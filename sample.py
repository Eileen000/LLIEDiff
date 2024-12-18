import argparse
import os
import time
from itertools import islice
from typing import List, Optional, Tuple, Union

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import yaml
from PIL import Image
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers import (
    ImagePipelineOutput,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import DDPMScheduler
from diffusers.schedulers.scheduling_repaint import RePaintSchedulerOutput
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from pytorch_lightning import seed_everything
from transformers import CLIPTextModel, CLIPTokenizer

from measurements import DarkenOperator


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# Load configurations
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def img_save(image, path):
    image = (image.detach() / 2 + 0.5).clamp(0, 1)
    image = (image.cpu().permute(0, 2, 3, 1)).squeeze(0).numpy()
    images = (image * 255).round().astype("uint8")
    images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, images)


# import torch.nn.functional as F

class L2Loss(nn.Module):
    def forward(self, input, target):
        return torch.linalg.norm(input - target)
        # return torch.norm(torch.sum(torch.abs(input - target)), p=1, dim=0)
        # return torch.sqrt(F.mse_loss(input, target, reduction='mean')) * 255.
        # return torch.sqrt(F.mse_loss(input, target, reduction='sum'))
        # return torch.sqrt(torch.sum((input - target) ** 2))


class SmoothL1Loss(nn.Module):
    def forward(self, input, target):
        loss_function = nn.SmoothL1Loss(beta=1.0)
        return loss_function(input, target)


class L1Loss(nn.Module):
    def forward(self, input, target):
        loss_function = nn.L1Loss()
        return loss_function(input, target)


def blur_operation(input, scale=1):
    I1 = input
    # H, W = input.shape[-2:]
    for s in range(scale):
        # if s > 0:
        #     H, W = H//2, W//2
        #     I1 = kornia.geometry.transform.resize(I1, (H, W))
        I1 = kornia.filters.gaussian_blur2d(I1, (5, 5), (1.5, 1.5))
    return I1


class GuidedScheduler(DDPMScheduler):
    order = 1

    def __init__(
            self,
            num_train_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            beta_schedule: str = "linear",
            eta: float = 0.0,
            trained_betas: Optional[np.ndarray] = None,
            clip_sample: bool = True,
    ):
        super(GuidedScheduler, self).__init__(
            num_train_timesteps, beta_start, beta_end, beta_schedule, trained_betas=trained_betas,
            clip_sample=clip_sample
        )
        self.final_alpha_cumprod = torch.tensor(1.0)
        self.eta = eta

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    def set_timesteps(
            self,
            num_inference_steps: int,
            jump_length: int = 10,
            jump_n_sample: int = 10,
            strength=1.0,
            device: Union[str, torch.device] = None,
    ):
        num_inference_steps = min(self.config.num_train_timesteps, num_inference_steps)
        self.num_inference_steps = num_inference_steps

        timesteps = []

        jumps = {}
        if jump_length > 0:
            for j in range(0, num_inference_steps - jump_length, jump_length):
                jumps[j] = jump_n_sample - 1

        # [get the original timestep using init_timestep]
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t = init_timestep
        while t >= 1:
            t = t - 1
            timesteps.append(t)

            if jump_length > 0 and jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_length):
                    t = t + 1
                    if t < init_timestep:
                        timesteps.append(t)

        timesteps = np.array(timesteps) * (self.config.num_train_timesteps // self.num_inference_steps)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def _get_variance(self, t):
        prev_timestep = t - self.config.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from
        # https://arxiv.org/pdf/2006.11239.pdf) and sample from it to get
        # previous sample x_{t-1} ~ N(pred_prev_sample, variance) == add
        # variance to pred_sample
        # Is equivalent to formula (16) in https://arxiv.org/pdf/2010.02502.pdf
        # without eta.
        # variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            sample: torch.FloatTensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
    ):
        t = timestep
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # We choose to follow RePaint Algorithm 1 to get x_{t-1}, however we
        # substitute formula (7) in the algorithm coming from DDPM paper
        # (formula (4) Algorithm 2 - Sampling) with formula (12) from DDIM paper.
        # DDIM schedule gives the same results as DDPM with eta = 1.0
        # Noise is being reused in 7. and 8., but no impact on quality has
        # been observed.

        # 5. Add noise
        device = model_output.device
        noise = randn_tensor(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
        std_dev_t = self.eta * self._get_variance(timestep) ** 0.5

        variance = 0
        if t > 0 and self.eta > 0:
            variance = std_dev_t * noise

        # 6. compute "direction pointing to x_t" of formula (12)
        # from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * model_output

        # 7. compute x_{t-1} of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction + variance

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return RePaintSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def undo_step(self, sample, timestep, generator=None):
        n = self.config.num_train_timesteps // self.num_inference_steps

        for i in range(n):
            beta = self.betas[timestep + i]
            if sample.device.type == "mps":
                # randn does not work reproducibly on mps
                noise = randn_tensor(sample.shape, dtype=sample.dtype, generator=generator)
                noise = noise.to(sample.device)
            else:
                noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)

            # 10. Algorithm 1 Line 10 https://arxiv.org/pdf/2201.09865.pdf
            sample = (1 - beta) ** 0.5 * sample + beta ** 0.5 * noise

        return sample

    def __len__(self):
        return self.config.num_train_timesteps


class PLSDPipeline(StableDiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler,
    ):
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        set_requires_grad(self.text_encoder, False)
        set_requires_grad(self.vae, False)
        set_requires_grad(self.unet, False)
        self.loss_fn = L2Loss()  # SimplePerceptualLoss() #SmoothL1Loss() #L2Loss()
        self.loss_fn_latent = L2Loss()  # SmoothL1Loss() L2Loss() #nn.SmoothL1Loss(reduction='sum')

    def freeze_vae(self):
        set_requires_grad(self.vae, False)

    def unfreeze_vae(self):
        set_requires_grad(self.vae, True)

    def freeze_unet(self):
        set_requires_grad(self.unet, False)

    def unfreeze_unet(self):
        set_requires_grad(self.unet, True)

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents_from_img(self, init_latents, timestep, batch_size, num_images_per_prompt, dtype, device,
                                 generator=None):
        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    @torch.enable_grad()
    def cond_fn(
            self,
            latents,
            timestep,
            index,
            text_embeddings,
            noise_pred_original,
            last_sample,
            last_grad,
            measurement: torch.Tensor = None,
            meas_inv: torch.Tensor = None,
            meas_inv_blur: torch.Tensor = None,
            init_latents: torch.Tensor = None,
            operator=None,
            save_path: Optional[str] = None,
            step_size: Optional[float] = 10.0,
            w_inpaint: Optional[float] = 0.5,
            w_inv: Optional[float] = 0.5,
            w_inv_b: Optional[float] = 0.5,
            w_meas: Optional[float] = 0.5,
            w_stepsize_max: Optional[float] = 1.0,
            print_freq: Optional[int] = 10,
            use_adpative_stepsize=False,
            use_adpative_beta=False,
    ):
        latents_original = latents.detach()
        latents = latents_original.requires_grad_()
        latent_model_input = self.scheduler.scale_model_input(latents, timestep)
        # predict the noise residual
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        # PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        # compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        # sample = pred_original_sample * (torch.sqrt(beta_prod_t)) + latents * (1 - torch.sqrt(beta_prod_t))
        # image = self.vae.decode(1 / self.vae.config.scaling_factor * sample).sample
        # image = (image / 2 + 0.5).clamp(0, 1)

        loss = 0
        pred_original_img = self.vae.decode(1 / self.vae.config.scaling_factor * pred_original_sample).sample

        meas_pred = operator.forward(pred_original_img, is_latent=True, adaptive_beta=use_adpative_beta)
        origi_pred_blur = blur_operation(pred_original_img, scale=2)
        # measurement_blur = blur_operation(measurement, scale=2)
        # meas_inv_blur = operator.transpose(measurement_blur, is_latent=True, adaptive_beta=use_adpative_beta)

        meas_error = self.loss_fn(measurement, meas_pred)
        if w_meas > 0:  # alpha
            loss = loss + meas_error * w_meas

        inv_error = self.loss_fn(meas_inv, pred_original_img)
        if w_inv > 0:  # omega
            # w_inv=1.0
            loss = loss + inv_error * w_inv

        inv_error_b = self.loss_fn(meas_inv_blur, origi_pred_blur)
        if w_inv_b > 0:  # omega
            # w_inv=1.0
            loss = loss + inv_error_b * w_inv_b

        # ortho_pred = operator.transpose(operator.forward(origi_pred, is_latent=True), is_latent=True)
        # ortho_project = origi_pred - ortho_pred
        # parallel_project = operator.transpose(measurement, is_latent=True)
        # inpainted_image = parallel_project + ortho_project
        # encoded_z_0 = self.vae.config.scaling_factor * vae.encode(
        #     # inpainted_image).latent_dist.mean
        #     # meas_inv).latent_dist.sample(generator)
        #     # meas_inv).latent_dist.mean
        #     meas_inv_blur).latent_dist.mean
        inpaint_error = self.loss_fn_latent(
            init_latents, self.vae.config.scaling_factor * vae.encode(origi_pred_blur).latent_dist.mean
            # pred_original_sample, self.vae.config.scaling_factor * vae.encode(meas_inv).latent_dist.mean
        )
        if w_inpaint > 0:  # gamma
            loss = loss + inpaint_error * w_inpaint

        loss.backward()

        # upgrade step_size
        step_size_adp = step_size
        if use_adpative_stepsize:
            if not (timestep.item() >= 999 or last_sample is None or last_grad is None):
                step_size_adp = torch.mul(pred_original_sample - last_sample,
                                          pred_original_sample - last_sample).sum() / torch.mul(
                    pred_original_sample - last_sample, (latents.grad - last_grad)).sum()
                step_size = max(min(step_size_adp, w_stepsize_max), 1 - 6)

        if index % print_freq == 0:
            print(f"[{index:04d}:{timestep:04d}] lr_adp: {step_size_adp:.2f} loss: {loss.item():.2f} = "
                  f"meas: {w_meas:.2f} * {0 if meas_error is None else meas_error.item():.2f} + "
                  f"inv: {w_inv:.2f} * {0 if inv_error is None else inv_error.item():.2f} + "
                  f"inv_b: {w_inv_b:.2f} * {0 if inv_error_b is None else inv_error_b.item():.2f} + "
                  f"inpaint: {w_inpaint:.2f} * {0 if inpaint_error is None else inpaint_error.item():.2f}")
            name, ext = os.path.splitext(save_path)
            img_save(pred_original_img, f"{name}_x0_pred_{timestep.item():04d}{ext}")
            img_save(origi_pred_blur, f"{name}_origi_pred_blur_{timestep.item():04d}{ext}")
        # noise_pred = noise_pred_original - torch.sqrt(beta_prod_t) * (-latents.grad) * step_size
        latents_original = latents_original - latents.grad * step_size
        return noise_pred_original, latents_original, pred_original_sample.detach(), latents.grad.detach()

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            strength: float = 0.8,
            num_inference_steps: Optional[int] = 1000,
            guidance_scale: Optional[float] = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            do_measure: bool = True,
            weight_dtype=torch.float32,
            jump_length: Optional[int] = 10,
            jump_n_sample: Optional[int] = 10,
            # callback
            measurement: torch.Tensor = None,
            meas_inv: torch.Tensor = None,
            meas_inv_blur: torch.Tensor = None,
            operator=None,
            save_path: Optional[str] = None,
            step_size: Optional[float] = 10.0,
            w_inpaint: Optional[float] = 0.5,
            w_inv: Optional[float] = 0.5,
            w_inv_b: Optional[float] = 0.5,
            w_meas: Optional[float] = 0.5,
            w_stepsize_max: Optional[float] = 1.0,
            print_freq: Optional[int] = 10,
            use_adpative_stepsize=False,
            use_adpative_beta=False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps=None,
            negative_prompt=negative_prompt,
        )

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        device = self.device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, jump_length, jump_n_sample, strength=strength, device=device)
        self.scheduler.eta = eta
        timesteps = self.scheduler.timesteps
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        print(f"timesteps (000): {timesteps[:500]}")
        if len(timesteps) > 500:
            print(f"timesteps (-500): {timesteps[-500:]}")

        # 4. Preprocess image
        init_latents = self.vae.config.scaling_factor * self.vae.encode(
            # meas_inv).latent_dist.sample(generator)
            meas_inv_blur).latent_dist.sample(generator)

        img_save(self.vae.decode(init_latents / self.vae.config.scaling_factor).sample,
                 f"{os.path.dirname(save_path)}/image_init.png")

        if latents is None:
            latents = self.prepare_latents_from_img(
                init_latents,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                weight_dtype,
                self.device,
                generator,
            )

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        t_last = timesteps[0] + 1

        last_sample, last_grad = None, None

        # 10. Denoising loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if t >= t_last:
                    # compute the reverse: x_t-1 -> x_t
                    latents = self.scheduler.undo_step(latents, t_last, generator)
                    progress_bar.update()
                    t_last = t
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                # perform classifier free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # perform clip guidance
                if do_measure:
                    text_embeddings_for_guidance = (
                        prompt_embeds.chunk(2)[1] if do_classifier_free_guidance else prompt_embeds
                    )
                    noise_pred, latents, last_sample, last_grad = self.cond_fn(
                        latents=latents,
                        timestep=t,
                        index=i,
                        text_embeddings=text_embeddings_for_guidance,
                        noise_pred_original=noise_pred,
                        last_sample=last_sample,
                        last_grad=last_grad,
                        # callbacks
                        measurement=measurement,
                        meas_inv=meas_inv,
                        meas_inv_blur=meas_inv_blur,
                        init_latents=init_latents,
                        operator=operator,
                        save_path=save_path,
                        step_size=step_size,
                        w_inpaint=w_inpaint,
                        w_inv=w_inv,
                        w_inv_b=w_inv_b,
                        w_meas=w_meas,
                        w_stepsize_max=w_stepsize_max,
                        print_freq=print_freq,
                        use_adpative_stepsize=use_adpative_stepsize,
                        use_adpative_beta=use_adpative_beta,
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    **extra_step_kwargs,
                ).prev_sample

                progress_bar.update()

                t_last = t

        # 11. Post-processing
        image = self.decode_latents(latents)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        # 13. Convert to PIL
        if not return_dict:
            return (image, None)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return ImagePipelineOutput(images=image[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir to read samples from",
        default="lol_samples/"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        # default="8k uhd, soft lighting, high quality, best quality, extremely detailed, clean, tidy",
        help="the prompt to render"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        nargs="?",
        default="",
        # default="8k uhd, soft lighting, high quality, best quality, extremely detailed, clean, tidy",
        help="the negative_prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/psld-samples-llie"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=999,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default='configs/LLIE_config_psld.yaml',
        help="task config yml file",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        # default='stabilityai/stable-diffusion-2-1',
        help="model id",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=10.0,
        help="reconstruction error",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.6,
        help="beta for lime",
    )
    parser.add_argument(
        "--lime_step",
        type=int,
        default=1,
        help="lime step",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=500,
        help="inpainting error",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=200.0,
        help="measurement error",
    )
    parser.add_argument(
        "--omega_b",
        type=float,
        default=200.0,
        help="blurry measurement error",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="reconstruction error",
    )
    parser.add_argument(
        "--adp_stepsize",
        action='store_true',
        default=False,
        help="measurements beta adaptive",
    )
    parser.add_argument(
        "--adp_beta",
        action='store_true',
        default=False,
        help="measurements beta adaptive",
    )
    parser.add_argument(
        "--do_measure",
        action='store_true',
        default=False,
        help="measurements beta adaptive",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="adaptive step_size scale max value",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.3,
        help="adpative beta",
    )
    parser.add_argument(
        "--jump_len",
        type=int,
        default=0,
        help="time travel",
    )
    parser.add_argument(
        "--jump_sample",
        type=int,
        default=0,
        help="time travel",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="init strength",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--test_id",
        type=str,
        default="0",
        help="number of test id",
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(opt.model_id, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(opt.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(opt.model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(opt.model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(opt.model_id, subfolder="scheduler")
    scheduler = GuidedScheduler.from_config(scheduler.config)

    # print(f"Done")
    # exit(0)

    vae.to(device=torch_device, dtype=weight_dtype)
    vae.enable_xformers_memory_efficient_attention()
    text_encoder.to(device=torch_device, dtype=weight_dtype)
    unet.to(device=torch_device, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()

    pipeline = PLSDPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    pipeline.to(torch_device=torch_device, torch_dtype=weight_dtype)

    batch_size = opt.n_samples
    assert opt.prompt is not None

    outdir = os.path.join(opt.outdir, str(opt.test_id))
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'measurements'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'inter'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'results'), exist_ok=True)

    imgsName = sorted(os.listdir(opt.indir))

    for i, imgName in enumerate(imgsName):

        #########################################################
        # set up operator and measurement
        #########################################################
        img_path = os.path.join(opt.indir, imgName)
        save_path = os.path.join(outdir, 'inter', f'inter_{imgName}')
        print('>>>>>>>>> ', str(opt.test_id), ' ', img_path, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # Prepare Operator and noise
        operator = DarkenOperator(
            device=torch_device, w=opt.W, h=opt.H,
            img_path=img_path, dtype=torch.float32, beta=opt.beta, sigma=opt.sigma,  # a=opt.a, b=opt.b
        )
        # noiser = get_noise(name='gaussian', sigma=0.01)

        img = operator.origin_img
        if img.ndim == 2:
            img = img[:, :, None]

        measurement = torch.from_numpy(img.transpose((2, 0, 1))).contiguous().unsqueeze(0).to(device=torch_device,
                                                                                              dtype=weight_dtype)

        degradation_map = operator.get_illum(opt.lime_step).to(device=torch_device, dtype=weight_dtype)
        meas_inv = operator.transpose(measurement, is_latent=False, adaptive_beta=opt.adp_beta)
        meas_inv_blur = blur_operation(meas_inv, scale=2)

        degradation_map = operator.get_illum(opt.lime_step).to(device=torch_device, dtype=torch.float32)
        vutils.save_image(measurement * degradation_map ** (1 - torch.exp(degradation_map) - 0.7),
                          os.path.join(outdir, 'measurements', f'I0_{imgName}'))

        #########################################################

        out = pipeline(
            prompt=opt.prompt,
            negative_prompt=opt.negative_prompt,
            height=opt.H,
            width=opt.W,
            num_inference_steps=opt.ddim_steps,
            guidance_scale=opt.scale,
            output_type="pil",
            return_dict=True,
            weight_dtype=weight_dtype,
            do_measure=opt.do_measure,
            jump_length=opt.jump_len,
            jump_n_sample=opt.jump_sample,
            strength=opt.strength,
            # callbacks
            measurement=measurement * 2.0 - 1.0,
            meas_inv=meas_inv * 2.0 - 1.0,
            meas_inv_blur=meas_inv_blur * 2.0 - 1.0,
            operator=operator,
            save_path=save_path,
            step_size=opt.step_size,
            w_inpaint=opt.gamma,
            w_inv=opt.omega,
            w_inv_b=opt.omega_b,
            w_meas=opt.alpha,
            w_stepsize_max=opt.eta,
            print_freq=opt.print_freq,
            use_adpative_stepsize=opt.adp_stepsize,
            use_adpative_beta=opt.adp_beta,
        ).images

        image = out

        out.save(os.path.join(outdir, 'results', f'{imgName}'))
        print('Done')
