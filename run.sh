python new_sample_xl.py \
  --prompt="8k uhd, soft lighting, high quality, best quality, extremely detailed, clean, tidy" \
  --negative_prompt="art, blur, blurry, deformed iris, deformed pupils, dehydrated, disgusting, drawing, haze, low quality, semi-realistic, sketch, text, ugly, worst quality" \
  --indir=data/size512 --outdir=outputs/psld-samples-llie --print_freq=5 \
  --do_measure --adp_stepsize --step_size=1.0 --eta=1.0 --alpha=10.0 --omega=1.0 --omega_b=1.0 --gamma=1.0 \
  --lime_step=1 --adp_beta --beta=0.8 --sigma=0.3 \
  --model_id=stabilityai/sdxl-turbo --H=512 --W=512 --ddim_steps=50 --jump_len=0 --jump_sample=0 --strength=0.4 --scale=0.0 \
  --test_id=sdxltb-latent-p-lr_adp-a10-o1-o1b-g1g
