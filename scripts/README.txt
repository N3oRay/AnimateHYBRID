python -m scripts.n3rHYBRID14 \
                         --pretrained-model-path "/mnt/62G/huggingface/miniSD" \
                         --config configs/prompts/2_animate/128.yaml \
                         --device cuda \
                         --vae-offload \
                         --fp16



# n3r_tiny.yaml
W: 128
H: 128
L: 4
steps: 10

init_image_scale: 0.5
creative_noise: 0.1

pretrained_model_path: "/mnt/62G/huggingface/miniSD"
dtype: float16

fps: 12
num_frames_per_image: 6
guidance_scale: 4.5
use_real_esrgan: true

#scheduler:
#  type: DDIMScheduler
#  steps: 22
#  beta_start: 0.00085
#  beta_end: 0.012

motion_module: scripts/modules/motion_module_tiny.py

seed: 1234

input_images:
  - input/image_128x0.png
  - input/image_128x1.png
  - input/image_128x2.png
  - input/image_128x3.png
  - input/image_128x4.png

#enable_xformers_memory_efficient_attention: true (Actif par défaut)
vae_path: "/mnt/62G/huggingface/vae/vae-ft-mse-840000-ema-pruned.safetensors"

prompt:
  - "best quality, 1 girl walking, natural dynamic pose, arms swinging, legs mid-step, balanced torso and head, flowing hair, consistent outfit and hairstyle, outdoor spring, cherry blossoms, petals, smooth motion across frames, coherent animation, vibrant colors, cinematic lighting"

n_prompt:
  - "low quality, blurry, deformed, stiff pose, unnatural movement, distorted anatomy, missing parts, extra limbs, broken motion, low resolution, inconsistent colors, messy background"
