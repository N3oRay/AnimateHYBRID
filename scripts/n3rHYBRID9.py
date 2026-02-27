# -------------------------
# n3rHYBRID8_WITH_VIDEO_ROBUST_FINAL.py
# -------------------------

import time
import argparse
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
import numpy as np
import csv
import cv2
import os

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_vae, test_vae_256
from scripts.utils.model_utils import load_pretrained_unet, get_text_embeddings, load_DDIMScheduler
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import load_images, decode_latents_correct, generate_latents_ai_5D_optimized

from transformers import CLIPTextModel, CLIPTokenizer

LATENT_SCALE = 0.18215
CLAMP_MAX = 1.0

torch.backends.cuda.matmul.allow_tf32 = True

# -------------------------
# Sauvegarde image
# -------------------------
def save_frame(img_array, filename):
    img_array = np.nan_to_num(img_array, nan=0.0, posinf=1.0, neginf=0.0)
    img_array_uint8 = (np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(img_array_uint8)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img.save(filename)

# -------------------------
# Encode image -> latents
# -------------------------
def encode_image_latents(image_tensor, vae, scale=LATENT_SCALE, dtype=torch.float16):
    vae_device = next(vae.parameters()).device
    img = image_tensor.to(device=vae_device,
                          dtype=torch.float32 if vae_device.type=="cpu" else dtype)
    with torch.no_grad():
        latents = vae.encode(img).latent_dist.sample() * scale
    return latents.unsqueeze(2)  # [B,C,1,H,W]

# -------------------------
# Pipeline principal
# -------------------------
def main(args):
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 and device=="cuda" else torch.float32

    # Paramètres
    fps = cfg.get("fps",12)
    num_fraps_per_image = cfg.get("num_fraps_per_image",12)
    steps = cfg.get("steps",35)
    seed = cfg.get("seed",42)
    guidance_scale = cfg.get("guidance_scale",4.5)
    init_image_scale = cfg.get("init_image_scale",0.85)
    creative_noise = cfg.get("creative_noise",0.0)

    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    prompt_text = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])

    total_frames = len(input_paths) * num_fraps_per_image * max(len(prompt_text),1)
    estimated_seconds = total_frames / fps

    print("📌 Paramètres de génération :")
    print(f"  fps                  : {fps}")
    print(f"  num_fraps_per_image : {num_fraps_per_image}")
    print(f"  steps                : {steps}")
    print(f"  guidance_scale       : {guidance_scale}")
    print(f"  init_image_scale     : {init_image_scale}")
    print(f"  creative_noise       : {creative_noise}")
    print(f"  seed                 : {seed}")
    print(f"⏱ Durée totale estimée : {estimated_seconds:.1f}s")

    # -------------------------
    # Tokenizer + Text Encoder
    # -------------------------
    tokenizer = CLIPTokenizer.from_pretrained(f"{args.pretrained_model_path}/tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(f"{args.pretrained_model_path}/text_encoder").to(device)
    if args.fp16:
        text_encoder = text_encoder.half()

    pos_embeds, neg_embeds = get_text_embeddings(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt_text,
        negative_prompt=negative_prompts,
        device=device,
        dtype=dtype
    )
    pos_embeds = pos_embeds.to(device=device, dtype=dtype)
    neg_embeds = neg_embeds.to(device=device, dtype=dtype)
    print("✅ Text encoder OK")

    # -------------------------
    # VAE
    # -------------------------
    vae_path = cfg.get("vae_path") or args.pretrained_model_path
    vae = safe_load_vae(vae_path, device=device, fp16=args.fp16, offload=args.vae_offload)
    test_vae_256(vae, Image.open("scripts/utils/logo.png").convert("RGB"))
    print("✅ VAE OK")

    # -------------------------
    # UNet + Scheduler
    # -------------------------
    unet = load_pretrained_unet(args.pretrained_model_path, device=device, dtype=dtype)
    unet.eval()
    try:
        unet.enable_xformers_memory_efficient_attention()
    except:
        pass
    scheduler = load_DDIMScheduler(args.pretrained_model_path)
    print("✅ UNet + Scheduler OK")

    # -------------------------
    # Motion module
    # -------------------------
    motion_module = load_motion_module(cfg.get("motion_module"), device=device) \
        if cfg.get("motion_module") else None

    # -------------------------
    # Output dirs
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/hybrid_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_frames"
    debug_dir.mkdir(exist_ok=True)

    csv_file = output_dir / "generation_log.csv"
    video = None
    frame_counter = 0

    # -------------------------
    # Génération
    # -------------------------
    with open(csv_file,"w",newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["frame","latent_min","latent_max","gen_time","decode_time"])

        for img_path in input_paths:

            input_image = load_images([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
            input_latents = encode_image_latents(input_image, vae, dtype=dtype)

            if args.vae_offload:
                vae.to(device)
                if args.fp16 and device=="cuda":
                    vae = vae.half()

            input_latents = input_latents.expand(-1,-1,num_fraps_per_image,-1,-1).clone()

            for f_idx in range(num_fraps_per_image):

                torch.cuda.empty_cache()

                latent_frame = input_latents[:,:,f_idx:f_idx+1,:,:].squeeze(2)
                frame_seed = seed + f_idx

                start = time.time()
                latent_frame = generate_latents_ai_5D_optimized(
                    latent_frame=latent_frame,
                    scheduler=scheduler,
                    pos_embeds=pos_embeds,
                    neg_embeds=neg_embeds,
                    unet=unet,
                    motion_module=motion_module,
                    device=device,
                    dtype=dtype,
                    guidance_scale=guidance_scale,
                    init_image_scale=init_image_scale,
                    creative_noise=creative_noise,
                    seed=frame_seed,
                    steps=steps
                )
                gen_time = time.time() - start

                if args.vae_offload:
                    vae.to(device)

                decode_start = time.time()
                frame_tensor = decode_latents_correct(latent_frame, vae)
                decode_time = time.time() - decode_start

                frame_array = frame_tensor.permute(0,2,3,1)[0].numpy()
                frame_array = np.nan_to_num(frame_array, nan=0.0, posinf=1.0, neginf=0.0)

                if video is None:
                    h, w = frame_array.shape[:2]
                    video_path = output_dir / "animation.mp4"
                    video = cv2.VideoWriter(str(video_path),
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            fps,(w,h))

                video.write(cv2.cvtColor((np.clip(frame_array,0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                save_frame(frame_array, debug_dir / f"frame_{frame_counter:05d}.png")

                writer.writerow([
                    frame_counter,
                    float(latent_frame.min()),
                    float(latent_frame.max()),
                    round(gen_time,4),
                    round(decode_time,4)
                ])

                del latent_frame
                del frame_tensor
                torch.cuda.empty_cache()
                frame_counter += 1

    if video:
        video.release()
    print("✅ Génération terminée.")

# -------------------------
# Entrée
# -------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--vae-offload", action="store_true")
    args = parser.parse_args()
    main(args)
