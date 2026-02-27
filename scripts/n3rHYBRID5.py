import time
import argparse
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
import numpy as np
import csv

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import (
    safe_load_vae, test_vae_256, decode_latents_safe
)
from scripts.utils.model_utils import load_pretrained_unet, load_scheduler, load_text_encoder, get_text_embeddings
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_ai_5D_optimized, load_images

from transformers import CLIPTextModel, CLIPTokenizer

LATENT_SCALE = 0.18215  # Tiny-SD 128x128

# -------------------------
# Encode image en latents
# -------------------------
def encode_image_latents(image_tensor, vae, scale=LATENT_SCALE, dtype=torch.float16):
    vae_device = next(vae.parameters()).device
    img = image_tensor.to(device=vae_device, dtype=torch.float32 if vae_device.type=="cpu" else dtype)
    with torch.no_grad():
        latents = vae.encode(img).latent_dist.sample() * scale
    return latents.unsqueeze(2)  # [B,C,1,H,W]

# -------------------------
# Main pipeline
# -------------------------
def main(args):

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 and device == "cuda" else torch.float32

    fps = cfg.get("fps", 12)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 12)
    steps = cfg.get("steps", 35)
    guidance_scale = cfg.get("guidance_scale", 4.5)
    init_image_scale = cfg.get("init_image_scale", 0.85)
    creative_noise = cfg.get("creative_noise", 0.0)

    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    prompt_text = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])

    total_frames = len(input_paths) * num_fraps_per_image * max(len(prompt_text), 1)
    estimated_seconds = total_frames / fps

    print("📌 Paramètres de génération :")
    print(f"  fps                  : {fps}")
    print(f"  num_fraps_per_image : {num_fraps_per_image}")
    print(f"  steps                : {steps}")
    print(f"  guidance_scale       : {guidance_scale}")
    print(f"  init_image_scale     : {init_image_scale}")
    print(f"  creative_noise       : {creative_noise}")
    print(f"⏱ Durée totale estimée de la vidéo : {estimated_seconds:.1f}s")

    # -------------------------
    # Charger Tokenizer + Text Encoder
    # -------------------------
    print("🔄 Chargement tokenizer et text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(f"{args.pretrained_model_path}/tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(f"{args.pretrained_model_path}/text_encoder").to(device)
    if args.fp16:
        text_encoder = text_encoder.half()
    print("✅ Tokenizer et Text Encoder chargés.")

    # -------------------------
    # Text embeddings
    # -------------------------
    pos_embeds, neg_embeds = get_text_embeddings(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt_text,
        negative_prompt=negative_prompts,
        device=device,
        dtype=dtype
    )

    # -------------------------
    # Load VAE
    # -------------------------
    print("🔄 Chargement VAE ...")
    vae_path = cfg.get("vae_path") or args.pretrained_model_path
    vae = safe_load_vae(vae_path, device=device, fp16=args.fp16, offload=args.vae_offload)
    if vae is None:
        raise RuntimeError("❌ Échec du chargement du VAE")
    test_image = Image.open("scripts/utils/logo.png").convert("RGB")
    test_vae_256(vae, test_image)
    print("✅ VAE opérationnel pour debug")

    # -------------------------
    # Load UNet + Scheduler
    # -------------------------
    print("🔄 Chargement UNet ...")
    unet = load_pretrained_unet(args.pretrained_model_path, device=device, dtype=dtype)
    print("✅ UNet chargé et prêt.")
    print("🔄 Chargement Scheduler ...")
    scheduler = load_scheduler(args.pretrained_model_path)
    print("✅ Scheduler chargé.")

    # -------------------------
    # Motion module
    # -------------------------
    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None
    if motion_module:
        print("✅ Motion module debug-ready")

    # -------------------------
    # Output setup
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/hybrid_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_frames"
    debug_dir.mkdir(exist_ok=True)

    to_pil = ToPILImage()
    frames_for_video = []
    frame_counter = 0

    start = time.time()
    print("Lancement time:", time.time() - start)

    # CSV log setup
    csv_file = output_dir / "generation_log.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Generation Time", "Decode Time", "Latent Min", "Latent Max"])

    # -------------------------
    # Generation loop
    # -------------------------
    for img_path in input_paths:
        input_image = load_images([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
        input_latents = encode_image_latents(input_image, vae, dtype=dtype)
        input_latents = input_latents.expand(-1, -1, num_fraps_per_image, -1, -1).clone()
        print(f"✅ Latents encodés pour {img_path}")

        for f_idx in range(num_fraps_per_image):
            latent_frame = input_latents[:, :, f_idx:f_idx+1, :, :].squeeze(2)
            print(f"[Frame {frame_counter}] Latents avant génération: min={latent_frame.min():.4f}, max={latent_frame.max():.4f}")

            # Start timing the frame generation
            frame_start = time.time()

            # génération frame
            latent_frame = generate_latents_ai_5D_optimized(
                latent_frame,
                motion_module=motion_module,
                device=device,
                dtype=dtype,
                guidance_scale=guidance_scale,
                init_image_scale=init_image_scale,
                creative_noise=creative_noise,
                seed=42 + f_idx,
                unet=unet,
                scheduler=scheduler,
                pos_embeds=pos_embeds,
                neg_embeds=neg_embeds
            )

            frame_generation_time = time.time() - frame_start

            # Décodage via VAE (sûr device/dtype)
            vae_device = next(vae.parameters()).device
            frame_tensor = vae.decode(latent_frame.to(device=vae_device, dtype=torch.float32 if vae_device.type=="cpu" else dtype)/LATENT_SCALE).sample

            frame_decode_time = time.time() - frame_start - frame_generation_time

            print("Frame générée en:", frame_generation_time)
            print("Frame décodée en:", frame_decode_time)

            # Log frame times in CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_counter, frame_generation_time, frame_decode_time, latent_frame.min().item(), latent_frame.max().item()])

            print(f"[Frame {frame_counter}] Frame sauvegardée")
            frame_counter += 1

    print(f"✅ Génération terminée. {len(frames_for_video)} frames sauvegardées dans {debug_dir}")
    print("Fin time:", time.time() - start)

# -------------------------
# Entrée
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--vae-offload", action="store_true")
    args = parser.parse_args()
    main(args)
