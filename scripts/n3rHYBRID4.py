# -------------------------
# n3rHYBRID3_PATCHED.py
# -------------------------
import time
import argparse
import csv
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
import numpy as np

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_vae, test_vae_256
from scripts.utils.model_utils import load_pretrained_unet, load_scheduler, load_text_encoder, get_text_embeddings
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_ai_5D, load_images

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
    dtype = torch.float16 if args.fp16 and device=="cuda" else torch.float32

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

    # CSV log
    csv_log = output_dir / "frame_times.csv"
    csv_file = open(csv_log, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_idx", "latents_time", "vae_decode_time", "total_time"])

    to_pil = ToPILImage()
    frames_for_video = []
    frame_counter = 0
    start_global = time.time()
    print("Lancement time:", time.time()-start_global)

    # -------------------------
    # Generation loop
    # -------------------------
    for img_path in input_paths:
        input_image = load_images([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
        t0_latent = time.time()
        input_latents = encode_image_latents(input_image, vae, dtype=dtype)
        input_latents = input_latents.expand(-1,-1,num_fraps_per_image,-1,-1).clone()
        t1_latent = time.time()
        print(f"✅ Latents encodés pour {img_path} en {t1_latent - t0_latent:.2f}s")

        for f_idx in range(num_fraps_per_image):
            t_frame_start = time.time()
            latent_frame = input_latents[:, :, f_idx:f_idx+1, :, :].squeeze(2)
            print(f"[Frame {frame_counter}] Latents avant génération: min={latent_frame.min():.4f}, max={latent_frame.max():.4f}")

            # génération
            t_latent_gen_start = time.time()
            latent_frame = generate_latents_ai_5D(
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
            t_latent_gen_end = time.time()

            # Décodage via VAE
            t_vae_start = time.time()
            vae_device = next(vae.parameters()).device
            frame_tensor = vae.decode(latent_frame.to(device=vae_device, dtype=torch.float32 if vae_device.type=="cpu" else dtype)/LATENT_SCALE).sample

            frame_tensor = frame_tensor.clamp(-1,1)
            frame_tensor = (frame_tensor + 1)/2
            # Ne garder que RGB
            if frame_tensor.shape[1] > 3:
                frame_tensor = frame_tensor[:, :3, :, :]
            frame_tensor = frame_tensor.cpu().permute(0,2,3,1)[0]
            frame_pil = to_pil(frame_tensor)
            t_vae_end = time.time()

            frame_pil.save(debug_dir / f"frame_{frame_counter:05d}.png")
            frames_for_video.append(frame_pil)

            # CSV log
            csv_writer.writerow([frame_counter, t_latent_gen_end - t_latent_gen_start, t_vae_end - t_vae_start, time.time() - t_frame_start])

            print(f"[Frame {frame_counter}] Frame sauvegardée, latents={t_latent_gen_end - t_latent_gen_start:.2f}s, VAE={t_vae_end - t_vae_start:.2f}s")
            frame_counter += 1

    csv_file.close()
    print(f"✅ Génération terminée. {len(frames_for_video)} frames sauvegardées dans {debug_dir}")
    print(f"✅ CSV log créé : {csv_log}")
    print("Fin time:", time.time() - start_global)

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
