# -------------------------
# n3rMASTER10.py - AnimateDiff 5D Debug Complet
# -------------------------
import argparse
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
import numpy as np

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import (
    safe_load_vae, encode_images_to_latents_ai, test_vae_256
)
from scripts.utils.model_utils import load_pretrained_unet, load_text_encoder, load_scheduler, get_text_embeddings
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_ai_5D, load_images

from transformers import CLIPTextModel, CLIPTokenizer

LATENT_SCALE = 0.18215  # Tiny-SD 128x128

# -------------------------
# Pipeline de génération 5D
# -------------------------
def main(args):
    # --- Config ---
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 and device == "cuda" else torch.float32

    fps = cfg.get("fps", 12)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 12)
    steps = cfg.get("steps", 35)
    guidance_scale = cfg.get("guidance_scale", 4.5)
    init_image_scale = cfg.get("init_image_scale", 0.85)
    creative_noise = cfg.get("creative_noise", 0.0)
    seed_global = cfg.get("seed", 42)

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

    torch.manual_seed(seed_global)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed_global)

    # --- Load tokenizer & text encoder ---
    print(f"🔄 Chargement tokenizer depuis {args.pretrained_model_path}/tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(f"{args.pretrained_model_path}/tokenizer")
    print("✅ Tokenizer chargé et prêt.")
    print(f"🔄 Chargement text_encoder depuis {args.pretrained_model_path}/text_encoder")
    text_encoder = CLIPTextModel.from_pretrained(f"{args.pretrained_model_path}/text_encoder").to(device)
    print("✅ Text encoder chargé et prêt.")

    # --- Embeddings textes ---
    pos_embeds, neg_embeds = get_text_embeddings(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt_text,
        negative_prompt=negative_prompts,
        device=device,
        dtype=dtype
    )

    # --- Load VAE ---
    vae_path = cfg.get("vae_path") or f"{args.pretrained_model_path}/vae"
    vae = safe_load_vae(vae_path, device=device, fp16=args.fp16, offload=args.vae_offload)
    test_image = Image.open("scripts/utils/logo.png").convert("RGB")
    test_vae_256(vae, test_image)
    print("✅ VAE opérationnel pour debug")

    # --- Load UNet & Scheduler ---
    print(f"🔄 Chargement UNet depuis {args.pretrained_model_path}")
    unet = load_pretrained_unet(args.pretrained_model_path, device=device, dtype=dtype)
    print("✅ UNet chargé et prêt.")
    print(f"🔄 Chargement scheduler depuis {args.pretrained_model_path}")
    scheduler = load_scheduler(args.pretrained_model_path)
    print("✅ Scheduler chargé.")

    # --- Motion module ---
    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None
    if motion_module:
        print("✅ Motion module debug-ready : toutes les propriétés requises initialisées")

    # --- Setup output ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/debug_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_frames"
    debug_dir.mkdir(exist_ok=True)
    to_pil = ToPILImage()

    frame_counter = 0
    for img_path in input_paths:
        # --- Initialisation latents ---
        if img_path is None:
            input_latents = torch.randn(
                1, 4, num_fraps_per_image, cfg["H"] // 8, cfg["W"] // 8,
                device=device, dtype=dtype
            ) * 0.01
            print("🔹 Latents initialisés avec petit bruit pour génération random")
        else:
            input_image = load_images([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
            input_image = input_image.to(device=vae.device, dtype=next(vae.parameters()).dtype)
            input_latents = encode_images_to_latents_ai(input_image, vae)
            input_latents = input_latents.unsqueeze(2).expand(-1, -1, num_fraps_per_image, -1, -1).clone()
            print(f"✅ Latents encodés pour {img_path}")

        # Save initial latents
        np.save(debug_dir / f"latents_init_{frame_counter:05d}.npy", input_latents.cpu().numpy())
        print(f"[Frame {frame_counter}] Latents init: min={input_latents.min():.4f}, max={input_latents.max():.4f}, mean={input_latents.mean():.4f}")

        # --- Frame loop ---
        for f_idx in range(num_fraps_per_image):
            latent_frame = input_latents[:, :, f_idx:f_idx+1, :, :].squeeze(2)

            # Génération du latent suivant
            latent_frame = generate_latents_ai_5D(
                latent_frame,
                motion_module=motion_module,
                device=device,
                dtype=dtype,
                guidance_scale=guidance_scale,
                init_image_scale=init_image_scale,
                creative_noise=creative_noise,
                seed=seed_global + f_idx,
                unet=unet,
                scheduler=scheduler,
                pos_embeds=pos_embeds,
                neg_embeds=neg_embeds
            )

            # Décodage via VAE
            try:
                frame_tensor = vae.decode(latent_frame / LATENT_SCALE).sample
                frame_tensor = frame_tensor.clamp(-1, 1)
                frame_tensor = (frame_tensor + 1.0) / 2.0
                frame_tensor = frame_tensor.cpu().permute(0, 2, 3, 1)[0]
                frame_pil = to_pil(frame_tensor)
                frame_pil.save(debug_dir / f"frame_{frame_counter:03d}.png")
                print(f"[Frame {frame_counter}] Frame sauvegardée ✅")
            except Exception as e:
                print(f"[Frame {frame_counter}] Erreur lors du décodage: {e}")

            frame_counter += 1

    print(f"✅ Génération terminée. Toutes les frames sont dans {debug_dir}")


# -------------------------
# Entrée script
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--vae-offload", action="store_true")
    args = parser.parse_args()
    main(args)
