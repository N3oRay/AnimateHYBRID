# -------------------------
# Script 5D AnimateDiff avec tile_size auto
# -------------------------

import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from datetime import datetime
import os
import shutil
from PIL import Image
from torchvision.transforms import ToPILImage

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_vae, safe_load_unet, safe_load_scheduler
from scripts.utils.vae_utils import decode_latents_to_image_tiled, encode_images_to_latents_ai, decode_latents_frame_ai, log_gpu_memory
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_ai_5D, load_image_file, generate_5D_video_auto, load_images

LATENT_SCALE = 0.18215  # Tiny-SD 128x128



# -------------------------
# Video save
# -------------------------
def save_frames_as_video(frames, output_path, fps=12):
    import ffmpeg
    temp_dir = Path("temp_frames")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    for idx, frame in enumerate(frames):
        frame.save(temp_dir / f"frame_{idx:05d}.png")
    (
        ffmpeg.input(f"{temp_dir}/frame_%05d.png", framerate=fps)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
    shutil.rmtree(temp_dir)

# -------------------------
# Main pipeline 5D
# -------------------------
def main(args):
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 and device == "cuda" else torch.float32

    fps = cfg.get("fps", 12)
    num_frames_per_image = cfg.get("num_frames_per_image", 12)
    steps = cfg.get("steps", 35)
    guidance_scale = cfg.get("guidance_scale", 4.5)
    init_image_scale = cfg.get("init_image_scale", 0.85)
    creative_noise = cfg.get("creative_noise", 0.0)
    seed_global = cfg.get("seed", 42)

    torch.manual_seed(seed_global)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed_global)

    input_paths = cfg.get("input_images") or [cfg.get("input_image")] if cfg.get("input_image") else []
    prompts = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])

    # -------------------------
    # Load models
    # -------------------------
    print("🔄 Chargement des modèles...")
    unet = safe_load_unet(args.pretrained_model_path, device, fp16=args.fp16)
    vae = safe_load_vae(args.pretrained_model_path, device, fp16=args.fp16, offload=args.vae_offload)
    scheduler = safe_load_scheduler(args.pretrained_model_path)
    if hasattr(scheduler, "set_timesteps"):
        scheduler.set_timesteps(steps, device=device)
        print(f"✅ Scheduler initialisé avec {steps} steps")
    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None

    # -------------------------
    # Text embeddings
    # -------------------------
    from transformers import CLIPTokenizerFast, CLIPTextModel
    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path, "text_encoder")).to(device)
    if args.fp16 and device == "cuda":
        text_encoder = text_encoder.half()

    embeddings = []
    for prompt_item in prompts:
        prompt_text = " ".join(prompt_item) if isinstance(prompt_item, list) else str(prompt_item)
        neg_text = " ".join(negative_prompts) if isinstance(negative_prompts, list) else str(negative_prompts)
        text_inputs = tokenizer(prompt_text, padding="max_length", truncation=True,
                                max_length=tokenizer.model_max_length, return_tensors="pt")
        neg_inputs = tokenizer(neg_text, padding="max_length", truncation=True,
                               max_length=tokenizer.model_max_length, return_tensors="pt")
        with torch.no_grad():
            pos_embeds = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state
            neg_embeds = text_encoder(neg_inputs.input_ids.to(device)).last_hidden_state
        embeddings.append((pos_embeds.to(dtype), neg_embeds.to(dtype)))

    if len(embeddings) == 0:
        print("⚠ Aucun prompt fourni → génération vide.")
        return

    # -------------------------
    # Output
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"

    to_pil = ToPILImage()
    frames_for_video = []

    # -------------------------
    # Generation loop 5D sécurisée
    # -------------------------
    if not input_paths:
        input_paths = [None]

    frame_counter = 0
    for img_path in input_paths:
        if img_path is None:
            input_latents = torch.randn(
                1, unet.in_channels, num_frames_per_image, cfg["H"] // 8, cfg["W"] // 8,
                device=device, dtype=dtype
            )
            print(f"[INFO] Latents initiaux aléatoires shape={input_latents.shape}")
        else:
            input_image = load_images([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
            input_latents = encode_images_to_latents_ai(input_image, vae)
            input_latents = input_latents.expand(-1, -1, num_frames_per_image, -1, -1).clone()
            print(f"[INFO] Latents encodés pour image {img_path}, shape={input_latents.shape}")

        for pos_embeds, neg_embeds in embeddings:
            for f_idx in range(num_frames_per_image):
                latent_frame = input_latents[:, :, f_idx:f_idx+1, :, :].squeeze(2)
                try:
                    latents_frame = generate_latents_ai_5D(
                        latent_frame,
                        pos_embeds,
                        neg_embeds,
                        unet,
                        scheduler,
                        motion_module=motion_module,
                        device=device,
                        dtype=dtype,
                        guidance_scale=guidance_scale,
                        init_image_scale=init_image_scale,
                        creative_noise=creative_noise,
                        seed=seed_global + frame_counter
                    )
                except Exception as e:
                    print(f"⚠ Erreur frame {frame_counter:05d} → reset léger: {e}")
                    latents_frame = torch.randn_like(latent_frame) * 0.5

                latents_frame = latents_frame.clamp(-3.0, 3.0)

                # --- Decode via VAE avec tile_size auto ---
                frame_tensor = decode_latents_frame_ai(latents_frame, vae)

                if frame_tensor.ndim == 4:
                    frame_tensor = frame_tensor[0]
                elif frame_tensor.ndim != 3:
                    frame_tensor = torch.zeros(3, cfg["H"], cfg["W"], device="cpu")

                frame_pil = to_pil(frame_tensor.cpu())
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frames_for_video.append(frame_pil)

                frame_counter += 1

    save_frames_as_video(frames_for_video, out_video, fps=fps)
    print(f"🎬 Vidéo générée : {out_video}")
    print("✅ Pipeline 5D terminé proprement.")

# Entrée
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--vae-offload", action="store_true")
    args = parser.parse_args()
    main(args)
