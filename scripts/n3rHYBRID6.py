# -------------------------
# n3rHYBRID5_WITH_VIDEO.py
# -------------------------
import time
import argparse
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
import numpy as np
import csv
import cv2

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import (
    safe_load_vae, test_vae_256
)
from scripts.utils.model_utils import load_pretrained_unet, load_scheduler, load_text_encoder, get_text_embeddings
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_ai_5D_optimized, load_images, encode_image_latents, decode_latents_safe

from transformers import CLIPTextModel, CLIPTokenizer

LATENT_SCALE = 0.18215  # Tiny-SD 128x128



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
    seed = cfg.get("seed", 42)
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
    print(f"  seed                 : {seed}")
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

    start_total = time.time()
    frame_counter = 0

    # CSV log setup
    csv_file = output_dir / "generation_log.csv"
    with open(csv_file, mode="w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["frame_index", "filename", "latent_min", "latent_max", "gen_time", "decode_time"])

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
                start_gen = time.time()

                latent_frame = generate_latents_ai_5D_optimized(
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
                    seed=seed, # + f_idx,
                    steps=steps  # <-- ajoute ça
                )

                frame_generation_time = time.time() - frame_start

                end_gen = time.time()

                start_decode = time.time()

                # Décodage via VAE (sûr device/dtype)
                vae_device = next(vae.parameters()).device
                frame_tensor = vae.decode(latent_frame.to(device=vae_device, dtype=torch.float32 if vae_device.type=="cpu" else dtype)/LATENT_SCALE).sample

                frame_decode_time = time.time() - frame_start - frame_generation_time

                print("Frame générée en:", frame_generation_time)
                print("Frame décodée en:", frame_decode_time)
                frame_tensor = frame_tensor.clamp(-1,1)
                frame_tensor = (frame_tensor+1)/2
                frame_tensor = frame_tensor.cpu().permute(0,2,3,1)[0]
                #frame_pil = Image.fromarray((frame_tensor.numpy()*255).astype("uint8"))
                frame_pil = Image.fromarray(
                    (frame_tensor.detach().cpu().numpy() * 255).astype("uint8")
                )

                end_decode = time.time()

                # Save PNG
                frame_name = f"frame_{frame_counter:05d}.png"
                frame_pil.save(debug_dir / frame_name)

                # Save CSV
                csv_writer.writerow([
                    frame_counter,
                    frame_name,
                    float(latent_frame.min()),
                    float(latent_frame.max()),
                    round(end_gen-start_gen, 4),
                    round(end_decode-start_decode, 4)
                ])

                frames_for_video.append(frame_pil)
                print(f"[Frame {frame_counter}] Frame sauvegardée")
                frame_counter += 1

    # -------------------------
    # Génération vidéo MP4
    # -------------------------
    if frames_for_video:
        height, width = frames_for_video[0].size[1], frames_for_video[0].size[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = output_dir / "animation.mp4"
        video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for frame_pil in frames_for_video:
            frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)

        video.release()
        print(f"✅ Vidéo sauvegardée: {video_path}")

    print(f"✅ Tout terminé en {time.time()-start_total:.2f}s")
    print("Fin time:", time.time() - start_total)


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

