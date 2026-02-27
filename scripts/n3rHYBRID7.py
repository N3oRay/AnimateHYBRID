# -------------------------
# n3rHYBRID7_WITH_VIDEO_ROBUST.py
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
from scripts.utils.n3r_utils import generate_latents_ai_5D_optimized, load_images, decode_latents_safe_ai, decode_latents_correct # encode_image_latents,

from transformers import CLIPTextModel, CLIPTokenizer

LATENT_SCALE = 0.18215  # Tiny-SD 128x128
CLAMP_MAX = 1.0          # Clamp strict avant VAE


import numpy as np
from PIL import Image
import os

def save_frame(img_array, filename):
    """
    Sauvegarde un tableau NumPy en PNG correctement.

    img_array : np.ndarray
        Tableau 2D (grayscale) ou 3D (H,W,C) avec valeurs float ou int
    filename : str
        Chemin du fichier de sortie (.png)
    """

    # Assurer que c'est un tableau NumPy
    img_array = np.array(img_array)

    # Si float (0..1 ou autres), normaliser entre 0 et 255
    if np.issubdtype(img_array.dtype, np.floating):
        img_array = np.clip(img_array, 0.0, 1.0)  # clamp à [0,1]
        img_array = (img_array * 255).astype(np.uint8)

    # Si int mais pas uint8, clip à [0,255] et convertir
    elif np.issubdtype(img_array.dtype, np.integer) and img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # Conversion canaux si nécessaire (OpenCV BGR → RGB)
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        # Si tu utilises OpenCV pour générer les images, elles sont BGR
        # img_array = img_array[..., ::-1]  # Décommente si les couleurs sont décalées
        pass

    # Créer un objet Image et sauvegarder
    img = Image.fromarray(img_array)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img.save(filename)

# Exemple d'utilisation
# img_array peut être ton output de AnimateHYBRID
# img_array = np.random.rand(256, 256, 3)  # float 0..1
# save_frame(img_array, "outputs/frame_00000.png")

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
                # Seed unique par frame pour animation
                frame_seed = seed + f_idx

                latent_frame = input_latents[:, :, f_idx:f_idx+1, :, :].squeeze(2)
                print(f"[Frame {frame_counter}] Latents avant génération: min={latent_frame.min():.4f}, max={latent_frame.max():.4f}")

                frame_start = time.time()
                start_gen = time.time()

                # -------------------------
                # Préparer latent batch pour guidance (fix UNet batch mismatch)
                # -------------------------
                batch_size = pos_embeds.shape[0]  # nombre de prompts positifs

                # Répéter latent pour correspondre aux embeddings concaténés (pos + neg)
                latent_model_input_in = latent_frame.repeat(batch_size * 2, 1, 1, 1)

                # Concaténer les embeddings positifs et négatifs pour guidance
                embeds = torch.cat([pos_embeds, neg_embeds], dim=0)

                # Appel correct de la fonction avec scheduler
                latent_frame = generate_latents_ai_5D_optimized(
                    latent_model_input_in,  # input latent répété
                    scheduler,              # <-- scheduler ajouté
                    embeds,                 # embeddings concaténés
                    unet,
                    motion_module=motion_module,
                    device=device,
                    dtype=dtype,
                    guidance_scale=guidance_scale,
                    init_image_scale=init_image_scale,
                    creative_noise=creative_noise,
                    seed=frame_seed,
                    steps=steps
                )

                frame_generation_time = time.time() - frame_start

                start_decode = time.time()

                # Clamp strict avant décodage
                # Limiter les valeurs pour éviter les NaN en fp16
                latent_frame = latent_frame.clamp(-CLAMP_MAX, CLAMP_MAX)
                #frame_tensor = vae.decode(latent_frame.to(device=next(vae.parameters()).device, dtype=torch.float32 if device=="cpu" else dtype)/LATENT_SCALE).sample
                # Compatible fp16, fp32 et VAE-offload.
                frame_tensor = decode_latents_correct(latent_frame, vae)

                frame_decode_time = time.time() - start_decode

                # Save PNG
                frame_tensor = frame_tensor.clamp(-1,1)
                frame_tensor = (frame_tensor+1)/2
                frame_array = frame_tensor.cpu().permute(0,2,3,1)[0].detach().cpu().numpy()
                frame_name = f"frame_{frame_counter:05d}.png"
                save_frame(frame_array, debug_dir / frame_name)

                end_decode = time.time()

                # Save CSV
                csv_writer.writerow([
                    frame_counter,
                    frame_name,
                    float(latent_frame.min()),
                    float(latent_frame.max()),
                    round(frame_generation_time, 4),
                    round(frame_decode_time, 4)
                ])

                frames_for_video.append(frame_array)
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

        for frame_array in frames_for_video:
            frame_bgr = cv2.cvtColor(np.array(frame_array), cv2.COLOR_RGB2BGR)
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
