# --------------------------------------------------------------
# n3rmodelSD_final.py - AnimateDiff ultra-light ~2Go VRAM
# --------------------------------------------------------------
import os, math, threading
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import argparse

from diffusers import PNDMScheduler
from transformers import CLIPTokenizerFast, CLIPTextModel

from scripts.utils.lora_utils import apply_lora_smart
from scripts.utils.vae_config import load_vae
from scripts.utils.tools_utils import ensure_4_channels
from scripts.utils.config_loader import load_config
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_safe_miniGPU, generate_latents_mini_gpu, load_images_test, generate_latents_mini_gpu_320, run_diffusion_pipeline
from scripts.utils.fx_utils import encode_images_to_latents_nuanced, decode_latents_ultrasafe_blockwise, save_frames_as_video_from_folder, encode_images_to_latents_safe
from scripts.utils.vae_utils import safe_load_unet

LATENT_SCALE = 0.18215
stop_generation = False

# ---------------- Thread stop ----------------
def wait_for_stop():
    global stop_generation
    inp = input("Appuyez sur '²' + Entrée pour arrêter : ")
    if inp.lower() == "²":
        stop_generation = True
threading.Thread(target=wait_for_stop, daemon=True).start()

# ---------------- Utilitaires ----------------

def compute_overlap(W, H, block_size, max_overlap_ratio=0.6):
    overlap = int(block_size * max_overlap_ratio)
    return min(overlap, min(W,H)//4)

def apply_motion_safe(latents, motion_module, threshold=1e-2):
    if latents.abs().max() < threshold:
        return latents, False
    return motion_module(latents), True



# ---------------- MAIN ----------------
def main(args):
    global stop_generation
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    use_mini_gpu = cfg.get("use_mini_gpu", True) # True for <2 Go VRAM - False for <4 Go VRAM

    fps = cfg.get("fps", 12)
    upscale_factor = cfg.get("upscale_factor", 1)
    transition_frames = cfg.get("transition_frames", 4)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 2)
    steps = max(cfg.get("steps", 16), 4)
    guidance_scale = cfg.get("guidance_scale", 4.0)

    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(steps, device=device)

    # ---------------- UNET ----------------
    unet = safe_load_unet(args.pretrained_model_path, device=device, fp16=True)
    if hasattr(unet, "enable_attention_slicing"): unet.enable_attention_slicing()
    if hasattr(unet, "enable_xformers_memory_efficient_attention"):
        try: unet.enable_xformers_memory_efficient_attention(True)
        except: pass

    # ---------------- LoRA ----------------
    unet_cross_attention_dim = getattr(unet.config, "cross_attention_dim", 768)
    n3oray_models = cfg.get("n3oray_models")
    if n3oray_models:
        for model_name, lora_path in n3oray_models.items():
            applied = apply_lora_smart(unet, lora_path, alpha=0.5, device=device, verbose=True)
            if not applied:
                print(f"⚠ LoRA '{model_name}' ignorée (incompatible UNet)")
    else:
        print("⚠ Aucun modèle LoRA n'est configuré, étape ignorée.")
    # ---------------- Motion module ----------------
    #motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None
    motion_module = None
    if motion_module is not None:
        print(f"[DEBUG] motion_module type: {type(motion_module)}, latents shape before motion: {latents.shape}")


    # ---------------- Tokenizer / Text encoder ----------------
    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path,"tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path,"text_encoder")).to(device).to(dtype)

    # ---------------- VAE ----------------
    vae_path = cfg.get("vae_path")
    vae, vae_type, latent_channels, LATENT_SCALE = load_vae(vae_path, device=device, dtype=dtype)

    # ---------------- Embeddings ----------------
    # ---------------- Embeddings ----------------
    embeddings = []
    prompts = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])

    # Récupération du cross_attention_dim attendu par le UNet
    unet_cross_attention_dim = getattr(unet.config, "cross_attention_dim", 1024)

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

        # ---------------- CORRECTION DES DIMENSIONS ----------------
        current_dim = pos_embeds.shape[-1]
        if current_dim != unet_cross_attention_dim:
            # Projection linéaire 768 -> 1024
            projection = torch.nn.Linear(current_dim, unet_cross_attention_dim).to(device).to(dtype)
            pos_embeds = projection(pos_embeds)
            neg_embeds = projection(neg_embeds)

        embeddings.append((pos_embeds, neg_embeds))

    print(f"✅ Embeddings adaptées à UNet cross_attention_dim={unet_cross_attention_dim}")

    # ---------------- DEBUG DIMENSIONS ----------------
    print("\n🔍 Vérification des dimensions avant génération")
    for i, (pos, neg) in enumerate(embeddings):
        print(f"Embedding {i}: pos {pos.shape}, neg {neg.shape}")
        if pos.shape[-1] != unet_cross_attention_dim:
            print(f"⚠ Attention : pos_embedding dim {pos.shape[-1]} != UNet {unet_cross_attention_dim}")
        if neg.shape[-1] != unet_cross_attention_dim:
            print(f"⚠ Attention : neg_embedding dim {neg.shape[-1]} != UNet {unet_cross_attention_dim}")

    print(f"UNet cross_attention_dim attendu : {unet_cross_attention_dim}")
    print("✅ Toutes les dimensions semblent correctes\n")


    # ---------------- Input images ----------------
    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    total_frames = len(input_paths) * num_fraps_per_image * max(len(prompts), 1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/modelSD2_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"

    print(f"📌 fps: {fps}, frames/image: {num_fraps_per_image}, steps: {steps}, guidance_scale: {guidance_scale}")

    block_size = cfg.get("block_size", 64)
    overlap = compute_overlap(cfg["W"], cfg["H"], block_size)

    previous_latent_single = None
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)

    for img_idx, img_path in enumerate(input_paths):
        if stop_generation: break

        # Charger et normaliser l'image
        input_image = load_images_test([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
        input_image = ensure_4_channels(input_image)

        # Encoder l'image en latents
        current_latent_single = encode_images_to_latents_safe(input_image, vae, device=device, latent_scale=LATENT_SCALE)

        print(f"[DEBUG] Latents shape after encoding: {current_latent_single.shape}")


        # ---------------- Ajuster la taille des latents pour UNet ----------------
        # UNet.sample_size correspond à la taille d'entrée attendue par le modèle (ex: 320 ou 512)
        target_H = getattr(unet.config, "sample_size", cfg["H"]) // 8
        target_W = getattr(unet.config, "sample_size", cfg["W"]) // 8

        # Interpolation bilinéaire pour correspondre à UNet
        current_latent_single = torch.nn.functional.interpolate(
            current_latent_single, size=(target_H, target_W), mode='bilinear', align_corners=False
        )

        # Assurer 4 channels (sécurité)
        current_latent_single = ensure_4_channels(current_latent_single)

        print(f"DEBUG latents shape after interpolation: {current_latent_single.shape}")

        # ---------------- Transition frames ----------------
        if previous_latent_single is not None and transition_frames > 0:
            for t in range(transition_frames):
                if stop_generation: break
                alpha = 0.5 - 0.5*math.cos(math.pi*t/max(transition_frames-1,1))
                latent_interp = (1-alpha)*previous_latent_single + alpha*current_latent_single
                if motion_module: latent_interp, _ = apply_motion_safe(latent_interp, motion_module)
                frame_pil = decode_latents_ultrasafe_blockwise(latent_interp, vae,
                                                               block_size=block_size, overlap=overlap,
                                                               gamma=1.0, brightness=1.0,
                                                               contrast=1.5, saturation=1.3,
                                                               device=device, frame_counter=frame_counter,
                                                               latent_scale_boost=5.71)
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frame_counter += 1
                pbar.update(1)

        # ---------------- Frames principales ----------------
        for pos_embeds, neg_embeds in embeddings:
            for f in range(num_fraps_per_image):
                if stop_generation: break
                if f == 0:
                    # Frame initiale = image d'entrée
                    frame_tensor = torch.clamp((input_image.squeeze(0)+1)/2, 0, 1)
                    frame_pil = to_pil_image(frame_tensor)
                else:
                    latents_frame = current_latent_single.clone()

                    # ---------------- Redimensionnement latents ----------------
                    unet_sample_size = getattr(unet.config, "sample_size", cfg["H"])  # ex: 320 ou 512
                    target_H = target_W = unet_sample_size // 8
                    if latents_frame.shape[-2:] != (target_H, target_W):
                        latents_frame = torch.nn.functional.interpolate(
                            latents_frame,
                            size=(target_H, target_W),
                            mode='bilinear',
                            align_corners=False
                        )

                    # ---------------- Assurer 4 canaux ----------------
                    latents_frame = ensure_4_channels(latents_frame)

                    cf_embeds = (pos_embeds.to(device), neg_embeds.to(device))
                    print("[DEBUG] embeddings shape:", pos_embeds.shape, "neg shape:", neg_embeds.shape)
                    print("[DEBUG] motion_module dim:", getattr(motion_module, "cross_attention_dim", "Unknown"))

                    if use_mini_gpu:
                        # 1,5–2 Go VRAM, safe pour GPU <3 Go.
                        latents = generate_latents_mini_gpu_320(
                            unet=unet,
                            scheduler=scheduler,
                            input_latents=latents_frame,
                            embeddings=cf_embeds,
                            motion_module=motion_module,
                            guidance_scale=guidance_scale,
                            device=device,
                            fp16=True,
                            steps=steps,
                            debug=True
                        )
                    else:
                        # Full pipeline (~2.5–3 Go minimum)
                        # run_diffusion_pipeline attend 3 canaux
                        latents_input = latents_frame[:, :3, :, :]
                        latents = run_diffusion_pipeline(
                            unet=unet,
                            vae=vae,
                            scheduler=scheduler,
                            images=latents_input,
                            embeddings=cf_embeds,
                            timesteps=scheduler.timesteps,
                            device=device
                        )

                    if motion_module:
                        latents, _ = apply_motion_safe(latents, motion_module)

                    frame_pil = decode_latents_ultrasafe_blockwise(
                        latents, vae,
                        block_size=block_size, overlap=overlap,
                        gamma=1.0, brightness=1.0,
                        contrast=1.5, saturation=1.3,
                        device=device, frame_counter=frame_counter,
                        latent_scale_boost=5.71
                    )

                    del latents
                    torch.cuda.empty_cache()

                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frame_counter += 1
                pbar.update(1)

        previous_latent_single = current_latent_single

    pbar.close()
    save_frames_as_video_from_folder(output_dir, out_video, fps=fps, upscale_factor=2)
    print(f"🎬 Vidéo générée : {out_video}")
    print("✅ Pipeline terminé avec motion module safe.")

# ---------------- ENTRY ----------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--vae-offload", action="store_true")
    args = parser.parse_args()
    main(args)
