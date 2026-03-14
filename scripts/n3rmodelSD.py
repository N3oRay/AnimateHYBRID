# --------------------------------------------------------------
# n3rmodelSD.py - AnimateDiff pipeline ultra light ~2Go VRAM
# --------------------------------------------------------------
import os, math, threading
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageEnhance
import argparse

from diffusers import AutoencoderKL, PNDMScheduler
from transformers import CLIPTokenizerFast, CLIPTextModel

from scripts.utils.lora_utils import apply_lora, apply_lora_smart
from scripts.utils.vae_config import load_vae
from scripts.utils.tools_utils import (
    ensure_4_channels,
    save_frames_as_video_from_folder
)
from scripts.utils.config_loader import load_config
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_safe_wrapper_v2, load_images_test
from scripts.utils.fx_utils import encode_images_to_latents_nuanced,decode_latents_ultrasafe_blockwise
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

# ---------------- Motion Module Safe ----------------
def apply_motion_safe(latents, motion_module, threshold=1e-2):
    if latents.abs().max() < threshold:
        return latents, False
    latents_after = motion_module(latents)
    return latents_after, True

def prepare_embeddings_for_unet(pos_embeds, neg_embeds, target_dim):
    """
    Adapte les embeddings texte pour correspondre au cross_attention_dim du UNet
    """

    current_dim = pos_embeds.shape[-1]

    if current_dim == target_dim:
        return pos_embeds, neg_embeds

    # tronquage
    if current_dim > target_dim:
        pos_embeds = pos_embeds[..., :target_dim]
        neg_embeds = neg_embeds[..., :target_dim]

    # padding
    elif current_dim < target_dim:
        pad = target_dim - current_dim
        pos_embeds = torch.nn.functional.pad(pos_embeds, (0, pad))
        neg_embeds = torch.nn.functional.pad(neg_embeds, (0, pad))

    return pos_embeds, neg_embeds
# ---------------- MAIN ----------------
def main(args):
    global stop_generation

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    fps = cfg.get("fps", 12)
    upscale_factor = cfg.get("upscale_factor", 1)
    transition_frames = cfg.get("transition_frames", 4)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 2)
    steps = max(cfg.get("steps", 16), 4)
    guidance_scale = cfg.get("guidance_scale", 4.0)

    scheduler = PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    scheduler.set_timesteps(steps, device=device)

    # ---------------- UNET ----------------
    unet = safe_load_unet(args.pretrained_model_path, device=device, fp16=True)
    if hasattr(unet, "enable_attention_slicing"):
        unet.enable_attention_slicing()
    if hasattr(unet, "enable_xformers_memory_efficient_attention"):
        try:
            unet.enable_xformers_memory_efficient_attention(True)
        except:
            pass

    # ---------------- LoRA ----------------
    cyber_skin_path = cfg.get("n3oray_models", {}).get("cybersamurai_v2")  # cybersamurai_v2 ou cyber_skin ou cyber_skin_girl ou cyberpunk_style_v3
    if cyber_skin_path is None:
        raise ValueError("❌ Impossible de trouver '{cyber_skin_path}' dans n3oray_models du YAML.")

    # Appliquer LoRA de manière intelligente
    unet = apply_lora_smart(unet, cyber_skin_path, alpha=0.5, device=device, verbose=True)

    unet_cross_attention_dim = None

    print("[MODEL INFO]")
    if hasattr(unet.config, "cross_attention_dim"):
        unet_cross_attention_dim = unet.config.cross_attention_dim
        print(f"[INFO] UNet cross_attention_dim = {unet_cross_attention_dim}")

    # ---------------- Motion module ----------------
    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None

    # ---------------- Tokenizer / Text encoder ----------------
    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path,"tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(
        os.path.join(args.pretrained_model_path,"text_encoder")
    ).to(device).to(dtype)

    print("Text encoder hidden_size:", text_encoder.config.hidden_size)

    # ---------------- VAE ----------------
    vae_path = cfg.get("vae_path")
    vae, vae_type, latent_channels, LATENT_SCALE = load_vae(vae_path, device=device, dtype=dtype)


    # ---------------- Embeddings ----------------
    prompts = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])
    embeddings = []

    for prompt_item in prompts:
        prompt_text = " ".join(prompt_item) if isinstance(prompt_item, list) else str(prompt_item)
        neg_text = " ".join(negative_prompts) if isinstance(negative_prompts, list) else str(negative_prompts)

        text_inputs = tokenizer(prompt_text, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
        neg_inputs = tokenizer(neg_text, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")

        #with torch.no_grad():
        #    pos_embeds = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state
        #    neg_embeds = text_encoder(neg_inputs.input_ids.to(device)).last_hidden_state

        CLIP_SKIP = 2  # valeur recommandée par ton LoRA

        with torch.no_grad():
            # Obtenir toutes les hidden_states
            pos_output = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
            neg_output = text_encoder(neg_inputs.input_ids.to(device), output_hidden_states=True)

            # Clip Skip = 2 → prendre la 2ème dernière couche
            pos_embeds = pos_output.hidden_states[-CLIP_SKIP]
            neg_embeds = neg_output.hidden_states[-CLIP_SKIP]

        # garder embeddings originaux
        pos_unet, neg_unet = prepare_embeddings_for_unet(
            pos_embeds,
            neg_embeds,
            unet_cross_attention_dim
        )
        print(f"[DEBUG] original embeds: {pos_embeds.shape}")

        embeddings.append((pos_unet, neg_unet))
        print(f"[DEBUG] UNet embeds: {pos_unet.shape}")

    # ---------------- Input images ----------------
    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    total_frames = len(input_paths) * num_fraps_per_image * max(len(prompts), 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/modelSD_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"

    print("📌 Paramètres de génération :")
    print(f"fps: {fps}, frames/image: {num_fraps_per_image}, steps: {steps}, guidance_scale: {guidance_scale}")

    previous_latent_single = None
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)

    for img_idx, img_path in enumerate(input_paths):
        if stop_generation:
            break

        # Charger et encoder image
        input_image = load_images_test([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
        current_latent_single = encode_images_to_latents_nuanced(input_image, vae, device=device, latent_scale=LATENT_SCALE)
        current_latent_single = ensure_4_channels(current_latent_single)

        # Transition frames
        if previous_latent_single is not None and transition_frames > 0:
            for t in range(transition_frames):
                if stop_generation:
                    break
                alpha = 0.5 - 0.5 * math.cos(math.pi * t / max(transition_frames - 1, 1))
                latent_interp = (1 - alpha) * previous_latent_single + alpha * current_latent_single
                latent_interp = latent_interp.clone()
                if motion_module:
                    latent_interp, _ = apply_motion_safe(latent_interp, motion_module)

                # Décodage ultra-safe
                frame_pil = decode_latents_ultrasafe_blockwise(
                    latent_interp, vae,
                    block_size=32, overlap=24,
                    gamma=1.0, brightness=1.0, contrast=1.5, saturation=1.3,
                    device=device, frame_counter=frame_counter,
                    output_dir=Path("."), epsilon=1e-5,
                    latent_scale_boost=5.71
                )
                if upscale_factor > 1:
                    frame_pil = frame_pil.resize((frame_pil.width*upscale_factor, frame_pil.height*upscale_factor), Image.BICUBIC)
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frame_counter += 1
                pbar.update(1)

        # ---------------- Frames principales ----------------
        for pos_embeds, neg_embeds in embeddings:
            for f in range(num_fraps_per_image):
                if stop_generation:
                    break

                if f == 0:
                    # Première frame = image d'origine
                    frame_tensor = (input_image.squeeze(0) + 1) / 2.0
                    frame_tensor = frame_tensor.clamp(0, 1)
                    frame_pil = to_pil_image(frame_tensor.cpu())
                else:
                    latents_frame = current_latent_single.clone()
                    cf_embeds = (pos_embeds.to(device), neg_embeds.to(device))

                    latents = generate_latents_safe_wrapper_v2(
                        unet=unet,
                        scheduler=scheduler,
                        input_latents=latents_frame,
                        embeddings=cf_embeds,
                        motion_module=None,  # Motion module post-génération
                        guidance_scale=guidance_scale,
                        device=device,
                        fp16=True,
                        steps=steps,
                        debug=False
                    )

                    if motion_module:
                        latents, _ = apply_motion_safe(latents, motion_module)

                    frame_pil = decode_latents_ultrasafe_blockwise(
                        latents, vae,
                        block_size=32, overlap=24,
                        gamma=1.0, brightness=1.0, contrast=1.5, saturation=1.3,
                        device=device, frame_counter=frame_counter,
                        output_dir=Path("."), epsilon=1e-5,
                        latent_scale_boost=5.71
                    )

                    if upscale_factor > 1:
                        frame_pil = frame_pil.resize(
                            (frame_pil.width*upscale_factor, frame_pil.height*upscale_factor),
                            Image.BICUBIC
                        )

                    del latents
                    torch.cuda.empty_cache()

                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frame_counter += 1
                pbar.update(1)

        previous_latent_single = current_latent_single

    pbar.close()
    save_frames_as_video_from_folder(output_dir, out_video, fps=fps)
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
    parser.add_argument("--n3_model", type=str, default="cybersamurai_v2") # cybersamurai_v2 - cyber_skin
    parser.add_argument("--scheduler", type=str, default="pndm")
    args = parser.parse_args()
    main(args)
