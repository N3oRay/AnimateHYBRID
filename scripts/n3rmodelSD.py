# --------------------------------------------------------------
# n3rmodel.py - AnimateDiff pipeline optimisé VRAM (DEBUG+Stable Latents)
# En quelques mots :
# Robustesse latents – ton encodage/décodage sécurisé évite les NaNs et crash VRAM, rare dans les scripts publics.
# Pipeline animé stable – combine UNet + motion module + latents répétitifs pour créer des vidéos fluides, pas trivial.
# Upscale et ajustements dynamiques – gamma, contraste, saturation appliqués frame par frame → qualité contrôlable.
# Modularité – tu peux changer modèle, scheduler, ou motion module facilement → flexibilité rare.
# --------------------------------------------------------------
import os, math, threading
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm
from torchvision.transforms import ToPILImage
import argparse
import torchvision.transforms as T
import copy

from diffusers import AutoencoderKL, PNDMScheduler
from transformers import CLIPTokenizerFast, CLIPTextModel
from safetensors.torch import load_file
from PIL import Image, ImageFilter, ImageEnhance

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_unet
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import load_image_file, load_images_test, generate_latents_safe_wrapper

LATENT_SCALE = 0.18215
stop_generation = False

# ---------------- Thread stop ----------------
def wait_for_stop():
    global stop_generation
    inp = input("Appuyez sur '²' + Entrée pour arrêter : ")
    if inp.lower() == "²":
        stop_generation = True
threading.Thread(target=wait_for_stop, daemon=True).start()

# ---------------- Utils ----------------
def prepare_frame_tensor(frame_tensor):
    if frame_tensor.ndim == 5: frame_tensor = frame_tensor.squeeze(2)
    if frame_tensor.ndim == 4: frame_tensor = frame_tensor.squeeze(0)
    if frame_tensor.ndim == 3 and frame_tensor.shape[0] != 3: frame_tensor = frame_tensor.permute(2,0,1)
    return frame_tensor.clamp(0,1)

def normalize_frame(frame_tensor):
    min_val = frame_tensor.min()
    max_val = frame_tensor.max()
    if max_val > min_val:
        frame_tensor = (frame_tensor - min_val)/(max_val-min_val)
    return frame_tensor.clamp(0,1)

def tensor_to_pil(frame_tensor):
    if frame_tensor.ndim==4: frame_tensor=frame_tensor[0]
    return T.ToPILImage()(frame_tensor.cpu().clamp(0,1))

def save_frames_as_video_from_folder(folder_path, output_path, fps=12):
    import ffmpeg
    folder_path = Path(folder_path)
    frame_files = sorted(folder_path.glob("frame_*.png"))
    if not frame_files:
        print("❌ Aucun frame trouvé")
        return
    pattern = str(folder_path / "frame_*.png")
    (
        ffmpeg.input(pattern, framerate=fps, pattern_type='glob')
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )

# ---------------- Encode / Decode ----------------
# ---------------- Décodage latents → image ultrasafe ----------------
def decode_latents_ultrasafe_blockwise(latents, vae, block_size=160, overlap=96, gamma=1.0, brightness=1.0, contrast=1.0, saturation=1.0):
    """
    Décodage latents en utilisant des blocs pour réduire la VRAM.
    """
    from torchvision.transforms import ToPILImage
    import torch
    import numpy as np

    B, C, H, W = latents.shape
    device = next(vae.parameters()).device

    # Patch grid
    stride = block_size - overlap
    h_steps = max(1, (H - overlap + stride - 1) // stride)
    w_steps = max(1, (W - overlap + stride - 1) // stride)

    # Image de sortie
    output = torch.zeros(B, C, H, W, device="cpu")
    weight_map = torch.zeros(B, 1, H, W, device="cpu")

    for i in range(h_steps):
        for j in range(w_steps):
            y0 = i * stride
            x0 = j * stride
            y1 = min(y0 + block_size, H)
            x1 = min(x0 + block_size, W)

            patch = latents[:, :, y0:y1, x0:x1].to(device)
            patch = torch.nan_to_num(patch, nan=0.0, posinf=5.0, neginf=-5.0)

            with torch.no_grad():
                patch_decoded = vae.decode(patch.to(vae.dtype)).sample
                patch_decoded = ((patch_decoded + 1)/2).clamp(0,1).cpu()

            # Weighted blend pour recouvrement
            h_patch, w_patch = patch_decoded.shape[2], patch_decoded.shape[3]
            mask = torch.ones(1, 1, h_patch, w_patch)
            output[:, :, y0:y0+h_patch, x0:x0+w_patch] += patch_decoded * mask
            weight_map[:, :, y0:y0+h_patch, x0:x0+w_patch] += mask

            torch.cuda.empty_cache()

    # Normalisation finale
    output = output / weight_map.clamp(min=1e-5)

    # Passage en PIL
    to_pil = ToPILImage()
    frame_pil = to_pil(output[0].clamp(0,1))

    # Ajustements image
    from PIL import ImageEnhance
    if gamma != 1.0: frame_pil = ImageEnhance.Brightness(frame_pil).enhance(gamma)
    if brightness != 1.0: frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
    if contrast != 1.0: frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
    if saturation != 1.0: frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)

    return frame_pil

# ---------------- Encodage images → latents (robuste) ----------------
def encode_images_to_latents_safe(images, vae, device="cuda", dtype=torch.float16):

    # image en float32 pour VAE
    images_t = images.to(device=device, dtype=torch.float32)

    # sauvegarde dtype VAE
    original_dtype = next(vae.parameters()).dtype

    # encoder en float32
    vae = vae.to(torch.float32)
    with torch.no_grad():
        latents = vae.encode(images_t).latent_dist.sample()

    # revenir au dtype original du VAE
    vae = vae.to(original_dtype)

    # --- correction clé : normalisation et conversion vers dtype attendu ---
    latents = latents * LATENT_SCALE
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    # normalisation pour UNet / motion module
    max_abs = latents.abs().max()
    if max_abs > 0:
        latents = latents / max_abs  # scale [-1,1]

    # conversion finale vers dtype de la pipeline (fp16 si demandé)
    latents = latents.to(dtype)

    # reshape pour pipeline [B,C,T,H,W]
    if latents.ndim == 4:
        latents = latents.unsqueeze(2)

    print(
        "[SAFE ENCODE]",
        "min=", latents.min().item(),
        "max=", latents.max().item(),
        "std=", latents.std().item(),
        "dtype=", latents.dtype,
        "device=", latents.device,
        "shape=", latents.shape
    )

    return latents

# ---------------- Decodage images → latents corrigé ----------------
def decode_latents_to_image_safe(latents, vae, gamma=0.7, brightness=1.2, contrast=1.1, saturation=1.15):
    latents = torch.nan_to_num(latents, nan=0.0, posinf=4.0, neginf=-4.0)
    if latents.ndim==5: latents = latents[:,:,0,:,:]
    latents = (latents / LATENT_SCALE).clamp(-5,5)
    with torch.no_grad():
        image = vae.decode(latents.to(vae.dtype)).sample
    image = (image+1)/2
    image = image.clamp(0,1)
    print(f"[DEBUG] decode_latents_to_image_safe: min={image.min().item()}, max={image.max().item()}, nan={torch.isnan(image).any()}")
    return image

# ---------------- MAIN ----------------
def main(args):
    global stop_generation
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32

    fps = cfg.get("fps",12)
    upscale_factor = cfg.get("upscale_factor",2)
    transition_frames = cfg.get("transition_frames",8)
    num_fraps_per_image = cfg.get("num_fraps_per_image",12)
    steps = max(cfg.get("steps",16),12)

    # ---------------- Scheduler ----------------
    base_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler = base_scheduler
    scheduler.set_timesteps(steps, device=device)

    guidance_scale = cfg.get("guidance_scale",4.5)
    init_image_scale = cfg.get("init_image_scale",0.85)
    creative_noise = cfg.get("creative_noise",0.0)

    # ---------------- UNet ----------------
    n3_model_path = cfg["n3oray_models"].get(args.n3_model)
    if n3_model_path is None: raise ValueError(f"N3 model '{args.n3_model}' non défini dans le YAML")
    print(f"✅ Chargement du modèle N3 '{args.n3_model}' depuis : {n3_model_path}")

    state_dict = load_file(n3_model_path, device="cpu")
    unet = safe_load_unet(args.pretrained_model_path, device=device, fp16=True)
    #if hasattr(unet, "enable_xformers_memory_efficient_attention"):
    #    try: unet.enable_xformers_memory_efficient_attention(); print("✅ xformers memory attention activé")
    #    except: pass
    # ⚠️ Important : doit être fait avant toute génération
    if hasattr(unet, "enable_xformers_memory_efficient_attention"):
        try:
            unet.enable_xformers_memory_efficient_attention(False)
        except Exception as e:
            print("⚠️ Impossible de désactiver xFormers, fallback:", e)
    unet.load_state_dict(state_dict, strict=False)
    if hasattr(unet,"enable_attention_slicing"): unet.enable_attention_slicing()
    if device=="cuda":
        try: unet = torch.compile(unet, mode="reduce-overhead", fullgraph=False)
        except: pass
    print(f"[DEBUG] UNet dtype/device: {next(unet.parameters()).dtype}/{next(unet.parameters()).device}")

    # ---------------- Motion module ----------------
    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None
    print(f"✅ Motion module chargé")

    # ---------------- Tokenizer & Text Encoder ----------------
    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path,"tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path,"text_encoder")).to(device).to(dtype)
    print(f"✅ tokenizer - text_encoder module chargé")

    # ---------------- VAE ----------------
    vae_path = cfg.get("vae_path")
    vae = AutoencoderKL.from_single_file(
        vae_path,
        torch_dtype=dtype
    )

    vae.to(device)
    vae.enable_slicing()
    vae.enable_tiling()
    print(f"✅ VAE chargé depuis : {vae_path} dtype={vae.dtype}")
    print("VAE parameters:", sum(p.numel() for p in vae.parameters()))

    # ---------------- Embeddings ----------------
    prompts = cfg.get("prompt",[])
    negative_prompts = cfg.get("n_prompt",[])
    embeddings = []
    for prompt_item in prompts:
        prompt_text = " ".join(prompt_item) if isinstance(prompt_item,list) else str(prompt_item)
        neg_text = " ".join(negative_prompts) if isinstance(negative_prompts,list) else str(negative_prompts)
        text_inputs = tokenizer(prompt_text, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
        neg_inputs = tokenizer(neg_text, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
        with torch.no_grad():
            pos_embeds = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state
            neg_embeds = text_encoder(neg_inputs.input_ids.to(device)).last_hidden_state
        embeddings.append((pos_embeds.to(dtype), neg_embeds.to(dtype)))
        print(f"[DEBUG] Embedding loaded: pos min/max={pos_embeds.min().item()}/{pos_embeds.max().item()}, neg min/max={neg_embeds.min().item()}/{neg_embeds.max().item()}")

    # ---------------- Input images ----------------
    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    total_frames = len(input_paths)*num_fraps_per_image + max(len(input_paths)-1,0)*transition_frames
    print(f"🎞 Frames totales estimées : {total_frames}")

    # ---------------- OUTPUT ----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/model_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"
    to_pil = ToPILImage()
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)
    previous_latent_single = None


    # ---------------- Main loop sécurisé ----------------
    # ---------------- MAIN LOOP SAFE ----------------
    for img_idx, img_path in enumerate(input_paths):
        if stop_generation: break

        # --- Chargement image ---
        input_image = load_images_test([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
        input_latents_single = encode_images_to_latents_safe(input_image, vae, device=device, dtype=dtype)

        # --- Assurer channels corrects ---
        if input_latents_single.shape[1] == 1:
            input_latents_single = input_latents_single.repeat(1, 4, 1, 1)

        # --- Répétition pour frames ---
        input_latents = input_latents_single.repeat(1, 1, num_fraps_per_image, 1, 1)
        current_latent_single = input_latents_single.clone()

        # ---------------- Transition frames ----------------
        if previous_latent_single is not None and transition_frames > 0:
            for t in range(transition_frames):
                if stop_generation: break
                alpha = 0.5 - 0.5 * math.cos(math.pi * t / (transition_frames-1))
                latent_interp = ((1-alpha)*previous_latent_single + alpha*current_latent_single).squeeze(2).clamp(-3.0,3.0)
                if latent_interp.shape[1] != 4:
                    latent_interp = latent_interp.repeat(1, 4, 1, 1)
                frame_tensor = decode_latents_to_image_safe(latent_interp, vae)
                frame_tensor = normalize_frame(frame_tensor)
                frame_tensor = prepare_frame_tensor(frame_tensor)
                frame_pil = to_pil(frame_tensor.cpu())
                if upscale_factor > 1:
                    frame_pil = frame_pil.resize((frame_pil.width*upscale_factor, frame_pil.height*upscale_factor), Image.BICUBIC)
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frame_counter += 1
                pbar.update(1)

        # ---------------- Generation frames ----------------
        for pos_embeds, neg_embeds in embeddings:
            for f in range(num_fraps_per_image):
                if f == 0:
                    # première frame = image d'entrée
                    frame_tensor = (input_image.squeeze(0) + 1) / 2.0
                    frame_tensor = frame_tensor.clamp(0, 1)
                    frame_pil = to_pil(frame_tensor.cpu())
                else:
                    # Récupération latent T=1
                    latents_frame = input_latents[:, :, f:f+1, :, :].clone()

                    # ✅ Assurer dtype/device cohérents
                    latents_frame = latents_frame.to(device=device, dtype=dtype)
                    unet = unet.to(device=device, dtype=dtype)
                    if motion_module is not None:
                        motion_module = motion_module.to(device=device, dtype=dtype) if hasattr(motion_module, "to") else motion_module

                    # Combiner embeddings positive + negative
                    combined_embeds = pos_embeds.to(device=device, dtype=dtype) + \
                                    guidance_scale * (pos_embeds.to(device=device, dtype=dtype) - neg_embeds.to(device=device, dtype=dtype))

                    # ---------------- GENERATION SAFE ----------------
                    latents = generate_latents_safe_wrapper(
                        unet=unet,
                        scheduler=scheduler,
                        input_latents=latents_frame,
                        embeddings=combined_embeds,
                        motion_module=motion_module,
                        guidance_scale=guidance_scale,
                        device=device,
                        fp16=args.fp16,
                        steps=steps,
                        init_image_scale=init_image_scale,
                        creative_noise=creative_noise,
                        debug=True
                    )

                    # Décodage ultrasafe
                    frame_pil = decode_latents_ultrasafe_blockwise(
                        latents,
                        vae,
                        block_size=160,
                        overlap=96,
                        gamma=0.7,
                        brightness=1.2,
                        contrast=1.1,
                        saturation=1.15
                    )

                # Upscale si demandé
                if upscale_factor > 1:
                    frame_pil = frame_pil.resize((frame_pil.width*upscale_factor, frame_pil.height*upscale_factor), Image.BICUBIC)

                # Sauvegarde frame
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frame_counter += 1

                # Libération VRAM
                if f != 0:
                    del latents_frame
                    torch.cuda.empty_cache()

                pbar.update(1)

        previous_latent_single = current_latent_single.clone()
        # ---------------- FIN Main loop ----------------
    pbar.close()
    save_frames_as_video_from_folder(output_dir, out_video, fps=fps)
    print(f"🎬 Vidéo générée : {out_video}")
    print("✅ Pipeline terminé proprement.")

# ---------------- ENTRY ----------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--vae-offload", action="store_true")
    parser.add_argument("--n3_model", type=str, default="cyber_skin") #cyber_skin , cyberpunk_style_v3
    parser.add_argument("--scheduler", type=str, default="dpm", help="Scheduler à utiliser: dpm, euler, pndm")
    args = parser.parse_args()
    main(args)
