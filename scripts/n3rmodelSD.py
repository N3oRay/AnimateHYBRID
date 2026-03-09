# --------------------------------------------------------------
# n3rmodel.py - AnimateDiff pipeline optimisé VRAM
# --------------------------------------------------------------
import os, math, threading
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageEnhance
from torchvision.transforms import ToPILImage
import argparse

from diffusers import AutoencoderKL
from transformers import CLIPTokenizerFast, CLIPTextModel
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_unet, safe_load_scheduler, generate_latents_robuste_model, decode_latents_to_image_vram_safe
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_robuste_safe_4go, load_image_file, decode_latents_to_image_auto, decode_latents_safe_vram

LATENT_SCALE = 0.18215
stop_generation = False


def prepare_frame_tensor(frame_tensor):
    """Assure que frame_tensor est [C,H,W] pour ToPIL"""
    if frame_tensor.ndim == 5:  # [B,C,T,H,W]
        frame_tensor = frame_tensor.squeeze(2)
    if frame_tensor.ndim == 4:  # [B,C,H,W]
        frame_tensor = frame_tensor.squeeze(0)
    if frame_tensor.ndim == 3 and frame_tensor.shape[0] != 3:  # [H,W,C] -> [C,H,W]
        frame_tensor = frame_tensor.permute(2,0,1)
    return frame_tensor

# ---------------- Thread stop ----------------
def wait_for_stop():
    global stop_generation
    inp = input("Appuyez sur '²' + Entrée pour arrêter : ")
    if inp.lower() == "²":
        stop_generation = True
threading.Thread(target=wait_for_stop, daemon=True).start()

# ---------------- Utils ----------------
def normalize_frame_ori(frame_tensor):
    if frame_tensor.min() < 0:
        frame_tensor = (frame_tensor + 1.0) / 2.0
    return frame_tensor.clamp(0, 1)

def normalize_frame(frame_tensor):
    min_val = frame_tensor.min()
    max_val = frame_tensor.max()
    if max_val > min_val:
        frame_tensor = (frame_tensor - min_val) / (max_val - min_val)
    return frame_tensor.clamp(0,1)

def compute_overlap(W, H, block_size, max_overlap_ratio=0.6):
    overlap = int(block_size * max_overlap_ratio)
    return min(overlap, min(W,H)//4)

def load_images(paths, W, H, device, dtype):
    all_tensors = []
    for p in paths:
        t = load_image_file(p, W, H, device, dtype)
        print(f"✅ Image chargée : {p}")
        all_tensors.append(t)
    return torch.stack(all_tensors, dim=0)

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

def encode_images_to_latents_old(images, vae):

    #images = images.to(device=vae.device, dtype=torch.float32)
    images = images.to(device=vae.device, dtype=vae.dtype)  # match dtype VAE
    images = images * 2 - 1

    with torch.inference_mode():
        latents = vae.encode(images).latent_dist.sample()

    latents = latents * LATENT_SCALE
    latents = latents.unsqueeze(2)

    return latents




def encode_images_to_latents(images, vae, device="cuda", dtype=torch.float16):

    import torch
    import torchvision.transforms as T

    LATENT_SCALE = 0.18215

    if not isinstance(images, list):
        images = [images]

    transform = T.ToTensor()
    processed = []

    for img in images:

        # PIL → Tensor
        if not isinstance(img, torch.Tensor):
            img = transform(img)

        # enlever batch dimension éventuelle
        if img.ndim == 4:
            img = img.squeeze(0)

        # grayscale H,W
        if img.ndim == 2:
            img = img.unsqueeze(0)

        # grayscale 1,H,W → 3,H,W
        if img.shape[0] == 1:
            img = img.repeat(3,1,1)

        img = img.clamp(0,1)

        processed.append(img)

    images = torch.stack(processed).to(device=device, dtype=dtype)

    # [0,1] → [-1,1]
    images = images * 2.0 - 1.0

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()

    latents = latents * LATENT_SCALE
    latents = torch.nan_to_num(latents)

    return latents

def decode_latents_to_image_auto_new(latents, vae):

    latents = torch.nan_to_num(latents, nan=0.0, posinf=4.0, neginf=-4.0)

    if latents.ndim == 5:
        latents = latents[:,:,0,:,:]

    latents = latents / LATENT_SCALE

    with torch.no_grad():
        # 🔹 S’assurer que les latents sont du même dtype que le VAE
        latents = latents.to(vae.dtype)
        image = vae.decode(latents).sample

    image = (image + 1) / 2
    image = image.clamp(0,1)

    return image


def decode_latents_to_image_bright_enhanced(latents, vae, gamma=0.7, brightness=1.2, contrast=1.1, saturation=1.15):
    """
    Décodage des latents en image PIL avec :
    - Correction gamma pour éclaircir
    - Augmentation de luminosité, contraste et saturation pour un rendu plus vivant
    """
    latents = torch.nan_to_num(latents, nan=0.0, posinf=4.0, neginf=-4.0)

    if latents.ndim == 5:  # [B,C,T,H,W]
        latents = latents[:, :, 0, :, :]

    # Revenir à l'échelle attendue par le VAE
    latents = latents / LATENT_SCALE

    with torch.no_grad():
        # 🔹 S’assurer que les latents sont du même dtype que le VAE
        latents = latents.to(vae.dtype)
        image = vae.decode(latents).sample

    # Normalisation [-1,1] -> [0,1]
    image = (image + 1.0) / 2.0
    image = image.clamp(0, 1)

    # Correction gamma
    image = image.pow(1.0 / gamma)

    # Convertir en PIL pour post-processing


    image = image[0]  # ✅ retire dimension batch
    pil_image = ToPILImage()(image.cpu().clamp(0, 1))

    # Boost luminosité, contraste et saturation
    pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)
    pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

    return pil_image


def tensor_to_pil(frame_tensor):
    import torchvision.transforms as T

    if frame_tensor.ndim == 4:
        frame_tensor = frame_tensor[0]

    frame_tensor = frame_tensor.clamp(0,1)

    return T.ToPILImage()(frame_tensor.cpu())

def build_scheduler(base_config, scheduler_name="dpm", steps=50, device="cuda"):
    """
    Crée un scheduler compatible avec différentes versions de diffusers
    et sécurise DPMSolverMultistep pour éviter l'IndexError.

    Args:
        base_config: configuration de base du scheduler (diffusers config)
        scheduler_name: "dpm", "euler", "pndm"
        steps: nombre de steps à exécuter
        device: "cuda" ou "cpu"

    Returns:
        scheduler: objet scheduler prêt à l'emploi
    """
    scheduler_name = scheduler_name.lower()
    try:
        if scheduler_name in ["dpm", "dpm++", "dpm_solver"]:
            from diffusers import DPMSolverMultistepScheduler
            scheduler = DPMSolverMultistepScheduler.from_config(
                base_config,
                algorithm_type="sde-dpmsolver++",
                use_karras_sigmas=True
            )
            # 🔹 Sécurisation steps vs sigmas
            max_safe_steps = max(len(scheduler.sigmas)-2, 1)
            safe_steps = min(steps, max_safe_steps)
            scheduler.set_timesteps(safe_steps, device=device)
            print(f"✅ Scheduler DPMSolverMultistep utilisé avec steps={safe_steps}")

        elif scheduler_name in ["euler"]:
            from diffusers import EulerDiscreteScheduler
            scheduler = EulerDiscreteScheduler.from_config(base_config)
            scheduler.set_timesteps(steps, device=device)
            print(f"✅ Scheduler EulerDiscreteScheduler utilisé avec steps={steps}")

        elif scheduler_name in ["pndm"]:
            from diffusers import PNDMScheduler
            scheduler = PNDMScheduler.from_config(base_config)
            scheduler.set_timesteps(steps, device=device)
            print(f"✅ Scheduler PNDMScheduler utilisé avec steps={steps}")

        else:
            from diffusers import PNDMScheduler
            scheduler = PNDMScheduler.from_config(base_config)
            scheduler.set_timesteps(steps, device=device)
            print("⚠️ Scheduler inconnu → fallback PNDM utilisé")

        return scheduler

    except Exception as e:
        print("⚠️ Impossible de créer le scheduler demandé:", e)
        from diffusers import PNDMScheduler
        scheduler = PNDMScheduler.from_config(base_config)
        scheduler.set_timesteps(min(steps, len(scheduler.sigmas)-2), device=device)
        print("⚠️ Fallback scheduler : PNDMScheduler utilisé")
        return scheduler

# ---------------- MAIN ----------------
def main(args):
    global stop_generation
    cfg = load_config(args.config)

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32

    fps = cfg.get("fps", 12)
    upscale_factor = cfg.get("upscale_factor", 2)
    transition_frames = cfg.get("transition_frames", 8)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 12)
    # 🔹 Contrôle minimal pour éviter IndexError DPM-Solver
    # Scheduler
    MIN_STEPS = 25  # minimum sûr pour DPM-Solver
    steps = max(cfg.get("steps", 50), MIN_STEPS)

    # ---------------- Scheduler ----------------
    # ---------------- Scheduler ----------------
    from diffusers import PNDMScheduler

    base_scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000
    )

    scheduler = build_scheduler(base_scheduler.config, args.scheduler)
    # 🔧 IMPORTANT
    scheduler.set_timesteps(steps, device=device)
    # -------------------------------------------

    # -------------------------------------------

    guidance_scale = cfg.get("guidance_scale", 4.5)
    init_image_scale = cfg.get("init_image_scale", 0.85)
    creative_noise = cfg.get("creative_noise", 0.0)

    # ---------------- LOAD UNET ----------------
    n3_model_name = args.n3_model
    n3_model_path = cfg["n3oray_models"].get(n3_model_name)
    if n3_model_path is None:
        raise ValueError(f"N3 model '{n3_model_name}' non défini dans le YAML")
    print(f"✅ Chargement du modèle N3 '{n3_model_name}' depuis : {n3_model_path}")

    # Charger le UNet depuis le checkpoint safetensors
    from safetensors.torch import load_file

    state_dict = load_file(n3_model_path, device="cpu")  # d'abord sur CPU
    unet = safe_load_unet(args.pretrained_model_path, device=device, fp16=True)
    unet.load_state_dict(state_dict, strict=False)
    if hasattr(unet, "enable_attention_slicing"):
        unet.enable_attention_slicing()
    print(f"✅ UNet N3 chargé correctement depuis {n3_model_path}")

    # Stabilisation diffusion
    if hasattr(scheduler.config, "use_karras_sigmas"):
        scheduler.config.use_karras_sigmas = True

    if hasattr(scheduler.config, "lower_order_final"):
        scheduler.config.lower_order_final = True

    # ---------------- Motion module ----------------
    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None
    print(f"✅ Motion module chargé")

    # ---------------- Tokenizer & Text Encoder ----------------
    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path,"tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path,"text_encoder")).to(device)
    #text_encoder = text_encoder.half()
    text_encoder = text_encoder.to(dtype)

    print(f"✅ tokenizer - text_encoder module chargé")

    # ---------------- VAE sur CPU ----------------
    # ---------------- VAE ----------------
    vae_path = cfg.get("vae_path")
    if not vae_path or not os.path.isfile(vae_path):
        raise ValueError(f"Chemin VAE invalide : {vae_path}")

    vae = AutoencoderKL.from_single_file(
        vae_path,
        torch_dtype=torch.float16 if args.fp16 else torch.float32
    )

    vae.to(device)
    vae.enable_slicing()
    print(f"✅ VAE chargé depuis : {vae_path} avec dtype={'float16' if args.fp16 else 'float32'}")

    # ---------------- Embeddings ----------------
    prompts = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])
    embeddings = []
    for prompt_item in prompts:
        prompt_text = " ".join(prompt_item) if isinstance(prompt_item,list) else str(prompt_item)
        neg_text = " ".join(negative_prompts) if isinstance(negative_prompts,list) else str(negative_prompts)
        text_inputs = tokenizer(prompt_text, padding="max_length", truncation=True,
                                max_length=tokenizer.model_max_length, return_tensors="pt")
        neg_inputs = tokenizer(neg_text, padding="max_length", truncation=True,
                               max_length=tokenizer.model_max_length, return_tensors="pt")
        with torch.no_grad():
            pos_embeds = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state
            neg_embeds = text_encoder(neg_inputs.input_ids.to(device)).last_hidden_state
        embeddings.append((pos_embeds.to(dtype), neg_embeds.to(dtype)))

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
    stop_generation = False

    # ---------------- Main loop ----------------
    for img_idx, img_path in enumerate(input_paths):
        if stop_generation:
            break
        input_image = load_images([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
        input_latents_single = encode_images_to_latents(input_image, vae)
        # 🔹 Sécurisation 1-channel
        if input_latents_single.shape[1] == 1:
            input_latents_single = input_latents_single.repeat(1, 4, 1, 1)
        input_latents = input_latents_single.repeat(1,1,num_fraps_per_image,1,1)
        current_latent_single = input_latents_single.clone()
        block_size = cfg.get("block_size",64)
        overlap = cfg.get("overlap", compute_overlap(cfg["W"], cfg["H"], block_size))

        # Transition latente
        if previous_latent_single is not None and transition_frames > 0:
            for t in range(transition_frames):
                if stop_generation: break
                alpha = 0.5 - 0.5*math.cos(math.pi*t/(transition_frames-1))
                latent_interp = ((1 - alpha) * previous_latent_single + alpha * current_latent_single)
                latent_interp = latent_interp.squeeze(2).clamp(-3.0, 3.0)

                # 🔹 Correctif reshape automatique pour transition latente
                if latent_interp.ndim == 4 and latent_interp.shape[0] != 4:
                    latent_interp = latent_interp.repeat(1,4,1,1)

                frame_tensor = decode_latents_to_image_auto_new(latent_interp, vae) # original
                # Normalisation finale (optionnel)
                frame_tensor = normalize_frame(frame_tensor)

                # ***** Correctif dimension:
                frame_tensor = prepare_frame_tensor(frame_tensor)
                if t == 0:
                    frame_pil = to_pil(frame_tensor.cpu())
                else:
                    frame_pil = to_pil(frame_tensor.cpu()).filter(ImageFilter.GaussianBlur(0.2))
                if upscale_factor>1:
                    frame_pil = frame_pil.resize((frame_pil.width*upscale_factor, frame_pil.height*upscale_factor), Image.BICUBIC)
                #frame_pil.save(Path(f"./outputs/frame_{frame_counter:05d}.png"))
                frame_path = output_dir / f"frame_{frame_counter:05d}.png"
                frame_pil.save(frame_path)
                frame_counter += 1
                pbar.update(1)

        # Génération frames
        for pos_embeds, neg_embeds in embeddings:
            for f in range(num_fraps_per_image):
                if f==0:
                    frame_tensor = (input_image.squeeze(0)+1.0)/2.0
                    frame_tensor = frame_tensor.clamp(0,1)
                    frame_pil = to_pil(frame_tensor.cpu())
                else:
                    # 🔹 Génération latents safe 4Go
                    latents_frame = input_latents[:,:,f:f+1,:,:].clone()
                    latents_frame = generate_latents_robuste_safe_4go(
                        latents_frame, pos_embeds, neg_embeds, unet, scheduler,
                        motion_module, device, dtype,
                        guidance_scale, init_image_scale, creative_noise, seed=frame_counter
                    )

                    # 🔹 Clamp / NaN
                    latents_frame = torch.nan_to_num(latents_frame, nan=0.0, posinf=5.0, neginf=-5.0)
                    latents_frame = latents_frame.clamp(-5.0, 5.0)

                    # 🔹 Décodage safe avec boost luminosité/contraste
                    frame_pil = decode_latents_to_image_bright_enhanced(
                        latents_frame, vae,
                        gamma=0.7,
                        brightness=1.2,
                        contrast=1.1,
                        saturation=1.15
                    )

                # 🔹 Upscale si demandé
                if upscale_factor>1:
                    frame_pil = frame_pil.resize(
                        (frame_pil.width*upscale_factor, frame_pil.height*upscale_factor),
                        Image.BICUBIC
                    )

                # 🔹 Sauvegarde frame
                frame_path = output_dir / f"frame_{frame_counter:05d}.png"
                frame_pil.save(frame_path)

                frame_counter += 1
                if f!=0: del latents_frame
                if frame_counter % 10 == 0: torch.cuda.empty_cache()
                pbar.update(1)
                # Génération frames
        previous_latent_single = current_latent_single.clone()

    pbar.close()
    save_frames_as_video_from_folder(output_dir, out_video, fps=fps)
    print(f"🎬 Vidéo générée : {out_video}")
    print("✅ Pipeline terminé proprement.")

# ---------------- ENTRY ---------------- --n3_model "cybersamurai_v2" \
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--vae-offload", action="store_true")
    parser.add_argument("--n3_model", type=str, default="cyberpunk_style_v3") # cyber_skin ou cyberpunk_style_v3 ou cybersamurai_v2
    parser.add_argument("--scheduler", type=str, default="dpm", help="Scheduler à utiliser: dpm, euler, pndm")
    args = parser.parse_args()
    main(args)
