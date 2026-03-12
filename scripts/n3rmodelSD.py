# --------------------------------------------------------------
# n3rmodelSD_vram2G.py - AnimateDiff pipeline ultra light ~2Go VRAM
# --------------------------------------------------------------
import os, math, threading
from pathlib import Path
from datetime import datetime
import csv  # <-- ajoute ça
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6,expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "garbage_collection_threshold:0.7,max_split_size_mb:64,expandable_segments:True"
import torch
from tqdm import tqdm
from torchvision.transforms import ToPILImage
import argparse
import torchvision.transforms as T
from diffusers import AutoencoderKL, PNDMScheduler
from transformers import CLIPTokenizerFast, CLIPTextModel
from safetensors.torch import load_file
from PIL import Image, ImageEnhance


from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import functional as F

from scripts.utils.lora_utils import apply_lora  # ou ton utilitaire n3oray

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_unet
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import load_images_test, generate_latents_safe_wrapper

LATENT_SCALE = 0.18215
stop_generation = False

# ---------------- Thread stop ----------------
def wait_for_stop():
    global stop_generation
    inp = input("Appuyez sur '²' + Entrée pour arrêter : ")
    if inp.lower() == "²": stop_generation = True
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
    if max_val > min_val: frame_tensor = (frame_tensor - min_val)/(max_val-min_val)
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

# ------------------- ENCODE -------------------
def encode_images_to_latents_safe(images, vae, device="cuda", epsilon=1e-5):
    """
    Encode des images en latents sûrs pour UNet.
    Retourne toujours un tensor [B, 4, H_latent, W_latent], dtype=vae.dtype.

    - epsilon : valeur minimale ajoutée aux latents nuls pour éviter frames noires
    """
    images_t = images.to(device=device, dtype=torch.float32)
    original_dtype = next(vae.parameters()).dtype

    vae = vae.to(device=device, dtype=torch.float32)  # safe pour l'encodage

    with torch.no_grad():
        latents = vae.encode(images_t).latent_dist.sample()

    print(f"[DEBUG encode] latents min/max après sample: {latents.min().item():.6f}/{latents.max().item():.6f}")

    # Scaling
    latents = latents * LATENT_SCALE
    print(f"[DEBUG encode] latents min/max après LATENT_SCALE: {latents.min().item():.6f}/{latents.max().item():.6f}")

    # Clamp NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    # ✅ Remplir les latents entièrement nuls
    if torch.all(latents == 0):
        latents += epsilon

    # Normalisation pour éviter overflow
    max_abs = latents.abs().max()
    if max_abs > 0:
        latents = latents / max_abs

    # Conversion en dtype final du VAE
    latents = latents.to(original_dtype)

    # ---------------- FORCE 4 CANAUX ----------------
    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)
    if latents.ndim == 5 and latents.shape[2] == 1:
        latents = latents.repeat(1, 1, 4, 1, 1)

    print(f"[DEBUG encode] shape finale latents: {latents.shape}, min/max: {latents.min().item():.6f}/{latents.max().item():.6f}")
    return latents


# ------------------- DECODE -------------------

def decode_latents_ultrasafe_blockwise(
    latents, vae,
    block_size=32, overlap=16,
    gamma=1.0, brightness=1.0,
    contrast=1.0, saturation=1.0,
    device="cuda", frame_counter=0, output_dir=Path("."),
    epsilon=1e-5  # valeur minimale pour éviter patches nuls
):
    """
    Décodage par blocs ultra-safe des latents en image PIL avec correction auto
    et debug min/max pour chaque patch et image finale.

    - latents: [B, 4, H, W] sur CPU ou GPU
    - vae: VAE déjà chargé (fp16 ou fp32)
    - block_size: taille des patches
    - overlap: chevauchement
    - gamma/brightness/contrast/saturation: correction finale
    - device: device du VAE (cuda ou cpu)
    - epsilon: valeur minimale pour patches nuls
    """
    vae_dtype = next(vae.parameters()).dtype
    B, C, H, W = latents.shape
    output_rgb = torch.zeros(B, 3, H * 8, W * 8, device=device, dtype=torch.float32)

    y_steps = list(range(0, H, block_size - overlap))
    x_steps = list(range(0, W, block_size - overlap))

    for y0 in y_steps:
        y1 = min(y0 + block_size, H)
        for x0 in x_steps:
            x1 = min(x0 + block_size, W)
            patch = latents[:, :, y0:y1, x0:x1].to(device=device, dtype=vae_dtype)

            # Debug avant VAE
            print(f"[DEBUG] patch avant VAE ({y0},{x0}): shape={patch.shape}, "
                  f"dtype={patch.dtype}, min={patch.min():.6f}, max={patch.max():.6f}")

            # ✅ Correction NaN / Inf et epsilon minimal
            patch = torch.nan_to_num(patch, nan=0.0, posinf=5.0, neginf=-5.0)
            if torch.all(patch == 0):
                patch += epsilon

            # log patch
            patch_idx = f"{y0}_{x0}"
            log_patch_stats(frame_idx=frame_counter, patch_idx=patch_idx, patch=patch, csv_path=output_dir / "patch_stats.csv")

            # Decode
            with torch.no_grad():
                patch_decoded = vae.decode(patch).sample  # [B, 3, h*8, w*8]
                patch_decoded = patch_decoded.to(torch.float32)

            # ✅ Recentrage automatique pour éviter frames sombres
            min_val = patch_decoded.min()
            max_val = patch_decoded.max()
            if max_val - min_val > 1e-6:
                patch_decoded = (patch_decoded - min_val) / (max_val - min_val)

            # Debug après VAE
            log_patch_stats(frame_idx=frame_counter, patch_idx=patch_idx+"_decoded", patch=patch_decoded, csv_path=output_dir / "patch_stats.csv")

            h_start, h_end = y0 * 8, y1 * 8
            w_start, w_end = x0 * 8, x1 * 8
            output_rgb[:, :, h_start:h_end, w_start:w_end] = patch_decoded

    # Debug avant clamp
    print(f"[DEBUG] output_rgb final avant clamp: shape={output_rgb.shape}, "
          f"min={output_rgb.min():.6f}, max={output_rgb.max():.6f}")

    output_rgb = output_rgb.clamp(0.0, 1.0)

    # Correction gamma / contraste / luminosité / saturation
    frame_pil_list = []
    for i in range(B):
        img = F.to_pil_image(output_rgb[i])
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        img = img.point(lambda x: (x / 255) ** (1 / gamma) * 255)
        frame_pil_list.append(img)

        # Debug post-correction
        pil_tensor = F.to_tensor(img)
        print(f"[DEBUG] frame {i} après PIL correction: shape={pil_tensor.shape}, "
              f"min={pil_tensor.min():.6f}, max={pil_tensor.max():.6f}")

    return frame_pil_list[0] if B == 1 else frame_pil_list


def log_latent_stats(frame_idx, latents, csv_path="latent_stats.csv"):
    """Écrit les stats latentes dans un CSV"""
    min_val = float(latents.min())
    max_val = float(latents.max())
    mean_val = float(latents.mean())
    std_val = float(latents.std())

    # Si le fichier n'existe pas, écrire l'en-tête
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["frame", "min", "max", "mean", "std"])
        writer.writerow([frame_idx, min_val, max_val, mean_val, std_val])


def log_patch_stats(frame_idx, patch_idx, patch, csv_path="patch_stats.csv"):
    """
    Écrit les stats de chaque patch VAE dans un CSV
    """
    min_val = float(patch.min())
    max_val = float(patch.max())
    mean_val = float(patch.mean())
    std_val = float(patch.std())
    shape_str = "x".join(map(str, patch.shape))
    dtype_str = str(patch.dtype)
    device_str = str(patch.device)
    any_nan = int(torch.isnan(patch).any())
    any_inf = int(torch.isinf(patch).any())

    # Mémoire GPU (si sur CUDA)
    if patch.is_cuda:
        mem_alloc = torch.cuda.memory_allocated()
        mem_reserved = torch.cuda.memory_reserved()
    else:
        mem_alloc = 0
        mem_reserved = 0

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "frame", "patch", "shape", "dtype", "device",
                "min", "max", "mean", "std", "NaN", "Inf",
                "gpu_alloc_bytes", "gpu_reserved_bytes"
            ])
        writer.writerow([
            frame_idx, patch_idx, shape_str, dtype_str, device_str,
            min_val, max_val, mean_val, std_val, any_nan, any_inf,
            mem_alloc, mem_reserved
        ])


def ensure_4_channels(latents):
    if latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)
    return latents

# ---------------- MAIN ----------------
def main(args):
    global stop_generation
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16


    fps = cfg.get("fps",12)
    upscale_factor = cfg.get("upscale_factor",1)
    transition_frames = cfg.get("transition_frames",4)
    num_fraps_per_image = cfg.get("num_fraps_per_image",4)
    steps = max(cfg.get("steps",16),3)
    guidance_scale = cfg.get("guidance_scale",4.0)
    init_image_scale = cfg.get("init_image_scale",0.85)
    creative_noise = cfg.get("creative_noise",0.0)

    # Scheduler
    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(steps, device=device)



    # UNet
    unet = safe_load_unet(args.pretrained_model_path, device=device, fp16=True)
    if hasattr(unet,"enable_attention_slicing"): unet.enable_attention_slicing()
    if hasattr(unet,"enable_xformers_memory_efficient_attention"):
        try: unet.enable_xformers_memory_efficient_attention(True)
        except: pass

    # Récupération du path cyber_skin depuis le YAML ----------------------------------- **********
    cyber_skin_path = cfg.get("n3oray_models", {}).get("cyber_skin")
    if cyber_skin_path is None:
        raise ValueError("❌ Impossible de trouver 'cyber_skin' dans n3oray_models du YAML.")

    # Fonction utilitaire pour appliquer LoRA/n3oray
    def apply_n3oray_to_unet(unet, n3oray_path, alpha=0.8):
        from scripts.utils.lora_utils import apply_lora  # ton utilitaire LoRA/n3oray
        print(f"📌 Application du style n3oray : {n3oray_path} (alpha={alpha})")
        unet = apply_lora(unet, n3oray_path, alpha=alpha)
        return unet

    # Appliquer cyber_skin depuis le YAML
    unet = apply_n3oray_to_unet(unet, cyber_skin_path, alpha=0.8)
    # ----------------------------------------------------------------------------------- **********

    # Motion module
    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None

    # Tokenizer & Text Encoder
    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path,"tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path,"text_encoder")).to(device).to(dtype)

    # VAE
    vae_path = cfg.get("vae_path")
    vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
    vae.to(device)
    vae.enable_slicing()
    vae.enable_tiling()

    # Embeddings
    prompts = cfg.get("prompt",[])
    negative_prompts = cfg.get("n_prompt",[])
    embeddings=[]
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

    # Input images
    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    total_frames = len(input_paths)*num_fraps_per_image*max(len(prompts),1)

    # statistique:
    # --------------
    total_frames = len(input_paths) * num_fraps_per_image * max(len(prompts), 1)
    estimated_seconds = total_frames / fps
    print("📌 Paramètres de génération :")
    print(f"  fps                  : {fps}")
    print(f"  num_fraps_per_image : {num_fraps_per_image}")
    print(f"  steps                : {steps}")
    print(f"  guidance_scale       : {guidance_scale}")
    print(f"  init_image_scale     : {init_image_scale}")
    print(f"  creative_noise       : {creative_noise}")
    print(f"⏱ Durée totale estimée de la vidéo : {estimated_seconds:.1f}s")

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/model_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"
    to_pil = ToPILImage()

    # Main loop
    # ---------------- MAIN LOOP MINI VRAM CORRIGE ----------------
    previous_latent_single = None
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)

    for img_idx, img_path in enumerate(input_paths):
        if stop_generation:
            break

        # ---------------- 1️⃣ Chargement image ----------------
        input_image = load_images_test([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)

        # ---------------- 2️⃣ Encodage latents ----------------
        input_latents_single = encode_images_to_latents_safe(input_image, vae, device=device, epsilon=1e-5)
        input_latents_single = ensure_4_channels(input_latents_single)  # ✅ force 4 canaux
        current_latent_single = input_latents_single.clone()  # RESTE sur GPU / dtype correct

        print(f"DEBUG: Shape latents après ensure_4_channels: {current_latent_single.shape}")

        # ---------------- 3️⃣ Transition frames ----------------
        if previous_latent_single is not None and transition_frames > 0:
            for t in range(transition_frames):
                if stop_generation:
                    break
                alpha = 0.5 - 0.5 * math.cos(math.pi * t / (transition_frames - 1))
                latent_interp = ((1 - alpha) * previous_latent_single + alpha * current_latent_single).clamp(-3, 3)
                latent_interp = latent_interp.to(device=device, dtype=dtype)

                # ⚡ Log stats latentes
                log_latent_stats(frame_counter, latent_interp, csv_path=output_dir / "latent_stats.csv")

                print(f"[DEBUG INTERPOL] frame {frame_counter}: min/max latents={latent_interp.min().item():.6f}/{latent_interp.max().item():.6f}")
                frame_pil = decode_latents_ultrasafe_blockwise(
                    latent_interp, vae,
                    block_size=32, overlap=16,
                    gamma=0.7, brightness=1.2,
                    contrast=1.1, saturation=1.15,
                    device=device, frame_counter=frame_counter, output_dir=output_dir, epsilon=1e-5)
                if upscale_factor > 1:
                    frame_pil = frame_pil.resize(
                        (frame_pil.width * upscale_factor, frame_pil.height * upscale_factor),
                        Image.BICUBIC
                    )
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frame_counter += 1
                pbar.update(1)

        # ---------------- 4️⃣ Génération frames ----------------
        # ---------------- 4️⃣ Génération frames (corrigé) ----------------
        for pos_embeds, neg_embeds in embeddings:
            for f in range(num_fraps_per_image):
                if stop_generation:
                    break

                if f == 0:
                    # Première frame = image d'entrée
                    frame_tensor = (input_image.squeeze(0) + 1) / 2.0
                    frame_tensor = frame_tensor.clamp(0, 1)
                    frame_pil = to_pil(frame_tensor.cpu())
                else:
                    # Latents frame
                    latents_frame = current_latent_single.clone()  # ✅ reste correct
                    latents_frame = ensure_4_channels(latents_frame)

                    # ⚡ CORRIGER LES FRAMES NOIRS : si latents nuls, reprendre le précédent
                    if previous_latent_single is not None:
                        if torch.all(latents_frame == 0):
                            latents_frame = previous_latent_single.clone()

                    # Appliquer embeddings
                    combined_embeds = pos_embeds.to(device=device, dtype=dtype) + guidance_scale * (pos_embeds - neg_embeds)

                    # UNet temporaire sur GPU
                    unet = unet.to(device)
                    with torch.inference_mode():
                        latents = generate_latents_safe_wrapper(
                            unet=unet,
                            scheduler=scheduler,
                            input_latents=latents_frame,
                            embeddings=combined_embeds,
                            motion_module=motion_module,
                            guidance_scale=guidance_scale,
                            device=device,
                            fp16=True,
                            steps=steps,
                            init_image_scale=init_image_scale,
                            creative_noise=creative_noise,
                            debug=False
                        )

                    # ⚡ CORRIGER LES FRAMES NOIRS : si latents générés nuls, reprendre le précédent
                    if previous_latent_single is not None:
                        if torch.all(latents == 0):
                            latents = previous_latent_single.clone()

                    unet = unet.to("cpu")
                    torch.cuda.empty_cache()

                    # ⚡ Log stats latentes
                    log_latent_stats(frame_counter, latents, csv_path=output_dir / "latent_stats.csv")

                    # ---------------- Decode frame ----------------
                    print(f"[DEBUG Génération frames] frame {frame_counter}: min/max latents={latents.min().item():.6f}/{latents.max().item():.6f}")
                    frame_pil = decode_latents_ultrasafe_blockwise(
                        latents, vae,
                        block_size=32, overlap=16,
                        gamma=0.7, brightness=1.2,
                        contrast=1.1, saturation=1.15,
                        device=device, frame_counter=frame_counter, output_dir=output_dir, epsilon=1e-5)
                    del latents
                    torch.cuda.empty_cache()

                # ---------------- Sauvegarde ----------------
                if upscale_factor > 1:
                    frame_pil = frame_pil.resize(
                        (frame_pil.width * upscale_factor, frame_pil.height * upscale_factor),
                        Image.BICUBIC
                    )
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frame_counter += 1
                pbar.update(1)

        # ---------------- Mémoriser latent pour transitions ----------------
        previous_latent_single = current_latent_single

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
    parser.add_argument("--n3_model", type=str, default="cyber_skin")
    parser.add_argument("--scheduler", type=str, default="pndm")
    args = parser.parse_args()
    main(args)
