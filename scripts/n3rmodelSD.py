# --------------------------------------------------------------
# n3rmodelSD.py - AnimateDiff pipeline ultra light ~2Go VRAM
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
from scripts.utils.logging_utils import log_latent_stats, log_patch_stats
from scripts.utils.tools_utils import (
    prepare_frame_tensor,
    normalize_frame,
    tensor_to_pil,
    ensure_4_channels,
    save_frames_as_video_from_folder
)
from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_unet
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import load_images_test, generate_latents_safe_wrapper, encode_images_to_latents_safe, decode_latents_ultrasafe_blockwise_test

LATENT_SCALE = 0.18215
stop_generation = False

# ---------------- Thread stop ----------------
def wait_for_stop():
    global stop_generation
    inp = input("Appuyez sur '²' + Entrée pour arrêter : ")
    if inp.lower() == "²": stop_generation = True
threading.Thread(target=wait_for_stop, daemon=True).start()


# ------------------- ENCODE -------------------


def encode_images_to_latents_nuanced(images, vae, device="cuda", latent_scale=LATENT_SCALE):
    """
    Encode une image en latents VAE tout en préservant le contraste et les nuances de couleur.
    - Utilise la moyenne de la distribution latente
    - Clamp minimal seulement pour sécurité
    - Force 4 canaux si nécessaire
    """

    images = images.to(device=device, dtype=torch.float32)
    vae = vae.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mean  # moyenne pour plus de stabilité

    # Appliquer le scaling mais garder la dynamique
    latents = latents * latent_scale

    # Sécurité NaN / Inf (mais pas normalisation globale)
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    # Forcer 4 canaux si nécessaire (VAE attend souvent 4)
    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    return latents
# ------------------- DECODE -------------------

def decode_latents_ultrasafe_blockwise(
    latents, vae,
    block_size=32, overlap=16,
    gamma=1.2, brightness=1.0,
    contrast=1.2, saturation=1.3,
    device="cuda", frame_counter=0, output_dir=Path("."),
    epsilon=1e-6,
    latent_scale_boost=1.1  # boost léger pour récupérer les nuances
):
    """
    Décodage ultra-safe par blocs des latents en image PIL.
    Optimisé pour préserver les nuances de couleur et réduire l'effet "photocopie".
    """

    # 🔹 Correctif : forcer le VAE sur le bon device et en float32
    vae = vae.to(device=device, dtype=torch.float32)
    vae.eval()

    B, C, H, W = latents.shape
    latents = latents.to(device=device, dtype=torch.float32) * latent_scale_boost

    # Dimensions finales
    out_H = H * 8
    out_W = W * 8
    output_rgb = torch.zeros(B, 3, out_H, out_W, device=device)
    weight = torch.zeros_like(output_rgb)

    stride = block_size - overlap

    # Calcul positions garanties pour full coverage
    y_positions = list(range(0, H - block_size + 1, stride)) or [0]
    x_positions = list(range(0, W - block_size + 1, stride)) or [0]

    if y_positions[-1] != H - block_size:
        y_positions.append(H - block_size)
    if x_positions[-1] != W - block_size:
        x_positions.append(W - block_size)

    for y in y_positions:
        for x in x_positions:
            y1 = y + block_size
            x1 = x + block_size

            patch = latents[:, :, y:y1, x:x1]

            # Sécurité : NaN / Inf / epsilon
            patch = torch.nan_to_num(patch, nan=0.0, posinf=5.0, neginf=-5.0)
            if torch.all(patch == 0):
                patch += epsilon

            # Décodage
            with torch.no_grad():
                decoded = vae.decode(patch).sample.to(torch.float32)

            # Intégration dans l'image finale
            iy0, ix0 = y*8, x*8
            iy1, ix1 = y1*8, x1*8
            output_rgb[:, :, iy0:iy1, ix0:ix1] += decoded
            weight[:, :, iy0:iy1, ix0:ix1] += 1.0

    # Moyenne pour blending final
    output_rgb = output_rgb / weight.clamp(min=1e-6)
    output_rgb = output_rgb.clamp(-1.0, 1.0)

    # Convertir en PIL et appliquer corrections gamma / contrast / saturation / brightness
    frame_pil_list = []
    for i in range(B):
        img = F.to_pil_image((output_rgb[i] + 1) / 2)  # [-1,1] -> [0,1]
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        if gamma != 1.0:
            img = img.point(lambda x: 255 * ((x / 255) ** (1 / gamma)))
        frame_pil_list.append(img)

    return frame_pil_list[0] if B == 1 else frame_pil_list




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
    steps = max(cfg.get("steps",16),4)
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
    def apply_n3oray_to_unet(unet, n3oray_path, alpha=0.5):
        from scripts.utils.lora_utils import apply_lora  # ton utilitaire LoRA/n3oray
        print(f"📌 Application du style n3oray : {n3oray_path} (alpha={alpha})")
        unet = apply_lora(unet, n3oray_path, alpha=alpha)
        return unet

    # Appliquer cyber_skin depuis le YAML
    unet = apply_n3oray_to_unet(unet, cyber_skin_path, alpha=0.5)
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
        #input_latents_single = encode_images_to_latents_safe(input_image, vae, device=device, epsilon=1e-5)
        input_latents_single = encode_images_to_latents_nuanced(input_image, vae, device=device, latent_scale=LATENT_SCALE)
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
                #frame_pil = decode_latents_ultrasafe_blockwise( latent_interp, vae, block_size=32, overlap=16, gamma=0.7, brightness=1.2, contrast=1.1, saturation=1.15, device=device, frame_counter=frame_counter, output_dir=output_dir, epsilon=1e-5)

                # Paramètres boostés
                gamma = 1.0  # 1.2
                brightness = 1.0 # 1.0
                contrast = 1.5 # 1.2
                saturation = 1.5 # 1.2
                upscale_factor = 2
                #frame_counter = 0  # pour debug/log si nécessaire

                # Décodage ultra-safe (blockwise comme ton code)
                vae = vae.to(device)  # <---- c'est le correctif
                frame_pil = decode_latents_ultrasafe_blockwise(
                    latent_interp, vae,
                    block_size=32, overlap=24,
                    gamma=gamma,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    device=device,
                    frame_counter=frame_counter,
                    output_dir=Path("."),
                    epsilon=1e-5,
                    latent_scale_boost=5.71
                )


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
                    # Paramètres boostés
                    gamma = 1.0  # 1.2
                    brightness = 1.0 # 1.0
                    contrast = 1.5 # 1.2
                    saturation = 1.5 # 1.2
                    upscale_factor = 2
                    #frame_counter = 0  # pour debug/log si nécessaire

                    # Décodage ultra-safe (blockwise comme ton code)
                    frame_pil = decode_latents_ultrasafe_blockwise(
                        latents, vae,
                        block_size=32, overlap=24,
                        gamma=gamma,
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        device=device,
                        frame_counter=frame_counter,
                        output_dir=Path("."),
                        epsilon=1e-5,
                        latent_scale_boost=5.71
                    )

                    #frame_pil = decode_latents_ultrasafe_blockwise( latents, vae, block_size=32, overlap=16, gamma=0.7, brightness=1.2, contrast=1.1, saturation=1.15, device=device, frame_counter=frame_counter, output_dir=output_dir, epsilon=1e-5)
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
