# --------------------------------------------------------------
# n3rmodelSD_vram2G.py - AnimateDiff pipeline ultra light ~2Go VRAM
# --------------------------------------------------------------
import os, math, threading
from pathlib import Path
from datetime import datetime
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

# ---------------- Encode / Decode ----------------
def encode_images_to_latents_safe(images, vae, device="cuda", dtype=torch.float16):
    images_t = images.to(device=device, dtype=torch.float32)
    original_dtype = next(vae.parameters()).dtype
    vae = vae.to(torch.float32)
    with torch.no_grad():
        latents = vae.encode(images_t).latent_dist.sample()
    vae = vae.to(original_dtype)
    latents = latents * LATENT_SCALE
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)
    max_abs = latents.abs().max()
    if max_abs > 0: latents = latents / max_abs
    latents = latents.to(dtype)
    if latents.ndim == 4: latents = latents.unsqueeze(2)
    return latents

def decode_latents_ultrasafe_blockwise(
    latents,
    vae,
    block_size=48,
    overlap=24,
    gamma=1.0,
    brightness=1.0,
    contrast=1.0,
    saturation=1.0
):
    """
    Décodage ultra-safe des latents en RGB, patch-wise.
    Supporte :
      - images : [B, C, H, W]
      - vidéos : [B, F, C, H, W]
    """
    device = next(vae.parameters()).device

    # Détecter si c'est une vidéo
    is_video = latents.ndim == 5
    if is_video:
        B, F, C, H, W = latents.shape
        latents = latents.reshape(B*F, C, H, W)
    else:
        B, C, H, W = latents.shape

    stride = block_size - overlap
    h_steps = max(1, (H - overlap + stride - 1) // stride)
    w_steps = max(1, (W - overlap + stride - 1) // stride)

    output_rgb = torch.zeros(latents.size(0), 3, H, W, device="cpu")
    weight_map = torch.zeros(latents.size(0), 1, H, W, device="cpu")

    for i in range(h_steps):
        for j in range(w_steps):
            y0, x0 = i * stride, j * stride
            y1, x1 = min(y0 + block_size, H), min(x0 + block_size, W)

            patch = latents[:, :, y0:y1, x0:x1].to(device)
            patch = torch.nan_to_num(patch, nan=0.0, posinf=5.0, neginf=-5.0)

            with torch.no_grad():
                patch = patch.to(vae.device, dtype=vae.dtype)
                patch_decoded = vae.decode(patch).sample
                patch_decoded = ((patch_decoded + 1) / 2).clamp(0, 1).cpu()

            del patch
            torch.cuda.empty_cache()

            # Redimensionner pour correspondre au bloc exact
            patch_decoded = torch.nn.functional.interpolate(
                patch_decoded,
                size=(y1 - y0, x1 - x0),
                mode='bilinear',
                align_corners=False
            )

            mask = torch.ones(1, 1, y1 - y0, x1 - x0)
            output_rgb[:, :, y0:y1, x0:x1] += patch_decoded * mask
            weight_map[:, :, y0:y1, x0:x1] += mask
            torch.cuda.empty_cache()

    output_rgb = output_rgb / weight_map.clamp(min=1e-5)

    # Conversion en frames PIL
    if is_video:
        frames_pil = []
        for f in range(F):
            frame_tensor = output_rgb[f::F]  # toutes les images pour cette frame
            frame_pil = ToPILImage()(frame_tensor[0].clamp(0, 1))
            # Ajustements gamma / brightness / contrast / saturation
            if gamma != 1.0:
                frame_pil = ImageEnhance.Brightness(frame_pil).enhance(gamma)
            if brightness != 1.0:
                frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
            if contrast != 1.0:
                frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
            if saturation != 1.0:
                frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)
            frames_pil.append(frame_pil)
        return frames_pil
    else:
        frame_pil = ToPILImage()(output_rgb[0].clamp(0, 1))
        if gamma != 1.0:
            frame_pil = ImageEnhance.Brightness(frame_pil).enhance(gamma)
        if brightness != 1.0:
            frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
        if contrast != 1.0:
            frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
        if saturation != 1.0:
            frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)
        return frame_pil


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

        # Chargement de l'image
        input_image = load_images_test([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)

        # Encodage en latents et forcage 4 canaux IMMÉDIATEMENT
        input_latents_single = encode_images_to_latents_safe(input_image, vae, device=device, dtype=dtype)
        input_latents_single = ensure_4_channels(input_latents_single)  # ✅ 4 canaux forcés ici
        current_latent_single = input_latents_single.clone().cpu()       # Latents sur CPU

        print(f"DEBUG: Shape latents après ensure_4_channels: {current_latent_single.shape}")

        # ---------------- Transition frames ----------------
        if previous_latent_single is not None and transition_frames > 0:
            for t in range(transition_frames):
                if stop_generation:
                    break
                alpha = 0.5 - 0.5 * math.cos(math.pi * t / (transition_frames - 1))
                latent_interp = ((1 - alpha) * previous_latent_single + alpha * current_latent_single).clamp(-3, 3)
                frame_tensor = prepare_frame_tensor(latent_interp)
                frame_pil = to_pil(frame_tensor)
                if upscale_factor > 1:
                    frame_pil = frame_pil.resize(
                        (frame_pil.width * upscale_factor, frame_pil.height * upscale_factor),
                        Image.BICUBIC
                    )
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frame_counter += 1
                pbar.update(1)

        # ---------------- Generation frames ----------------
        # ---------------- Generation frames (corrigé) ----------------
        for pos_embeds, neg_embeds in embeddings:
            for f in range(num_fraps_per_image):
                if stop_generation:
                    break

                if f == 0:
                    # Première frame = image d'entrée
                    frame_tensor = (input_image.squeeze(0) + 1) / 2.0
                    frame_tensor = frame_tensor.clamp(0, 1)
                    frame_pil_or_list = to_pil(frame_tensor.cpu())
                else:
                    # Latents frame -> CPU
                    latents_frame = current_latent_single.clone().cpu()

                    # Assure 4 canaux
                    latents_frame = ensure_4_channels(latents_frame)

                    # Aplatir la dimension F si c'est 1
                    if latents_frame.ndim == 5 and latents_frame.shape[2] == 1:
                        latents_frame = latents_frame.squeeze(2)  # devient [B, C, H, W]

                    # Passer sur GPU
                    latents_frame = latents_frame.to(device="cuda", dtype=dtype)

                    # Combiner embeddings positif/négatif
                    combined_embeds = pos_embeds.to(device="cuda", dtype=dtype) + guidance_scale * (pos_embeds - neg_embeds)

                    # UNet temporaire sur GPU
                    unet = unet.to("cuda")
                    with torch.inference_mode():
                        latents = generate_latents_safe_wrapper(
                            unet=unet,
                            scheduler=scheduler,
                            input_latents=latents_frame,
                            embeddings=combined_embeds,
                            motion_module=motion_module,
                            guidance_scale=guidance_scale,
                            device="cuda",
                            fp16=True,
                            steps=steps,
                            init_image_scale=init_image_scale,
                            creative_noise=creative_noise,
                            debug=False
                        )
                    unet = unet.to("cpu")
                    torch.cuda.empty_cache()

                    # Décodage frame par frame (mini VRAM)
                    frame_pil_or_list = decode_latents_ultrasafe_blockwise(
                        latents.cpu(), vae,
                        block_size=32, overlap=16,
                        gamma=0.7, brightness=1.2,
                        contrast=1.1, saturation=1.15
                    )
                    del latents
                    torch.cuda.empty_cache()

                # ---------------- Sauvegarde ----------------
                if isinstance(frame_pil_or_list, list):
                    for frame_pil in frame_pil_or_list:
                        if upscale_factor > 1:
                            frame_pil = frame_pil.resize(
                                (frame_pil.width * upscale_factor, frame_pil.height * upscale_factor),
                                Image.BICUBIC
                            )
                        frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                        frame_counter += 1
                        pbar.update(1)
                else:
                    frame_pil = frame_pil_or_list
                    if upscale_factor > 1:
                        frame_pil = frame_pil.resize(
                            (frame_pil.width * upscale_factor, frame_pil.height * upscale_factor),
                            Image.BICUBIC
                        )
                    frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                    frame_counter += 1
                    pbar.update(1)

        # Mémoriser le latent courant pour transitions
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
