# -------------------------
# n3rHYBRID17_SAFE_256_VRAM_TILED_LOGS.py
# -------------------------
import os, time, csv
from pathlib import Path
from datetime import datetime
import torch, numpy as np
from PIL import Image
import cv2

from transformers import CLIPTokenizerFast, CLIPTextModel

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_vae_stable, safe_load_unet, safe_load_scheduler
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import load_images, generate_latents_ai_5D_stable, decode_latents_correct

LATENT_SCALE = 0.18215
torch.backends.cuda.matmul.allow_tf32 = True

# -------------------------
# Fonctions utilitaires
# -------------------------
def save_frame(img_array, filename):
    """Sauvegarde une image [H,W,3] en PNG"""
    img_array = np.clip(img_array, 0.0, 1.0)
    img_uint8 = (img_array * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    Image.fromarray(img_uint8).save(filename)

def compute_rgb_stats(img_tensor):
    """Retourne min/max/moyenne par canal R,G,B pour un tensor [1,3,H,W]"""
    t = img_tensor[0].permute(1,2,0).cpu().numpy()  # [H,W,3]
    stats = []
    for i in range(3):
        channel = t[:,:,i]
        stats.extend([float(channel.min()), float(channel.max()), float(channel.mean())])
    return stats  # [Rmin,Rmax,Rmean,Gmin,Gmax,Gmean,Bmin,Bmax,Bmean]

def encode_tile(tile_rgb, vae, fp16=False):
    """Encode une tile RGB [1,3,H,W] -> latent [1,4,H/8,W/8]"""
    device = next(vae.parameters()).device
    dtype = torch.float16 if fp16 else torch.float32
    tile_rgb = tile_rgb.to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = vae.encode(tile_rgb).latent_dist.sample() * LATENT_SCALE
    return latent

def tile_image(img_tensor, tile_size=128, overlap=32):
    """Découpe un tensor [B,3,H,W] en tiles RGB"""
    B,C,H,W = img_tensor.shape
    stride = tile_size - overlap
    tiles, positions = [], []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, y2 = y, min(y+tile_size,H)
            x1, x2 = x, min(x+tile_size,W)
            tile = img_tensor[:,:,y1:y2,x1:x2]
            tiles.append(tile)
            positions.append((y1,y2,x1,x2))
    return tiles, positions

def merge_tiles(tiles, positions, H, W):
    """Fusionne les tiles [1,3,h,w] en image finale [1,3,H,W]"""
    device = tiles[0].device
    out = torch.zeros((tiles[0].shape[0], 3, H, W), device=device)
    count = torch.zeros((tiles[0].shape[0], 3, H, W), device=device)
    for t,(y1,y2,x1,x2) in zip(tiles,positions):
        th,tw = t.shape[2], t.shape[3]
        out[:,:,y1:y1+th,x1:x1+tw] += t
        count[:,:,y1:y1+th,x1:x1+tw] += 1.0
    out /= count.clamp(min=1.0)
    return out

# -------------------------
# MAIN
# -------------------------
def main(args):
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32

    fps = cfg.get("fps",12)
    num_fraps_per_image = cfg.get("num_fraps_per_image",12)
    steps = cfg.get("steps",10)
    seed = cfg.get("seed",42)
    guidance_scale = cfg.get("guidance_scale",7.5)
    creative_noise = cfg.get("creative_noise",0.03)
    tile_size = cfg.get("tile_size",128)
    tile_overlap = cfg.get("tile_overlap",32)

    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    prompts = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])

    # -------------------------
    # Load models
    # -------------------------
    unet = safe_load_unet(args.pretrained_model_path, device, fp16=args.fp16)
    vae = safe_load_vae_stable(args.pretrained_model_path, device, fp16=False, offload=args.vae_offload)
    scheduler = safe_load_scheduler(args.pretrained_model_path)
    if not vae or not unet or not scheduler:
        print("❌ Un ou plusieurs modèles manquent.")
        return

    vae = vae.float()
    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None

    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path,"tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path,"text_encoder")).to(device)
    if args.fp16: text_encoder = text_encoder.half()

    # Encode prompts
    embeddings = []
    for prompt_item in prompts:
        prompt_text = " ".join(prompt_item) if isinstance(prompt_item,list) else str(prompt_item)
        neg_text = " ".join(negative_prompts) if isinstance(negative_prompts,list) else str(negative_prompts)
        text_inputs = tokenizer(prompt_text,padding="max_length",truncation=True,
                                max_length=tokenizer.model_max_length,return_tensors="pt")
        neg_inputs = tokenizer(neg_text,padding="max_length",truncation=True,
                               max_length=tokenizer.model_max_length,return_tensors="pt")
        with torch.no_grad():
            pos_embeds = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state
            neg_embeds = text_encoder(neg_inputs.input_ids.to(device)).last_hidden_state
        embeddings.append((pos_embeds.to(dtype),neg_embeds.to(dtype)))

    unet.eval()
    try: unet.enable_xformers_memory_efficient_attention()
    except: pass

    # -------------------------
    # OUTPUT
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/hybrid17_256_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_frames"; debug_dir.mkdir(exist_ok=True)
    csv_file = output_dir / "generation_log.csv"

    video = None
    frame_counter = 0
    with open(csv_file,"w",newline="") as f_csv:
        writer = csv.writer(f_csv)
        # Ajout des stats R/G/B pour les tiles et frame finale
        writer.writerow([
            "frame","latent_min","latent_max",
            "tile_R_min","tile_R_max","tile_R_mean",
            "tile_G_min","tile_G_max","tile_G_mean",
            "tile_B_min","tile_B_max","tile_B_mean",
            "final_R_min","final_R_max","final_R_mean",
            "final_G_min","final_G_max","final_G_mean",
            "final_B_min","final_B_max","final_B_mean",
            "gen_time","decode_time"
        ])

        for img_path in input_paths:
            input_image = load_images([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)

            # --- Tiling ---
            tiles, positions = tile_image(input_image, tile_size, tile_overlap)

            for f_idx in range(num_fraps_per_image):
                decoded_tiles = []
                # stats tiles accumulées
                tile_stats_acc = []

                for tile_rgb in tiles:
                    # Encode tile RGB -> latent
                    tile_latent = encode_tile(tile_rgb, vae, fp16=False)

                    # Génération latent
                    gen_start = time.time()
                    pos_embeds, neg_embeds = embeddings[0]
                    batch_latents = generate_latents_ai_5D_stable(
                        latent_frame=tile_latent,
                        scheduler=scheduler,
                        pos_embeds=pos_embeds,
                        neg_embeds=neg_embeds,
                        unet=unet,
                        motion_module=motion_module,
                        device=device,
                        dtype=dtype,
                        guidance_scale=guidance_scale,
                        creative_noise=creative_noise,
                        seed=seed + f_idx,
                        steps=steps
                    )
                    gen_time = time.time() - gen_start

                    # Décodage tile
                    decode_start = time.time()
                    if args.vae_offload: vae.to(device)
                    decoded_tile = decode_latents_correct(batch_latents, vae)
                    decode_time = time.time() - decode_start
                    if args.vae_offload:
                        vae.cpu(); torch.cuda.synchronize() if device.startswith("cuda") else None

                    decoded_tiles.append(decoded_tile)
                    tile_stats_acc.append(compute_rgb_stats(decoded_tile))

                # fusion tiles
                final_frame = merge_tiles(decoded_tiles, positions, H=input_image.shape[2], W=input_image.shape[3])
                final_stats = compute_rgb_stats(final_frame)

                # frame finale sauvegarde PIL
                frame_array = final_frame[0].clamp(0,1).permute(1,2,0).cpu().numpy()
                save_frame(frame_array, debug_dir/f"frame_{frame_counter:05d}_pil.png")

                # sauvegarde video OpenCV
                if video is None:
                    h,w = frame_array.shape[:2]
                    video_path = output_dir/"animation.mp4"
                    video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
                video.write(cv2.cvtColor((frame_array*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                # --- Ecriture CSV ---
                # moyenne des stats de tiles
                tile_stats_mean = np.mean(np.array(tile_stats_acc), axis=0)
                writer.writerow([
                    frame_counter,
                    float(batch_latents.min()), float(batch_latents.max()),
                    *tile_stats_mean,
                    *final_stats,
                    round(gen_time,4), round(decode_time,4)
                ])
                frame_counter += 1

    if video: video.release()
    print("✅ Génération 256x256 VRAM-safe avec tiles et logs RGB terminée.")

# -------------------------
# Entrée
# -------------------------
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path",type=str,required=True)
    parser.add_argument("--config",type=str,required=True)
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--fp16",action="store_true")
    parser.add_argument("--vae-offload",action="store_true")
    args = parser.parse_args()
    main(args)
