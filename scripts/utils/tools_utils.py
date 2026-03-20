# --------------------------------------------------------------
# tools_utils.py - Fonctions utilitaires génériques
# --------------------------------------------------------------
import os, math
import hashlib
from PIL import Image, ImageEnhance
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from .fx_utils import apply_post_processing_adaptive

LATENT_SCALE = 0.18215  # valeur globale, peut être importée si nécessaire

import json
import torch
from pathlib import Path

def save_input_frame(input_image, output_dir, frame_counter, pbar=None,
                     blur_radius=0.0, contrast=1.0, saturation=1.0, apply_post=False):
    try:
        from torchvision.transforms.functional import to_pil_image

        # Tensor → CPU + clamp
        img = input_image[0].detach().cpu().clamp(-1, 1)

        # [-1,1] → [0,1]
        img = (img + 1) / 2

        # → PIL
        img_pil = to_pil_image(img)

        # Option post-process
        if apply_post:
            img_pil = apply_post_processing_adaptive(
                img_pil,
                blur_radius=blur_radius,
                contrast=contrast,
                brightness=1.0,
                saturation=saturation
            )

        # Save
        img_pil.save(output_dir / f"frame_{frame_counter+1:05d}_input.png")
        print(f"[INPUT SAVE Frame {frame_counter:03d}]")

        # Update compteur + progress bar
        frame_counter += 1
        if pbar:
            pbar.update(1)

        return frame_counter

    except Exception as e:
        print(f"[INPUT SAVE ERROR] {e}")
        return frame_counter

def get_dynamic_latent_injection(frame_counter, total_frames, start=0.90, end=0.55, mode="cosine"):
    """
    Calcule latent_injection pour chaque frame, avec protection contre division par zéro.
    """
    if total_frames <= 1:
        return start  # Pas de progression possible

    t = frame_counter / (total_frames - 1)  # toujours safe
    if mode == "cosine":
        alpha = 0.5 - 0.5 * math.cos(math.pi * t)
    else:
        alpha = t
    latent_injection = start + (end - start) * alpha
    return min(max(latent_injection, 0.0), 1.0)

# -------------------------------------------------------------------------------------------
# --- Sélection simple des embeddings prompts par frame ---
def get_embeddings_for_frame(frame_idx, frames_per_prompt, pos_list, neg_list, device="cuda"):
    #Retourne les embeddings du prompt correspondant à la frame_idx. Chaque prompt produit `frames_per_prompt` frames consécutives.
    num_prompts = len(pos_list)
    prompt_idx = min(frame_idx // frames_per_prompt, num_prompts - 1)
    return pos_list[prompt_idx].to(device), neg_list[prompt_idx].to(device)


def adapt_embeddings_to_unet(pos_embeds, neg_embeds, target_dim):
    """Adapte automatiquement les embeddings texte pour correspondre au cross_attention_dim du UNet."""
    current_dim = pos_embeds.shape[-1]
    if current_dim == target_dim:
        return pos_embeds, neg_embeds
    # Troncature
    if current_dim > target_dim:
        pos_embeds = pos_embeds[..., :target_dim]
        neg_embeds = neg_embeds[..., :target_dim]
    # Padding
    elif current_dim < target_dim:
        pad = target_dim - current_dim
        pos_embeds = torch.nn.functional.pad(pos_embeds, (0, pad))
        neg_embeds = torch.nn.functional.pad(neg_embeds, (0, pad))
    return pos_embeds, neg_embeds

def compute_weighted_params(frame_idx, total_frames,
                            init_start=0.85, init_end=0.5,
                            noise_start=0.0, noise_end=0.08,
                            guidance_start=3.5, guidance_end=4.5,
                            mode="cosine"):
    """
    Calcule init_image_scale, creative_noise et guidance_scale de manière pondérée.
    Ajuste guidance en fonction du signal de l'image et du bruit.
    """
    # interpolation linéaire ou cosinus
    def interp(a, b, t, mode="cosine"):
        if mode=="cosine":
            mu = (1 - math.cos(math.pi * t)) / 2
        else:
            mu = t
        return a*(1-mu) + b*mu

    t = frame_idx / max(total_frames-1,1)
    init_scale = interp(init_start, init_end, t, mode)
    creative_noise = interp(noise_start, noise_end, t, mode)

    # pondération guidance_scale :
    # si init_scale élevé → fidèle → guidance plus faible
    # si init_scale faible → moins fidèle → guidance plus créative
    base_guidance = interp(guidance_start, guidance_end, t, mode)
    weighted_guidance = base_guidance * (1 + 0.5*(1-init_scale)) * (1 - 0.5*creative_noise)

    return init_scale, creative_noise, weighted_guidance

def load_external_embedding_as_latent(path, target_shape):
    from safetensors.torch import load_file
    emb = load_file(path)
    clip_vec = list(emb.values())[0]

    # projection simple
    latent = clip_vec.mean() * torch.randn(target_shape)
    return latent

def inject_external_embeddings(
    latents,
    external_embeddings,
    device,
    normalize=True,
    clamp_range=(-1.0, 1.0)
):
    """
    Injecte des embeddings latents externes dans les latents principaux.

    Args:
        latents (torch.Tensor): latents [B,C,H,W]
        external_embeddings (list[dict]): liste de dicts avec :
            - "latent": tensor
            - "weight": float
            - "type": "positive" ou "negative"
        device (str): device cible
        normalize (bool): normalise les embeddings pour éviter domination
        clamp_range (tuple): clamp final

    Returns:
        torch.Tensor: latents modifiés
    """

    if not external_embeddings:
        return latents

    latents = latents.to(device)

    for emb in external_embeddings:
        try:
            ext = emb.get("latent", None)
            weight = float(emb.get("weight", 0.0))
            emb_type = emb.get("type", "positive")

            if ext is None or weight == 0.0:
                continue

            # --- Device + dtype safe ---
            ext = ext.to(device=device, dtype=latents.dtype)

            # --- Resize si nécessaire ---
            if ext.shape != latents.shape:
                ext = torch.nn.functional.interpolate(
                    ext,
                    size=latents.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

                # Ajustement batch/channel si besoin
                if ext.shape[1] != latents.shape[1]:
                    ext = ext[:, :latents.shape[1], :, :]

            # --- Nettoyage ---
            ext = torch.nan_to_num(ext)

            # --- Normalisation (important) ---
            if normalize:
                ext_std = ext.std()
                lat_std = latents.std()

                if ext_std > 1e-6:
                    ext = ext * (lat_std / ext_std)

            # --- Injection ---
            if emb_type == "negative":
                latents = latents - weight * ext
            else:
                latents = latents + weight * ext

        except Exception as e:
            print(f"[inject_external_embeddings ERROR] {e}")
            continue

    # --- Clamp + sécurité ---
    latents = torch.clamp(latents, clamp_range[0], clamp_range[1])
    latents = torch.nan_to_num(latents)

    return latents


def update_n3r_memory(memory_dict, cf_embeds, n3r_latents, memory_alpha=0.15):
    prompt_bytes = cf_embeds[0,0].cpu().numpy().tobytes()
    prompt_key = hashlib.sha256(prompt_bytes).hexdigest()
    print(f"[DEBUG] clé mémoire : {prompt_key[:8]}..., latents fusionnés")
    previous_memory = memory_dict.get(prompt_key, torch.zeros_like(n3r_latents))
    fused_latents = (1 - memory_alpha) * previous_memory.to(n3r_latents.device) + memory_alpha * n3r_latents
    memory_dict[prompt_key] = fused_latents.detach().cpu()
    return fused_latents

# ------------------- Sauvegarde mémoire -------------------
def save_memory(memory_dict, memory_file: Path):
    """
    Sauvegarde la mémoire N3R au format JSON.
    Convertit les tensors en listes pour compatibilité JSON.
    """
    serializable_memory = {k: v.tolist() for k, v in memory_dict.items()}
    memory_file = memory_file.with_suffix(".json")
    memory_file.parent.mkdir(parents=True, exist_ok=True)  # créer dossier si absent
    with open(memory_file, "w") as f:
        json.dump(serializable_memory, f, indent=2)
    print(f"💾 Mémoire N3R sauvegardée : {memory_file}")


# ------------------- Chargement mémoire -------------------
def load_memory(memory_file: Path):
    """
    Charge la mémoire N3R depuis un fichier JSON.
    Convertit les listes en tensors.
    """
    memory_file = memory_file.with_suffix(".json")
    if memory_file.exists():
        with open(memory_file, "r") as f:
            mem = json.load(f)
        # Convertir listes → tensors
        memory_dict = {k: torch.tensor(v) for k, v in mem.items()}
        print(f"✅ Mémoire N3R chargée depuis {memory_file}")
        return memory_dict
    else:
        print("⚡ Nouvelle mémoire N3R initialisée")
        return {}

def stabilize_latents_before_decode(
    latents,
    latent_scale,
    clamp_val=0.95,
    smooth_kernel=3,
    enable_smoothing=True
):
    """
    Stabilise les latents avant décodage pour éviter les artefacts de tiles. Args: latents (torch.Tensor): latents [B,C,H,W] latent_scale (float): facteur VAE (ex: 0.18215) clamp_val (float): limite de clamp (0.9–1.0 recommandé) smooth_kernel (int): taille du noyau de smoothing enable_smoothing (bool): active le lissage spatial Returns: torch.Tensor: latents prêts pour decode
    """

    # 🔥 sécurité NaN / inf
    latents = torch.nan_to_num(latents)

    # 🔥 clamp doux (évite contrastes violents entre tiles)
    latents = torch.clamp(latents, -clamp_val, clamp_val)

    # 🔥 lissage spatial (corrige les seams)
    if enable_smoothing and smooth_kernel > 1:
        latents = torch.nn.functional.avg_pool2d(
            latents,
            kernel_size=smooth_kernel,
            stride=1,
            padding=smooth_kernel // 2
        )

    # 🔥 continuité mémoire (important pour decode blockwise)
    latents = latents.contiguous()

    # 🔥 scale VAE
    latents = latents / latent_scale

    return latents


def get_interpolated_embeddings(frame_idx, frames_per_prompt, pos_list, neg_list, device="cuda"):
    num_prompts = len(pos_list)

    # index de base
    idx = frame_idx // frames_per_prompt
    idx_next = min(idx + 1, num_prompts - 1)

    # progression locale dans le segment
    t = (frame_idx % frames_per_prompt) / frames_per_prompt

    # cosine smooth (beaucoup mieux que linéaire)
    t = 0.5 - 0.5 * math.cos(math.pi * t)

    pos = (1 - t) * pos_list[idx] + t * pos_list[idx_next]
    neg = (1 - t) * neg_list[idx] + t * neg_list[idx_next]

    return pos.to(device), neg.to(device)


def compute_overlap(W, H, block_size, max_overlap_ratio=0.6):
    overlap = int(block_size * max_overlap_ratio)
    return min(overlap, min(W,H)//4)

# ---------------- DEBUG UTILS ----------------
def log_debug(message, level="INFO", verbose=True):
    """
    Affiche le message si verbose=True.
    level: "INFO", "DEBUG", "WARNING"
    """
    if verbose:
        print(f"[{level}] {message}")
# -------------------------------------------------------------------------------------------
def sanitize_latents(latents):
    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)

    # clamp doux (évite saturation brutale)
    latents = torch.clamp(latents, -1.2, 1.2)
    # normalisation légère si explosion
    if latents.std() > 1.5:
        latents = latents / latents.std()

    return latents

# -------------------------------------------------------------------------------------------
def stabilize_latents_advanced(latents, strength=0.99, knee=0.7):
    # sécurité
    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)

    # clamp doux
    latents = torch.clamp(latents, -1.2, 1.2)
    # normalisation si explosion
    std = latents.std()
    if std > 1.5:
        latents = latents / std
    # 🔥 compression non-linéaire (anti crispy blanc)
    latents = torch.tanh(latents * (1.0 / knee)) * knee
    # léger scaling global
    latents = latents * strength

    return latents



def print_generation_params(params: dict):
    """
    Affiche les paramètres de génération dans un tableau clair.

    params : dict
        Dictionnaire contenant tous les paramètres nécessaires.
        Exemple de clés attendues :
        'fps', 'upscale_factor', 'num_fraps_per_image', 'steps',
        'guidance_scale', 'init_image_scale', 'creative_noise',
        'latent_scale_boost', 'final_latent_scale', 'seed'
    """
    print("📌 Paramètres de génération :")
    print(f"{'Paramètre':<20} {'Valeur':>10}   {'Paramètre':<20} {'Valeur':>10}")

    left_keys = ['fps', 'num_fraps_per_image', 'guidance_scale', 'guidance_scale_end', 'creative_noise', 'creative_noise_end','final_latent_scale', 'transition_frames', 'use_n3r_model']
    right_keys = ['use_mini_gpu', 'upscale_factor', 'steps', 'init_image_scale', 'init_image_scale_end', 'latent_scale_boost', 'seed', 'latent_injection', 'block_size']

    for l, r in zip(left_keys, right_keys):
        print(f"{l:<20} {params.get(l, ''):>10}   {r:<20} {params.get(r, ''):>10}")


# ---------------- Tensor / PIL utils ----------------
def prepare_frame_tensor(frame_tensor):
    """
    Prépare un tensor de frame pour traitement (squeeze / permute / clamp)
    """
    if frame_tensor.ndim == 5: frame_tensor = frame_tensor.squeeze(2)
    if frame_tensor.ndim == 4: frame_tensor = frame_tensor.squeeze(0)
    if frame_tensor.ndim == 3 and frame_tensor.shape[0] != 3:
        frame_tensor = frame_tensor.permute(2,0,1)
    return frame_tensor.clamp(0,1)


def normalize_frame(frame_tensor):
    """
    Normalise un tensor image dans l'intervalle [0, 1].

    Cette fonction évite les problèmes d'overflow ou de valeurs hors plage
    en re-scalant dynamiquement les valeurs du tensor.

    Args:
        frame_tensor (torch.Tensor):
            Tensor image de forme [C, H, W] ou [B, C, H, W],
            avec des valeurs arbitraires.

    Returns:
        torch.Tensor:
            Tensor normalisé dans [0, 1].

    Notes:
        - Si min == max, aucune normalisation n'est appliquée.
        - Un clamp final garantit la stabilité numérique.
    """
    min_val = frame_tensor.min()
    max_val = frame_tensor.max()

    if max_val > min_val:
        frame_tensor = (frame_tensor - min_val) / (max_val - min_val)

    return frame_tensor.clamp(0, 1)


def tensor_to_pil(frame_tensor):
    """
    Convertit un tensor torch en image PIL.

    Args:
        frame_tensor (torch.Tensor):
            Tensor image de forme [C, H, W] ou [1, C, H, W],
            avec des valeurs attendues dans [0, 1].

    Returns:
        PIL.Image.Image:
            Image PIL prête à être sauvegardée ou affichée.

    Notes:
        - Si un batch est fourni ([B, C, H, W]), seule la première image est utilisée.
        - Les valeurs sont automatiquement clampées dans [0, 1].
        - Le tensor est déplacé sur CPU avant conversion.
    """
    if frame_tensor.ndim == 4:
        frame_tensor = frame_tensor[0]

    return ToPILImage()(frame_tensor.cpu().clamp(0, 1))


def ensure_4_channels(latents):
    """
    Garantit que le tensor latent possède 4 canaux (format attendu par les modèles SD).

    Args:
        latents (torch.Tensor):
            Tensor latent de forme [B, C, H, W].

    Returns:
        torch.Tensor:
            Tensor avec exactement 4 canaux.

    Notes:
        - Si C == 1, les canaux sont dupliqués pour obtenir 4 canaux.
        - Si C == 4, le tensor est retourné tel quel.
        - Ne gère pas les cas C != 1 et C != 4 (à étendre si besoin).
    """
    if latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    return latents


# ---------------- Video utils ----------------
def save_frames_as_video_from_folder(folder_path, output_path, fps=12):
    """Sauvegarde un dossier de frames PNG en vidéo mp4 via ffmpeg"""
    import ffmpeg
    from pathlib import Path
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
