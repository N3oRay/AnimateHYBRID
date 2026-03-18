# --------------------------------------------------------------
# tools_utils.py - Fonctions utilitaires génériques
# --------------------------------------------------------------
import os, math
import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F

LATENT_SCALE = 0.18215  # valeur globale, peut être importée si nécessaire

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

    left_keys = ['fps', 'num_fraps_per_image', 'guidance_scale', 'creative_noise', 'final_latent_scale']
    right_keys = ['upscale_factor', 'steps', 'init_image_scale', 'latent_scale_boost', 'seed']

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
    """Normalise un tensor [0,1] pour éviter overflow"""
    min_val = frame_tensor.min()
    max_val = frame_tensor.max()
    if max_val > min_val:
        frame_tensor = (frame_tensor - min_val)/(max_val - min_val)
    return frame_tensor.clamp(0,1)


def tensor_to_pil(frame_tensor):
    """Convertit un tensor torch en PIL Image"""
    if frame_tensor.ndim == 4:
        frame_tensor = frame_tensor[0]
    return ToPILImage()(frame_tensor.cpu().clamp(0,1))


def ensure_4_channels(latents):
    """Force un tensor latent à 4 canaux si nécessaire"""
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
