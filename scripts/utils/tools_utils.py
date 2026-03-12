# --------------------------------------------------------------
# tools_utils.py - Fonctions utilitaires génériques
# --------------------------------------------------------------
import os
import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F

LATENT_SCALE = 0.18215  # valeur globale, peut être importée si nécessaire


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
