# fx_utils.py
# ------------------- ENCODE -------------------

import os, math, threading
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

import torch
import numpy as np
import subprocess
from torchvision.transforms import functional as F
import torch.nn.functional as Fu
import torch.nn.functional as TF
import torch.nn.functional as FF
LATENT_SCALE = 0.18215


def apply_post_processing(frame_pil,
                          blur_radius=0.05,
                          contrast=1.15,
                          brightness=1.05,
                          saturation=0.85,
                          sharpen=False,
                          sharpen_radius=1,
                          sharpen_percent=90,
                          sharpen_threshold=2):
    """
    Appliquer des effets post-decode sur une frame PIL.

    Args:
        frame_pil (PIL.Image): L'image décodée depuis les latents
        blur_radius (float): Rayon du flou gaussien
        contrast (float): Facteur de contraste (1.0 = inchangé)
        brightness (float): Facteur de luminosité (1.0 = inchangé)
        saturation (float): Facteur de saturation (1.0 = inchangé)
        sharpen (bool): Appliquer un sharpen/unsharp mask si True
        sharpen_radius (float): Rayon pour l'UnsharpMask
        sharpen_percent (float): Intensité pour l'UnsharpMask
        sharpen_threshold (float): Seuil pour l'UnsharpMask

    Returns:
        PIL.Image: Image modifiée
    """
    # GaussianBlur simple
    if blur_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Ajustements optionnels (contrast, brightness, saturation)
    if contrast != 1.0:
        frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
    if brightness != 1.0:
        frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
    if saturation != 1.0:
        frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)

    # Sharp / UnsharpMask
    if sharpen:
        frame_pil = frame_pil.filter(ImageFilter.UnsharpMask(
            radius=sharpen_radius,
            percent=sharpen_percent,
            threshold=sharpen_threshold
        ))

    return frame_pil


def apply_post_processing_blur(frame_pil, blur_radius=0.2, contrast=1.0, brightness=1.0, saturation=1.0):
    """
    Appliquer des effets post-decode sur une frame PIL.

    Args:
        frame_pil (PIL.Image): L'image décodée depuis les latents
        blur_radius (float): Rayon du flou gaussien
        contrast (float): Facteur de contraste (1.0 = inchangé)
        brightness (float): Facteur de luminosité (1.0 = inchangé)
        saturation (float): Facteur de saturation (1.0 = inchangé)

    Returns:
        PIL.Image: Image modifiée
    """
    # GaussianBlur simple
    if blur_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Ajustements optionnels (contrast, brightness, saturation)
    if contrast != 1.0:
        frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
    if brightness != 1.0:
        frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
    if saturation != 1.0:
        frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)

    return frame_pil




def encode_images_to_latents_safe(images, vae, device="cuda", latent_scale=0.18215):
    """
    Encode une image en latents VAE en gardant la stabilité et le contraste.
    """
    images = images.to(device=device, dtype=torch.float32)
    vae = vae.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mean  # moyenne pour stabilité

    latents = latents * latent_scale
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    return latents


def save_frames_as_video_from_folder(
    folder_path,
    out_path,
    fps=12,
    upscale_factor=1
):
    """
    Sauvegarde toutes les images PNG d'un dossier en vidéo MP4.
    - Utilise ffmpeg directement (plus besoin de imageio)
    - Supporte l'upscale des images
    """
    folder_path = Path(folder_path)
    images = sorted(folder_path.glob("*.png"))

    if not images:
        raise ValueError(f"Aucune image trouvée dans {folder_path}")

    # Créer un dossier temporaire pour les images upscale
    tmp_dir = folder_path / "_tmp_upscaled"
    tmp_dir.mkdir(exist_ok=True)

    # Redimensionner les images si nécessaire
    for idx, img_path in enumerate(images):
        img = Image.open(img_path)
        if upscale_factor != 1:
            img = img.resize((img.width * upscale_factor, img.height * upscale_factor), Image.BICUBIC)
        tmp_file = tmp_dir / f"frame_{idx:05d}.png"
        img.save(tmp_file)

    # Appel ffmpeg pour créer la vidéo
    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-framerate", str(fps),
        "-i", str(tmp_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path)
    ]

    print("⚡ Génération vidéo avec ffmpeg…")
    subprocess.run(cmd, check=True)
    print(f"🎬 Vidéo sauvegardée : {out_path}")

    # Optionnel : supprimer le dossier temporaire
    for f in tmp_dir.glob("*.png"):
        f.unlink()
    tmp_dir.rmdir()


def encode_images_to_latents_nuanced_v1(images, vae, device="cuda", latent_scale=LATENT_SCALE):
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




def encode_images_to_latents_nuanced(images, vae, unet, device="cuda", latent_scale=LATENT_SCALE):
    """
    Encode une image en latents VAE, en préservant nuances et contraste,
    et redimensionne dynamiquement pour correspondre à la taille attendue par le UNet.
    """
    images = images.to(device=device, dtype=torch.float32)
    vae = vae.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # Encoder l'image → latents
        latents = vae.encode(images).latent_dist.mean  # moyenne pour stabilité

    # Appliquer le scaling
    latents = latents * latent_scale

    # Sécurité NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    # Forcer 4 canaux si nécessaire
    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    # 🔹 Redimensionner dynamiquement pour correspondre au UNet
    # On regarde la taille attendue à partir de l'UNet (ou ses skip connections)
    # Supposons que le UNet a un module `config` avec "sample_size" ou attention_resolutions
    try:
        # Si UNet a `sample_size` ou autre attribut
        target_H = getattr(unet.config, "sample_size", latents.shape[2])
        target_W = getattr(unet.config, "sample_size", latents.shape[3])
    except AttributeError:
        # fallback : garder la taille actuelle
        target_H, target_W = latents.shape[2], latents.shape[3]

    # Interpolation bilinéaire pour adapter la taille
    if (latents.shape[2], latents.shape[3]) != (target_H, target_W):
        latents = TF.interpolate(latents, size=(target_H, target_W), mode="bilinear", align_corners=False)
        print(f"[DEBUG] Latents resized to ({target_H}, {target_W})")

    return latents

# ------------------- ENCODE -------------------

def encode_images_to_latents_target(images, vae, device="cuda", latent_scale=LATENT_SCALE, target_size=64):
    """
    Encode une image en latents VAE, compatible MiniSD/AnimateDiff ultra-light (~2Go VRAM)
    - garde la dynamique et contraste
    - clamp minimal pour sécurité
    - force 4 canaux
    - resize latents à target_size (MiniSD)
    """
    images = images.to(device=device, dtype=torch.float32)
    vae = vae.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mean  # moyenne pour plus de stabilité

    # Appliquer le scaling
    latents = latents * latent_scale

    # Sécurité NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    # Forcer 4 canaux si nécessaire
    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    # Redimensionner à target_size x target_size
    if latents.shape[2] != target_size or latents.shape[3] != target_size:
        latents = torch.nn.functional.interpolate(
            latents, size=(target_size, target_size), mode="bilinear", align_corners=False
        )

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
