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


# ---------------- Fusion N3R/VAE adaptative ----------------

def fuse_n3r_latents_adaptive(latents_frame, n3r_latents, latent_injection=0.7, clamp_val=1.0, creative_noise=0.0):
    n3r_latents = n3r_latents.clone()

    # Normalisation **canal par canal**
    for c in range(3):  # RGB uniquement
        n3r_c = n3r_latents[:,c:c+1,:,:]
        frame_c = latents_frame[:,c:c+1,:,:]
        mean, std = n3r_c.mean(), n3r_c.std()
        n3r_c = (n3r_c - mean) / (std + 1e-6)
        n3r_c = n3r_c * frame_c.std() + frame_c.mean()
        n3r_latents[:,c:c+1,:,:] = n3r_c

    # Ajouter un bruit créatif léger si nécessaire
    if creative_noise > 0.0:
        noise = torch.randn_like(n3r_latents) * creative_noise
        n3r_latents += noise

    # Clamp stricte pour éviter débordement
    n3r_latents = torch.clamp(n3r_latents, -clamp_val, clamp_val)
    latents_frame = torch.clamp(latents_frame, -clamp_val, clamp_val)

    # Fusion finale
    fused_latents = latent_injection * latents_frame + (1 - latent_injection) * n3r_latents
    fused_latents = torch.clamp(fused_latents, -clamp_val, clamp_val)

    print(f"[N3R fusion frame] mean/std par canal: {fused_latents.mean(dim=(2,3))}, injection={latent_injection:.2f}")
    return fused_latents


def interpolate_param_fast(start_val, end_val, current_frame, total_frames, mode="linear", speed=2.0):
    """
    Interpolation accélérée pour faire varier les paramètres plus rapidement au début.
    speed > 1 → plus rapide, speed < 1 → plus lent
    """
    t = current_frame / max(total_frames-1, 1)
    t = min(1.0, t * speed)  # accélère la progression

    if mode == "linear":
        return start_val + (end_val - start_val) * t
    elif mode == "cosine":
        t = (1 - math.cos(math.pi * t)) / 2
        return start_val + (end_val - start_val) * t
    elif mode == "ease_in_out":
        t = t*t*(3 - 2*t)
        return start_val + (end_val - start_val) * t
    else:
        return start_val + (end_val - start_val) * t


def interpolate_param(start_val, end_val, current_frame, total_frames, mode="linear"):
    """
    Interpolation d'un paramètre entre start_val -> end_val sur total_frames.
    Modes disponibles: 'linear', 'cosine', 'ease_in_out'
    """
    t = current_frame / max(total_frames-1,1)
    if mode == "linear":
        return start_val + (end_val - start_val) * t
    elif mode == "cosine":
        # Cosine easing pour un départ/arrivée plus doux
        t = (1 - math.cos(math.pi * t)) / 2
        return start_val + (end_val - start_val) * t
    elif mode == "ease_in_out":
        t = t*t*(3 - 2*t)
        return start_val + (end_val - start_val) * t
    else:
        return start_val + (end_val - start_val) * t


def estimate_sharpness(image):
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)
    laplacian = np.abs(
        arr[:-2,1:-1] + arr[2:,1:-1] + arr[1:-1,:-2] + arr[1:-1,2:] - 4*arr[1:-1,1:-1]
    )
    return laplacian.mean()

def adaptive_post_process(image):
    sharpness = estimate_sharpness(image)

    # seuils empiriques (à ajuster)
    if sharpness < 8:
        # image floue → sharpen fort
        return apply_post_processing(
            image,
            blur_radius=0.02,
            contrast=1.1,
            brightness=1.05,
            saturation=0.9,
            sharpen=True,
            sharpen_radius=1,
            sharpen_percent=120,
            sharpen_threshold=2
        )

    elif sharpness > 15:
        # image déjà très nette → adoucir
        return apply_post_processing(
            image,
            blur_radius=0.15,
            contrast=1.05,
            brightness=1.02,
            saturation=0.85,
            sharpen=False
        )

    else:
        # équilibré
        return apply_post_processing(
            image,
            blur_radius=0.05,
            contrast=1.1,
            brightness=1.05,
            saturation=0.9,
            sharpen=True,
            sharpen_radius=1,
            sharpen_percent=80,
            sharpen_threshold=2
        )




def apply_post_processing(frame_pil,
                          blur_radius=0.05,
                          contrast=1.15,
                          brightness=1.05,
                          saturation=0.85,
                          vibrance=1.1,  # <-- ajout vibrance
                          sharpen=False,
                          sharpen_radius=1,
                          sharpen_percent=90,
                          sharpen_threshold=2):
    # GaussianBlur
    if blur_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Contrast / Brightness / Saturation
    if contrast != 1.0:
        frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
    if brightness != 1.0:
        frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
    if saturation != 1.0:
        frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)

    # Vibrance: booster les couleurs peu saturées
    if vibrance != 1.0:
        frame_hsv = frame_pil.convert("HSV")
        h, s, v = frame_hsv.split()
        s = s.point(lambda i: min(255, int(i * vibrance) if i < 128 else i))
        frame_pil = Image.merge("HSV", (h, s, v)).convert("RGB")

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


def encode_images_to_latents_hybrid(images, vae, device="cuda", latent_scale=LATENT_SCALE):
    """
    Encodage hybride pour conserver la fidélité et la richesse de détails.
    - Utilise un échantillon de la distribution (plus vivant que mean)
    - Clamp léger pour éviter débordements extrêmes mais garder micro-contrastes
    - Assure 4 channels si nécessaire pour compatibilité UNet/N3R
    """
    images = images.to(device=device, dtype=torch.float32)
    vae = vae.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()  # <-- sample() pour micro-variations

    latents = latents * latent_scale
    latents = torch.clamp(latents, -5.0, 5.0)  # Clamp léger, pas de nan_to_num

    # Assurer 4 channels si nécessaire
    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    return latents

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
