# fx_utils.py
# ------------------- ENCODE -------------------

import os, math, threading
from pathlib import Path
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as F
import torch
LATENT_SCALE = 0.18215


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
