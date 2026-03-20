# n3rProNet_utils.py
from .tools_utils import ensure_4_channels, sanitize_latents, log_debug
import torch
import math
import numpy as np
from PIL import Image, ImageFilter
import torch.nn.functional as F

def soft_tone_map(img):
    arr = np.array(img).astype(np.float32) / 255.0

    # 🔥 compression plus douce (log-like)
    arr = np.log1p(arr * 1.5) / np.log1p(1.5)

    # 🔥 léger adoucissement des contrastes
    arr = np.power(arr, 0.95)

    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def soft_tone_map1(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr / (arr + 0.2)
    arr = np.power(arr, 0.95)
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def apply_n3r_pro_net(latents, model=None, strength=0.3, sanitize_fn=None):
    if model is None or strength <= 0:
        return latents

    try:
        latents = latents.to(next(model.parameters()).dtype)
        refined = model(latents)

        # 🔥 différence (detail map)
        detail = refined - latents

        # 🔥 SMOOTH du détail (clé !!!)
        detail = F.avg_pool2d(detail, kernel_size=3, stride=1, padding=1)

        # 🔥 injection contrôlée
        latents = latents + strength * detail

        if sanitize_fn:
            latents = sanitize_fn(latents)

        return latents

    except Exception as e:
        print(f"[N3RProNet ERROR] {e}")
        return latents


def apply_n3r_pro_net1(latents, model=None, strength=0.3, sanitize_fn=None):
    if model is None or strength <= 0:
        return latents

    try:
        dtype = next(model.parameters()).dtype
        latents = latents.to(dtype)

        refined = model(latents)

        # 🔥 CLAMP SAFE (évite explosion)
        refined = torch.clamp(refined, -2.5, 2.5)

        # 🔥 BLEND DOUX (beaucoup plus stable)
        latents = (1 - strength) * latents + strength * refined

        # 🔥 NORMALISATION LÉGÈRE
        latents = latents / (latents.std(dim=[1,2,3], keepdim=True) + 1e-6)

        if sanitize_fn:
            latents = sanitize_fn(latents)

        return latents

    except Exception as e:
        print(f"[N3RProNet ERROR] {e}")
        return latents


def apply_n3r_pro_net_v1(latents, model=None, strength=0.3, sanitize_fn=None, frame_idx=None, total_frames=None):
    if model is None or strength <= 0:
        return latents

    try:
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        latents = latents.to(dtype=model_dtype, device=model_device)
        latents = ensure_4_channels(latents)

        if frame_idx is not None and total_frames is not None:
            adaptive_strength = strength * (0.3 + 0.7 * 0.5 * (1 - math.cos(math.pi * frame_idx / total_frames)))
        else:
            adaptive_strength = strength

        refined = model(latents)

        # 🔹 Normalisation du delta pour éviter saturation
        delta = refined - latents
        max_delta = delta.abs().amax(dim=(1,2,3), keepdim=True).clamp(min=1e-5)
        delta = delta / max_delta
        latents = latents + adaptive_strength * delta

        # 🔹 Clamp léger pour stabilité
        latents = latents / latents.abs().amax(dim=(1,2,3), keepdim=True).clamp(min=1.0)

        if sanitize_fn:
            latents = sanitize_fn(latents)

        return latents

    except Exception as e:
        print(f"[N3RProNet ERROR] {e}")
        return latents
