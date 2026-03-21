# n3rProNet_utils.py
from .tools_utils import ensure_4_channels, sanitize_latents, log_debug
import torch
import math
import numpy as np
from PIL import Image, ImageFilter
import torch.nn.functional as F
from pathlib import Path


def save_frame_verbose(frame: Image.Image, output_dir: Path, frame_counter: int, suffix: str = "00", psave: bool = True):
    """
    Sauvegarde une frame avec suffixe et affiche un message si verbose=True

    Args:
        frame (Image.Image): Image PIL à sauvegarder
        output_dir (Path): Dossier de sortie
        frame_counter (int): Numéro de frame
        suffix (str): Suffixe pour différencier les étapes
        verbose (bool): Affiche le message si True
    """
    file_path = output_dir / f"frame_{frame_counter:05d}_{suffix}.png"

    if psave:
        print(f"[SAVE Frame {frame_counter:03d}_{suffix}] -> {file_path}")
        frame.save(file_path)
    return file_path

def neutralize_color_cast(img, strength=0.45, warm_bias=0.015, green_bias=-0.07):
    """
    Neutralise la dominante de couleur tout en corrigeant un excès de vert.

    Args:
        img (PIL.Image): image à corriger
        strength (float): intensité de neutralisation (0.0 = off, 1.0 = full)
        warm_bias (float): réchauffe légèrement (rouge+/bleu-)
        green_bias (float): ajuste le vert (-0.07 = moins 7%)
    """
    import numpy as np
    from PIL import Image

    arr = np.array(img).astype(np.float32)

    mean = arr.mean(axis=(0,1))
    gray = mean.mean()

    gain = gray / (mean + 1e-6)
    gain = 1.0 + (gain - 1.0) * strength

    arr[..., 0] *= gain[0] * (1 + warm_bias)  # rouge +
    arr[..., 1] *= gain[1] * (1 + green_bias) # vert corrigé
    arr[..., 2] *= gain[2] * (1 - warm_bias)  # bleu -

    arr = np.clip(arr, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def neutralize_color_cast_clean(img, strength=0.6, warm_bias=0.02):
    import numpy as np
    from PIL import Image

    arr = np.array(img).astype(np.float32)

    mean = arr.mean(axis=(0,1))
    gray = mean.mean()

    gain = gray / (mean + 1e-6)
    gain = 1.0 + (gain - 1.0) * strength

    arr[..., 0] *= gain[0] * (1 + warm_bias)  # 🔥 léger rouge +
    arr[..., 1] *= gain[1]
    arr[..., 2] *= gain[2] * (1 - warm_bias)  # 🔥 léger bleu -

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def neutralize_color_cast_str(img, strength=0.6):
    import numpy as np
    from PIL import Image

    arr = np.array(img).astype(np.float32)

    mean = arr.mean(axis=(0,1))
    gray = mean.mean()

    gain = gray / (mean + 1e-6)

    # 🔥 interpolation (clé)
    gain = 1.0 + (gain - 1.0) * strength

    arr[..., 0] *= gain[0]
    arr[..., 1] *= gain[1]
    arr[..., 2] *= gain[2]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def neutralize_color_cast_simple(img):
    import numpy as np
    arr = np.array(img).astype(np.float32)

    mean = arr.mean(axis=(0,1))

    # cible gris neutre
    gray = mean.mean()

    gain = gray / (mean + 1e-6)

    arr[..., 0] *= gain[0]
    arr[..., 1] *= gain[1]
    arr[..., 2] *= gain[2]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def kelvin_to_rgb(temp):
    """
    Approximation réaliste Kelvin → RGB (inspiré photographie)
    """
    temp = temp / 100.0

    # Rouge
    if temp <= 66:
        r = 255
    else:
        r = temp - 60
        r = 329.698727446 * (r ** -0.1332047592)

    # Vert
    if temp <= 66:
        g = temp
        g = 99.4708025861 * math.log(g) - 161.1195681661
    else:
        g = temp - 60
        g = 288.1221695283 * (g ** -0.0755148492)

    # Bleu
    if temp >= 66:
        b = 255
    elif temp <= 19:
        b = 0
    else:
        b = temp - 10
        b = 138.5177312231 * math.log(b) - 305.0447927307

    return (
        max(0, min(255, r)) / 255.0,
        max(0, min(255, g)) / 255.0,
        max(0, min(255, b)) / 255.0
    )


def adjust_color_temperature(image, target_temp=10000, reference_temp=6500, strength=0.5):
    import numpy as np

    img = np.array(image).astype(np.float32) / 255.0

    r1, g1, b1 = kelvin_to_rgb(reference_temp)
    r2, g2, b2 = kelvin_to_rgb(target_temp)

    # 🔥 interpolation (clé)
    r_gain = (1 - strength) + strength * (r2 / r1)
    g_gain = (1 - strength) + strength * (g2 / g1)
    b_gain = (1 - strength) + strength * (b2 / b1)

    img[..., 0] *= r_gain
    img[..., 1] *= g_gain
    img[..., 2] *= b_gain

    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))

def adjust_color_temperature_simple(image, target_temp=3500, reference_temp=8000):
    import numpy as np

    img = np.array(image).astype(np.float32) / 255.0

    # Gains relatifs (IMPORTANT → comme GIMP)
    r1, g1, b1 = kelvin_to_rgb(reference_temp)
    r2, g2, b2 = kelvin_to_rgb(target_temp)

    r_gain = r2 / r1
    g_gain = g2 / g1
    b_gain = b2 / b1

    img[..., 0] *= r_gain
    img[..., 1] *= g_gain
    img[..., 2] *= b_gain

    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def soft_tone_map(img):
    import numpy as np

    arr = np.array(img).astype(np.float32) / 255.0

    # 🔥 contraste léger (au lieu de compression)
    mean = arr.mean(axis=(0,1), keepdims=True)
    arr = (arr - mean) * 1.1 + mean

    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def soft_tone_map_unreal(img, exposure=1.0):
    import numpy as np

    arr = np.array(img).astype(np.float32) / 255.0

    # 🔥 exposure
    arr = arr * exposure

    # 🔥 tone mapping type Reinhard (plus naturel)
    mapped = arr / (1.0 + arr)

    # 🔥 léger contraste local (clé !)
    mapped = np.power(mapped, 0.9)

    return Image.fromarray((np.clip(mapped, 0, 1) * 255).astype(np.uint8))


def soft_tone_map_v1(img):
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
