# n3rProNet_utils.py
from .tools_utils import ensure_4_channels, sanitize_latents, log_debug
import torch
import math
import numpy as np
from PIL import Image, ImageFilter
import torch.nn.functional as F
from pathlib import Path


def apply_chromatic_soft_glow(frame_pil,
                              glow_strength=0.25,
                              exposure=1.05,
                              blur_radius=2.0,
                              luminance_threshold=0.8,
                              color_saturation=1.05,
                              sharpen=True):
    """
    Soft Glow chromatique localisé :
    - Glow appliqué sur pixels clairs selon leur canal (R/G/B)
    - Zones sombres préservées
    - Détails conservés
    """
    from PIL import Image, ImageFilter, ImageChops, ImageEnhance
    import numpy as np

    arr = np.array(frame_pil).astype(np.float32) / 255.0
    arr = np.clip(arr * exposure, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))

    # -----------------------
    # Masque par canal
    # -----------------------
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    mask_r = np.clip((r - luminance_threshold) / (1.0 - luminance_threshold), 0, 1)
    mask_g = np.clip((g - luminance_threshold) / (1.0 - luminance_threshold), 0, 1)
    mask_b = np.clip((b - luminance_threshold) / (1.0 - luminance_threshold), 0, 1)

    # -----------------------
    # Glow par canal
    # -----------------------
    bright = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    bright_arr = np.array(bright).astype(np.float32) / 255.0

    # Mélange selon masque couleur
    arr[...,0] = np.clip(arr[...,0] + glow_strength * mask_r * bright_arr[...,0], 0, 1)
    arr[...,1] = np.clip(arr[...,1] + glow_strength * mask_g * bright_arr[...,1], 0, 1)
    arr[...,2] = np.clip(arr[...,2] + glow_strength * mask_b * bright_arr[...,2], 0, 1)

    img = Image.fromarray((arr*255).astype(np.uint8))

    # -----------------------
    # Saturation douce
    # -----------------------
    img = ImageEnhance.Color(img).enhance(color_saturation)

    # -----------------------
    # Micro sharpen subtil
    # -----------------------
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=30, threshold=2))

    return img


def apply_localized_soft_glow(frame_pil,
                              glow_strength=0.25,
                              exposure=1.05,
                              blur_radius=2.0,
                              luminance_threshold=0.6,
                              color_saturation=1.05,
                              sharpen=True):
    """
    Filtre 'Soft Glow Localisé':
    - Glow appliqué seulement sur les zones lumineuses
    - Effet subtil, préserve les zones sombres
    - Maintien des détails
    """
    from PIL import Image, ImageFilter, ImageChops, ImageEnhance
    import numpy as np

    # -----------------------
    # 1️⃣ Convertir en float + exposure
    # -----------------------
    arr = np.array(frame_pil).astype(np.float32) / 255.0
    arr = np.clip(arr * exposure, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))

    # -----------------------
    # 2️⃣ Masque de luminosité
    # -----------------------
    gray = img.convert("L")
    lum_arr = np.array(gray).astype(np.float32) / 255.0
    mask = np.clip((lum_arr - luminance_threshold) / (1.0 - luminance_threshold), 0, 1)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    # -----------------------
    # 3️⃣ Glow léger
    # -----------------------
    bright = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    glow_img = ImageChops.screen(img, bright)
    # Appliquer glow uniquement là où mask > 0
    glow_img = Image.composite(glow_img, img, mask_img)
    img = Image.blend(img, glow_img, glow_strength)

    # -----------------------
    # 4️⃣ Saturation douce
    # -----------------------
    img = ImageEnhance.Color(img).enhance(color_saturation)

    # -----------------------
    # 5️⃣ Micro sharpen subtil
    # -----------------------
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=30, threshold=2))

    return img


def apply_soft_glow(frame_pil,
                    glow_strength=0.25,
                    exposure=1.05,
                    blur_radius=2.0,
                    color_saturation=1.05,
                    sharpen=True):
    """
    Filtre 'Soft Glow' :
    - Surexposition douce sur les zones claires
    - Glow léger et subtil
    - Maintien des détails et textures
    """
    from PIL import Image, ImageFilter, ImageChops, ImageEnhance
    import numpy as np

    # -----------------------
    # 1️⃣ Convertir en float + exposure léger
    # -----------------------
    arr = np.array(frame_pil).astype(np.float32) / 255.0
    arr = np.clip(arr * exposure, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))

    # -----------------------
    # 2️⃣ Glow subtil (Light Bloom)
    # -----------------------
    bright = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    img = ImageChops.screen(img, bright)
    img = Image.blend(img, bright, glow_strength)

    # -----------------------
    # 3️⃣ Saturation douce
    # -----------------------
    img = ImageEnhance.Color(img).enhance(color_saturation)

    # -----------------------
    # 4️⃣ Micro sharpen subtil
    # -----------------------
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=30, threshold=2))

    return img


def apply_cinematic_neon_glow(frame_pil,
                              glow_strength=0.25,
                              edge_strength=0.15,
                              color_saturation=1.15,
                              exposure=1.05,
                              contrast=1.25,
                              blur_radius=0.4,
                              sharpen=True):
    """
    Filtre original 'Cinematic Neon Glow':
    - Glow subtil autour des zones claires
    - Couleurs saturées style néon / cinématographique
    - Bords légèrement lumineux type sketch
    """
    from PIL import Image, ImageFilter, ImageChops, ImageEnhance
    import numpy as np

    # -----------------------
    # 1️⃣ Convertir en float
    # -----------------------
    arr = np.array(frame_pil).astype(np.float32) / 255.0

    # -----------------------
    # 2️⃣ Exposure léger
    # -----------------------
    arr *= exposure
    arr = np.clip(arr, 0, 1)

    img = Image.fromarray((arr * 255).astype(np.uint8))

    # -----------------------
    # 3️⃣ Glow subtil (Light Bloom)
    # -----------------------
    bright = img.filter(ImageFilter.GaussianBlur(radius=5))
    img = ImageChops.screen(img, bright)  # effet lumineux
    img = Image.blend(img, bright, glow_strength)

    # -----------------------
    # 4️⃣ Edge sketch léger
    # -----------------------
    gray = img.convert("L").filter(ImageFilter.GaussianBlur(radius=1.0))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageChops.invert(edges)
    edges_rgb = Image.merge("RGB", (edges, edges, edges))
    img = ImageChops.blend(img, edges_rgb, edge_strength)

    # -----------------------
    # 5️⃣ Saturation & Contraste
    # -----------------------
    img = ImageEnhance.Color(img).enhance(color_saturation)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # -----------------------
    # 6️⃣ Micro blur anti-pixel
    # -----------------------
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # -----------------------
    # 7️⃣ Sharpen subtil
    # -----------------------
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=40, threshold=2))

    return img


def apply_post_processing_sketch(frame_pil, edge_strength=0.2, blur_radius=0.3, sharpen=True,
                                           contrast_boost=1.6,   # +60% contraste
                                           exposure=0.80):       # -20% brillance
    """
    Effet dessin subtil / croquis clair ajusté :
    - Contours légèrement visibles (blancs doux)
    - +40% contraste, -10% brillance
    - Lisse les pixels isolés
    - Ne dénature pas les couleurs de base
    """
    from PIL import Image, ImageFilter, ImageChops, ImageEnhance
    import numpy as np

    # -----------------------
    # 1️⃣ Edge detection doux
    # -----------------------
    gray = frame_pil.convert("L").filter(ImageFilter.GaussianBlur(radius=0.5))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.MedianFilter(size=3))   # supprime points isolés
    edges = edges.filter(ImageFilter.GaussianBlur(radius=0.6))  # lissage
    edges = ImageEnhance.Contrast(edges).enhance(1.2)
    edges = ImageChops.invert(edges)
    edge_rgb = Image.merge("RGB", (edges, edges, edges))

    # -----------------------
    # 2️⃣ Fusion douce des edges
    # -----------------------
    img = ImageChops.blend(frame_pil, edge_rgb, edge_strength)

    # -----------------------
    # 3️⃣ Exposure / Brillance
    # -----------------------
    img = ImageEnhance.Brightness(img).enhance(exposure)

    # -----------------------
    # 4️⃣ Contraste
    # -----------------------
    img = ImageEnhance.Contrast(img).enhance(contrast_boost)

    # -----------------------
    # 5️⃣ Blur léger anti-pixel
    # -----------------------
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # -----------------------
    # 6️⃣ Sharp subtil
    # -----------------------
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=40, threshold=2))

    return img



def apply_post_processing_drawing(frame_pil,
                                  edge_strength=0.7,
                                  color_levels=48,
                                  saturation=0.95,
                                  contrast=1.10,
                                  sharpen=True):
    """
    Post-processing dessin type line-art.
    Simplifie les couleurs, ajoute des contours au crayon blanc,
    supprime les points noirs et garde un rendu net.
    """

    from PIL import Image, ImageFilter, ImageEnhance, ImageChops
    import numpy as np

    # -----------------------
    # 1️⃣ Color simplification douce
    # -----------------------
    arr = np.array(frame_pil).astype(np.float32)
    levels = color_levels
    arr = np.round(arr / (256 / levels)) * (256 / levels)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # -----------------------
    # 2️⃣ Edge detection propre
    # -----------------------
    gray = frame_pil.convert("L").filter(ImageFilter.GaussianBlur(radius=0.6))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.GaussianBlur(radius=0.8))
    edges = edges.filter(ImageFilter.MedianFilter(size=3))  # supprime points isolés
    edges = ImageEnhance.Contrast(edges).enhance(1.4)
    edges = edges.point(lambda x: 0 if x < 15 else int(x * 1.2))
    edges = ImageChops.invert(edges)
    edge_rgb = Image.merge("RGB", (edges, edges, edges))

    # -----------------------
    # 3️⃣ Fusion douce contours
    # -----------------------
    img_edges = ImageChops.multiply(img, edge_rgb)
    img = Image.blend(img, img_edges, edge_strength * 0.85)

    # -----------------------
    # 4️⃣ Color / Contrast / Sharpen
    # -----------------------
    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.6, percent=60, threshold=3))

    return img




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
