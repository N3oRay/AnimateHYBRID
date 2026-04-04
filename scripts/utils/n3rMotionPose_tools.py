#********************************************
# n3rOpenPose_utils.py
#********************************************
import torch
from diffusers import ControlNetModel
import math
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import traceback


#from .n3rMotionPose_tools import gaussian_blur_tensor, feather_mask, feather_mask_fast, feather_outside_only, feather_inside,feather_inside_strict, debug_draw_openpose_skeleton, rotate_mask_around_torso_simple, rotate_mask_around_torso


def debug_draw_openpose_skeleton(
    pose_full_image,
    keypoints_tensor,
    debug_dir,
    frame_counter,
):
    """
    Dessine les keypoints + squelette OpenPose sur l'image de pose.

    Args:
        pose_full_image (torch.Tensor): [B,3,H,W] normalisé [-1,1]
        keypoints_tensor (torch.Tensor): [B,18,3] (x,y,conf) normalisé [0,1]
        debug_dir (str): dossier de sauvegarde
        frame_counter (int): index frame
    """

    os.makedirs(debug_dir, exist_ok=True)

    # ---------------------------
    # 🔹 Convert tensor → image
    # ---------------------------
    pose_img = pose_full_image[0].permute(1, 2, 0).detach().cpu().numpy()
    pose_img = (pose_img + 1.0) / 2.0
    pose_img = (pose_img * 255).astype(np.uint8).copy()

    H, W, _ = pose_img.shape

    keypoints = keypoints_tensor[0].detach().cpu().numpy()

    def to_pixel(x, y):
        return int(x * W), int(y * H)

    # ---------------------------
    # 🎨 Couleurs
    # ---------------------------
    COLORS = {
        "head": (255, 0, 255),   # rose
        "eyes": (0, 0, 255),     # rouge
        "nose": (128, 0, 128),   # violet
        "arms": (0, 255, 0),     # vert
        "torso": (255, 0, 0),    # bleu
        "default": (200, 200, 200)
    }

    # ---------------------------
    # 🔹 Draw points
    # ---------------------------
    for i, (x, y, conf) in enumerate(keypoints):
        if conf < 0.1:
            continue

        px, py = to_pixel(x, y)

        if i in [16, 17]:        # ears
            color = COLORS["head"]
        elif i in [14, 15]:      # eyes
            color = COLORS["eyes"]
        elif i == 0:             # nose
            color = COLORS["nose"]
        elif i in [2,3,19,20,4,5,6,7]: # arms
            color = COLORS["arms"]
        elif i in [1,8,11]:      # torso
            color = COLORS["torso"]
        else:
            color = COLORS["default"]

        cv2.circle(pose_img, (px, py), 6, color, -1)

    # ---------------------------
    # 🔹 Skeleton connections
    # ---------------------------
    skeleton = [
        (0,1),        # nose → neck
        (1,19), (19,2), (2,3), (3,4),   # right arm (1,2), (2,3), (3,4),   # right arm
        (1,20), (20,5), (5,6), (6,7),   # left arm (1,5), (5,6), (6,7),   # left arm
        (1,8), (1,11),         # torso
        (14,0), (15,0),        # eyes → nose
        (16,14), (17,15),      # ears → eyes
    ]

    for i, j in skeleton:
        xi, yi, ci = keypoints[i]
        xj, yj, cj = keypoints[j]

        if ci < 0.1 or cj < 0.1:
            continue

        p1 = to_pixel(xi, yi)
        p2 = to_pixel(xj, yj)

        # couleurs par type de lien
        if (i, j) == (0,1):
            color = (128, 0, 128)  # nez → cou (violet)
        elif i in [2,3,4,5,6,7]:
            color = (0,255,0)      # bras
        elif i in [1,8,11]:
            color = (255,0,0)      # torse
        else:
            color = (255,255,255)

        cv2.line(pose_img, p1, p2, color, 2)

    # ---------------------------
    # 💾 Save
    # ---------------------------
    save_path = f"{debug_dir}/skeleton_{frame_counter:05d}.png"
    cv2.imwrite(save_path, cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))

    print(f"[DEBUG] Skeleton sauvegardé : {save_path}")

# sigma correspond a la valeur du flou
def gaussian_blur_tensor(x, kernel_size=3, sigma=0.5):
    # x: [B,C,H,W]
    B, C, H, W = x.shape

    # Générer un kernel 1D
    def gauss1d(k, sigma):
        a = torch.arange(k).float() - (k - 1) / 2
        g = torch.exp(-(a**2)/(2*sigma**2))
        return g / g.sum()

    k = kernel_size
    g = gauss1d(k, sigma)
    kernel2d = g[:,None] * g[None,:]      # [k,k]
    kernel2d = kernel2d.to(x.device, dtype=x.dtype)
    kernel2d = kernel2d.expand(C, 1, k, k)  # [C,1,k,k] pour grouped conv
    pad = k // 2
    x = F.conv2d(x, kernel2d, padding=pad, groups=C)
    return x

# blur sur le coté du masque type photoshop
def feather_mask(mask, radius=3):
    """
    mask: [B,1,H,W] (0 ou 1)
    radius: épaisseur du bord en pixels
    """

    # approx distance via blur répété (rapide GPU)
    dist = mask.clone()

    for _ in range(radius):
        dist = F.max_pool2d(dist, kernel_size=3, stride=1, padding=1)

    # bande extérieure uniquement
    band = (dist - mask).clamp(0, 1)

    # normaliser pour faire un dégradé progressif
    band = band / (band.max().clamp(min=1e-6))

    # lisser légèrement (optionnel mais joli)
    band = gaussian_blur_tensor(band, kernel_size=5, sigma=1.0)

    # reconstruire
    mask = mask + band * (1 - mask)

    return torch.clamp(mask, 0, 1)

# blur sur le coté du masque type photoshop
def feather_mask_fast(mask, radius=3):
    k = 2 * radius + 1

    dist = F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)

    band = (dist - mask).clamp(0, 1)
    band = gaussian_blur_tensor(band, kernel_size=5, sigma=1.0)

    return torch.clamp(mask + band * (1 - mask), 0, 1)


def feather_outside_only(mask, radius=3, blur_kernel=5, sigma=1.0):
    """
    mask: [B,1,H,W] (0 ou 1)
    radius: largeur du dégradé extérieur (pixels)
    """

    # 1. dilatation contrôlée (couronne extérieure)
    k = 2 * radius + 1
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)

    # 2. isoler UNIQUEMENT l'extérieur
    band = (dilated - mask).clamp(0, 1)

    # 3. lisser cette bande (sans toucher l'intérieur)
    band = gaussian_blur_tensor(band, kernel_size=blur_kernel, sigma=sigma)

    # 4. normaliser pour un vrai dégradé
    band = band / (band.max().clamp(min=1e-6))

    # 5. reconstruction :
    # intérieur intact + dégradé extérieur uniquement
    result = mask + band * (1 - mask)

    return torch.clamp(result, 0, 1)


def feather_inside(mask, radius=5, blur_kernel=5, sigma=1.0):
    """
    mask: [B,1,H,W] (0 ou 1)
    radius: largeur de la bande à l'intérieur du mask
    """

    # 1. érosion (réduire le mask) pour créer la bande interne
    k = 2 * radius + 1
    eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=radius)  # erosion via max pooling sur négatif

    # 2. isoler la bande intérieure
    band = (mask - eroded).clamp(0, 1)

    # 3. lisser légèrement pour le feather
    band = gaussian_blur_tensor(band, kernel_size=blur_kernel, sigma=sigma)

    # 4. correction alpha pour éviter l’étirement
    band = band * band * (3 - 2 * band)  # smoothstep

    # 5. reconstruire le mask : intérieur net + dégradé à l’intérieur
    result = eroded + band

    return torch.clamp(result, 0, 1)


def feather_inside_strict(mask, radius=5, blur_kernel=5, sigma=1.0):
    """
    mask: [B,1,H,W] (0 ou 1)
    radius: largeur du feather à l'intérieur du mask
    """

    # 1️⃣ érosion : zone intérieure parfaitement nette
    k = 2 * radius + 1
    eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=radius)

    # 2️⃣ bande interne à flouter (feather)
    band = (mask - eroded).clamp(0, 1)

    # 3️⃣ flou uniquement sur la bande
    if band.max() > 0:  # éviter division par zéro
        band_blur = gaussian_blur_tensor(band, kernel_size=blur_kernel, sigma=sigma)
        # 4️⃣ correction alpha smoothstep pour éviter étirement
        band_blur = band_blur * band_blur * (3 - 2 * band_blur)
    else:
        band_blur = band

    # 5️⃣ reconstruction : intérieur net + bande floutée
    result = eroded + band_blur

    return torch.clamp(result, 0, 1)


def rotate_mask_around_torso_simple(mask, torso_points_px, angle, device="cuda"):
    """
    Rotate a mask around the torso center on X-axis only (horizontal rotation),
    corrected version with proper broadcasting.

    mask: [B, C, H, W]
    torso_points_px: [B, 2, N]
    angle: [B] tensor, rotation in radians
    device: device

    Returns:
        mask_rotated: [B, C, H, W]
    """
    import torch
    import torch.nn.functional as F

    B, C, H, W = mask.shape

    # Centre du torse
    torso_center = torso_points_px.mean(dim=2)  # [B, 2]
    cx = torso_center[:, 0].view(B, 1, 1)
    cy = torso_center[:, 1].view(B, 1, 1)

    # Coordonnées pixels
    xx = torch.arange(W, device=device).view(1, 1, W).float()  # [1,1,W]
    yy = torch.arange(H, device=device).view(1, H, 1).float()  # [1,H,1]

    # Décalage horizontal seulement
    x_shift = xx - cx
    y_shift = yy  # vertical inchangé

    # Rotation horizontale
    cos_a = torch.cos(angle).view(B, 1, 1)
    x_rot = cos_a * x_shift + cx  # rotation autour du centre X
    y_rot = y_shift

    # Broadcast pour stack
    x_norm = 2.0 * x_rot / (W - 1) - 1.0
    y_norm = 2.0 * y_rot / (H - 1) - 1.0
    # broadcast sur [B,H,W]
    x_norm = x_norm.expand(B, H, W)
    y_norm = y_norm.expand(B, H, W)

    grid = torch.stack((x_norm, y_norm), dim=-1)  # [B,H,W,2]

    # Rotation bilinéaire
    mask_rotated = F.grid_sample(
        mask, grid, mode='bilinear', padding_mode='zeros', align_corners=False
    )

    return mask_rotated

def rotate_mask_around_torso(mask, torso_points_px, angle, H, W, device="cuda"):

    B, C, H_mask, W_mask = mask.shape

    # 🔥 utiliser H_mask / W_mask partout
    H, W = H_mask, W_mask

    # -------------------- centre du torse --------------------
    torso_center = torso_points_px.mean(dim=2)  # [B, 2]

    # -------------------- grid pixel --------------------
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    xx = xx.float().unsqueeze(0).expand(B, -1, -1)
    yy = yy.float().unsqueeze(0).expand(B, -1, -1)

    # -------------------- rotation autour centre --------------------
    cx = torso_center[:, 0].view(B, 1, 1)
    cy = torso_center[:, 1].view(B, 1, 1)

    x_shift = xx - cx
    y_shift = yy - cy

    cos_angle = torch.cos(angle).view(B, 1, 1)
    sin_angle = torch.sin(angle).view(B, 1, 1)

    x_rot = cos_angle * x_shift - sin_angle * y_shift
    y_rot = sin_angle * x_shift + cos_angle * y_shift

    x_final = x_rot + cx
    y_final = y_rot + cy

    # -------------------- NORMALISATION CORRECTE --------------------
    x_norm = 2.0 * x_final / (W - 1) - 1.0
    y_norm = 2.0 * y_final / (H - 1) - 1.0

    grid = torch.stack((x_norm, y_norm), dim=-1)

    # 🔥 FIX CRITIQUE : align_corners=False
    mask_rotated = F.grid_sample(
        mask,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    # 🔥 GARANTIE ABSOLUE
    assert mask_rotated.shape[2:] == (H, W), \
        f"Shape mismatch: {mask_rotated.shape} vs {(B,C,H,W)}"

    return mask_rotated


def apply_breathing_xy(latents, previous_latent, frame_counter, breathing=True):
    """
    Applique une respiration réaliste sur les latents : décalage vertical centré.
    """
    if previous_latent is not None and breathing:
        B, C, H, W = latents.shape
        breath = 0.004 * math.sin(frame_counter * 0.15)  # amplitude réduite
        # créer un shift vertical uniforme
        yy, xx = torch.meshgrid(torch.linspace(-1,1,H,device=latents.device),
                                torch.linspace(-1,1,W,device=latents.device), indexing='ij')
        grid = torch.stack((xx, yy + breath, ), dim=-1)  # on ajoute seulement sur y
        grid = grid.unsqueeze(0).repeat(B,1,1,1)  # batch
        latents = torch.nn.functional.grid_sample(latents, grid, align_corners=True)
    return latents

def apply_breathing(latents, previous_latent, frame_counter, breathing=True):
    """
    Applique une respiration réaliste sur les latents : décalage vertical uniquement.
    """
    if previous_latent is not None and breathing:
        B, C, H, W = latents.shape
        device = latents.device

        # amplitude respiration
        breath = 0.004 * math.sin(frame_counter * 0.15)  # amplitude réduite

        # grille pixels avec meshgrid
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )  # yy, xx → [H, W]

        # ajouter dimension batch et stack
        grid = torch.stack((xx, yy + breath), dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).expand(B, H, W, 2)   # [B, H, W, 2]

        # appliquer le décalage vertical uniquement
        latents = F.grid_sample(latents, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return latents



def save_impact_map(latents, latents_in, debug_dir, frame_counter):
    """
    Sauvegarde une carte d'impact montrant les différences entre latents et latents_in.

    Args:
        latents (torch.Tensor): Latents modifiés [B, C, H, W]
        latents_in (torch.Tensor): Latents originaux [B, C, H, W]
        debug_dir (str): Répertoire où sauvegarder l'image
        frame_counter (int): Index de la frame pour le nom de fichier
    """
    if debug_dir is None:
        return

    os.makedirs(debug_dir, exist_ok=True)
    impact_map = torch.abs(latents - latents_in).mean(1, keepdim=True)
    impact_np = impact_map[0,0].detach().cpu().numpy()
    impact_np -= impact_np.min()
    if impact_np.max() > 0:
        impact_np /= impact_np.max()
    save_path = os.path.join(debug_dir, f"impact_map_driven_{frame_counter:05d}.png")
    Image.fromarray((impact_np*255).astype(np.uint8)).save(save_path)
    print(f"[DEBUG] Impact map saved: {save_path}")
