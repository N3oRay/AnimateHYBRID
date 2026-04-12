#********************************************
# n3rOpenPose_utils.py
#********************************************
import torch
import time
from diffusers import ControlNetModel
import math
import torch.nn.functional as F
from .n3rControlNet import create_canny_control, control_to_latent, match_latent_size
from .tools_utils import ensure_4_channels, print_generation_params, sanitize_latents
from .n3rMotionPose_tools import gaussian_blur_tensor, debug_draw_openpose_skeleton, rotate_mask_around_torso_simple, rotate_mask_around_visage, save_impact_map, apply_breathing_xy, smooth_noise, feather_dynamic_vectorized, compute_delta, stabilize_latents_motion, save_debug_pose_image_with_skeleton, apply_hair_motion_3D, apply_hair_motion_cycle, apply_breathing_simple, feather_inside_strict2, feather_outside_only_alpha2
from .n3rMotionPoseClass import Pose
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import traceback
from torchvision.utils import save_image


#------extract_keypoints_from_pose

def extract_keypoints_from_pose(
    pose_full_image,
    device="cuda",
    debug=False,
    debug_dir=None,
    frame_counter=None
):
    """
    FR:
    Extraction MANUELLE des keypoints (format COCO 18 points).
    Les coordonnées sont normalisées entre [0,1] dans l'espace IMAGE.

    EN:
    MANUAL keypoints extraction (COCO 18 format).
    Coordinates are normalized in [0,1] in IMAGE space.

    Output:
        keypoints_tensor: [B, 18, 3]  (x, y, confidence)
    """

    B, C, H, W = pose_full_image.shape  # H=height, W=width

    # ---------------------------
    # 🔥 KEYPOINTS MANUELS / MANUAL KEYPOINTS
    # ---------------------------
    # FR:
    # x = pixel_x / image_width
    # y = pixel_y / image_height
    # IMPORTANT: utiliser la taille de l'image originale (pose_full), PAS le latent

    # EN:
    # x = pixel_x / image_width
    # y = pixel_y / image_height
    # IMPORTANT: use original image size (pose_full), NOT latent size

    keypoints_template = [
        [422/896, 408/1280, 1.0],  # 0 nose / nez 👃 Nose detected: (422, 408)
        [418/896, 490/1280, 1.0],  # 1 neck / cou 🦵 Neck detected: {'center': (418, 490)}

        [562/896, 506/1280, 1.0],  # 2 right_shoulder / épaule droite 🦾 Clavicules detected: [(275, 519), (562, 506)]
        [627/896, 896/1280, 1.0],  # 3 right_elbow / coude droit 🦾 Elbows detected/estimated: [[179, 896], [627, 896]]
        [488/896, 1040/1280, 1.0], # 4 right_wrist / poignet droit ✋ Wrists detected: fallback

        [275/896, 519/1280, 1.0],  # 5 left_shoulder / épaule gauche 🦾 Clavicules detected: [(275, 519), (562, 506)]
        [197/896, 944/1280, 1.0],  # 6 left_elbow / coude gauche 🦾 Elbows detected/estimated: [[179, 896], [627, 896]]
        [431/896, 1087/1280, 1.0], # 7 left_wrist / poignet gauche ✋ Wrists detected: fallback

        [564/896, 1102/1280, 1.0], # 8 right_hip / hanche droite 🦿📍 Hips detected: left=(564, 1102), right=(308, 1129)
        [0.0, 0.0, 0.0],           # 9 right_knee (absent)
        [0.0, 0.0, 0.0],           # 10 right_ankle (absent)

        [308/896, 1129/1280, 1.0], # 11 left_hip / hanche gauche 🦿📍 Hips detected: left=(564, 1102), right=(308, 1129)
        [0.0, 0.0, 0.0],           # 12 left_knee (absent)
        [0.0, 0.0, 0.0],           # 13 left_ankle (absent)

        [492/896, 355/1280, 1.0],  # 14 right_eye / œil droit 👁 Eyes detected: [(492, 355), (375, 323)]
        [374/896, 323/1280, 1.0],  # 15 left_eye / œil gauche 👁 Eyes detected: [(326, 379), (359, 490)]
        [529/896, 349/1280, 1.0],  # 16 right_ear / oreille droite 👂 Ears detected: [(308, 282), (529, 349)]
        [308/896, 282/1280, 1.0],  # 17 left_ear / oreille gauche
        [406/896, 446/1280, 1.0],  # 18 mouth / bouche 👄 Mouth detected: [(405, 446)]
        [562/896, 506/1280, 1.0],  # 19 right_clavicle 🦾 Clavicules detected: [(275, 519), (562, 506)]
        [275/896, 519/1280, 1.0],  # 20 left_clavicle 🦾 Clavicules detected: [(275, 519), (562, 506)]

        # Nouveaux points du cou
        [387/896, 480/1280, 1.0],  # 21 chin / menton
        [307/896, 282/1280, 1.0],  # 22 left_side_neck / gauche cou
        [529/896, 349/1280, 1.0],  # 23 right_side_neck / droite cou
        [422/896, 428/1280, 1.0],  # 24 anchor / point d'ancrage cou
    ]

    #48–54 : lèvres supérieures (coin gauche → coin droit)
    #54–60 : lèvres inférieures (coin droit → coin gauche)
    #60–67 : contour interne de la bouche (pour l’ouverture, micro-mouvements)

    # ---------------------------
    # 🔹 Conversion numpy → tensor
    # ---------------------------
    keypoints_np = np.array(keypoints_template, dtype=np.float32)

    # FR: sécurité pour éviter valeurs hors [0,1]
    # EN: safety clamp to keep values in [0,1]
    keypoints_np = np.clip(keypoints_np, 0.0, 1.0)

    # FR: duplication pour batch
    # EN: repeat for batch
    keypoints_np = np.expand_dims(keypoints_np, axis=0)  # [1,18,3]
    keypoints_np = np.repeat(keypoints_np, B, axis=0)    # [B,18,3]

    keypoints_tensor = torch.from_numpy(keypoints_np).to(device)

    # ---------------------------
    # 🔹 DEBUG VISUEL / VISUAL DEBUG
    # ---------------------------
    if debug and debug_dir is not None and frame_counter is not None:
        # Affichage des points en debug
        debug_draw_openpose_skeleton(
            pose_full_image=pose_full_image,
            keypoints_tensor=keypoints_tensor,
            debug_dir=debug_dir,
            frame_counter=frame_counter
        )

    return keypoints_tensor


def update_keypoints_from_pose(
    pose_full_image,
    current_keypoints,
    nose_coords,
    neck_coords,
    shoulders_coords,
    clavicules_coords,
    elbow_coords,
    wrists_coords,
    hips_coords,
    eye_coords,
    ear_coords,
    mouth_coords,
    device="cuda",
    debug=False,
    debug_dir=None,
    frame_counter=None
):
    """
    Mise à jour safe des keypoints (25 points + extras) à partir des nouvelles coordonnées.
    Ne met à jour que les points valides. Les points invalides restent inchangés.

    Inputs:
        pose_full_image: tensor [B,C,H,W]
        current_keypoints: tensor [B,25,3] déjà existant
        *_coords: nouvelles coordonnées détectées
        device: 'cuda' ou 'cpu'
        debug: bool pour affichage visuel
        debug_dir: répertoire pour debug visuel
        frame_counter: numéro de frame pour debug

    Output:
        keypoints_tensor: [B,25,3] (x,y,conf) mis à jour
    """

    B, C, H, W = pose_full_image.shape

    # ---------------------------
    # 🔹 Fonction utilitaire safe
    # ---------------------------
    def safe_update(idx, new_coord, keypoints_np, label=""):
        x, y = new_coord
        old_x, old_y = keypoints_np[idx, 0]*W, keypoints_np[idx, 1]*H
        if x == 0 and y == 0:
            if old_x != 0 or old_y != 0:
                # On garde l'ancienne valeur sans warning
                return
            else:
                print(f"⚠ Coordonnée {label} invalide, aucune valeur précédente.")
        else:
            keypoints_np[idx, 0] = x / W
            keypoints_np[idx, 1] = y / H
            keypoints_np[idx, 2] = 1.0

    # ---------------------------
    # 🔹 Conversion current_keypoints → numpy
    # ---------------------------
    keypoints_np = current_keypoints.clone().cpu().numpy()[0]  # [25,3]

    # ---------------------------
    # 🔹 Récupération safe des coordonnées
    # ---------------------------
    def safe_xy(coord):
        if coord is None:
            return (0,0)
        if isinstance(coord, list) and len(coord) == 1:
            return tuple(coord[0])
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            return tuple(coord)
        return (0,0)

    nose = safe_xy(nose_coords)
    neck_center = safe_xy(neck_coords.get("center") if isinstance(neck_coords, dict) else neck_coords)
    chin = safe_xy(neck_coords.get("chin") if isinstance(neck_coords, dict) else None)
    left_side_neck = safe_xy(neck_coords.get("left") if isinstance(neck_coords, dict) else None)
    right_side_neck = safe_xy(neck_coords.get("right") if isinstance(neck_coords, dict) else None)
    anchor = safe_xy(neck_coords.get("anchor") if isinstance(neck_coords, dict) else None)

    left_shoulder, right_shoulder = [safe_xy(p) for p in shoulders_coords] if shoulders_coords else ((0,0),(0,0))
    left_clavicle, right_clavicle = [safe_xy(p) for p in clavicules_coords] if clavicules_coords else ((0,0),(0,0))
    left_elbow, right_elbow = [safe_xy(p) for p in elbow_coords] if elbow_coords else ((0,0),(0,0))
    left_wrist, right_wrist = [safe_xy(p) for p in wrists_coords] if wrists_coords else ((0,0),(0,0))
    left_hip, right_hip = [safe_xy(p) for p in hips_coords] if hips_coords else ((0,0),(0,0))
    left_eye, right_eye = [safe_xy(p) for p in eye_coords] if eye_coords else ((0,0),(0,0))
    left_ear, right_ear = [safe_xy(p) for p in ear_coords] if ear_coords else ((0,0),(0,0))
    mouth = safe_xy(mouth_coords)

    # ---------------------------
    # 🔹 Mise à jour safe des keypoints
    # ---------------------------
    safe_update(0, nose, keypoints_np, "nose")
    safe_update(1, neck_center, keypoints_np, "neck")

    safe_update(2, right_shoulder, keypoints_np, "right_shoulder")
    safe_update(3, right_elbow, keypoints_np, "right_elbow")
    safe_update(4, right_wrist, keypoints_np, "right_wrist")

    safe_update(5, left_shoulder, keypoints_np, "left_shoulder")
    safe_update(6, left_elbow, keypoints_np, "left_elbow")
    safe_update(7, left_wrist, keypoints_np, "left_wrist")

    safe_update(8, right_hip, keypoints_np, "right_hip")
    safe_update(11, left_hip, keypoints_np, "left_hip")

    safe_update(14, right_eye, keypoints_np, "right_eye")
    safe_update(15, left_eye, keypoints_np, "left_eye")
    safe_update(16, right_ear, keypoints_np, "right_ear")
    safe_update(17, left_ear, keypoints_np, "left_ear")
    safe_update(18, mouth, keypoints_np, "mouth")

    safe_update(19, right_clavicle, keypoints_np, "right_clavicle")
    safe_update(20, left_clavicle, keypoints_np, "left_clavicle")

    safe_update(21, chin, keypoints_np, "chin")
    safe_update(22, left_side_neck, keypoints_np, "left_side_neck")
    safe_update(23, right_side_neck, keypoints_np, "right_side_neck")
    safe_update(24, anchor, keypoints_np, "anchor")

    # ---------------------------
    # 🔹 Conversion numpy → tensor
    # ---------------------------
    keypoints_np = np.expand_dims(keypoints_np, axis=0)  # [1,25,3]
    keypoints_np = np.repeat(keypoints_np, B, axis=0)    # [B,25,3]
    keypoints_tensor = torch.from_numpy(keypoints_np).to(device)

    # ---------------------------
    # 🔹 DEBUG VISUEL / VISUAL DEBUG
    # ---------------------------
    if debug and debug_dir is not None and frame_counter is not None:
        debug_draw_openpose_skeleton(
            pose_full_image=pose_full_image,
            keypoints_tensor=keypoints_tensor,
            debug_dir=debug_dir,
            frame_counter=frame_counter
        )

    return keypoints_tensor
#----------------------------------------------------------------------------------------------------------------

def resize_pose(pose_tile, H_latent, W_latent):
    target_h = H_latent * 8
    target_w = W_latent * 8

    if pose_tile.shape[-2:] != (target_h, target_w):
        return F.interpolate(
            pose_tile,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
    return pose_tile


def prepare_inputs(latent_tile, pose_tile, cf_embeds, device):
    pos_embeds, neg_embeds = cf_embeds

    latent_fp32 = latent_tile.to(device=device, dtype=torch.float32)
    pose_fp32 = pose_tile.to(device=device, dtype=torch.float32)

    pos_fp32 = pos_embeds.to(device=device, dtype=torch.float32)
    neg_fp32 = neg_embeds.to(device=device, dtype=torch.float32) if neg_embeds is not None else None

    return latent_fp32, pose_fp32, pos_fp32, neg_fp32


def add_noise(latent, scheduler, t, noise_strength=0.5):
    noise = torch.randn_like(latent) * noise_strength
    latent_noisy = scheduler.add_noise(latent, noise, t)
    return torch.clamp(latent_noisy, -20, 20)


def apply_cfg(latent_input, pos_embeds, neg_embeds, guidance_scale):
    if neg_embeds is not None:
        latent_input = torch.cat([latent_input] * 2)
        embeds = torch.cat([neg_embeds, pos_embeds])
        return latent_input, embeds, True
    return latent_input, pos_embeds, False


def compute_noise_pred(unet, controlnet, latent_input, t, embeds, pose):
    down_samples, mid_sample = controlnet(
        latent_input,
        t,
        encoder_hidden_states=embeds,
        controlnet_cond=pose,
        return_dict=False
    )

    noise_pred = unet(
        latent_input,
        t,
        encoder_hidden_states=embeds,
        down_block_additional_residuals=down_samples,
        mid_block_additional_residual=mid_sample,
        return_dict=False
    )[0]

    return torch.nan_to_num(noise_pred, nan=0.0, posinf=1.0, neginf=-1.0)


def merge_cfg(noise_pred, guidance_scale, use_cfg):
    if use_cfg:
        noise_uncond, noise_text = noise_pred.chunk(2)
        return noise_uncond + guidance_scale * (noise_text - noise_uncond)
    return noise_pred


def compute_adaptive_importance(latent):
    with torch.no_grad():
        blurred = F.avg_pool2d(latent, kernel_size=3, stride=1, padding=1)
        high_freq = torch.abs(latent - blurred)
        importance = high_freq.mean(dim=1, keepdim=True)
        # normalisation douce
        importance = importance / (importance.mean() + 1e-6)
        # compression pour éviter extrêmes
        importance = torch.sqrt(importance)
        return torch.clamp(importance, 0.7, 1.3)





def controlnet_tile_fn(
    latent_tile,
    pose_tile,
    frame_counter,
    unet,
    controlnet,
    scheduler,
    cf_embeds,
    current_guidance_scale,
    controlnet_scale,
    device,
    target_dtype,
    **kwargs
):

    B, C, H_latent, W_latent = latent_tile.shape

    # =========================================================
    # 1️⃣ Resize pose
    # =========================================================
    pose_resized = resize_pose(pose_tile, H_latent, W_latent)
    # =========================================================
    # 2️⃣ Inputs
    # =========================================================
    latent_fp32, pose_fp32, pos_embeds, neg_embeds = prepare_inputs(
        latent_tile, pose_resized, cf_embeds, device
    )
    # =========================================================
    # 3️⃣ Timestep
    # =========================================================
    t = scheduler.timesteps[min(frame_counter, len(scheduler.timesteps) - 1)]

    # =========================================================
    # 4️⃣ Noise
    # =========================================================
    latent_noisy = add_noise(latent_fp32, scheduler, t)

    latent_input = scheduler.scale_model_input(latent_noisy, t)

    # =========================================================
    # 5️⃣ CFG
    # =========================================================
    latent_input, embeds, use_cfg = apply_cfg(
        latent_input, pos_embeds, neg_embeds, current_guidance_scale
    )

    latent_input = latent_input.to(target_dtype)
    embeds = embeds.to(target_dtype)
    pose_fp32 = pose_fp32.to(target_dtype)

    # =========================================================
    # 6️⃣ UNet + ControlNet
    # =========================================================
    noise_pred = compute_noise_pred(
        unet, controlnet, latent_input, t, embeds, pose_fp32
    )

    noise_pred = merge_cfg(noise_pred, current_guidance_scale, use_cfg)

    # =========================================================
    # 7️⃣ Scheduler step
    # =========================================================
    latents_out = scheduler.step(noise_pred, t, latent_noisy).prev_sample

    # =========================================================
    # 🔥 8️⃣ Adaptive blending (clé)
    # =========================================================
    importance = compute_adaptive_importance(latent_fp32)

    delta = compute_delta(
        latents_out,
        latent_fp32,
        controlnet_scale,
        importance
    )

    latents_final = latent_fp32 + delta

    return latents_final.to(target_dtype)

#---------------------------------------------------------------------------------------------------------------------------------------------



def log_frame_error(img_path, error: Exception, verbose: bool = True):
    print(f"\n[FRAME ERROR] {img_path}")
    print(f"Type d'erreur : {type(error).__name__}")
    print(f"Message d'erreur : {error}")

    if verbose:
        print("Traceback complet :")
        traceback.print_exc()


def prepare_controlnet(
    controlnet,
    freeze: bool = True,
    enable_slicing: bool = True,
    device=None,
    dtype=None,
    verbose: bool = True
):
    """
    Prépare un ControlNet :
    - eval mode
    - freeze des poids
    - attention slicing (si dispo)
    - move device / dtype
    - init pose_sequence

    Returns:
        controlnet, pose_sequence (None par défaut)
    """

    # ---- eval mode
    controlnet.eval()
    if verbose:
        print("✅ ControlNet en mode eval")

    # ---- freeze
    if freeze:
        for p in controlnet.parameters():
            p.requires_grad = False
        if verbose:
            print("✅ Paramètres gelés")

    # ---- attention slicing
    if enable_slicing:
        fn = getattr(controlnet, "enable_attention_slicing", None)
        if callable(fn):
            fn()
            if verbose:
                print("✅ Attention slicing activé")
        else:
            if verbose:
                print("⚠ enable_attention_slicing non disponible")

    # ---- device / dtype
    if device is not None or dtype is not None:
        controlnet = controlnet.to(device=device, dtype=dtype)
        if verbose:
            print(f"✅ Déplacé sur {device} / {dtype}")

    # ---- init pose
    pose_sequence = None

    return controlnet, pose_sequence

def fix_pose_sequence(
    pose_sequence: torch.Tensor,
    total_frames: int,
    device=None,
    dtype=None,
    verbose: bool = True
) -> torch.Tensor:
    """
    Ajuste une séquence de poses au bon nombre de frames avec interpolation.

    Args:
        pose_sequence: Tensor (F, C, H, W)
        total_frames: nombre de frames cible
        device: device cible (optionnel)
        dtype: dtype cible (optionnel)
        verbose: afficher logs

    Returns:
        Tensor (F, C, H, W)
    """
    print(f"🎞 fix_pose_sequence - Frames JSON: {pose_sequence.shape[0]}")
    print(f"🎞 fix_pose_sequence - Frames attendues: {total_frames}")

    if pose_sequence.shape[0] != total_frames:
        if verbose:
            print("⚠ Ajustement du nombre de frames OpenPose")

        # (F, C, H, W) → (1, C, F, H, W)
        pose_sequence = pose_sequence.permute(1, 0, 2, 3).unsqueeze(0)

        pose_sequence = F.interpolate(
            pose_sequence,
            size=(total_frames, pose_sequence.shape[-2], pose_sequence.shape[-1]),
            mode='trilinear',
            align_corners=False
        )

        # retour → (F, C, H, W)
        pose_sequence = pose_sequence.squeeze(0).permute(1, 0, 2, 3)

    # Fix device + dtype
    if device is not None or dtype is not None:
        pose_sequence = pose_sequence.to(device=device, dtype=dtype)

    if verbose:
        print(
            "✅ PoseSequence final:",
            pose_sequence.shape,
            pose_sequence.device,
            pose_sequence.dtype
        )

    return pose_sequence



def tensor_to_pil(tensor):
    """
    Convertit un tensor torch [C,H,W] ou [H,W] en PIL.Image RGB.
    """
    if tensor.dim() == 3:
        C, H, W = tensor.shape
        if C == 1:
            array = tensor[0].cpu().numpy()  # [H,W]
            pil_img = Image.fromarray(array).convert("RGB")
        elif C == 3:
            array = tensor.permute(1, 2, 0).cpu().numpy()  # [H,W,C]
            pil_img = Image.fromarray(array)
        else:
            raise ValueError(f"Tensor avec {C} canaux non supporté")
    elif tensor.dim() == 2:
        pil_img = Image.fromarray(tensor.cpu().numpy()).convert("RGB")
    else:
        raise ValueError(f"Tensor shape non supportée: {tensor.shape}")
    return pil_img



def save_debug_pose_image(pose_tensor, frame_counter, output_dir, cfg=None, prefix="openpose"):
    """
    Sauvegarde une image de pose pour contrôle visuel.

    pose_tensor : torch.Tensor [C,H,W] ou [H,W]
    frame_counter : int, numéro de frame
    output_dir : str, dossier où sauvegarder
    cfg : dict ou None, peut contenir paramètre 'visual_debug' pour activer/désactiver
    prefix : str, préfixe du fichier
    """

    # Vérifie si le debug visuel est activé
    if cfg is not None and cfg.get("visual_debug") is False:
        return

    # Convertir tensor en uint8 [0,255]
    pose_img = (pose_tensor * 255).clamp(0, 255).byte()

    # Fonction interne pour gérer tous les formats [C,H,W], [H,W]
    def tensor_to_pil(tensor):
        if tensor.dim() == 3:
            C, H, W = tensor.shape
            if C == 1:
                array = tensor[0].cpu().numpy()  # [H,W]
                pil_img = Image.fromarray(array).convert("RGB")
            elif C == 3:
                array = tensor.permute(1, 2, 0).cpu().numpy()  # [H,W,C]
                pil_img = Image.fromarray(array)
            else:
                raise ValueError(f"Tensor avec {C} canaux non supporté")
        elif tensor.dim() == 2:
            pil_img = Image.fromarray(tensor.cpu().numpy()).convert("RGB")
        else:
            # Si la tensor a une forme inattendue, on essaie de la "squeezer"
            tensor = tensor.squeeze()
            if tensor.dim() in [2, 3]:
                return tensor_to_pil(tensor)
            raise ValueError(f"Tensor shape non supportée: {tensor.shape}")
        return pil_img

    pil_pose = tensor_to_pil(pose_img)

    # Création du dossier si nécessaire
    os.makedirs(output_dir, exist_ok=True)

    # Nom du fichier : openpose_00001.png
    filename = f"{prefix}_{frame_counter:05d}.png"
    path = os.path.join(output_dir, filename)

    pil_pose.save(path)
    print(f"[DEBUG] Pose sauvegardée : {path}")

def save_debug_pose_image_mini(pose_tensor, frame_counter, output_dir, cfg=None, prefix="openpose"):
    """
    Sauvegarde la pose détectée pour vérification visuelle.

    Args:
        pose_tensor (torch.Tensor): Tensor BCHW ou CHW (1,3,H,W ou 3,H,W)
        frame_counter (int): numéro de la frame
        output_dir (Path): dossier de sortie pour sauvegarde
        cfg (dict, optional): configuration, active si cfg.get("debug_pose_visual", False) est True
        prefix (str): préfixe du fichier image (default: 'openpose')
    """
    if cfg is None or not cfg.get("debug_pose_visual", False):
        return

    # S'assurer que le tensor est BCHW
    if pose_tensor.ndim == 3:  # CHW -> BCHW
        pose_tensor = pose_tensor.unsqueeze(0)

    pose_tensor = pose_tensor[0]  # retirer batch

    # Limiter à 3 canaux
    if pose_tensor.shape[0] > 3:
        pose_tensor = pose_tensor[:3, :, :]

    # CHW -> HWC
    pose_np = pose_tensor.permute(1, 2, 0).cpu().numpy()
    # Normalisation 0-255
    pose_np = (pose_np - pose_np.min()) / (pose_np.max() - pose_np.min() + 1e-8) * 255.0
    pose_np = pose_np.astype("uint8")
    img = Image.fromarray(pose_np)

    # Nom de fichier : openpose_0001.png
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{prefix}_{frame_counter:04d}.png"
    img.save(filename)

def debug_pose_visual(pose_tensor, frame_counter, cfg=None, title="Pose Debug"):
    """
    Affiche la pose détectée pour vérification visuelle.

    Args:
        pose_tensor (torch.Tensor): Tensor BCHW ou CHW (1,3,H,W ou 3,H,W)
        frame_counter (int): numéro de la frame
        cfg (dict, optional): configuration, active si cfg.get("debug_pose_visual", False) est True
        title (str): titre pour l'affichage
    """
    if cfg is None or not cfg.get("debug_pose_visual", False):
        return

    # S'assurer que le tensor est BCHW
    if pose_tensor.ndim == 3:  # CHW -> BCHW
        pose_tensor = pose_tensor.unsqueeze(0)

    pose_tensor = pose_tensor[0]  # retirer batch

    # Limiter à 3 canaux
    if pose_tensor.shape[0] > 3:
        pose_tensor = pose_tensor[:3, :, :]

    # CHW -> HWC pour PIL
    pose_np = pose_tensor.permute(1, 2, 0).cpu().numpy()
    pose_np = (pose_np - pose_np.min()) / (pose_np.max() - pose_np.min() + 1e-8) * 255.0
    pose_np = pose_np.astype("uint8")
    img = Image.fromarray(pose_np)

    # Affichage rapide avec matplotlib
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{title} - Frame {frame_counter}")
    plt.show(block=False)
    plt.pause(0.1)  # court délai pour rafraîchir
    plt.close()


#------------- JSON TO POSE SEQUENCE --------------------

def convert_json_to_pose_sequence(anim_data, H=512, W=512,
                                  device="cuda", dtype=torch.float16,
                                  debug=False, output_dir=None):
    """
    Convertit un JSON d'animation OpenPose en tensor ControlNet, avec centrage et scaling automatique.
    Output : [num_frames, 3, H, W], dtype et device configurables.
    """
    frames = anim_data.get("animation", [])
    pose_images = []

    # --- Détecter le bounding box global des keypoints ---
    all_x = []
    all_y = []
    for frame in frames:
        for kp in frame.get("keypoints", []):
            all_x.append(kp["x"])
            all_y.append(kp["y"])

    if len(all_x) == 0 or len(all_y) == 0:
        raise ValueError("Aucun keypoint trouvé dans le JSON.")

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Scale et translation pour centrer et remplir le canvas
    scale_x = (W - 20) / (max_x - min_x + 1e-6)  # marge 10px
    scale_y = (H - 20) / (max_y - min_y + 1e-6)
    scale = min(scale_x, scale_y)

    offset_x = (W - (max_x - min_x) * scale) / 2 - min_x * scale
    offset_y = (H - (max_y - min_y) * scale) / 2 - min_y * scale

    for idx, frame in enumerate(frames):
        keypoints = frame.get("keypoints", [])
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # --- Dessin des points ---
        for kp in keypoints:
            x = int(kp["x"] * scale + offset_x)
            y = int(kp["y"] * scale + offset_y)
            conf = kp.get("confidence", 1.0)
            if conf > 0.3:
                cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1)

        # --- Dessin des connexions ---
        skeleton = [
            (0, 1),  # tête → torse
            (1, 2),  # torse → bras gauche
            (1, 3),  # torse → bras droit
            (1, 4),  # torse → jambe gauche
            (1, 5),  # torse → jambe droite
        ]
        for a, b in skeleton:
            if a < len(keypoints) and b < len(keypoints):
                x1 = int(keypoints[a]["x"] * scale + offset_x)
                y1 = int(keypoints[a]["y"] * scale + offset_y)
                x2 = int(keypoints[b]["x"] * scale + offset_x)
                y2 = int(keypoints[b]["y"] * scale + offset_y)
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        img = torch.from_numpy(canvas).float() / 255.0
        img = img.permute(2, 0, 1)  # C,H,W
        pose_images.append(img)

        # Debug
        if debug and output_dir is not None:
            cv2.imwrite(f"{output_dir}/debug_pose_{idx:03d}.png", canvas)

    pose_sequence = torch.stack(pose_images).to(device=device, dtype=dtype)
    pose_sequence = pose_sequence * 2.0 - 1.0  # [-1,1]

    if debug:
        print(f"[JSON->POSE] shape: {pose_sequence.shape}")
        print(f"[JSON->POSE] min/max: {pose_sequence.min().item()} / {pose_sequence.max().item()}")

    return pose_sequence

def convert_json_to_pose_sequence_debug(anim_data, H=512, W=512, original_w=512, original_h=512,
                                  device="cuda", dtype=torch.float16, debug=False, output_dir=None):
    """
    Convertit un JSON d'animation OpenPose simplifié en tensor utilisable par ControlNet.

    Args:
        anim_data: dict JSON avec "animation" -> frames -> keypoints
        H, W: résolution finale du canvas
        original_w, original_h: résolution originale des keypoints
        device: "cuda" ou "cpu"
        dtype: torch dtype (ex: torch.float16)
        debug: bool, sauvegarde les images pour visualisation
        output_dir: chemin pour debug images (optionnel)

    Returns:
        pose_sequence: tensor [num_frames, 3, H, W] (RGB type)
    """
    frames = anim_data.get("animation", [])
    pose_images = []

    for idx, frame in enumerate(frames):
        keypoints = frame.get("keypoints", [])

        # Image noire
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # --- Dessin des points ---
        for kp in keypoints:
            # remapping keypoints vers la résolution finale
            x = int(kp["x"] * W / original_w)
            y = int(kp["y"] * H / original_h)
            conf = kp.get("confidence", 1.0)

            if conf > 0.3:
                cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1)

        # --- Dessin des connexions (squelette simple) ---
        skeleton = [
            (0, 1),  # tête → torse
            (1, 2),  # torse → bras gauche
            (1, 3),  # torse → bras droit
            (1, 4),  # torse → jambe gauche
            (1, 5),  # torse → jambe droite
        ]

        for a, b in skeleton:
            if a < len(keypoints) and b < len(keypoints):
                x1 = int(keypoints[a]["x"] * W / original_w)
                y1 = int(keypoints[a]["y"] * H / original_h)
                x2 = int(keypoints[b]["x"] * W / original_w)
                y2 = int(keypoints[b]["y"] * H / original_h)
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # --- Conversion en tensor ---
        img = torch.from_numpy(canvas).float() / 255.0  # [H, W, C]
        img = img.permute(2, 0, 1)  # → [C, H, W]

        pose_images.append(img)

        # --- Debug save ---
        if debug and output_dir is not None:
            debug_path = f"{output_dir}/debug_pose_{idx:03d}.png"
            cv2.imwrite(debug_path, (canvas).astype(np.uint8))

    # --- Stack frames + normalisation [-1,1] ---
    pose_sequence = torch.stack(pose_images).to(device=device, dtype=dtype)
    pose_sequence = pose_sequence * 2.0 - 1.0  # [0,1] → [-1,1]

    if debug:
        print(f"[JSON->POSE] shape: {pose_sequence.shape}")
        print(f"[JSON->POSE] min/max: {pose_sequence.min().item()} / {pose_sequence.max().item()}")

    return pose_sequence



def build_control_latent_debug(input_pil, vae, device="cuda", latent_scale=0.18215):

    print("\n================ CONTROL LATENT DEBUG ================")

    # 1. Canny
    control = create_canny_control(input_pil)

    print("[STEP 1] RAW CONTROL")
    print(" shape:", control.shape)
    print(" dtype:", control.dtype)
    print(" min/max:", control.min().item(), control.max().item())

    # 2. 1 → 3 channels
    if control.shape[1] == 1:
        control = control.repeat(1, 3, 1, 1)

    # 3. Normalize PROPERLY (CRUCIAL)
    control = control.clamp(0, 1)          # sécurité
    control = control * 2.0 - 1.0          # [-1,1]

    print("[STEP 2] NORMALIZED")
    print(" min/max:", control.min().item(), control.max().item())

    # 4. Move to device FP32
    control = control.to(device=device, dtype=torch.float32)

    print("[STEP 3] DEVICE")
    print(" device:", control.device)
    print(" dtype:", control.dtype)

    # 5. Sync VAE
    print("[STEP 4] VAE STATE")
    print(" vae dtype:", next(vae.parameters()).dtype)
    print(" vae device:", next(vae.parameters()).device)

    # 🔥 FORCER cohérence VAE
    vae = vae.to(device=device, dtype=torch.float32)

    # 6. Encode SAFE (no autocast)
    with torch.no_grad():
        try:
            latent_dist = vae.encode(control).latent_dist
            latent = latent_dist.sample()
        except Exception as e:
            print("❌ VAE ENCODE CRASH:", e)
            raise

    print("[STEP 5] LATENT RAW")
    print(" min/max:", latent.min().item(), latent.max().item())
    print(" NaN:", torch.isnan(latent).sum().item())

    # 🚨 CHECK NaN
    if torch.isnan(latent).any():
        print("⚠️ NaN DETECTED → applying fallback")

        # fallback 1: zero latent
        latent = torch.zeros_like(latent)

        # fallback 2 (optionnel):
        # latent = torch.randn_like(latent) * 0.1

    # 7. Scale (SD standard)
    latent = latent * latent_scale

    print("[STEP 6] SCALED LATENT")
    print(" min/max:", latent.min().item(), latent.max().item())

    # 8. Back to FP16
    latent = latent.to(dtype=torch.float16)

    print("[FINAL]")
    print(" dtype:", latent.dtype)
    print(" device:", latent.device)
    print("=====================================================\n")

    return latent

# ---------------- Control -> Latent sécurisé ----------------
def control_to_latent_safe(control_tensor, vae, device="cuda", LATENT_SCALE=1.0):
    # 🔥 FORCE VAE EN FP32
    vae = vae.to(device=device, dtype=torch.float32)

    control_tensor = control_tensor.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latent = vae.encode(control_tensor).latent_dist.sample()

    return latent * LATENT_SCALE

def process_latents_streamed(control_latent, mini_latents=None, mini_weight=0.5, device="cuda"):
    """
    Fusionne ControlNet / mini-latents frame par frame, patch par patch
    pour réduire l'empreinte VRAM.
    """
    # On garde tout en float16 tant que possible
    control_latent = control_latent.to(device=device, dtype=torch.float16)

    if mini_latents is not None:
        mini_latents = mini_latents.to(device=device, dtype=torch.float16)

    # Initialisation finale du tensor latents en float16
    latents = control_latent.clone()

    # Si mini_latents existe, on fait un mix patch par patch
    if mini_latents is not None:
        B, C, H, W = latents.shape
        patch_size = 16  # petit patch pour limiter la VRAM
        for y in range(0, H, patch_size):
            y1 = min(y + patch_size, H)
            for x in range(0, W, patch_size):
                x1 = min(x + patch_size, W)

                # Sélection patch
                patch_main = latents[:, :, y:y1, x:x1]
                patch_mini = mini_latents[:, :, y:y1, x:x1]

                # Mix float16 → float16 pour VRAM
                patch_main = (1 - mini_weight) * patch_main + mini_weight * patch_mini

                # Écriture patch back
                latents[:, :, y:y1, x:x1] = patch_main

                # Nettoyage immédiat pour libérer VRAM
                del patch_main, patch_mini
                torch.cuda.empty_cache()

    return latents


def match_latent_size(latents_main, *tensors):
    """
    Interpole tous les tensors pour correspondre à la taille HxW de latents_main.
    """
    matched = []
    for t in tensors:
        if t.shape[2:] != latents_main.shape[2:]:
            t = F.interpolate(t, size=latents_main.shape[2:], mode='bilinear', align_corners=False)
        matched.append(t)
    return matched if len(matched) > 1 else matched[0]


def pad_to_multiple(x, mult=8):
    B, C, H, W = x.shape
    pad_H = (mult - H % mult) % mult
    pad_W = (mult - W % mult) % mult
    if pad_H == 0 and pad_W == 0:
        return x
    return F.pad(x, (0, pad_W, 0, pad_H))  # pad right & bottom

def gaussian_blend_mask(H, W, overlap):
    """Crée un masque gaussien pour fusionner les tiles avec overlap."""

    y = np.linspace(-1,1,H)
    x = np.linspace(-1,1,W)
    xv, yv = np.meshgrid(x,y)
    mask = np.exp(-(xv**2 + yv**2) / 0.5)  # ajuste le sigma si nécessaire
    mask = torch.tensor(mask, dtype=torch.float32)
    return mask


#---------------------------------------------------------

# 🔹 Récupère les coordonnées (x,y) d’un keypoint spécifique dans le batch
# 🔹 Récupère les coordonnées (x,y) d’un keypoint spécifique dans le batch
def get_point(kp_tensor, idx):
    return kp_tensor[:, idx, :2]  # [B,2]

# 🔹 Applique la différence entre latents avant/après OpenPose
#   Permet de conserver l’impact du pose controlnet.
def apply_openpose_delta(latents, latents_before, latents_after, mask):
    if latents_before is not None and latents_after is not None:
        delta = latents_after - latents_before
        delta = torch.clamp(delta, -0.15, 0.15)
        latents = latents + delta * mask * 0.5
    return latents

# -------------------- Fonction utilitaire --------------------
def compute_torso_angle(keypoints):
    """
    Calcule l'angle du torse selon les épaules (radians).
    """
    right_shoulder = get_point(keypoints, 2)
    left_shoulder = get_point(keypoints, 5)
    vec = right_shoulder - left_shoulder
    angle = torch.atan2(vec[:,1], vec[:,0])  # [B]
    torso_center = (right_shoulder + left_shoulder) * 0.5
    return angle, torso_center


# -------------------- Fonction principale -----------------------------------------------------------------


def apply_hair_motion(
    latents,
    mask_hair,
    grid,
    H,
    W,
    frame_counter,
    device,
    delta_px=None,
    prev_hair_field=None,
    debug=False,
    debug_dir=None
):
    """
    Hair motion version cinéma amplifiée :
    - mouvements plus marqués
    - vent + gravité + micro-souplesse
    - inertie adaptative pour fluidité
    """

    B = latents.shape[0]
    t = frame_counter
    t_wind1 = torch.tensor(t / 15.0, device=device)
    t_wind2 = torch.tensor(t / 60.0, device=device)

    # -------------------- Multi-échelle bruit --------------------
    def multi_noise(grid, t, scales=[0.05,0.15,0.3], weights=[1.0,0.5,0.25]):
        val = 0
        for s, w in zip(scales, weights):
            val += w * smooth_noise(grid, t, scale=s)
        return val

    noise_x = multi_noise(grid, t)
    noise_y = multi_noise(grid, t + 123, scales=[0.08,0.2,0.4], weights=[1.0,0.5,0.25])

    # -------------------- Champ delta de base amplifié --------------------
    hair_delta_field = torch.zeros_like(grid)
    hair_delta_field[...,0] = 0.06 * noise_x   # x2 vs original
    hair_delta_field[...,1] = 0.10 * noise_y   # x2 vs original

    # -------------------- Vent dynamique --------------------
    wind_dir = torch.tensor([[1.0,0.2],[0.3,0.1]], device=device).mean(dim=0).view(1,1,1,2)
    wind_strength = 0.04 + 0.02 * torch.sin(t_wind1) + 0.01 * torch.sin(t_wind2)
    wind_delta = wind_dir * wind_strength

    # -------------------- Gravité légère --------------------
    gravity_delta = torch.zeros_like(grid)
    gravity_delta[...,1] = 0.008  # plus tombant

    # -------------------- Influence du torse --------------------
    if delta_px is not None:
        speed = torch.norm(delta_px, dim=-1, keepdim=True)
        hair_delta_field *= (1.0 + 3.5 * speed)
        wind_delta *= (1.0 + 2.0 * speed)
        gravity_delta *= (1.0 + 0.8 * speed)

    # -------------------- Inertie adaptative --------------------
    inertia = 0.7  # moins amorti, plus cinématique
    if prev_hair_field is not None:
        hair_delta_field = inertia * prev_hair_field + (1 - inertia) * hair_delta_field

    # -------------------- Masque + falloff racine→pointe --------------------
    mask_hair_expand = mask_hair.permute(0,2,3,1)
    yy = torch.linspace(0,1,H,device=device).view(1,H,1,1)
    smooth_falloff = yy**2.5 * (3 - 2*yy**1.5)  # accentue le mouvement sur les pointes
    mask_hair_expand = mask_hair_expand * smooth_falloff

    # -------------------- Micro-souplesse physique --------------------
    spring = 0.006 * torch.sin(t*0.5 + grid[...,1:2]*3.0)
    hair_delta_field[...,1:2] += spring

    # -------------------- Micro noise --------------------
    micro_noise = 0.002 * (torch.rand_like(hair_delta_field)-0.5)
    hair_delta_field += micro_noise

    # -------------------- Application --------------------
    grid_hair = grid + hair_delta_field * mask_hair_expand
    grid_hair += wind_delta * mask_hair_expand
    grid_hair += gravity_delta * mask_hair_expand

    # -------------------- Normalisation pour grid_sample --------------------
    grid_hair[...,0] = 2.0 * grid_hair[...,0] / (W-1) - 1.0
    grid_hair[...,1] = 2.0 * grid_hair[...,1] / (H-1) - 1.0

    # -------------------- Sampling --------------------
    latents_out = F.grid_sample(latents, grid_hair, align_corners=True)

    if debug:
        print("[DEBUG] Hair motion cinema amplified applied")

    return latents_out, hair_delta_field

def apply_hair_motion_cinema(  # version cinéma
    latents,
    mask_hair,
    grid,
    H,
    W,
    frame_counter,
    device,
    delta_px=None,
    prev_hair_field=None,
    debug=False
):
    B = latents.shape[0]

    # -------------------- Temps --------------------
    t = frame_counter
    t_wind1 = torch.tensor(t / 15.0, device=device)
    t_wind2 = torch.tensor(t / 60.0, device=device)

    # -------------------- Multi-échelle bruit --------------------
    def multi_noise(grid, t, scales=[0.05,0.15,0.3], weights=[1.0,0.5,0.25]):
        val = 0
        for s,w in zip(scales, weights):
            val += w * smooth_noise(grid, t, scale=s)
        return val

    noise_x = multi_noise(grid, t)
    noise_y = multi_noise(grid, t + 123, scales=[0.08,0.2,0.4], weights=[1.0,0.5,0.25])

    # -------------------- Champ delta de base --------------------
    hair_delta_field = torch.zeros_like(grid)
    hair_delta_field[...,0] = 0.03 * noise_x
    hair_delta_field[...,1] = 0.05 * noise_y

    # -------------------- Vent dynamique --------------------
    wind_dir = torch.tensor([[1.0,0.2],[0.3,0.1]], device=device).mean(dim=0).view(1,1,1,2)
    wind_strength = 0.02 + 0.01 * torch.sin(t_wind1) + 0.005 * torch.sin(t_wind2)
    wind_delta = wind_dir * wind_strength

    # -------------------- Gravité légère --------------------
    gravity_delta = torch.zeros_like(grid)
    gravity_delta[...,1] = 0.004  # constant downwards

    # -------------------- Influence du torse --------------------
    if delta_px is not None:
        speed = torch.norm(delta_px, dim=-1, keepdim=True)
        hair_delta_field *= (1.0 + 2.5*speed)
        wind_delta *= (1.0 + 1.5*speed)
        gravity_delta *= (1.0 + 0.5*speed)

    # -------------------- Inertie adaptative --------------------
    inertia = 0.85
    if prev_hair_field is not None:
        hair_delta_field = inertia * prev_hair_field + (1-inertia) * hair_delta_field

    # -------------------- Masque + falloff racine→pointe --------------------
    mask_hair_expand = mask_hair.permute(0,2,3,1)
    yy = torch.linspace(0,1,H,device=device).view(1,H,1,1)
    smooth_falloff = yy**2 * (3-2*yy)
    mask_hair_expand = mask_hair_expand * smooth_falloff

    # -------------------- Micro-souplesse physique --------------------
    spring = 0.003 * torch.sin(t*0.5 + grid[...,1:2]*3.0)
    hair_delta_field[...,1:2] += spring

    # -------------------- Micro noise --------------------
    micro_noise = 0.001 * (torch.rand_like(hair_delta_field)-0.5)
    hair_delta_field += micro_noise

    # -------------------- Application --------------------
    grid_hair = grid + hair_delta_field * mask_hair_expand
    grid_hair += wind_delta * mask_hair_expand
    grid_hair += gravity_delta * mask_hair_expand

    # -------------------- Normalisation --------------------
    grid_hair[...,0] = 2.0 * grid_hair[...,0] / (W-1) - 1.0
    grid_hair[...,1] = 2.0 * grid_hair[...,1] / (H-1) - 1.0

    # -------------------- Sampling --------------------
    latents_out = F.grid_sample(latents, grid_hair, align_corners=True)

    if debug:
        print("[DEBUG] Hair motion cinema applied")

    return latents_out, hair_delta_field

def apply_hair_motion_v2(
    latents,
    mask_hair,
    grid,
    H,
    W,
    frame_counter,
    device,
    delta_px=None,
    prev_hair_field=None,
    debug=False
):
    B = latents.shape[0]

    # -------------------- Temps --------------------
    t_dict = {
        "noise": frame_counter,
        "wind1": torch.tensor(frame_counter / 15.0, device=device),
        "wind2": torch.tensor(frame_counter / 60.0, device=device),
    }

    # -------------------- Bruit multi-échelle --------------------
    def multi_scale_noise(grid, t, scales=[0.05, 0.15, 0.3], weights=[1.0, 0.5, 0.25]):
        result = 0
        for s, w in zip(scales, weights):
            result += w * smooth_noise(grid, t, scale=s)
        return result

    noise_x = multi_scale_noise(grid, t_dict["noise"])
    noise_y = multi_scale_noise(grid, t_dict["noise"] + 123, scales=[0.08, 0.2, 0.4], weights=[1.0,0.5,0.25])

    # -------------------- Hair delta --------------------
    hair_delta_field = torch.zeros_like(grid)
    hair_delta_field[..., 0] = 0.04 * noise_x
    hair_delta_field[..., 1] = 0.06 * noise_y

    # -------------------- Vent dynamique --------------------
    wind_dir = torch.tensor([1.0, 0.3], device=device).view(1,1,1,2)
    wind_strength = 0.03
    wind_delta = wind_dir * (wind_strength +
                             0.01 * torch.sin(t_dict["wind1"]) +
                             0.005 * torch.sin(t_dict["wind2"]))

    # -------------------- Influence du mouvement du torse --------------------
    if delta_px is not None:
        speed = torch.norm(delta_px, dim=-1, keepdim=True)
        hair_delta_field *= (1.0 + 2.5 * speed)  # plus naturel

    # -------------------- Inertie --------------------
    inertia = 0.85 if prev_hair_field is not None else 0.0
    if prev_hair_field is not None:
        hair_delta_field = inertia * prev_hair_field + (1 - inertia) * hair_delta_field

    # -------------------- Masque + Falloff racine→pointe --------------------
    mask_hair_expand = mask_hair.permute(0,2,3,1)

    yy = torch.linspace(0, 1, H, device=device).view(1,H,1,1)
    # Smoothstep pour transition plus douce
    falloff_root = yy**2 * (3 - 2*yy)  # smoothstep approximation
    mask_hair_expand = mask_hair_expand * falloff_root

    # -------------------- Micro noise supplémentaire --------------------
    micro_noise = 0.002 * (torch.rand_like(hair_delta_field) - 0.5)
    hair_delta_field += micro_noise

    # -------------------- Application --------------------
    grid_hair = grid + hair_delta_field * mask_hair_expand + wind_delta * mask_hair_expand

    # -------------------- Normalisation --------------------
    grid_hair[...,0] = 2.0 * grid_hair[...,0] / (W-1) - 1.0
    grid_hair[...,1] = 2.0 * grid_hair[...,1] / (H-1) - 1.0

    # -------------------- Sampling --------------------
    latents_out = F.grid_sample(latents, grid_hair, align_corners=True)

    if debug:
        print("[DEBUG] Hair motion applied with improved quality")

    return latents_out, hair_delta_field


def apply_torso_warp(
    latents,
    pose,
    mask_torso,
    grid,
    H,
    W,
    device,
    prev_delta_px=None,
    debug=False,
    debug_dir=None
):
    B = latents.shape[0]

    # -------------------- Centre du torse --------------------
    points_idx = [2, 5, 8, 11]
    pts = torch.stack([pose.get_point(i) for i in points_idx], dim=1)

    torso_center = pts.mean(dim=1)
    torso_center_px = torso_center * torch.tensor([W-1, H-1], device=device)
    torso_center_px = torso_center_px.view(B,1,1,2)

    # -------------------- Delta torse --------------------
    delta_px = pose.delta.clone()
    delta_px[...,0] *= W
    delta_px[...,1] *= H
    delta_px = delta_px.view(B,1,1,2)

    # -------------------- Lissage temporel --------------------
    if prev_delta_px is not None:
        alpha = 0.7
        delta_px = alpha * delta_px + (1 - alpha) * prev_delta_px

    # -------------------- Feather dynamique --------------------
    mask_torso = feather_dynamic_vectorized(
        mask_torso,
        delta_px,
        base_radius=3,
        sigma=1.5,
        scale=2.0
    )

    mask_expand = mask_torso.permute(0,2,3,1)

    # -------------------- Déformation non-linéaire (IMPORTANT) --------------------
    offset = grid - torso_center_px
    distance = torch.norm(offset, dim=-1, keepdim=True)

    # falloff spatial → centre bouge plus que les bords
    falloff = torch.exp(-distance / (0.35 * W))

    # -------------------- Warp torse --------------------
    grid_torso = grid + delta_px * mask_expand * falloff

    # -------------------- Normalisation --------------------
    grid_torso[...,0] = 2.0 * grid_torso[...,0] / (W-1) - 1.0
    grid_torso[...,1] = 2.0 * grid_torso[...,1] / (H-1) - 1.0

    # -------------------- Sampling --------------------
    latents_out = F.grid_sample(latents, grid_torso, align_corners=True)

    if debug:
        print("[DEBUG] Torso warp applied")

    return latents_out, delta_px


def apply_global_pose(
    latents,
    pose,
    prev_pose=None,
    H=None,
    W=None,
    device="cuda",
    strength=0.4,   # 🔥 IMPORTANT: boost réel
    debug=False,
    debug_dir=None
):
    B = latents.shape[0]

    if prev_pose is None:
        return latents, torch.zeros((B,1,1,2), device=device)

    # -------------------- centre --------------------
    idx = [2, 5, 8, 11]
    pts = torch.stack([pose.get_point(i) for i in idx], dim=1)
    prev_pts = torch.stack([prev_pose.get_point(i) for i in idx], dim=1)

    center = pts.mean(dim=1)
    prev_center = prev_pts.mean(dim=1)

    # -------------------- delta --------------------
    delta = center - prev_center

    delta_px = delta.clone()
    delta_px[..., 0] *= W
    delta_px[..., 1] *= H

    # 🔥 anti extinction du mouvement
    delta_px = torch.tanh(delta_px / 5.0) * 5.0

    delta_px = delta_px.view(B, 1, 1, 2)

    # -------------------- grid --------------------
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B, 1, 1, 1)

    # 🔥 BOOST IMPORTANT ICI
    grid_global = grid + delta_px * strength * 3.0

    # -------------------- normalize --------------------
    grid_global[..., 0] = 2.0 * grid_global[..., 0] / (W - 1) - 1.0
    grid_global[..., 1] = 2.0 * grid_global[..., 1] / (H - 1) - 1.0

    # -------------------- warp --------------------
    latents_out = F.grid_sample(latents, grid_global, align_corners=True)

    if debug:
        print(f"[DEBUG] Global delta px mean: {delta_px.abs().mean().item():.4f}")

    return latents_out, delta_px

def apply_global_pose_v1(
    latents,
    pose,
    prev_pose=None,
    H=None,
    W=None,
    device="cuda",
    strength=0.1,   # 🔥 beaucoup plus faible !
    debug=False,
    debug_dir=None
):
    B = latents.shape[0]

    # -------------------- Si pas de frame précédente → rien faire --------------------
    if prev_pose is None:
        return latents, torch.zeros((B,1,1,2), device=device)

    # -------------------- Centre actuel --------------------
    points_idx = [2, 5, 8, 11]
    pts = torch.stack([pose.get_point(i) for i in points_idx], dim=1)
    center = pts.mean(dim=1)

    # -------------------- Centre précédent --------------------
    prev_pts = torch.stack([prev_pose.get_point(i) for i in points_idx], dim=1)
    prev_center = prev_pts.mean(dim=1)

    # -------------------- Delta réel --------------------
    delta = center - prev_center  # 🔥 vrai mouvement

    # passage en pixels
    delta_px = delta.clone()
    delta_px[...,0] *= W
    delta_px[...,1] *= H
    delta_px = delta_px.view(B,1,1,2)

    # -------------------- Clamp pour stabilité --------------------
    delta_px = torch.clamp(delta_px, min=-10.0, max=10.0)

    # -------------------- Grille --------------------
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B,1,1,1)

    grid_global = grid + delta_px * strength

    # -------------------- Normalisation --------------------
    grid_global[...,0] = 2.0 * grid_global[...,0] / (W-1) - 1.0
    grid_global[...,1] = 2.0 * grid_global[...,1] / (H-1) - 1.0

    # -------------------- Warp --------------------
    latents_out = F.grid_sample(latents, grid_global, align_corners=True)

    if debug:
        print(f"[DEBUG] Global delta px: {delta_px.mean().item():.4f}")

    return latents_out, delta_px


#------------------------------------------------------------------------------------------
def apply_micro_boost_v1(latents, frame_counter, device, masks):
    """
    Applique un micro-boost sinusoidal sur différentes zones pour rendre l'image vivante.

    Args:
        latents (torch.Tensor): latents [B,C,H,W]
        frame_counter (int): compteur de frame
        device (str / torch.device)
        masks (dict): dictionnaire de la forme
            {
                "torso": (mask_tensor, phase, amplitude),
                "hair":  (mask_tensor, phase, amplitude),
                "face":  (mask_tensor, phase, amplitude)
            }

    Returns:
        torch.Tensor: latents modifiés
    """

    t = torch.tensor(frame_counter / 6.0, device=device)  # base temporelle

    for zone_name, (mask, phase, amp) in masks.items():
        if mask is None:
            continue
        latents += amp * mask * torch.sin(t + phase)

    return latents

def apply_micro_boost_v2(latents, frame_counter, device, masks):
    t = torch.tensor(frame_counter / 6.0, device=device)

    total = torch.zeros_like(latents)

    for zone_name, (mask, phase, amp) in masks.items():
        if mask is None:
            continue

        total = total + amp * mask * torch.sin(t + phase)

    return latents + total


def apply_micro_boost(latents, frame_counter, device, masks, keypoints, prev_keypoints=None):

    t = torch.tensor(frame_counter / 6.0, device=device, dtype=latents.dtype)

    total = torch.zeros_like(latents)

    # safety guard
    if prev_keypoints is None:
        motion_strength = 0.0
    else:
        motion_strength = (keypoints[:, :, :2] - prev_keypoints[:, :, :2]).abs().mean()
        motion_strength = torch.clamp(motion_strength, 0.0, 0.01)
        motion_strength = 0.002 + motion_strength

    for zone_name, (mask, phase, amp) in masks.items():
        if mask is None:
            continue

        total += amp * mask * motion_strength * torch.sin(t + phase)

    return latents + total


def apply_micro_motion(latents: torch.Tensor, frame_counter: int, device, masks: dict, randomize: bool = True):
    """
    Applique un micro-boost sinusoidal sur différentes zones pour rendre l'image vivante.

    Args:
        latents (torch.Tensor): latents [B,C,H,W]
        frame_counter (int): compteur de frame
        device (str / torch.device)
        masks (dict): dictionnaire de la forme
            {
                "torso": (mask_tensor, phase, amplitude),
                "hair":  (mask_tensor, phase, amplitude),
                "face":  (mask_tensor, phase, amplitude),
                "mouth": (mask_tensor, phase, amplitude),  # optionnel
            }
        randomize (bool): si True, ajoute une légère variation aléatoire sur la sinusoïde

    Returns:
        torch.Tensor: latents modifiés
    """
    t = torch.tensor(frame_counter / 6.0, device=device)  # base temporelle

    for zone_name, params in masks.items():
        if params is None:
            continue
        mask, phase, amplitude = params
        if mask is None:
            continue

        # S'assurer que le mask est float et sur le bon device
        mask = mask.to(dtype=latents.dtype, device=device)

        # Variation aléatoire légère pour éviter un mouvement trop mécanique
        noise = torch.rand_like(mask) * 0.005 if randomize else 0.0

        # Calcul du micro-mouvement sinusoidal
        delta = amplitude * mask * torch.sin(t + phase + noise)

        latents = latents + delta

    return latents

def calibrate_amplitude(mask, base_amp=0.002, max_amp=0.005):
    """
    Calibre automatiquement l'amplitude d'un micro-boost en fonction de la taille du masque.

    Args:
        mask (torch.Tensor): masque binaire [B,H,W] ou [H,W], valeurs 0-1
        base_amp (float): amplitude minimale
        max_amp (float): amplitude maximale

    Returns:
        float: amplitude calibrée
    """
    # Calculer proportion de pixels activés
    mask_area_ratio = mask.mean().item()  # entre 0 et 1

    # Interpolation linéaire
    amplitude = base_amp + (max_amp - base_amp) * mask_area_ratio

    return amplitude

def apply_face_warp(
    latents,
    pose,
    mask_face,
    grid,
    H,
    W,
    frame_counter,
    device=None,
    debug=False,
    debug_dir=None,
    smooth=0.85,
    prev_grid=None
):
    """
    Warp global du visage (hors bouche) avec micro-mouvements et vent.
    Supporte le lissage temporel via prev_grid si fourni.
    """
    if device is None:
        device = latents.device

    B, C, H, W = latents.shape
    latents_in = latents.clone()

    # =========================
    # Points faciaux
    # =========================
    facial_points = pose.estimate_facial_points_full(smooth=smooth)
    pose.set_prev_facial_points(facial_points)

    # =========================
    # Temps
    # =========================
    t_micro  = torch.tensor(frame_counter / 5.0, device=device)
    t_wind1  = torch.tensor(frame_counter / 15.0, device=device)
    t_wind2  = torch.tensor(frame_counter / 60.0, device=device)

    # =========================
    # Déplacements micro-visage
    # =========================
    face_delta = torch.zeros((B,1,1,2), device=device)
    face_delta[...,0] += 0.006 * torch.sin(t_micro)
    face_delta[...,1] += 0.009 * torch.sin(t_micro * 1.5)
    face_delta_exp = face_delta.expand(B,H,W,2).clone()

    # =========================
    # Micro-oscillations visage (vent)
    # =========================
    mask_face_exp = mask_face.permute(0,2,3,1)
    face_delta_exp[...,0] += mask_face_exp[...,0] * 0.004 * torch.sin(t_wind1)
    face_delta_exp[...,1] += mask_face_exp[...,0] * 0.003 * torch.sin(t_wind2)

    # =========================
    # Centre visage (nez)
    # =========================
    face_center = pose.get_point(0)
    face_center_px = face_center * torch.tensor([W-1, H-1], device=device)
    face_center_px = face_center_px.view(B,1,1,2)

    # =========================
    # Warp
    # =========================
    grid_face = grid.clone() - face_center_px
    grid_face = grid_face + face_delta_exp
    grid_face = grid_face + face_center_px

    # =========================
    # Lissage temporel si prev_grid fourni
    # =========================
    if prev_grid is not None:
        alpha = 0.85  # coefficient de lissage
        grid_face = alpha * prev_grid + (1.0 - alpha) * grid_face

    grid_face[...,0] = 2.0 * grid_face[...,0] / (W-1) - 1.0
    grid_face[...,1] = 2.0 * grid_face[...,1] / (H-1) - 1.0

    latents_out = F.grid_sample(latents, grid_face, align_corners=True)

    if debug:
        save_impact_map(latents_out, latents_in, debug_dir, frame_counter, prefix="face")
        print("[DEBUG] Face warp applied")

    return latents_out, face_delta_exp, facial_points


def apply_face_warp_v1(
    latents,
    pose,
    mask_face,
    grid,
    H,
    W,
    frame_counter,
    device=None,
    debug=False,
    debug_dir=None,
    smooth=0.85,
    prev_grid=None
):
    """
    Warp global du visage (hors bouche) avec micro-mouvements et vent.
    """
    if device is None:
        device = latents.device

    B, C, H, W = latents.shape
    latents_in = latents.clone()

    # =========================
    # Points faciaux
    # =========================
    facial_points = pose.estimate_facial_points_full(smooth=smooth)
    pose.set_prev_facial_points(facial_points)

    # =========================
    # Temps
    # =========================
    t_micro  = torch.tensor(frame_counter / 5.0, device=device)
    t_wind1  = torch.tensor(frame_counter / 15.0, device=device)
    t_wind2  = torch.tensor(frame_counter / 60.0, device=device)

    # =========================
    # Déplacements micro-visage
    # =========================
    face_delta = torch.zeros((B,1,1,2), device=device)
    face_delta[...,0] += 0.006 * torch.sin(t_micro)
    face_delta[...,1] += 0.009 * torch.sin(t_micro * 1.5)

    face_delta_exp = face_delta.expand(B,H,W,2).clone()

    # =========================
    # Micro-oscillations visage (vent)
    # =========================
    mask_face_exp = mask_face.permute(0,2,3,1)
    face_delta_exp[...,0] += mask_face_exp[...,0] * 0.004 * torch.sin(t_wind1)
    face_delta_exp[...,1] += mask_face_exp[...,0] * 0.003 * torch.sin(t_wind2)

    # =========================
    # Centre visage (nez)
    # =========================
    face_center = pose.get_point(0)
    face_center_px = face_center * torch.tensor([W-1, H-1], device=device)
    face_center_px = face_center_px.view(B,1,1,2)

    # =========================
    # Warp
    # =========================
    grid_face = grid.clone() - face_center_px
    grid_face = grid_face + face_delta_exp
    grid_face = grid_face + face_center_px

    grid_face[...,0] = 2.0 * grid_face[...,0] / (W-1) - 1.0
    grid_face[...,1] = 2.0 * grid_face[...,1] / (H-1) - 1.0

    latents_out = F.grid_sample(latents, grid_face, align_corners=True)

    if debug:
        save_impact_map(latents_out, latents_in, debug_dir, frame_counter, prefix="face")
        print("[DEBUG] Face warp applied")

    return latents_out, face_delta_exp, facial_points

def apply_mouth_smil(
    latents,
    pose,
    mask_mouth,
    grid,
    H,
    W,
    frame_counter,
    device=None,
    debug=False,
    debug_dir=None,
    smooth=0.85
):
    """
    Warp dynamique de la bouche (version visible) :
    - Sourire animé plus prononcé
    - Respiration bouche amplifiée
    - Micro-oscillations
    - Masque arrondi avec bord glow
    """
    if device is None:
        device = latents.device

    B, C, H, W = latents.shape
    latents_in = latents.clone()

    # =========================
    # Points faciaux
    # =========================
    facial_points = pose.estimate_facial_points_full(smooth=smooth)
    mouth_left  = facial_points['mouth_left']  # (B,2)
    mouth_right = facial_points['mouth_right']

    # =========================
    # Masque bouche
    # =========================
    mask_mouth = feather_inside_strict2(mask_mouth, radius=2, blur_kernel=3, sigma=1.0)
    mask_mouth = feather_outside_only_alpha2(mask_mouth, radius=1, sigma=1.0)  # rayon plus petit pour plus de force
    mask_mouth_exp = mask_mouth.permute(0,2,3,1)  # (B,H,W,1)

    # =========================
    # Sourire dynamique
    # =========================
    t_smile = torch.tensor(frame_counter / 8.0, device=device)
    smile_strength = 0.25  # plus visible
    delta_amp = 0.06       # amplitude verticale

    mouth_left_px  = mouth_left  * torch.tensor([W-1,H-1], device=device)
    mouth_right_px = mouth_right * torch.tensor([W-1,H-1], device=device)

    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x_grid = x_grid.unsqueeze(0).expand(B,-1,-1)
    y_grid = y_grid.unsqueeze(0).expand(B,-1,-1)

    # Distance aux coins plus concentrée
    dist_left  = torch.clamp(1.0 - torch.sqrt((x_grid - mouth_left_px[:,0,None,None])**2 +
                                              (y_grid - mouth_left_px[:,1,None,None])**2)/10.0, 0,1)
    dist_right = torch.clamp(1.0 - torch.sqrt((x_grid - mouth_right_px[:,0,None,None])**2 +
                                              (y_grid - mouth_right_px[:,1,None,None])**2)/10.0, 0,1)

    # Déplacement coins amplifié
    delta_left  = torch.zeros((B,H,W,2), device=device)
    delta_right = torch.zeros((B,H,W,2), device=device)
    delta_left[...,0]  = -smile_strength * torch.sin(t_smile) * dist_left
    delta_left[...,1]  = -delta_amp * torch.sin(t_smile) * dist_left
    delta_right[...,0] =  smile_strength * torch.sin(t_smile) * dist_right
    delta_right[...,1] = -delta_amp * torch.sin(t_smile) * dist_right

    # =========================
    # Respiration bouche amplifiée
    # =========================
    t_breath = torch.tensor(frame_counter / 12.0, device=device)
    breath_strength = 0.04  # plus visible
    breath_delta = breath_strength * torch.sin(t_breath)

    # =========================
    # Combinaison des deltas
    # =========================
    face_delta = torch.zeros((B,1,1,2), device=device)
    face_delta_expanded = face_delta.expand(B,H,W,2).clone()
    face_delta_expanded[...,0] += (delta_left[...,0] + delta_right[...,0]) * mask_mouth_exp[...,0]
    face_delta_expanded[...,1] += (delta_left[...,1] + delta_right[...,1] + breath_delta) * mask_mouth_exp[...,0]

    # =========================
    # Grid warp
    # =========================
    face_center = pose.get_point(0)  # nez
    face_center_px = face_center * torch.tensor([W-1,H-1], device=device)
    face_center_px = face_center_px.view(B,1,1,2)

    grid_mouth = grid.clone() - face_center_px
    grid_mouth = grid_mouth + face_delta_expanded
    grid_mouth = grid_mouth + face_center_px

    grid_mouth[...,0] = 2.0 * grid_mouth[...,0] / (W-1) - 1.0
    grid_mouth[...,1] = 2.0 * grid_mouth[...,1] / (H-1) - 1.0

    # =========================
    # Warp final
    # =========================
    latents_out = F.grid_sample(latents, grid_mouth, align_corners=True)

    if debug and debug_dir is not None:
        save_impact_map(latents_out, latents_in, debug_dir, frame_counter, prefix="mouth")
        print("[DEBUG] Mouth warp applied (visible)")

    return latents_out, face_delta_expanded, facial_points


def apply_mouth_warp(
    latents,
    pose,
    mask_mouth,
    grid,
    H,
    W,
    frame_counter,
    device=None,
    debug=False,
    debug_dir=None,
    smooth=0.85
):
    """
    Warp spécifique de la bouche :
    - Sourire dynamique
    - Respiration bouche
    - Micro-oscillations
    """
    if device is None:
        device = latents.device

    B, C, H, W = latents.shape
    latents_in = latents.clone()

    # =========================
    # Points faciaux
    # =========================
    facial_points = pose.estimate_facial_points_full(smooth=smooth)

    # =========================
    # Masque bouche
    # =========================
    #mouth_mask = pose.create_mouth_mask(H, W, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter)
    mouth_mask_exp = mask_mouth.permute(0, 2, 3, 1)  # (B,H,W,1) pour broadcast

    # =========================
    # Sourire (coins de la bouche)
    # =========================
    mouth_left  = facial_points['mouth_left']
    mouth_right = facial_points['mouth_right']

    # Conversion pixels
    mouth_left_px  = mouth_left  * torch.tensor([W-1,H-1], device=device)
    mouth_right_px = mouth_right * torch.tensor([W-1,H-1], device=device)

    # Grille pixel
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)

    # Distance normalisée aux coins
    dist_left  = torch.clamp(1.0 - torch.sqrt((x_grid - mouth_left_px[:,0,None,None])**2 +
                                              (y_grid - mouth_left_px[:,1,None,None])**2)/20.0, 0,1)
    dist_right = torch.clamp(1.0 - torch.sqrt((x_grid - mouth_right_px[:,0,None,None])**2 +
                                              (y_grid - mouth_right_px[:,1,None,None])**2)/20.0, 0,1)

    # Déplacement coins
    t_smile = torch.tensor(frame_counter / 8.0, device=device)
    smile_strength = 0.12
    delta_left  = torch.zeros((B,H,W,2), device=device)
    delta_right = torch.zeros((B,H,W,2), device=device)

    delta_left[...,0]  = -smile_strength * torch.sin(t_smile) * dist_left
    delta_left[...,1]  = -0.03 * torch.sin(t_smile) * dist_left
    delta_right[...,0] =  smile_strength * torch.sin(t_smile) * dist_right
    delta_right[...,1] = -0.03 * torch.sin(t_smile) * dist_right

    # =========================
    # Respiration bouche
    # =========================
    t_breath = torch.tensor(frame_counter / 12.0, device=device)
    breath_strength = 0.018
    breath_delta = breath_strength * torch.sin(t_breath)

    # =========================
    # Combinaison des deltas
    # =========================
    face_delta = torch.zeros((B,1,1,2), device=device)
    face_delta_expanded = face_delta.expand(B,H,W,2).clone()
    face_delta_expanded[...,0] += (delta_left[...,0] + delta_right[...,0]) * mouth_mask_exp[...,0]
    face_delta_expanded[...,1] += (delta_left[...,1] + delta_right[...,1] + breath_delta) * mouth_mask_exp[...,0]

    # =========================
    # Grid warp
    # =========================
    # Centre visage (pour référence, optionnel)
    face_center = pose.get_point(0)  # nez
    face_center_px = face_center * torch.tensor([W-1,H-1], device=device)
    face_center_px = face_center_px.view(B,1,1,2)

    grid_mouth = grid.clone() - face_center_px
    grid_mouth = grid_mouth + face_delta_expanded
    grid_mouth = grid_mouth + face_center_px

    # Normalisation [-1,1]
    grid_mouth[...,0] = 2.0 * grid_mouth[...,0] / (W-1) - 1.0
    grid_mouth[...,1] = 2.0 * grid_mouth[...,1] / (H-1) - 1.0

    # =========================
    # Warp final
    # =========================
    latents_out = F.grid_sample(latents, grid_mouth, align_corners=True)

    # =========================
    # Debug
    # =========================
    if debug and debug_dir is not None:
        save_impact_map(latents_out, latents_in, debug_dir, frame_counter, prefix="mouth")
        print("[DEBUG] Mouth warp applied")

    return latents_out, face_delta_expanded, facial_points
#-------------------------------------------------test -----------------------------------------
    """
    Ultra PRO 2.0 motion pipeline:
    - Global Pose + Stabilisation avancée par keypoints
    - Torso Warp + Breathing dynamique
    - Face Warp stateful + temporal smoothing
    - Mouth & Corner micro-expressions
    - Eyes micro-motion
    - Hair Motion Cycle with temporal buffer
    - Micro-boost per zone
    - Full debug and timings
    """
def apply_pose_driven_motion_ultra2(
    latents,
    keypoints,
    prev_keypoints=None,
    frame_counter=0,
    device="cuda",
    breathing=True,
    debug=False,
    debug_dir=None
):
    timings = {}
    B, C, H, W = latents.shape
    device = latents.device
    latents = latents.float()
    latents_in = latents.clone()

    # =========================
    # 🔹 Pose et deltas
    # =========================
    start = time.time()
    pose = Pose(keypoints.to(device))
    pose.compute_torso_delta(latent_h=H, latent_w=W)
    prev_pose = Pose(prev_keypoints.to(device)) if prev_keypoints is not None else None
    timings["Pose"] = time.time() - start

    # =========================
    # 🔹 Global compensation
    # =========================
    global_shift = torch.zeros((B,1,1,2), device=device)
    if prev_pose is not None:
        c1 = pose.get_center()[..., :2]
        c0 = prev_pose.get_center()[..., :2]
        delta = c1 - c0
        delta = torch.clamp(delta, -5.0, 5.0)
        global_shift = delta.view(B,1,1,2)

    # =========================
    # 🔹 Grid
    # =========================
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B,1,1,1)
    grid_base = grid + global_shift

    # =========================
    # 🔹 Masks
    # =========================
    mask_face  = torch.clamp(pose.create_face_mask(H,W, debug=debug, debug_dir=debug_dir),0,1).float()
    mask_mouth, _ = pose.create_mouth_mask(H,W, debug=debug, debug_dir=debug_dir)
    mask_mouth = torch.clamp(mask_mouth,0,1).float()
    mask_mouth_corners, _ = pose.create_mouth_corners_mask(H,W, debug=debug, debug_dir=debug_dir)
    mask_mouth_corners = torch.clamp(mask_mouth_corners,0,1).float()
    mask_torso = torch.clamp(pose.create_upper_body_mask(H,W, debug=debug, debug_dir=debug_dir),0,1).float()
    mask_hair = torch.clamp(pose.create_hair_mask(H,W, debug=debug, debug_dir=debug_dir),0,1).float()
    mask_left_eye = torch.clamp(pose.create_left_eye_mask(H,W, debug=debug, debug_dir=debug_dir),0,1).float()
    mask_right_eye = torch.clamp(pose.create_right_eye_mask(H,W, debug=debug, debug_dir=debug_dir),0,1).float()

    mask_torso_exp = mask_torso * (1.0 - mask_face)
    mask_hair_exp = mask_hair * (1.0 - mask_face)
    mask_face_exp = mask_face
    mask_mouth_exp = mask_mouth

    # =========================
    # 🔹 Global pose & stabilisation avancée
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_global, global_delta = apply_global_pose(latents, pose, prev_pose, H, W, device, debug=debug, debug_dir=debug_dir)
    if prev_pose is not None:
        key_joints = ['neck','left_shoulder','right_shoulder','left_hip','right_hip']
        for joint in key_joints:
            idx = pose.FACIAL_POINT_IDX[joint]
            diff = keypoints[:,idx,:2] - prev_keypoints[:,idx,:2]
            diff = torch.clamp(diff, -3.0, 3.0)
            latents_global += diff.mean() * 0.001
    latents = latents_global * (1.0 - mask_face_exp) + latents_before * mask_face_exp
    timings["GLOBAL"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="torso_global")
        print("[DEBUG] Ultra2 Global pose applied, delta mean:", global_delta.abs().mean())

    # =========================
    # 🔹 Torso + breathing dynamique
    # =========================
    latents_before = latents.clone()
    latents_torso, delta_px = apply_torso_warp(latents, pose, mask_torso, grid, H, W, device, debug=debug, debug_dir=debug_dir)
    t = torch.tensor(frame_counter/10.0, device=device)
    breath_strength = 0.2 + 0.1 * torch.sin(t)
    latents = latents_torso*(breath_strength*mask_torso_exp) + latents_before*(1.0 - breath_strength*mask_torso_exp)
    timings["TORSO+BREATH"] = time.time() - start

    # =========================
    # 🔹 Face + temporal smoothing
    # =========================
    if not hasattr(apply_pose_driven_motion_ultra2,"prev_face_grid"):
        apply_pose_driven_motion_ultra2.prev_face_grid = [None]*B
    start = time.time()
    latents, face_delta, facial_points = apply_face_warp(
        latents, pose, mask_face, grid_base, H, W, frame_counter,
        device=device, debug=debug, debug_dir=debug_dir, smooth=0.85,
        prev_grid=apply_pose_driven_motion_ultra2.prev_face_grid[0] if B==1 else None
    )
    apply_pose_driven_motion_ultra2.prev_face_grid[0] = grid_base.clone() if B==1 else None
    timings["FACE"] = time.time() - start

    # =========================
    # 🔹 Mouth & micro-expressions
    # =========================
    start = time.time()
    latents, mouth_delta, _ = apply_mouth_smil(
        latents, pose, mask_mouth, grid_base, H, W, frame_counter,
        device=device, debug=debug, debug_dir=debug_dir, smooth=0.85
    )

    # Broadcasting correct pour la bouche
    mask_mouth_corners_broadcast = mask_mouth_corners.repeat(1, C, 1, 1)
    latents += 0.002 * (mask_mouth_corners_broadcast * torch.sin(t*2.0))

    # Broadcasting correct pour les yeux
    mask_left_eye_broadcast  = mask_left_eye.repeat(1, C, 1, 1)
    mask_right_eye_broadcast = mask_right_eye.repeat(1, C, 1, 1)
    latents += 0.0015 * (mask_left_eye_broadcast  * torch.sin(t*3.0))
    latents += 0.0015 * (mask_right_eye_broadcast * torch.cos(t*3.0))

    timings["MOUTH+EYES"] = time.time() - start

    # =========================
    # 🔹 Eyes micro-motion
    # =========================
    # Correct broadcasting pour les yeux
    mask_left_eye_broadcast  = mask_left_eye.repeat(1, C, 1, 1)
    mask_right_eye_broadcast = mask_right_eye.repeat(1, C, 1, 1)

    latents += 0.0015 * (mask_left_eye_broadcast  * torch.sin(t*3.0))
    latents += 0.0015 * (mask_right_eye_broadcast * torch.cos(t*3.0))

    # =========================
    # 🔹 Hair motion cycle
    # =========================
    if not hasattr(apply_pose_driven_motion_ultra2,"prev_hair_fields"):
        apply_pose_driven_motion_ultra2.prev_hair_fields = [None]*B
    start = time.time()
    latents_before = latents.clone()
    latents_hair, hair_delta = apply_hair_motion_cycle(
        latents, mask_hair, grid, H, W, frame_counter, device, delta_px,
        prev_hair_field=apply_pose_driven_motion_ultra2.prev_hair_fields[0] if B==1 else None,
        debug=debug, debug_dir=debug_dir
    )
    latents = latents_hair * mask_hair_exp + latents_before * (1.0 - mask_hair_exp)
    apply_pose_driven_motion_ultra2.prev_hair_fields[0] = hair_delta
    timings["HAIR"] = time.time() - start

    # =========================
    # 🔹 Micro boost global
    # =========================
    masks = {
        "torso": (mask_torso_exp, 0.1, calibrate_amplitude(mask_torso_exp,0.002,0.004)),
        "hair": (mask_hair_exp,0.2,calibrate_amplitude(mask_hair_exp,0.003,0.0035)),
        "face": (mask_face_exp,0.3,calibrate_amplitude(mask_face_exp,0.002,0.006)),
        "mouth": (mask_mouth_exp,0.3,calibrate_amplitude(mask_mouth_exp,0.003,0.008)),
        "left_eye": (mask_left_eye,0.5,calibrate_amplitude(mask_left_eye,0.0015,0.004)),
        "right_eye": (mask_right_eye,0.6,calibrate_amplitude(mask_right_eye,0.0015,0.004)),
        "mouth_corners": (mask_mouth_corners,0.3,calibrate_amplitude(mask_mouth_corners,0.002,0.006))
    }
    start = time.time()
    latents = apply_micro_boost(latents, frame_counter, device, masks, keypoints, prev_keypoints)

    for key, (mask, speed, amplitude) in masks.items():
        # mask: [B, 1, H, W] ou [B, H, W]
        if mask.ndim == 5:
            mask = mask.squeeze(2)  # supprime la dimension singleton inutile
        mask_exp = mask.repeat(1, C, 1, 1)  # broadcast sur les canaux
        latents += amplitude * mask_exp * torch.sin(t * speed)

    latents = apply_micro_motion(latents, frame_counter, device, masks, randomize=True)
    timings["MICRO_BOOST"] = time.time() - start

    # =========================
    # 🔹 DEBUG FINAL
    # =========================
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="final")
        print("[DEBUG] Ultra2 Full motion pipeline applied")
        print("[DEBUG] Timings per step:", timings)

    return latents

    """
    Ultra PRO motion pipeline:
    - Global Pose + Stabilisation
    - Torso Warp + Breathing
    - Face Warp (stateful)
    - Mouth & Corners Warp
    - Hair Motion Cycle
    - Eyes micro-motion
    - Micro-boost per zone
    - Full debug and timing outputs
    """

def apply_pose_driven_motion(
    latents,
    keypoints,
    prev_keypoints=None,
    frame_counter=0,
    device="cuda",
    breathing=True,
    debug=False,
    debug_dir=None
):


    timings = {}
    B, C, H, W = latents.shape
    device = latents.device
    latents = latents.float()
    latents_in = latents.clone()

    # =========================
    # 🔹 Pose Processing
    # =========================
    start = time.time()
    pose = Pose(keypoints.to(device))
    pose.compute_torso_delta(latent_h=H, latent_w=W)

    prev_pose = Pose(prev_keypoints.to(device)) if prev_keypoints is not None else None
    timings["Pose"] = time.time() - start

    # =========================
    # 🔹 Global Compensation
    # =========================
    global_shift = None
    if prev_pose is not None:
        # Compute pixel centers for global motion
        c1 = pose.get_center()[..., :2]  # shape (B,2)
        c0 = prev_pose.get_center()[..., :2]
        global_shift = (c1 - c0).view(B, 1, 1, 2).to(device)
    else:
        global_shift = torch.zeros((B,1,1,2), device=device)

    # =========================
    # 🔹 Grid
    # =========================
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B,1,1,1)
    grid_base = grid + global_shift

    # =========================
    # 🔹 Masks
    # =========================
    mask_face  = torch.clamp(pose.create_face_mask(H,W, debug=debug, debug_dir=debug_dir), 0,1).float()
    mask_mouth, _ = pose.create_mouth_mask(H,W, debug=debug, debug_dir=debug_dir)
    mask_mouth = torch.clamp(mask_mouth,0,1).float()
    mask_mouth_corners, _ = pose.create_mouth_corners_mask(H,W, debug=debug, debug_dir=debug_dir)
    mask_mouth_corners = torch.clamp(mask_mouth_corners,0,1).float()
    mask_torso = torch.clamp(pose.create_upper_body_mask(H,W, debug=debug, debug_dir=debug_dir),0,1).float()
    mask_hair = torch.clamp(pose.create_hair_mask(H,W, debug=debug, debug_dir=debug_dir),0,1).float()
    mask_left_eye = torch.clamp(pose.create_left_eye_mask(H,W, debug=debug, debug_dir=debug_dir),0,1).float()
    mask_right_eye = torch.clamp(pose.create_right_eye_mask(H,W, debug=debug, debug_dir=debug_dir),0,1).float()

    mask_torso_exp = mask_torso * (1.0 - mask_face)
    mask_hair_exp = mask_hair * (1.0 - mask_face)
    mask_face_exp = mask_face
    mask_mouth_exp = mask_mouth

    # =========================
    # 🔹 GLOBAL POSE
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_global, global_delta = apply_global_pose(latents, pose, prev_pose, H, W, device, debug=debug, debug_dir=debug_dir)
    latents = latents_global * (1.0 - mask_face_exp) + latents_before * mask_face_exp
    timings["GLOBAL"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="torso_global")
        print("[DEBUG] Global pose applied, delta mean:", global_delta.abs().mean())

    # =========================
    # 🔹 TORSO
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_torso, delta_px = apply_torso_warp(latents, pose, mask_torso, grid, H, W, device, debug=debug, debug_dir=debug_dir)
    latents = latents_torso * mask_torso_exp + latents_before * (1.0 - mask_torso_exp)
    timings["TORSO"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="torso_warp")
        print("[DEBUG] Torso warp applied")

    # =========================
    # 🔹 FACE (stateful)
    # =========================
    start = time.time()
    latents, face_delta, facial_points = apply_face_warp(latents, pose, mask_face, grid_base, H, W, frame_counter, device=device, debug=debug, debug_dir=debug_dir, smooth=0.85)
    timings["face_warp"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="face_warp")
        print("[DEBUG] Face warp applied")

    # =========================
    # 🔹 MOUTH & CORNERS
    # =========================
    start = time.time()
    latents, mouth_delta, _ = apply_mouth_smil(latents, pose, mask_mouth, grid_base, H, W, frame_counter, device=device, debug=debug, debug_dir=debug_dir, smooth=0.85)
    timings["mouth_warp"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="mouth_warp")
        print("[DEBUG] Mouth warp applied")

    # =========================
    # 🔹 HAIR
    # =========================
    if not hasattr(apply_pose_driven_motion, "prev_hair_fields"):
        apply_pose_driven_motion.prev_hair_fields = [None]*B
    start = time.time()
    latents_before = latents.clone()
    latents_hair, hair_delta = apply_hair_motion_cycle(latents, mask_hair, grid, H, W, frame_counter, device, delta_px, prev_hair_field=apply_pose_driven_motion.prev_hair_fields[0] if B==1 else None, debug=debug, debug_dir=debug_dir)
    latents = latents_hair * mask_hair_exp + latents_before * (1.0 - mask_hair_exp)
    apply_pose_driven_motion.prev_hair_fields[0] = hair_delta
    timings["HAIR"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="hair")
        print("[DEBUG] Hair motion applied")

    # =========================
    # 🔹 BREATHING
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_breath = apply_breathing_simple(latents, mask_torso_exp, frame_counter, breathing, debug=debug, debug_dir=debug_dir)
    t = torch.tensor(frame_counter/10.0, device=latents.device)
    breath_strength = 0.2 + 0.1 * torch.sin(t)
    latents = latents_before*(1.0-breath_strength*mask_torso_exp) + latents_breath*(breath_strength*mask_torso_exp)
    timings["breathing"] = time.time() - start
    if debug:
        print("[DEBUG] Breathing applied")

    # =========================
    # 🔹 STABILISATION
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_stab = stabilize_latents_motion(latents)
    latents = latents_stab*(1.0 - mask_face_exp) + latents_before*mask_face_exp
    timings["stabilisation"] = time.time() - start
    if debug:
        print("[DEBUG] Stabilization applied")

    # =========================
    # 🔹 MICRO BOOST
    # =========================
    masks = {
        "torso": (mask_torso_exp, 0.1, calibrate_amplitude(mask_torso_exp, 0.002,0.004)),
        "hair": (mask_hair_exp, 0.2, calibrate_amplitude(mask_hair_exp, 0.003,0.0035)),
        "face": (mask_face_exp, 0.3, calibrate_amplitude(mask_face_exp,0.002,0.006)),
        "mouth": (mask_mouth_exp, 0.3, calibrate_amplitude(mask_mouth_exp,0.003,0.008)),
        "left_eye": (mask_left_eye,0.5, calibrate_amplitude(mask_left_eye,0.0015,0.004)),
        "right_eye": (mask_right_eye,0.6, calibrate_amplitude(mask_right_eye,0.0015,0.004)),
        "mouth_corners": (mask_mouth_corners,0.3, calibrate_amplitude(mask_mouth_corners,0.002,0.006))
    }
    start = time.time()
    latents = apply_micro_boost(latents, frame_counter, device, masks, keypoints, prev_keypoints)
    latents = apply_micro_motion(latents, frame_counter, device, masks, randomize=True)
    timings["micro_boost"] = time.time() - start

    # =========================
    # 🔹 FINAL DEBUG
    # =========================
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="final")
        print("[DEBUG] Full motion pipeline applied")
        print("[DEBUG] Timings per step:", timings)

    return latents

def apply_pose_driven_motion_stable(
    latents,
    keypoints,
    prev_keypoints=None,
    frame_counter=0,
    device="cuda",
    breathing=True,
    debug=False,
    debug_dir=None
):
    """
    Pipeline motion PRO (stable + vivant + isolé) :
    - Global Pose
    - Torso Warp
    - Face Warp (stateful)
    - Hair Motion (alt normal/extreme)
    - Breathing (torso only)
    - Stabilisation (face protected)
    - Micro-boost par zone pour éviter le rendu statique
    """
    # dictionnaire pour stocker les temps
    timings = {}
    # =========================
    # 🔹 SETUP
    # =========================
    B, C, H, W = latents.shape
    device = latents.device
    latents = latents.float()
    latents_in = latents.clone()

    # =========================
    # 🔹 POSE (NOW CONSISTENT)
    # =========================
    start = time.time()
    pose = Pose(keypoints.to(device))
    pose.compute_torso_delta(latent_h=H, latent_w=W)

    prev_pose = Pose(prev_keypoints.to(device)) if prev_keypoints is not None else None
    timings["Pose"] = time.time() - start

    # =========================
    # 🔥 GLOBAL COMPENSATION
    # =========================


    global_shift = None

    if prev_pose is not None:
        # centers en pixels directement (IMPORTANT)
        c1 = pose.get_center()
        c0 = prev_pose.get_center()

        c1 = pose.get_center()[..., :2]
        c0 = prev_pose.get_center()[..., :2]

        global_shift = (c1 - c0).to(device)
        global_shift = global_shift.view(B, 1, 1, 2)

    # =========================
    # 🔹 Grid
    # =========================
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B, 1, 1, 1)

    # =========================
    # 🔹 Masks (CLAMP SAFE)
    # =========================
    mask_face  = torch.clamp(pose.create_face_mask(H, W, debug=debug, debug_dir=debug_dir), 0, 1).float()
    mask_mouth, mouth_points = pose.create_mouth_mask(H, W, debug=debug, debug_dir=debug_dir)
    mask_mouth = torch.clamp(mask_mouth, 0, 1).float()
    mask_mouth_corners, corners_points_batch = pose.create_mouth_corners_mask(H, W, debug=debug, debug_dir=debug_dir)
    mask_mouth_corners = torch.clamp(mask_mouth_corners, 0, 1).float()

    mask_torso = torch.clamp(pose.create_upper_body_mask(H, W, debug=debug, debug_dir=debug_dir), 0, 1).float()
    mask_hair  = torch.clamp(pose.create_hair_mask(H, W, debug=debug, debug_dir=debug_dir), 0, 1).float()
    mask_left_eye = pose.create_left_eye_mask(H, W, debug=debug, debug_dir=debug_dir)
    mask_left_eye = torch.clamp(mask_left_eye, 0, 1).float()
    mask_right_eye = pose.create_right_eye_mask(H, W, debug=debug, debug_dir=debug_dir)
    mask_right_eye = torch.clamp(mask_right_eye, 0, 1).float()


    mask_face_exp  = mask_face
    mask_mouth_exp  = mask_mouth
    mask_torso_exp = mask_torso * (1.0 - mask_face_exp)
    mask_hair_exp  = mask_hair  * (1.0 - mask_face_exp)
    mask_right_eye_exp = mask_right_eye
    mask_left_eye_exp = mask_left_eye

    # =========================
    # 🔹 GLOBAL POSE
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_global, global_delta = apply_global_pose(
        latents=latents, pose=pose, prev_pose=prev_pose, H=H, W=W, device=device,
        debug=debug,
        debug_dir=debug_dir
    )
    if debug:
        print("GLOBAL delta mean:", global_delta.abs().mean())



    latents = latents_global * (1.0 - mask_face_exp) + latents_before * mask_face_exp
    timings["GLOBAL"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="torso_global")
        print("[DEBUG] Global pose applied")

    # =========================
    # 🔹 TORSO
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_torso, delta_px = apply_torso_warp(
        latents=latents, pose=pose, mask_torso=mask_torso, grid=grid, H=H, W=W, device=device,
        debug=debug,
        debug_dir=debug_dir
    )
    latents = latents_torso * mask_torso_exp + latents_before * (1.0 - mask_torso_exp)
    timings["TORSO"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="torso_warp")
        print("[DEBUG] Torso warp applied")

    # ========================
    # FIX FACE + BOUCHE
    # ========================
    grid_base = grid

    global_shift = global_shift if global_shift is not None else torch.zeros((B,1,1,2), device=device)
    grid_base = grid_base + global_shift
    # =========================
    # 🔹 FACE (STATEFUL)
    # =========================
    face_grid = grid_base
    start = time.time()
    latents, face_delta, facial_points = apply_face_warp(
        latents=latents, pose=pose, mask_face=mask_face, grid=face_grid, H=H, W=W, frame_counter=frame_counter, device=device, debug=debug, debug_dir=debug_dir,
        smooth=0.85
    )
    timings["face_warp"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="face_warp")
        print("[DEBUG] Face warp applied")
    # =========================
    # 🔹 BOUCHE (STATEFUL)
    # =========================
    mouth_grid = grid_base
    start = time.time()
    latents, mouth_delta, _ = apply_mouth_smil(
        latents=latents, pose=pose, mask_mouth=mask_mouth, grid=mouth_grid, H=H, W=W, frame_counter=frame_counter, device=device, debug=debug, debug_dir=debug_dir,
        smooth=0.85
    )
    timings["mouth_warp"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="mouth_warp")
        print("[DEBUG] Mouth warp applied")


    # =========================
    # 🔹 HAIR (ALTERNANCE CINÉMA)
    # =========================
    # Définir un dictionnaire global ou un buffer par batch
    if not hasattr(apply_pose_driven_motion, "prev_hair_fields"):
        apply_pose_driven_motion.prev_hair_fields = [None] * B

    # =========================
    # 🔹 HAIR (ALTERNANCE CINÉMA)
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_hair, hair_delta = apply_hair_motion_cycle(
        latents=latents, mask_hair=mask_hair, grid=grid, H=H, W=W, frame_counter=frame_counter, device=device, delta_px=delta_px,
        prev_hair_field=apply_pose_driven_motion.prev_hair_fields[0] if B==1 else None,
        debug=debug,
        debug_dir=debug_dir
    )
    latents = latents_hair * mask_hair_exp + latents_before * (1.0 - mask_hair_exp)
    # Stocker pour la prochaine frame
    apply_pose_driven_motion.prev_hair_fields[0] = hair_delta
    timings["HAIR"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="hair")
        print("[DEBUG] Hair motion applied")

    # =========================
    # 🔹 BREATHING (TORSO ONLY)
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_breath = apply_breathing_simple( latents, mask_torso_exp, frame_counter, breathing, debug=debug, debug_dir=debug_dir )
    t = torch.tensor(frame_counter / 10.0, device=latents.device)
    breath_strength = 0.2 + 0.1 * torch.sin(t)
    latents = (
        latents_before * (1.0 - breath_strength * mask_torso_exp)
        + latents_breath * (breath_strength * mask_torso_exp)
    )
    timings["breathing"] = time.time() - start
    if debug:
        print("[DEBUG] Breathing applied")

    # =========================
    # 🔹 STABILISATION
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_stab = stabilize_latents_motion(latents)
    latents = latents_stab * (1.0 - mask_face_exp) + latents_before * mask_face_exp
    timings["stabilisation"] = time.time() - start

    if debug:
        print("[DEBUG] Stabilization applied")

    # =========================
    # 🔹 MICRO BOOST GLOBAL PAR ZONE
    # =========================
    masks = {
        "torso": (mask_torso_exp, 0.1, calibrate_amplitude(mask_torso_exp, base_amp=0.002, max_amp=0.004)),
        "hair":  (mask_hair_exp,  0.2, calibrate_amplitude(mask_hair_exp, base_amp=0.003, max_amp=0.0035)),
        "face":  (mask_face_exp,  0.3, calibrate_amplitude(mask_face_exp, base_amp=0.002, max_amp=0.006)),
        "mouth":  (mask_mouth_exp,  0.3, calibrate_amplitude(mask_mouth_exp, base_amp=0.003, max_amp=0.008)),
        # Yeux (clignements / micro-mouvements)
        "left_eye":  (mask_left_eye,  0.5, calibrate_amplitude(mask_left_eye, 0.0015, 0.004)),
        "right_eye": (mask_right_eye, 0.6, calibrate_amplitude(mask_right_eye, 0.0015, 0.004)),
        # Coins de bouche pour sourire subtil
        "mouth_corners": (mask_mouth_corners, 0.3, calibrate_amplitude(mask_mouth_corners, 0.002, 0.006)),
    }

    start = time.time()
    latents = apply_micro_boost(latents, frame_counter, device, masks, keypoints, prev_keypoints)
    latents = apply_micro_motion(latents, frame_counter, device, masks, randomize = True)

    timings["micro_boost"] = time.time() - start

    # =========================
    # 🔹 DEBUG FINAL
    # =========================
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="final")
        print("[DEBUG] Full motion pipeline applied")
        print("[DEBUG] Timings per step:", timings)

    return latents

#-------Gestion de l'animation -----------------------------------------------------------------------------------
def update_pose_sequence_from_keypoints_batch(
    keypoints_tensor,
    prev_keypoints=None,
    frame_idx=0,
    alpha=0.7,        # lissage temporel plus fort
    add_motion=True,
    debug=False
):

    kp = keypoints_tensor.clone()
    B, N, _ = kp.shape
    device = kp.device

    # =========================
    # 🔹 1. SMOOTH TEMPOREL
    # =========================
    if prev_keypoints is not None:
        kp = alpha * kp + (1 - alpha) * prev_keypoints

    # =========================
    # 🔹 2. MOTION PROCÉDURAL
    # =========================
    if add_motion:
        t = frame_idx * 0.1

        # Respiration (vertical torso + épaules)
        breath = 0.009 * math.sin(t * 0.15)
        kp[:, 2, 1] += breath
        kp[:, 5, 1] += breath

        # Balancement gauche/droite
        sway = 0.010 * math.sin(t * 0.08)
        #kp[:, :, 0] += sway
        torso_ids = [0,1,2,5,8,11]
        kp[:, torso_ids, 0] += sway

        # Head motion
        head_idx = 0
        kp[:, head_idx, 0] += 0.006 * math.sin(t * 0.2)
        kp[:, head_idx, 1] += 0.006 * math.cos(t * 0.18)

        # Drift lent
        drift_x = 0.002 * math.sin(t * 0.03)
        drift_y = 0.002 * math.cos(t * 0.025)
        #kp[:, :, 0] += drift_x
        #kp[:, :, 1] += drift_y
        valid_ids = [0,1,2,5,8,11,14,15,16,17,18,19,20,21,22,23,24]

        kp[:, valid_ids, 0] += drift_x
        kp[:, valid_ids, 1] += drift_y

        # Micro noise (anti-freeze)
        noise = torch.randn_like(kp[..., :2]) * 0.0015
        kp[..., :2] += noise

    # =========================
    # 🔹 3. STABILISATION
    # =========================
    kp[..., :2] = torch.clamp(kp[..., :2], -1.2, 1.2)

    # =========================
    # 🔹 4. DEBUG
    # =========================
    if debug:
        motion_strength = (kp - keypoints_tensor).abs().mean()
        print(f"[DEBUG] Keypoint motion strength: {motion_strength.item():.6f}")

    return kp


