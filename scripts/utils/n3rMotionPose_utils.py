#********************************************
# n3rOpenPose_utils.py
#********************************************
import torch
import time
from diffusers import ControlNetModel
import math
import torch.nn.functional as F
from .n3rcoords import pair, safe_xy, safe_update, norm, build_upper_body_inputs, animate_upper_body
from .n3rControlNet import create_canny_control, control_to_latent, match_latent_size
from .tools_utils import ensure_4_channels, print_generation_params, sanitize_latents
from .n3rMotionPose_tools import gaussian_blur_tensor, debug_draw_openpose_skeleton, rotate_mask_around_torso_simple, rotate_mask_around_visage, save_impact_map, apply_breathing_xy, smooth_noise, feather_dynamic_vectorized, compute_delta, stabilize_latents_motion, save_debug_pose_image_with_skeleton, apply_hair_motion_cycle, apply_breathing_real, apply_breathing_soft, feather_inside_strict2, feather_outside_only_alpha2, apply_micro_motion, apply_micro_boost

from .n3rMotionPoseClass import Pose
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torchvision
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

        [308/896, 1129/1280, 1.0], # 8 right_hip / hanche droite 🦿📍 Hips detected: left=(564, 1102), right=(308, 1129)
        [0.0, 0.0, 0.0],           # 9 right_knee (absent)
        [0.0, 0.0, 0.0],           # 10 right_ankle (absent)

        [564/896, 1102/1280, 1.0], # 11 left_hip / hanche gauche 🦿📍 Hips detected: left=(564, 1102), right=(308, 1129)
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
    strength=0.35,
    debug=False,
    debug_dir=None,
    frame_counter=None
):
    """
    Version factorisée PRO :
    - Mapping centralisé
    - Safe update générique
    """

    B, C, H, W = pose_full_image.shape

    # =========================
    # 🔹 INIT KEYPOINTS
    # =========================
    keypoints_np = current_keypoints.clone().cpu().numpy()[0]

    # =========================
    # 🔹 NORMALISATION INPUTS
    # =========================
    left_shoulder, right_shoulder  = pair(shoulders_coords, debug=debug)  # épaule_droite, épaule_gauche
    left_clavicle, right_clavicle  = pair(clavicules_coords, debug=debug) # clavicule_droite, clavicule_gauche
    left_elbow, right_elbow  = pair(elbow_coords, debug=debug) # coude_droit, coude_gauche
    left_wrist, right_wrist = pair(wrists_coords, debug=debug) # poignet_gauche, poignet_droit
    left_hip, right_hip = pair(hips_coords, debug=debug) # hanche_gauche, hanche_droite
    left_eye, right_eye = pair(eye_coords, debug=debug) # œil_gauche, œil_droit
    left_ear, right_ear = pair(ear_coords, debug=debug) # oreille_gauche, oreille_droite

    # Neck dict safe
    if isinstance(neck_coords, dict):
        neck_map = {
            "neck": neck_coords.get("center"),
            "chin": neck_coords.get("chin"),
            "left_side_neck": neck_coords.get("left"),
            "right_side_neck": neck_coords.get("right"),
            "anchor": neck_coords.get("anchor"),
        }
    else:
        neck_map = {"neck": neck_coords}

    # =========================
    # 🔹 MAPPING CENTRALISÉ 🔥
    # =========================

    updates = {
        0: ("nose", nose_coords),
        1: ("neck", norm(neck_map.get("neck"))),

        2: ("right_shoulder", right_shoulder),
        3: ("right_elbow", right_elbow),
        4: ("right_wrist", right_wrist),

        5: ("left_shoulder", left_shoulder),
        6: ("left_elbow", left_elbow),
        7: ("left_wrist", left_wrist),

        8: ("right_hip", right_hip),
        11: ("left_hip", left_hip),

        14: ("right_eye", right_eye),
        15: ("left_eye", left_eye),
        16: ("right_ear", right_ear),
        17: ("left_ear", left_ear),

        18: ("mouth", mouth_coords),

        19: ("right_clavicle", right_clavicle),
        20: ("left_clavicle", left_clavicle),

        21: ("chin", neck_map.get("chin")),
        22: ("left_side_neck", neck_map.get("left_side_neck")),
        23: ("right_side_neck", neck_map.get("right_side_neck")),
        24: ("anchor", neck_map.get("anchor")),
    }

    # =========================
    # 🔹 APPLY UPDATES
    # =========================
    for idx, (label, coord) in updates.items():
        safe_update(idx, coord, keypoints_np, W, H, label, debug=debug)


    # ========= Valeur initial================
    # 🔹 TO TENSOR
    # =========================
    keypoints_np = np.expand_dims(keypoints_np, axis=0)
    keypoints_np = np.repeat(keypoints_np, B, axis=0)
    keypoints_tensor = torch.from_numpy(keypoints_np).to(device)

    # =========================
    # 🔹 DEBUG
    # =========================
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

    # 1️⃣ Resize pose
    pose_resized = resize_pose(pose_tile, H_latent, W_latent)
    # 2️⃣ Inputs
    latent_fp32, pose_fp32, pos_embeds, neg_embeds = prepare_inputs(
        latent_tile, pose_resized, cf_embeds, device
    )
    # 3️⃣ Timestep
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

    delta = compute_delta( latents_out, latent_fp32, controlnet_scale, importance )

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

def convert_json_to_pose_sequence(anim_data, H=512, W=512, device="cuda", dtype=torch.float16, debug=False, output_dir=None):
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

def apply_torso_warp(
    latents,
    pose,
    mask_torso,
    grid,
    H,
    W,
    device,
    prev_delta_px=None,
    strength=0.3,   # 🔥 NEW 0.3–1.2
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
    print("delta_px max:", delta_px.abs().max().item())

    # -------------------- Lissage temporel --------------------
    if prev_delta_px is not None:
        alpha = 0.7
        delta_px = alpha * delta_px + (1 - alpha) * prev_delta_px

    # -------------------- strength control --------------------
    #delta_px = delta_px * strength
    delta_px = torch.tanh(delta_px) * 5.0 * strength

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
    falloff = torch.exp(-distance / (0.35 * W))

    # -------------------- Warp torse --------------------
    warp = delta_px * strength
    # safety clamp (IMPORTANT)
    warp = torch.tanh(warp / 5.0) * 5.0
    grid_torso = grid + strength * delta_px * mask_expand * falloff
    # -------------------- Normalisation --------------------
    grid_torso[...,0] = 2.0 * grid_torso[...,0] / (W-1) - 1.0
    grid_torso[...,1] = 2.0 * grid_torso[...,1] / (H-1) - 1.0

    # -------------------- Sampling --------------------
    latents_out = F.grid_sample(latents, grid_torso, align_corners=True)

    if debug:
        print("[DEBUG][TORSO] strength:", strength)
        print("[DEBUG][TORSO] delta_px mean:", delta_px.abs().mean().item())
        print("[DEBUG][TORSO] delta_px max:", delta_px.abs().max().item())
        print("[DEBUG][TORSO] falloff mean:", falloff.mean().item())
        print("[DEBUG][TORSO] mask mean:", mask_expand.mean().item())

    return latents_out, delta_px

# version corriger
def apply_global_pose(
    latents,
    pose,
    prev_pose=None,
    H=None,
    W=None,
    device="cuda",
    strength=0.4,
    clamp=True,
    debug=True,
    debug_dir=None
):

    B, C, H_lat, W_lat = latents.shape
    device = latents.device

    t0 = time.time()

    # =========================================================
    # 🔹 Identity fallback
    # =========================================================
    if prev_pose is None:
        yy, xx = torch.meshgrid(
            torch.arange(H_lat, device=device),
            torch.arange(W_lat, device=device),
            indexing="ij"
        )
        grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B, 1, 1, 1)
        return latents, torch.zeros((B,1,1,2), device=device), grid, None

    # =========================================================
    # 🔹 Key joints
    # =========================================================
    joints = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    idx = [pose.FACIAL_POINT_IDX[j] for j in joints]

    pts = torch.stack([pose.get_point(i) for i in idx], dim=1)        # [B,4,2]
    prev_pts = torch.stack([prev_pose.get_point(i) for i in idx], 1)

    # =========================================================
    # 🔹 Global center motion (translation)
    # =========================================================
    center = pts.mean(1)
    prev_center = prev_pts.mean(1)
    delta = center - prev_center

    # =========================================================
    # 🔹 Spine axis (NEW)
    # =========================================================
    spine_top = (pts[:,0] + pts[:,1]) * 0.5   # shoulders
    spine_bottom = (pts[:,2] + pts[:,3]) * 0.5  # hips

    spine_vec = spine_top - spine_bottom
    prev_spine_vec = (prev_pts[:,0]+prev_pts[:,1])*0.5 - (prev_pts[:,2]+prev_pts[:,3])*0.5

    spine_len = torch.norm(spine_vec, dim=-1, keepdim=True).clamp(1e-6)
    prev_len = torch.norm(prev_spine_vec, dim=-1, keepdim=True).clamp(1e-6)

    spine_dir = spine_vec / spine_len
    prev_dir = prev_spine_vec / prev_len

    # angle via dot/cross stable
    dot = (spine_dir * prev_dir).sum(-1, keepdim=True).clamp(-1, 1)
    angle = torch.acos(dot)

    # signed approx
    cross = spine_dir[...,0]*prev_dir[...,1] - spine_dir[...,1]*prev_dir[...,0]
    angle = angle * torch.sign(cross).unsqueeze(-1)

    # =========================================================
    # 🔹 Temporal smoothing (IMPORTANT)
    # =========================================================
    delta = 0.8 * delta + 0.2 * getattr(pose, "_vel", torch.zeros_like(delta))
    pose._vel = delta.detach()

    # =========================================================
    # 🔹 Pixel conversion
    # =========================================================
    delta_px = torch.zeros_like(delta)
    delta_px[...,0] = delta[...,0] * W_lat
    delta_px[...,1] = delta[...,1] * H_lat

    if clamp:
        delta_px = torch.tanh(delta_px / 4.0) * 4.0

    delta_px = delta_px.view(B,1,1,2)

    # =========================================================
    # 🔹 Base grid
    # =========================================================
    yy, xx = torch.meshgrid(
        torch.arange(H_lat, device=device),
        torch.arange(W_lat, device=device),
        indexing="ij"
    )

    grid = torch.stack((xx, yy), -1).float().unsqueeze(0).repeat(B,1,1,1)

    # =========================================================
    # 🔹 CENTER-BASED ROTATION (stable)
    # =========================================================
    center_px = center.clone()
    center_px[...,0] *= W_lat
    center_px[...,1] *= H_lat
    center_px = center_px.view(B,1,1,2)

    theta = angle * strength * 0.6

    cos_t = torch.cos(theta).view(B,1,1,1)
    sin_t = torch.sin(theta).view(B,1,1,1)

    x = grid[...,0:1] - center_px[...,0:1]
    y = grid[...,1:2] - center_px[...,1:2]

    rot_x = x*cos_t - y*sin_t
    rot_y = x*sin_t + y*cos_t

    grid = torch.cat([rot_x + center_px[...,0:1],
                      rot_y + center_px[...,1:2]], dim=-1)

    # =========================================================
    # 🔹 SPINE AXIS DEFORMATION (NEW CORE FEATURE)
    # =========================================================
    spine_mid = (spine_top + spine_bottom) * 0.5
    spine_mid_px = spine_mid.clone()
    spine_mid_px[...,0] *= W_lat
    spine_mid_px[...,1] *= H_lat
    spine_mid_px = spine_mid_px.view(B,1,1,2)

    spine_dir_px = spine_dir.clone()
    spine_dir_px = spine_dir_px.view(B,1,1,2)

    # projection of pixel onto spine axis
    rel = grid - spine_mid_px
    proj = (rel * spine_dir_px).sum(-1, keepdim=True) * spine_dir_px

    ortho = rel - proj

    # squash/stretch VERY subtle
    stretch = 1.0 + torch.tanh(delta_px.norm(dim=-1, keepdim=True)) * 0.03
    squash = 1.0 - torch.tanh(delta_px.norm(dim=-1, keepdim=True)) * 0.02

    grid = spine_mid_px + proj * stretch + ortho * squash

    # =========================================================
    # 🔹 FINAL MOTION
    # =========================================================
    global_gain = 1.6

    grid = grid + delta_px * strength * global_gain

    # =========================================================
    # 🔹 Normalize for grid_sample
    # =========================================================
    grid_norm = grid.clone()
    grid_norm[...,0] = 2.0 * grid_norm[...,0] / (W_lat - 1) - 1.0
    grid_norm[...,1] = 2.0 * grid_norm[...,1] / (H_lat - 1) - 1.0

    # =========================================================
    # 🔹 Apply warp
    # =========================================================
    latents_out = F.grid_sample(latents, grid_norm, align_corners=True)

    # =========================================================
    # 🔹 DEBUG
    # =========================================================
    if debug:
        print("\n[DEBUG][GLOBAL] =====================")
        print(f"Time: {time.time()-t0:.4f}s")
        print(f"Strength: {strength}")
        print(f"Angle spine: {angle.mean().item():.5f}")
        print(f"Delta px mean: {delta_px.abs().mean().item():.5f}")

        for i, j in enumerate(joints):
            mv = (pts[:,i]-prev_pts[:,i]) * torch.tensor([W_lat, H_lat], device=device)
            print(f"{j}: {mv.squeeze().tolist()}")

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            img = latents_out[0].detach().float().cpu()
            img = img[:3] if img.shape[0] > 3 else img

            torchvision.utils.save_image(
                (img+1)/2,
                os.path.join(debug_dir, "global_pose_debug.png")
            )

    return latents_out, delta_px, grid, grid_norm


def apply_global_pose_check(
    latents,
    pose,
    prev_pose=None,
    H=None,
    W=None,
    device="cuda",
    strength=0.4,
    clamp=True,
    debug=True,
    debug_dir=None
):
    import time
    import os
    import torch
    import torch.nn.functional as F
    import torchvision

    B = latents.shape[0]
    device = latents.device

    t0 = time.time()

    # =========================================================
    # 🔹 Identity fallback
    # =========================================================
    if prev_pose is None:
        if debug:
            print("[DEBUG][GLOBAL] No prev_pose → identity warp")

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B, 1, 1, 1)

        return latents, torch.zeros((B, 1, 1, 2), device=device), grid, None

    # =========================================================
    # 🔹 Key torso joints
    # =========================================================
    key_joints = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    idx_list = [pose.FACIAL_POINT_IDX[j] for j in key_joints]

    pts = torch.stack([pose.get_point(i) for i in idx_list], dim=1)
    prev_pts = torch.stack([prev_pose.get_point(i) for i in idx_list], dim=1)

    # =========================================================
    # 🔹 Center motion (translation)
    # =========================================================
    center = pts.mean(dim=1)
    prev_center = prev_pts.mean(dim=1)

    delta = center - prev_center

    # =========================================================
    # 🔹 Shoulder orientation (rotation estimate)
    # =========================================================
    cur_vec = pts[:, 1] - pts[:, 0]
    prev_vec = prev_pts[:, 1] - prev_pts[:, 0]

    cur_angle = torch.atan2(cur_vec[:, 1], cur_vec[:, 0])
    prev_angle = torch.atan2(prev_vec[:, 1], prev_vec[:, 0])

    angle_delta = (cur_angle - prev_angle).unsqueeze(-1)

    # =========================================================
    # 🔹 Temporal smoothing (inertia model)
    # =========================================================
    if not hasattr(pose, "_global_velocity"):
        pose._global_velocity = torch.zeros_like(delta)

    pose._global_velocity = 0.85 * pose._global_velocity + 0.15 * delta
    delta = pose._global_velocity

    # =========================================================
    # 🔹 Pixel conversion
    # =========================================================
    delta_px = delta.clone()
    delta_px[..., 0] *= W
    delta_px[..., 1] *= H

    # clamp soft
    if clamp:
        delta_px = torch.tanh(delta_px / 5.0) * 5.0

    delta_px = delta_px.view(B, 1, 1, 2)

    # =========================================================
    # 🔹 Scale adaptatif (distance-aware)
    # =========================================================
    scale_ref = torch.norm(pts[:, 0] - pts[:, 1], dim=-1, keepdim=True)
    scale_factor = torch.clamp(scale_ref, 0.05, 0.5).view(B, 1, 1, 1)

    # =========================================================
    # 🔹 Base grid
    # =========================================================
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    grid_raw = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B, 1, 1, 1)

    # =========================================================
    # 🔹 Optional rotation warp (rigid-body effect)
    # =========================================================
    center_px = center.clone()
    center_px[..., 0] *= W
    center_px[..., 1] *= H
    center_px = center_px.view(B, 1, 1, 2)

    theta = angle_delta * strength * 0.5

    cos_t = torch.cos(theta).view(B, 1, 1, 1)
    sin_t = torch.sin(theta).view(B, 1, 1, 1)

    x = grid_raw[..., 0:1] - center_px[..., 0:1]
    y = grid_raw[..., 1:2] - center_px[..., 1:2]

    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t

    grid_raw = torch.cat(
        [x_rot + center_px[..., 0:1],
         y_rot + center_px[..., 1:2]],
        dim=-1
    )

    # =========================================================
    # 🔹 Final warp (NO MAGIC CONSTANTS)
    # =========================================================
    GLOBAL_GAIN = 1.8

    grid_warp = grid_raw + delta_px * strength * GLOBAL_GAIN * scale_factor

    # =========================================================
    # 🔹 Normalize for grid_sample
    # =========================================================
    grid_norm = grid_warp.clone()
    grid_norm[..., 0] = 2.0 * grid_norm[..., 0] / (W - 1) - 1.0
    grid_norm[..., 1] = 2.0 * grid_norm[..., 1] / (H - 1) - 1.0

    # =========================================================
    # 🔹 Apply warp
    # =========================================================
    latents_out = F.grid_sample(latents, grid_norm, align_corners=True)

    # =========================================================
    # 🔹 DEBUG
    # =========================================================
    if debug:
        print("\n[DEBUG][GLOBAL] ===============================")
        print(f"[DEBUG][GLOBAL] Time: {time.time() - t0:.5f}s")
        print(f"[DEBUG][GLOBAL] Strength: {strength}")

        for i, j in enumerate(key_joints):
            move_px = (pts[:, i] - prev_pts[:, i]) * torch.tensor([W, H], device=device)
            print(f"[DEBUG][GLOBAL] {j} movement px: {move_px.squeeze().tolist()}")

        print(f"[DEBUG][GLOBAL] Mean delta_px: {delta_px.abs().mean().item():.4f}")
        print(f"[DEBUG][GLOBAL] Angle delta: {angle_delta.mean().item():.4f}")

        print(f"[DEBUG][GLOBAL] grid_raw min/max: {grid_raw.min().item():.2f} / {grid_raw.max().item():.2f}")
        print(f"[DEBUG][GLOBAL] grid_warp min/max: {grid_warp.min().item():.2f} / {grid_warp.max().item():.2f}")

        # optional debug image
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)

            img = latents_out[0].detach().float().cpu()
            if img.shape[0] > 3:
                img = img[:3]

            torchvision.utils.save_image(
                (img + 1.0) / 2.0,
                os.path.join(debug_dir, "global_pose_debug.png")
            )

            print(f"[DEBUG][GLOBAL] Saved: {debug_dir}/global_pose_debug.png")

    return latents_out, delta_px, grid_raw, grid_norm

def apply_global_pose_warp(
    latents,
    pose,
    prev_pose=None,
    H=None,
    W=None,
    device="cuda",
    strength=0.4,
    clamp=True,
    debug=True,
    debug_dir=None
):

    B = latents.shape[0]
    device = latents.device

    t0 = time.time()

    # =========================================================
    # 🔹 No previous pose → no motion
    # =========================================================
    if prev_pose is None:
        if debug:
            print("[DEBUG][GLOBAL] No prev_pose → identity warp")

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B,1,1,1)

        return latents, torch.zeros((B,1,1,2), device=device), grid, None

    # =========================================================
    # 🔹 Torso joints
    # =========================================================
    key_joints = ["left_shoulder","right_shoulder","left_hip","right_hip"]
    idx_list = [pose.FACIAL_POINT_IDX[j] for j in key_joints]

    pts = torch.stack([pose.get_point(i) for i in idx_list], dim=1)
    prev_pts = torch.stack([prev_pose.get_point(i) for i in idx_list], dim=1)

    center = pts.mean(dim=1)
    prev_center = prev_pts.mean(dim=1)

    delta = center - prev_center

    # pixels
    delta_px = delta.clone()
    delta_px[..., 0] *= W
    delta_px[..., 1] *= H

    if clamp:
        delta_px = torch.tanh(delta_px / 5.0) * 5.0

    delta_px = delta_px.view(B,1,1,2)

    # =========================================================
    # 🔹 Grid (RAW)
    # =========================================================
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    grid_raw = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B,1,1,1)

    # =========================================================
    # 🔹 Warp
    # =========================================================
    grid_warp = grid_raw + delta_px * strength * 3.0

    # =========================================================
    # 🔹 Normalize for grid_sample
    # =========================================================
    grid_norm = grid_warp.clone()
    grid_norm[..., 0] = 2.0 * grid_norm[..., 0] / (W - 1) - 1.0
    grid_norm[..., 1] = 2.0 * grid_norm[..., 1] / (H - 1) - 1.0

    # =========================================================
    # 🔹 Apply warp
    # =========================================================
    latents_out = F.grid_sample(latents, grid_norm, align_corners=True)

    # =========================================================
    # 🔹 DEBUG
    # =========================================================
    if debug:

        print("\n[DEBUG][GLOBAL] ===============================")
        print(f"[DEBUG][GLOBAL] Time: {time.time() - t0:.5f}s")
        print(f"[DEBUG][GLOBAL] Strength: {strength}")

        # --- joint motion ---
        for i, j in enumerate(key_joints):
            move_px = (pts[:, i, :] - prev_pts[:, i, :]) * torch.tensor([W, H], device=device)
            print(f"[DEBUG][GLOBAL] {j} movement (px): {move_px.squeeze().tolist()}")

        print(f"[DEBUG][GLOBAL] Mean delta_px: {delta_px.abs().mean().item():.4f}")

        # --- sanity checks ---
        print(f"[DEBUG][GLOBAL] grid_raw min/max: {grid_raw.min().item():.2f} / {grid_raw.max().item():.2f}")
        print(f"[DEBUG][GLOBAL] grid_warp min/max: {grid_warp.min().item():.2f} / {grid_warp.max().item():.2f}")

        # =====================================================
        # 🔹 optional debug image
        # =====================================================
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)

            img = latents_out[0].detach().float().cpu()
            if img.shape[0] > 3:
                img = img[:3]

            torchvision.utils.save_image(
                (img + 1.0) / 2.0,
                os.path.join(debug_dir, "global_pose_debug.png")
            )

            print(f"[DEBUG][GLOBAL] Saved: {debug_dir}/global_pose_debug.png")

    return latents_out, delta_px, grid_raw, grid_norm


#------------------------------------------------------------------------------------------
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
    prev_grid=None,
    strength=2.0
):
    if device is None:
        device = latents.device

    B, C, H, W = latents.shape
    latents_in = latents.clone()

    # =========================
    # Facial points
    # =========================
    facial_points = pose.estimate_facial_points_full(smooth=smooth)
    pose.set_prev_facial_points(facial_points)

    # =========================
    # Time
    # =========================
    t = frame_counter / 10.0
    t = torch.tensor(t, device=device)

    # =========================
    # 🔥 BASE MOTION (boosté volontairement)
    # =========================
    face_delta = torch.zeros((B, H, W, 2), device=device)

    # mouvement global visage
    dx = 0.03 * torch.sin(t * 2.0)
    dy = 0.04 * torch.sin(t * 1.7)

    face_delta[..., 0] += strength * dx
    face_delta[..., 1] += strength * dy

    # =========================
    # 🔥 WIND MICRO MOTION (structure corrigée)
    # =========================
    mask = mask_face
    if mask.ndim == 4:
        mask = mask.squeeze(1)

    mask = mask.unsqueeze(-1)  # [B,H,W,1]

    wind1 = torch.sin(t * 1.5)
    wind2 = torch.cos(t * 1.1)

    face_delta[..., 0] += strength * mask[..., 0] * 0.02 * wind1
    face_delta[..., 1] += strength * mask[..., 0] * 0.015 * wind2

    if debug:
        print("FACE DELTA MEAN:", face_delta.abs().mean().item())

    # =========================
    # 🔥 FACE CENTER (nose)
    # =========================
    face_center = pose.get_point(0)
    face_center_px = face_center * torch.tensor([W - 1, H - 1], device=device)
    face_center_px = face_center_px.view(B, 1, 1, 2)

    # =========================
    # GRID WARP
    # =========================
    grid_face = grid - face_center_px
    grid_face = grid_face + face_delta
    grid_face = grid_face + face_center_px

    # =========================
    # 🔥 TEMPORAL SMOOTHING
    # =========================
    if prev_grid is not None:
        alpha = 0.7
        grid_face = alpha * prev_grid + (1.0 - alpha) * grid_face

    # =========================
    # NORMALIZATION GRID_SAMPLE
    # =========================
    grid_face = grid_face.clone()

    grid_face[..., 0] = 2.0 * grid_face[..., 0] / (W - 1) - 1.0
    grid_face[..., 1] = 2.0 * grid_face[..., 1] / (H - 1) - 1.0

    # =========================
    # WARP
    # =========================
    latents_out = F.grid_sample(
        latents,
        grid_face,
        align_corners=True,
        mode="bilinear",
        padding_mode="reflection"
    )

    if debug:
        print("[DEBUG] Face warp applied OK")
        print("  grid mean:", grid_face.abs().mean().item())

    return latents_out, face_delta, facial_points

#--------------------------------------------------------------------

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
    smooth=0.85,
    strength=2.0,
    npy=False
):

    if device is None:
        device = latents.device

    B, C, H, W = latents.shape
    latents_in = latents.clone()

    # =========================
    # 🔥 NEW: compute delta propre
    # =========================
    delta, facial_points = Pose.compute_mouth_delta(
        pose=pose,
        mask_mouth=mask_mouth,
        H=H,
        W=W,
        frame_counter=frame_counter,
        device=device,
        smooth=smooth,
        strength=strength,
        debug=debug
    )

    # =========================
    # 8. GRID WARP (inchangé)
    # =========================
    face_center = pose.get_point(0)
    face_center_px = face_center * torch.tensor([W-1, H-1], device=device)
    face_center_px = face_center_px.view(B,1,1,2)

    grid_mouth = grid.clone() - face_center_px
    grid_mouth = grid_mouth + delta
    grid_mouth = grid_mouth + face_center_px

    grid_mouth[...,0] = 2.0 * grid_mouth[...,0] / (W-1) - 1.0
    grid_mouth[...,1] = 2.0 * grid_mouth[...,1] / (H-1) - 1.0

    # =========================
    # 9. SAMPLE (amélioré)
    # =========================
    latents_out = F.grid_sample(
        latents,
        grid_mouth,
        mode='bilinear',
        padding_mode='border',  # 🔥 important
        align_corners=True
    )

    # =========================
    # 10. DEBUG (inchangé)
    # =========================
    if debug and debug_dir is not None:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            save_impact_map( latents_out, latents_in, debug_dir, frame_counter, prefix="mouth" )
            if npy:
                np.save( os.path.join(debug_dir, f"mouth_delta_{frame_counter:05d}.npy"), delta.detach().cpu().numpy() )
            print("[DEBUG] Mouth warp applied OK + delta saved")
        except Exception as e:
            print(f"[WARN] mouth debug failed: {e}")

    return latents_out, delta, facial_points

#-------------------------------------------------test -----------------------------------------
def normalize_mask(mask):
    m = mask.mean().clamp(1e-6, 0.2)
    mask = mask / (m + 1e-6)
    mask = torch.clamp(mask, 0.0, 3.0)
    return mask
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
def should_freeze(frame_idx, frame_pause):
    return (frame_pause is not None) and (frame_idx % frame_pause == 0)

def get_breathing_mode(frame_counter, freeze):
    if freeze:
        return "soft", 0.6
    else:
        return "real", 1.0


def time_sin(frame_counter, freq=2.0, device="cuda"):
    t = torch.tensor(frame_counter / 10.0, device=device)
    return torch.sin(t * freq)


def get_time(frame_counter, fps=10.0, device=None):
    return torch.tensor(frame_counter / fps, device=device, dtype=torch.float32)


def apply_breathing(
    latents_world,
    pose,
    mask_torso_exp,
    frame_counter,
    breathing,
    mode="soft",
    debug=False,
    debug_dir=None,
):
    if mode == "real":
        return apply_breathing_real(latents_world, pose, mask_torso_exp, frame_counter, breathing, debug=debug, debug_dir=debug_dir)
    else:
        return apply_breathing_soft(latents_world, pose, mask_torso_exp, frame_counter, breathing, debug=debug, debug_dir=debug_dir)

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
    latents_base = latents.clone()
    latents_world = latents.clone()

    # =========================
    # 🔹 Pose et deltas
    # =========================
    start = time.time()
    pose = Pose(keypoints.to(device))
    pose.compute_torso_delta(latent_h=H, latent_w=W)
    prev_pose = Pose(prev_keypoints.to(device)) if prev_keypoints is not None else None
    timings["Pose"] = time.time() - start

    # =========================
    # 🔹 Animation Upper Body
    # =========================
    try:
        # On récupère les coordonnées brutes depuis keypoints
        upper_body_inputs = {
            "nose": keypoints[:, pose.FACIAL_POINT_IDX["nose"], :2],
            "neck": keypoints[:, pose.FACIAL_POINT_IDX["neck"], :2],
            "right_shoulder": keypoints[:, pose.FACIAL_POINT_IDX["right_shoulder"], :2],
            "right_elbow": keypoints[:, pose.FACIAL_POINT_IDX["right_elbow"], :2],
            "right_wrist": keypoints[:, pose.FACIAL_POINT_IDX["right_wrist"], :2],
            "left_shoulder": keypoints[:, pose.FACIAL_POINT_IDX["left_shoulder"], :2],
            "left_elbow": keypoints[:, pose.FACIAL_POINT_IDX["left_elbow"], :2],
            "left_wrist": keypoints[:, pose.FACIAL_POINT_IDX["left_wrist"], :2],
            "right_clavicle": keypoints[:, pose.FACIAL_POINT_IDX.get("right_clavicle", 19), :2],
            "left_clavicle": keypoints[:, pose.FACIAL_POINT_IDX.get("left_clavicle", 20), :2],
        }

        # Mise à jour via animate_upper_body
        pose_copy = Pose(pose.keypoints.clone())
        updated_upper_body = animate_upper_body(
            state=pose_copy,
            inputs=upper_body_inputs,
            mode="smooth",
            strength=0.35, debug=debug
        )
        n = min(pose.keypoints.shape[1], updated_upper_body.shape[1])
        pose.keypoints[:, :n] = updated_upper_body[:, :n]

        if debug:
            print(f"[DEBUG] Upper body animated, first shoulder delta:",
                (pose.keypoints[0, pose.FACIAL_POINT_IDX['left_shoulder'], :2] -
                keypoints[0, pose.FACIAL_POINT_IDX['left_shoulder'], :2]))

    except Exception as e:
        print("[WARN] Upper body animation failed:", e)

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
    #grid_base = grid + global_shift

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

    print("mask_hair_exp mean:", mask_hair_exp.mean().item())
    print("mask_face_exp mean:", mask_face_exp.mean().item())
    print("mask_mouth_exp mean:", mask_mouth_exp.mean().item())

    # 🔥 NOUVEAU MASQUE DÉCOR
    mask_decor = pose.create_decor_mask(H, W, mask_face, mask_torso, mask_hair, debug=debug, debug_dir=debug_dir)  # doit devenir
    mask_decor = torch.clamp(mask_decor, 0, 1).float()

    #============================================== PARTI WORD ========================================
    # =========================
    # 🔹 Global pose & stabilisation avancée
    # =========================
    if should_freeze(frame_counter, 1): # Pause traitement
        start = time.time()
        latents_world, global_delta, grid_raw, grid_global = apply_global_pose(latents_world, pose, prev_pose, H, W, device=device, strength=2.0, debug=debug, debug_dir=debug_dir)
        if prev_pose is not None:
            key_joints = ['neck','left_shoulder','right_shoulder','left_hip','right_hip']
            for joint in key_joints:
                idx = pose.FACIAL_POINT_IDX[joint]
                diff = keypoints[:,idx,:2] - prev_keypoints[:,idx,:2]
                diff = torch.clamp(diff, -3.0, 3.0)
                latents_world += diff.mean() * 0.001
        timings["GLOBAL"] = time.time() - start
        if debug:
            print("[DEBUG] GLOBAL WARP REPORT")
            print("  - delta mean px:", global_delta.abs().mean().item())
            print("  - delta max px:", global_delta.abs().max().item())
            save_impact_map(latents_world, latents_base, debug_dir, frame_counter, prefix="torso_global")


    # =========================
    # 🔹 Torso
    # =========================
    t = torch.tensor(frame_counter/10.0, device=device)
    delta_px = pose.delta.clone()
    delta_px[...,0] *= W
    delta_px[...,1] *= H
    delta_px = delta_px.view(B,1,1,2)

    #if frame_counter % 2 == 0:
    start = time.time()
    latents_before = latents_world.clone()
    latents_torso, delta_px = apply_torso_warp(latents_world, pose, mask_torso, grid, H, W, device=device, debug=debug, debug_dir=debug_dir)
    breath_strength = 0.2 + 0.1 * torch.sin(t)
    #latents_world = latents_torso*(breath_strength*mask_torso_exp) + latents_before*(1.0 - breath_strength*mask_torso_exp)
    latents_world = latents_before*(1.0-breath_strength*mask_torso_exp) + latents_torso*(breath_strength*mask_torso_exp)
    timings["TORSO+BREATH"] = time.time() - start
    if debug:
        save_impact_map(latents_world, latents_before, debug_dir, frame_counter, prefix="torso_warp")
    # =====================================================================
    # 🔹 BREATHING
    # =====================================================================
    start = time.time()
    freeze = should_freeze(frame_counter, 3)
    mode, mode_strength = get_breathing_mode(frame_counter, freeze)
    latents_before = latents_world
    # breathing field (single output)
    latents_breath = apply_breathing( latents_world, pose, mask_torso_exp, frame_counter, breathing, mode=mode, debug=debug, debug_dir=debug_dir )
    # temporal modulation
    t = frame_counter / 10.0
    breath_strength = (0.2 + 0.2 * math.sin(t)) * mode_strength
    # =========================================================
    # ✔️ RESIDUAL INJECTION (IMPORTANT CHANGE)
    # =========================================================
    mask = mask_torso_exp ** 1.5   # smoother falloff
    latents_world = latents_before + ( (latents_breath - latents_before) * mask * breath_strength )
    timings["breathing"] = time.time() - start
    if debug:
        print(f"[DEBUG] Breathing applied ({mode})")

    #================== PARTI LOCAL ========================================

    latents_local = latents_world.clone()
    # =========================
    # 🔹 Face + temporal smoothing
    # =========================
    if not hasattr(apply_pose_driven_motion_ultra2,"prev_face_grid"):
        apply_pose_driven_motion_ultra2.prev_face_grid = [None]*B
    start = time.time()
    latents_local, face_delta, facial_points = apply_face_warp(
        latents_local, pose, mask_face, grid, H, W, frame_counter,
        device=device, debug=debug, debug_dir=debug_dir, smooth=0.85,
        prev_grid=apply_pose_driven_motion_ultra2.prev_face_grid[0] if B==1 else None
    )
    apply_pose_driven_motion_ultra2.prev_face_grid[0] = grid.clone() if B==1 else None

    face_mix = feather_inside_strict2(mask_face_exp, radius=6, blur_kernel=5, sigma=1.5)

    face_strength = 0.9
    latents_local = (
        latents_world * (1 - face_strength * face_mix) +
        latents_local * (face_strength * face_mix)
    )

    face_strength_mouth = 0.9
    latents_local = (
        latents_world * (1 - face_strength_mouth * mask_mouth_exp) +
        latents_local * (face_strength_mouth * mask_mouth_exp)
    )
    timings["FACE"] = time.time() - start

    # ===================================
    # 🔹 Mouth & micro-expressions - OK
    # ==================================
    if should_freeze(frame_counter, 4): # Pause traitement
        start = time.time()
        latents_local, mouth_delta, _ = apply_mouth_smil(
            latents_local, pose, mask_mouth, grid, H, W, frame_counter,
            device=device, debug=debug, debug_dir=debug_dir, smooth=0.85
        )
        print("MOUTH DELTA MEAN:", mouth_delta.abs().mean().item())

        # Broadcasting correct pour la bouche
        mask_mouth_corners_broadcast = mask_mouth_corners.repeat(1, C, 1, 1)
        #latents_local += 0.002 * (mask_mouth_corners_broadcast * torch.sin(t*2.0))

        phase = time_sin(frame_counter, device=latents_local.device)
        latents_local += 0.002 * mask_mouth_corners_broadcast * phase

        # Broadcasting correct pour les yeux
        mask_left_eye_broadcast  = mask_left_eye.repeat(1, C, 1, 1)
        mask_right_eye_broadcast = mask_right_eye.repeat(1, C, 1, 1)


        t = time_sin(frame_counter, freq=3.0, device=latents_world.device)
        eye_motion = 0.1 * (mask_left_eye_broadcast * t +
                        mask_right_eye_broadcast * time_sin(frame_counter, freq=3.0, device=latents_world.device))

        #eye_motion = 0.1 * (mask_left_eye_broadcast  * torch.sin(t*3.0) + mask_right_eye_broadcast * torch.cos(t*3.0))

        eye_motion = eye_motion * mask_face_exp
        latents_local += eye_motion

        timings["MOUTH+EYES"] = time.time() - start

    # ==============================
    # 🔹 Hair motion cycle - OK
    # ==============================
    if should_freeze(frame_counter, 6): # Pause traitement
        if not hasattr(apply_pose_driven_motion_ultra2,"prev_hair_fields"):
            apply_pose_driven_motion_ultra2.prev_hair_fields = [None]*B
        start = time.time()
        latents_before = latents_local.clone()
        latents_hair, hair_delta = apply_hair_motion_cycle(
            latents_local, mask_hair, grid, H, W, frame_counter, device, delta_px,
            prev_hair_field=apply_pose_driven_motion_ultra2.prev_hair_fields[0] if B==1 else None,
            debug=debug, debug_dir=debug_dir
        )
        latents_local = latents_hair * mask_hair_exp + latents_before * (1.0 - mask_hair_exp)
        print("HAIR DELTA MEAN:", hair_delta.abs().mean().item())
        apply_pose_driven_motion_ultra2.prev_hair_fields[0] = hair_delta
        timings["HAIR"] = time.time() - start


    # ===========================
    # 🔹 Decor motion cycle - OK
    # ===========================
    if should_freeze(frame_counter, 6): # Pause traitement
        if not hasattr(apply_pose_driven_motion_ultra2,"prev_decor_fields"):
            apply_pose_driven_motion_ultra2.prev_decor_fields = [None]*B
        start = time.time()
        latents_before = latents_local.clone()

        latents_decor, decor_delta = apply_hair_motion_cycle(
            latents_local, mask_decor, grid, H, W, frame_counter, device, delta_px,
            prev_hair_field=apply_pose_driven_motion_ultra2.prev_decor_fields[0] if B==1 else None,
            debug=debug, debug_dir=debug_dir
        )
        latents_local = latents_decor * mask_decor + latents_before * (1.0 - mask_decor)
        print("DECOR DELTA MEAN:", decor_delta.abs().mean().item())
        apply_pose_driven_motion_ultra2.prev_decor_fields[0] = decor_delta
        timings["DECOR"] = time.time() - start

    # =========================
    # 🔹 Micro boost global
    # =========================

    MICRO_GAIN = 2.0   # contrôle global unique

    masks = {
        "torso": (mask_torso_exp, 0.05, calibrate_amplitude(mask_torso_exp, 0.002, 0.006)),
        "hair": (mask_hair_exp, 0.20, calibrate_amplitude(mask_hair_exp, 0.002, 0.006)),
        "face": (mask_face_exp, 0.15, calibrate_amplitude(mask_face_exp, 0.002, 0.006)),
        "left_eye": (mask_left_eye, 0.6, calibrate_amplitude(mask_left_eye, 0.0008, 0.0025)),
        "right_eye": (mask_right_eye, 0.6, calibrate_amplitude(mask_right_eye, 0.0008, 0.0025)),
        "mouth": (mask_mouth_exp, 0.25, calibrate_amplitude(mask_mouth_exp, 0.01, 0.08)),
        "mouth_corners": (mask_mouth_corners, 0.2, calibrate_amplitude(mask_mouth_corners, 0.005, 0.03)),
        "decor": (mask_decor, 0.02, calibrate_amplitude(mask_decor, 0.0005, 0.0015))
    }

    start = time.time()

    # =========================
    # PREPROCESS MASKS (clean)
    # =========================
    for k, (mask, phase, amp) in masks.items():
        if mask is None:
            continue

        mask = normalize_mask(mask)

        # stabilisation douce (OK)
        mask = torch.sqrt(mask.clamp(0, 1))

        masks[k] = (mask, phase, amp)

    # =========================
    # BASE LATENTS MIX
    # =========================
    latents_mix = 0.7 * latents_local + 0.3 * latents_world

    # =========================
    # MICRO BOOST CORE (1 seule source)
    # =========================
    latents = apply_micro_boost( latents_mix, frame_counter, device, masks, keypoints, prev_keypoints, strength=1.0, debug=debug )

    # =========================
    # SECONDARY SINUS BOOST (optionnel mais propre)
    # =========================
    for key, (mask, speed, amplitude) in masks.items():
        if mask is None:
            continue

        # sécurité dimension
        if mask.ndim == 5:
            mask = mask.squeeze(2)

        mask_exp = mask.repeat(1, C, 1, 1)

        # UNIQUE scaling propre
        t = get_time(frame_counter, device=latents.device)
        signal = torch.sin(t * speed) * torch.exp(-0.1 * t)

        latents = latents + MICRO_GAIN * amplitude * mask_exp * signal

    # =========================
    # MICRO MOTION FINAL (very light)
    # =========================
    latents = apply_micro_motion( latents, frame_counter, device, masks, strength=0.05, randomize=True, debug=debug )

    timings["MICRO_BOOST"] = time.time() - start

    # =========================
    # 🔹 DECOR MASK (post-process)
    # =========================
    decor_strength = 0.25  # 🔥 réglable
    decor_mix = 0.2  # conserve un peu du mouvement
    decor_mask_soft = mask_decor * 0.8  # 🔥 réduit impact

    latents = latents * (1.0 - decor_strength * decor_mask_soft) + \
            (latents_world * (1.0 - decor_mix) + latents * decor_mix) * (decor_strength * decor_mask_soft)

    # =========================
    # 🔹 DEBUG FINAL
    # =========================
    if debug:
        save_impact_map(latents, latents_base, debug_dir, frame_counter, prefix="final")
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
    latents_breath = apply_breathing_real( latents, mask_torso_exp, frame_counter, breathing, debug=debug, debug_dir=debug_dir )
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


def compute_time(frame_idx, frame_pause, base_dt=0.03):
    if frame_pause is None:
        return frame_idx * base_dt

    frozen_blocks = frame_idx // frame_pause
    intra = frame_idx % frame_pause

    # ralentissement progressif dans le bloc
    decay = intra / frame_pause
    return (frozen_blocks + decay * 0.3) * base_dt
# --- En DEV: synchroniser ce freeze intelligent avec ton apply_global_pose pour éviter les conflits entre keypoints et warp global.
def update_pose_sequence_from_keypoints_batch(
    keypoints_tensor,
    prev_keypoints=None,
    frame_idx=0,
    alpha=0.9,
    add_motion=True,
    motion_scale=0.4,
    freeze_threshold=0.0015,   # 🔥 NEW
    freeze_strength=0.25,       # 0 = full freeze, 1 = no freeze
    debug=False
):

    kp = keypoints_tensor.clone()
    B, N, _ = kp.shape

    # =========================================================
    # 🔹 1. TEMPORAL SMOOTHING
    # =========================================================
    if prev_keypoints is not None:
        kp = alpha * kp + (1 - alpha) * prev_keypoints

    if not add_motion:
        return kp

    t = frame_idx * 0.03

    # =========================================================
    # 🔹 2. MOTION ENERGY ESTIMATION (NEW CORE)
    # =========================================================
    motion_energy = 0.0
    if prev_keypoints is not None:
        motion_energy = (kp - prev_keypoints).abs().mean()

    # =========================================================
    # 🔹 3. FREEZE GATE (soft)
    # =========================================================
    freeze_gate = 1.0
    if motion_energy < freeze_threshold:
        # plus le mouvement est faible, plus on freeze
        freeze_gate = motion_energy / freeze_threshold
        freeze_gate = torch.clamp(torch.tensor(freeze_gate), 0.0, 1.0)

        # inverse blending (freeze dominant)
        freeze_gate = freeze_strength + (1 - freeze_strength) * freeze_gate

    # =========================================================
    # 🔹 4. GROUPS
    # =========================================================
    pelvis_ids   = [11, 8]
    spine_ids    = [1, 2, 5, 8]
    shoulder_ids = [2, 5]
    head_id      = 0
    limb_ids     = list(range(min(N, 25)))

    # =========================================================
    # 🔹 5. GLOBAL MOTION (reduced by freeze)
    # =========================================================
    slow_sway = 0.003 * math.sin(t * 0.6)
    drift_x   = 0.001 * math.sin(t * 0.15)
    drift_y   = 0.001 * math.cos(t * 0.13)

    global_offset = torch.zeros_like(kp[:, :, :2])
    global_offset[..., 0] += slow_sway + drift_x
    global_offset[..., 1] += drift_y

    kp[..., :2] += global_offset * motion_scale * freeze_gate

    # =========================================================
    # 🔹 6. BREATHING (soft freeze-aware)
    # =========================================================
    breath = 0.004 * math.sin(t * 0.7)

    kp[:, spine_ids, 1] += breath * freeze_gate
    kp[:, shoulder_ids, 1] += breath * 0.6 * freeze_gate

    # =========================================================
    # 🔹 7. SPINE WAVE
    # =========================================================
    spine_wave = 0.002 * math.sin(t * 0.8)

    kp[:, spine_ids, 1] += spine_wave * freeze_gate
    kp[:, spine_ids, 0] += spine_wave * 0.2 * freeze_gate

    # =========================================================
    # 🔹 8. HEAD MOTION (inertial + freeze)
    # =========================================================
    head_x = 0.0025 * math.sin(t * 0.9)
    head_y = 0.0020 * math.cos(t * 0.85)

    kp[:, head_id, 0] += head_x * freeze_gate
    kp[:, head_id, 1] += head_y * freeze_gate

    kp[:, head_id] += (kp[:, 2] - kp[:, 5]) * 0.03 * freeze_gate

    # =========================================================
    # 🔹 9. LIMBS (strong freeze sensitivity)
    # =========================================================
    limb_noise = 0.0006 * torch.randn_like(kp[:, limb_ids, :2])
    kp[:, limb_ids, :2] += limb_noise * freeze_gate

    # =========================================================
    # 🔹 10. MICRO STABILIZATION (reduced when frozen)
    # =========================================================
    kp[..., :2] += torch.randn_like(kp[..., :2]) * 0.0004 * freeze_gate

    # =========================================================
    # 🔹 11. CLAMP
    # =========================================================
    kp[..., :2] = torch.clamp(kp[..., :2], -1.2, 1.2)

    # =========================================================
    # 🔹 DEBUG
    # =========================================================
    if debug:
        print("\n[DEBUG][FREEZE]")
        print(f"motion_energy: {motion_energy}")
        print(f"freeze_gate: {freeze_gate}")
        print(f"motion_scale: {motion_scale}")

    return kp

def update_pose_sequence_from_keypoints_batch_stable(
    keypoints_tensor,
    prev_keypoints=None,
    frame_idx=0,
    alpha=0.9,
    add_motion=True,
    motion_scale=0.4,   # 🔥 NOUVEAU: contrôle global vitesse
    debug=False
):

    kp = keypoints_tensor.clone()
    B, N, _ = kp.shape

    # =========================================================
    # 🔹 1. TEMPORAL SMOOTHING (plus fort)
    # =========================================================
    if prev_keypoints is not None:
        kp = alpha * kp + (1 - alpha) * prev_keypoints

    if not add_motion:
        return kp

    t = frame_idx * 0.03   # 🔥 RALENTI x3

    # =========================================================
    # 🔹 2. GROUPS
    # =========================================================
    pelvis_ids   = [11, 8]
    spine_ids    = [1, 2, 5, 8]
    shoulder_ids = [2, 5]
    head_id      = 0
    limb_ids     = list(range(min(N, 25)))

    # =========================================================
    # 🔹 3. GLOBAL MOTION (VERY SLOW drift)
    # =========================================================
    slow_sway = 0.003 * math.sin(t * 0.6)
    drift_x   = 0.001 * math.sin(t * 0.15)
    drift_y   = 0.001 * math.cos(t * 0.13)

    global_offset = torch.zeros_like(kp[:, :, :2])
    global_offset[..., 0] += slow_sway + drift_x
    global_offset[..., 1] += drift_y

    kp[..., :2] += global_offset * motion_scale

    # =========================================================
    # 🔹 4. BREATHING (ultra slow)
    # =========================================================
    breath = 0.004 * math.sin(t * 0.7)  # 🔥 beaucoup plus lent

    kp[:, spine_ids, 1] += breath
    kp[:, shoulder_ids, 1] += breath * 0.6

    # =========================================================
    # 🔹 5. SPINE WAVE (low frequency)
    # =========================================================
    spine_wave = 0.002 * math.sin(t * 0.8)

    kp[:, spine_ids, 1] += spine_wave
    kp[:, spine_ids, 0] += spine_wave * 0.2

    # =========================================================
    # 🔹 6. HEAD MOTION (very slow inertia)
    # =========================================================
    head_x = 0.0025 * math.sin(t * 0.9)
    head_y = 0.0020 * math.cos(t * 0.85)

    kp[:, head_id, 0] += head_x
    kp[:, head_id, 1] += head_y

    # inertia head (très réduit)
    kp[:, head_id] += (kp[:, 2] - kp[:, 5]) * 0.03

    # =========================================================
    # 🔹 7. LIMBS (quasi static)
    # =========================================================
    limb_noise = 0.0006 * torch.randn_like(kp[:, limb_ids, :2])
    kp[:, limb_ids, :2] += limb_noise

    # =========================================================
    # 🔹 8. MICRO STABILIZATION
    # =========================================================
    kp[..., :2] += torch.randn_like(kp[..., :2]) * 0.0004

    # =========================================================
    # 🔹 9. CLAMP
    # =========================================================
    kp[..., :2] = torch.clamp(kp[..., :2], -1.2, 1.2)

    # =========================================================
    # 🔹 DEBUG
    # =========================================================
    if debug:
        motion_strength = (kp - keypoints_tensor).abs().mean()
        print(f"[DEBUG] motion: {motion_strength.item():.6f}")
        print(f"[DEBUG] motion_scale: {motion_scale}")
        print(f"[DEBUG] freq t: {t:.4f}")

    return kp

