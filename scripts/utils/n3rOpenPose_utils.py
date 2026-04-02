#********************************************
# n3rOpenPose_utils.py
#********************************************
import torch
from diffusers import ControlNetModel
import math
import torch.nn.functional as F
from .n3rControlNet import create_canny_control, control_to_latent, match_latent_size
from .tools_utils import ensure_4_channels, print_generation_params, sanitize_latents
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import traceback




def gaussian_blur_tensor(x, kernel_size=3, sigma=1.0):
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
#---------------------------------- CLASS POSE ------------------------------------------------------
class Pose:
    def __init__(self, keypoints: torch.Tensor):
        """
        keypoints : [B, 17, 3]  -> normalized [0,1] + confidence
        """
        self.keypoints = keypoints.clone()
        self.B = keypoints.shape[0]
        self.delta = None
        self.angles = None
        self.device = keypoints.device

    # ----------------- Keypoint utils -----------------
    def get_point(self, idx):
        """Récupère un keypoint spécifique [B,2]"""
        return self.keypoints[:, idx, :2]

    # ----------------- Torso delta -----------------
    def compute_torso_delta(self, latent_h: int, latent_w: int, scale: float = 0.8):
        """Calcule le déplacement du torse pour grid warp"""
        r_shoulder = self.get_point(2)
        l_shoulder = self.get_point(5)
        torso_center = (r_shoulder + l_shoulder) * 0.5  # [B,2]

        # Transformation en delta
        delta = torso_center * scale
        delta = torch.tanh(delta * 2.0) * 0.5  # stabilisation
        self.delta = delta
        return delta

    # ----------------- Torso angle -----------------
    def compute_torso_angle(self):
        """Calcule l'angle du torse selon les épaules"""
        r_shoulder = self.get_point(2)
        l_shoulder = self.get_point(5)
        vec = r_shoulder - l_shoulder
        angle = torch.atan2(vec[:,1], vec[:,0]).unsqueeze(1)  # [B,1]
        self.angles = angle
        return angle

    def create_upper_body_mask(self, H: int, W: int, kernel_size: int = 15, sigma: float = 5.0,
                           debug: bool = False, debug_dir: str = None, frame_counter: int = 0):
        """
        Crée un masque polygonal flouté torse + bras.
        Le torse est basé sur les épaules, et les bras sur coudes et poignets.
        """
        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        for b in range(self.B):
            # Récupère les keypoints
            r_sh = self.get_point(2)[b].cpu().numpy()
            r_el = self.get_point(3)[b].cpu().numpy()
            r_wr = self.get_point(4)[b].cpu().numpy()
            l_sh = self.get_point(5)[b].cpu().numpy()
            l_el = self.get_point(6)[b].cpu().numpy()
            l_wr = self.get_point(7)[b].cpu().numpy()

            # Convertir en pixels
            def to_px(kp):
                return [int(kp[0]*(W-1)), int(kp[1]*(H-1))]
            pts = np.array([
                to_px(r_sh), to_px(r_el), to_px(r_wr),
                to_px(l_wr), to_px(l_el), to_px(l_sh)
            ], dtype=np.int32)

            # Remplir le polygone
            mask_np = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask_np, [pts], 255)

            # Convertir en tensor et assigner
            mask[b,0] = torch.from_numpy(mask_np / 255.0).to(self.device)

        # Floutage pour adoucir les bords
        mask = gaussian_blur_tensor(mask, kernel_size=kernel_size, sigma=sigma)
        mask = torch.clamp(mask, 0, 1)

        # -------------------- Debug --------------------
        if debug and debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            debug_scale = 4
            mask_np_debug = (mask[0,0].detach().cpu().numpy() * 255).astype(np.uint8)
            mask_debug = cv2.resize(mask_np_debug, (W*debug_scale, H*debug_scale), interpolation=cv2.INTER_NEAREST)
            mask_debug_rgb = cv2.cvtColor(mask_debug, cv2.COLOR_GRAY2BGR)
            save_path = os.path.join(debug_dir, f"skeleton_mask_{frame_counter:05d}.png")
            cv2.imwrite(save_path, mask_debug_rgb)
            print(f"[DEBUG] Upper body mask saved (scale {debug_scale}): {save_path}")

        return mask
#----------------------------------------------------------------------------------------------------------------------

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
        elif i in [2,3,4,5,6,7]: # arms
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
        (1,2), (2,3), (3,4),   # right arm
        (1,5), (5,6), (6,7),   # left arm
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


def compute_delta(latents_out, latent_ref, controlnet_scale, importance):
    delta = latents_out - latent_ref
    delta = torch.nan_to_num(delta, nan=0.0, posinf=1.0, neginf=-1.0)

    # 🔥 adaptive blending ici
    delta = torch.tanh(delta) * 0.15 * importance

    return delta * controlnet_scale


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
    """
    Log propre d'une erreur sur une frame.

    Args:
        img_path: chemin de l'image/frame
        error: exception capturée
        verbose: afficher le traceback complet
    """

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

import os
from PIL import Image
import torch

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
    import torch

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


def apply_controlnet_openpose_step(
    latents,
    t,
    unet,
    controlnet,
    scheduler,
    pose_image,
    pos_embeds,
    neg_embeds=None,
    guidance_scale=5.0,
    controlnet_scale=0.7,
    device="cuda",
    dtype=torch.float16,
    debug=False
):
    import torch

    latents = latents.to(device=device, dtype=dtype)
    pose_image = pose_image.to(device=device, dtype=dtype)

    # 🔁 classifier-free guidance
    if neg_embeds is not None:
        latent_model_input = torch.cat([latents] * 2)
        encoder_states = torch.cat([neg_embeds, pos_embeds])
        pose_input = torch.cat([pose_image] * 2)
    else:
        latent_model_input = latents
        encoder_states = pos_embeds
        pose_input = pose_image

    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # 🔥 ControlNet
    down_samples, mid_sample = controlnet(
        latent_model_input,
        t,
        encoder_hidden_states=encoder_states,
        controlnet_cond=pose_input,
        return_dict=False
    )

    # 🔥 UNet avec ControlNet
    noise_pred = unet(
        latent_model_input,
        t,
        encoder_hidden_states=encoder_states,
        down_block_additional_residuals=[d * controlnet_scale for d in down_samples],
        mid_block_additional_residual=mid_sample * controlnet_scale
    ).sample

    # 🔁 CFG
    if neg_embeds is not None:
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

    # 🔥 Scheduler step
    latents = scheduler.step(noise_pred, t, latents).prev_sample

    if debug:
        print(f"[ControlNet] t={t}, latents min/max: {latents.min().item():.3f}/{latents.max().item():.3f}")

    return latents


def generate_pose_sequence(
    base_pose,
    num_frames=16,
    motion_type="idle",   # "idle", "sway", "zoom", "breath"
    amplitude=5.0,
    device="cuda",
    dtype=None,
    debug=False
):
    """
    Génère une séquence de poses animées (OpenPose-like).

    Args:
        base_pose: tensor [1,3,H,W] ou [1,1,H,W] (image pose)
        num_frames: nombre de frames
        motion_type: type animation
        amplitude: intensité mouvement
        device: device
        debug: print infos

    Returns:
        List[Tensor]: liste de control_tensor
    """


    if dtype is None:
        dtype = base_pose.dtype

    base_pose = base_pose.to(device=device, dtype=dtype)
    B, C, H, W = base_pose.shape

    poses = []

    for t in range(num_frames):

        alpha = t / max(1, num_frames - 1)
        phase = 2 * math.pi * alpha

        pose = base_pose.clone()

        # --------------------------------------------------
        # 🎯 MOTION TYPES
        # --------------------------------------------------

        # 🔹 1. IDLE (micro mouvements naturels)
        if motion_type == "idle":
            dx = math.sin(phase) * amplitude * 0.3
            dy = math.cos(phase) * amplitude * 0.2

        # 🔹 2. SWAY (balancement)
        elif motion_type == "sway":
            dx = math.sin(phase) * amplitude
            dy = 0

        # 🔹 3. BREATH (zoom subtil)
        elif motion_type == "breath":
            scale = 1.0 + math.sin(phase) * 0.03
            pose = F.interpolate(
                pose,
                scale_factor=scale,
                mode="bilinear",
                align_corners=False
            )
            pose = F.interpolate(pose, size=(H, W))
            dx, dy = 0, 0

        # 🔹 4. ZOOM léger + drift
        elif motion_type == "zoom":
            scale = 1.0 + math.sin(phase) * 0.05
            pose = F.interpolate(
                pose,
                scale_factor=scale,
                mode="bilinear",
                align_corners=False
            )
            pose = F.interpolate(pose, size=(H, W))
            dx = math.sin(phase) * amplitude * 0.2
            dy = math.cos(phase) * amplitude * 0.2

        else:
            dx, dy = 0, 0

        # --------------------------------------------------
        # 🔹 Translation affine (ultra stable)
        # --------------------------------------------------
        if motion_type != "breath":
            theta = torch.tensor([
                [1, 0, dx / (W/2)],
                [0, 1, dy / (H/2)]
            ], device=device, dtype=dtype).unsqueeze(0)

            grid = F.affine_grid(theta, pose.size(), align_corners=False)
            pose = F.grid_sample(pose, grid, align_corners=False)

        # --------------------------------------------------
        # 🔒 Sécurité
        # --------------------------------------------------
        pose = torch.nan_to_num(pose, 0.0)
        pose = pose.clamp(0, 1)

        poses.append(pose)

    # --------------------------------------------------
    # 🔍 Debug
    # --------------------------------------------------
    if debug:
        print(f"[PoseSeq] frames: {num_frames}")
        print(f"[PoseSeq] type: {motion_type}")
        print(f"[PoseSeq] shape: {poses[0].shape}")

    return poses




# Chargement par defaut:
# /mnt/62G/AnimateDiff main* 54s
# animatediff ❯ ls -l /mnt/62G/huggingface/sd-controlnet-openpose/
# .rw-r--r--@ 1,4G n3oray 25 mars  22:50  diffusion_pytorch_model.safetensors
# .rw-r--r--@   67 n3oray 25 mars  22:51  Note.txt

def load_controlnet_openpose_local(
    local_model_path="/mnt/62G/huggingface/sd-controlnet-openpose",
    device="cuda",
    dtype=torch.float16,
    use_fp16=True,
    debug=True
):
    """
    Charge ControlNet OpenPose depuis un dossier local contenant :
      - diffusion_pytorch_model.safetensors
      - config.json

    Args:
        local_model_path (str): chemin vers le dossier local du modèle
        device (str): "cuda" ou "cpu"
        dtype (torch.dtype): dtype cible
        use_fp16 (bool): force fp16 si possible
        debug (bool): logs détaillés

    Returns:
        controlnet (ControlNetModel)
    """
    print(f"Chargement ControlNet OpenPose depuis dossier local : {local_model_path}")
    print(f"device   : {device}")
    print(f"dtype    : {dtype}")

    try:
        # 🔹 Choix dtype intelligent
        load_dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            local_model_path,
            torch_dtype=load_dtype,
            local_files_only=True
        )

        # 🔹 Move device
        controlnet = controlnet.to(device)

        # 🔹 Vérification paramètres
        total_params = sum(p.numel() for p in controlnet.parameters()) / 1e6
        if debug:
            print(f"🧠 ControlNet prêt")
            print(f"   params : {total_params:.1f}M")
            print(f"   dtype  : {next(controlnet.parameters()).dtype}")
            print(f"   device : {next(controlnet.parameters()).device}")

        # 🔹 Mode eval
        controlnet.eval()

        # 🔹 Nettoyage mémoire GPU
        torch.cuda.empty_cache()

        return controlnet

    except Exception as e:
        print("❌ ERREUR chargement ControlNet depuis dossier local")
        print(str(e))

        # 🔥 fallback CPU (évite crash)
        if device.startswith("cuda"):
            print("⚠ Fallback CPU...")
            return load_controlnet_openpose_local(
                local_model_path=local_model_path,
                device="cpu",
                dtype=torch.float32,
                use_fp16=False,
                debug=debug
            )

        raise e


def load_controlnet_openpose(
    device="cuda",
    dtype=torch.float16,
    model_id="lllyasviel/sd-controlnet-openpose",
    use_fp16=True,
    debug=True
):
    """
    Charge ControlNet OpenPose avec gestion propre GPU / CPU / dtype.

    Args:
        device (str): "cuda" ou "cpu"
        dtype (torch.dtype): dtype cible (fp16 recommandé)
        model_id (str): repo HF
        use_fp16 (bool): force fp16 si possible
        debug (bool): logs détaillés

    Returns:
        controlnet (ControlNetModel)
    """
    print(f"Chargement ControlNet OpenPose - Parametres recommander:")
    print(f"guidance_scale = 5.0 → 6.0")
    print(f"controlnet_strength = 0.7 → 0.9")
    print(f"latents update factor = 0.1  ✅ (ne pas monter)")

    if debug:
        print("🔄 Chargement ControlNet OpenPose...")
        print(f"   model_id : {model_id}")
        print(f"   device   : {device}")
        print(f"   dtype    : {dtype}")

    try:
        # 🔹 Choix dtype intelligent
        load_dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=load_dtype
        )

        if debug:
            print("✅ Modèle chargé depuis HuggingFace")

        # 🔹 Move device
        controlnet = controlnet.to(device)

        # 🔹 Vérification paramètres
        total_params = sum(p.numel() for p in controlnet.parameters()) / 1e6

        if debug:
            print(f"🧠 ControlNet prêt")
            print(f"   params : {total_params:.1f}M")
            print(f"   dtype  : {next(controlnet.parameters()).dtype}")
            print(f"   device : {next(controlnet.parameters()).device}")

        # 🔹 Mode eval
        controlnet.eval()

        # 🔹 Sécurité mémoire (important pour 4GB)
        torch.cuda.empty_cache()

        return controlnet

    except Exception as e:
        print("❌ ERREUR chargement ControlNet")
        print(str(e))

        # 🔥 fallback CPU (évite crash)
        if device == "cuda":
            print("⚠ Fallback CPU...")
            return load_controlnet_openpose(
                device="cpu",
                dtype=torch.float32,
                use_fp16=False,
                debug=debug
            )

        raise e

def apply_controlnet_openpose_step_ultrasafe(
    latents,
    t,
    unet,
    controlnet,
    scheduler,
    pose_image,
    pos_embeds,
    neg_embeds=None,
    guidance_scale=5.0,
    controlnet_scale=0.7,
    device="cuda",
    dtype=torch.float16,
    debug=False
):
    import traceback

    # Backup latents
    latents_prev = latents.clone().to(device=device, dtype=dtype)

    # 🔹 Cast strict au dtype du modèle
    latents = latents.to(device=device, dtype=dtype)
    pose_image = pose_image.to(device=device, dtype=dtype)
    pos_embeds = pos_embeds.to(device=device, dtype=dtype)
    if neg_embeds is not None:
        neg_embeds = neg_embeds.to(device=device, dtype=dtype)

    # 🔹 CFG batch
    if neg_embeds is not None:
        latent_model_input = torch.cat([latents] * 2)
        encoder_states = torch.cat([neg_embeds, pos_embeds])
        pose_input = torch.cat([pose_image] * 2)
    else:
        latent_model_input = latents
        encoder_states = pos_embeds
        pose_input = pose_image

    latent_model_input = latent_model_input.to(dtype=dtype)
    encoder_states = encoder_states.to(dtype=dtype)
    pose_input = pose_input.to(dtype=dtype)

    # Scheduler scale
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # ControlNet
    try:
        down_samples, mid_sample = controlnet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_states,
            controlnet_cond=pose_input,
            return_dict=False
        )
    except Exception as e:
        if debug:
            print(f"[ControlNet ERROR] {e}")
            traceback.print_exc()
        return latents_prev

    # Normalisation
    down_samples = [torch.nan_to_num(d / (d.std() + 1e-6), nan=0.0, posinf=1.0, neginf=-1.0) for d in down_samples]
    mid_sample = torch.nan_to_num(mid_sample / (mid_sample.abs().mean() + 1e-6), nan=0.0, posinf=1.0, neginf=-1.0)

    # UNet
    try:
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_states,
            down_block_additional_residuals=[d * controlnet_scale for d in down_samples],
            mid_block_additional_residual=mid_sample * controlnet_scale
        ).sample
    except Exception as e:
        if debug:
            print(f"[UNet ERROR] {e}")
            traceback.print_exc()
        return latents_prev

    # CFG
    if neg_embeds is not None:
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

    # Scheduler step
    try:
        latents = scheduler.step(noise_pred, t, latent_model_input).prev_sample
    except Exception as e:
        if debug:
            print(f"[Scheduler ERROR] {e}")
            traceback.print_exc()
        return latents_prev

    if neg_embeds is not None:
        latents = latents.chunk(2)[0]

    # Final safety
    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)
    latents = torch.clamp(latents, -0.85, 0.85)

    if debug:
        print(f"[ControlNet OK] t={t}, dtype={latents.dtype}, min/max={latents.min().item():.3f}/{latents.max().item():.3f}")

    return latents


def apply_openpose_tilewise(
    latents,
    pose,
    apply_fn,
    block_size=64,
    overlap=32,
    device='cuda',
    debug=False,
    debug_dir=None,
    frame_idx=0,
    full_res=(1024, 512),
    scale=4
):
    """
    Applique une fonction tile-wise ControlNet/UNet sur les latents
    en gérant correctement les bords et la taille des tiles.
    """
    B, C, H, W = latents.shape
    stride = block_size - overlap

    # --- Gaussian weight mask ---
    def make_weight_mask(size):
        coords = torch.linspace(-1, 1, size, device=device)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        sigma = 0.5
        mask = torch.exp(-(dist**2) / (2 * sigma**2))
        return mask

    weight_mask_full = make_weight_mask(block_size)

    # --- Accumulation buffers ---
    latents_accum = torch.zeros_like(latents, device=device)
    weight_accum = torch.zeros((1, 1, H, W), device=device)

    if debug:
        impact_map = torch.zeros((H, W), device=device)

    tiles = []

    # --- Generate tiles ---
    for i in range(0, H, stride):
        for j in range(0, W, stride):
            i_start = i
            j_start = j
            i_end = min(i_start + block_size, H)
            j_end = min(j_start + block_size, W)
            tiles.append((i_start, i_end, j_start, j_end))

    # --- Process tiles ---
    for tile_id, (i_start, i_end, j_start, j_end) in enumerate(tiles):
        latent_tile = latents[:, :, i_start:i_end, j_start:j_end].to(device)
        tile_coords = (j_start, i_start, j_end, i_end)
        try:
            latent_tile_processed = apply_fn(latent_tile, tile_coords)
        except Exception as e:
            print(f"[WARNING] Tile {tile_id} failed: {e}")
            traceback.print_exc()
            latent_tile_processed = latent_tile

        # Adapt weight mask for edges
        h, w = i_end - i_start, j_end - j_start
        weight_mask = weight_mask_full[:h, :w].unsqueeze(0).unsqueeze(0)

        # Accumulate
        latents_accum[:, :, i_start:i_end, j_start:j_end] += latent_tile_processed * weight_mask
        weight_accum[:, :, i_start:i_end, j_start:j_end] += weight_mask

        if debug:
            diff = (latent_tile_processed - latent_tile).abs().mean().item()
            impact_map[i_start:i_end, j_start:j_end] += diff

    # --- Normalize ---
    latents_out = latents_accum / (weight_accum + 1e-8)

    # --- Debug map ---
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        impact_map_full = F.interpolate(
            impact_map.unsqueeze(0).unsqueeze(0),
            size=full_res,
            mode='bilinear',
            align_corners=False
        ).squeeze()
        impact_np = impact_map_full.detach().cpu().numpy()
        impact_np -= impact_np.min()
        if impact_np.max() > 0:
            impact_np /= impact_np.max()
        Image.fromarray((impact_np * 255).astype(np.uint8)).save(
            os.path.join(debug_dir, f"impact_map_{frame_idx:05d}.png")
        )

    return latents_out



import torch
import torch.nn.functional as F
import numpy as np

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


from torchvision.utils import save_image
import os
from PIL import Image

#--------------------------------------------------------

def gaussian_blur_tensor2(x, kernel_size=3, sigma=0.5):
    """Applique un blur gaussien 2D sur un tensor [B,C,H,W]"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    # créer kernel 1D
    coords = torch.arange(kernel_size) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()
    # kernel 2D
    kernel2d = g[:, None] @ g[None, :]
    kernel2d = kernel2d.to(x.device, x.dtype)
    kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)  # shape [1,1,K,K]
    kernel2d = kernel2d.repeat(x.shape[1], 1, 1, 1)  # [C,1,K,K]
    x = F.pad(x, (kernel_size//2,)*4, mode='reflect')
    return F.conv2d(x, kernel2d, groups=x.shape[1])


def gaussian_blur_tensor2_soft(x, kernel_size=3, sigma=0.5, strength=0.5):
    blurred = gaussian_blur_tensor2(x, kernel_size, sigma)
    return x * (1 - strength) + blurred * strength



def unsharp_mask(latents, blur_kernel=3, blur_sigma=0.5, strength=1.0):
    """
    Renforce la netteté des latents après flou.
    - latents : tensor [B,C,H,W]
    - blur_kernel, blur_sigma : pour générer le flou
    - strength : poids du renforcement des détails
    """
    latents_blur = gaussian_blur_tensor(latents, kernel_size=blur_kernel, sigma=blur_sigma)
    high_freq = latents - latents_blur
    latents_sharp = latents + strength * high_freq
    return torch.clamp(latents_sharp, -1.5, 1.5)  # clamp pour éviter saturation

def unsharp_mask_adaptive(x, blur_kernel=3, blur_sigma=0.5, strength=1.0):
    """Unsharp adaptatif basé sur variance locale pour contours et volumes"""
    blurred = gaussian_blur_tensor2(x, kernel_size=blur_kernel, sigma=blur_sigma)
    detail = x - blurred
    # pondération adaptative selon variance locale
    local_var = F.avg_pool2d(detail**2, kernel_size=3, stride=1, padding=1)
    adaptive_strength = torch.clamp(local_var * 10.0, 0.0, 1.0) * strength
    return x + detail * adaptive_strength



def apply_openpose_tilewise_safe(
    latents,
    pose,
    apply_fn,
    past_latents=None,
    temporal_smoothing=0.3,
    block_size=64,
    overlap=32,
    device='cuda',
    debug=False,
    debug_dir=None,
    frame_idx=None,
    savetile=False
):
    B, C, H_latent, W_latent = latents.shape
    H_full, W_full = pose.shape[2], pose.shape[3]
    latents_out = torch.zeros_like(latents)
    weight_map = torch.zeros_like(latents)
    stride = block_size - overlap

    # Tile-wise processing
    for i in range(0, H_latent, stride):
        for j in range(0, W_latent, stride):
            i_end = min(i + block_size, H_latent)
            j_end = min(j + block_size, W_latent)
            tile_h, tile_w = i_end - i, j_end - j
            latent_tile = latents[:, :, i:i_end, j:j_end]
            i_full_start, i_full_end = int(i * H_full / H_latent), int(i_end * H_full / H_latent)
            j_full_start, j_full_end = int(j * W_full / W_latent), int(j_end * W_full / W_latent)
            pose_tile = pose[:, :, i_full_start:i_full_end, j_full_start:j_full_end]
            pad_h, pad_w = block_size - tile_h, block_size - tile_w
            if pad_h > 0 or pad_w > 0:
                latent_tile = F.pad(latent_tile, (0, pad_w, 0, pad_h))
                pose_tile = F.pad(pose_tile, (0, pad_w, 0, pad_h))
            try:
                latent_tile_processed = apply_fn(latent_tile, pose_tile)
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Tile failed at ({i},{j}) frame {frame_idx}: {e}")
                latent_tile_processed = latent_tile
            finally:
                torch.cuda.empty_cache()
            latent_tile_processed = latent_tile_processed[:, :, :tile_h, :tile_w]
            latents_out[:, :, i:i_end, j:j_end] += latent_tile_processed
            weight_map[:, :, i:i_end, j:j_end] += 1.0

    # Moyenne sur overlaps
    weight_map[weight_map == 0] = 1.0
    latents_out = latents_out / weight_map

    # Post-traitement multi-échelle
    latents_out = gaussian_blur_tensor2_soft(latents_out, kernel_size=3, sigma=0.5, strength=0.3)
    latents_out = unsharp_mask_adaptive(latents_out, blur_kernel=3, blur_sigma=0.5, strength=1.0)
    latents_out = unsharp_mask_adaptive(latents_out, blur_kernel=5, blur_sigma=1.0, strength=0.7)

    # --- Shadow / volume boost (ultra subtil) ---
    luminance = latents_out.mean(dim=1, keepdim=True)
    shadow_mask = torch.clamp((0.5 - luminance) * 2.0, 0.0, 1.0)
    latents_out = latents_out + shadow_mask * (latents_out - gaussian_blur_tensor2_soft(latents_out, 3, 0.5, 0.5)) * 0.3



    # Clamp et ajustement delta pour éviter saturation
    min_val, max_val = latents.min().item() - 0.3, latents.max().item() + 0.3
    latents_out = torch.clamp(latents_out, min_val, max_val)
    delta = latents_out - latents
    latents_out = latents + torch.sigmoid(delta*2.0) * delta

    # Lissage temporel
    if past_latents and temporal_smoothing > 0.0:
        smoothed = sum(past_latents[-3:] + [latents_out]) / (len(past_latents[-3:]) + 1)
        latents_out = latents_out * (1 - temporal_smoothing) + smoothed * temporal_smoothing

    # Impact map debug
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        impact_map = torch.abs(latents_out - latents).mean(1, keepdim=True)
        impact_map_full = F.interpolate(impact_map, size=(H_full, W_full),
                                        mode='bilinear', align_corners=False)
        impact_np = impact_map_full[0,0].detach().cpu().numpy()
        impact_np -= impact_np.min()
        if impact_np.max() > 0:
            impact_np /= impact_np.max()
        impact_img = (impact_np * 255).astype(np.uint8)
        Image.fromarray(impact_img).save(
            os.path.join(debug_dir, f"impact_map_{frame_idx:05d}.png")
        )

    return latents_out



#---------------------------------
def apply_breathing_latents(
    latents: torch.Tensor,
    previous_latent: torch.Tensor | None,
    latents_before_openpose: torch.Tensor | None = None,
    latents_after_openpose: torch.Tensor | None = None,
    frame_counter: int = 0,
    device: torch.device = torch.device("cuda"),
    alpha_base: float = 0.15,
    alpha_amp: float = 0.05,
    noise_strength: float = 0.0005,
    delta_clamp: float = 0.05,
) -> torch.Tensor:
    """
    Applique un effet de "respiration" sur les latents d'une frame,
    en interpolant avec la frame précédente et en limitant l'impact d'OpenPose.

    latents: latents actuels
    previous_latent: latents de la frame précédente
    latents_before_openpose: latents avant OpenPose (optionnel)
    latents_after_openpose: latents après OpenPose (optionnel)
    frame_counter: index de la frame pour oscillation
    device: device PyTorch
    alpha_base / alpha_amp: paramètres de respiration
    noise_strength: bruit aléatoire pour fluidité
    delta_clamp: clamp du delta OpenPose pour éviter artefacts
    """

    # ------------------- Calcul alpha respiration -------------------
    alpha = alpha_base + alpha_amp * math.sin(frame_counter * 0.2)

    # ------------------- Interpolation avec frame précédente -------------------
    if previous_latent is not None:
        latents = alpha * previous_latent.to(device) + (1 - alpha) * latents

    # ------------------- Petit bruit pour éviter rigidité -------------------
    latents = latents + noise_strength * torch.randn_like(latents)

    # ------------------- Clamp général -------------------
    latents = torch.clamp(latents, -1.0, 1.0)

    # ------------------- Si OpenPose utilisé, limiter delta -------------------
    if latents_before_openpose is not None and latents_after_openpose is not None:
        delta = latents_after_openpose - latents_before_openpose
        delta = torch.clamp(delta, -delta_clamp, delta_clamp)
        latents = latents_before_openpose + delta

    # ------------------- Nettoyage final -------------------
    latents = torch.nan_to_num(latents)
    latents = sanitize_latents(latents)

    return latents


# ---------------------------
# Masque haut du corps basé sur keypoints OpenPose
# ---------------------------
# ---------------------------
# 1️⃣ Build upper body mask from OpenPose keypoints
# ---------------------------
def build_upper_body_mask(keypoints, latent_shape, margin=0.05, device='cpu', debug_dir=None, frame_counter=None):
    """
    Crée un mask pour le haut du corps (torse, bras, épaules, tête) à partir des keypoints OpenPose.

    Args:
        keypoints : tensor [B, num_joints, 3] (x, y, confidence), valeurs normalisées [0,1]
        latent_shape : tuple (B, C, H, W)
        margin : proportion pour étendre le masque autour du point
        device : torch device
        debug_dir : dossier pour sauvegarde debug mask
        frame_counter : numéro de frame pour debug

    Returns:
        mask : tensor [B,1,H,W] float32
    """
    B, C, H, W = latent_shape
    mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

    # indices COCO OpenPose pour haut du corps
    upper_body_joints = [0, 1, 2, 5, 6, 7, 8, 11, 12]  # nez, yeux, épaules, coudes, hanches

    for b in range(B):
        for joint_idx in upper_body_joints:
            x, y, conf = keypoints[b, joint_idx]
            if conf < 0.1:
                continue
            # points en pixels dans le latent
            xi = int(x * W)
            yi = int(y * H)
            dx = max(1, int(W * margin))
            dy = max(1, int(H * margin))
            x0, x1 = max(0, xi - dx), min(W, xi + dx)
            y0, y1 = max(0, yi - dy), min(H, yi + dy)
            mask[b, 0, y0:y1, x0:x1] = 1.0

    # Gaussian blur via OpenCV (appliquer par batch si besoin)
    mask_np = mask.cpu().numpy()
    for b in range(B):
        mask_np[b, 0] = cv2.GaussianBlur(mask_np[b, 0], (7,7), sigmaX=1.0)

    mask = torch.tensor(mask_np, device=device)

    # Debug sauvegarde
    if debug_dir is not None and frame_counter is not None:
        import os
        os.makedirs(debug_dir, exist_ok=True)
        for b in range(B):
            vis = (mask[b,0].cpu().numpy()*255).astype(np.uint8)
            save_path = f"{debug_dir}/upper_body_mask_{frame_counter:05d}_b{b}.png"
            cv2.imwrite(save_path, vis)

    return mask


# ---------------------------
# 2️⃣ Apply breathing + upper body motion
# ---------------------------
def apply_upper_body_motion(latents, previous_latent, latents_before_openpose, latents_after_openpose,
                            keypoints, frame_counter, device, breathing=True, debug=False, debug_dir=None, ):
    """
    latents : current latents [B,C,H,W]
    previous_latent : previous frame latents [B,C,H,W] or None
    latents_before_openpose, latents_after_openpose : latents frame delta
    keypoints : OpenPose keypoints [B,25,3]
    frame_counter : int
    """
    B, C, H, W = latents.shape

    # ------------------- respiration légère -------------------
    if breathing and previous_latent is not None:
        # forte respiration sinusoidale
        alpha = 0.5 + 0.4 * math.sin(frame_counter * 0.2)
        latents = alpha * previous_latent.to(device) + (1 - alpha) * latents
        latents = latents + 0.001 * torch.randn_like(latents)
        latents = torch.clamp(latents, -1.0, 1.0)

    # ------------------- masque haut du corps -------------------
    #mask = build_upper_body_mask(keypoints, latents.shape, device=device).to(device)
    mask = build_upper_body_mask(keypoints, latents.shape, device=device, debug_dir=debug_dir, frame_counter=frame_counter).to(device)

    # ------------------- appliquer delta OpenPose sur haut du corps -------------------
    if latents_before_openpose is not None and latents_after_openpose is not None:
        delta = latents_after_openpose - latents_before_openpose
        motion_energy = delta.abs().mean(dim=1, keepdim=True)
        motion_boost = 1.0 + torch.pow(torch.clamp(motion_energy * 3.0, 0.0, 1.0), 0.7)
        latents = latents_before_openpose + delta * mask * motion_boost

    # ------------------- cleanup -------------------
    latents = torch.nan_to_num(latents)
    latents_max = latents.abs().amax(dim=(2,3), keepdim=True)
    latents = latents / (latents_max + 1e-6) * 1.05
    latents = torch.clamp(latents, -1.3, 1.3)

    return latents


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

    # 👄 Mouth detected: [(404, 446)]

    keypoints_template = [
        [418/896, 418/1280, 1.0],  # 0 nose / nez
        [383/896, 515/1280, 1.0],  # 1 neck / cou

        [627/896, 533/1280, 1.0],  # 2 right_shoulder / épaule droite
        [612/896, 838/1280, 1.0],  # 3 right_elbow / coude droit
        [488/896, 1040/1280, 1.0], # 4 right_wrist / poignet droit

        [121/896, 553/1280, 1.0],  # 5 left_shoulder / épaule gauche
        [197/896, 944/1280, 1.0],  # 6 left_elbow / coude gauche
        [431/896, 1087/1280, 1.0], # 7 left_wrist / poignet gauche

        [619/896, 1048/1280, 1.0], # 8 right_hip / hanche droite
        [0.0, 0.0, 0.0],           # 9 right_knee (absent)
        [0.0, 0.0, 0.0],           # 10 right_ankle (absent)

        [260/896, 1139/1280, 1.0], # 11 left_hip / hanche gauche
        [0.0, 0.0, 0.0],           # 12 left_knee (absent)
        [0.0, 0.0, 0.0],           # 13 left_ankle (absent)

        [495/896, 353/1280, 1.0],  # 14 right_eye / œil droit - 👁 Eyes detected:
        [373/896, 322/1280, 1.0],  # 15 left_eye / œil gauche - 👁 Eyes detected:

        [608/896, 304/1280, 1.0],  # 16 right_ear / oreille droite
        [290/896, 244/1280, 1.0],  # 17 left_ear / oreille gauche
        [404/896, 446/1280, 1.0],  # 18 mouth / Bouche
    ]

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

#---------------------------------------------------------

# 🔹 Récupère les coordonnées (x,y) d’un keypoint spécifique dans le batch
# 🔹 Récupère les coordonnées (x,y) d’un keypoint spécifique dans le batch
def get_point(kp_tensor, idx):
    return kp_tensor[:, idx, :2]  # [B,2]

# 🔹 Recentre tous les keypoints par rapport au torse (entre épaules)
#   Cela évite que le personnage se déplace vers le coin haut-gauche.
def normalize_keypoints(kp_tensor):
    kp = kp_tensor.clone()
    r_shoulder = get_point(kp, 2)
    l_shoulder = get_point(kp, 5)
    torso_center = (r_shoulder + l_shoulder) * 0.5
    kp[...,0] = kp[...,0] - torso_center[:,0].unsqueeze(1)  # recentre X
    kp[...,1] = kp[...,1] - torso_center[:,1].unsqueeze(1)  # recentre Y
    return kp

# 🔹 Applique un léger effet de respiration (scaling sinusoïdal) sur les latents
def apply_breathing(latents, previous_latent, frame_counter, breathing=True):
    """
    Applique une légère respiration sinusoidale sur les latents.
    """
    import math
    if previous_latent is not None and breathing:
        breath = 0.03 * math.sin(frame_counter * 0.15)
        latents = latents * (1.0 + breath)
    return latents

# 🔹 Calcule le déplacement du torse par rapport à la frame précédente
#   Utilisé pour translater les latents afin de suivre le mouvement.

def compute_delta_torso(kp, latent_h, latent_w, scale=0.8):
    """
    Calcule le déplacement du torse en coordonnées latentes.
    Le centre du warp est aligné sur le torse du personnage.
    """

    # Extraire les épaules
    r_shoulder = get_point(kp, 2)  # [B,2]
    l_shoulder = get_point(kp, 5)

    # Centre du torse
    torso_center = (r_shoulder + l_shoulder) * 0.5  # [B,2]

    # Normaliser par rapport à l'image (0-1)
    # On suppose que kp est déjà normalisé sur H,W [0,1]
    torso_center_norm = torso_center.clone()

    # Calculer offset depuis le centre du latent
    center_offset_x = (torso_center_norm[:,0] - 0.5) * latent_w
    center_offset_y = (torso_center_norm[:,1] - 0.5) * latent_h

    delta_torso = torch.stack([center_offset_x, center_offset_y], dim=1) * scale

    # 🔒 Stabilisation pour éviter les jumps
    delta_torso = torch.tanh(delta_torso * 2.0) * 0.5

    return delta_torso

# 🔹 Applique une translation sur les latents en utilisant un grid warp
#   Déplace visuellement le personnage selon le delta du torse.
def warp_latents(latents, delta_torso, H, W, device):
    import torch
    import torch.nn.functional as F

    B = latents.shape[0]

    dx = delta_torso[:, 0].reshape(B,1,1) * W
    dy = delta_torso[:, 1].reshape(B,1,1) * H

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B,1,1,1)

    delta_grid = torch.cat([dx*2/W, dy*2/H], dim=-1).unsqueeze(2)
    grid = grid + delta_grid

    latents_warped = F.grid_sample(
        latents,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    return latents_warped, dx, dy, grid

# 🔹 Fusionne le latents original avec le latents warpé
#   Seule la partie haut du corps (mask) est affectée par le mouvement.
def fuse_upper_body(latents, latents_warped, keypoints, device):
    mask = build_upper_body_mask(keypoints, latents.shape, device=device)
    motion_strength = 0.6
    fused = (1 - mask * motion_strength) * latents + (mask * motion_strength) * latents_warped
    return fused, mask

# 🔹 Applique la différence entre latents avant/après OpenPose
#   Permet de conserver l’impact du pose controlnet.
def apply_openpose_delta(latents, latents_before, latents_after, mask):
    if latents_before is not None and latents_after is not None:
        delta = latents_after - latents_before
        delta = torch.clamp(delta, -0.15, 0.15)
        latents = latents + delta * mask * 0.5
    return latents

# 🔹 Stabilise les latents pour éviter NaN ou valeurs extrêmes
#   Normalisation et clamp pour rester dans [-1.2,1.2].
def stabilize_latents_motion(latents):
    latents = torch.nan_to_num(latents)
    latents_max = latents.abs().amax(dim=(2,3), keepdim=True)
    latents = latents / (latents_max + 1e-6)
    latents = latents * 0.95
    return torch.clamp(latents, -1.2, 1.2)


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

def rotate_mask_around_point_v1(mask, center, angle, H, W, device):
    """
    Rotate mask around center point (B,C,H,W).
    center: [B,2] with normalized coordinates [0,1]
    angle: [B] radians
    """
    B, C, h, w = mask.shape
    # grid [-1,1]
    yy, xx = torch.meshgrid(torch.linspace(-1,1,h,device=device), torch.linspace(-1,1,w,device=device), indexing='ij')
    grid = torch.stack((xx,yy),dim=-1).unsqueeze(0).repeat(B,1,1,1)  # [B,H,W,2]

    # Rotation matrix
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    rot_mat = torch.zeros(B,2,2,device=device)
    rot_mat[:,0,0] = cos
    rot_mat[:,0,1] = -sin
    rot_mat[:,1,0] = sin
    rot_mat[:,1,1] = cos

    # center in [-1,1]
    center_norm = center.clone()
    center_norm[:,0] = center[:,0] * 2 - 1
    center_norm[:,1] = center[:,1] * 2 - 1

    # translate grid relative to center
    grid_flat = grid.reshape(B, -1, 2)
    grid_centered = grid_flat - center_norm[:,None,:]

    # apply rotation
    grid_rot = torch.bmm(grid_centered, rot_mat.transpose(1,2))
    grid_final = grid_rot + center_norm[:,None,:]
    grid_final = grid_final.reshape(B,H,W,2)

    mask_rotated = F.grid_sample(mask, grid_final, align_corners=True)
    return mask_rotated

def rotate_mask_around_point_v2(mask, center_px, angle, H, W, device):
    """
    Rotate mask around center point (pixels) using F.grid_sample
    mask: [B,C,H,W]
    center_px: [B,2] (x,y in pixels)
    angle: [B] radians
    """
    B, C, h, w = mask.shape

    # Meshgrid [H,W,2]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    grid = torch.stack((xx, yy), dim=-1)  # [H,W,2]
    grid = grid.unsqueeze(0).repeat(B,1,1,1)  # [B,H,W,2]

    # Déplacement relatif au centre
    grid_centered = grid - center_px[:, None, None, :]  # [B,H,W,2]

    cos = torch.cos(angle)[:, None, None]  # [B,1,1]
    sin = torch.sin(angle)[:, None, None]

    rot_x = cos * grid_centered[...,0] - sin * grid_centered[...,1]
    rot_y = sin * grid_centered[...,0] + cos * grid_centered[...,1]

    grid_rot = torch.stack((rot_x, rot_y), dim=-1) + center_px[:, None, None, :]  # [B,H,W,2]

    # Normalisation [-1,1] pour F.grid_sample
    grid_rot[...,0] = 2 * grid_rot[...,0] / (W-1) - 1
    grid_rot[...,1] = 2 * grid_rot[...,1] / (H-1) - 1

    mask_rotated = F.grid_sample(mask, grid_rot, align_corners=True)
    return mask_rotated




def rotate_mask_around_torso(mask, torso_points_px, angle, H, W, device="cuda"):
    """
    Rotate a mask around the torso center instead of the image center.

    mask: [B, 1, H, W] ou [B, C, H, W]
    torso_points_px: [B, 2, N] points clés du torse (x, y en pixels)
    angle: [B] en radians
    """
    B, C, H_mask, W_mask = mask.shape
    print(f"[LOG] mask.shape = {mask.shape}, torso_points_px.shape = {torso_points_px.shape}, angle.shape = {angle.shape}")

    # Calcul du centre du torse
    torso_center = torso_points_px.mean(dim=2)  # [B, 2]
    print(f"[LOG] torso_center (px) = {torso_center}")

    # Création de la grid
    yy, xx = torch.meshgrid(torch.arange(H_mask, device=device),
                            torch.arange(W_mask, device=device), indexing='ij')
    xx = xx.float().unsqueeze(0).expand(B, -1, -1)
    yy = yy.float().unsqueeze(0).expand(B, -1, -1)
    print(f"[LOG] xx min/max: {xx.min().item()}/{xx.max().item()}, yy min/max: {yy.min().item()}/{yy.max().item()}")

    # Translation vers le centre du torse
    x_shift = xx - torso_center[:, 0].view(B, 1, 1)
    y_shift = yy - torso_center[:, 1].view(B, 1, 1)
    print(f"[LOG] x_shift min/max: {x_shift.min().item()}/{x_shift.max().item()}, y_shift min/max: {y_shift.min().item()}/{y_shift.max().item()}")

    cos_angle = torch.cos(angle).view(B, 1, 1)
    sin_angle = torch.sin(angle).view(B, 1, 1)

    x_rot = cos_angle * x_shift - sin_angle * y_shift
    y_rot = sin_angle * x_shift + cos_angle * y_shift
    print(f"[LOG] x_rot min/max: {x_rot.min().item()}/{x_rot.max().item()}, y_rot min/max: {y_rot.min().item()}/{y_rot.max().item()}")

    x_final = x_rot + torso_center[:, 0].view(B, 1, 1)
    y_final = y_rot + torso_center[:, 1].view(B, 1, 1)
    print(f"[LOG] x_final min/max: {x_final.min().item()}/{x_final.max().item()}, y_final min/max: {y_final.min().item()}/{y_final.max().item()}")

    # Normalisation [-1, 1]
    x_norm = 2.0 * x_final / (W - 1) - 1.0
    y_norm = 2.0 * y_final / (H - 1) - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1)
    print(f"[LOG] grid min/max: x {grid[...,0].min().item()}/{grid[...,0].max().item()}, y {grid[...,1].min().item()}/{grid[...,1].max().item()}")

    # Rotation avec grid_sample
    mask_rotated = F.grid_sample(mask, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    print(f"[LOG] mask_rotated min/max/mean: {mask_rotated.min().item()}/{mask_rotated.max().item()}/{mask_rotated.mean().item()}")

    return mask_rotated

def compute_sternum_offset_y(neck_coords, shoulder_coords, mouth_coords=None, ratio=0.5):
    """
    Calcule un offset vertical pour centrer le haut du torse vers le sternum.

    Args:
        neck_coords (Tensor): [B,2] coordonnées (x,y) du cou
        shoulder_coords (Tensor): [B,2] coordonnées moyennes des épaules (x,y)
        mouth_coords (Tensor, optional): [B,2] coordonnées de la bouche
        ratio (float): proportion de la distance cou-épaules à utiliser pour l'offset

    Returns:
        Tensor: [B] offset vertical en coordonnées normalisées (0 à 1)
    """
    # Distance verticale cou → épaules
    vertical_dist_neck_shoulders = shoulder_coords[:,1] - neck_coords[:,1]

    if mouth_coords is not None:
        # Distance cou → bouche
        vertical_dist_neck_mouth = mouth_coords[:,1] - neck_coords[:,1]
        # On combine les deux distances pour un offset plus réaliste
        combined_dist = 0.7 * vertical_dist_neck_shoulders + 0.3 * vertical_dist_neck_mouth
    else:
        combined_dist = vertical_dist_neck_shoulders

    sternum_offset_y = ratio * combined_dist
    return sternum_offset_y

# -------------------- Fonction principale --------------------
def apply_pose_driven_motion(
    latents,
    previous_latent,
    latents_before_openpose,
    latents_after_openpose,
    keypoints,
    prev_keypoints=None,
    frame_counter=0,
    device="cuda",
    breathing=True,
    debug=False,
    debug_dir=None
):
    import os
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import numpy as np

    B, C, H, W = latents.shape
    device = latents.device
    latents_in = latents.clone()

    # -------------------- Respiration --------------------
    latents = apply_breathing(latents, previous_latent, frame_counter, breathing)
    if debug: print("[DEBUG] Respiration applied")

    # -------------------- Pose --------------------
    pose = Pose(keypoints.to(device))
    pose.compute_torso_delta(latent_h=H, latent_w=W, scale=0.8)
    angle = pose.compute_torso_angle()
    mask = pose.create_upper_body_mask(H, W, kernel_size=15, sigma=5.0,
                                       debug=debug, debug_dir=debug_dir, frame_counter=frame_counter)
    if debug:
        print(f"[DEBUG] Torso delta: {pose.delta}")
        print(f"[DEBUG] Torso angle (rad): {angle}")

    # -------------------- Grid warp --------------------
    latents_warped, dx, dy, _ = warp_latents(latents, pose.delta, H, W, device)
    if debug:
        print(f"[DEBUG] Grid warp applied")
        print(f"[DEBUG] dx min/max: {dx.min().item()}/{dx.max().item()}")
        print(f"[DEBUG] dy min/max: {dy.min().item()}/{dy.max().item()}")

    # -------------------- Recentrage automatique sur haut du torse ----
    # Points neck + épaules + bouche
    points_idx = [1, 2, 5, 18]  # neck, right_shoulder, left_shoulder
    pts = torch.stack([pose.get_point(i) for i in points_idx], dim=1)  # [B,3,2]

    neck = pts[:, 0, :]                # cou
    shoulders = pts[:, 1:, :]          # épaules
    avg_shoulders = shoulders.mean(dim=1)  # moyenne gauche/droite

    # bouche (si détectée, ici keypoint 18)
    mouth = pts[:, 3, :] if pts.shape[1] > 18 else None

    sternum_offset_y = compute_sternum_offset_y(neck, avg_shoulders, mouth_coords=mouth, ratio=0.5)
    content_center = pts.mean(dim=1, keepdim=True) + torch.tensor([0.0, sternum_offset_y], device=device)
    content_points_px = content_center * torch.tensor([W-1, H-1], device=device)

    # Rotation du masque autour du barycentre des épaules
    torso_center = pts.mean(dim=1)
    torso_points_px = torso_center * torch.tensor([W-1, H-1], device=device)
    torso_points_px = torso_points_px.view(B,2,1,1)
    mask_rotated = rotate_mask_around_torso(mask, torso_points_px, angle.view(-1), H, W, device)

    # -------------------- Offset calculation basé sur le masque --------------------
    mask_rotated = mask_rotated.clamp(0,1)
    grid_x = torch.arange(W, device=device).view(1,1,1,W).expand(B,1,H,W)
    grid_y = torch.arange(H, device=device).view(1,1,H,1).expand(B,1,H,W)

    # Centre du masque (barycentre)
    mask_sum = mask_rotated.sum(dim=[2,3], keepdim=True) + 1e-6
    mask_center_x = (mask_rotated * grid_x).sum(dim=[2,3], keepdim=True) / mask_sum
    mask_center_y = (mask_rotated * grid_y).sum(dim=[2,3], keepdim=True) / mask_sum

    # -------------------- Déplacement du contenu aligné au haut du torse ----
    content_center_px = content_points_px[:,0]  # [B,2]

    yy, xx = torch.meshgrid(
        torch.linspace(-1,1,H,device=device),
        torch.linspace(-1,1,W,device=device),
        indexing='ij'
    )
    grid = torch.stack((xx,yy),dim=-1).unsqueeze(0).repeat(B,1,1,1)

    # Décalage basé sur barycentre du haut du torse (épaules)
    shift_x = (mask_center_x[:,0,0,0] - content_center_px[:,0]) * 2 / (W-1)
    shift_y = (mask_center_y[:,0,0,0] - content_center_px[:,1]) * 2 / (H-1)

    # On réduit le décalage horizontal pour éviter que le masque ne glisse trop
    shift_x *= 0.8  # ajustable entre 0.4 et 0.6 selon image

    grid[...,0] -= shift_x[:,None,None]
    grid[...,1] -= shift_y[:,None,None]

    latents_warped = F.grid_sample(latents_warped, grid, align_corners=True)

    # -------------------- Fusion latents --------------------
    mask_boosted = torch.clamp(mask_rotated ** 0.7 * 1.5, 0, 1)
    latents = latents * (1 - mask_boosted) + latents_warped * mask_boosted

    # -------------------- OpenPose delta --------------------
    latents = apply_openpose_delta(latents, latents_before_openpose, latents_after_openpose, mask_rotated)

    # -------------------- Stabilisation --------------------
    latents = stabilize_latents_motion(latents)

    # -------------------- Impact map (debug) --------------------
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        impact_map = torch.abs(latents - latents_in).mean(1, keepdim=True)
        impact_np = impact_map[0,0].detach().cpu().numpy()
        impact_np -= impact_np.min()
        if impact_np.max() > 0:
            impact_np /= impact_np.max()
        Image.fromarray((impact_np*255).astype(np.uint8)).save(
            os.path.join(debug_dir, f"impact_map_driven_{frame_counter:05d}.png")
        )

    return latents

def apply_pose_driven_motion_v2(
    latents,
    previous_latent,
    latents_before_openpose,
    latents_after_openpose,
    keypoints,
    prev_keypoints=None,
    frame_counter=0,
    device="cuda",
    breathing=True,
    debug=False,
    debug_dir=None
):
    import os
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import numpy as np

    B, C, H, W = latents.shape
    device = latents.device
    latents_in = latents.clone()

    # -------------------- Respiration --------------------
    latents = apply_breathing(latents, previous_latent, frame_counter, breathing)
    if debug: print("[DEBUG] Respiration applied")

    # -------------------- Pose --------------------
    pose = Pose(keypoints.to(device))
    pose.compute_torso_delta(latent_h=H, latent_w=W, scale=0.8)
    angle = pose.compute_torso_angle()
    mask = pose.create_upper_body_mask(H, W, kernel_size=15, sigma=5.0,
                                       debug=debug, debug_dir=debug_dir, frame_counter=frame_counter)
    if debug:
        print(f"[DEBUG] Torso delta: {pose.delta}")
        print(f"[DEBUG] Torso angle (rad): {angle}")

    # -------------------- Grid warp --------------------
    latents_warped, dx, dy, _ = warp_latents(latents, pose.delta, H, W, device)
    if debug:
        print(f"[DEBUG] Grid warp applied")
        print(f"[DEBUG] dx min/max: {dx.min().item()}/{dx.max().item()}")
        print(f"[DEBUG] dy min/max: {dy.min().item()}/{dy.max().item()}")

    # -------------------- Recentrage automatique sur haut du torse ----
    # Points neck + épaules
    points_idx = [1, 2, 5]  # neck, right_shoulder, left_shoulder
    pts = torch.stack([pose.get_point(i) for i in points_idx], dim=1)  # [B,3,2]

    # Centrage sur haut du torse avec petit offset vertical pour sternum
    sternum_offset_y = 0.03  # très léger
    content_center = pts.mean(dim=1, keepdim=True) + torch.tensor([0.0, sternum_offset_y], device=device)

    # Conversion en pixels
    content_points_px = content_center * torch.tensor([W-1, H-1], device=device)

    # Rotation du masque autour du barycentre des épaules
    torso_center = pts.mean(dim=1)
    torso_points_px = torso_center * torch.tensor([W-1, H-1], device=device)
    torso_points_px = torso_points_px.view(B,2,1,1)
    mask_rotated = rotate_mask_around_torso(mask, torso_points_px, angle.view(-1), H, W, device)

    # -------------------- Offset calculation basé sur le masque --------------------
    mask_rotated = mask_rotated.clamp(0,1)

    grid_x = torch.arange(W, device=device).view(1,1,1,W).expand(B,1,H,W)
    grid_y = torch.arange(H, device=device).view(1,1,H,1).expand(B,1,H,W)

    # Centre du masque (barycentre)
    mask_sum = mask_rotated.sum(dim=[2,3], keepdim=True) + 1e-6
    mask_center_x = (mask_rotated * grid_x).sum(dim=[2,3], keepdim=True) / mask_sum
    mask_center_y = (mask_rotated * grid_y).sum(dim=[2,3], keepdim=True) / mask_sum

    # -------------------- Déplacement du contenu aligné au haut du torse ----
    shift_x = (mask_center_x[:,0,0,0] - content_points_px[:,0,0]) * 2 / (W-1)
    shift_y = (mask_center_y[:,0,0,0] - content_points_px[:,0,1]) * 2 / (H-1)

    yy, xx = torch.meshgrid(
        torch.linspace(-1,1,H,device=device),
        torch.linspace(-1,1,W,device=device),
        indexing='ij'
    )
    grid = torch.stack((xx,yy),dim=-1).unsqueeze(0).repeat(B,1,1,1)
    grid[...,0] -= shift_x[:,None,None]
    grid[...,1] -= shift_y[:,None,None]

    latents_warped = F.grid_sample(latents_warped, grid, align_corners=True)

    # -------------------- Fusion latents --------------------
    mask_boosted = torch.clamp(mask_rotated ** 0.7 * 1.5, 0, 1)
    latents = latents * (1 - mask_boosted) + latents_warped * mask_boosted

    # -------------------- OpenPose delta --------------------
    latents = apply_openpose_delta(latents, latents_before_openpose, latents_after_openpose, mask_rotated)

    # -------------------- Stabilisation --------------------
    latents = stabilize_latents_motion(latents)

    # -------------------- Impact map (debug) --------------------
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        impact_map = torch.abs(latents - latents_in).mean(1, keepdim=True)
        impact_np = impact_map[0,0].detach().cpu().numpy()
        impact_np -= impact_np.min()
        if impact_np.max() > 0:
            impact_np /= impact_np.max()
        Image.fromarray((impact_np*255).astype(np.uint8)).save(
            os.path.join(debug_dir, f"impact_map_driven_{frame_counter:05d}.png")
        )

    return latents

def apply_pose_driven_motion_v1(
    latents,
    previous_latent,
    latents_before_openpose,
    latents_after_openpose,
    keypoints,
    prev_keypoints=None,
    frame_counter=0,
    device="cuda",
    breathing=True,
    debug=False,
    debug_dir=None
):
    import os
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import numpy as np

    B, C, H, W = latents.shape
    device = latents.device
    latents_in = latents.clone()

    # -------------------- Respiration --------------------
    latents = apply_breathing(latents, previous_latent, frame_counter, breathing)
    if debug: print("[DEBUG] Respiration applied")

    # -------------------- Pose --------------------
    pose = Pose(keypoints.to(device))
    pose.compute_torso_delta(latent_h=H, latent_w=W, scale=0.8)
    angle = pose.compute_torso_angle()
    mask = pose.create_upper_body_mask(H, W, kernel_size=15, sigma=5.0, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter)
    if debug:
        print(f"[DEBUG] Torso delta: {pose.delta}")
        print(f"[DEBUG] Torso angle (rad): {angle}")

    # -------------------- Grid warp --------------------
    latents_warped, dx, dy, _ = warp_latents(latents, pose.delta, H, W, device)
    if debug:
        print(f"[DEBUG] Grid warp applied")
        print(f"[DEBUG] dx min/max: {dx.min().item()}/{dx.max().item()}")
        print(f"[DEBUG] dy min/max: {dy.min().item()}/{dy.max().item()}")

    # -------------------- Recentrage automatique --------------------
    # Points des épaules pour le barycentre du torse
    points_idx = [11, 12]  # gauche/droite épaule
    pts = torch.stack([pose.get_point(i) for i in points_idx], dim=1)  # [B,2,2]

    # Ajout d'un petit offset vers le bas pour approx sternum
    sternum_offset = torch.tensor([0.0, 0.15], device=device).view(1,1,2)
    content_center = pts.mean(dim=1, keepdim=True) + sternum_offset  # [B,1,2]
    content_points_px = content_center * torch.tensor([W-1, H-1], device=device)

    # Rotation du masque autour du torse
    torso_center = pts.mean(dim=1)  # [B,2]
    torso_points_px = torso_center * torch.tensor([W-1, H-1], device=device)
    torso_points_px = torso_points_px.view(B,2,1,1)
    mask_rotated = rotate_mask_around_torso(mask, torso_points_px, angle.view(-1), H, W, device)

    # -------------------- Offset calculation basé sur le masque --------------------
    mask_rotated = mask_rotated.clamp(0,1)

    grid_x = torch.arange(W, device=device).view(1,1,1,W).expand(B,1,H,W)
    grid_y = torch.arange(H, device=device).view(1,1,H,1).expand(B,1,H,W)

    # Centre du masque (barycentre)
    mask_sum = mask_rotated.sum(dim=[2,3], keepdim=True) + 1e-6
    mask_center_x = (mask_rotated * grid_x).sum(dim=[2,3], keepdim=True) / mask_sum
    mask_center_y = (mask_rotated * grid_y).sum(dim=[2,3], keepdim=True) / mask_sum

    # -------------------- Déplacement du contenu aligné au barycentre du torse ----
    shift_x = (mask_center_x[:,0,0,0] - content_points_px[:,0,0]) * 2 / (W-1)
    shift_y = (mask_center_y[:,0,0,0] - content_points_px[:,0,1]) * 2 / (H-1)

    yy, xx = torch.meshgrid(
        torch.linspace(-1,1,H,device=device),
        torch.linspace(-1,1,W,device=device),
        indexing='ij'
    )
    grid = torch.stack((xx,yy),dim=-1).unsqueeze(0).repeat(B,1,1,1)
    grid[...,0] -= shift_x[:,None,None]
    grid[...,1] -= shift_y[:,None,None]

    latents_warped = F.grid_sample(latents_warped, grid, align_corners=True)

    # -------------------- Fusion latents --------------------
    mask_boosted = torch.clamp(mask_rotated ** 0.7 * 1.5, 0, 1)
    latents = latents * (1 - mask_boosted) + latents_warped * mask_boosted

    # -------------------- OpenPose delta --------------------
    latents = apply_openpose_delta(latents, latents_before_openpose, latents_after_openpose, mask_rotated)

    # -------------------- Stabilisation --------------------
    latents = stabilize_latents_motion(latents)

    # -------------------- Impact map (debug) --------------------
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        impact_map = torch.abs(latents - latents_in).mean(1, keepdim=True)
        impact_np = impact_map[0,0].detach().cpu().numpy()
        impact_np -= impact_np.min()
        if impact_np.max() > 0:
            impact_np /= impact_np.max()
        Image.fromarray((impact_np*255).astype(np.uint8)).save(
            os.path.join(debug_dir, f"impact_map_driven_{frame_counter:05d}.png")
        )

    return latents




def apply_pose_driven_motion_test(
    latents,
    previous_latent,
    latents_before_openpose,
    latents_after_openpose,
    keypoints,
    prev_keypoints=None,
    frame_counter=0,
    device="cuda",
    breathing=True,
    debug=False,
    debug_dir=None
):
    import os
    from PIL import Image
    import torch.nn.functional as F
    import numpy as np

    B, C, H, W = latents.shape
    device = latents.device
    latents_in = latents.clone()

    # -------------------- Respiration --------------------
    latents = apply_breathing(latents, previous_latent, frame_counter, breathing)
    if debug: print("[DEBUG] Respiration applied")

    # -------------------- Création de l'objet Pose --------------------
    pose = Pose(keypoints.to(device))
    pose.compute_torso_delta(latent_h=H, latent_w=W, scale=0.8)
    angle = pose.compute_torso_angle()
    mask = pose.create_upper_body_mask(H, W, kernel_size=15, sigma=5.0, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter)
    if debug:
        print(f"[DEBUG] Torso delta: {pose.delta}")
        print(f"[DEBUG] Torso angle (rad): {angle}")

    # -------------------- Grid warp --------------------
    latents_warped, dx, dy, _ = warp_latents(latents, pose.delta, H, W, device)
    if debug:
        print(f"[DEBUG] Grid warp applied")
        print(f"[DEBUG] dx min/max: {dx.min().item()}/{dx.max().item()}")
        print(f"[DEBUG] dy min/max: {dy.min().item()}/{dy.max().item()}")

    # -------------------- Recentrage automatique --------------------
    # Si pose.get_point(i) est normalisé [0,1], convertir en pixels
    # Points pour le torse (épaule et hanche)
    points_idx = [2, 5, 8, 11]
    # Récupération des keypoints pour tout le batch
    pts = torch.stack([pose.get_point(i) for i in points_idx], dim=1)  # [B,4,2]
    # Centre barycentrique du torse
    torso_center = pts.mean(dim=1)  # [B,2]
    # Conversion en pixels
    torso_points_px = torso_center * torch.tensor([W-1, H-1], device=device)
    torso_points_px = torso_points_px.view(B,2,1,1)  # [B,2,1,1]
    mask_rotated = rotate_mask_around_torso(mask, torso_points_px, angle.view(-1), H, W, device)

    # -------------------- Offset calculation --------------------
    mask_sum = mask_rotated.sum(dim=[2,3], keepdim=True) + 1e-6
    mask_center_x = (mask_rotated * torch.arange(W, device=device)[None,None,None,:]).sum(dim=[2,3], keepdim=True) / mask_sum
    mask_center_y = (mask_rotated * torch.arange(H, device=device)[None,None,:,None]).sum(dim=[2,3], keepdim=True) / mask_sum
    mask_center = torch.cat([mask_center_x, mask_center_y], dim=1)  # [B,2,1,1]

    # centre du contenu (AVANT warp)
    # Grilles de coordonnées
    grid_x = torch.arange(W, device=device).view(1,1,1,W).expand(B,1,H,W)
    grid_y = torch.arange(H, device=device).view(1,1,H,1).expand(B,1,H,W)

    # Centre du contenu pondéré par le masque
    content_center_x = (latents * mask_rotated * grid_x).sum(dim=[1,2,3], keepdim=True) / ((latents * mask_rotated).sum(dim=[1,2,3], keepdim=True) + 1e-6)
    #content_center_y = (latents * mask_rotated * grid_y).sum(dim=[1,2,3], keepdim=True) / ((latents * mask_rotated).sum(dim=[1,2,3], keepdim=True) + 1e-6)

    content_center_y = (latents * mask_rotated * grid_y).sum(...) / ((latents * mask_rotated).sum(...) + 1e-6)

    content_center = torch.cat([content_center_x, content_center_y], dim=1)  # [B,2,1,1]

    offset = (mask_center - content_center)
    # ✅ Limitation pour éviter de sortir de l'image
    offset = offset.clamp(-20, 20)
    if debug: print("offset px:", offset[0,:,0,0])

    if debug:
        print(f"[DEBUG] mask_center: {mask_center[0,:,0,0]}")
        print(f"[DEBUG] content_center: {content_center[0,:,0,0]}")
        print(f"[DEBUG] applied offset: {offset[0,:,0,0]}")

    # -------------------- Shift avec grid_sample --------------------
    yy, xx = torch.meshgrid(
        torch.linspace(-1,1,H,device=device),
        torch.linspace(-1,1,W,device=device),
        indexing='ij'
    )
    grid = torch.stack((xx,yy),dim=-1).unsqueeze(0).repeat(B,1,1,1)
    shift_x = offset[:,0,0,0] * 2 / (W-1)
    shift_y = offset[:,1,0,0] * 2 / (H-1)
    grid[...,0] -= shift_x[:,None,None]
    grid[...,1] -= shift_y[:,None,None]

    latents_warped = F.grid_sample(latents_warped, grid, align_corners=True)

    # -------------------- Fusion latents --------------------
    mask_boosted = mask_rotated ** 0.7
    mask_boosted = mask_boosted * 1.5
    mask_boosted = torch.clamp(mask_boosted, 0, 1)
    latents = latents * (1 - mask_boosted) + latents_warped * mask_boosted

    if debug:
        print("[DEBUG] Upper body mask fusion applied (inclined)")
        print(f"[DEBUG] mask_rotated: min={mask_rotated.min().item():.3f}, max={mask_rotated.max().item():.3f}, mean={mask_rotated.mean().item():.3f}")
        print(f"[DEBUG] mask_boosted: min={mask_boosted.min().item():.3f}, max={mask_boosted.max().item():.3f}, mean={mask_boosted.mean().item():.3f}")
        warped_contribution = (latents_warped * mask_boosted).mean().item()
        print(f"[DEBUG] Approx. latents_warped contribution (mean): {warped_contribution:.3f}")

        # ✅ Debug: sauvegarde du masque après shift
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            mask_img = mask_boosted[0,0].detach().cpu().numpy()
            Image.fromarray((mask_img*255).astype(np.uint8)).save(
                os.path.join(debug_dir, f"mask_boosted_{frame_counter:05d}.png")
            )
            print(f"[DEBUG] Mask boosted saved for frame {frame_counter}")

    # -------------------- OpenPose delta --------------------
    latents = apply_openpose_delta(latents, latents_before_openpose, latents_after_openpose, mask_rotated)
    if debug: print("[DEBUG] OpenPose delta applied")

    # -------------------- Stabilisation --------------------
    latents = stabilize_latents_motion(latents)
    if debug: print("[DEBUG] Latents stabilized")

    # -------------------- Impact map --------------------
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        impact_map = torch.abs(latents - latents_in).mean(1, keepdim=True)
        impact_np = impact_map[0,0].detach().cpu().numpy()
        impact_np -= impact_np.min()
        if impact_np.max() > 0:
            impact_np /= impact_np.max()
        Image.fromarray((impact_np*255).astype(np.uint8)).save(
            os.path.join(debug_dir, f"impact_map_driven_{frame_counter:05d}.png")
        )
        print(f"[DEBUG] Impact map saved for frame {frame_counter}")

    return latents

#-----------------------------------------------------------------------------------------

def create_upper_body_mask(keypoints, H, W, device):
    """
    Crée un masque simple du torse + bras.
    """
    B = keypoints.shape[0]
    mask = torch.zeros(B,1,H,W,device=device)
    # Pour simplifier, on met 1 là où sont les keypoints du torse et bras
    for i in [2,3,4,5,6,7]:  # épaules + coudes + poignets
        kp = get_point(keypoints, i)
        x = (kp[:,0] * (W-1)).long()
        y = (kp[:,1] * (H-1)).long()
        mask[torch.arange(B),0,y,x] = 1.0
    mask = gaussian_blur_tensor(mask, kernel_size=15, sigma=5.0)
    mask = torch.clamp(mask,0,1)
    return mask



#------------------------------------------------------------------------------------------
def update_pose_sequence_from_keypoints_batch(
    keypoints_tensor,
    prev_keypoints=None,
    frame_idx=0,
    alpha=0.5,
    add_motion=True,
    debug=False
):
    """
    Génère une évolution temporelle des keypoints.
    Fonctionne même avec pose statique (motion procédural).
    """

    import torch
    import math

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

        t = frame_idx

        # -------- Breathing (vertical torso + épaules)
        breath = 0.015 * math.sin(t * 0.15)

        # épaules (indices OpenPose)
        kp[:, 2, 1] += breath   # right shoulder Y
        kp[:, 5, 1] += breath   # left shoulder Y

        # -------- Sway (balancement gauche/droite)
        sway = 0.02 * math.sin(t * 0.08)

        kp[:, :, 0] += sway  # léger mouvement global en X

        # -------- Head motion (indépendant)
        head_idx = 0
        kp[:, head_idx, 0] += 0.01 * math.sin(t * 0.2)
        kp[:, head_idx, 1] += 0.01 * math.cos(t * 0.18)

        # -------- Drift lent (très faible)
        drift_x = 0.005 * math.sin(t * 0.03)
        drift_y = 0.005 * math.cos(t * 0.025)

        kp[:, :, 0] += drift_x
        kp[:, :, 1] += drift_y

        # -------- Micro noise (anti-freeze)
        noise = torch.randn_like(kp[..., :2]) * 0.003
        kp[..., :2] += noise

    # =========================
    # 🔹 3. STABILISATION
    # =========================
    kp[..., :2] = torch.clamp(kp[..., :2], -1.2, 1.2)

    # =========================
    # 🔹 DEBUG
    # =========================
    if debug:
        motion_strength = (kp - keypoints_tensor).abs().mean()
        print(f"[DEBUG] Keypoint motion strength: {motion_strength.item():.6f}")

    return kp

#--------------------------------------------------------------------------------------------------------------

def save_debug_pose_image_with_skeleton(
    pose_tensor,
    keypoints_tensor,
    frame_counter,
    output_dir,
    cfg=None,
    prefix="openpose"
):
    """
    Sauvegarde une image de pose ET un squelette OpenPose pour contrôle visuel.

    Args:
        pose_tensor (torch.Tensor): [B,3,H,W] normalisé [-1,1] ou [C,H,W]
        keypoints_tensor (torch.Tensor): [B,18,3] (x,y,conf) normalisé [0,1]
        frame_counter (int): numéro de frame
        output_dir (str): dossier où sauvegarder
        cfg (dict, optional): peut contenir 'visual_debug' pour activer/désactiver
        prefix (str, optional): préfixe du fichier
    """

    if cfg is not None and cfg.get("visual_debug") is False:
        return

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------
    # 🔹 Convertir pose_tensor en image RGB [0,255]
    # ----------------------------
    pose_img = pose_tensor[0].detach().cpu()
    if pose_img.ndim == 3 and pose_img.shape[0] == 3:
        pose_img = pose_img.permute(1,2,0)  # H,W,C
    pose_img = ((pose_img + 1.0)/2.0 * 255).clamp(0,255).byte().numpy()

    # Sauvegarde simple de l'image de pose
    filename_pose = f"{prefix}_{frame_counter:05d}.png"
    path_pose = os.path.join(output_dir, filename_pose)
    cv2.imwrite(path_pose, cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))
    print(f"[DEBUG] Pose sauvegardée : {path_pose}")

    # ---------------------------
    # 🔹 Dessin du squelette via debug_draw_openpose_skeleton
    # ---------------------------
    if keypoints_tensor is not None:
        debug_draw_openpose_skeleton(
            pose_full_image=pose_tensor.unsqueeze(0) if pose_tensor.ndim==3 else pose_tensor,
            keypoints_tensor=keypoints_tensor,
            debug_dir=output_dir,
            frame_counter=frame_counter
        )
