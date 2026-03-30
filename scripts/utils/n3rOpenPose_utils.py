#********************************************
# n3rOpenPose_utils.py
#********************************************
import torch
from diffusers import ControlNetModel
import math
import torch.nn.functional as F
from .n3rControlNet import create_canny_control, control_to_latent, match_latent_size
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import traceback

import torch
import torch.nn.functional as F

def gaussian_blur_tensor(x, kernel_size=5, sigma=1.0):
    """Applique un flou gaussien sur un tensor 2D ou 4D (B,C,H,W)."""
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif x.ndim == 3:
        x = x.unsqueeze(0)  # [1,C,H,W]

    # créer kernel gaussien 1D
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # kernel 2D par produit extérieur
    kernel2d = gauss[:, None] * gauss[None, :]
    kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)  # [1,1,K,K]
    kernel2d = kernel2d.to(x.device, dtype=x.dtype)

    # padding "same"
    pad = kernel_size // 2
    x = F.conv2d(x, kernel2d, padding=pad)

    if x.shape[0] == 1 and x.shape[1] == 1:
        x = x.squeeze()
    return x

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

def convert_json_to_pose_sequence(anim_data, H=512, W=512, device="cuda", dtype=torch.float16, debug=False):
    """
    Convertit un JSON d'animation OpenPose simplifié en tensor utilisable par ControlNet.

    Output:
        pose_sequence: tensor [num_frames, 3, H, W] (RGB image type)
    """

    frames = anim_data.get("animation", [])
    pose_images = []

    for idx, frame in enumerate(frames):
        keypoints = frame.get("keypoints", [])

        # Image noire
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # --- Dessin des points ---
        for kp in keypoints:
            x = int(kp["x"])
            y = int(kp["y"])
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
                x1, y1 = int(keypoints[a]["x"]), int(keypoints[a]["y"])
                x2, y2 = int(keypoints[b]["x"]), int(keypoints[b]["y"])
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # --- Conversion en tensor ---
        img = torch.from_numpy(canvas).float() / 255.0  # [H, W, C]
        img = img.permute(2, 0, 1)  # → [C, H, W]

        pose_images.append(img)

    pose_sequence = torch.stack(pose_images).to(device=device, dtype=dtype)

    if debug:
        print(f"[JSON->POSE] shape: {pose_sequence.shape}")
        print(f"[JSON->POSE] min/max: {pose_sequence.min().item()} / {pose_sequence.max().item()}")

    return pose_sequence


def apply_controlnet_openpose_step_safe(
    latents,
    timestep,
    unet,
    controlnet,
    scheduler,
    pose_image,
    pos_embeds,
    neg_embeds,
    guidance_scale,
    controlnet_scale=0.25,
    device="cuda",
    dtype=torch.float16,
    debug=False
):
    """
    Wrapper sécurisé pour apply_controlnet_openpose_step
    - gère CPU/GPU
    - corrige dtype
    - convertit timestep en long pour scheduler
    """
    # --- CPU float32 pour ControlNet ---
    latents_cpu = latents.to("cpu", dtype=torch.float32)
    unet_cpu = unet.to("cpu", dtype=torch.float32)
    controlnet_cpu = controlnet.to("cpu", dtype=torch.float32)
    pose_cpu = pose_image.to("cpu", dtype=torch.float32)
    pos_embeds_cpu = pos_embeds.to("cpu", dtype=torch.float32)
    neg_embeds_cpu = neg_embeds.to("cpu", dtype=torch.float32)

    # --- Préparer timestep ---
    if timestep.ndim == 0:
        timestep = timestep.unsqueeze(0)
    batch_size = latents_cpu.shape[0]
    timestep = timestep.repeat(batch_size).to(torch.long).to("cpu")

    # --- Appel ControlNet OpenPose ---
    latents_cpu = apply_controlnet_openpose_step(
        latents=latents_cpu,
        t=timestep,
        unet=unet_cpu,
        controlnet=controlnet_cpu,
        scheduler=scheduler,
        pose_image=pose_cpu,
        pos_embeds=pos_embeds_cpu,
        neg_embeds=neg_embeds_cpu,
        guidance_scale=guidance_scale,
        controlnet_scale=controlnet_scale,
        device="cpu",
        dtype=torch.float32,
        debug=debug
    )

    # --- Retour sur GPU et dtype final ---
    latents_out = latents_cpu.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)

    return latents_out

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


def match_latent_size_v1(latents_main, latents_mini):
    """
    Assure que latents_mini a la même taille HxW que latents_main.
    """
    if latents_mini.shape[2:] != latents_main.shape[2:]:
        latents_mini = F.interpolate(
            latents_mini,
            size=latents_main.shape[2:],  # H, W
            mode='bilinear',
            align_corners=False
        )
    return latents_mini


# ****************************** A TESTER ********************************


def apply_openpose_tilewise_48(
    latents,
    pose,
    apply_fn,
    block_size=48,
    overlap=16,
    device='cuda',
    debug=False,
    debug_dir=None,
    frame_idx=0,
    full_res=(960, 512)
):
    import torch
    import torch.nn.functional as F
    import os
    import numpy as np
    from PIL import Image

    B, C_lat, H_lat, W_lat = latents.shape

    # 🔹 Redimensionner pose aux latents
    pose_resized = F.interpolate(
        pose, size=(H_lat, W_lat), mode='bilinear', align_corners=False
    ).clamp(0.0, 1.0)

    stride = block_size - overlap

    # 🔹 Padding pour que H et W soient multiples de stride
    pad_h = (stride - H_lat % stride) % stride
    pad_w = (stride - W_lat % stride) % stride
    latents_padded = F.pad(latents, (0, pad_w, 0, pad_h), mode='reflect')
    H_pad, W_pad = H_lat + pad_h, W_lat + pad_w

    latents_out = latents_padded.clone()
    if debug:
        impact_map = torch.zeros((H_pad, W_pad), device=device)

    tile_id = 0
    for i in range(0, H_pad, stride):
        for j in range(0, W_pad, stride):
            # 🔹 Forcer le dernier tile à toucher le bord
            i_start = max(0, min(i, H_pad - block_size))
            j_start = max(0, min(j, W_pad - block_size))
            i_end = i_start + block_size
            j_end = j_start + block_size

            latent_tile = latents_padded[:, :, i_start:i_end, j_start:j_end].to(device=device, dtype=torch.float16)
            tile_coords = (j_start, i_start, j_end, i_end)

            # Apply tile
            try:
                latent_tile_processed = apply_fn(latent_tile, tile_coords)
                latent_tile_processed = latent_tile_processed.clamp(-5.0, 5.0)
            except Exception as e:
                print(f"[WARNING] Tile {tile_id} ({i_start}:{i_end},{j_start}:{j_end}) failed: {e}")
                latent_tile_processed = latent_tile

            # 🔹 Blend overlaps
            blend_mask = torch.ones((1, 1, i_end-i_start, j_end-j_start), device=device)
            fade_h = min(overlap, j_end-j_start)
            fade_v = min(overlap, i_end-i_start)

            # horizontal fade
            if j_start != 0:
                blend_mask[:, :, :, :fade_h] *= torch.linspace(0,1,fade_h, device=device).view(1,1,1,-1)
            if j_end != W_pad:
                blend_mask[:, :, :, -fade_h:] *= torch.linspace(1,0,fade_h, device=device).view(1,1,1,-1)

            # vertical fade
            if i_start != 0:
                blend_mask[:, :, :fade_v, :] *= torch.linspace(0,1,fade_v, device=device).view(1,1,-1,1)
            if i_end != H_pad:
                blend_mask[:, :, -fade_v:, :] *= torch.linspace(1,0,fade_v, device=device).view(1,1,-1,1)

            latents_out[:, :, i_start:i_end, j_start:j_end] = (
                latents_out[:, :, i_start:i_end, j_start:j_end] * (1 - blend_mask) +
                latent_tile_processed * blend_mask
            )

            if debug:
                diff_map = (latent_tile_processed - latent_tile).abs().mean(dim=1)
                impact_map[i_start:i_end, j_start:j_end] += diff_map.squeeze(0)
                print(f"[TILE {tile_id}] diff={diff_map.mean().item():.6f}")

            tile_id += 1

    # 🔹 Retirer padding
    latents_out = latents_out[:, :, :H_lat, :W_lat]

    # 🔹 DEBUG impact map
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        impact_map_full = impact_map[:H_lat, :W_lat].unsqueeze(0).unsqueeze(0)
        impact_map_full = F.interpolate(impact_map_full, size=full_res, mode='bilinear', align_corners=False).squeeze()
        impact_np = impact_map_full.detach().cpu().numpy()
        impact_np = impact_np - impact_np.min()
        if impact_np.max() > 0:
            impact_np = impact_np / impact_np.max()
        Image.fromarray((impact_np*255).astype(np.uint8)).save(
            os.path.join(debug_dir, f"impact_map_{frame_idx:05d}.png")
        )

    return latents_out



def apply_openpose_tilewise_32(
    latents,
    pose,
    apply_fn,
    block_size=32,
    overlap=16,
    device='cuda',
    debug=False,
    debug_dir=None,
    frame_idx=0,
    full_res=(960, 512)
):
    import torch
    import torch.nn.functional as F
    import os
    import numpy as np
    from PIL import Image

    B, C_lat, H_lat, W_lat = latents.shape

    # Redimensionner pose aux latents
    pose_resized = F.interpolate(
        pose, size=(H_lat, W_lat), mode='bilinear', align_corners=False
    ).clamp(0.0, 1.0)

    stride = block_size - overlap

    # 🔹 Padding pour éviter tiles vides
    pad_h = (stride - H_lat % stride) % stride
    pad_w = (stride - W_lat % stride) % stride
    latents_padded = F.pad(latents, (0, pad_w, 0, pad_h), mode='reflect')
    H_pad, W_pad = H_lat + pad_h, W_lat + pad_w

    latents_out = latents_padded.clone()
    if debug:
        impact_map = torch.zeros((H_pad, W_pad), device=device)

    tile_id = 0
    for i in range(0, H_pad, stride):
        for j in range(0, W_pad, stride):
            i_end = min(i + block_size, H_pad)
            j_end = min(j + block_size, W_pad)
            i_start = max(0, i_end - block_size)
            j_start = max(0, j_end - block_size)

            # 🔹 Ignorer les tiles invalides
            if (i_end - i_start) <= 0 or (j_end - j_start) <= 0:
                continue

            latent_tile = latents_padded[:, :, i_start:i_end, j_start:j_end].to(device=device, dtype=torch.float16)
            tile_coords = (j_start, i_start, j_end, i_end)

            # Apply avec fallback
            try:
                latent_tile_processed = apply_fn(latent_tile, tile_coords)
                latent_tile_processed = latent_tile_processed.clamp(-5.0, 5.0)
            except Exception as e:
                print(f"[WARNING] Tile {tile_id} ({i_start}:{i_end},{j_start}:{j_end}) failed: {e}")
                latent_tile_processed = latent_tile

            # 🔹 Blend sur les overlaps
            blend_mask = torch.ones((1, 1, i_end-i_start, j_end-j_start), device=device)
            # horizontal
            if j_start != 0:
                fade = torch.linspace(0, 1, min(overlap, j_end-j_start), device=device).view(1,1,1,-1)
                blend_mask[:, :, :, :fade.shape[-1]] *= fade
            if j_end != W_pad:
                fade = torch.linspace(1, 0, min(overlap, j_end-j_start), device=device).view(1,1,1,-1)
                blend_mask[:, :, :, -fade.shape[-1]:] *= fade
            # vertical
            if i_start != 0:
                fade = torch.linspace(0, 1, min(overlap, i_end-i_start), device=device).view(1,1,-1,1)
                blend_mask[:, :, :fade.shape[-2], :] *= fade
            if i_end != H_pad:
                fade = torch.linspace(1, 0, min(overlap, i_end-i_start), device=device).view(1,1,-1,1)
                blend_mask[:, :, -fade.shape[-2]:, :] *= fade

            latents_out[:, :, i_start:i_end, j_start:j_end] = (
                latents_out[:, :, i_start:i_end, j_start:j_end] * (1 - blend_mask) +
                latent_tile_processed * blend_mask
            )

            if debug:
                diff_map = (latent_tile_processed - latent_tile).abs().mean(dim=1)
                impact_map[i_start:i_end, j_start:j_end] += diff_map.squeeze(0)
                print(f"[TILE {tile_id}] diff={diff_map.mean().item():.6f}")

            tile_id += 1

    # Retirer padding
    latents_out = latents_out[:, :, :H_lat, :W_lat]

    # 🔹 DEBUG impact map
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        impact_map_full = impact_map[:H_lat, :W_lat].unsqueeze(0).unsqueeze(0)
        impact_map_full = F.interpolate(impact_map_full, size=full_res, mode='bilinear', align_corners=False).squeeze()
        impact_np = impact_map_full.detach().cpu().numpy()
        impact_np = impact_np - impact_np.min()
        if impact_np.max() > 0:
            impact_np = impact_np / impact_np.max()
        Image.fromarray((impact_np*255).astype(np.uint8)).save(
            os.path.join(debug_dir, f"impact_map_{frame_idx:05d}.png")
        )

    return latents_out





def apply_openpose_tilewise_scale8(
    latents,
    pose,
    apply_fn,
    block_size=64,
    overlap=32,
    device='cuda',
    debug=False,
    debug_dir=None,
    frame_idx=0,
    full_res=(1024, 512)
):
    import torch
    import torch.nn.functional as F
    import os
    import numpy as np
    from PIL import Image

    B, C, H, W = latents.shape

    stride = block_size - overlap

    # =========================================================
    # 🔥 GAUSSIAN WEIGHT MASK (clé du rendu propre)
    # =========================================================
    def make_weight_mask(size):
        coords = torch.linspace(-1, 1, size, device=device)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        sigma = 0.5
        mask = torch.exp(-(dist**2) / (2 * sigma**2))
        return mask  # [H, W]

    weight_mask_full = make_weight_mask(block_size)  # [bs, bs]

    # =========================================================
    # 🔥 ACCUMULATION BUFFERS
    # =========================================================
    latents_accum = torch.zeros_like(latents, device=device)
    weight_accum = torch.zeros((1, 1, H, W), device=device)

    # DEBUG MAP
    if debug:
        impact_map = torch.zeros((H, W), device=device)

    tiles = []

    # =========================================================
    # 🔥 TILE GENERATION (ROBUSTE)
    # =========================================================
    for i in range(0, H, stride):
        for j in range(0, W, stride):

            i_start = i
            j_start = j
            i_end = i + block_size
            j_end = j + block_size

            if i_end > H:
                i_end = H
                i_start = max(0, H - block_size)

            if j_end > W:
                j_end = W
                j_start = max(0, W - block_size)

            if i_end <= i_start or j_end <= j_start:
                continue

            tiles.append((i_start, i_end, j_start, j_end))

    tiles = list(set(tiles))  # remove duplicates

    # =========================================================
    # 🔥 PROCESS TILES
    # =========================================================
    for tile_id, (i_start, i_end, j_start, j_end) in enumerate(tiles):

        latent_tile = latents[:, :, i_start:i_end, j_start:j_end].to(device)

        tile_coords = (j_start, i_start, j_end, i_end)

        try:
            latent_tile_processed = apply_fn(latent_tile, tile_coords)
        except Exception as e:
            print(f"[WARNING] Tile {tile_id} failed: {e}")
            latent_tile_processed = latent_tile

        # =====================================================
        # 🔥 WEIGHT MASK ADAPTATION (bord)
        # =====================================================
        h = i_end - i_start
        w = j_end - j_start

        weight_mask = weight_mask_full[:h, :w]  # crop si bord
        weight_mask = weight_mask.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]

        # =====================================================
        # 🔥 ACCUMULATION
        # =====================================================
        latents_accum[:, :, i_start:i_end, j_start:j_end] += latent_tile_processed * weight_mask
        weight_accum[:, :, i_start:i_end, j_start:j_end] += weight_mask

        # DEBUG
        if debug:
            diff = (latent_tile_processed - latent_tile).abs().mean().item()
            impact_map[i_start:i_end, j_start:j_end] += diff

    # =========================================================
    # 🔥 NORMALISATION FINALE (clé)
    # =========================================================
    latents_out = latents_accum / (weight_accum + 1e-8)

    # =========================================================
    # 🔹 DEBUG VISU
    # =========================================================
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

        impact_img = (impact_np * 255).astype(np.uint8)
        Image.fromarray(impact_img).save(
            os.path.join(debug_dir, f"impact_map_{frame_idx:05d}.png")
        )

    return latents_out

def apply_openpose_tilewise_good(
    latents,
    pose,
    apply_fn,
    block_size=64,
    overlap=32,
    device='cuda',
    debug=False,
    debug_dir=None,
    frame_idx=0,
    full_res=(1024, 512)  # résolution originale pour l'impact map
):
    import torch
    import torch.nn.functional as F
    import os
    import numpy as np
    from PIL import Image

    B, C_lat, H_lat, W_lat = latents.shape

    # 🔹 Redimensionner le pose à la résolution des latents et clamp
    pose_resized = F.interpolate(
        pose, size=(H_lat, W_lat),
        mode='bilinear', align_corners=False
    ).clamp(0.0, 1.0)

    latents_out = latents.clone()

    # 🔹 DEBUG MAP (impact spatial)
    if debug:
        impact_map = torch.zeros((H_lat, W_lat), device=device)

    stride = block_size - overlap
    tile_id = 0

    for i in range(0, H_lat, stride):
        for j in range(0, W_lat, stride):

            i_end = min(i + block_size, H_lat)
            j_end = min(j + block_size, W_lat)

            i_start = max(0, i_end - block_size)
            j_start = max(0, j_end - block_size)

            latent_tile = latents[:, :, i_start:i_end, j_start:j_end].to(device=device, dtype=torch.float16)
            tile_coords = (j_start, i_start, j_end, i_end)

            # 🔹 APPLY avec fallback en cas de crash
            try:
                latent_tile_processed = apply_fn(latent_tile, tile_coords)
                latent_tile_processed = latent_tile_processed.clamp(-5.0, 5.0)
            except Exception as e:
                print(f"[WARNING] Tile {tile_id} ({i_start}:{i_end},{j_start}:{j_end}) failed: {e}")
                traceback.print_exc()

                latent_tile_processed = latent_tile  # fallback

            # 🔹 DEBUG METRICS
            if debug:
                diff_map = (latent_tile_processed - latent_tile).abs().mean(dim=1)
                impact_map[i_start:i_end, j_start:j_end] += diff_map.squeeze(0)

                diff_mean = diff_map.mean().item()
                min_val = latent_tile_processed.min().item()
                max_val = latent_tile_processed.max().item()
                print(f"[TILE {tile_id}] ({i_start}:{i_end},{j_start}:{j_end}) "
                      f"diff={diff_mean:.6f} min={min_val:.4f} max={max_val:.4f}")

            # 🔹 BLEND au lieu d'overwrite
            latents_out[:, :, i_start:i_end, j_start:j_end] = (
                0.7 * latents_out[:, :, i_start:i_end, j_start:j_end] +
                0.3 * latent_tile_processed
            )

            tile_id += 1

    # 🔹 VISUAL DEBUG GLOBAL
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

        # 🔹 Upsample à la résolution originale
        impact_map_full = F.interpolate(
            impact_map.unsqueeze(0).unsqueeze(0),
            size=full_res,
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # 🔹 Lissage pour réduire le bruit
        impact_map_full = gaussian_blur_tensor(impact_map_full, kernel_size=5, sigma=1.0)

        # 🔹 Normalisation pour visualisation
        impact_np = impact_map_full.detach().cpu().numpy()
        impact_np = impact_np - impact_np.min()
        if impact_np.max() > 0:
            impact_np = impact_np / impact_np.max()
        impact_img = (impact_np * 255).astype(np.uint8)
        impact_img = Image.fromarray(impact_img)

        impact_img.save(os.path.join(debug_dir, f"impact_map_{frame_idx:05d}.png"))
        print(f"[DEBUG] Impact map saved")

    return latents_out

def apply_openpose_tilewise_v2(
    latents,
    pose,
    apply_fn,
    block_size=64,
    overlap=32,
    device='cuda',
    debug=False,
    debug_dir=None,
    frame_idx=0,
    full_res=(960, 512)  # résolution originale pour l'impact map
):
    import torch
    import torch.nn.functional as F
    import os
    import numpy as np
    from PIL import Image

    B, C_lat, H_lat, W_lat = latents.shape

    # 🔹 Redimensionner le pose à la résolution des latents
    pose_resized = F.interpolate(
        pose, size=(H_lat, W_lat),
        mode='bilinear', align_corners=False
    )

    latents_out = latents.clone()

    # 🔹 DEBUG MAP (impact spatial)
    if debug:
        impact_map = torch.zeros((H_lat, W_lat), device=device)

    stride = block_size - overlap
    tile_id = 0

    for i in range(0, H_lat, stride):
        for j in range(0, W_lat, stride):

            i_end = min(i + block_size, H_lat)
            j_end = min(j + block_size, W_lat)

            i_start = max(0, i_end - block_size)
            j_start = max(0, j_end - block_size)

            latent_tile = latents[:, :, i_start:i_end, j_start:j_end]
            tile_coords = (j_start, i_start, j_end, i_end)

            # 🔹 APPLY
            latent_tile_processed = apply_fn(latent_tile, tile_coords)

            # 🔹 DEBUG METRICS
            if debug:
                diff_map = (latent_tile_processed - latent_tile).abs().mean(dim=1)
                impact_map[i_start:i_end, j_start:j_end] += diff_map.squeeze(0)

                diff_mean = diff_map.mean().item()
                min_val = latent_tile_processed.min().item()
                max_val = latent_tile_processed.max().item()
                print(f"[TILE {tile_id}] ({i_start}:{i_end},{j_start}:{j_end}) "
                      f"diff={diff_mean:.6f} min={min_val:.4f} max={max_val:.4f}")

            # 🔹 BLEND au lieu d'overwrite
            latents_out[:, :, i_start:i_end, j_start:j_end] = (
                0.7 * latents_out[:, :, i_start:i_end, j_start:j_end] +
                0.3 * latent_tile_processed
            )

            tile_id += 1

    # 🔹 VISUAL DEBUG GLOBAL
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

        # 🔹 Upsample à la résolution originale
        impact_map_full = F.interpolate(
            impact_map.unsqueeze(0).unsqueeze(0),
            size=full_res,
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # 🔹 Lissage pour réduire le bruit
        impact_map_full = gaussian_blur_tensor(impact_map_full, kernel_size=5, sigma=1.0)

        # 🔹 Normalisation pour visualisation
        impact_np = impact_map_full.detach().cpu().numpy()
        impact_np = impact_np - impact_np.min()
        if impact_np.max() > 0:
            impact_np = impact_np / impact_np.max()
        impact_img = (impact_np * 255).astype(np.uint8)
        impact_img = Image.fromarray(impact_img)

        impact_img.save(os.path.join(debug_dir, f"impact_map_{frame_idx:05d}.png"))
        print(f"[DEBUG] Impact map saved")

    return latents_out


def apply_openpose_tilewise_v1(
    latents,
    pose,
    apply_fn,
    block_size=64,
    overlap=32,
    device='cuda',
    debug=False,
    debug_dir=None,
    frame_idx=0
):
    import torch
    import torch.nn.functional as F
    import os
    import numpy as np
    from PIL import Image

    B, C_lat, H_lat, W_lat = latents.shape

    # 🔹 Redimensionner pose pour matcher les latents
    pose_resized = F.interpolate(
        pose, size=(H_lat, W_lat),
        mode='bilinear', align_corners=False
    )

    latents_out = latents.clone()

    # 🔹 DEBUG MAP (impact spatial)
    if debug:
        impact_map = torch.zeros((H_lat, W_lat), device=device)

    stride = block_size - overlap
    tile_id = 0

    for i in range(0, H_lat, stride):
        for j in range(0, W_lat, stride):
            i_end = min(i + block_size, H_lat)
            j_end = min(j + block_size, W_lat)
            i_start = max(0, i_end - block_size)
            j_start = max(0, j_end - block_size)

            latent_tile = latents[:, :, i_start:i_end, j_start:j_end]
            pose_tile = pose_resized[:, :, i_start:i_end, j_start:j_end]

            tile_coords = (j_start, i_start, j_end, i_end)

            # 🔹 APPLY
            latent_tile_processed = apply_fn(latent_tile, tile_coords)

            # 🔹 DEBUG METRICS
            if debug:
                # Diff par pixel (moyenne sur canaux)
                diff_map = (latent_tile_processed - latent_tile).abs().mean(dim=1).squeeze(0)
                impact_map[i_start:i_end, j_start:j_end] += diff_map

                diff_scalar = diff_map.mean().item()
                min_val = latent_tile_processed.min().item()
                max_val = latent_tile_processed.max().item()

                print(f"[TILE {tile_id}] ({i_start}:{i_end},{j_start}:{j_end}) "
                      f"diff={diff_scalar:.6f} min={min_val:.4f} max={max_val:.4f}")

                # Détecter tile mort
                if max_val == 0 and min_val == 0:
                    print(f"[⚠️ ZERO TILE] {tile_id}")

            # 🔹 BLEND au lieu d'overwrite
            latents_out[:, :, i_start:i_end, j_start:j_end] = (
                0.7 * latents_out[:, :, i_start:i_end, j_start:j_end] +
                0.3 * latent_tile_processed
            )

            tile_id += 1

    # 🔹 VISUAL DEBUG GLOBAL
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

        impact_np = impact_map.detach().float().cpu().numpy()
        # normalisation visuelle
        impact_np = impact_np - impact_np.min()
        if impact_np.max() > 0:
            impact_np = impact_np / impact_np.max()

        impact_img = (impact_np * 255).astype(np.uint8)
        impact_img = Image.fromarray(impact_img)
        impact_img.save(os.path.join(debug_dir, f"impact_map_{frame_idx:05d}.png"))

        print(f"[DEBUG] Impact map saved")

    return latents_out

def apply_openpose_tilewise_old(
    latents,
    pose,
    apply_fn,
    block_size=64,
    overlap=32,
    device='cuda',
    debug=False,
    debug_dir=None,
    frame_idx=0
):
    import torch
    import torch.nn.functional as F
    import os
    import numpy as np
    from PIL import Image

    B, C_lat, H_lat, W_lat = latents.shape

    pose_resized = F.interpolate(
        pose, size=(H_lat, W_lat),
        mode='bilinear', align_corners=False
    )

    latents_out = latents.clone()

    # 🔥 DEBUG MAP (impact spatial)
    if debug:
        impact_map = torch.zeros((H_lat, W_lat), device=device)

    stride = block_size - overlap

    tile_id = 0

    for i in range(0, H_lat, stride):
        for j in range(0, W_lat, stride):

            i_end = min(i + block_size, H_lat)
            j_end = min(j + block_size, W_lat)

            #i_start = i_end - block_size if i_end - i < block_size else i
            #j_start = j_end - block_size if j_end - j < block_size else j

            i_start = max(0, i_end - block_size)
            j_start = max(0, j_end - block_size)

            latent_tile = latents[:, :, i_start:i_end, j_start:j_end]
            pose_tile = pose_resized[:, :, i_start:i_end, j_start:j_end]

            tile_coords = (j_start, i_start, j_end, i_end)

            # 🔹 APPLY
            latent_tile_processed = apply_fn(latent_tile, tile_coords)

            # 🔥 DEBUG METRICS
            if debug:
                diff = (latent_tile_processed - latent_tile).abs().mean().item()
                min_val = latent_tile_processed.min().item()
                max_val = latent_tile_processed.max().item()

                print(f"[TILE {tile_id}] ({i_start}:{i_end},{j_start}:{j_end}) "
                      f"diff={diff:.6f} min={min_val:.4f} max={max_val:.4f}")

                # détecter tile mort
                if max_val == 0 and min_val == 0:
                    print(f"[⚠️ ZERO TILE] {tile_id}")

                # remplir impact map
                impact_map[i_start:i_end, j_start:j_end] += diff

            # 🔥 IMPORTANT → BLEND au lieu d'overwrite
            latents_out[:, :, i_start:i_end, j_start:j_end] = (
                0.7 * latents_out[:, :, i_start:i_end, j_start:j_end] +
                0.3 * latent_tile_processed
            )

            tile_id += 1

    # 🔥 VISUAL DEBUG GLOBAL
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

        impact_np = impact_map.detach().float().cpu().numpy()

        # normalisation visuelle
        impact_np = impact_np - impact_np.min()
        if impact_np.max() > 0:
            impact_np = impact_np / impact_np.max()

        impact_img = (impact_np * 255).astype(np.uint8)
        impact_img = Image.fromarray(impact_img)

        impact_img.save(os.path.join(debug_dir, f"impact_map_{frame_idx:05d}.png"))

        print(f"[DEBUG] Impact map saved")

    return latents_out

def apply_openpose_tilewise_ori(latents, pose, apply_fn, block_size=64, overlap=96, device='cuda'):
    """
    Applique OpenPose tile par tile sur le latent.

    latents : torch.Tensor [B, 4, H_latent, W_latent] (diffusion latents)
    pose    : torch.Tensor [B, 3, H_img, W_img] (full OpenPose)
    apply_fn: fonction qui applique ControlNet sur un tile
    """
    B, C_lat, H_lat, W_lat = latents.shape
    B, C_pose, H_img, W_img = pose.shape

    # Redimensionner le pose pour matcher le latent full size
    pose_resized = F.interpolate(pose, size=(H_lat, W_lat), mode='bilinear', align_corners=False)

    # Créer une copie pour modification
    latents_out = latents.clone()

    stride = block_size - overlap
    for i in range(0, H_lat, stride):
        for j in range(0, W_lat, stride):
            # Limites du tile (clip si dépasse)
            i_end = min(i + block_size, H_lat)
            j_end = min(j + block_size, W_lat)
            i_start = i_end - block_size if i_end - i < block_size else i
            j_start = j_end - block_size if j_end - j < block_size else j

            # Extraire tile latent et tile pose
            latent_tile = latents[:, :, i_start:i_end, j_start:j_end]
            pose_tile = pose_resized[:, :, i_start:i_end, j_start:j_end]

            # Appliquer ControlNet sur le tile
            #latent_tile_processed = apply_fn(latent_tile, pose_tile)
            tile_coords = (j_start, i_start, j_end, i_end)  # ⚠️ ordre important
            latent_tile_processed = apply_fn(latent_tile, tile_coords)

            # Écraser dans latents_out
            latents_out[:, :, i_start:i_end, j_start:j_end] = latent_tile_processed

    return latents_out


def apply_controlnet_openpose_step_safe(
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
    # 🔹 Déplacement sur device / dtype
    latents = latents.to(device=device, dtype=dtype)
    pose_image = pose_image.to(device=device, dtype=dtype)

    # 🔹 Préparer batch pour classifier-free guidance
    if neg_embeds is not None:
        latent_model_input = torch.cat([latents] * 2)
        encoder_states = torch.cat([neg_embeds, pos_embeds])
        pose_input = torch.cat([pose_image] * 2)
    else:
        latent_model_input = latents
        encoder_states = pos_embeds
        pose_input = pose_image

    # 🔹 Scheduler scale
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # 🔹 ControlNet
    down_samples, mid_sample = controlnet(
        latent_model_input,
        t,
        encoder_hidden_states=encoder_states,
        controlnet_cond=pose_input,
        return_dict=False
    )

    # 🔹 Safe normalization des résidus ControlNet
    down_samples = [d / (d.abs().mean() + 1e-6) for d in down_samples]
    mid_sample = mid_sample / (mid_sample.abs().mean() + 1e-6)

    # 🔹 UNet avec ControlNet
    noise_pred = unet(
        latent_model_input,
        t,
        encoder_hidden_states=encoder_states,
        down_block_additional_residuals=[d * controlnet_scale for d in down_samples],
        mid_block_additional_residual=mid_sample * controlnet_scale
    ).sample

    # 🔹 Classifier-Free Guidance
    if neg_embeds is not None:
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

    # 🔹 Scheduler step avec batch safe
    latents_input = torch.cat([latents] * 2) if neg_embeds is not None else latents
    latents = scheduler.step(noise_pred, t, latents_input).prev_sample

    # 🔹 Récupérer batch original si CFG
    if neg_embeds is not None:
        latents = latents.chunk(2)[0]

    # 🔹 Clamp final pour sécurité
    latents = torch.clamp(latents, -1.0, 1.0)

    if debug:
        print(f"[ControlNet SAFE] t={t}, latents min/max: {latents.min().item():.3f}/{latents.max().item():.3f}")

    return latents


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


def apply_controlnet_openpose_step_v1(
    latents,
    t,
    unet,
    controlnet,
    control_tensor,
    pos_embeds=None,
    neg_embeds=None,
    guidance_scale=1.0,
    controlnet_strength=1.0,
    device="cuda",
    debug=False
):
    """
    Applique ControlNet OpenPose sur un step UNet avec CFG.

    Args:
        latents: [B,C,H,W]
        t: timestep
        unet: modèle UNet
        controlnet: modèle ControlNet
        control_tensor: [B,1,H,W] ou [B,3,H,W] (pose image)
        pos_embeds: embeddings positifs
        neg_embeds: embeddings négatifs
        guidance_scale: CFG strength
        controlnet_strength: influence pose
        device: device
        debug: print infos

    Returns:
        latents mis à jour
    """

    # --------------------------------------------------
    # 🔒 Sécurisation inputs
    # --------------------------------------------------
    latents = torch.nan_to_num(latents, 0.0)
    control_tensor = torch.nan_to_num(control_tensor, 0.0)

    control_tensor = control_tensor.clamp(0, 1)

    if control_tensor.shape[1] == 1:
        control_tensor = control_tensor.repeat(1, 3, 1, 1)

    control_tensor = control_tensor.to(device=device, dtype=latents.dtype)

    if debug:
        print(f"[ControlNet] latents: {latents.shape}")
        print(f"[ControlNet] control: {control_tensor.shape}")
        print(f"[ControlNet] timestep: {t}")

    # --------------------------------------------------
    # 🔹 POS PASS
    # --------------------------------------------------
    down_pos, mid_pos = controlnet(
        latents,
        t,
        encoder_hidden_states=pos_embeds,
        controlnet_cond=control_tensor,
        return_dict=False
    )

    down_pos = [d * controlnet_strength for d in down_pos]
    mid_pos = mid_pos * controlnet_strength

    noise_pos = unet(
        latents,
        t,
        encoder_hidden_states=pos_embeds,
        down_block_additional_residuals=down_pos,
        mid_block_additional_residual=mid_pos
    ).sample

    # --------------------------------------------------
    # 🔹 NEG PASS (si CFG)
    # --------------------------------------------------
    if neg_embeds is not None:

        down_neg, mid_neg = controlnet(
            latents,
            t,
            encoder_hidden_states=neg_embeds,
            controlnet_cond=control_tensor,
            return_dict=False
        )

        down_neg = [d * controlnet_strength for d in down_neg]
        mid_neg = mid_neg * controlnet_strength

        noise_neg = unet(
            latents,
            t,
            encoder_hidden_states=neg_embeds,
            down_block_additional_residuals=down_neg,
            mid_block_additional_residual=mid_neg
        ).sample

        # 🔥 CFG
        noise_pred = noise_neg + guidance_scale * (noise_pos - noise_neg)

    else:
        noise_pred = noise_pos

    # --------------------------------------------------
    # 🔹 Update latents (diffusion step simplifié)
    # --------------------------------------------------
    latents = latents + noise_pred * 0.1   # facteur stable (évite explosion)

    # 🔒 Clamp sécurité
    latents = torch.clamp(latents, -1.5, 1.5)

    # --------------------------------------------------
    # 🔍 Debug
    # --------------------------------------------------
    if debug:
        print(f"[ControlNet] noise min/max: {noise_pred.min():.3f}/{noise_pred.max():.3f}")
        print(f"[ControlNet] latents min/max: {latents.min():.3f}/{latents.max():.3f}")

    return latents

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

def apply_openpose_tilewise_good(
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
    scale=8  # 🔹 nouveau paramètre scale
):
    import torch
    import torch.nn.functional as F
    import os
    import numpy as np
    from PIL import Image

    B, C, H, W = latents.shape
    stride = block_size - overlap

    # =========================================================
    # 🔹 GAUSSIAN WEIGHT MASK (clé du rendu propre)
    # =========================================================
    def make_weight_mask(size):
        coords = torch.linspace(-1, 1, size, device=device)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        sigma = 0.5
        mask = torch.exp(-(dist**2) / (2 * sigma**2))
        return mask  # [H, W]

    weight_mask_full = make_weight_mask(block_size)

    # =========================================================
    # 🔹 ACCUMULATION BUFFERS
    # =========================================================
    latents_accum = torch.zeros_like(latents, device=device)
    weight_accum = torch.zeros((1, 1, H, W), device=device)

    if debug:
        impact_map = torch.zeros((H, W), device=device)

    tiles = []

    # =========================================================
    # 🔹 TILE GENERATION
    # =========================================================
    for i in range(0, H, stride):
        for j in range(0, W, stride):
            i_start = i
            j_start = j
            i_end = i + block_size
            j_end = j + block_size

            if i_end > H:
                i_end = H
                i_start = max(0, H - block_size)
            if j_end > W:
                j_end = W
                j_start = max(0, W - block_size)
            if i_end <= i_start or j_end <= j_start:
                continue
            tiles.append((i_start, i_end, j_start, j_end))

    tiles = list(set(tiles))

    # =========================================================
    # 🔹 PROCESS TILES
    # =========================================================
    for tile_id, (i_start, i_end, j_start, j_end) in enumerate(tiles):

        latent_tile = latents[:, :, i_start:i_end, j_start:j_end].to(device)

        # 🔹 scale passé à apply_fn
        tile_coords = (j_start, i_start, j_end, i_end)
        try:
            latent_tile_processed = apply_fn(latent_tile, tile_coords)
        except Exception as e:
            print(f"[WARNING] Tile {tile_id} failed: {e}")
            traceback.print_exc()
            latent_tile_processed = latent_tile

        # =====================================================
        # 🔹 WEIGHT MASK ADAPTATION (bord)
        # =====================================================
        h = i_end - i_start
        w = j_end - j_start
        weight_mask = weight_mask_full[:h, :w].unsqueeze(0).unsqueeze(0)

        # =====================================================
        # 🔹 ACCUMULATION
        # =====================================================
        latents_accum[:, :, i_start:i_end, j_start:j_end] += latent_tile_processed * weight_mask
        weight_accum[:, :, i_start:i_end, j_start:j_end] += weight_mask

        if debug:
            diff = (latent_tile_processed - latent_tile).abs().mean().item()
            impact_map[i_start:i_end, j_start:j_end] += diff

    # =========================================================
    # 🔹 NORMALISATION FINALE
    # =========================================================
    latents_out = latents_accum / (weight_accum + 1e-8)

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

        impact_img = (impact_np * 255).astype(np.uint8)
        Image.fromarray(impact_img).save(
            os.path.join(debug_dir, f"impact_map_{frame_idx:05d}.png")
        )

    return latents_out




import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
import traceback

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

def pad_to_multiple(x, mult=8):
    B, C, H, W = x.shape
    pad_H = (mult - H % mult) % mult
    pad_W = (mult - W % mult) % mult
    if pad_H == 0 and pad_W == 0:
        return x
    return F.pad(x, (0, pad_W, 0, pad_H))  # pad right & bottom

def gaussian_blend_mask(H, W, overlap):
    """Crée un masque gaussien pour fusionner les tiles avec overlap."""
    import numpy as np
    y = np.linspace(-1,1,H)
    x = np.linspace(-1,1,W)
    xv, yv = np.meshgrid(x,y)
    mask = np.exp(-(xv**2 + yv**2) / 0.5)  # ajuste le sigma si nécessaire
    mask = torch.tensor(mask, dtype=torch.float32)
    return mask

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os

def apply_openpose_tilewise_safe_v1(
    latents,
    pose,
    apply_fn,
    block_size=64,
    overlap=32,
    device='cuda',
    debug=False,
    debug_dir=None,
    frame_idx=None
):
    """
    Applique un OpenPose/ControlNet sur des latents en tiles de manière robuste.

    Arguments:
        latents: torch.Tensor [B, C, H, W] - latents à traiter
        pose: torch.Tensor [B, 3, H_full, W_full] - pose ou condition
        apply_fn: fonction à appliquer sur chaque tile (latent_tile, pose_tile)
        block_size: taille de la tile
        overlap: superposition des tiles
        device: 'cuda' ou 'cpu'
        debug: True pour afficher logs
        debug_dir: dossier pour sauvegarder les tiles debug
        frame_idx: index de frame pour debug
    """
    B, C, H, W = latents.shape
    latents_out = torch.zeros_like(latents)
    weight_map = torch.zeros_like(latents)

    stride = block_size - overlap

    for i in range(0, H, stride):
        for j in range(0, W, stride):
            # Coordonnées du tile
            i_end = min(i + block_size, H)
            j_end = min(j + block_size, W)
            tile_h = i_end - i
            tile_w = j_end - j

            latent_tile = latents[:, :, i:i_end, j:j_end]
            pose_tile = pose[:, :, i:i_end, j:j_end]

            # Padding si nécessaire
            pad_h = block_size - tile_h
            pad_w = block_size - tile_w
            if pad_h > 0 or pad_w > 0:
                latent_tile = F.pad(latent_tile, (0, pad_w, 0, pad_h))
                pose_tile = F.pad(pose_tile, (0, pad_w, 0, pad_h))

            try:
                # Application de la tile
                latent_tile_processed = apply_fn(latent_tile, pose_tile)
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Tile failed at ({i},{j}) frame {frame_idx}: {e}")
                latent_tile_processed = latent_tile  # fallback

            # Retirer le padding pour fusion
            latent_tile_processed = latent_tile_processed[:, :, :tile_h, :tile_w]

            # Fusion avec weight_map pour overlap
            latents_out[:, :, i:i_end, j:j_end] += latent_tile_processed
            weight_map[:, :, i:i_end, j:j_end] += 1.0

            # Sauvegarde debug si demandé
            if debug and debug_dir is not None:
                os.makedirs(debug_dir, exist_ok=True)
                tile_save_path = os.path.join(debug_dir, f"tile_{frame_idx}_{i}_{j}.png")
                save_image((latent_tile_processed[0] + 1) / 2, tile_save_path)

    # Moyenne sur les overlaps
    weight_map[weight_map == 0] = 1.0  # éviter division par 0
    latents_out = latents_out / weight_map

    return latents_out



import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

def apply_openpose_tilewise_safe_v1(
    latents,
    pose,
    apply_fn,
    block_size=64,
    overlap=32,
    device='cuda',
    debug=False,
    debug_dir=None,
    frame_idx=None,
    savetile=False
):
    B, C, H, W = latents.shape
    latents_out = torch.zeros_like(latents)
    weight_map = torch.zeros_like(latents)

    stride = block_size - overlap



    for i in range(0, H, stride):
        for j in range(0, W, stride):
            i_end = min(i + block_size, H)
            j_end = min(j + block_size, W)
            tile_h = i_end - i
            tile_w = j_end - j

            latent_tile = latents[:, :, i:i_end, j:j_end]
            pose_tile = pose[:, :, i:i_end, j:j_end]

            pad_h = block_size - tile_h
            pad_w = block_size - tile_w
            if pad_h > 0 or pad_w > 0:
                latent_tile = F.pad(latent_tile, (0, pad_w, 0, pad_h))
                pose_tile = F.pad(pose_tile, (0, pad_w, 0, pad_h))

            try:
                latent_tile_processed = apply_fn(latent_tile, pose_tile)
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Tile failed at ({i},{j}) frame {frame_idx}: {e}")
                latent_tile_processed = latent_tile  # fallback

            # Retirer le padding
            latent_tile_processed = latent_tile_processed[:, :, :tile_h, :tile_w]

            latents_out[:, :, i:i_end, j:j_end] += latent_tile_processed
            weight_map[:, :, i:i_end, j:j_end] += 1.0

            if debug and savetile and debug_dir is not None:
                os.makedirs(debug_dir, exist_ok=True)
                tile_save_path = os.path.join(debug_dir, f"tile_{frame_idx}_{i}_{j}.png")
                save_image((latent_tile_processed[0] + 1) / 2, tile_save_path)

    weight_map[weight_map == 0] = 1.0
    latents_out = latents_out / weight_map

    # --- Debug map global ---
    # --- Debug map global ---
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

        # impact_map = |latents_out - latents| moyenne sur les channels
        impact_map = torch.abs(latents_out - latents).mean(1, keepdim=True)  # shape [B,1,H_latent,W_latent]

        # Redimensionner à la taille finale approximative (latents * 8)
        H_frame, W_frame = latents.shape[2]*8, latents.shape[3]*8
        impact_map_full = F.interpolate(impact_map, size=(H_frame, W_frame), mode='bilinear', align_corners=False)

        # Normalisation 0-255
        impact_np = impact_map_full[0,0].detach().cpu().numpy()
        impact_np -= impact_np.min()
        if impact_np.max() > 0:
            impact_np /= impact_np.max()
        impact_img = (impact_np * 255).astype(np.uint8)

        Image.fromarray(impact_img).save(os.path.join(debug_dir, f"impact_map_{frame_idx:05d}.png"))

    return latents_out



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
    import torch
    import torch.nn.functional as F

    B, C, H_latent, W_latent = latent_tile.shape
    _, _, H_pose, W_pose = pose_tile.shape

    # 🔥 AUTO-DETECTION SCALE (évite tous les bugs)
    scale_y = H_pose / H_latent
    scale_x = W_pose / W_latent

    # Cas 1 : pose déjà en full res (ex: 1024 vs 128 → scale 8)
    if scale_y >= 4:
        pose_resized = F.interpolate(
            pose_tile,
            size=(H_latent * 8, W_latent * 8),
            mode='bilinear',
            align_corners=False
        )

    # Cas 2 : pose en latent (ex: 128x64)
    else:
        pose_resized = F.interpolate(
            pose_tile,
            size=(H_latent * 8, W_latent * 8),
            mode='bilinear',
            align_corners=False
        )

    # --- Préparation embeddings ---
    pos_embeds, neg_embeds = cf_embeds

    latent_tile_fp32 = latent_tile.to(device=device, dtype=torch.float32)
    pose_tile_fp32 = pose_resized.to(device=device, dtype=torch.float32)

    pos_embeds_fp32 = pos_embeds.to(device=device, dtype=torch.float32)
    neg_embeds_fp32 = (
        neg_embeds.to(device=device, dtype=torch.float32)
        if neg_embeds is not None else None
    )

    # --- Scheduler noise ---
    timesteps = scheduler.timesteps
    t = timesteps[min(frame_counter, len(timesteps)-1)]

    noise = torch.randn_like(latent_tile_fp32)
    latent_noisy = scheduler.add_noise(latent_tile_fp32, noise, t)
    latent_noisy = torch.clamp(latent_noisy, -20, 20)

    latent_model_input = scheduler.scale_model_input(latent_noisy, t)

    # --- CFG ---
    if neg_embeds_fp32 is not None:
        latent_model_input = torch.cat([latent_model_input] * 2)
        embeds = torch.cat([neg_embeds_fp32, pos_embeds_fp32])
    else:
        embeds = pos_embeds_fp32

    latent_model_input = latent_model_input.to(target_dtype)
    embeds = embeds.to(target_dtype)
    pose_tile_fp32 = pose_tile_fp32.to(target_dtype)

    # 🔥 DEBUG CRUCIAL
    # print("latent:", latent_model_input.shape, "pose:", pose_tile_fp32.shape)

    # --- ControlNet ---
    down_samples, mid_sample = controlnet(
        latent_model_input,
        t,
        encoder_hidden_states=embeds,
        controlnet_cond=pose_tile_fp32,
        return_dict=False
    )

    # --- UNet ---
    noise_pred = unet(
        latent_model_input,
        t,
        encoder_hidden_states=embeds,
        down_block_additional_residuals=down_samples,
        mid_block_additional_residual=mid_sample,
        return_dict=False
    )[0]

    noise_pred = torch.nan_to_num(noise_pred, nan=0.0, posinf=1.0, neginf=-1.0)

    # --- CFG merge ---
    if neg_embeds_fp32 is not None:
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + current_guidance_scale * (noise_text - noise_uncond)

    latents_out = scheduler.step(noise_pred, t, latent_noisy).prev_sample

    # --- Delta safe ---
    delta = latents_out - latent_noisy
    delta = torch.nan_to_num(delta, nan=0.0, posinf=1.0, neginf=-1.0)
    delta = torch.clamp(delta, -0.25, 0.25)
    delta = delta * controlnet_scale

    return (latent_tile_fp32 + delta).to(target_dtype)




def apply_openpose_tilewise_safe(
    latents,
    pose,
    apply_fn,
    block_size=64,
    overlap=32,
    device='cuda',
    debug=False,
    debug_dir=None,
    frame_idx=None,
    savetile=False
):
    """
    Applique un OpenPose/ControlNet sur des latents en tiles et génère une impact_map
    à la taille finale de l'image (en tenant compte du scale de decoding, ici 4).
    """
    B, C, H, W = latents.shape
    latents_out = torch.zeros_like(latents)
    weight_map = torch.zeros_like(latents)

    stride = block_size - overlap

    for i in range(0, H, stride):
        for j in range(0, W, stride):
            i_end = min(i + block_size, H)
            j_end = min(j + block_size, W)
            tile_h = i_end - i
            tile_w = j_end - j

            latent_tile = latents[:, :, i:i_end, j:j_end]
            pose_tile = pose[:, :, i:i_end, j:j_end]

            pad_h = block_size - tile_h
            pad_w = block_size - tile_w
            if pad_h > 0 or pad_w > 0:
                latent_tile = F.pad(latent_tile, (0, pad_w, 0, pad_h))
                pose_tile = F.pad(pose_tile, (0, pad_w, 0, pad_h))

            try:
                latent_tile_processed = apply_fn(latent_tile, pose_tile)
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Tile failed at ({i},{j}) frame {frame_idx}: {e}")
                latent_tile_processed = latent_tile  # fallback
            finally:
                torch.cuda.empty_cache()

            # Retirer le padding
            latent_tile_processed = latent_tile_processed[:, :, :tile_h, :tile_w]

            latents_out[:, :, i:i_end, j:j_end] += latent_tile_processed
            weight_map[:, :, i:i_end, j:j_end] += 1.0

            # Sauvegarde debug tile si besoin
            if debug and savetile and debug_dir is not None:
                os.makedirs(debug_dir, exist_ok=True)
                tile_save_path = os.path.join(debug_dir, f"tile_{frame_idx}_{i}_{j}.png")
                save_image((latent_tile_processed[0] + 1) / 2, tile_save_path)

    # Moyenne sur les overlaps
    weight_map[weight_map == 0] = 1.0
    latents_out = latents_out / weight_map

    # --- Impact map à la taille finale ---
    if debug and debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

        # différence moyenne sur les canaux
        impact_map = torch.abs(latents_out - latents).mean(1, keepdim=True)  # [B,1,H_latent,W_latent]

        # Redimensionner selon le scale final utilisé dans decode_latents
        final_H, final_W = H * 4, W * 4  # 🔹 scale=4
        impact_map_full = F.interpolate(impact_map, size=(final_H, final_W), mode='bilinear', align_corners=False)

        # Normalisation 0-255
        impact_np = impact_map_full[0,0].detach().cpu().numpy()
        impact_np -= impact_np.min()
        if impact_np.max() > 0:
            impact_np /= impact_np.max()
        impact_img = (impact_np * 255).astype(np.uint8)

        Image.fromarray(impact_img).save(os.path.join(debug_dir, f"impact_map_{frame_idx:05d}.png"))

    return latents_out





