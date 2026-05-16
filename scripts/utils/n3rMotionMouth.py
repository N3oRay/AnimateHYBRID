#n3rMotionMouth.py
import torch
import torch.nn.functional as F
import os
import numpy as np
from .n3rMotionPose_tools import save_impact_map
from .n3rMotionPoseClass import Pose

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
    # Sélectionner les points de la bouche (exemple d'indices pour les coins de la bouche)
    # =========================
    mouth_points_idx = [40, 41, 18]  # Vous pouvez ajuster en fonction des indices exacts de votre modèle.

    # On récupère les points de la bouche et on les empile en un tensor
    mouth_points = torch.stack([pose.get_point(i) for i in mouth_points_idx], dim=1)  # [B, 4, 2]

    # Calculer le centre de la bouche (moyenne des 4 points)
    mouth_center = mouth_points.mean(dim=1)  # [B, 2]

    # =========================
    # Appliquer les déformations en fonction du centre de la bouche
    # =========================
    mouth_center_px = mouth_center * torch.tensor([W-1, H-1], device=device)
    mouth_center_px = mouth_center_px.view(B, 1, 1, 2)

    # Calculer les décalages de la bouche
    grid_mouth = grid.clone() - mouth_center_px
    grid_mouth = grid_mouth + delta
    grid_mouth = grid_mouth + mouth_center_px

    # Normaliser les coordonnées du grid pour grid_sample
    grid_mouth[..., 0] = 2.0 * grid_mouth[..., 0] / (W-1) - 1.0
    grid_mouth[..., 1] = 2.0 * grid_mouth[..., 1] / (H-1) - 1.0

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
