#n3rMotionMouth.py
import numpy as np
import os
import datetime
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .tools_utils import ensure_4_channels, log_debug, sanitize_latents
from .n3r_EMA import motion_aware_ema_fusion, compute_high_freq_energy


from .n3rMotionPose_tools import save_impact_map
from .n3rMotionPoseClass import Pose


def apply_mouth_smil_old(
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


    mouth_points_idx = [
            40, 41,
            70,71,72,73,74,75,76,
            77,78,79,80,81,82,83
        ]

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

""""
        38: ("mouth_left_ext", mouth_left_ext),
        39: ("mouth_right_ext", mouth_right_ext),

        40: ("mouth_left", mouth_left),
        41: ("mouth_right", mouth_right),
        70: ("mouth_top_mid_r3", mouth_top_mid_r3), #OK
        71: ("mouth_top_mid_r2", mouth_top_mid_r2), #OK
        72: ("mouth_top_mid_r1", mouth_top_mid_r1), #OK
        73: ("mouth_top_mid", mouth_top_mid), #OK
        74: ("mouth_top_mid_l1", mouth_top_mid_l1), #OK
        75: ("mouth_top_mid_l2", mouth_top_mid_l2), #OK
        76: ("mouth_top_mid_l3", mouth_top_mid_l3), #OK

        77: ("mouth_bot_mid_r3", mouth_bot_mid_r3), #OK
        78: ("mouth_bot_mid_r2", mouth_bot_mid_r2), #OK
        79: ("mouth_bot_mid_r1", mouth_bot_mid_r1), #OK
        80: ("mouth_bot_mid", mouth_bot_mid), #OK
        81: ("mouth_bot_mid_l1", mouth_bot_mid_l1), #OK
        82: ("mouth_bot_mid_l2", mouth_bot_mid_l2), #OK
        83: ("mouth_bot_mid_l3", mouth_bot_mid_l3), #OK


        mouth_points_idx = [
            40, 41,
            70,71,72,73,74,75,76,
            77,78,79,80,81,82,83
        ]

"""

class MouthMotionModel(nn.Module):

    def __init__(self, in_channels=4, hidden=32):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
        )

        self.landmark_proj = nn.Linear(32, hidden)

        self.fusion = nn.Conv2d(hidden * 2, hidden, 1)

        self.motion_gate = nn.Conv2d(hidden, 1, 1)

        self.flow_head = nn.Conv2d(hidden, 2, 3, padding=1)

    def forward(self, x, landmarks):

        h = self.encoder(x)

        l = self.landmark_proj(landmarks)
        l = l[:,:,None,None].expand(-1,-1,h.shape[2],h.shape[3])

        h = self.fusion(torch.cat([h,l], dim=1))

        gate = torch.sigmoid(self.motion_gate(h))

        flow = torch.tanh(self.flow_head(h))

        flow = flow * gate

        return {
            "flow": flow,
            "gate": gate
        }




# instance par défaut pour ton pipeline
mouth_model = MouthMotionModel().cuda()

def apply_mouth_smil(
    latents,
    pose,
    mask_mouth,
    grid,
    mouth_model,
    H=None,
    W=None,
    frame_counter=0,
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

    # =====================================================
    # LANDMARKS
    # =====================================================
    mouth_points_idx = [
        40, 41,
        70,71,72,73,74,75,76,
        77,78,79,80,81,82,83
    ]

    mouth_points = torch.stack(
        [pose.get_point(i) for i in mouth_points_idx],
        dim=1
    )

    mouth_center = mouth_points.mean(dim=1)

    scale_tensor = torch.tensor(
        [W - 1, H - 1],
        device=device,
        dtype=latents.dtype
    )

    mouth_center_px = (mouth_center * scale_tensor).view(B, 1, 1, 2)

    # =====================================================
    # DELTA
    # =====================================================
    if mouth_model is None:

        print("[MOTION MOUTH] ANALYTIC DELTA ✅")

        delta, _ = Pose.compute_mouth_delta(
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

        # assume BCHW -> convert to BHWC
        if delta.shape[1] == 2:
            delta = delta.permute(0, 2, 3, 1).contiguous()

        motion_gate = torch.ones((B, H, W, 1), device=device)

    else:

        print("[MOTION MOUTH] NEURAL DELTA ✅")

        landmarks = mouth_points.reshape(B, -1)

        pred = mouth_model(latents, landmarks)

        # BCHW -> BHWC
        delta = pred["flow"].permute(0, 2, 3, 1).contiguous()
        motion_gate = pred["gate"].permute(0, 2, 3, 1).contiguous()

        # scale flow
        delta[..., 0] *= W * 0.08
        delta[..., 1] *= H * 0.08


    # =====================================================
    # MASK PROCESSING
    # =====================================================
    if mask_mouth.dim() == 3:
        mask = mask_mouth.unsqueeze(1)
    else:
        mask = mask_mouth

    mask = mask.float()

    # anisotropic dilation (good mouth behavior)
    mask_h = F.max_pool2d(mask, kernel_size=(5, 13), stride=1, padding=(2, 6))
    mask_v = F.avg_pool2d(mask_h, kernel_size=(3, 5), stride=1, padding=(1, 2))

    mask_soft = F.interpolate(
        mask_v,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    # BCHW -> BHWC
    mask_soft = mask_soft.permute(0, 2, 3, 1).contiguous()


    # =====================================================
    # INERTIA
    # =====================================================
    if not hasattr(apply_mouth_smil, "prev_delta"):
        apply_mouth_smil.prev_delta = torch.zeros_like(delta)

    delta = 0.85 * delta + 0.15 * apply_mouth_smil.prev_delta

    # =====================================================
    # TEMPORAL (APPLY BEFORE MASK)
    # =====================================================
    t = torch.tensor(frame_counter / 10.0, device=delta.device, dtype=delta.dtype)

    sin_t = torch.sin(t)
    cos_t = torch.cos(t * 0.8)

    temporal_vec = torch.stack([
        torch.full_like(delta[..., 0], sin_t),
        torch.full_like(delta[..., 1], cos_t)
    ], dim=-1)

    # Occilation constantes
    #delta = delta + temporal_vec * 0.7
    # Occilation variables
    temporal_weight = 0.25 + 0.15 * torch.sin(
        torch.tensor(frame_counter * 0.2, device=delta.device, dtype=delta.dtype)
    )

    delta = delta + temporal_vec * temporal_weight

    # =====================================================
    # CONSTRAINTS (MASK + GATE)
    # =====================================================
    #delta = delta * mask_soft * motion_gate

    combined_gate = mask_soft * (0.7 + 0.3 * motion_gate)
    delta = delta * combined_gate

    # =====================================================
    # SCALING (ONLY ONCE)
    # =====================================================
    delta = delta * strength

    # =====================================================
    # NEW CODE
    # =====================================================
    norm = torch.sqrt(delta[..., 0]**2 + delta[..., 1]**2 + 1e-8)
    scale = torch.tanh(norm * 0.8) / (norm + 1e-8)

    delta = delta * scale.unsqueeze(-1)

    # =====================================================
    # UPDATE INERTIA BUFFER
    # =====================================================
    apply_mouth_smil.prev_delta = delta.detach()

    # =====================================================
    # GRID WARP
    # =====================================================
    base_grid = grid.clone()

    # ensure grid BHWC
    if base_grid.shape[-1] != 2:
        base_grid = base_grid.permute(0, 2, 3, 1).contiguous()

    grid_mouth = base_grid + delta

    # normalize
    grid_norm = grid_mouth.clone()
    grid_norm[..., 0] = 2.0 * grid_norm[..., 0] / (W - 1) - 1.0
    grid_norm[..., 1] = 2.0 * grid_norm[..., 1] / (H - 1) - 1.0

    grid_norm = torch.clamp(grid_norm, -1.2, 1.2)

    # =====================================================
    # GRID SAMPLE
    # =====================================================

    if grid_norm.shape[-1] != 2:
        grid_norm = grid_norm.permute(0, 2, 3, 1).contiguous()

    latents_out = F.grid_sample(
        latents,
        grid_norm,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    # =====================================================
    # BLENDING (SAFE)
    # =====================================================
    blend = mask_soft.permute(0, 3, 1, 2).contiguous()

    if blend.shape[-2:] != latents.shape[-2:]:
        blend = F.interpolate(
            blend,
            size=latents.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    latents_out = latents * (1.0 - blend) + latents_out * blend

    # =====================================================
    # DEBUG
    # =====================================================
    delta_mean = delta.abs().mean().item()
    delta_max = delta.abs().max().item()

    print("[MOTION MOUTH] delta mean:", delta_mean)
    print("[MOTION MOUTH] delta max :", delta_max)

    if debug and debug_dir is not None:
        try:
            os.makedirs(debug_dir, exist_ok=True)

            save_impact_map(
                latents_out,
                latents_in,
                debug_dir,
                frame_counter,
                prefix="mouth"
            )

            if npy:
                np.save(
                    os.path.join(debug_dir, f"mouth_delta_{frame_counter:05d}.npy"),
                    delta.detach().cpu().numpy()
                )

        except Exception as e:
            print("[WARN] debug failed:", e)

    return latents_out, delta, mouth_points

