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

        # séparation des intentions
        self.motion_head = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.flow_head = nn.Conv2d(hidden, 2, 3, padding=1)
        self.gate_head = nn.Conv2d(hidden, 1, 1)

        self.register_buffer("prev_flow", None)

    def forward(self, x, landmarks):

        h = self.encoder(x)

        l = self.landmark_proj(landmarks)
        l = l[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])

        h = self.fusion(torch.cat([h, l], dim=1))

        motion = self.motion_head(h)

        gate = torch.sigmoid(self.gate_head(motion))

        flow = torch.tanh(self.flow_head(motion))

        flow = flow * gate

        # temporal smoothing inside model
        if self.prev_flow is not None:
            flow = 0.7 * flow + 0.3 * self.prev_flow

        self.prev_flow = flow.detach()

        return {
            "flow": flow,
            "gate": gate
        }

class MouthMotionModel_v1(nn.Module):

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

def build_mask_v1(idx_list, base_scale=0.08):
    pts = torch.stack([pose.get_point(i) for i in idx_list], dim=1)
    pts_px = pts * torch.tensor([W-1, H-1], device=device, dtype=pts.dtype)

    yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
    )

    grid_xy = torch.stack([xx, yy], dim=-1).float().unsqueeze(0)

    dist = torch.norm(
        grid_xy.unsqueeze(1) - pts_px.view(B, len(idx_list), 1, 1, 2),
        dim=-1
    )

    return torch.exp(-dist.min(dim=1).values * base_scale).unsqueeze(-1)


def build_mask(idx_list, pose, W, H, device, base_scale=0.08):

    pts = torch.stack([pose.get_point(i) for i in idx_list], dim=1)  # [B,N,2]

    pts_px = pts * torch.tensor(
        [W - 1, H - 1],
        device=device,
        dtype=pts.dtype
    )

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    grid_xy = torch.stack([xx, yy], dim=-1).float()  # [H,W,2]
    grid_xy = grid_xy.unsqueeze(0)  # [1,H,W,2]

    # reshape pour broadcast propre
    pts_px = pts_px.unsqueeze(2).unsqueeze(2)  # [B,N,1,1,2]

    dist = torch.norm(grid_xy.unsqueeze(1) - pts_px, dim=-1)  # [B,N,H,W]

    min_dist = dist.min(dim=1).values  # [B,H,W]

    mask = torch.exp(-min_dist * base_scale)

    return mask.unsqueeze(-1)

def apply_mouth_smil(
    latents,
    pose,
    mask_mouth,
    grid,
    frame_counter,
    mouth_model,
    H=None,
    W=None,
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
    # =====================================================
    # WIDE SOFT MOUTH FIELD
    # =====================================================

    mask_h = F.max_pool2d(mask, kernel_size=(5, 13), stride=1, padding=(2, 6))
    mask_v = F.avg_pool2d(mask_h, kernel_size=(3, 5), stride=1, padding=(1, 2))

    mask_soft = F.interpolate(
        mask_v,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    # BOOST STRUCTURE (important)
    #mask_soft = torch.pow(mask_soft, 0.7)
    #mask_soft = torch.clamp(mask_soft * 2.5, 0.0, 1.0)

    # BOOST STRUCTURE (important)
    # diffusion douce
    mask_soft = torch.pow(mask_soft, 0.7)

    # amplification
    mask_soft = torch.clamp(mask_soft * 4.0, 0.0, 1.0)
    # =====================================================
    # BCHW -> BHWC
    mask_soft = mask_soft.permute(0, 2, 3, 1).contiguous()

    print("[MASK MAX]", mask_soft.max().item())
    print("[MASK MEAN]", mask_soft.mean().item())


    # =====================================================
    # INERTIA
    # =====================================================
    if not hasattr(apply_mouth_smil, "prev_delta"):
        apply_mouth_smil.prev_delta = torch.zeros_like(delta)

    # inertia only on BASE motion, NOT after temporal
    base_delta = delta

    alpha = 0.55  # beaucoup plus réactif
    smoothed = alpha * base_delta + (1 - alpha) * apply_mouth_smil.prev_delta
    apply_mouth_smil.prev_delta = base_delta.detach()


    apply_mouth_smil.prev_delta = smoothed.detach()

    delta = smoothed

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
    temporal_weight = 0.6 + 0.4 * torch.sin(
        torch.tensor(frame_counter * 0.25, device=delta.device, dtype=delta.dtype)
    )

    #delta = delta + temporal_vec * temporal_weight
    delta = delta + (temporal_vec * temporal_weight * (1.0 + 0.5 * torch.tanh(delta.abs())))

    # =====================================================
    # TEMPORAL DYNAMICS (SAFE)
    # =====================================================

    if not hasattr(apply_mouth_smil, "prev_motion"):
        apply_mouth_smil.prev_motion = torch.zeros_like(delta)

    motion_change = delta - apply_mouth_smil.prev_motion

    motion_energy = torch.norm(
        motion_change,
        dim=-1,
        keepdim=True
    )

    motion_boost = 1.0 + torch.clamp(
        motion_energy * 0.15,
        0.0,
        0.35
    )

    delta = delta * motion_boost

    apply_mouth_smil.prev_motion = delta.detach()

    # =====================================================
    # DEBUG: MOTION ANALYSIS CORE
    # =====================================================

    if not hasattr(apply_mouth_smil, "debug_prev"):
        apply_mouth_smil.debug_prev = torch.zeros_like(delta)

    delta_change = (delta - apply_mouth_smil.debug_prev).abs().mean().item()
    delta_norm = torch.norm(delta, dim=-1).mean().item()

    motion_inertia = (apply_mouth_smil.prev_delta - delta).abs().mean().item()

    temporal_energy = (temporal_vec.abs().mean().item() * float(temporal_weight))

    mask_energy = mask_soft.mean().item()

    gate_energy = motion_gate.mean().item() if motion_gate is not None else 0.0

    print(f"""
    [MOUTH DEBUG]
    frame: {frame_counter}
    delta_mean: {delta.abs().mean().item():.6f}
    delta_max : {delta.abs().max().item():.6f}

    delta_change (t vs t-1): {delta_change:.6f}  <-- IMPORTANT
    delta_norm: {delta_norm:.6f}

    inertia_effect: {motion_inertia:.6f}
    temporal_energy: {temporal_energy:.6f}

    mask_energy: {mask_energy:.3f}
    gate_energy: {gate_energy:.3f}
    """)

    motion_alive = delta_change / (delta_norm + 1e-6)

    print(f"[MOTION ALIVE SCORE] {motion_alive:.6f}")

    apply_mouth_smil.debug_prev = delta.detach().clone()

    # =====================================================
    # CONSTRAINTS (MASK + GATE)
    # =====================================================
    #delta = delta * mask_soft * motion_gate

    combined_gate = mask_soft * (0.7 + 0.3 * motion_gate)
    delta = delta * combined_gate

    if delta.abs().mean().item() < 1e-4:
        print("[WARNING] delta almost zero → motion collapsed by mask/gate/inertia")

    # =====================================================
    # SCALING (ONLY ONCE)
    # =====================================================
    delta = delta * strength

    # =====================================================
    # REGION MASKS (CORE / CORNER / ANCHOR)
    # =====================================================

    core_idx = list(range(70, 84))
    corner_idx = [40, 41]
    anchor_idx = [38, 39]



    core_mask   = build_mask(core_idx, pose, W, H, device,  0.10)
    corner_mask = build_mask(corner_idx, pose, W, H, device, 0.06)
    anchor_mask = build_mask(anchor_idx, pose, W, H, device, 0.04)

    # =====================================================
    # Deblocage des coins
    # =====================================================
    corner_mask = corner_mask * (0.9 + 0.1 * torch.sin(
        torch.tensor(frame_counter * 0.3, device=device, dtype=delta.dtype)
    ))

    # =====================================================
    # COMPUTE DELTA
    # =====================================================
    core_delta   = delta * core_mask
    corner_delta = delta * corner_mask
    anchor_delta = delta * anchor_mask

    # inertia séparée
    if not hasattr(apply_mouth_smil, "corner_prev"):
        apply_mouth_smil.corner_prev = torch.zeros_like(delta)
    if not hasattr(apply_mouth_smil, "anchor_prev"):
        apply_mouth_smil.anchor_prev = torch.zeros_like(delta)

    corner_delta = 0.85 * corner_delta + 0.15 * apply_mouth_smil.corner_prev
    anchor_delta = 0.95 * anchor_delta + 0.05 * apply_mouth_smil.anchor_prev

    apply_mouth_smil.corner_prev = corner_delta.detach()
    apply_mouth_smil.anchor_prev = anchor_delta.detach()

    # recomposition hiérarchique
    delta = delta * (1 - core_mask - corner_mask - anchor_mask) + core_delta + corner_delta + anchor_delta

    # =====================================================
    # UPDATE INERTIA BUFFER
    # =====================================================
    apply_mouth_smil.prev_delta = delta.detach()

    # =====================================================
    # Noise effect anti robot
    # =====================================================
    #noise = torch.randn_like(delta) * 0.02
    #delta = delta + noise * mask_soft
    # =====================================================
    # GRID WARP
    # =====================================================
    base_grid = grid.clone()

    # ensure grid BHWC
    if base_grid.shape[-1] != 2:
        base_grid = base_grid.permute(0, 2, 3, 1).contiguous()


    # =====================================================
    # LIP ARTICULATION FIELD
    # =====================================================

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )

    yy = yy.unsqueeze(0).unsqueeze(-1)
    xx = xx.unsqueeze(0).unsqueeze(-1)

    # séparation verticale lèvres
    upper_field = torch.exp(-((yy + 0.08) ** 2) * 40.0)
    lower_field = torch.exp(-((yy - 0.08) ** 2) * 40.0)

    # dynamique opposée
    lip_field_y = (lower_field - upper_field)

    # léger asymétrique horizontal
    lip_field_x = torch.sin(xx * 3.14) * 0.15

    articulation = torch.cat([
        lip_field_x,
        lip_field_y
    ], dim=-1)

    # modulation temporelle
    articulation_strength = 0.18 + 0.08 * torch.sin(
        torch.tensor(frame_counter * 0.25, device=device)
    )

    delta = delta + articulation * articulation_strength * mask_soft


    # =====================================================
    # RIGIDITY FIELD (CRITICAL FOR SHARPNESS)
    # =====================================================

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )

    yy = yy.unsqueeze(0).unsqueeze(-1)
    xx = xx.unsqueeze(0).unsqueeze(-1)

    # centre bouche = mobile
    center_weight = torch.exp(-(xx**2) * 6.0)

    # lèvres hautes plus rigides
    upper_rigid = torch.exp(-((yy + 0.15) ** 2) * 60.0)

    # lèvres basses mobiles
    lower_mobile = torch.exp(-((yy - 0.10) ** 2) * 30.0)

    # coins rigides
    corner_rigid = 1.0 - torch.exp(-(xx**2) * 2.5)

    # rigidité finale
    rigidity = (
        0.25
        + 0.75 * center_weight * lower_mobile
    )

    rigidity = rigidity * (1.0 - 0.6 * upper_rigid)
    rigidity = rigidity * (1.0 - 0.5 * corner_rigid)

    rigidity = rigidity.clamp(0.1, 1.0)

    delta = delta * rigidity


    # =====================================================
    # LIP COMPRESSION FIELD
    # =====================================================

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )

    yy = yy.unsqueeze(0).unsqueeze(-1)
    xx = xx.unsqueeze(0).unsqueeze(-1)

    # centre horizontal bouche
    center_x = torch.exp(-(xx**2) * 10.0)

    # ligne fermeture lèvres
    lip_line = torch.exp(-(yy**2) * 80.0)

    compression = center_x * lip_line

    # force verticale opposée
    #compress_y = -yy * compression * 0.12

    upper_part = torch.clamp(-yy, 0.0, 1.0)
    lower_part = torch.clamp(yy, 0.0, 1.0)

    compress_y = (
        -upper_part * 0.04
        + lower_part * 0.16
    ) * compression

    print( "[COMPRESS_Y]", compress_y.mean().item(), compress_y.min().item(), compress_y.max().item() )

    compression_field = torch.cat([
        torch.zeros_like(compress_y),
        compress_y
    ], dim=-1)

    delta = delta + compression_field * mask_soft

    # =====================================================
    # LOG
    # =====================================================
    lip_opening = delta[...,1].mean().item()

    upper_motion = delta[...,1][yy.squeeze(-1) < 0].abs().mean().item()
    lower_motion = delta[...,1][yy.squeeze(-1) > 0].abs().mean().item()

    print(f"[LIP OPENING] {lip_opening:.6f}")
    print(f"[UPPER LIP MOTION] {upper_motion:.6f}")
    print(f"[LOWER LIP MOTION] {lower_motion:.6f}")

    # =====================================================
    # GRID
    # =====================================================

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

    grid_shift = (grid_mouth - base_grid).abs().mean().item()

    print(f"[GRID DEBUG] shift mean: {grid_shift:.6f}")

    if grid_shift < 1e-4:
        print("[WARNING] grid is static → no visible deformation")

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

            save_impact_map( latents_out, latents_in, debug_dir, frame_counter, prefix="mouth" )

            if npy:
                np.save( os.path.join(debug_dir, f"mouth_delta_{frame_counter:05d}.npy"), delta.detach().cpu().numpy() )

        except Exception as e:
            print("[WARN] debug failed:", e)

    return latents_out, delta, mouth_points

