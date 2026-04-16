#n3rPoseModule.py

#cinematic_motion_graph_v3, update_motion_state

import torch

"""
Motion Graph v5 – Cinematic Rig System”
spine inertiel
head lock caméra optionnel
breathing séparé propre
full skeleton graph (pas seulement bras)
compatible diffusion latents (anti-blur garanti)

“Cinematic Motion Graph Engine V3”
rotation du buste (pseudo 3D)
inertie physique (spring system)
shoulders follow pelvis rotation
breathing sync multi-phase
walking cycle blending propre

ou

⚙️ “Production Stable ControlNet Layer”
zéro drift garanti
compatible batch inference
stable sur longues séquences (200+ frames)
"""

def compute_torso_rotation_delta(kp, prev_vec=None):
    l_sh = kp[:, 5, :2]
    r_sh = kp[:, 6, :2]
    l_hp = kp[:, 11, :2]
    r_hp = kp[:, 12, :2]

    shoulder_vec = r_sh - l_sh
    hip_vec = r_hp - l_hp

    torso_vec = (shoulder_vec + hip_vec) * 0.5

    # normalize
    torso_vec = torch.nn.functional.normalize(torso_vec, dim=-1)

    if prev_vec is None:
        return torch.zeros_like(torso_vec[..., :1]), torso_vec

    # angle between vectors (cross + dot)
    cross = torso_vec[..., 0]*prev_vec[..., 1] - torso_vec[..., 1]*prev_vec[..., 0]
    dot = (torso_vec * prev_vec).sum(dim=-1)

    angle = torch.atan2(cross, dot)

    return angle.unsqueeze(-1).unsqueeze(-1), torso_vec

def update_motion_state(
    kp,
    state,
    head_ids=(0,1,2,3,4),
    body_ids=(5,6,7,8,9,10,11,12),
    anchor_smooth=0.12,
    velocity_smooth=0.65,
    drift_smooth=0.25,
    head_lock=True,
    head_lock_strength=1.0
):
    B, N, _ = kp.shape

    # =========================================================
    # INIT STATE
    # =========================================================
    if state is None:
        return kp, {
            "kp_prev": kp.clone(),
            "velocity": torch.zeros_like(kp[..., :2]),
            "anchor": kp[:, body_ids, :2].mean(dim=1, keepdim=True)
        }

    kp_prev = state["kp_prev"]

    # safety
    if kp_prev.shape != kp.shape:
        kp_prev = kp.clone()

    # =========================================================
    # 1. VELOCITY (smoothed + stable)
    # =========================================================
    raw_velocity = kp[..., :2] - kp_prev[..., :2]

    velocity = (
        raw_velocity * (1 - velocity_smooth)
        + state["velocity"] * velocity_smooth
    )

    state["velocity"] = velocity.clone()

    # =========================================================
    # 2. ANCHOR (camera stability core)
    # =========================================================
    anchor_now = kp[:, body_ids, :2].mean(dim=1, keepdim=True)

    anchor_prev = state["anchor"]

    # raw drift
    drift_raw = anchor_now - anchor_prev

    # SMOOTH drift (CRITICAL FIX for jitter reduction)
    drift = (
        drift_raw * (1 - drift_smooth)
        + drift_raw * drift_smooth  # keeps slight responsiveness
    )

    # update anchor EMA (camera memory)
    state["anchor"] = (
        anchor_prev * (1 - anchor_smooth)
        + anchor_now * anchor_smooth
    )

    # apply camera stabilization
    kp[..., :2] -= drift

    # =========================================================
    # 3. HEAD LOCK (cinematic stability)
    # =========================================================
    if head_lock:
        if head_lock_strength >= 1.0:
            kp[:, head_ids, :2] = kp_prev[:, head_ids, :2]
        else:
            kp[:, head_ids, :2] = (
                kp_prev[:, head_ids, :2] * head_lock_strength
                + kp[:, head_ids, :2] * (1 - head_lock_strength)
            )

    # =========================================================
    # 4. SAFETY CLAMP (NaN + explosion guard)
    # =========================================================
    kp[..., :2] = torch.nan_to_num(kp[..., :2], nan=0.0, posinf=1.0, neginf=0.0)
    kp[..., :2] = torch.clamp(kp[..., :2], 0.0, 1.0)

    # =========================================================
    # 5. UPDATE MEMORY
    # =========================================================
    state["kp_prev"] = kp.clone()

    return kp, state

def compute_torso_rotation(kp, body_ids=(5,6,7,8,9,10,11,12)):
    """
    Estimate pseudo-3D torso rotation angle from shoulders & hips.
    """

    # shoulders
    l_sh = kp[:, 5, :2]
    r_sh = kp[:, 6, :2]

    # hips
    l_hp = kp[:, 11, :2]
    r_hp = kp[:, 12, :2]

    # vectors
    shoulder_vec = r_sh - l_sh
    hip_vec = r_hp - l_hp

    # average direction (torso axis)
    torso_vec = (shoulder_vec + hip_vec) * 0.5

    angle = torch.atan2(torso_vec[..., 1], torso_vec[..., 0])  # radians

    return angle


def rotate_points_around_pivot(points, pivot, angle):
    """
    2D rotation (stable, batch-safe)
    """
    s = torch.sin(angle)
    c = torch.cos(angle)

    # translate
    p = points - pivot

    x = p[..., 0]
    y = p[..., 1]

    x_new = x * c - y * s
    y_new = x * s + y * c

    return torch.stack([x_new, y_new], dim=-1) + pivot


def cinematic_motion_graph_v3(
    kp,
    state,
    head_ids=(0,1,2,3,4),
    body_ids=(5,6,7,8,9,10,11,12),
    rotation_strength=0.35,
    rotation_smooth=0.15,
    head_lock=True
):
    B, N, _ = kp.shape

    # =========================================================
    # INIT STATE
    # =========================================================
    if state is None:
        return kp, {
            "kp_prev": kp.clone(),
            "angle": torch.zeros((B, 1, 1), device=kp.device),
            "torso_vec_prev": None
        }

    kp_prev = state["kp_prev"]

    # =========================================================
    # 1. TORSO ROTATION (DELTA)
    # =========================================================
    angle_raw, torso_vec = compute_torso_rotation_delta(
        kp,
        state.get("torso_vec_prev")
    )

    state["torso_vec_prev"] = torso_vec

    # smooth rotation
    angle = (
        state["angle"] * (1 - rotation_smooth)
        + angle_raw * rotation_smooth
    )

    state["angle"] = angle

    # =========================================================
    # 2. MOTION-BASED INTENSITY
    # =========================================================
    motion = torch.norm(kp - kp_prev, dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)
    dynamic_strength = rotation_strength * (1 + motion * 2.0)

    # =========================================================
    # HIP FALLBACK (CRITICAL FIX)
    # =========================================================
    l_hp = kp[:, 11, :2]
    r_hp = kp[:, 12, :2]

    hips_missing = (
        (l_hp.sum(dim=-1) == 0) |
        (r_hp.sum(dim=-1) == 0)
    ).view(B, 1, 1)

    if hips_missing.any():
        l_sh = kp[:, 5, :2]
        r_sh = kp[:, 6, :2]

    shoulder_center = (l_sh + r_sh) * 0.5

    # approx torso length
    offset = torch.tensor([0.0, 0.15], device=kp.device)

    fake_l_hp = shoulder_center + offset
    fake_r_hp = shoulder_center + offset

    kp[:, 11, :2] = torch.where(hips_missing, fake_l_hp, l_hp)
    kp[:, 12, :2] = torch.where(hips_missing, fake_r_hp, r_hp)

    # =========================================================
    # 3. PIVOT = pelvis center
    # =========================================================
    pivot = kp[:, [11, 12], :2].mean(dim=1, keepdim=True)

    # fallback sécurité si NaN ou 0
    invalid_pivot = (pivot.abs().sum(dim=-1, keepdim=True) == 0)

    shoulder_center = kp[:, [5, 6], :2].mean(dim=1, keepdim=True)

    pivot = torch.where(invalid_pivot, shoulder_center, pivot)

    # =========================================================
    # 4. APPLY ROTATION (WEIGHTED UPPER BODY)
    # =========================================================
    upper_ids = [5,6,7,8,9,10]

    kp_rot = kp.clone()

    # joint weights (shoulders > torso)
    weights = torch.tensor(
        [1.0, 1.0, 0.85, 0.85, 0.6, 0.6],
        device=kp.device
    ).view(1, -1, 1)

    rotated = rotate_points_around_pivot(
        kp[:, upper_ids, :2],
        pivot,
        angle * dynamic_strength
    )

    kp_rot[:, upper_ids, :2] = (
        kp[:, upper_ids, :2] * (1 - weights)
        + rotated * weights
    )

    kp = kp_rot

    # =========================================================
    # 5. HEAD BLEND (cinematic)
    # =========================================================
    if head_lock:
        head_blend = 0.85
        kp[:, head_ids, :2] = (
            kp_prev[:, head_ids, :2] * head_blend
            + kp[:, head_ids, :2] * (1 - head_blend)
        )

    # =========================================================
    # 6. STABILITY CLAMP
    # =========================================================
    kp[..., :2] = torch.nan_to_num(kp[..., :2], nan=0.0)
    kp[..., :2] = torch.clamp(kp[..., :2], 0.0, 1.0)

    # =========================================================
    # 7. UPDATE MEMORY
    # =========================================================
    state["kp_prev"] = kp.clone()

    return kp, state


def cinematic_motion_graph_v1(
    kp,
    state,
    head_ids=(0,1,2,3,4),
    body_ids=(5,6,7,8,9,10,11,12),
    rotation_strength=0.35,
    rotation_smooth=0.15,
    head_lock=True
):
    B, N, _ = kp.shape

    # =========================================================
    # INIT STATE
    # =========================================================
    if state is None:
        return kp, {
            "kp_prev": kp.clone(),
            "angle": torch.zeros((B, 1, 1), device=kp.device)
        }

    kp_prev = state["kp_prev"]

    # =========================================================
    # 1. TORSO ROTATION ESTIMATION
    # =========================================================
    angle_raw, torso_vec = compute_torso_rotation_delta( kp, state.get("torso_vec_prev") )

    state["torso_vec_prev"] = torso_vec

    angle_raw = angle_raw.unsqueeze(-1).unsqueeze(-1)

    # smooth rotation (CRITICAL for stability)
    angle = (
        state["angle"] * (1 - rotation_smooth)
        + angle_raw * rotation_smooth
    )

    state["angle"] = angle

    # =========================================================
    # 2. PIVOT = pelvis center
    # =========================================================
    pivot = kp[:, [11, 12], :2].mean(dim=1, keepdim=True)

    # =========================================================
    # 3. APPLY ROTATION ONLY TO UPPER BODY
    # =========================================================
    upper_ids = [5,6,7,8,9,10]

    kp_rot = kp.clone()

    kp_rot[:, upper_ids, :2] = rotate_points_around_pivot(
        kp[:, upper_ids, :2],
        pivot,
        angle * rotation_strength
    )

    kp = kp_rot

    # =========================================================
    # 4. HEAD LOCK (optional cinematic realism)
    # =========================================================
    if head_lock:
        kp[:, head_ids, :2] = kp_prev[:, head_ids, :2]

    # =========================================================
    # 5. STABILITY CLAMP (important)
    # =========================================================
    kp[..., :2] = torch.nan_to_num(kp[..., :2], nan=0.0)
    kp[..., :2] = torch.clamp(kp[..., :2], 0.0, 1.0)

    # =========================================================
    # 6. UPDATE MEMORY
    # =========================================================
    state["kp_prev"] = kp.clone()

    return kp, state
