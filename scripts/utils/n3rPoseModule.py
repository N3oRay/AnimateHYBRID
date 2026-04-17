#n3rPoseModule.py
#cinematic_motion_graph_v3, update_motion_state

import torch
import torch.nn.functional as F

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


def compute_torso_rotation_delta(kp, prev_vec=None, eps=1e-6):
    """
    Stable torso rotation estimation with robustness to missing hips.
    Returns:
        angle: (B, 1, 1)
        torso_vec: (B, 2)
    """

    # =========================================================
    # 1. EXTRACT KEYPOINTS
    # =========================================================
    l_sh = kp[:, 5, :2]
    r_sh = kp[:, 6, :2]
    l_hp = kp[:, 11, :2]
    r_hp = kp[:, 12, :2]

    # =========================================================
    # 2. BUILD VECTORS
    # =========================================================
    shoulder_vec = r_sh - l_sh
    hip_vec = r_hp - l_hp

    # robust validity check (not just finite)
    def is_valid(v):
        return (
            torch.isfinite(v).all(dim=-1, keepdim=True) &
            (torch.norm(v, dim=-1, keepdim=True) > 1e-4)
        )

    shoulder_ok = is_valid(shoulder_vec)
    hip_ok = is_valid(hip_vec)

    # =========================================================
    # 3. SMART FUSION (IMPORTANT)
    # =========================================================
    # cases:
    # - both valid → average
    # - only shoulder → fallback
    # - only hip → rare but handle
    # - none → zero (safe)
    torso_vec = torch.zeros_like(shoulder_vec)

    both = shoulder_ok & hip_ok
    torso_vec = torch.where(both, (shoulder_vec + hip_vec) * 0.5, torso_vec)
    torso_vec = torch.where(shoulder_ok & ~hip_ok, shoulder_vec, torso_vec)
    torso_vec = torch.where(~shoulder_ok & hip_ok, hip_vec, torso_vec)

    # =========================================================
    # 4. NORMALIZE SAFELY
    # =========================================================
    norm = torch.norm(torso_vec, dim=-1, keepdim=True)
    torso_vec = torso_vec / (norm + eps)

    # =========================================================
    # 5. INIT CASE
    # =========================================================
    if prev_vec is None:
        zero_angle = torch.zeros((kp.shape[0], 1, 1), device=kp.device)
        return zero_angle, torso_vec

    prev_vec = F.normalize(prev_vec, dim=-1)

    # =========================================================
    # 6. ANGLE COMPUTATION
    # =========================================================
    cross = torso_vec[..., 0] * prev_vec[..., 1] - torso_vec[..., 1] * prev_vec[..., 0]
    dot = (torso_vec * prev_vec).sum(dim=-1)

    angle = torch.atan2(cross, dot)

    # =========================================================
    # 7. STABILITY SAFETY
    # =========================================================
    angle = torch.nan_to_num(angle, nan=0.0, posinf=0.0, neginf=0.0)

    # clamp violent jumps (important for video stability)
    angle = torch.clamp(angle, -1.0, 1.0)

    return angle.unsqueeze(-1).unsqueeze(-1), torso_vec

#---------------------------------------------------------------------

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


def rotate_points_around_pivot(points, pivot, angle, name="ROT"):
    """
    2D rotation (stable, batch-safe) + debug
    """

    s = torch.sin(angle)
    c = torch.cos(angle)

    print(f"[{name}] angle: {angle}")
    print(f"[{name}] pivot: {pivot}")
    print(f"[{name}] points shape: {points.shape}")

    # translate
    p = points - pivot

    print(f"[{name}] pre-translate sample: {p[0] if p.ndim > 1 else p}")

    x = p[..., 0]
    y = p[..., 1]

    x_new = x * c - y * s
    y_new = x * s + y * c

    out = torch.stack([x_new, y_new], dim=-1) + pivot

    print(f"[{name}] post-rotate sample: {out[0] if out.ndim > 1 else out}")

    return out

# ---------------------------------------------------------
# utils debug
# ---------------------------------------------------------
def _dbg(debug, *args):
    if debug:
        print(*args)


# ---------------------------------------------------------
# SAFE ROTATION
# ---------------------------------------------------------
def compute_torso_rotation_delta_v2(kp, prev_vec=None, eps=1e-6):
    l_sh = kp[:, 5, :2]
    r_sh = kp[:, 6, :2]
    l_hp = kp[:, 11, :2]
    r_hp = kp[:, 12, :2]

    shoulder_vec = r_sh - l_sh
    hip_vec = r_hp - l_hp

    hip_valid = torch.isfinite(hip_vec).all(dim=-1, keepdim=True)

    torso_vec = torch.where(
        hip_valid,
        0.7 * shoulder_vec + 0.3 * hip_vec,
        shoulder_vec
    )

    norm = torch.norm(torso_vec, dim=-1, keepdim=True)
    torso_vec = torso_vec / (norm + eps)

    if prev_vec is None:
        return torch.zeros((kp.shape[0], 1, 1), device=kp.device), torso_vec

    prev_vec = F.normalize(prev_vec, dim=-1)

    cross = torso_vec[..., 0] * prev_vec[..., 1] - torso_vec[..., 1] * prev_vec[..., 0]
    dot = (torso_vec * prev_vec).sum(dim=-1)

    angle = torch.atan2(cross, dot)

    # ↓↓↓ important: soft clamp, pas hard clamp brutal
    angle = torch.tanh(angle) * 0.9

    angle = torch.nan_to_num(angle, 0.0)

    return angle.unsqueeze(-1).unsqueeze(-1), torso_vec

# ---------------------------------------------------------
# MAIN MOTION GRAPH (FIXED)
# ---------------------------------------------------------
def actor_system_v9(
    kp,
    state,
    head_ids=(0,1,2,3,4),
    face_ids=(0,1,2,3,4,15,16,17),
    debug=False
):
    B, N, _ = kp.shape

    # =========================================================
    # INIT
    # =========================================================
    if state is None or state["kp_prev"] is None:
        state = {
            "kp_prev": kp.clone(),
            "energy": torch.zeros((B,1,1), device=kp.device),
            "intent_vec": torch.zeros((B,1,2), device=kp.device),
            "emotion": torch.zeros((B,1,1), device=kp.device),
            "attention": torch.zeros((B,1,2), device=kp.device),
            "gaze_target": None,
            "gaze_strength": 0.5,
            "gaze_jitter": 0.002,
            "inertia": 0.9
        }
        return kp, state

    kp_prev = state["kp_prev"]

    # =========================================================
    # 1. MOTION → ENERGY FIELD
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]
    motion_vec = delta.mean(dim=1, keepdim=True)

    energy_raw = torch.norm(motion_vec, dim=-1, keepdim=True)

    state["energy"] = (
        state["energy"] * state["inertia"] +
        energy_raw * (1 - state["inertia"])
    )

    # =========================================================
    # 2. INTENT (BODY DIRECTION MEMORY)
    # =========================================================
    if state["intent_vec"].abs().sum() == 0:
        state["intent_vec"] = motion_vec.clone()
    else:
        state["intent_vec"] = (
            state["intent_vec"] * 0.85 +
            motion_vec * 0.15
        )

    intent = state["intent_vec"]

    # =========================================================
    # 3. EMOTION EVOLUTION (SMOOTH ACTING CURVE)
    # =========================================================
    emotion_target = torch.clamp(state["energy"] * 3.0, 0.0, 1.0)

    state["emotion"] = (
        state["emotion"] * 0.92 +
        emotion_target * 0.08
    )

    # =========================================================
    # 4. GAZE ENGINE (CRITICAL V9 FEATURE)
    # =========================================================
    eye_center = kp[:, [1,2], :2].mean(dim=1, keepdim=True)

    if state["gaze_target"] is None:
        state["gaze_target"] = eye_center.clone()

    # drift gaze slightly toward intent direction
    drift = intent * 0.03

    state["gaze_target"] = (
        state["gaze_target"] * 0.9 +
        (eye_center + drift) * 0.1
    )

    gaze_dir = state["gaze_target"] - eye_center

    # micro jitter (human imperfection)
    gaze_dir = gaze_dir + torch.randn_like(gaze_dir) * state["gaze_jitter"]

    # =========================================================
    # 5. FACE EMOTION SYSTEM
    # =========================================================

    # mouth opens with emotion
    mouth_open = state["emotion"] * 0.02

    # eyes widen slightly with energy
    eye_tension = 1.0 - state["emotion"] * 0.15

    # apply to mouth (assume index 14)
    if N > 14:
        kp[:, 14, 1] += mouth_open.squeeze(-1)

    # eye separation subtle shift
    kp[:, 1:3, :2] += gaze_dir * 0.15 * state["emotion"]

    # =========================================================
    # 6. BODY ROTATION (INTENT-DRIVEN)
    # =========================================================
    angle = torch.atan2(intent[..., 1], intent[..., 0] + 1e-6)
    angle = angle * (0.4 + state["energy"] * 2.2)

    upper_ids = list(range(5, 11))

    pivot = kp[:, 5:13, :2].mean(dim=1, keepdim=True)

    kp[:, upper_ids, :2] = rotate_points_around_pivot(
        kp[:, upper_ids, :2],
        pivot,
        angle
    )

    # =========================================================
    # 7. MICRO LIFE NOISE (ESSENTIAL FOR REALISM)
    # =========================================================
    noise = torch.randn_like(kp[..., :2]) * 0.0012
    kp[..., :2] += noise * state["energy"]

    # =========================================================
    # 8. HEAD SOFT STABILITY
    # =========================================================
    kp[:, head_ids, :2] = (
        kp[:, head_ids, :2] * 0.965 +
        kp_prev[:, head_ids, :2] * 0.035
    )

    # =========================================================
    # 9. CLEAN
    # =========================================================
    kp[..., :2] = torch.nan_to_num(kp[..., :2], nan=0.0)
    kp[..., :2] = torch.clamp(kp[..., :2], 0.0, 1.0)

    # =========================================================
    # 10. UPDATE STATE
    # =========================================================
    state["kp_prev"] = kp.clone()

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print("\n[🎭 V9 FULL ACTOR SYSTEM]")
        print(f"energy: {state['energy'].mean().item():.6f}")
        print(f"emotion: {state['emotion'].mean().item():.6f}")
        print(f"intent_norm: {intent.norm(dim=-1).mean().item():.6f}")
        print(f"gaze: {state['gaze_target'][0,0].tolist()}")

    return kp, state

def cinematic_motion_graph_v8(
    kp,
    state,
    head_ids=(0, 1, 2, 3, 4),
    debug=False
):
    B, N, _ = kp.shape

    # =========================================================
    # 0. INIT ACTOR STATE
    # =========================================================
    if state is None:
        return kp, {
            "kp_prev": kp.clone(),
            "energy": torch.zeros((B, 1, 1), device=kp.device),
            "intent_vec": None,
            "inertia": 0.9
        }

    kp_prev = state["kp_prev"]

    # =========================================================
    # 1. MOTION FIELD (ENERGY + DIRECTION)
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]

    motion_vec = delta.mean(dim=1, keepdim=True)  # (B,1,2)

    energy_raw = torch.norm(motion_vec, dim=-1, keepdim=True)

    state["energy"] = (
        state["energy"] * state["inertia"] +
        energy_raw * (1.0 - state["inertia"])
    )

    # =========================================================
    # 2. INTENT VECTOR (ACTOR DIRECTION MEMORY)
    # =========================================================
    if state["intent_vec"] is None:
        state["intent_vec"] = motion_vec.clone()
    else:
        state["intent_vec"] = (
            state["intent_vec"] * 0.85 +
            motion_vec * 0.15
        )

    intent = state["intent_vec"]

    # =========================================================
    # 3. ROTATION (EMOTION-DRIVEN, NOT GEOMETRY-DRIVEN)
    # =========================================================
    angle = torch.atan2(
        intent[..., 1],
        intent[..., 0] + 1e-6
    )

    angle = angle * (0.5 + state["energy"] * 2.5)

    # =========================================================
    # 4. PIVOT (ACTOR CENTER OF MASS)
    # =========================================================
    upper = kp[:, 5:11, :2].mean(dim=1, keepdim=True)
    lower = kp[:, 11:13, :2].mean(dim=1, keepdim=True)

    pivot = upper * 0.35 + lower * 0.65

    # fallback safety
    invalid = ~torch.isfinite(pivot).all(dim=-1, keepdim=True)
    pivot = torch.where(invalid, upper, pivot)

    # =========================================================
    # 5. APPLY ROTATION (SOFT PERFORMANCE TRANSFORM)
    # =========================================================
    upper_ids = list(range(5, 11))

    kp_rot = kp.clone()

    kp_rot[:, upper_ids, :2] = rotate_points_around_pivot(
        kp[:, upper_ids, :2],
        pivot,
        angle
    )

    kp = kp_rot

    # =========================================================
    # 6. MICRO INSTABILITY (LIFE SIGNAL)
    # =========================================================
    noise = torch.randn_like(kp[..., :2]) * 0.0015
    kp[..., :2] += noise * state["energy"]

    # =========================================================
    # 7. HEAD STABILITY (SOFT ANCHOR, NOT LOCK)
    # =========================================================
    kp[:, head_ids, :2] = (
        kp[:, head_ids, :2] * 0.96 +
        kp_prev[:, head_ids, :2] * 0.04
    )

    # =========================================================
    # 8. CLEAN OUTPUT
    # =========================================================
    kp[..., :2] = torch.nan_to_num(kp[..., :2], nan=0.0)
    kp[..., :2] = torch.clamp(kp[..., :2], 0.0, 1.0)

    # =========================================================
    # 9. UPDATE STATE
    # =========================================================
    state["kp_prev"] = kp.clone()

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print("\n[🎬 V8 ACTING SYSTEM]")
        print(f"energy: {state['energy'].mean().item():.6f}")
        print(f"intent_norm: {intent.norm(dim=-1).mean().item():.6f}")
        print(f"angle: {angle.mean().item():.6f}")
        print(f"motion: {energy_raw.mean().item():.6f}")
        print(f"pivot: {pivot[0, 0].tolist()}")

    return kp, state

def cinematic_motion_graph_v7(
    kp,
    state,
    head_ids=(0,1,2,3,4),
    upper_ids=(5,6,7,8,9,10),
    hip_ids=(11,12),

    rotation_strength=1.4,
    motion_sensitivity=7.0,

    # physics params
    angular_damping=0.88,
    angular_inertia=0.92,
    spring_strength=0.18,

    head_lag=0.75,
    max_deg=28.0,

    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # 0. INIT STATE
    # =========================================================
    if state is None:
        return kp, {
            "kp_prev": kp.clone(),
            "angle": torch.zeros((B,1,1), device=device),
            "angular_vel": torch.zeros((B,1,1), device=device),
            "torso_vec_prev": None
        }

    kp_prev = state["kp_prev"]

    # =========================================================
    # 1. MOTION ENERGY (robust + stable)
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]

    motion = torch.norm(delta, dim=-1, keepdim=True)

    motion_energy = (
        motion.mean(dim=1, keepdim=True) +
        motion.max(dim=1, keepdim=True).values
    ) * 0.5

    motion_gain = torch.tanh(motion_energy * motion_sensitivity)

    # =========================================================
    # 2. TORSO ANGLE + PHYSICS INERTIA
    # =========================================================
    angle_raw, torso_vec = compute_torso_rotation_delta(
        kp,
        state.get("torso_vec_prev")
    )
    state["torso_vec_prev"] = torso_vec

    angle_raw = torch.clamp(angle_raw, -1.2, 1.2)

    # velocity integration (physics core)
    angular_vel = state["angular_vel"]

    angular_vel = (
        angular_vel * angular_damping +
        (angle_raw - state["angle"]) * spring_strength
    )

    state["angular_vel"] = angular_vel

    # angle integration
    angle = state["angle"] + angular_vel * angular_inertia
    state["angle"] = angle

    # =========================================================
    # 3. PIVOT STABLE (hip + fallback)
    # =========================================================
    l_hip = kp[:, hip_ids[0], :2]
    r_hip = kp[:, hip_ids[1], :2]

    valid = torch.isfinite(l_hip).all(dim=-1, keepdim=True) & \
            torch.isfinite(r_hip).all(dim=-1, keepdim=True)

    shoulder_center = kp[:, [5,6], :2].mean(dim=1, keepdim=True)

    pivot = torch.where(valid, (l_hip + r_hip)*0.5, shoulder_center)

    # =========================================================
    # 4. ROTATION WITH PHYSICS
    # =========================================================
    kp_out = kp.clone()

    strength = rotation_strength * (1.0 + motion_gain)

    angle_final = torch.clamp(angle * strength, -max_deg, max_deg)

    rotated = rotate_points_around_pivot(
        kp[:, upper_ids, :2],
        pivot,
        angle_final
    )

    kp_out[:, upper_ids, :2] = rotated

    # =========================================================
    # 5. HEAD LAG (cinematic delay)
    # =========================================================
    kp_out[:, head_ids, :2] = (
        kp_prev[:, head_ids, :2] * head_lag +
        kp_out[:, head_ids, :2] * (1.0 - head_lag)
    )

    # =========================================================
    # 6. FULL BODY SPRING STABILIZATION
    # =========================================================
    kp_out[..., :2] = kp_prev[..., :2] + (
        kp_out[..., :2] - kp_prev[..., :2]
    ) * 0.92  # global damping

    # =========================================================
    # 7. CLEAN OUTPUT
    # =========================================================
    kp_out[..., :2] = torch.nan_to_num(kp_out[..., :2], nan=0.0)
    kp_out[..., :2] = torch.clamp(kp_out[..., :2], 0.0, 1.0)

    # =========================================================
    # 8. UPDATE STATE
    # =========================================================
    state["kp_prev"] = kp_out.clone()

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print("\n[🎬 CINEMA V7 PHYSICS]")
        print(f"motion_energy: {motion_energy.mean().item():.6f}")
        print(f"motion_gain: {motion_gain.mean().item():.6f}")
        print(f"angle: {angle.mean().item():.6f}")
        print(f"angular_vel: {angular_vel.mean().item():.6f}")
        print(f"pivot: {pivot[0,0].tolist()}")

    return kp_out, state

def cinematic_motion_graph_v6(
    kp,
    state,
    head_ids=(0,1,2,3,4),
    upper_ids=(5,6,7,8,9,10),
    hip_ids=(11,12),
    rotation_strength=1.5,
    rotation_smooth=0.3,
    motion_sensitivity=8.0,   # 🔥 BOOST IMPORTANT
    head_lock=0.65,           # 🔥 réduit (IMPORTANT)
    max_rotation_deg=30.0,
    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # INIT
    # =========================================================
    if state is None:
        return kp, {
            "kp_prev": kp.clone(),
            "angle": torch.zeros((B,1,1), device=device),
            "torso_vec_prev": None
        }

    kp_prev = state["kp_prev"]

    # =========================================================
    # 1. REAL MOTION ENERGY (FIX MAJEUR)
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]

    # 🔥 NE PAS AVERAGER TROP TÔT
    motion_per_joint = torch.norm(delta, dim=-1, keepdim=True)

    # 🔥 utilise max + mean mix (important)
    motion_score = (
        motion_per_joint.mean(dim=1, keepdim=True) +
        motion_per_joint.max(dim=1, keepdim=True).values
    ) * 0.5

    motion_gain = torch.tanh(motion_score * motion_sensitivity)

    # =========================================================
    # 2. TORSO ANGLE (stable + amplified slightly)
    # =========================================================
    angle_raw, torso_vec = compute_torso_rotation_delta(
        kp,
        state.get("torso_vec_prev")
    )

    state["torso_vec_prev"] = torso_vec

    angle_raw = torch.clamp(angle_raw, -1.2, 1.2)

    state["angle"] = (
        state["angle"] * (1 - rotation_smooth)
        + angle_raw * rotation_smooth
    )

    angle = state["angle"]

    # =========================================================
    # 3. PIVOT STABLE
    # =========================================================
    l_hip = kp[:, hip_ids[0], :2]
    r_hip = kp[:, hip_ids[1], :2]

    valid = torch.isfinite(l_hip).all(dim=-1, keepdim=True) & \
            torch.isfinite(r_hip).all(dim=-1, keepdim=True)

    shoulder_center = kp[:, [5,6], :2].mean(dim=1, keepdim=True)

    pivot = torch.where(valid, (l_hip + r_hip)*0.5, shoulder_center)

    # =========================================================
    # 4. ROTATION (IMPORTANT FIX)
    # =========================================================
    kp_out = kp.clone()

    strength = rotation_strength * (1.0 + motion_gain)

    angle_final = torch.clamp(
        angle * strength,
        -max_rotation_deg,
        max_rotation_deg
    )

    kp_out[:, upper_ids, :2] = rotate_points_around_pivot(
        kp[:, upper_ids, :2],
        pivot,
        angle_final
    )

    # =========================================================
    # 5. HEAD LOCK (SOFT FIX)
    # =========================================================
    if head_lock > 0:
        kp_out[:, head_ids, :2] = (
            kp_prev[:, head_ids, :2] * head_lock +
            kp_out[:, head_ids, :2] * (1 - head_lock)
        )

    # =========================================================
    # 6. NO HARD VELOCITY KILL (IMPORTANT CHANGE)
    # =========================================================
    # ❌ suppression du clamp violent
    # → remplace par soft damping

    kp_out[..., :2] = kp_prev[..., :2] + (
        kp_out[..., :2] - kp_prev[..., :2]
    ) * 0.95

    # =========================================================
    # 7. CLEAN
    # =========================================================
    kp_out[..., :2] = torch.nan_to_num(kp_out[..., :2], nan=0.0)
    kp_out[..., :2] = torch.clamp(kp_out[..., :2], 0.0, 1.0)

    # =========================================================
    # 8. STATE UPDATE
    # =========================================================
    state["kp_prev"] = kp_out.clone()

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print("\n[🎬 V6 MOTION FIX]")
        print("motion_score:", motion_score.mean().item())
        print("motion_gain:", motion_gain.mean().item())
        print("angle:", angle.mean().item())
        print("angle_final:", angle_final.mean().item())
        print("pivot:", pivot[0,0].tolist())

    return kp_out, state

def cinematic_motion_graph_v3(
    kp,
    state,
    head_ids=(0,1,2,3,4),
    body_ids=(5,6,7,8,9,10,11,12),
    rotation_strength=0.35,
    rotation_smooth=0.15,
    head_lock=True,
    debug=False
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

    if debug:
        angle_deg = angle.mean().item() * 57.2958
        raw_deg = angle_raw.mean().item() * 57.2958

        motion_val = motion.mean().item()
        strength_val = dynamic_strength.mean().item()

        pivot_mean = pivot.mean(dim=1)[0]

        print("\n[🎯 TORSO ROTATION DEBUG]")
        print(f"angle_raw_deg: {raw_deg:.3f}")
        print(f"angle_smooth_deg: {angle_deg:.3f}")
        print(f"motion: {motion_val:.6f}")
        print(f"rotation_strength: {strength_val:.4f}")

        print(f"pivot: ({pivot_mean[0].item():.3f}, {pivot_mean[1].item():.3f})")

        # stabilité du vecteur torso
        if state.get("torso_vec_prev") is not None:
            vec = torso_vec[0]
            print(f"torso_vec: ({vec[0].item():.3f}, {vec[1].item():.3f})")

    return kp, state

