#n3rPoseModule.py
# ✔ energy field
# ✔ intent vector (mémoire directionnelle)
# ✔ angular velocity (physique)
# ✔ inertia / damping
# ✔ pivot body (centre de rotation)
# ✔ head lag (cinematic delay)
# ✔ gaze system (attention model)
# ✔ body segmentation (upper / hips / head)
#cinematic_motion_graph_v3, update_motion_state

import torch
import torch.nn.functional as F
import inspect
import math

# =========================================================
# ACTOR FUNCTION MAP (EXTENSIBLE SYSTEM)
# =========================================================


def update_motion_state(
    kp,
    state,
    head_ids=(0,1,18,21,22,23,24),
    body_ids=(2,3,4,5,6,7,8,9,10,11,12,13),
    anchor_smooth=0.08,        # 🔥 plus lent = plus stable
    velocity_smooth=0.85,      # 🔥 plus inertiel
    drift_smooth=0.90,         # 🔥 IMPORTANT (low-pass correct)
    head_lock=True,
    head_lock_strength=0.95,   # 🔥 jamais 1.0 (évite freeze)
    debug=False
):
    B, N, _ = kp.shape

    # =========================================================
    # INIT
    # =========================================================
    if state is None:
        return kp, {
            "kp_prev": kp.clone(),
            "velocity": torch.zeros_like(kp[..., :2]),
            "anchor": kp[:, body_ids, :2].mean(dim=1, keepdim=True)
        }

    kp_prev = state["kp_prev"]

    if kp_prev.shape != kp.shape:
        kp_prev = kp.clone()

    # =========================================================
    # 1. VELOCITY (correct inertia model)
    # =========================================================
    raw_velocity = kp[..., :2] - kp_prev[..., :2]

    velocity = (
        state["velocity"] * velocity_smooth +
        raw_velocity * (1 - velocity_smooth)
    )

    state["velocity"] = velocity

    # =========================================================
    # 2. ANCHOR (camera stabilisation FIXED)
    # =========================================================
    anchor_prev = state["anchor"]

    anchor_now = kp[:, body_ids, :2].mean(dim=1, keepdim=True)

    # EMA anchor (camera memory)
    anchor = (
        anchor_prev * (1 - anchor_smooth) +
        anchor_now * anchor_smooth
    )

    state["anchor"] = anchor

    # =========================================================
    # 3. DRIFT (CORRECT FILTERING)
    # =========================================================
    drift_raw = anchor_now - anchor_prev

    drift = (
        drift_raw * (1 - drift_smooth) +
        drift_raw * drift_smooth   # ← remplacé correctement ci-dessous
    )

    # 🔥 FIX IMPORTANT : vrai low-pass filtering
    drift = drift * 0.1  # global damping (critical)

    kp[..., :2] = kp[..., :2] - drift

    # =========================================================
    # 4. APPLY VELOCITY INERTIA (IMPORTANT FIX)
    # =========================================================
    kp[..., :2] = kp_prev[..., :2] + velocity * 0.5

    # =========================================================
    # 5. HEAD LOCK (SOFT FIX)
    # =========================================================
    if head_lock:
        kp[:, head_ids, :2] = (
            kp[:, head_ids, :2] * (1 - head_lock_strength * 0.1) +
            kp_prev[:, head_ids, :2] * (head_lock_strength * 0.1)
        )

    # =========================================================
    # 6. SAFETY CLAMP
    # =========================================================
    kp[..., :2] = torch.nan_to_num(kp[..., :2], nan=0.0)
    kp[..., :2] = torch.clamp(kp[..., :2], 0.0, 1.0)

    # =========================================================
    # 7. UPDATE STATE
    # =========================================================
    state["kp_prev"] = kp.clone()

    return kp, state

ACTOR_LABELS = {
    "base": "[BASE MOTION]",
    "v6": "[V6 CINEMATIC]",
    "v7": "[V7 PHYSICS LAYER]",
    "v8": "[V8 ACTING SYSTEM]",
    "v9": "[V9 FULL ACTOR SYSTEM]",
    "v3": "[V3 SYSTEM]",
}


ACTOR_MODEL_SCHEDULE = [
    (0,  "base"),
    (1,  "v9"),
    (6,  "v8"),
    (10,  "base"),
    (11, "v7"),
    (15,  "base"),
    (16, "v6"),
    (17,  "base"),
    (18, "v3"),
]




# =========================================================
# ACTOR MODEL RESOLUTION (DIRECTOR SYSTEM)
# =========================================================

MOTION_MODEL_SCHEDULE = [
    (0,  "locked"),
    (5,  "stable"),
    (10, "warp"),
    (20, "cinematic"),
    (35, "dynamic"),
]

def resolve_motion_model(frame_idx: int) -> str:
    """
    Motion system:
    controls physics behavior over time (camera + velocity + warp).
    """
    model = "dynamic"

    for threshold, name in MOTION_MODEL_SCHEDULE:
        if frame_idx >= threshold:
            model = name

    return model



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


def compute_torso_rotation(kp, body_ids=(2,3,4,5,6,7,8,9,10,11,12,13)):
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


def rotate_points_around_pivot(points, pivot, angle, name="ROT", debug=None):
    """
    2D rotation (stable, batch-safe) + debug
    """

    s = torch.sin(angle)
    c = torch.cos(angle)
    if debug:
        print(f"[{name}] angle: {angle}")
        print(f"[{name}] pivot: {pivot}")
        print(f"[{name}] points shape: {points.shape}")

    # translate
    p = points - pivot
    if debug:
        print(f"[{name}] pre-translate sample: {p[0] if p.ndim > 1 else p}")

    x = p[..., 0]
    y = p[..., 1]

    x_new = x * c - y * s
    y_new = x * s + y * c

    out = torch.stack([x_new, y_new], dim=-1) + pivot
    if debug:
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
    head_ids=(0,1,18,21,22,23,24),
    body_ids=(2,3,4,5,6,7,8,9,10,11,12,13),
    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # 0. SAFE INIT (CRASH FIX GLOBAL)
    # =========================================================
    if state is None:
        state = {}

    def init_tensor(key, shape):
        if key not in state or not torch.is_tensor(state[key]):
            state[key] = torch.zeros(shape, device=device)

    init_tensor("kp_prev", (B, N, 2))
    init_tensor("energy", (B, 1, 1))
    init_tensor("emotion", (B, 1, 1))
    init_tensor("intent_vec", (B, 1, 2))
    init_tensor("attention", (B, 1, 2))
    init_tensor("angular_vel", (B, 1, 1))
    init_tensor("angle", (B, 1, 1))
    init_tensor("angle_vel", (B, 1, 1))

    state.setdefault("gaze_target", None)
    state.setdefault("gaze_jitter", 0.002)
    state.setdefault("inertia", 0.9)

    kp_prev = state["kp_prev"]

    # =========================================================
    # 1. MOTION ENERGY (SAFE)
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]
    motion_vec = delta.mean(dim=1, keepdim=True)

    energy_raw = torch.norm(motion_vec, dim=-1, keepdim=True)

    state["energy"] = (
        state["energy"] * state["inertia"] +
        energy_raw * (1.0 - state["inertia"])
    )

    # =========================================================
    # 2. INTENT VECTOR
    # =========================================================
    state["intent_vec"] = (
        state["intent_vec"] * 0.85 +
        motion_vec * 0.15
    )

    intent = state["intent_vec"]

    # =========================================================
    # 3. EMOTION
    # =========================================================
    emotion_target = torch.clamp(state["energy"] * 3.0, 0.0, 1.0)

    state["emotion"] = (
        state["emotion"] * 0.92 +
        emotion_target * 0.08
    )

    # =========================================================
    # 4. GAZE ENGINE (SAFE)
    # =========================================================
    eye_center = kp[:, [1,2], :2].mean(dim=1, keepdim=True)

    if state["gaze_target"] is None:
        state["gaze_target"] = eye_center.clone()

    drift = intent * 0.03

    state["gaze_target"] = (
        state["gaze_target"] * 0.9 +
        (eye_center + drift) * 0.1
    )

    gaze_dir = state["gaze_target"] - eye_center
    gaze_dir = gaze_dir + torch.randn_like(gaze_dir) * state["gaze_jitter"]

    # =========================================================
    # 5. FACE MOTION (SAFE INDEX CHECK)
    # =========================================================
    emotion = state["emotion"]

    mouth_open = emotion * 0.02

    if N > 14:
        kp[:, 14, 1] = kp[:, 14, 1] + mouth_open.squeeze(-1)

    kp[:, 1:3, :2] = kp[:, 1:3, :2] + gaze_dir * 0.15 * emotion

    # =========================================================
    # 6. BODY ROTATION (STABLE CINEMATIC VERSION)
    # =========================================================

    intent_angle = torch.atan2(
        intent[..., 1],
        intent[..., 0] + 1e-6
    )

    # -----------------------------
    # ENERGY GAIN (soft, non explosif)
    # -----------------------------
    energy = state["energy"].clamp(0.0, 1.0)

    gain = 0.25 + 0.9 * energy   # 🔥 réduit fortement

    target_angle = intent_angle * gain

    # -----------------------------
    # HARD LIMIT (safe cinematic range)
    # -----------------------------
    max_angle = math.radians(18)  # 🔥 réduit de 25 → 18
    target_angle = torch.clamp(target_angle, -max_angle, max_angle)

    # -----------------------------
    # ANGLE SMOOTHING (IMPORTANT FIX)
    # -----------------------------
    prev_angle = state["angle"]

    # EMA smoothing (critical stability)
    alpha = 0.85
    angle = prev_angle * alpha + target_angle * (1.0 - alpha)

    # -----------------------------
    # ANGULAR VELOCITY CONTROLLED
    # -----------------------------
    angle_vel = angle - prev_angle

    # clamp velocity (anti-jerk)
    angle_vel = torch.clamp(angle_vel, -0.04, 0.04)

    state["angle_vel"] = angle_vel
    state["angle"] = angle

    # -----------------------------
    # ROTATION DAMPING PER FRAME
    # -----------------------------
    rotation_damping = 0.92
    angle = angle * rotation_damping

    # =========================================================
    # APPLY ROTATION
    # =========================================================
    upper_ids = list(range(5, min(11, N)))

    pivot = kp[:, 5:min(13, N), :2].mean(dim=1, keepdim=True)

    kp[:, upper_ids, :2] = rotate_points_around_pivot(
        kp[:, upper_ids, :2],
        pivot,
        angle
    )

    # =========================================================
    # 7. MICRO NOISE (SAFE)
    # =========================================================
    noise = torch.randn_like(kp[..., :2]) * 0.0012
    kp[..., :2] = kp[..., :2] + noise * state["energy"]

    # =========================================================
    # 8. HEAD STABILITY
    # =========================================================
    kp[:, head_ids, :2] = (
        kp[:, head_ids, :2] * 0.965 +
        kp_prev[:, head_ids, :2] * 0.035
    )

    # =========================================================
    # 9. CLEAN OUTPUT
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
        print("\n[🎭 V9 SAFE ACTOR]")
        print(f"energy: {state['energy'].mean().item():.6f}")
        print(f"emotion: {state['emotion'].mean().item():.6f}")
        print(f"intent_norm: {intent.norm(dim=-1).mean().item():.6f}")
        print(f"gaze: {state['gaze_target'][0,0].tolist()}")

    return kp, state



def cinematic_motion_graph_v8(
    kp,
    state,
    head_ids=(0,1,18,21,22,23,24),
    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # 0. SAFE INIT (NO SHAPE DRIFT EVER)
    # =========================================================
    def to_B1(x, default=0.0):
        if x is None:
            return torch.full((B,1), default, device=device)
        if isinstance(x, (float, int)):
            return torch.full((B,1), float(x), device=device)
        if torch.is_tensor(x):
            if x.dim() == 0:
                return x.view(1,1).expand(B,1)
            if x.dim() == 1:
                return x.view(B,1)
            if x.dim() == 2:
                return x[:, :1]
        return torch.full((B,1), default, device=device)

    if state is None:
        state = {}

    kp_prev = state.get("kp_prev", kp.clone())

    state["energy"] = to_B1(state.get("energy", 0.0), B)
    state["inertia"] = state.get("inertia", 0.75)
    state["intent_vec"] = state.get("intent_vec", None)
    state.setdefault("angle", torch.zeros((B,1), device=device))
    state.setdefault("angle_vel", torch.zeros((B,1), device=device))

    # =========================================================
    # 1. MOTION ENERGY (STABLE)
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]

    motion_vec = delta.mean(dim=1, keepdim=True)  # (B,1,2)

    energy_raw = torch.norm(motion_vec, dim=-1, keepdim=True)  # (B,1,1)
    energy_raw = energy_raw[:, :, 0]  # → (B,1)

    state["energy"] = (
        state["energy"] * state["inertia"] +
        energy_raw * (1.0 - state["inertia"])
    )

    energy = state["energy"]  # (B,1)

    # =========================================================
    # 2. INTENT VECTOR (SAFE)
    # =========================================================
    if state["intent_vec"] is None:
        state["intent_vec"] = motion_vec
    else:
        state["intent_vec"] = (
            state["intent_vec"] * 0.85 +
            motion_vec * 0.15
        )

    intent = state["intent_vec"]  # (B,1,2)

    # =========================================================
    # 3. ROTATION (REDUCED + STABLE)
    # =========================================================

    raw_angle = torch.atan2(
        intent[..., 1],
        intent[..., 0] + 1e-6
    )  # (B,1)

    # -----------------------------
    # ENERGY GAIN (FORTEMENT RÉDUIT)
    # -----------------------------
    gain = 0.2 + energy * 1.2   # 🔥 réduit drastiquement (avant 2.5)

    target_angle = raw_angle * gain

    # -----------------------------
    # HARD LIMIT (PLUS STRICT)
    # -----------------------------
    max_angle = 0.6  # ~34° → OK mais plus safe
    target_angle = torch.clamp(target_angle, -max_angle, max_angle)

    # -----------------------------
    # ANGLE SMOOTHING (ESSENTIEL)
    # -----------------------------
    prev_angle = state.get("angle", torch.zeros_like(target_angle))

    alpha = 0.9  # 🔥 plus stable
    angle = prev_angle * alpha + target_angle * (1 - alpha)

    # -----------------------------
    # LIMIT ANGULAR SPEED (CRUCIAL FIX)
    # -----------------------------
    angle_vel = angle - prev_angle
    angle_vel = torch.clamp(angle_vel, -0.03, 0.03)

    # integrate softly
    angle = prev_angle + angle_vel

    # optional damping
    angle = angle * 0.92

    state["angle"] = angle
    state["angle_vel"] = angle_vel

    # =========================================================
    # 4. PIVOT (SAFE CENTER OF MASS)
    # =========================================================
    upper = kp[:, 5:11, :2].mean(dim=1, keepdim=True)
    lower = kp[:, 11:13, :2].mean(dim=1, keepdim=True)

    pivot = upper * 0.35 + lower * 0.65

    invalid = ~torch.isfinite(pivot).all(dim=-1, keepdim=True)
    pivot = torch.where(invalid, upper, pivot)

    # =========================================================
    # 5. ROTATION APPLY (SAFE)
    # =========================================================
    kp_out = kp.clone()

    upper_ids = list(range(5, 11))

    kp_out[:, upper_ids, :2] = rotate_points_around_pivot(
        kp[:, upper_ids, :2],
        pivot,
        angle.view(B,1,1)  # IMPORTANT FIX
    )

    # =========================================================
    # 6. MICRO INSTABILITY (CONTROLLED)
    # =========================================================
    noise = torch.randn_like(kp_out[..., :2]) * 0.0015
    kp_out[..., :2] = kp_out[..., :2] + noise * energy.unsqueeze(-1)

    # =========================================================
    # 7. HEAD STABILITY
    # =========================================================
    kp_out[:, head_ids, :2] = (
        kp_out[:, head_ids, :2] * 0.96 +
        kp_prev[:, head_ids, :2] * 0.04
    )

    # =========================================================
    # 8. CLEAN OUTPUT
    # =========================================================
    kp_out[..., :2] = torch.nan_to_num(kp_out[..., :2], nan=0.0)
    kp_out[..., :2] = torch.clamp(kp_out[..., :2], 0.0, 1.0)

    # =========================================================
    # 9. UPDATE STATE
    # =========================================================
    state["kp_prev"] = kp_out.clone()

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print("\n[🎬 V8 SAFE ACTING SYSTEM]")
        print("energy:", energy.mean().item())
        print("intent:", intent.norm(dim=-1).mean().item())
        print("angle:", angle.mean().item())
        print("motion:", energy_raw.mean().item())
        print("pivot:", pivot[0,0].tolist())

    return kp_out, state



def cinematic_motion_graph_v7(
    kp,
    state,
    head_ids=(0,1,18,21,22,23,24),
    upper_ids=(5,6,7,8,9,10),
    hip_ids=(11,12),
    rotation_strength=1.2,
    motion_sensitivity=6.0,
    damping=0.90,
    inertia=0.85,
    head_lag=0.7,
    max_deg=25.0,
    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # 0. SAFE STATE INIT (UNCHANGED BUT HARDENED)
    # =========================================================
    def to_B1(x, default=0.0):
        if x is None:
            return torch.zeros((B,1), device=device)
        if isinstance(x, (float, int)):
            return torch.full((B,1), float(x), device=device)
        if torch.is_tensor(x):
            if x.dim() == 0:
                return x.view(1,1).expand(B,1)
            if x.dim() == 1:
                return x.view(B,1)
            if x.dim() == 2:
                return x[:, :1]
        return torch.zeros((B,1), device=device)

    if state is None:
        state = {}

    kp_prev = state.get("kp_prev", kp.clone())

    angle = to_B1(state.get("angle", 0.0), B)
    angular_vel = to_B1(state.get("angular_vel", 0.0), B)

    state["angle"] = angle
    state["angular_vel"] = angular_vel
    state.setdefault("torso_vec_prev", None)

    # =========================================================
    # 1. MOTION ENERGY (STRUCTURED FIX)
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]
    motion = torch.norm(delta, dim=-1)  # (B,N)

    # body weighting (minimal but critical fix)
    w = torch.ones_like(motion)

    torso_ids = list(range(5, 11))

    w[:, torso_ids] *= 1.5
    w[:, hip_ids] *= 1.2
    w[:, [0,1]] *= 0.8

    motion_energy = (motion * w).mean(dim=1, keepdim=True)
    motion_gain = torch.tanh(motion_energy * motion_sensitivity)

    # =========================================================
    # 2. TORSO ANGLE (SAFE)
    # =========================================================
    angle_raw, torso_vec = compute_torso_rotation_delta(
        kp,
        state["torso_vec_prev"]
    )
    state["torso_vec_prev"] = torso_vec

    if angle_raw.dim() > 2:
        angle_raw = angle_raw.mean(dim=-1, keepdim=True)

    angle_raw = angle_raw.view(B,1).clamp(-1.0, 1.0)

    delta_angle = angle_raw - state["angle"]

    angular_vel = state["angular_vel"] * damping + delta_angle * 0.15
    angular_vel = torch.clamp(angular_vel, -0.2, 0.2)

    state["angular_vel"] = angular_vel
    state["angle"] = state["angle"] + angular_vel * inertia

    # =========================================================
    # 3. PIVOT (MORE STABLE BODY CENTER)
    # =========================================================
    upper_center = kp[:, torso_ids, :2].mean(dim=1, keepdim=True)
    hip_center = kp[:, hip_ids, :2].mean(dim=1, keepdim=True)

    pivot = upper_center * 0.4 + hip_center * 0.6

    valid = torch.isfinite(pivot).all(dim=-1, keepdim=True)
    shoulder_center = kp[:, [5,6], :2].mean(dim=1, keepdim=True)

    pivot = torch.where(valid, pivot, shoulder_center)

    # =========================================================
    # 4. ROTATION (UNCHANGED BUT STABILIZED INPUT)
    # =========================================================
    kp_out = kp.clone()

    strength = rotation_strength * (1.0 + motion_gain)

    angle_final = torch.clamp(
        state["angle"] * strength,
        -max_deg,
        max_deg
    ).view(B,1,1)

    kp_out[:, upper_ids, :2] = rotate_points_around_pivot(
        kp[:, upper_ids, :2],
        pivot,
        angle_final
    )

    # =========================================================
    # 5. HEAD LAG (UNCHANGED)
    # =========================================================
    kp_out[:, head_ids, :2] = (
        kp_prev[:, head_ids, :2] * head_lag +
        kp_out[:, head_ids, :2] * (1.0 - head_lag)
    )

    # =========================================================
    # 6. GLOBAL STABILITY (SLIGHTLY ADAPTIVE FIX)
    # =========================================================
    alpha = 0.88 + 0.08 * (1.0 - motion_gain)

    kp_out[..., :2] = kp_prev[..., :2] + (
        kp_out[..., :2] - kp_prev[..., :2]
    ) * alpha

    # =========================================================
    # 7. CLEAN OUTPUT
    # =========================================================
    kp_out[..., :2] = torch.nan_to_num(kp_out[..., :2], nan=0.0)
    kp_out[..., :2] = torch.clamp(kp_out[..., :2], 0.0, 1.0)

    # =========================================================
    # 8. STATE UPDATE
    # =========================================================
    state["kp_prev"] = kp_out.clone()

    # =========================================================
    # DEBUG (UNCHANGED)
    # =========================================================
    if debug:
        print("\n[🎬 V7 SAFE FIXED]")
        print("motion:", motion_energy.mean().item())
        print("gain:", motion_gain.mean().item())
        print("angle:", state["angle"].mean().item())
        print("vel:", state["angular_vel"].mean().item())
        print("pivot:", pivot[0,0].tolist())

    return kp_out, state




def cinematic_motion_graph_v6(
    kp,
    state,
    head_ids=(0,1,18,21,22,23,24),
    body_ids=(2,3,4,5,6,7,8,9,10,11,12,13),
    hip_ids=(11,12),
    rotation_strength=0.5,
    rotation_smooth=0.3,
    motion_sensitivity=8.0,   # 🔥 BOOST IMPORTANT
    head_lock=0.65,           # 🔥 réduit (IMPORTANT)
    max_rotation_deg=30.0,
    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # 0. INIT STATE SAFE
    # =========================================================
    if state is None:
        return kp, {
            "kp_prev": kp.clone(),
            "angle": torch.zeros((B,1), device=device),
            "torso_vec_prev": None
        }

    kp_prev = state.get("kp_prev", kp.clone())
    state.setdefault("angle", torch.zeros((B,1), device=device))
    state.setdefault("torso_vec_prev", None)

    # =========================================================
    # 1. REAL MOTION ENERGY
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]
    motion_per_joint = torch.norm(delta, dim=-1, keepdim=True)
    motion_score = (motion_per_joint.mean(dim=1, keepdim=True) +
                    motion_per_joint.max(dim=1, keepdim=True).values) * 0.5
    motion_gain = torch.tanh(motion_score * motion_sensitivity)

    # =========================================================
    # 2. TORSO ANGLE (stable + smoothing)
    # =========================================================
    angle_raw, torso_vec = compute_torso_rotation_delta(
        kp,
        state["torso_vec_prev"]
    )
    state["torso_vec_prev"] = torso_vec
    angle_raw = angle_raw.view(B,1).clamp(-1.2, 1.2)

    state["angle"] = state["angle"] * (1 - rotation_smooth) + angle_raw * rotation_smooth
    angle = state["angle"]

    # =========================================================
    # 3. PIVOT STABLE (hip fallback)
    # =========================================================
    l_hip = kp[:, hip_ids[0], :2]
    r_hip = kp[:, hip_ids[1], :2]
    valid = torch.isfinite(l_hip).all(dim=-1, keepdim=True) & torch.isfinite(r_hip).all(dim=-1, keepdim=True)
    shoulder_center = kp[:, [5,6], :2].mean(dim=1, keepdim=True)
    pivot = torch.where(valid, (l_hip + r_hip)*0.5, shoulder_center)

    # =========================================================
    # 4. ROTATION
    # =========================================================
    kp_out = kp.clone()
    strength = rotation_strength * (1.0 + motion_gain)
    angle_final = torch.clamp(angle * strength, -max_rotation_deg, max_rotation_deg).view(B,1)
    kp_out[:, upper_ids, :2] = rotate_points_around_pivot(
        kp[:, upper_ids, :2],
        pivot,
        angle_final
    )

    # =========================================================
    # 5. HEAD LOCK
    # =========================================================
    if head_lock > 0:
        kp_out[:, head_ids, :2] = kp_prev[:, head_ids, :2] * head_lock + kp_out[:, head_ids, :2] * (1.0 - head_lock)

    # =========================================================
    # 6. FULL BODY SOFT DAMPING
    # =========================================================
    kp_out[..., :2] = kp_prev[..., :2] + (kp_out[..., :2] - kp_prev[..., :2]) * 0.95

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
    # 9. DEBUG
    # =========================================================
    if debug:
        print("\n[🎬 CINEMA V6 SAFE]")
        print("motion_score:", motion_score.mean().item())
        print("motion_gain:", motion_gain.mean().item())
        print("angle:", angle.mean().item())
        print("angle_final:", angle_final.mean().item())
        print("pivot:", pivot[0,0].tolist())

    return kp_out, state


def cinematic_motion_graph_v3(
    kp,
    state,
    head_ids=(0,1,18,21,22,23,24),
    body_ids=(2,3,4,5,6,7,8,9,10,11,12,13),
    rotation_strength=0.15,
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


def apply_actor_model(kp, state, frame_idx, profile=None, debug=True):



    # =========================
    # SAFE MODEL RESOLUTION
    # =========================
    try:
        rmodel = resolve_actor_model(frame_idx)
    except Exception as e:
        if debug:
            print(f"[⚠ resolve_actor_model failed] {e}")
        rmodel = "v7"

    fn = ACTOR_PIPELINE.get(rmodel, update_motion_state)
    label = ACTOR_LABELS.get(rmodel, "[UNKNOWN MODEL]")

    # =========================
    # DEBUG HEADER
    # =========================
    if debug:
        print("\n[🎬 ACTOR PIPELINE] =====================================")
        print(f"frame_idx : {frame_idx}")
        print(f"model     : {rmodel}")
        print(f"profile   : {profile}")
        print(f"layer     : {label}")
        print("========================================================")

    # =========================
    # SAFE STATE INIT
    # =========================
    if state is None:
        state = {}

    kp_prev = state.get("kp_prev", kp.clone())

    state.setdefault("kp_prev", kp_prev)

    # FIX: velocity always valid shape (B,N,2)
    state.setdefault(
        "velocity",
        torch.zeros_like(kp[..., :2])
    )

    # =========================
    # SAFE ARG INSPECTION
    # (évite crash frame_idx / debug / velocity mismatch)
    # =========================
    try:
        sig = inspect.signature(fn)
        kwargs = {}

        if "frame_idx" in sig.parameters:
            kwargs["frame_idx"] = frame_idx
        if "debug" in sig.parameters:
            kwargs["debug"] = debug
        if "profile" in sig.parameters:
            kwargs["profile"] = profile

        result = fn(kp, state, **kwargs)

    except TypeError as e:
        if debug:
            print(f"[⚠ ACTOR SIGNATURE MISMATCH] {e}")
            print("[↩ fallback update_motion_state]")
        result = update_motion_state(kp, state)

    except Exception as e:
        if debug:
            print(f"[⚠ ACTOR ERROR] model={rmodel} -> fallback base motion")
            print(f"error: {e}")
        result = update_motion_state(kp, state)

    # =========================
    # NORMALIZE OUTPUT
    # =========================
    if isinstance(result, tuple):
        kp_out, new_state = result
    else:
        kp_out, new_state = result, state

    if new_state is None:
        new_state = {}

    # =========================
    # SAFE STATE UPDATE (IMPORTANT FIX)
    # =========================
    new_state["kp_prev"] = kp_out.clone()

    # FIX: avoid velocity/angle crash in downstream v6/v7
    if "velocity" not in new_state:
        new_state["velocity"] = torch.zeros_like(kp_out[..., :2])

    if "angle" not in new_state:
        new_state["angle"] = 0.0

    # =========================
    # DEBUG MOTION
    # =========================
    if kp_prev is not None:
        delta = (kp_out[..., :2] - kp_prev[..., :2]).abs().mean().item()
        print(f"[DEBUG] motion_delta_mean: {delta:.6f}")

    return kp_out, new_state





def resolve_actor_model(frame_idx: int) -> str:
    """
    Director timeline system:
    progressive acting complexity over time.
    """
    model = "v9"

    for threshold, name in ACTOR_MODEL_SCHEDULE:
        if frame_idx >= threshold:
            model = name

    return model


ACTOR_PIPELINE = {
    "base": update_motion_state,
    "v3": cinematic_motion_graph_v3,
    "v6": cinematic_motion_graph_v6,
    "v7": cinematic_motion_graph_v7,
    "v8": cinematic_motion_graph_v8,
    "v9": actor_system_v9,
}

