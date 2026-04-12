#n3rcoords.py

import torch

def prepare_pose_tensor(pose_full, device, target_dtype):
    """
    Convertit un tensor image en format BCHW, corrige les channels,
    normalise en [-1, 1], et envoie sur le bon device/dtype.

    Args:
        pose_full (torch.Tensor): input tensor (HWC, CHW ou BCHW)
        device (torch.device): device cible
        target_dtype (torch.dtype): dtype cible

    Returns:
        torch.Tensor: tensor prêt à l'emploi (B,3,H,W)
    """

    # --- Format BCHW ---
    if pose_full.ndim == 3:
        if pose_full.shape[0] in [1, 3]:  # CHW
            pose_full = pose_full.unsqueeze(0)
        else:  # HWC
            pose_full = pose_full.permute(2, 0, 1).unsqueeze(0)

    # --- Fix channels ---
    if pose_full.shape[1] > 3:
        pose_full = pose_full[:, :3]
    elif pose_full.shape[1] == 1:
        pose_full = pose_full.repeat(1, 3, 1, 1)

    # --- Normalisation [-1,1] ---
    pose_full = (pose_full - 0.5) * 2.0
    pose_full = torch.clamp(pose_full, -1.0, 1.0)

    # --- Device + dtype ---
    pose_full = pose_full.to(device=device, dtype=target_dtype)

    # --- Debug ---
    print(f"[DEBUG] Pose full {pose_full.shape} dtype={pose_full.dtype}")

    return pose_full

# --- Vérifier que les listes ne sont pas vides avant l'appel ---
def has_valid_coords(face_coords_dict):
    for k, coords in face_coords_dict.items():
        if coords and all(isinstance(c, (list, tuple)) and len(c) == 2 for c in coords):
            return True
        return False


def sanitize_coords(coords):
    valid = []
    if isinstance(coords, dict) and "center" in coords:
        coords = [coords["center"]]
    for p in coords:
        try:
            if len(p) != 2:
                print(f"⚠ Coordonnée ignorée car invalide: {p}")
                continue
            x, y = int(p[0]), int(p[1])
            valid.append([x, y])
        except (ValueError, TypeError):
            print(f"⚠ Coordonnée ignorée car invalide: {p}")
            continue
    return valid

# Convertir toutes les coordonnées en [[x, y]] et sécuriser
def process_coords(coords, label="coords"):
    print(f"{label}: {coords}")
    return [[float(x), float(y)] for x, y in coords]


def prepare_face_coords(
    eye_coords,
    mouth_coords,
    ear_coords,
    nose_coords,
    process_coords
):
    print("🟢 Debug coords:")

    # --- Normalisation initiale ---
    process_coords(eye_coords, "eye_coords")
    process_coords(mouth_coords, "mouth_coords")
    process_coords(ear_coords, "ear_coords")

    print(f"nose_coords: {nose_coords}")

    # --- Sanitize ---
    eye_coords_list = sanitize_coords(eye_coords)
    print(f"eye_coords sanitize_coords: {eye_coords_list}")

    mouth_coords_list = sanitize_coords(mouth_coords)
    print(f"mouth_coords_list sanitize_coords: {mouth_coords_list}")

    ear_coords_list = sanitize_coords(ear_coords)
    print(f"ear_coords_list sanitize_coords: {ear_coords_list}")

    nose_coords_list = sanitize_coords(nose_coords)
    print(f"nose_coords_list sanitize_coords: {nose_coords_list}")

    # --- Nez dict (conservé pour ailleurs) ---
    nose_coords_dict = nose_coords if nose_coords else None

    # --- Dict global ---
    face_coords_dict = {
        "eyes": eye_coords_list,
        "mouth": mouth_coords_list,
        "ears": ear_coords_list,
        "nose": nose_coords_list,
    }

    print(f"Face coords dict: {face_coords_dict}")

    return face_coords_dict, nose_coords_dict, eye_coords_list, mouth_coords_list, ear_coords_list, nose_coords_list

# =========================
# 🔹 NORMALISATION INPUTS
# =========================
def pair(coords, debug=False):
    if coords:
        out = [safe_xy(c, debug=debug) for c in coords]
        if debug:
            print(f"[pair] input={coords} → output={out}")
        return out
    if debug:
        print("[pair] coords=None → fallback (0,0)")
    return [(0, 0), (0, 0)]

# =========================
# 🔹 SAFE UTILS
# =========================
def safe_xy(coord, debug=False):
    if coord is None:
        if debug:
            print("[safe_xy] None → (0,0)")
        return (0, 0)

    if isinstance(coord, list) and len(coord) == 1:
        val = tuple(coord[0])
        if debug:
            print(f"[safe_xy] list[1] {coord} → {val}")
        return val

    if isinstance(coord, (list, tuple)) and len(coord) == 2:
        val = tuple(coord)
        if debug:
            print(f"[safe_xy] tuple/list {coord} → {val}")
        return val

    if debug:
        print(f"[safe_xy] format inconnu {coord} → (0,0)")
    return (0, 0)

def norm(coord):
    if isinstance(coord, (list, tuple)) and len(coord) == 1:
        return coord[0]
    return coord


def safe_update(idx, coord, keypoints_np, W, H, label="", debug=False):
    # =========================
    # 🔹 NORMALISATION ROBUSTE
    # =========================
    if coord is None:
        x, y = 0, 0
    elif isinstance(coord, dict):
        # sécurité si jamais MediaPipe ou autre passe un dict
        x, y = 0, 0
    elif isinstance(coord, (list, tuple)):

        # cas [[x,y]]
        if len(coord) == 1 and isinstance(coord[0], (list, tuple)):
            coord = coord[0]

        if len(coord) == 2:
            x, y = coord
        else:
            x, y = 0, 0
    else:
        x, y = 0, 0

    old_x, old_y = keypoints_np[idx, 0] * W, keypoints_np[idx, 1] * H

    # =========================
    # 🔹 LOGIQUE SAFE IDENTIQUE V1
    # =========================
    if x == 0 and y == 0:
        if old_x != 0 or old_y != 0:
            if debug:
                print(f"[safe_update] {label} fallback → garde ({old_x:.1f},{old_y:.1f})")
        else:
            if debug:
                print(f"[safe_update] ⚠ {label} ignoré (aucune valeur valide)")
        return

    # clamp sécurité (optionnel mais utile)
    x = max(0, min(x, W))
    y = max(0, min(y, H))

    if debug:
        print(f"[safe_update] {label}: ({old_x:.1f},{old_y:.1f}) → ({x:.1f},{y:.1f})")

    keypoints_np[idx, 0] = x / W
    keypoints_np[idx, 1] = y / H
    keypoints_np[idx, 2] = 1.0
