#n3rcoords.py


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
