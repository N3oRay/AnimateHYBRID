import torch
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw
from .n3rMotionPose_tools import save_debug_mask, feather_inside_strict, feather_mask, feather_mask_fast, feather_outside_only, feather_inside,feather_inside_strict, feather_outside_only_alpha
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
        self._prev_facial_points = None
        # Ajustement
        self.torce_expand_w=1.6
        self.torce_shrink_h=1.2

        # -----------------------------
        # Mapping global des points faciaux
        # -----------------------------
        self.FACIAL_POINT_IDX = {
            'nose': 0,
            'neck': 1,
            'right_shoulder': 2,
            'right_elbow': 3,
            'right_wrist': 4,
            'left_shoulder': 5,
            'left_elbow': 6,
            'left_wrist': 7,
            'right_hip': 8,
            'left_hip': 11,
            'right_eye': 14,
            'left_eye': 15,
            'right_ear': 16,
            'left_ear': 17,
            'mouth': 18,
            'chin': 21,
            'left_side_neck': 22,
            'right_side_neck': 23,
            'anchor_neck': 24,
            'mouth_left': 48,    # coin gauche des lèvres supérieures
            'mouth_right': 49,   # coin droit des lèvres supérieures
            'mouth_top': 50,     # index approximatif du haut de la bouche (à ajuster selon ton keypoints)
            'mouth_bottom': 51,  # index approximatif du bas de la bouche
        }

    # -----------------------------
    # 🔹 UPDATE HAUT DU CORPS (smooth)
    # -----------------------------
    def update_upper_body(self, coords_dict, strength=0.35):
        """
        coords_dict : dict avec les points détectés (nose, neck, shoulder, elbow, wrist, clavicle, eyes, ears, mouth)
        strength : interpolation entre l'ancien et le nouveau (0=full old, 1=full new)
        """
        keypoints_np = self.keypoints.clone().cpu().numpy()[0]

        for name, idx in self.FACIAL_POINT_IDX.items():
            if name in coords_dict:
                new_coord = np.array(coords_dict[name], dtype=np.float32)

                # Smooth
                prev_coord = keypoints_np[idx] if self._prev_keypoints is None else self._prev_keypoints[0, idx].cpu().numpy()
                keypoints_np[idx] = prev_coord * (1 - strength) + new_coord * strength

        # Mise à jour du tensor
        keypoints_np = np.expand_dims(keypoints_np, axis=0)
        keypoints_tensor = torch.from_numpy(keypoints_np).to(self.device)

        # Update memory
        self._prev_keypoints = keypoints_tensor.clone()
        self.keypoints = keypoints_tensor.clone()

        return keypoints_tensor

    # -----------------------------
    # 🔹 ROTATION TORSE
    # -----------------------------
    def rotate_torso(self, angle_deg: float, strength: float = 1.0):
        """
        Applique une rotation 2D du torse (hips + épaules) autour du centre du buste.
        angle_deg : rotation en degrés (+ = sens horaire)
        strength  : 0 = pas de rotation, 1 = rotation totale
        """
        if self.B == 0:
            return

        # Points du torse à affecter
        torso_points = ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip']
        idxs = [self.FACIAL_POINT_IDX[p] for p in torso_points]

        # Centre du torse
        center = self.keypoints[:, idxs, :2].mean(dim=1, keepdim=True)  # [B,1,2]

        # Conversion angle -> radians
        angle_rad = angle_deg * (np.pi / 180.0) * strength

        # Matrice de rotation 2D
        rot_matrix = torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad)],
            [torch.sin(angle_rad),  torch.cos(angle_rad)]
        ], device=self.device)

        # Appliquer rotation
        torso_coords = self.keypoints[:, idxs, :2] - center  # centrer
        torso_coords_rot = torch.matmul(torso_coords, rot_matrix.T) + center  # rotation + recentrage

        # Mise à jour keypoints
        self.keypoints[:, idxs, :2] = torso_coords_rot
        return self.keypoints[:, idxs, :2]

    # =========================
    # GETTER
    # =========================

    def get_prev_facial_points(self):
        return self._prev_facial_points

    #self.get_torce_expand_w()
    def get_torce_expand_w(self):
        return self.torce_expand_w

    #self.get_torce_shrink_h()
    def get_torce_shrink_h(self):
        return self.torce_shrink_h

    def has_point(self, name: str) -> bool:
        """
        Vérifie si un point existe dans le mapping ET dans le tensor.
        """
        idx = self.FACIAL_POINT_IDX.get(name, None)
        if idx is None:
            return False
        return idx < self.keypoints.shape[1]

    # =========================
    # SETTER
    # =========================
    def set_prev_facial_points(self, points):
        self._prev_facial_points = points

    def get_center(self):
        # suppose keypoints shape: (B, N, 2) ou (N, 2)
        kpts = self.keypoints

        if kpts.dim() == 3:
            kpts = kpts[0]

        valid = kpts[kpts[..., 0] > 0]  # filtre simple si coords invalides
        if valid.shape[0] == 0:
            return torch.zeros(2, device=kpts.device)

        return valid.mean(dim=0)

    def estimate_facial_points_full(self, smooth=0.8):
        """
        Version complète + temporelle + stable
        """
        points = {}

        nose = self.keypoints[:, 0, :2]
        right_eye = self.keypoints[:, 14, :2] if self.keypoints.shape[1] > 14 else None
        left_eye  = self.keypoints[:, 15, :2] if self.keypoints.shape[1] > 15 else None

        if right_eye is not None and left_eye is not None:
            eye_dist = (left_eye[:, 0] - right_eye[:, 0]).abs().unsqueeze(1)
        else:
            eye_dist = torch.full((self.B,1), 0.12, device=self.device)

        # -------------------- Base structure --------------------
        mouth_center = nose + torch.cat([torch.zeros_like(eye_dist), eye_dist * 1.0], dim=1)

        mouth_left   = mouth_center + torch.cat([-0.3 * eye_dist, torch.zeros_like(eye_dist)], dim=1)
        mouth_right  = mouth_center + torch.cat([ 0.3 * eye_dist, torch.zeros_like(eye_dist)], dim=1)
        mouth_top    = mouth_center + torch.cat([torch.zeros_like(eye_dist), -0.15 * eye_dist], dim=1)
        mouth_bottom = mouth_center + torch.cat([torch.zeros_like(eye_dist),  0.15 * eye_dist], dim=1)

        points.update({
            "mouth_center": mouth_center,
            "mouth_left": mouth_left,
            "mouth_right": mouth_right,
            "mouth_top": mouth_top,
            "mouth_bottom": mouth_bottom,
        })

        # -------------------- YEUX enrichis --------------------
        if right_eye is not None and left_eye is not None:
            eye_vec = left_eye - right_eye

            points["eye_center"] = (left_eye + right_eye) / 2

            points["right_eye_inner"] = right_eye + 0.2 * eye_vec
            points["right_eye_outer"] = right_eye - 0.3 * eye_vec

            points["left_eye_inner"] = left_eye - 0.2 * eye_vec
            points["left_eye_outer"] = left_eye + 0.3 * eye_vec

        # =========================
        # 🔥 SMOOTH TEMPOREL AUTO
        # =========================
        prev = self.get_prev_facial_points()

        if prev is not None:
            for k in points:
                if k in prev:
                    points[k] = smooth * prev[k] + (1 - smooth) * points[k]

        # =========================
        # 🔥 UPDATE MÉMOIRE
        # =========================
        self.set_prev_facial_points(points)

        return points


    def estimate_facial_points(self, prev_points=None, smooth=0.8):
        points = {}
        nose = self.keypoints[:,0,:2]
        right_eye = self.keypoints[:,14,:2] if self.keypoints.shape[1] > 14 else None
        left_eye = self.keypoints[:,15,:2] if self.keypoints.shape[1] > 15 else None

        # Distance relative yeux-nez
        eye_dist = (left_eye[:,0]-right_eye[:,0]).abs() if right_eye is not None else 0.12
        eye_dist = eye_dist.unsqueeze(1)

        mouth_center = nose + torch.tensor([0.0, eye_dist.mean()*1.0], device=self.device)

        mouth_left  = mouth_center + torch.tensor([-0.3,0.0], device=self.device) * eye_dist
        mouth_right = mouth_center + torch.tensor([0.3,0.0], device=self.device) * eye_dist
        mouth_top   = mouth_center + torch.tensor([0.0,-0.15], device=self.device) * eye_dist
        mouth_bottom= mouth_center + torch.tensor([0.0,0.15], device=self.device) * eye_dist

        points['mouth_center'], points['mouth_left'], points['mouth_right'] = mouth_center, mouth_left, mouth_right
        points['mouth_top'], points['mouth_bottom'] = mouth_top, mouth_bottom

        # Lissage si points précédents fournis
        if prev_points is not None:
            for k in points:
                points[k] = smooth*prev_points[k] + (1-smooth)*points[k]

        return points

    # ----------------- Calcul des points de la bouche -----------------
    def estimate_missing_facial_points(self):
        """
        Estime les points manquants du visage (bouche, coins de lèvres, coins des yeux)
        à partir des points détectés (nez, yeux, oreilles, bouche si disponibles).

        Retourne un dictionnaire {nom_point: tensor [B,2]}.
        """
        estimated_points = {}
        B = self.B
        device = self.device

        # ----------------- Points de base -----------------
        nose = self.keypoints[:, 0, :2]       # point 0 = nez
        mouth_detected = self.keypoints[:, 18, :2]  # point 18 = bouche

        # Yeux
        right_eye = self.keypoints[:, 14, :2] if self.keypoints.shape[1] > 14 else None
        left_eye = self.keypoints[:, 15, :2] if self.keypoints.shape[1] > 15 else None

        # Oreilles
        right_ear = self.keypoints[:, 16, :2] if self.keypoints.shape[1] > 16 else None
        left_ear  = self.keypoints[:, 17, :2] if self.keypoints.shape[1] > 17 else None

        # ----------------- BOUCHE -----------------
        # Si bouche détectée, centre = point détecté
        mouth_center = mouth_detected.clone()

        # Largeur de la bouche : si oreilles connues, sinon distance yeux
        if right_ear is not None and left_ear is not None:
            mouth_width = (left_ear[:,0] - right_ear[:,0]) * 0.4  # proportionnelle à la largeur de la tête
        elif right_eye is not None and left_eye is not None:
            mouth_width = (left_eye[:,0] - right_eye[:,0]) * 0.5
        else:
            mouth_width = torch.tensor(0.12, device=device).expand(B)  # fallback

        # Hauteur approximative de la bouche
        mouth_height = mouth_width * 0.25

        # Coins gauche/droite
        mouth_left = mouth_center.clone()
        mouth_left[:,0] -= mouth_width / 2
        mouth_right = mouth_center.clone()
        mouth_right[:,0] += mouth_width / 2

        # Haut/bas
        mouth_top = mouth_center.clone()
        mouth_top[:,1] -= mouth_height / 2
        mouth_bottom = mouth_center.clone()
        mouth_bottom[:,1] += mouth_height / 2

        estimated_points['mouth_center'] = mouth_center
        estimated_points['mouth_left'] = mouth_left
        estimated_points['mouth_right'] = mouth_right
        estimated_points['mouth_top'] = mouth_top
        estimated_points['mouth_bottom'] = mouth_bottom

        # ----------------- COINS DES YEUX (optionnel) -----------------
        if right_eye is not None and left_eye is not None:
            eye_width = (left_eye[:,0] - right_eye[:,0])
            eye_height = 0.1 * eye_width

            estimated_points['right_eye_inner'] = right_eye.clone()
            estimated_points['right_eye_outer'] = right_eye.clone()
            estimated_points['right_eye_outer'][:,0] += 0.5*eye_width
            estimated_points['right_eye_inner'][:,0] -= 0.2*eye_width

            estimated_points['left_eye_inner'] = left_eye.clone()
            estimated_points['left_eye_outer'] = left_eye.clone()
            estimated_points['left_eye_outer'][:,0] -= 0.5*eye_width
            estimated_points['left_eye_inner'][:,0] += 0.2*eye_width

        return estimated_points

    # ----------------- Keypoint utils -----------------
    def get_point(self, idx):
        if isinstance(idx, str):
            idx = self.FACIAL_POINT_IDX[idx]
        return self.keypoints[:, idx, :2]

    # ----------------- Torso angle -----------------
    def compute_torso_angle(self):
        """Calcule l'angle du torse selon les épaules"""
        r_shoulder = self.get_point(2)
        l_shoulder = self.get_point(5)
        vec = r_shoulder - l_shoulder
        angle = torch.atan2(vec[:,1], vec[:,0]).unsqueeze(1)  # [B,1]
        self.angles = angle
        return angle
    # ----------------- Masque Bouche -----------------
    def get_mouth_region(
        self,
        H: int,
        W: int,
        device=None,
        debug: bool = False,
        debug_dir: str = None,
        frame_counter: int = 0,
        expand_w=0.3,
        expand_h=0.25,
        min_size=4  # 🔥 taille minimale en pixels
    ):
        """
        Masque bouche robuste [B,1,H,W]
        - sécurisé contre NaN / points invalides
        - taille minimale garantie
        - fallback intelligent si détection foire
        """

        if device is None:
            device = self.device

        B = self.B
        mask = torch.zeros((B, 1, H, W), device=device)

        # =========================
        # 🔹 Récupération points
        # =========================
        try:
            points_dict = self.estimate_missing_facial_points()
        except Exception as e:
            print(f"[WARN] mouth_region: estimation failed → fallback empty ({e})")
            return mask

        required_keys = ['mouth_left', 'mouth_right', 'mouth_top', 'mouth_bottom']

        # Vérification des clés
        for k in required_keys:
            if k not in points_dict:
                print(f"[WARN] mouth_region: missing key {k}")
                return mask

        mouth_left   = points_dict['mouth_left']
        mouth_right  = points_dict['mouth_right']
        mouth_top    = points_dict['mouth_top']
        mouth_bottom = points_dict['mouth_bottom']

        # =========================
        # 🔹 Construction masque
        # =========================
        for b in range(B):

            # 🔹 check NaN / invalid
            pts = torch.stack([
                mouth_left[b],
                mouth_right[b],
                mouth_top[b],
                mouth_bottom[b]
            ])

            if torch.isnan(pts).any():
                print(f"[WARN] mouth_region: NaN detected (batch {b})")
                continue

            # 🔹 centre
            x_center = (mouth_left[b,0] + mouth_right[b,0]) / 2
            y_center = (mouth_top[b,1] + mouth_bottom[b,1]) / 2

            # 🔹 dimensions (safe)
            width  = torch.abs(mouth_right[b,0] - mouth_left[b,0])
            height = torch.abs(mouth_bottom[b,1] - mouth_top[b,1])

            # fallback si trop petit
            if width < 1e-4 or height < 1e-4:
                width  = torch.tensor(0.05, device=device)
                height = torch.tensor(0.05, device=device)

            # 🔹 expansion
            width  = width  * (1 + expand_w)
            height = height * (1 + expand_h)

            # 🔹 conversion pixels
            x_min = int((x_center - width/2) * (W-1))
            x_max = int((x_center + width/2) * (W-1))
            y_min = int((y_center - height/2) * (H-1))
            y_max = int((y_center + height/2) * (H-1))

            # 🔹 clamp sécurisé
            x_min = max(0, min(W-1, x_min))
            x_max = max(0, min(W-1, x_max))
            y_min = max(0, min(H-1, y_min))
            y_max = max(0, min(H-1, y_max))

            # 🔥 taille minimale garantie
            if (x_max - x_min) < min_size:
                cx = (x_min + x_max) // 2
                x_min = max(0, cx - min_size // 2)
                x_max = min(W-1, cx + min_size // 2)

            if (y_max - y_min) < min_size:
                cy = (y_min + y_max) // 2
                y_min = max(0, cy - min_size // 2)
                y_max = min(H-1, cy + min_size // 2)

            # 🔹 fill
            mask[b,0,y_min:y_max+1, x_min:x_max+1] = 1.0

        # =========================
        # 🔹 Feather (safe)
        # =========================
        try:
            mask = feather_outside_only_alpha(mask, radius=3, sigma=1.5)
        except Exception as e:
            print(f"[WARN] mouth_region: feather failed ({e})")

        # =========================
        # 🔹 Debug
        # =========================
        if debug and debug_dir is not None:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                mask_np = (mask[0,0].detach().cpu().numpy() * 255).astype(np.uint8)

                mask_debug = cv2.resize(
                    mask_np,
                    (W*4, H*4),
                    interpolation=cv2.INTER_NEAREST
                )

                mask_debug_rgb = cv2.cvtColor(mask_debug, cv2.COLOR_GRAY2BGR)

                save_path = os.path.join(
                    debug_dir,
                    f"mouth_mask_{frame_counter:05d}.png"
                )

                cv2.imwrite(save_path, mask_debug_rgb)
                print(f"[DEBUG] Mouth mask saved: {save_path}")

            except Exception as e:
                print(f"[WARN] mouth_region: debug failed ({e})")

        return mask

    # ----------------- Torso delta -----------------
    # Attention le expand_w et le shrink_h doit être similaire pour compute_torso_delta et create_upper_body_mask
    def compute_torso_delta(
        self,
        latent_h: int,
        latent_w: int,
        expand_w=None,
        shrink_h=None
    ):
        """
        Delta torse aligné avec le masque ovale adaptatif
        - largeur adaptative épaules/hanche
        - centre pondéré chest/belly
        - lissage temporel possible
        """
        # Valeurs par défaut dynamiques
        if expand_w is None:
            expand_w = self.get_torce_expand_w()
        if shrink_h is None:
            shrink_h = self.get_torce_shrink_h()

        # =========================
        # 🔹 Points torse
        # =========================
        r_sh = self.get_point(19)  # right_shoulder
        l_sh = self.get_point(20)  # left_shoulder
        r_hip = self.get_point(8)  # right_hip
        l_hip = self.get_point(11) # left_hip

        # Stack pour traitement batch
        pts = torch.stack([r_sh, l_sh, r_hip, l_hip], dim=1)  # [B,4,2]

        # =========================
        # 🔹 Centres anatomiques
        # =========================
        shoulder_center = (r_sh + l_sh) / 2
        hip_center = (r_hip + l_hip) / 2

        chest = shoulder_center * 0.7 + hip_center * 0.3
        belly = shoulder_center * 0.3 + hip_center * 0.7

        # Centre pondéré
        center = (chest + belly) / 2  # [B,2]

        # =========================
        # 🔹 Largeur/hauteur adaptative
        # =========================
        shoulder_width = torch.norm(r_sh - l_sh, dim=1, keepdim=True)  # [B,1]
        hip_width = torch.norm(r_hip - l_hip, dim=1, keepdim=True)    # [B,1]

        width_top = shoulder_width * expand_w
        width_bottom = hip_width * expand_w * 0.9
        height = torch.norm(hip_center - shoulder_center, dim=1, keepdim=True) * shrink_h

        # =========================
        # 🔹 Bounding box du torse (adapté à l’ellipse)
        # =========================
        x_min = center[:,0:1] - width_top/2
        x_max = center[:,0:1] + width_top/2
        y_min = center[:,1:2] - height/2
        y_max = center[:,1:2] + height/2

        # Recalcule centre exact
        torso_center_x = (x_min + x_max) * 0.5
        torso_center_y = (y_min + y_max) * 0.5
        torso_center = torch.cat([torso_center_x, torso_center_y], dim=1)  # [B,2]

        # =========================
        # 🔹 Delta normalisé
        # =========================
        delta = torso_center - 0.5
        delta = torch.tanh(delta * 1.5) * 0.12

        # =========================
        # 🔹 Smooth temporel (optionnel)
        # =========================
        if hasattr(self, "_prev_torso_delta"):
            alpha = 0.85
            delta = alpha * self._prev_torso_delta + (1 - alpha) * delta

        self._prev_torso_delta = delta
        self.delta = delta

        return delta

    def compute_torso_delta_v2(
        self,
        latent_h: int,
        latent_w: int,
        expand_w=1.0,
        shrink_h=0.9
    ):
        """
        Delta torse aligné avec le masque ovale (plus stable + plus lisible)
        """

        # =========================
        # 🔹 Points torse
        # =========================
        pts = torch.stack([
            self.get_point(19),  # r_shoulder
            self.get_point(20),  # l_shoulder
            self.get_point(8),   # r_hip
            self.get_point(11)   # l_hip
        ], dim=1)  # [B,4,2]

        # =========================
        # 🔹 Centre brut
        # =========================
        cx = pts[:, :, 0].mean(dim=1, keepdim=True)
        cy = pts[:, :, 1].mean(dim=1, keepdim=True)

        # =========================
        # 🔹 Appliquer EXACTEMENT la même déformation que le masque
        # =========================
        pts_x = cx + (pts[:, :, 0] - cx) * expand_w
        pts_y = cy + (pts[:, :, 1] - cy) * shrink_h

        # =========================
        # 🔹 Bounding box du torse (plus robuste que mean)
        # =========================
        x_min = pts_x.min(dim=1, keepdim=True)[0]
        x_max = pts_x.max(dim=1, keepdim=True)[0]

        y_min = pts_y.min(dim=1, keepdim=True)[0]
        y_max = pts_y.max(dim=1, keepdim=True)[0]

        # Centre réel du masque
        torso_center_x = (x_min + x_max) * 0.5
        torso_center_y = (y_min + y_max) * 0.5

        torso_center = torch.cat([torso_center_x, torso_center_y], dim=1)  # [B,2]

        # =========================
        # 🔹 Delta normalisé (stable)
        # =========================
        delta = torso_center - 0.5

        # 🔥 très important : non-linéarité douce
        delta = torch.tanh(delta * 1.5) * 0.12

        # =========================
        # 🔹 Option PRO : inertie temporelle (🔥 gros gain qualité)
        # =========================
        if hasattr(self, "_prev_torso_delta"):
            alpha = 0.85  # smooth temporel
            delta = alpha * self._prev_torso_delta + (1 - alpha) * delta

        self._prev_torso_delta = delta

        # =========================
        # 🔹 Store
        # =========================
        self.delta = delta

        return delta

    def compute_torso_delta_v1(self, latent_h: int, latent_w: int, expand_w=0.95, shrink_h=0.70):
        pts = torch.stack([
            self.get_point(19),  # r_shoulder
            self.get_point(20),  # l_shoulder
            self.get_point(8),   # r_hip
            self.get_point(11)   # l_hip
        ], dim=1)  # [B,4,2]

        # Centre exact du polygone
        cx = pts[:,:,0].mean(dim=1, keepdim=True)
        cy = pts[:,:,1].mean(dim=1, keepdim=True)

        # Ajuster largeur/hauteur pour matcher le masque
        pts[:,:,0] = cx + (pts[:,:,0] - cx) * expand_w
        pts[:,:,1] = cy + (pts[:,:,1] - cy) * shrink_h

        # Delta : simple translation du centre du polygone (sans scale excessif)
        torso_center = torch.stack([cx.squeeze(1), cy.squeeze(1)], dim=1)  # [B,2]
        delta = torso_center - 0.5  # recentrer autour du milieu
        delta = torch.tanh(delta * 1.0) * 0.1  # petite amplitude, stabilisation légère

        self.delta = delta
        return delta
    # --------------- Mask decor ---------------------------------------------
    def create_decor_mask( self, H: int, W: int, mask_face=None, mask_torso=None, mask_hair=None, debug: bool = False, debug_dir: str = None, frame_counter: int = 0, expand=2.2, vertical_bias=1.2, falloff_strength=2.0 ):
        device = self.device
        B = self.B

        mask = torch.zeros(B, 1, H, W, device=device)

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )

        def to_px(kp):
            return kp * torch.tensor([W - 1, H - 1], device=device)

        for b in range(B):

            # 🔹 Points clés
            r_sh = to_px(self.get_point(self.FACIAL_POINT_IDX["right_shoulder"])[b])
            l_sh = to_px(self.get_point(self.FACIAL_POINT_IDX["left_shoulder"])[b])
            r_hip = to_px(self.get_point(self.FACIAL_POINT_IDX["right_hip"])[b])
            l_hip = to_px(self.get_point(self.FACIAL_POINT_IDX["left_hip"])[b])
            head = to_px(self.get_point(self.FACIAL_POINT_IDX["nose"])[b])

            pts = torch.stack([r_sh, l_sh, r_hip, l_hip])

            # 🔹 Bounding
            min_xy = pts.min(dim=0).values
            max_xy = pts.max(dim=0).values
            center = (min_xy + max_xy) / 2

            width = (max_xy[0] - min_xy[0]) * expand
            height = (max_xy[1] - head[1]) * expand * vertical_bias

            # 🔹 Ellipse corps
            dx = xx - center[0]
            dy = yy - center[1]

            ellipse = (dx / (width / 2 + 1e-6))**2 + (dy / (height / 2 + 1e-6))**2
            inside = torch.exp(-ellipse * falloff_strength)

            # 🔥 Décor = extérieur
            mask[b, 0] = 1.0 - inside

        # =========================
        # 🔹 Exclusion simple
        # =========================
        if mask_face is not None:
            mask *= (1.0 - mask_face)
        if mask_torso is not None:
            mask *= (1.0 - mask_torso)
        if mask_hair is not None:
            mask *= (1.0 - mask_hair)

        # =========================
        # 🔹 Edge falloff (clé)
        # =========================
        yy_n, xx_n = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )

        radial = torch.sqrt(xx_n**2 + yy_n**2)
        edge_falloff = (1.0 - radial).clamp(0, 1)

        mask *= edge_falloff.unsqueeze(0).unsqueeze(0)

        # =========================
        # 🔹 Smooth léger
        # =========================
        mask = torch.nn.functional.avg_pool2d(
            mask,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # =========================
        # 🔹 Clamp final
        # =========================
        mask = torch.clamp(mask, 0.0, 1.0)

        # =========================
        # 🔹 Debug
        # =========================
        if debug and debug_dir is not None:
            save_debug_mask(mask, H, W, debug_dir, frame_counter, prefix="decor_mask")

        return mask
    # -------- version pose
    def create_decor_outpose_mask( self, H: int, W: int, debug: bool = False, debug_dir: str = None, frame_counter: int = 0, expand=2.2, vertical_bias=1.2, falloff_strength=2.0 ):
        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        yy, xx = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij"
        )

        def to_px(kp):
            return kp * torch.tensor([W-1, H-1], device=self.device)

        for b in range(self.B):

            # =========================
            # 🔹 Points principaux
            # =========================
            r_sh = to_px(self.get_point(self.FACIAL_POINT_IDX["right_shoulder"])[b])
            l_sh = to_px(self.get_point(self.FACIAL_POINT_IDX["left_shoulder"])[b])
            r_hip = to_px(self.get_point(self.FACIAL_POINT_IDX["right_hip"])[b])
            l_hip = to_px(self.get_point(self.FACIAL_POINT_IDX["left_hip"])[b])

            neck = to_px(self.get_point(self.FACIAL_POINT_IDX["neck"])[b])
            head = to_px(self.get_point(self.FACIAL_POINT_IDX["nose"])[b])

            # =========================
            # 🔹 Bras (optionnel mais crucial)
            # =========================
            points_width = [r_sh, l_sh, r_hip, l_hip]

            if self.has_point("right_elbow"):
                points_width.append(to_px(self.get_point(self.FACIAL_POINT_IDX["right_elbow"])[b]))
            if self.has_point("left_elbow"):
                points_width.append(to_px(self.get_point(self.FACIAL_POINT_IDX["left_elbow"])[b]))

            pts = torch.stack(points_width)

            # =========================
            # 🔹 Bounding dynamique
            # =========================
            min_xy = pts.min(dim=0).values
            max_xy = pts.max(dim=0).values

            center = (min_xy + max_xy) / 2

            width = (max_xy[0] - min_xy[0]) * expand
            height = (max_xy[1] - head[1]) * expand * vertical_bias

            # =========================
            # 🔹 Ellipse
            # =========================
            dx = xx - center[0]
            dy = yy - center[1]

            ellipse = (dx / (width / 2 + 1e-6))**2 + (dy / (height / 2 + 1e-6))**2

            inside = torch.exp(-ellipse * falloff_strength)

            # 🔥 inversion
            outside = 1.0 - inside
            outside = torch.clamp(outside, 0, 1)

            mask[b, 0] = outside

        # =========================
        # 🔹 Feather
        # =========================
        mask = feather_inside_strict(mask, radius=8, blur_kernel=5, sigma=2.0)

        if debug and debug_dir is not None:
            save_debug_mask(mask, H, W, debug_dir, frame_counter, prefix="decor_mask")

        return mask

    def create_upper_body_mask(
        self,
        H: int,
        W: int,
        debug: bool = False,
        debug_dir: str = None,
        frame_counter: int = 0,
        expand_w=None,
        shrink_h=None,
        roundness=0.7
    ):
        """
        Masque torse PRO++ :
        - largeur adaptative épaules / hanches
        - forme elliptique anatomique
        - pondération gaussienne
        """
        # Valeurs par défaut dynamiques
        expand_w = expand_w or self.get_torce_expand_w()
        shrink_h = shrink_h or self.get_torce_shrink_h()

        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        # Grille unique
        yy, xx = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij"
        )

        def to_px(kp):
            return kp * torch.tensor([W-1, H-1], device=self.device)

        for b in range(self.B):
            r_sh = to_px(self.get_point(19)[b])
            l_sh = to_px(self.get_point(20)[b])
            r_hip = to_px(self.get_point(8)[b])
            l_hip = to_px(self.get_point(11)[b])

            # Centres anatomiques et largeur/hauteur
            shoulder_center, hip_center = (r_sh + l_sh)/2, (r_hip + l_hip)/2
            center = (shoulder_center * 0.5 + hip_center * 0.5)

            width_top = torch.norm(r_sh - l_sh) * expand_w
            width_bottom = torch.norm(r_hip - l_hip) * expand_w * 0.9
            height = torch.norm(hip_center - shoulder_center) * shrink_h

            # Ellipse adaptative
            dx, dy = xx - center[0], yy - center[1]
            t = torch.clamp((dy / height) + 0.5, 0, 1)
            width_interp = width_top * (1 - t) + width_bottom * t
            ellipse = (dx / (width_interp / 2))**2 + (dy / (height / 2))**2

            mask[b,0] = torch.exp(-ellipse * 2.5)  # Gaussian falloff

        # Clamp + feather
        mask = torch.clamp(mask, 0, 1)
        mask = feather_inside_strict(mask, radius=5, blur_kernel=3, sigma=1.2)

        # Debug
        if debug and debug_dir is not None:
            save_debug_mask(mask, H, W, debug_dir, frame_counter, prefix="torso_mask_PRO")

        return mask

    def create_upper_body_mask_v2(
        self,
        H: int,
        W: int,
        debug: bool = False,
        debug_dir: str = None,
        frame_counter: int = 0,
        expand_w=1.0,
        shrink_h=0.9,
        roundness=0.6
    ):
        """
        Masque torse PRO : capsule (rectangle + ellipse), plus naturel que polygon
        """
        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        def to_px(kp):
            return np.array([kp[0]*(W-1), kp[1]*(H-1)])

        for b in range(self.B):
            # Points clés
            pts = np.array([
                to_px(self.get_point(19)[b].cpu().numpy()),  # r_sh
                to_px(self.get_point(20)[b].cpu().numpy()),  # l_sh
                to_px(self.get_point(8)[b].cpu().numpy()),   # r_hip
                to_px(self.get_point(11)[b].cpu().numpy())   # l_hip
            ])

            # Centre et dimensions
            x_min, y_min = pts[:,0].min(), pts[:,1].min()
            x_max, y_max = pts[:,0].max(), pts[:,1].max()
            cx, cy = (x_min + x_max)/2, (y_min + y_max)/2
            width, height = (x_max - x_min) * expand_w, (y_max - y_min) * shrink_h

            # Rectangle central
            rect_h = int(height * (1 - roundness))
            rect_w = int(width)
            x0, y0 = int(cx - rect_w/2), int(cy - rect_h/2)

            mask_np = np.zeros((H, W), dtype=np.uint8)
            cv2.rectangle(mask_np, (x0, y0), (x0 + rect_w, y0 + rect_h), 255, -1)

            # Ellipses haut/bas
            ellipse_h = int(height * roundness)
            for y_center, start_angle, end_angle in [(y0, 0, 180), (y0 + rect_h, 180, 360)]:
                cv2.ellipse(mask_np, (int(cx), int(y_center)), (rect_w//2, ellipse_h), 0, start_angle, end_angle, 255, -1)

            # Morphologie pour lisser
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

            mask[b,0] = torch.from_numpy(mask_np / 255.0).to(self.device)

        # Feather final
        mask = feather_inside_strict(mask, radius=5, blur_kernel=3, sigma=1.2)

        # Debug
        if debug and debug_dir is not None:
            save_debug_mask(mask, H, W, debug_dir, frame_counter, prefix="torso_mask_")

        return mask

    def create_upper_body_mask_v1(self, H: int, W: int,
                            debug: bool = False, debug_dir: str = None, frame_counter: int = 0,
                            expand_w=0.95, shrink_h=0.70):
        """
        Crée un masque polygonal flouté torse uniquement, basé sur épaules + hanches.
        expand_w: facteur pour élargir le masque horizontalement (>1 = plus large)
        shrink_h: facteur pour réduire le masque verticalement (<1 = plus petit)
        """

        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        for b in range(self.B):
            # Récupère les keypoints : épaules et hanches
            r_sh = self.get_point(19)[b].cpu().numpy()
            l_sh = self.get_point(20)[b].cpu().numpy()
            r_hip = self.get_point(8)[b].cpu().numpy()
            l_hip = self.get_point(11)[b].cpu().numpy()

            # Convertir en pixels
            def to_px(kp):
                return np.array([kp[0]*(W-1), kp[1]*(H-1)])

            pts = np.array([
                to_px(r_sh),
                to_px(l_sh),
                to_px(l_hip),
                to_px(r_hip)
            ], dtype=np.float32)

            # 🔹 Ajuster largeur et hauteur
            # Centre horizontal et vertical du polygone
            cx = np.mean(pts[:,0])
            cy = np.mean(pts[:,1])

            # Appliquer le facteur
            pts[:,0] = cx + (pts[:,0] - cx) * expand_w    # élargir horizontalement
            pts[:,1] = cy + (pts[:,1] - cy) * shrink_h    # réduire verticalement

            # Convertir en int pour cv2
            pts = pts.astype(np.int32)

            # Remplir le polygone
            mask_np = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask_np, [pts], 255)

            # Convertir en tensor
            mask[b,0] = torch.from_numpy(mask_np / 255.0).to(self.device)

        # Appliquer feather intérieur strict
        mask = feather_inside_strict(mask, radius=5, blur_kernel=3, sigma=1.0)

        # -------------------- Debug --------------------
        if debug and debug_dir is not None:
            save_debug_mask(mask, H, W, debug_dir, frame_counter, prefix="skeleton_mask_")

        return mask


    def create_hair_mask(self, H: int, W: int,
                             debug: bool = False, debug_dir: str = None, frame_counter: int = 0,
                             top_extend=0.5, side_extend=0.2, height_factor=1.0):
        """
        Crée un masque cheveux en forme d'ellipse au-dessus du visage.
        - top_extend : fraction de la hauteur du visage à ajouter au-dessus pour les cheveux
        - side_extend : fraction de la largeur du visage à ajouter sur les côtés
        - height_factor : facteur d'étirement vertical de l'ellipse
        """
        mask_face = self.create_face_mask(H, W, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter)

        mask_hair = torch.zeros_like(mask_face)

        for b in range(self.B):
            coords = torch.nonzero(mask_face[b, 0], as_tuple=False)
            if coords.shape[0] == 0:
                continue
            y_min, x_min = coords.min(dim=0)[0]
            y_max, x_max = coords.max(dim=0)[0]

            h_face = y_max - y_min + 1
            w_face = x_max - x_min + 1

            # Centre de l'ellipse
            cx = (x_min + x_max) / 2
            cy = y_min - h_face * top_extend / 2  # centre légèrement au-dessus du visage

            # Rayons de l'ellipse
            rx = w_face / 2 * (1 + side_extend)
            ry = h_face / 2 * (1 + top_extend) * height_factor

            # Créer masque numpy pour l'ellipse
            mask_np = np.zeros((H, W), dtype=np.uint8)
            cv2.ellipse(mask_np,
                        (int(cx), int(cy)),
                        (int(rx), int(ry)),
                        angle=0,
                        startAngle=0,
                        endAngle=360,
                        color=255,
                        thickness=-1)

            # Convertir en tensor
            mask_hair[b, 0] = torch.from_numpy(mask_np / 255.0).to(self.device)

        # Retirer le visage
        mask_hair = mask_hair * (1 - mask_face)

        # Intensification
        mask_hair = torch.clamp(mask_hair, 0, 1)
        mask_hair = mask_hair ** 2.5

        # Feather léger
        mask_hair = feather_outside_only_alpha(mask_hair, radius=3, sigma=1.5)

        if debug and debug_dir is not None:
            save_debug_mask(mask_hair, H, W, debug_dir, frame_counter, prefix="hair_mask_ellipse_")

        return mask_hair
    # version dynamique pro
    def create_left_eye_mask(self, H: int, W: int, debug=False, debug_dir=None, frame_counter=0, expand_w=0.2, expand_h=0.2):
        """
        Masque dynamique pour l’œil gauche, ovale réaliste.
        """
        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        for b in range(self.B):
            points = [self.get_point(15)[b].cpu().numpy()]  # left_eye keypoints
            pts = np.array([[p[0]*(W-1), p[1]*(H-1)] for p in points], dtype=np.float32)

            if len(pts) == 0:
                continue

            cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
            w_eye = np.max(pts[:,0]) - np.min(pts[:,0])
            h_eye = np.max(pts[:,1]) - np.min(pts[:,1])

            # Expansion proportionnelle
            w_eye *= (1.0 + expand_w)
            h_eye *= (1.0 + expand_h)

            mask_np = np.zeros((H, W), dtype=np.uint8)
            center = (int(cx), int(cy))
            axes = (max(1, int(w_eye/2)), max(1, int(h_eye/2)))
            cv2.ellipse(mask_np, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

            mask[b,0] = torch.from_numpy(mask_np/255.0).to(self.device)

        # Feather interne/externe pour adoucir
        mask = feather_inside_strict(mask, radius=2, blur_kernel=3, sigma=0.8)
        mask = feather_outside_only_alpha(mask, radius=2, sigma=1.2)

        if debug and debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            save_path = os.path.join(debug_dir, f"left_eye_mask_{frame_counter:05d}.png")
            mask_img = (mask[0,0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(save_path)
            print(f"[DEBUG] Left eye mask saved: {save_path}")

        return mask


    def create_right_eye_mask(self, H: int, W: int, debug=False, debug_dir=None, frame_counter=0, expand_w=0.2, expand_h=0.2):
        """
        Masque dynamique pour l’œil droit, ovale réaliste.
        """

        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        for b in range(self.B):
            points = [self.get_point(14)[b].cpu().numpy()]  # right_eye keypoints
            pts = np.array([[p[0]*(W-1), p[1]*(H-1)] for p in points], dtype=np.float32)

            if len(pts) == 0:
                continue

            cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
            w_eye = np.max(pts[:,0]) - np.min(pts[:,0])
            h_eye = np.max(pts[:,1]) - np.min(pts[:,1])

            # Expansion proportionnelle
            w_eye *= (1.0 + expand_w)
            h_eye *= (1.0 + expand_h)

            mask_np = np.zeros((H, W), dtype=np.uint8)
            center = (int(cx), int(cy))
            axes = (max(1, int(w_eye/2)), max(1, int(h_eye/2)))
            cv2.ellipse(mask_np, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

            mask[b,0] = torch.from_numpy(mask_np/255.0).to(self.device)

        # Feather interne/externe pour adoucir
        mask = feather_inside_strict(mask, radius=2, blur_kernel=3, sigma=0.8)
        mask = feather_outside_only_alpha(mask, radius=2, sigma=1.2)

        if debug and debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            save_path = os.path.join(debug_dir, f"right_eye_mask_{frame_counter:05d}.png")
            mask_img = (mask[0,0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(save_path)
            print(f"[DEBUG] Right eye mask saved: {save_path}")

        return mask
    #-----------------------------------------------------------------------------------------------------------------------------------
    def create_mouth_mask(self, H: int, W: int, debug=False, debug_dir=None, frame_counter=0, expand_w=0.2, expand_h=0.2):
        """
        Masque dynamique pour la bouche uniquement, arrondi avec bord glow.
        expand_w / expand_h permettent d'élargir le rectangle de la bouche.
        Retourne également les points calculés de la bouche pour suivi.
        """
        mask = torch.zeros(self.B, 1, H, W, device=self.device)
        mouth_points_batch = []

        for b in range(self.B):
            # Récupération du centre approximatif de la bouche
            mouth_center = self.get_point('mouth')[b].cpu().numpy()
            cx = mouth_center[0] * (W-1)
            cy = mouth_center[1] * (H-1)

            # Dimensions approximatives de la bouche
            w_mouth = 0.06 * W * (1 + expand_w)
            h_mouth = 0.04 * H * (1 + expand_h)

            # Calcul des coins et haut/bas
            mouth_left   = np.array([cx - w_mouth/2, cy])
            mouth_right  = np.array([cx + w_mouth/2, cy])
            mouth_top    = np.array([cx, cy - h_mouth/2])
            mouth_bottom = np.array([cx, cy + h_mouth/2])

            mouth_points_batch.append({
                'mouth_center': mouth_center,
                'mouth_left': mouth_left / [W-1, H-1],
                'mouth_right': mouth_right / [W-1, H-1],
                'mouth_top': mouth_top / [W-1, H-1],
                'mouth_bottom': mouth_bottom / [W-1, H-1],
            })

            # Création d'un ovale pour la bouche
            mask_np = np.zeros((H, W), dtype=np.uint8)
            center = (int(cx), int(cy))
            axes = (int(w_mouth/2), int(h_mouth/2))
            cv2.ellipse(mask_np, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

            # Conversion en tensor et ajout au batch
            mask[b,0] = torch.from_numpy(mask_np/255.0).to(self.device)

        # Feather interne/externe pour adoucir le bord
        mask = feather_inside_strict(mask, radius=3, blur_kernel=3, sigma=1.0)
        mask = feather_outside_only_alpha(mask, radius=3, sigma=1.5)

        # Debug
        if debug and debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            save_path = os.path.join(debug_dir, f"mouth_mask_{frame_counter:05d}.png")
            mask_img = (mask[0,0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(save_path)
            print(f"[DEBUG] Mouth mask saved: {save_path}")

        return mask, mouth_points_batch
    #-----------------------------------------------------------------------------------------------------------------------------------
    def create_mouth_corners_mask(self, H: int, W: int, debug=False, debug_dir=None, frame_counter=0, expand_w=0.2, expand_h=0.2):
        """
        Masque dynamique pour les coins de la bouche uniquement (pour sourire subtil, micro-mouvements).
        Les coins sont calculés à partir du point central de la bouche.

        Args:
            H, W: hauteur et largeur de l'image/latent
            expand_w, expand_h: élargissement du masque horizontal/vertical
            debug: si True, sauvegarde l'image du masque
            debug_dir: répertoire de sauvegarde debug
            frame_counter: numéro de frame (pour debug)

        Returns:
            mask (torch.Tensor): [B,1,H,W] masque des coins de bouche
            corners_points_batch (list[dict]): coordonnées normalisées des coins par batch
        """
        mask = torch.zeros(self.B, 1, H, W, device=self.device)
        corners_points_batch = []

        for b in range(self.B):
            # Récupération du centre approximatif de la bouche
            mouth_center = self.get_point('mouth')[b].cpu().numpy()
            cx = mouth_center[0] * (W-1)
            cy = mouth_center[1] * (H-1)

            # Dimensions approximatives pour coins de bouche
            w_corner = 0.03 * W * (1 + expand_w)
            h_corner = 0.02 * H * (1 + expand_h)

            # Calcul coins gauche/droite
            mouth_left  = np.array([cx - w_corner/2, cy])
            mouth_right = np.array([cx + w_corner/2, cy])

            corners_points_batch.append({
                'mouth_center': mouth_center,
                'mouth_left': mouth_left / [W-1, H-1],
                'mouth_right': mouth_right / [W-1, H-1],
            })

            # Création masque ovale pour les coins
            mask_np = np.zeros((H, W), dtype=np.uint8)
            cv2.ellipse(mask_np, (int(mouth_left[0]), int(mouth_left[1])), (int(w_corner/2), int(h_corner/2)), 0, 0, 360, 255, -1)
            cv2.ellipse(mask_np, (int(mouth_right[0]), int(mouth_right[1])), (int(w_corner/2), int(h_corner/2)), 0, 0, 360, 255, -1)

            mask[b,0] = torch.from_numpy(mask_np/255.0).to(self.device)

        # Feather interne/externe pour adoucir le bord
        mask = feather_inside_strict(mask, radius=2, blur_kernel=3, sigma=1.0)
        mask = feather_outside_only_alpha(mask, radius=2, sigma=1.0)

        # Debug
        if debug and debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            save_path = os.path.join(debug_dir, f"mouth_corners_mask_{frame_counter:05d}.png")
            mask_img = (mask[0,0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(save_path)
            print(f"[DEBUG] Mouth corners mask saved: {save_path}")

        return mask, corners_points_batch
    #-----------------------------------------------------------------------------------------------------------------------------------
    def create_face_mask(self, H: int, W: int, debug=False, debug_dir=None, frame_counter=0):
        """
        Masque facial dynamique et professionnel, en forme ovale réaliste, incluant bouche.
        """
        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        # Masque bouche
        mouth_mask = self.get_mouth_region(
            H, W, device=self.device, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter,
            expand_w=0.2, expand_h=0.2
        )

        for b in range(self.B):
            # Points clés du visage (yeux, nez, oreilles)
            points = [
                self.get_point(14)[b].cpu().numpy(),  # right_eye
                self.get_point(15)[b].cpu().numpy(),  # left_eye
                self.get_point(0)[b].cpu().numpy(),   # nose
                self.get_point(16)[b].cpu().numpy(),  # right_ear
                self.get_point(17)[b].cpu().numpy()   # left_ear
            ]
            pts = np.array([[p[0]*(W-1), p[1]*(H-1)] for p in points], dtype=np.float32)

            # Centre et dimensions de l'ovale
            cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
            w_face = np.max(pts[:,0]) - np.min(pts[:,0])
            h_face = np.max(pts[:,1]) - np.min(pts[:,1])

            # Ajustement pour un ovale réaliste (plus haut que large)
            w_face *= 1.15  # élargir légèrement
            h_face *= 1.4   # hauteur accentuée

            # Création masque ovale
            mask_np = np.zeros((H, W), dtype=np.uint8)
            center = (int(cx), int(cy))
            axes = (int(w_face/2), int(h_face/2))
            cv2.ellipse(mask_np, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

            # Combiner avec le masque bouche
            mask[b,0] = torch.maximum(torch.from_numpy(mask_np/255.0).to(self.device), mouth_mask[b,0])

        # Feather interne et externe pour adoucir
        mask = feather_inside_strict(mask, radius=3, blur_kernel=3, sigma=1.0)
        mask = feather_outside_only_alpha(mask, radius=3, sigma=1.5)

        # Debug

        if debug and debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            save_path = os.path.join(debug_dir, f"face_mask_pro_{frame_counter:05d}.png")
            mask_img = (mask[0,0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(save_path)
            print(f"[DEBUG] Professional face mask saved: {save_path}")

        return mask

        # -------------------- Debug --------------------
        if debug and debug_dir is not None:
            save_debug_mask(mask, H, W, debug_dir, frame_counter, prefix="face_mask_")

        return mask

# ----------------------------------------------------------------------------------------------------------------------------------
# Pose PoseAnimator en DEV
# ----------------------------------------------------------------------------------------------------------------------------------

class PoseAnimator:
    def __init__(self, pose: Pose, latent_h: int, latent_w: int):
        self.pose = pose
        self.H = latent_h
        self.W = latent_w

    # ----------------- Préparer les masques -----------------
    def prepare_masks(self, debug=False, debug_dir=None, frame_counter=0):
        # Masque torse
        self.torso_mask = self.pose.create_upper_body_mask(
            H=self.H, W=self.W,
            expand_w=0.9, shrink_h=0.65,
            debug=debug, debug_dir=debug_dir, frame_counter=frame_counter
        )

        # Masque visage dynamique
        self.face_mask = self.pose.create_face_mask(
            H=self.H, W=self.W,
            debug=debug, debug_dir=debug_dir, frame_counter=frame_counter
        )

        # Masque cheveux
        self.hair_mask = self.pose.create_hair_mask(
            H=self.H, W=self.W,
            debug=debug, debug_dir=debug_dir, frame_counter=frame_counter
        )

    # ----------------- Calcul des deltas -----------------
    def compute_deltas(self, torso_scale=1.0, face_scale=0.3, x_factor=0.3):
        # Torse
        self.pose.compute_torso_delta(latent_h=self.H, latent_w=self.W)
        self.torso_delta = self.pose.delta.clone()
        self.torso_delta[:,0] *= x_factor  # limiter le déplacement X

        # Face : micro-mouvements
        face_center = (self.pose.get_point(14) + self.pose.get_point(15) + self.pose.get_point(0)) / 3.0
        face_delta = face_center * face_scale
        face_delta = torch.tanh(face_delta * 2.0) * 0.3
        self.face_delta = face_delta.clone()
        self.face_delta[:,0] *= x_factor  # réduire X

    # ----------------- Appliquer les deltas sur le latent -----------------
    def apply_to_latent(self, latent: torch.Tensor):
        """
        latent: [B, C, H, W]
        Retourne latent avec torse + visage animés
        """

        # Copier pour éviter d'écraser
        out = latent.clone()

        # Appliquer delta torse
        torso_mask = self.torso_mask
        out = self.grid_warp(out, torso_mask, self.torso_delta)

        # Appliquer delta visage
        face_mask = self.face_mask
        out = self.grid_warp(out, face_mask, self.face_delta)

        return out

    # ----------------- Grid warp localisé -----------------
    @staticmethod
    def grid_warp(latent, mask, delta):
        """
        Applique un déplacement (delta) uniquement sur la zone du mask
        """
        B, C, H, W = latent.shape

        # Créer grid [B,H,W,2]
        yy, xx = torch.meshgrid(torch.arange(H, device=latent.device),
                                torch.arange(W, device=latent.device),
                                indexing='ij')
        grid = torch.stack([xx.float(), yy.float()], dim=-1)  # [H,W,2]
        grid = grid.unsqueeze(0).repeat(B,1,1,1)  # [B,H,W,2]

        # Ajouter delta multiplié par mask
        mask_expand = mask.permute(0,2,3,1)  # [B,H,W,1]
        delta_expand = delta.unsqueeze(1).unsqueeze(1)  # [B,1,1,2]
        grid = grid + delta_expand * mask_expand

        # Normaliser grid entre -1 et 1
        grid_norm = grid.clone()
        grid_norm[...,0] = 2.0 * grid[...,0] / (W-1) - 1.0
        grid_norm[...,1] = 2.0 * grid[...,1] / (H-1) - 1.0

        # Sample latent
        out = torch.nn.functional.grid_sample(latent, grid_norm, align_corners=True)
        return out
