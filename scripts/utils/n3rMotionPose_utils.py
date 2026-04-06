#********************************************
# n3rOpenPose_utils.py
#********************************************
import torch
from diffusers import ControlNetModel
import math
import torch.nn.functional as F
from .n3rControlNet import create_canny_control, control_to_latent, match_latent_size
from .tools_utils import ensure_4_channels, print_generation_params, sanitize_latents
from .n3rMotionPose_tools import gaussian_blur_tensor, feather_mask, feather_mask_fast, feather_outside_only, feather_inside,feather_inside_strict, debug_draw_openpose_skeleton, rotate_mask_around_torso_simple, rotate_mask_around_visage, save_impact_map, apply_breathing_xy, apply_breathing, feather_outside_only_alpha, smooth_noise, feather_dynamic_vectorized
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import traceback
from torchvision.utils import save_image



#------extract_keypoints_from_pose

def extract_keypoints_from_pose(
    pose_full_image,
    device="cuda",
    debug=False,
    debug_dir=None,
    frame_counter=None
):
    """
    FR:
    Extraction MANUELLE des keypoints (format COCO 18 points).
    Les coordonnées sont normalisées entre [0,1] dans l'espace IMAGE.

    EN:
    MANUAL keypoints extraction (COCO 18 format).
    Coordinates are normalized in [0,1] in IMAGE space.

    Output:
        keypoints_tensor: [B, 18, 3]  (x, y, confidence)
    """

    B, C, H, W = pose_full_image.shape  # H=height, W=width

    # ---------------------------
    # 🔥 KEYPOINTS MANUELS / MANUAL KEYPOINTS
    # ---------------------------
    # FR:
    # x = pixel_x / image_width
    # y = pixel_y / image_height
    # IMPORTANT: utiliser la taille de l'image originale (pose_full), PAS le latent

    # EN:
    # x = pixel_x / image_width
    # y = pixel_y / image_height
    # IMPORTANT: use original image size (pose_full), NOT latent size

    # 👄 Mouth detected: [(404, 446)]

    keypoints_template = [
        [418/896, 418/1280, 1.0],  # 0 nose / nez (ok) 👃 Nose detected: [(422, 408)]
        [383/896, 515/1280, 1.0],  # 1 neck / cou 🦵 Neck detected: [(420, 518)]

        [627/896, 533/1280, 1.0],  # 2 right_shoulder / épaule droite 🦾 Shoulders detected: [(77.5, 576.2), (761.5, 542.6)]
        [612/896, 838/1280, 1.0],  # 3 right_elbow / coude droit 🦾 Elbows detected/estimated: [[179, 896], [627, 896]]
        [488/896, 1040/1280, 1.0], # 4 right_wrist / poignet droit ✋ Wrists detected: [(179, 1152.0), (627, 1152.0)]

        [121/896, 553/1280, 1.0],  # 5 left_shoulder / épaule gauche 🦾 Shoulders detected: [(77.5, 576.2), (761.5, 542.6)]
        [197/896, 944/1280, 1.0],  # 6 left_elbow / coude gauche 🦾 Elbows detected/estimated: [[179, 896], [627, 896]]
        [431/896, 1087/1280, 1.0], # 7 left_wrist / poignet gauche ✋ Wrists detected: [(179, 1152.0), (627, 1152.0)]

        [619/896, 1048/1280, 1.0], # 8 right_hip / hanche droite  🦿📍 Hips hanches detected: left=(564, 1102), right=(308, 1129)
        [0.0, 0.0, 0.0],           # 9 right_knee (absent)
        [0.0, 0.0, 0.0],           # 10 right_ankle (absent)

        [260/896, 1139/1280, 1.0], # 11 left_hip / hanche gauche 🦿📍 Hips hanches detected: left=(564, 1102), right=(308, 1129)
        [0.0, 0.0, 0.0],           # 12 left_knee (absent)
        [0.0, 0.0, 0.0],           # 13 left_ankle (absent)

        [359/896, 490/1280, 1.0],  # 14 right_eye / œil droit - 👁 Eyes detected: (ok)
        [379/896, 326/1280, 1.0],  # 15 left_eye / œil gauche - 👁 Eyes detected: [(326, 379), (359, 490)]
        [608/896, 304/1280, 1.0],  # 16 right_ear / oreille droite
        [290/896, 244/1280, 1.0],  # 17 left_ear / oreille gauche
        [404/896, 446/1280, 1.0],  # 18 mouth / Bouche (ok) - 👄 Mouth detected: [(404, 446)]
        [562/896, 514/1280, 1.0],  # 19 🦾 right_clavicules detected: [562/896, 514/1280, 1.0],
        [277/896, 528/1280, 1.0],  # 20 🦾 left_clavicules detected: [(277, 528), (562, 514)]
    ]

    #48–54 : lèvres supérieures (coin gauche → coin droit)
    #54–60 : lèvres inférieures (coin droit → coin gauche)
    #60–67 : contour interne de la bouche (pour l’ouverture, micro-mouvements)

    # ---------------------------
    # 🔹 Conversion numpy → tensor
    # ---------------------------
    keypoints_np = np.array(keypoints_template, dtype=np.float32)

    # FR: sécurité pour éviter valeurs hors [0,1]
    # EN: safety clamp to keep values in [0,1]
    keypoints_np = np.clip(keypoints_np, 0.0, 1.0)

    # FR: duplication pour batch
    # EN: repeat for batch
    keypoints_np = np.expand_dims(keypoints_np, axis=0)  # [1,18,3]
    keypoints_np = np.repeat(keypoints_np, B, axis=0)    # [B,18,3]

    keypoints_tensor = torch.from_numpy(keypoints_np).to(device)

    # ---------------------------
    # 🔹 DEBUG VISUEL / VISUAL DEBUG
    # ---------------------------
    if debug and debug_dir is not None and frame_counter is not None:
        # Affichage des points en debug
        debug_draw_openpose_skeleton(
            pose_full_image=pose_full_image,
            keypoints_tensor=keypoints_tensor,
            debug_dir=debug_dir,
            frame_counter=frame_counter
        )

    return keypoints_tensor



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

    # =========================
    # GETTER
    # =========================
    def get_prev_facial_points(self):
        return self._prev_facial_points

    # =========================
    # SETTER
    # =========================
    def set_prev_facial_points(self, points):
        self._prev_facial_points = points

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
        à partir des points existants (nez, yeux, oreilles).

        Retourne un dictionnaire {nom_point: tensor [B,2]}.
        """
        estimated_points = {}
        B = self.B
        device = self.device

        # ----------------- BOUCHE -----------------
        # On suppose que le point 0 = nez
        nose = self.keypoints[:, 0, :2]  # [B,2]

        # Points yeux si existants
        right_eye = self.keypoints[:, 14, :2] if self.keypoints.shape[1] > 14 else None
        left_eye = self.keypoints[:, 15, :2] if self.keypoints.shape[1] > 15 else None

        # Base pour la bouche : légèrement sous le nez
        mouth_center = nose.clone()
        mouth_center[:,1] += 0.12  # proportion approximative verticale

        # Largeur de la bouche estimée à partir distance yeux
        if right_eye is not None and left_eye is not None:
            eye_dist = (left_eye[:,0] - right_eye[:,0]).abs()
        else:
            eye_dist = torch.tensor(0.12, device=device).expand(B)  # fallback

        # Coins gauche/droite
        mouth_left = mouth_center.clone()
        mouth_left[:,0] -= eye_dist * 0.3
        mouth_right = mouth_center.clone()
        mouth_right[:,0] += eye_dist * 0.3

        # Haut/bas
        mouth_top = mouth_center.clone()
        mouth_top[:,1] -= eye_dist * 0.15
        mouth_bottom = mouth_center.clone()
        mouth_bottom[:,1] += eye_dist * 0.15

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
        """Récupère un keypoint spécifique [B,2]"""
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
    def get_mouth_region(self, H: int, W: int, device=None,
                     debug: bool = False, debug_dir: str = None, frame_counter: int = 0,
                     expand=0.1):
        """
        Retourne un masque bouche [B,1,H,W] basé sur les points estimés.
        - expand : fraction de largeur/hauteur pour agrandir le masque légèrement
        """
        if device is None:
            device = self.device

        B = self.B
        mask = torch.zeros((B, 1, H, W), device=device)

        # Estimation des points de la bouche
        points_dict = self.estimate_missing_facial_points()
        mouth_left = points_dict['mouth_left']
        mouth_right = points_dict['mouth_right']
        mouth_top = points_dict['mouth_top']
        mouth_bottom = points_dict['mouth_bottom']

        for b in range(B):
            # Calcul coordonnées en pixels
            x_min = int((mouth_left[b,0] - expand) * (W-1))
            x_max = int((mouth_right[b,0] + expand) * (W-1))
            y_min = int((mouth_top[b,1] - expand) * (H-1))
            y_max = int((mouth_bottom[b,1] + expand) * (H-1))

            # Clamp pour rester dans l'image
            x_min = max(0, x_min)
            x_max = min(W-1, x_max)
            y_min = max(0, y_min)
            y_max = min(H-1, y_max)

            # Remplir le masque
            mask[b,0,y_min:y_max+1,x_min:x_max+1] = 1.0

        # Feather léger pour adoucir les bords
        mask = feather_outside_only_alpha(mask, radius=3, sigma=1.5)

        # Debug
        if debug and debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            mask_np = (mask[0,0].detach().cpu().numpy() * 255).astype(np.uint8)
            mask_debug = cv2.resize(mask_np, (W*4,H*4), interpolation=cv2.INTER_NEAREST)
            mask_debug_rgb = cv2.cvtColor(mask_debug, cv2.COLOR_GRAY2BGR)
            save_path = os.path.join(debug_dir, f"mouth_mask_{frame_counter:05d}.png")
            cv2.imwrite(save_path, mask_debug_rgb)
            print(f"[DEBUG] Mouth mask saved (scale 4): {save_path}")

        return mask


    # ----------------- Torso delta -----------------
    # Attention le expand_w et le shrink_h doit être similaire pour compute_torso_delta et create_upper_body_mask
    def compute_torso_delta(self, latent_h: int, latent_w: int, expand_w=0.95, shrink_h=0.70):
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

    def create_upper_body_mask(self, H: int, W: int,
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
            os.makedirs(debug_dir, exist_ok=True)
            debug_scale = 4
            mask_np_debug = (mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            mask_debug = cv2.resize(mask_np_debug, (W*debug_scale, H*debug_scale), interpolation=cv2.INTER_NEAREST)
            mask_debug_rgb = cv2.cvtColor(mask_debug, cv2.COLOR_GRAY2BGR)
            save_path = os.path.join(debug_dir, f"skeleton_mask_{frame_counter:05d}.png")
            cv2.imwrite(save_path, mask_debug_rgb)
            print(f"[DEBUG] Upper body mask saved (scale {debug_scale}): {save_path}")

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
            os.makedirs(debug_dir, exist_ok=True)
            debug_scale = 4
            mask_np_debug = (mask_hair[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            mask_debug = cv2.resize(mask_np_debug, (W*debug_scale, H*debug_scale), interpolation=cv2.INTER_NEAREST)
            mask_debug_rgb = cv2.cvtColor(mask_debug, cv2.COLOR_GRAY2BGR)
            save_path = os.path.join(debug_dir, f"hair_mask_ellipse_{frame_counter:05d}.png")
            cv2.imwrite(save_path, mask_debug_rgb)
            print(f"[DEBUG] Hair ellipse mask saved (scale {debug_scale}): {save_path}")

        return mask_hair

    def create_face_mask(self, H: int, W: int,
                     debug: bool = False, debug_dir: str = None, frame_counter: int = 0,
                     expand_w=1.1, expand_h=1.2):
        """
        Crée un masque polygonal pour le visage (yeux, nez, bouche, oreilles).
        expand_w / expand_h: élargir ou augmenter la hauteur du polygone pour inclure cheveux/front.
        """
        mask = torch.zeros(self.B, 1, H, W, device=self.device)

        for b in range(self.B):
            # Points clés du visage
            points = [
                self.get_point(14)[b].cpu().numpy(),  # right_eye
                self.get_point(15)[b].cpu().numpy(),  # left_eye
                self.get_point(0)[b].cpu().numpy(),   # nose
                self.get_point(18)[b].cpu().numpy(),  # mouth
                self.get_point(16)[b].cpu().numpy(),  # right_ear
                self.get_point(17)[b].cpu().numpy()   # left_ear
            ]

            pts = np.array([ [p[0]*(W-1), p[1]*(H-1)] for p in points ], dtype=np.float32)

            # Centre du polygone
            cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])

            # Ajustement largeur / hauteur
            pts[:,0] = cx + (pts[:,0] - cx) * expand_w
            pts[:,1] = cy + (pts[:,1] - cy) * expand_h

            # Convertir en int pour cv2
            pts = pts.astype(np.int32)

            # Remplir le polygone
            mask_np = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask_np, [pts], 255)

            # Convertir en tensor
            mask[b,0] = torch.from_numpy(mask_np / 255.0).to(self.device)

        # Feather interne strict
        mask = feather_inside_strict(mask, radius=3, blur_kernel=3, sigma=1.0)

        # -------------------- Debug --------------------
        if debug and debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            debug_scale = 4
            mask_np_debug = (mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            mask_debug = cv2.resize(mask_np_debug, (W*debug_scale, H*debug_scale), interpolation=cv2.INTER_NEAREST)
            mask_debug_rgb = cv2.cvtColor(mask_debug, cv2.COLOR_GRAY2BGR)
            save_path = os.path.join(debug_dir, f"face_mask_{frame_counter:05d}.png")
            cv2.imwrite(save_path, mask_debug_rgb)
            print(f"[DEBUG] Face mask saved (scale {debug_scale}): {save_path}")

        return mask
#----------------------------------------------------------------------------------------------------------------------
class PoseAnimator:
    def __init__(self, pose: Pose, latent_h: int, latent_w: int):
        self.pose = pose
        self.H = latent_h
        self.W = latent_w

    # ----------------- Préparer les masques -----------------
    def prepare_masks(self):
        # Masque torse
        self.torso_mask = self.pose.create_upper_body_mask(
            H=self.H, W=self.W,
            expand_w=0.9, shrink_h=0.65
        )

        # Masque visage
        self.face_mask = self.pose.create_face_mask(
            H=self.H, W=self.W,
            expand_w=1.1, expand_h=1.2
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
#----------------------------------------------------------------------------------------------------------------

def resize_pose(pose_tile, H_latent, W_latent):
    target_h = H_latent * 8
    target_w = W_latent * 8

    if pose_tile.shape[-2:] != (target_h, target_w):
        return F.interpolate(
            pose_tile,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
    return pose_tile


def prepare_inputs(latent_tile, pose_tile, cf_embeds, device):
    pos_embeds, neg_embeds = cf_embeds

    latent_fp32 = latent_tile.to(device=device, dtype=torch.float32)
    pose_fp32 = pose_tile.to(device=device, dtype=torch.float32)

    pos_fp32 = pos_embeds.to(device=device, dtype=torch.float32)
    neg_fp32 = neg_embeds.to(device=device, dtype=torch.float32) if neg_embeds is not None else None

    return latent_fp32, pose_fp32, pos_fp32, neg_fp32


def add_noise(latent, scheduler, t, noise_strength=0.5):
    noise = torch.randn_like(latent) * noise_strength
    latent_noisy = scheduler.add_noise(latent, noise, t)
    return torch.clamp(latent_noisy, -20, 20)


def apply_cfg(latent_input, pos_embeds, neg_embeds, guidance_scale):
    if neg_embeds is not None:
        latent_input = torch.cat([latent_input] * 2)
        embeds = torch.cat([neg_embeds, pos_embeds])
        return latent_input, embeds, True
    return latent_input, pos_embeds, False


def compute_noise_pred(unet, controlnet, latent_input, t, embeds, pose):
    down_samples, mid_sample = controlnet(
        latent_input,
        t,
        encoder_hidden_states=embeds,
        controlnet_cond=pose,
        return_dict=False
    )

    noise_pred = unet(
        latent_input,
        t,
        encoder_hidden_states=embeds,
        down_block_additional_residuals=down_samples,
        mid_block_additional_residual=mid_sample,
        return_dict=False
    )[0]

    return torch.nan_to_num(noise_pred, nan=0.0, posinf=1.0, neginf=-1.0)


def merge_cfg(noise_pred, guidance_scale, use_cfg):
    if use_cfg:
        noise_uncond, noise_text = noise_pred.chunk(2)
        return noise_uncond + guidance_scale * (noise_text - noise_uncond)
    return noise_pred


def compute_adaptive_importance(latent):
    with torch.no_grad():
        blurred = F.avg_pool2d(latent, kernel_size=3, stride=1, padding=1)
        high_freq = torch.abs(latent - blurred)
        importance = high_freq.mean(dim=1, keepdim=True)
        # normalisation douce
        importance = importance / (importance.mean() + 1e-6)
        # compression pour éviter extrêmes
        importance = torch.sqrt(importance)
        return torch.clamp(importance, 0.7, 1.3)


def compute_delta(latents_out, latent_ref, controlnet_scale, importance):
    delta = latents_out - latent_ref
    delta = torch.nan_to_num(delta, nan=0.0, posinf=1.0, neginf=-1.0)
    # 🔥 adaptive blending ici
    delta = torch.tanh(delta) * 0.15 * importance
    return delta * controlnet_scale


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

    B, C, H_latent, W_latent = latent_tile.shape

    # =========================================================
    # 1️⃣ Resize pose
    # =========================================================
    pose_resized = resize_pose(pose_tile, H_latent, W_latent)
    # =========================================================
    # 2️⃣ Inputs
    # =========================================================
    latent_fp32, pose_fp32, pos_embeds, neg_embeds = prepare_inputs(
        latent_tile, pose_resized, cf_embeds, device
    )
    # =========================================================
    # 3️⃣ Timestep
    # =========================================================
    t = scheduler.timesteps[min(frame_counter, len(scheduler.timesteps) - 1)]

    # =========================================================
    # 4️⃣ Noise
    # =========================================================
    latent_noisy = add_noise(latent_fp32, scheduler, t)

    latent_input = scheduler.scale_model_input(latent_noisy, t)

    # =========================================================
    # 5️⃣ CFG
    # =========================================================
    latent_input, embeds, use_cfg = apply_cfg(
        latent_input, pos_embeds, neg_embeds, current_guidance_scale
    )

    latent_input = latent_input.to(target_dtype)
    embeds = embeds.to(target_dtype)
    pose_fp32 = pose_fp32.to(target_dtype)

    # =========================================================
    # 6️⃣ UNet + ControlNet
    # =========================================================
    noise_pred = compute_noise_pred(
        unet, controlnet, latent_input, t, embeds, pose_fp32
    )

    noise_pred = merge_cfg(noise_pred, current_guidance_scale, use_cfg)

    # =========================================================
    # 7️⃣ Scheduler step
    # =========================================================
    latents_out = scheduler.step(noise_pred, t, latent_noisy).prev_sample

    # =========================================================
    # 🔥 8️⃣ Adaptive blending (clé)
    # =========================================================
    importance = compute_adaptive_importance(latent_fp32)

    delta = compute_delta(
        latents_out,
        latent_fp32,
        controlnet_scale,
        importance
    )

    latents_final = latent_fp32 + delta

    return latents_final.to(target_dtype)

#---------------------------------------------------------------------------------------------------------------------------------------------



def log_frame_error(img_path, error: Exception, verbose: bool = True):
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


#------------- JSON TO POSE SEQUENCE --------------------

def convert_json_to_pose_sequence(anim_data, H=512, W=512,
                                  device="cuda", dtype=torch.float16,
                                  debug=False, output_dir=None):
    """
    Convertit un JSON d'animation OpenPose en tensor ControlNet, avec centrage et scaling automatique.
    Output : [num_frames, 3, H, W], dtype et device configurables.
    """
    frames = anim_data.get("animation", [])
    pose_images = []

    # --- Détecter le bounding box global des keypoints ---
    all_x = []
    all_y = []
    for frame in frames:
        for kp in frame.get("keypoints", []):
            all_x.append(kp["x"])
            all_y.append(kp["y"])

    if len(all_x) == 0 or len(all_y) == 0:
        raise ValueError("Aucun keypoint trouvé dans le JSON.")

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Scale et translation pour centrer et remplir le canvas
    scale_x = (W - 20) / (max_x - min_x + 1e-6)  # marge 10px
    scale_y = (H - 20) / (max_y - min_y + 1e-6)
    scale = min(scale_x, scale_y)

    offset_x = (W - (max_x - min_x) * scale) / 2 - min_x * scale
    offset_y = (H - (max_y - min_y) * scale) / 2 - min_y * scale

    for idx, frame in enumerate(frames):
        keypoints = frame.get("keypoints", [])
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # --- Dessin des points ---
        for kp in keypoints:
            x = int(kp["x"] * scale + offset_x)
            y = int(kp["y"] * scale + offset_y)
            conf = kp.get("confidence", 1.0)
            if conf > 0.3:
                cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1)

        # --- Dessin des connexions ---
        skeleton = [
            (0, 1),  # tête → torse
            (1, 2),  # torse → bras gauche
            (1, 3),  # torse → bras droit
            (1, 4),  # torse → jambe gauche
            (1, 5),  # torse → jambe droite
        ]
        for a, b in skeleton:
            if a < len(keypoints) and b < len(keypoints):
                x1 = int(keypoints[a]["x"] * scale + offset_x)
                y1 = int(keypoints[a]["y"] * scale + offset_y)
                x2 = int(keypoints[b]["x"] * scale + offset_x)
                y2 = int(keypoints[b]["y"] * scale + offset_y)
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        img = torch.from_numpy(canvas).float() / 255.0
        img = img.permute(2, 0, 1)  # C,H,W
        pose_images.append(img)

        # Debug
        if debug and output_dir is not None:
            cv2.imwrite(f"{output_dir}/debug_pose_{idx:03d}.png", canvas)

    pose_sequence = torch.stack(pose_images).to(device=device, dtype=dtype)
    pose_sequence = pose_sequence * 2.0 - 1.0  # [-1,1]

    if debug:
        print(f"[JSON->POSE] shape: {pose_sequence.shape}")
        print(f"[JSON->POSE] min/max: {pose_sequence.min().item()} / {pose_sequence.max().item()}")

    return pose_sequence

def convert_json_to_pose_sequence_debug(anim_data, H=512, W=512, original_w=512, original_h=512,
                                  device="cuda", dtype=torch.float16, debug=False, output_dir=None):
    """
    Convertit un JSON d'animation OpenPose simplifié en tensor utilisable par ControlNet.

    Args:
        anim_data: dict JSON avec "animation" -> frames -> keypoints
        H, W: résolution finale du canvas
        original_w, original_h: résolution originale des keypoints
        device: "cuda" ou "cpu"
        dtype: torch dtype (ex: torch.float16)
        debug: bool, sauvegarde les images pour visualisation
        output_dir: chemin pour debug images (optionnel)

    Returns:
        pose_sequence: tensor [num_frames, 3, H, W] (RGB type)
    """
    frames = anim_data.get("animation", [])
    pose_images = []

    for idx, frame in enumerate(frames):
        keypoints = frame.get("keypoints", [])

        # Image noire
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # --- Dessin des points ---
        for kp in keypoints:
            # remapping keypoints vers la résolution finale
            x = int(kp["x"] * W / original_w)
            y = int(kp["y"] * H / original_h)
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
                x1 = int(keypoints[a]["x"] * W / original_w)
                y1 = int(keypoints[a]["y"] * H / original_h)
                x2 = int(keypoints[b]["x"] * W / original_w)
                y2 = int(keypoints[b]["y"] * H / original_h)
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # --- Conversion en tensor ---
        img = torch.from_numpy(canvas).float() / 255.0  # [H, W, C]
        img = img.permute(2, 0, 1)  # → [C, H, W]

        pose_images.append(img)

        # --- Debug save ---
        if debug and output_dir is not None:
            debug_path = f"{output_dir}/debug_pose_{idx:03d}.png"
            cv2.imwrite(debug_path, (canvas).astype(np.uint8))

    # --- Stack frames + normalisation [-1,1] ---
    pose_sequence = torch.stack(pose_images).to(device=device, dtype=dtype)
    pose_sequence = pose_sequence * 2.0 - 1.0  # [0,1] → [-1,1]

    if debug:
        print(f"[JSON->POSE] shape: {pose_sequence.shape}")
        print(f"[JSON->POSE] min/max: {pose_sequence.min().item()} / {pose_sequence.max().item()}")

    return pose_sequence



def build_control_latent_debug(input_pil, vae, device="cuda", latent_scale=0.18215):

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


def pad_to_multiple(x, mult=8):
    B, C, H, W = x.shape
    pad_H = (mult - H % mult) % mult
    pad_W = (mult - W % mult) % mult
    if pad_H == 0 and pad_W == 0:
        return x
    return F.pad(x, (0, pad_W, 0, pad_H))  # pad right & bottom

def gaussian_blend_mask(H, W, overlap):
    """Crée un masque gaussien pour fusionner les tiles avec overlap."""

    y = np.linspace(-1,1,H)
    x = np.linspace(-1,1,W)
    xv, yv = np.meshgrid(x,y)
    mask = np.exp(-(xv**2 + yv**2) / 0.5)  # ajuste le sigma si nécessaire
    mask = torch.tensor(mask, dtype=torch.float32)
    return mask


#---------------------------------------------------------

# 🔹 Récupère les coordonnées (x,y) d’un keypoint spécifique dans le batch
# 🔹 Récupère les coordonnées (x,y) d’un keypoint spécifique dans le batch
def get_point(kp_tensor, idx):
    return kp_tensor[:, idx, :2]  # [B,2]

# 🔹 Recentre tous les keypoints par rapport au torse (entre épaules)
#   Cela évite que le personnage se déplace vers le coin haut-gauche.
def normalize_keypoints(kp_tensor):
    kp = kp_tensor.clone()
    r_shoulder = get_point(kp, 2)
    l_shoulder = get_point(kp, 5)
    torso_center = (r_shoulder + l_shoulder) * 0.5
    kp[...,0] = kp[...,0] - torso_center[:,0].unsqueeze(1)  # recentre X
    kp[...,1] = kp[...,1] - torso_center[:,1].unsqueeze(1)  # recentre Y
    return kp

# 🔹 Calcule le déplacement du torse par rapport à la frame précédente
#   Utilisé pour translater les latents afin de suivre le mouvement.

def compute_delta_torso(kp, latent_h, latent_w, scale=0.8):
    """
    Calcule le déplacement du torse en coordonnées latentes.
    Le centre du warp est aligné sur le torse du personnage.
    """

    # Extraire les épaules
    r_shoulder = get_point(kp, 2)  # [B,2]
    l_shoulder = get_point(kp, 5)

    # Centre du torse
    torso_center = (r_shoulder + l_shoulder) * 0.5  # [B,2]

    # Normaliser par rapport à l'image (0-1)
    # On suppose que kp est déjà normalisé sur H,W [0,1]
    torso_center_norm = torso_center.clone()

    # Calculer offset depuis le centre du latent
    center_offset_x = (torso_center_norm[:,0] - 0.5) * latent_w
    center_offset_y = (torso_center_norm[:,1] - 0.5) * latent_h

    delta_torso = torch.stack([center_offset_x, center_offset_y], dim=1) * scale

    # 🔒 Stabilisation pour éviter les jumps
    delta_torso = torch.tanh(delta_torso * 2.0) * 0.5

    return delta_torso

# 🔹 Applique une translation sur les latents en utilisant un grid warp
#   Déplace visuellement le personnage selon le delta du torse.
def warp_latents(latents, delta_torso, H, W, device):

    B = latents.shape[0]

    dx = delta_torso[:, 0].reshape(B,1,1) * W
    dy = delta_torso[:, 1].reshape(B,1,1) * H

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B,1,1,1)

    delta_grid = torch.cat([dx*2/W, dy*2/H], dim=-1).unsqueeze(2)
    grid = grid + delta_grid

    latents_warped = F.grid_sample(
        latents,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    return latents_warped, dx, dy, grid


def warp_latents_local(latents, delta, mask, center, H, W, device):

    B, C, _, _ = latents.shape

    # -------------------- Préparation --------------------

    # centre en pixels
    center_px = center * torch.tensor([W-1, H-1], device=device)
    center_px = center_px.view(B,1,1,2)

    # delta en pixels
    delta_px = delta * torch.tensor([W, H], device=device)
    delta_px = delta_px.view(B,1,1,2)

    # grille pixel
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid = torch.stack((xx, yy), dim=-1).float()
    grid = grid.unsqueeze(0).repeat(B,1,1,1)

    # masque
    mask_expand = mask.permute(0,2,3,1) ** 1.5

    # -------------------- 💥 warp pivot --------------------

    grid = grid - center_px
    grid = grid + delta_px * mask_expand
    grid = grid + center_px

    # -------------------- normalisation --------------------

    grid_norm = grid.clone()
    grid_norm[...,0] = 2.0 * grid[...,0] / (W-1) - 1.0
    grid_norm[...,1] = 2.0 * grid[...,1] / (H-1) - 1.0

    # -------------------- sampling --------------------

    latents_warped = F.grid_sample(
        latents,
        grid_norm,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )

    return latents_warped



# 🔹 Applique la différence entre latents avant/après OpenPose
#   Permet de conserver l’impact du pose controlnet.
def apply_openpose_delta(latents, latents_before, latents_after, mask):
    if latents_before is not None and latents_after is not None:
        delta = latents_after - latents_before
        delta = torch.clamp(delta, -0.15, 0.15)
        latents = latents + delta * mask * 0.5
    return latents

# 🔹 Stabilise les latents pour éviter NaN ou valeurs extrêmes
#   Normalisation et clamp pour rester dans [-1.2,1.2].
def stabilize_latents_motion(latents):
    latents = torch.nan_to_num(latents)
    latents_max = latents.abs().amax(dim=(2,3), keepdim=True)
    latents = latents / (latents_max + 1e-6)
    latents = latents * 0.95
    return torch.clamp(latents, -1.2, 1.2)


# -------------------- Fonction utilitaire --------------------
def compute_torso_angle(keypoints):
    """
    Calcule l'angle du torse selon les épaules (radians).
    """
    right_shoulder = get_point(keypoints, 2)
    left_shoulder = get_point(keypoints, 5)
    vec = right_shoulder - left_shoulder
    angle = torch.atan2(vec[:,1], vec[:,0])  # [B]
    torso_center = (right_shoulder + left_shoulder) * 0.5
    return angle, torso_center


# -------------------- Fonction principale -----------------------------------------------------------------
def apply_hair_motion_extreme(
    latents,
    mask_hair,
    grid,
    H,
    W,
    frame_counter,
    device,
    delta_px=None,
    prev_hair_field=None,
    debug=False
):
    """
    Hair motion version cinéma EXTRÊME :
    - mouvements très amples
    - vent + gravité + torse amplifiés
    - inertie légère pour réactivité maximale
    - pointes accentuées
    """

    B = latents.shape[0]
    t = frame_counter
    t_wind1 = torch.tensor(t / 10.0, device=device)
    t_wind2 = torch.tensor(t / 40.0, device=device)

    # -------------------- Multi-échelle bruit --------------------
    def multi_noise(grid, t, scales=[0.05,0.15,0.3], weights=[1.0,0.5,0.25]):
        val = 0
        for s, w in zip(scales, weights):
            val += w * smooth_noise(grid, t, scale=s)
        return val

    noise_x = multi_noise(grid, t)
    noise_y = multi_noise(grid, t + 123, scales=[0.08,0.2,0.4], weights=[1.0,0.5,0.25])

    # -------------------- Champ delta de base extrême --------------------
    hair_delta_field = torch.zeros_like(grid)
    hair_delta_field[...,0] = 0.12 * noise_x   # x4 vs version cinéma
    hair_delta_field[...,1] = 0.18 * noise_y   # x4 vs version cinéma

    # -------------------- Vent extrême --------------------
    wind_dir = torch.tensor([[1.0,0.3],[0.5,0.2]], device=device).mean(dim=0).view(1,1,1,2)
    wind_strength = 0.08 + 0.04 * torch.sin(t_wind1) + 0.02 * torch.sin(t_wind2)
    wind_delta = wind_dir * wind_strength

    # -------------------- Gravité plus marquée --------------------
    gravity_delta = torch.zeros_like(grid)
    gravity_delta[...,1] = 0.015  # fort tombant

    # -------------------- Influence du torse amplifiée --------------------
    if delta_px is not None:
        speed = torch.norm(delta_px, dim=-1, keepdim=True)
        hair_delta_field *= (1.0 + 5.0 * speed)
        wind_delta *= (1.0 + 3.0 * speed)
        gravity_delta *= (1.0 + 1.5 * speed)

    # -------------------- Inertie légère --------------------
    inertia = 0.5  # moins d'amortissement → mouvements plus réactifs
    if prev_hair_field is not None:
        hair_delta_field = inertia * prev_hair_field + (1-inertia) * hair_delta_field

    # -------------------- Masque + falloff racine→pointe --------------------
    mask_hair_expand = mask_hair.permute(0,2,3,1)
    yy = torch.linspace(0,1,H,device=device).view(1,H,1,1)
    extreme_falloff = yy**3 * (3 - 2*yy**1.5)  # pointes très dynamiques
    mask_hair_expand = mask_hair_expand * extreme_falloff

    # -------------------- Micro-souplesse physique --------------------
    spring = 0.01 * torch.sin(t*0.8 + grid[...,1:2]*5.0)
    hair_delta_field[...,1:2] += spring

    # -------------------- Micro noise --------------------
    micro_noise = 0.003 * (torch.rand_like(hair_delta_field)-0.5)
    hair_delta_field += micro_noise

    # -------------------- Application --------------------
    grid_hair = grid + hair_delta_field * mask_hair_expand
    grid_hair += wind_delta * mask_hair_expand
    grid_hair += gravity_delta * mask_hair_expand

    # -------------------- Normalisation --------------------
    grid_hair[...,0] = 2.0 * grid_hair[...,0] / (W-1) - 1.0
    grid_hair[...,1] = 2.0 * grid_hair[...,1] / (H-1) - 1.0

    # -------------------- Sampling --------------------
    latents_out = F.grid_sample(latents, grid_hair, align_corners=True)

    if debug:
        print("[DEBUG] Hair motion EXTREME applied")

    return latents_out, hair_delta_field

def apply_hair_motion(
    latents,
    mask_hair,
    grid,
    H,
    W,
    frame_counter,
    device,
    delta_px=None,
    prev_hair_field=None,
    debug=False
):
    """
    Hair motion version cinéma amplifiée :
    - mouvements plus marqués
    - vent + gravité + micro-souplesse
    - inertie adaptative pour fluidité
    """

    B = latents.shape[0]
    t = frame_counter
    t_wind1 = torch.tensor(t / 15.0, device=device)
    t_wind2 = torch.tensor(t / 60.0, device=device)

    # -------------------- Multi-échelle bruit --------------------
    def multi_noise(grid, t, scales=[0.05,0.15,0.3], weights=[1.0,0.5,0.25]):
        val = 0
        for s, w in zip(scales, weights):
            val += w * smooth_noise(grid, t, scale=s)
        return val

    noise_x = multi_noise(grid, t)
    noise_y = multi_noise(grid, t + 123, scales=[0.08,0.2,0.4], weights=[1.0,0.5,0.25])

    # -------------------- Champ delta de base amplifié --------------------
    hair_delta_field = torch.zeros_like(grid)
    hair_delta_field[...,0] = 0.06 * noise_x   # x2 vs original
    hair_delta_field[...,1] = 0.10 * noise_y   # x2 vs original

    # -------------------- Vent dynamique --------------------
    wind_dir = torch.tensor([[1.0,0.2],[0.3,0.1]], device=device).mean(dim=0).view(1,1,1,2)
    wind_strength = 0.04 + 0.02 * torch.sin(t_wind1) + 0.01 * torch.sin(t_wind2)
    wind_delta = wind_dir * wind_strength

    # -------------------- Gravité légère --------------------
    gravity_delta = torch.zeros_like(grid)
    gravity_delta[...,1] = 0.008  # plus tombant

    # -------------------- Influence du torse --------------------
    if delta_px is not None:
        speed = torch.norm(delta_px, dim=-1, keepdim=True)
        hair_delta_field *= (1.0 + 3.5 * speed)
        wind_delta *= (1.0 + 2.0 * speed)
        gravity_delta *= (1.0 + 0.8 * speed)

    # -------------------- Inertie adaptative --------------------
    inertia = 0.7  # moins amorti, plus cinématique
    if prev_hair_field is not None:
        hair_delta_field = inertia * prev_hair_field + (1 - inertia) * hair_delta_field

    # -------------------- Masque + falloff racine→pointe --------------------
    mask_hair_expand = mask_hair.permute(0,2,3,1)
    yy = torch.linspace(0,1,H,device=device).view(1,H,1,1)
    smooth_falloff = yy**2.5 * (3 - 2*yy**1.5)  # accentue le mouvement sur les pointes
    mask_hair_expand = mask_hair_expand * smooth_falloff

    # -------------------- Micro-souplesse physique --------------------
    spring = 0.006 * torch.sin(t*0.5 + grid[...,1:2]*3.0)
    hair_delta_field[...,1:2] += spring

    # -------------------- Micro noise --------------------
    micro_noise = 0.002 * (torch.rand_like(hair_delta_field)-0.5)
    hair_delta_field += micro_noise

    # -------------------- Application --------------------
    grid_hair = grid + hair_delta_field * mask_hair_expand
    grid_hair += wind_delta * mask_hair_expand
    grid_hair += gravity_delta * mask_hair_expand

    # -------------------- Normalisation pour grid_sample --------------------
    grid_hair[...,0] = 2.0 * grid_hair[...,0] / (W-1) - 1.0
    grid_hair[...,1] = 2.0 * grid_hair[...,1] / (H-1) - 1.0

    # -------------------- Sampling --------------------
    latents_out = F.grid_sample(latents, grid_hair, align_corners=True)

    if debug:
        print("[DEBUG] Hair motion cinema amplified applied")

    return latents_out, hair_delta_field

def apply_hair_motion_cinema(  # version cinéma
    latents,
    mask_hair,
    grid,
    H,
    W,
    frame_counter,
    device,
    delta_px=None,
    prev_hair_field=None,
    debug=False
):
    B = latents.shape[0]

    # -------------------- Temps --------------------
    t = frame_counter
    t_wind1 = torch.tensor(t / 15.0, device=device)
    t_wind2 = torch.tensor(t / 60.0, device=device)

    # -------------------- Multi-échelle bruit --------------------
    def multi_noise(grid, t, scales=[0.05,0.15,0.3], weights=[1.0,0.5,0.25]):
        val = 0
        for s,w in zip(scales, weights):
            val += w * smooth_noise(grid, t, scale=s)
        return val

    noise_x = multi_noise(grid, t)
    noise_y = multi_noise(grid, t + 123, scales=[0.08,0.2,0.4], weights=[1.0,0.5,0.25])

    # -------------------- Champ delta de base --------------------
    hair_delta_field = torch.zeros_like(grid)
    hair_delta_field[...,0] = 0.03 * noise_x
    hair_delta_field[...,1] = 0.05 * noise_y

    # -------------------- Vent dynamique --------------------
    wind_dir = torch.tensor([[1.0,0.2],[0.3,0.1]], device=device).mean(dim=0).view(1,1,1,2)
    wind_strength = 0.02 + 0.01 * torch.sin(t_wind1) + 0.005 * torch.sin(t_wind2)
    wind_delta = wind_dir * wind_strength

    # -------------------- Gravité légère --------------------
    gravity_delta = torch.zeros_like(grid)
    gravity_delta[...,1] = 0.004  # constant downwards

    # -------------------- Influence du torse --------------------
    if delta_px is not None:
        speed = torch.norm(delta_px, dim=-1, keepdim=True)
        hair_delta_field *= (1.0 + 2.5*speed)
        wind_delta *= (1.0 + 1.5*speed)
        gravity_delta *= (1.0 + 0.5*speed)

    # -------------------- Inertie adaptative --------------------
    inertia = 0.85
    if prev_hair_field is not None:
        hair_delta_field = inertia * prev_hair_field + (1-inertia) * hair_delta_field

    # -------------------- Masque + falloff racine→pointe --------------------
    mask_hair_expand = mask_hair.permute(0,2,3,1)
    yy = torch.linspace(0,1,H,device=device).view(1,H,1,1)
    smooth_falloff = yy**2 * (3-2*yy)
    mask_hair_expand = mask_hair_expand * smooth_falloff

    # -------------------- Micro-souplesse physique --------------------
    spring = 0.003 * torch.sin(t*0.5 + grid[...,1:2]*3.0)
    hair_delta_field[...,1:2] += spring

    # -------------------- Micro noise --------------------
    micro_noise = 0.001 * (torch.rand_like(hair_delta_field)-0.5)
    hair_delta_field += micro_noise

    # -------------------- Application --------------------
    grid_hair = grid + hair_delta_field * mask_hair_expand
    grid_hair += wind_delta * mask_hair_expand
    grid_hair += gravity_delta * mask_hair_expand

    # -------------------- Normalisation --------------------
    grid_hair[...,0] = 2.0 * grid_hair[...,0] / (W-1) - 1.0
    grid_hair[...,1] = 2.0 * grid_hair[...,1] / (H-1) - 1.0

    # -------------------- Sampling --------------------
    latents_out = F.grid_sample(latents, grid_hair, align_corners=True)

    if debug:
        print("[DEBUG] Hair motion cinema applied")

    return latents_out, hair_delta_field

def apply_hair_motion_v2(
    latents,
    mask_hair,
    grid,
    H,
    W,
    frame_counter,
    device,
    delta_px=None,
    prev_hair_field=None,
    debug=False
):
    B = latents.shape[0]

    # -------------------- Temps --------------------
    t_dict = {
        "noise": frame_counter,
        "wind1": torch.tensor(frame_counter / 15.0, device=device),
        "wind2": torch.tensor(frame_counter / 60.0, device=device),
    }

    # -------------------- Bruit multi-échelle --------------------
    def multi_scale_noise(grid, t, scales=[0.05, 0.15, 0.3], weights=[1.0, 0.5, 0.25]):
        result = 0
        for s, w in zip(scales, weights):
            result += w * smooth_noise(grid, t, scale=s)
        return result

    noise_x = multi_scale_noise(grid, t_dict["noise"])
    noise_y = multi_scale_noise(grid, t_dict["noise"] + 123, scales=[0.08, 0.2, 0.4], weights=[1.0,0.5,0.25])

    # -------------------- Hair delta --------------------
    hair_delta_field = torch.zeros_like(grid)
    hair_delta_field[..., 0] = 0.04 * noise_x
    hair_delta_field[..., 1] = 0.06 * noise_y

    # -------------------- Vent dynamique --------------------
    wind_dir = torch.tensor([1.0, 0.3], device=device).view(1,1,1,2)
    wind_strength = 0.03
    wind_delta = wind_dir * (wind_strength +
                             0.01 * torch.sin(t_dict["wind1"]) +
                             0.005 * torch.sin(t_dict["wind2"]))

    # -------------------- Influence du mouvement du torse --------------------
    if delta_px is not None:
        speed = torch.norm(delta_px, dim=-1, keepdim=True)
        hair_delta_field *= (1.0 + 2.5 * speed)  # plus naturel

    # -------------------- Inertie --------------------
    inertia = 0.85 if prev_hair_field is not None else 0.0
    if prev_hair_field is not None:
        hair_delta_field = inertia * prev_hair_field + (1 - inertia) * hair_delta_field

    # -------------------- Masque + Falloff racine→pointe --------------------
    mask_hair_expand = mask_hair.permute(0,2,3,1)

    yy = torch.linspace(0, 1, H, device=device).view(1,H,1,1)
    # Smoothstep pour transition plus douce
    falloff_root = yy**2 * (3 - 2*yy)  # smoothstep approximation
    mask_hair_expand = mask_hair_expand * falloff_root

    # -------------------- Micro noise supplémentaire --------------------
    micro_noise = 0.002 * (torch.rand_like(hair_delta_field) - 0.5)
    hair_delta_field += micro_noise

    # -------------------- Application --------------------
    grid_hair = grid + hair_delta_field * mask_hair_expand + wind_delta * mask_hair_expand

    # -------------------- Normalisation --------------------
    grid_hair[...,0] = 2.0 * grid_hair[...,0] / (W-1) - 1.0
    grid_hair[...,1] = 2.0 * grid_hair[...,1] / (H-1) - 1.0

    # -------------------- Sampling --------------------
    latents_out = F.grid_sample(latents, grid_hair, align_corners=True)

    if debug:
        print("[DEBUG] Hair motion applied with improved quality")

    return latents_out, hair_delta_field


def apply_torso_warp(
    latents,
    pose,
    mask_torso,
    grid,
    H,
    W,
    device,
    prev_delta_px=None,
    debug=False
):
    B = latents.shape[0]

    # -------------------- Centre du torse --------------------
    points_idx = [2, 5, 8, 11]
    pts = torch.stack([pose.get_point(i) for i in points_idx], dim=1)

    torso_center = pts.mean(dim=1)
    torso_center_px = torso_center * torch.tensor([W-1, H-1], device=device)
    torso_center_px = torso_center_px.view(B,1,1,2)

    # -------------------- Delta torse --------------------
    delta_px = pose.delta.clone()
    delta_px[...,0] *= W
    delta_px[...,1] *= H
    delta_px = delta_px.view(B,1,1,2)

    # -------------------- Lissage temporel --------------------
    if prev_delta_px is not None:
        alpha = 0.7
        delta_px = alpha * delta_px + (1 - alpha) * prev_delta_px

    # -------------------- Feather dynamique --------------------
    mask_torso = feather_dynamic_vectorized(
        mask_torso,
        delta_px,
        base_radius=3,
        sigma=1.5,
        scale=2.0
    )

    mask_expand = mask_torso.permute(0,2,3,1)

    # -------------------- Déformation non-linéaire (IMPORTANT) --------------------
    offset = grid - torso_center_px
    distance = torch.norm(offset, dim=-1, keepdim=True)

    # falloff spatial → centre bouge plus que les bords
    falloff = torch.exp(-distance / (0.35 * W))

    # -------------------- Warp torse --------------------
    grid_torso = grid + delta_px * mask_expand * falloff

    # -------------------- Normalisation --------------------
    grid_torso[...,0] = 2.0 * grid_torso[...,0] / (W-1) - 1.0
    grid_torso[...,1] = 2.0 * grid_torso[...,1] / (H-1) - 1.0

    # -------------------- Sampling --------------------
    latents_out = F.grid_sample(latents, grid_torso, align_corners=True)

    if debug:
        print("[DEBUG] Torso warp applied")

    return latents_out, delta_px

def apply_global_pose(
    latents,
    pose,
    prev_pose=None,
    H=None,
    W=None,
    device="cuda",
    strength=0.1,   # 🔥 beaucoup plus faible !
    debug=False
):
    B = latents.shape[0]

    # -------------------- Si pas de frame précédente → rien faire --------------------
    if prev_pose is None:
        return latents, torch.zeros((B,1,1,2), device=device)

    # -------------------- Centre actuel --------------------
    points_idx = [2, 5, 8, 11]
    pts = torch.stack([pose.get_point(i) for i in points_idx], dim=1)
    center = pts.mean(dim=1)

    # -------------------- Centre précédent --------------------
    prev_pts = torch.stack([prev_pose.get_point(i) for i in points_idx], dim=1)
    prev_center = prev_pts.mean(dim=1)

    # -------------------- Delta réel --------------------
    delta = center - prev_center  # 🔥 vrai mouvement

    # passage en pixels
    delta_px = delta.clone()
    delta_px[...,0] *= W
    delta_px[...,1] *= H
    delta_px = delta_px.view(B,1,1,2)

    # -------------------- Clamp pour stabilité --------------------
    delta_px = torch.clamp(delta_px, min=-10.0, max=10.0)

    # -------------------- Grille --------------------
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B,1,1,1)

    grid_global = grid + delta_px * strength

    # -------------------- Normalisation --------------------
    grid_global[...,0] = 2.0 * grid_global[...,0] / (W-1) - 1.0
    grid_global[...,1] = 2.0 * grid_global[...,1] / (H-1) - 1.0

    # -------------------- Warp --------------------
    latents_out = F.grid_sample(latents, grid_global, align_corners=True)

    if debug:
        print(f"[DEBUG] Global delta px: {delta_px.mean().item():.4f}")

    return latents_out, delta_px

def apply_face_warp_v2(
    latents,
    pose,
    mask_face,
    grid,
    H,
    W,
    frame_counter,
    device=None,
    debug=False,
    debug_dir=None,
    smooth=0.8
):
    """
    Warp du visage avec micro-expressions et respiration.
    Version stateful avec gestion interne des points faciaux.
    """

    if device is None:
        device = latents.device

    B, C, H, W = latents.shape
    latents_in = latents.clone()

    # =========================
    # 🔥 Récupération + update auto des points
    # =========================
    prev_facial_points = pose.get_prev_facial_points()

    # Estimation avec lissage temporel (la méthode n'a PAS prev_points)
    facial_points = pose.estimate_facial_points_full(smooth=smooth)

    # Sauvegarde pour la frame suivante
    pose.set_prev_facial_points(facial_points)

    mouth_center = facial_points['mouth_center']
    mouth_top    = facial_points['mouth_top']
    mouth_bottom = facial_points['mouth_bottom']

    # =========================
    # 🔹 Temps (tensor safe)
    # =========================
    t_resp_y = torch.tensor(frame_counter / 8.0, device=device)
    t_resp_x = torch.tensor(frame_counter / 10.0, device=device)
    t_micro  = torch.tensor(frame_counter / 5.0, device=device)

    # =========================
    # 🔹 Base motion visage
    # =========================
    face_delta = torch.zeros((B,1,1,2), device=device)
    face_delta[...,1] += 0.01  * torch.sin(t_resp_y)
    face_delta[...,0] += 0.005 * torch.sin(t_resp_x)
    face_delta[...,0] += 0.002 * torch.sin(t_micro)
    face_delta[...,1] += 0.003 * torch.sin(t_micro * 1.5)

    # =========================
    # 🔹 Micro sourire
    # =========================
    mouth_mask = pose.get_mouth_region(H, W, device=device, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter)
    mouth_mask_exp = mouth_mask.permute(0,2,3,1)  # [B,H,W,1]

    smile_strength = 0.02
    smile_delta = smile_strength * torch.sin(torch.tensor(frame_counter / 10.0, device=device))

    face_delta_expanded = face_delta.expand(B,H,W,2).clone()
    face_delta_expanded[...,0] += mouth_mask_exp[...,0] * smile_delta

    # =========================
    # 🔹 Respiration bouche
    # =========================
    breath_strength = 0.005
    breath_delta = breath_strength * torch.sin(torch.tensor(frame_counter / 12.0, device=device))
    face_delta_expanded[...,1] += mouth_mask_exp[...,0] * breath_delta

    # =========================
    # 🔹 Centre visage (nez)
    # =========================
    face_center = pose.get_point(0)
    face_center_px = face_center * torch.tensor([W-1, H-1], device=device)
    face_center_px = face_center_px.view(B,1,1,2)

    # =========================
    # 🔹 Grid warp
    # =========================
    grid_face = grid.clone() - face_center_px
    mask_face_exp = mask_face.permute(0,2,3,1)
    grid_face = grid_face + face_delta_expanded * mask_face_exp
    grid_face = grid_face + face_center_px

    # =========================
    # 🔹 Normalisation
    # =========================
    grid_face[...,0] = 2.0 * grid_face[...,0] / (W-1) - 1.0
    grid_face[...,1] = 2.0 * grid_face[...,1] / (H-1) - 1.0

    # =========================
    # 🔹 Warp final
    # =========================
    latents_out = F.grid_sample(latents, grid_face, align_corners=True)

    # =========================
    # 🔹 Debug
    # =========================
    if debug:
        save_impact_map(latents_out, latents_in, debug_dir, frame_counter)
        print("[DEBUG] Face warp v2 (stateful) applied")

    return latents_out, face_delta_expanded, facial_points

def apply_face_warp(
    latents,
    pose,
    mask_face,
    grid,
    H,
    W,
    frame_counter,
    device,
    debug=False,
    debug_dir=None
):
    """
    Warp du visage avec micro expressions et respiration.
    - latents : tenseur [B,C,H,W]
    - pose : objet Pose
    - mask_face : masque visage [B,1,H,W]
    - grid : grille de base [B,H,W,2]
    - frame_counter : compteur de frame pour animations
    """

    B, C, H, W = latents.shape
    latents_in = latents.clone()

    # -------------------- Delta visage --------------------
    face_center = pose.get_point(0)  # nez
    face_center_px = face_center * torch.tensor([W-1, H-1], device=device)
    face_center_px = face_center_px.view(B,1,1,2)

    # Base delta pour respiration/micro-mouvements
    t_dict = {
        "resp_y": torch.tensor(frame_counter / 8.0, device=device),
        "resp_x": torch.tensor(frame_counter / 10.0, device=device),
        "micro": torch.tensor(frame_counter / 5.0, device=device),
        "wind1": torch.tensor(frame_counter / 15.0, device=device),
        "wind2": torch.tensor(frame_counter / 60.0, device=device),
    }

    face_delta = torch.zeros((B,1,1,2), device=device)
    face_delta[...,1] += 0.01 * torch.sin(t_dict["resp_y"])
    face_delta[...,0] += 0.005 * torch.sin(t_dict["resp_x"])
    face_delta[...,0] += 0.002 * torch.sin(t_dict["micro"])
    face_delta[...,1] += 0.003 * torch.sin(t_dict["micro"] * 1.5)

    # -------------------- Micro sourire --------------------
    mouth_mask = pose.get_mouth_region(H, W, device=device, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter)
    mouth_mask_exp = mouth_mask.permute(0,2,3,1)  # [B,H,W,1] pour broadcasting

    smile_strength = 0.02
    smile_delta = smile_strength * torch.sin(torch.tensor(frame_counter / 10.0, device=device))

    # face_delta étendu à [B,H,W,2] pour grid_sample
    face_delta_expanded = face_delta.expand(B,H,W,2).clone()
    face_delta_expanded[...,0] += mouth_mask_exp[...,0] * smile_delta

    # -------------------- Construction grille --------------------
    grid_face = grid.clone()
    grid_face = grid_face - face_center_px
    mask_face_exp = mask_face.permute(0,2,3,1)  # [B,H,W,1]
    grid_face = grid_face + face_delta_expanded * mask_face_exp
    grid_face = grid_face + face_center_px

    # Normalisation [-1,1] pour grid_sample
    grid_face[...,0] = 2.0*grid_face[...,0]/(W-1) - 1.0
    grid_face[...,1] = 2.0*grid_face[...,1]/(H-1) - 1.0

    # -------------------- Warp --------------------
    latents_out = F.grid_sample(latents, grid_face, align_corners=True)

    if debug:
        save_impact_map(latents_out, latents_in, debug_dir, frame_counter)
        print("[DEBUG] Face warp applied")

    return latents_out, face_delta_expanded

#------------------------------------------------------------------------------------------

def apply_pose_driven_motion(
    latents,
    previous_latent,
    latents_before_openpose,
    latents_after_openpose,
    keypoints,
    prev_keypoints=None,
    frame_counter=0,
    device="cuda",
    breathing=True,
    debug=False,
    debug_dir=None
):
    """
    Pipeline motion PRO (stable + vivant + isolé):
    - Global Pose
    - Torso Warp
    - Face Warp (stateful)
    - Hair Motion (alt normal/extreme)
    - Breathing (torso only)
    - Stabilisation (face protected)
    """

    # =========================
    # 🔹 Setup
    # =========================
    B, C, H, W = latents.shape
    device = latents.device

    latents = latents.float()
    latents_in = latents.clone()

    # =========================
    # 🔹 Pose
    # =========================
    pose = Pose(keypoints.to(device))
    pose.compute_torso_delta(latent_h=H, latent_w=W)

    prev_pose = Pose(prev_keypoints.to(device)) if prev_keypoints is not None else None

    # =========================
    # 🔹 Grid
    # =========================
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    grid = torch.stack((xx, yy), dim=-1).float()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    # =========================
    # 🔹 Masks (CLAMP SAFE)
    # =========================
    mask_face  = torch.clamp(pose.create_face_mask(H, W), 0, 1).float()
    mask_torso = torch.clamp(pose.create_upper_body_mask(H, W), 0, 1).float()
    mask_hair  = torch.clamp(pose.create_hair_mask(H, W), 0, 1).float()

    # PRIORITÉ VISAGE ABSOLUE
    mask_face_exp  = mask_face
    mask_torso_exp = mask_torso * (1.0 - mask_face_exp)
    mask_hair_exp  = mask_hair  * (1.0 - mask_face_exp)

    # =========================
    # 🔹 GLOBAL POSE
    # =========================
    latents_before = latents.clone()

    latents_global, global_delta = apply_global_pose(
        latents=latents,
        pose=pose,
        prev_pose=prev_pose,
        H=H,
        W=W,
        device=device
    )

    # 🔥 protection visage stricte
    latents = latents_global * (1.0 - mask_face_exp) + latents_before * mask_face_exp

    if debug:
        print("[DEBUG] Global pose applied")

    # =========================
    # 🔹 TORSO
    # =========================
    latents_before = latents.clone()

    latents_torso, delta_px = apply_torso_warp(
        latents=latents,
        pose=pose,
        mask_torso=mask_torso,
        grid=grid,
        H=H,
        W=W,
        device=device
    )

    latents = latents_torso * mask_torso_exp + latents_before * (1.0 - mask_torso_exp)

    if debug:
        print("[DEBUG] Torso warp applied")

    # =========================
    # 🔹 FACE (STATEFUL)
    # =========================
    latents, face_delta, facial_points = apply_face_warp_v2(
        latents=latents,
        pose=pose,
        mask_face=mask_face,
        grid=grid,
        H=H,
        W=W,
        frame_counter=frame_counter,
        device=device,
        debug=debug,
        debug_dir=debug_dir,
        smooth=0.85  # 🔥 un peu plus smooth = vivant mais stable
    )

    if debug:
        print("[DEBUG] Face warp applied")

    # =========================
    # 🔹 HAIR (ALTERNANCE CINÉMA)
    # =========================
    latents_before = latents.clone()

    if (frame_counter // 6) % 2 == 0:
        # 🔹 phase douce
        latents_hair, hair_delta = apply_hair_motion(
            latents, mask_hair, grid, H, W,
            frame_counter, device,
            delta_px=delta_px,
            debug=debug
        )
    else:
        # 🔹 phase dynamique
        latents_hair, hair_delta = apply_hair_motion_extreme(
            latents, mask_hair, grid, H, W,
            frame_counter, device,
            delta_px=delta_px,
            debug=debug
        )

    latents = latents_hair * mask_hair_exp + latents_before * (1.0 - mask_hair_exp)

    if debug:
        print("[DEBUG] Hair motion applied")

    # =========================
    # 🔹 BREATHING (TORSO ONLY)
    # =========================
    latents_before = latents.clone()

    latents_breath = apply_breathing(
        latents,
        previous_latent,
        frame_counter,
        breathing
    )

    latents = latents_breath * mask_torso_exp + latents_before * (1.0 - mask_torso_exp)

    if debug:
        print("[DEBUG] Breathing applied")

    # =========================
    # 🔹 STABILISATION
    # =========================
    latents_before = latents.clone()

    latents_stab = stabilize_latents_motion(latents)

    # 🔥 visage jamais touché
    latents = latents_stab * (1.0 - mask_face_exp) + latents_before * mask_face_exp

    if debug:
        print("[DEBUG] Stabilization applied")

    # =========================
    # 🔹 MICRO BOOST GLOBAL (🔥 IMPORTANT)
    # =========================
    # 👉 sinon rendu trop statique
    micro_amp = 0.002
    t = torch.tensor(frame_counter / 6.0, device=device)

    latents = latents + micro_amp * torch.sin(t)

    # =========================
    # 🔹 DEBUG FINAL
    # =========================
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter)
        print("[DEBUG] Full motion pipeline applied")

    return latents



#------------------------------------------------------------------------------------------
def update_pose_sequence_from_keypoints_batch(
    keypoints_tensor,
    prev_keypoints=None,
    frame_idx=0,
    alpha=0.7,        # lissage temporel plus fort
    add_motion=True,
    debug=False
):

    kp = keypoints_tensor.clone()
    B, N, _ = kp.shape
    device = kp.device

    # =========================
    # 🔹 1. SMOOTH TEMPOREL
    # =========================
    if prev_keypoints is not None:
        kp = alpha * kp + (1 - alpha) * prev_keypoints

    # =========================
    # 🔹 2. MOTION PROCÉDURAL
    # =========================
    if add_motion:
        t = frame_idx

        # Respiration (vertical torso + épaules)
        breath = 0.009 * math.sin(t * 0.15)
        kp[:, 2, 1] += breath
        kp[:, 5, 1] += breath

        # Balancement gauche/droite
        sway = 0.010 * math.sin(t * 0.08)
        kp[:, :, 0] += sway

        # Head motion
        head_idx = 0
        kp[:, head_idx, 0] += 0.006 * math.sin(t * 0.2)
        kp[:, head_idx, 1] += 0.006 * math.cos(t * 0.18)

        # Drift lent
        drift_x = 0.002 * math.sin(t * 0.03)
        drift_y = 0.002 * math.cos(t * 0.025)
        kp[:, :, 0] += drift_x
        kp[:, :, 1] += drift_y

        # Micro noise (anti-freeze)
        noise = torch.randn_like(kp[..., :2]) * 0.0015
        kp[..., :2] += noise

    # =========================
    # 🔹 3. STABILISATION
    # =========================
    kp[..., :2] = torch.clamp(kp[..., :2], -1.2, 1.2)

    # =========================
    # 🔹 4. DEBUG
    # =========================
    if debug:
        motion_strength = (kp - keypoints_tensor).abs().mean()
        print(f"[DEBUG] Keypoint motion strength: {motion_strength.item():.6f}")

    return kp

#--------------------------------------------------------------------------------------------------------------

def save_debug_pose_image_with_skeleton(
    pose_tensor,
    keypoints_tensor,
    frame_counter,
    output_dir,
    cfg=None,
    prefix="openpose"
):
    """
    Sauvegarde une image de pose ET un squelette OpenPose pour contrôle visuel.

    Args:
        pose_tensor (torch.Tensor): [B,3,H,W] normalisé [-1,1] ou [C,H,W]
        keypoints_tensor (torch.Tensor): [B,18,3] (x,y,conf) normalisé [0,1]
        frame_counter (int): numéro de frame
        output_dir (str): dossier où sauvegarder
        cfg (dict, optional): peut contenir 'visual_debug' pour activer/désactiver
        prefix (str, optional): préfixe du fichier
    """

    if cfg is not None and cfg.get("visual_debug") is False:
        return

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------
    # 🔹 Convertir pose_tensor en image RGB [0,255]
    # ----------------------------
    pose_img = pose_tensor[0].detach().cpu()
    if pose_img.ndim == 3 and pose_img.shape[0] == 3:
        pose_img = pose_img.permute(1,2,0)  # H,W,C
    pose_img = ((pose_img + 1.0)/2.0 * 255).clamp(0,255).byte().numpy()

    # Sauvegarde simple de l'image de pose
    filename_pose = f"{prefix}_{frame_counter:05d}.png"
    path_pose = os.path.join(output_dir, filename_pose)
    cv2.imwrite(path_pose, cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))
    print(f"[DEBUG] Pose sauvegardée : {path_pose}")

    # ---------------------------
    # 🔹 Dessin du squelette via debug_draw_openpose_skeleton
    # ---------------------------
    if keypoints_tensor is not None:
        debug_draw_openpose_skeleton(
            pose_full_image=pose_tensor.unsqueeze(0) if pose_tensor.ndim==3 else pose_tensor,
            keypoints_tensor=keypoints_tensor,
            debug_dir=output_dir,
            frame_counter=frame_counter
        )
