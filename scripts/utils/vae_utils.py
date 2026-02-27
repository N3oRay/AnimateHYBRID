# utils/vae_utils.py
import torch
from diffusers import AutoencoderKL
from pathlib import Path
import os
from diffusers import UNet2DConditionModel, AutoencoderKL, DPMSolverMultistepScheduler
from torch.nn.functional import interpolate
from torch.nn.functional import pad
from safetensors.torch import load_file
from torchvision.transforms import ToPILImage

import torchvision.transforms as T
from PIL import Image


LATENT_SCALE = 0.18215

# scripts/utils/vae_utils.py



# -------------------------
# Decode frame via VAE (compatible vae_offload)
# -------------------------
def decode_latents_safe(latents, vae, device, tile_size=128, overlap=64):
    """
    Décodage sécurisé des latents en image PIL, compatible avec VAE sur CPU (vae_offload)
    et avec latents sur GPU.
    """
    # Déplacer latents sur le device du VAE
    vae_device = next(vae.parameters()).device
    latents = latents.to(vae_device).float()

    # Éviter NaN/Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=0.0)

    # Décodage en tiles pour VRAM limitée
    frame_tensor = decode_latents_to_image_tiled128(
        latents,
        vae,
        tile_size=tile_size,
        overlap=overlap,
        device=vae_device
    ).clamp(0, 1)

    # Renvoie un tensor CPU float32 pour sauvegarde
    return frame_tensor.cpu()


# ---------------------------------
# Debug UNet sur latents
# ---------------------------------
# Ajoutons des fonctions de test pour valider les latents avant de les passer dans UNet et VAE.

def test_unet_on_latents(latents, unet, device):
    """
    Teste l'entrée latente avant de la passer dans le UNet
    Cette fonction aide à vérifier si la forme des latents est correcte avant de les utiliser.
    """
    print(f"[Test UNet] Latents avant UNet - min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}")

    # Assurer que les latents ont la forme correcte (batch, channels, height, width)
    if latents.ndimension() != 4:
        raise ValueError(f"Les latents doivent avoir 4 dimensions, mais ils en ont {latents.ndimension()}")

    # Test rapide avec le UNet pour vérifier si la forme est correcte
    try:
        # Assurez-vous que le UNet est configuré avec les bons paramètres (timestep, encoder_hidden_states)
        # Simulez les entrées nécessaires à un UNet standard si nécessaire
        timestep = torch.tensor([0]).to(device)  # Exemple de timestep (à ajuster selon votre modèle)
        encoder_hidden_states = torch.zeros((latents.size(0), 77, latents.size(2)), device=device)  # Ajustez selon votre taille

        output = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        print(f"[Test UNet] Latents après UNet - min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        return output
    except Exception as e:
        print(f"[Test UNet] Erreur pendant l'exécution du UNet : {e}")
        return None

def test_vae_on_latents(latents, vae, device):
    """
    Teste l'entrée latente avant de la passer au VAE
    Cette fonction vérifie la forme et le contenu des latents avant le décodage avec VAE.
    """
    print(f"[Test VAE] Latents avant VAE - min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}")

    # Assurez-vous que les latents ont la forme correcte pour le VAE
    if latents.ndimension() != 4:
        raise ValueError(f"Les latents doivent avoir 4 dimensions, mais ils en ont {latents.ndimension()}")

    # Essayons de décoder avec le VAE
    try:
        decoded_image = vae.decode(latents).sample  # Ajustez selon la fonction exacte du VAE
        print(f"[Test VAE] Image décodée - min={decoded_image.min():.4f}, max={decoded_image.max():.4f}, mean={decoded_image.mean():.4f}")
        return decoded_image
    except Exception as e:
        print(f"[Test VAE] Erreur pendant le décodage avec le VAE : {e}")
        return None


# -------------------------
# Décodage tiled (pour grandes images)
# -------------------------
def encode_images_to_latents_ai(images, vae):
    """
    Encode une batch d'images [B, 3, H, W] en latents [B, 4, H/8, W/8].
    """
    device = images.device
    vae_dtype = next(vae.parameters()).dtype

    with torch.no_grad():
        # On encode en latent avec le VAE, en s'assurant que la sortie a bien 4 canaux
        latents = vae.encode(images.to(vae_dtype)).latent_dist.sample()

    # Assure que les latents sont en 4 canaux, ce qui est attendu pour le VAE
    if latents.shape[1] != 4:
        raise ValueError(f"Latents doivent avoir 4 canaux, mais ont {latents.shape[1]} canaux.")

    # Scale pour correspondre au SD
    latents = latents * LATENT_SCALE
    return latents

def decode_latents_to_image_tiled128(latents, vae, tile_size=128, overlap=64, device="cuda"):
    """
    Decode les latents [B, 4, H, W] en images [B, 3, H, W] avec tiling pour éviter OOM.
    Supporte latents 5D [B, C, T, H, W] en reshaping automatique.
    """
    vae_dtype = next(vae.parameters()).dtype

    # Support 5D : [B, C, T, H, W] -> [B*T, C, H, W]
    if latents.ndim == 5:
        B, C, T, H, W = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
    elif latents.ndim == 4:
        B, C, H, W = latents.shape
    else:
        raise ValueError(f"Latents attendus 4D ou 5D, got {latents.shape}")

    # Assure que C=4
    if C != 4:
        raise ValueError(f"Latents doivent avoir 4 canaux, got {C} canaux")

    # Convert dtype pour correspondre au VAE
    latents = latents.to(vae_dtype)

    H_out, W_out = H, W
    output = torch.zeros(B if latents.ndim == 4 else B*T, 3, H_out, W_out, device=device, dtype=torch.float32)

    # Décodage en tiling si image > tile_size
    if max(H, W) <= tile_size:
        with torch.no_grad():
            decoded = vae.decode(latents / LATENT_SCALE).sample
        return decoded.clamp(0, 1)

    # Tiling (optionnel, pour très grandes images)
    # Ici simplifié : on peut ajouter tiling si nécessaire
    with torch.no_grad():
        decoded = vae.decode(latents / LATENT_SCALE).sample
    return decoded.clamp(0, 1)
# -------------------------
# vae_utils.py (version corrigée)
# -------------------------
# -------------------------
# Encode images en latents
# -------------------------
def encode_images_to_latents_ai_old(images, vae):
    """
    Encode une batch d'images [B, 3, H, W] en latents [B, 4, H/8, W/8].
    """
    device = images.device
    vae_dtype = next(vae.parameters()).dtype

    with torch.no_grad():
        latents = vae.encode(images.to(vae_dtype)).latent_dist.sample()
    # Scale pour correspondre au SD
    latents = latents * LATENT_SCALE
    return latents

# -------------------------
# Décodage tiled des latents 5D
# -------------------------
def decode_latents_to_image_tiled128_5D(latents, vae, tile_size=128, overlap=64, device="cuda"):
    """
    Decode les latents [B, 4, H, W] en images [B, 3, H, W] avec tiling pour éviter OOM.
    Supporte latents 5D [B, C, T, H, W] en reshaping automatique.
    """
    vae_dtype = next(vae.parameters()).dtype

    # Support 5D : [B, C, T, H, W] -> [B*T, C, H, W]
    if latents.ndim == 5:
        B, C, T, H, W = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
    elif latents.ndim == 4:
        B, C, H, W = latents.shape
    else:
        raise ValueError(f"Latents attendus 4D ou 5D, got {latents.shape}")

    # Assure que C=4
    assert C == 4, f"Latents doivent avoir 4 canaux, got {C}"

    # Convert dtype pour correspondre au VAE
    latents = latents.to(vae_dtype)

    H_out, W_out = H, W
    output = torch.zeros(B if latents.ndim == 4 else B*T, 3, H_out, W_out, device=device, dtype=torch.float32)

    # Décodage en tiling si image > tile_size
    if max(H, W) <= tile_size:
        with torch.no_grad():
            decoded = vae.decode(latents / LATENT_SCALE).sample
        return decoded.clamp(0, 1)

    # Tiling (optionnel, pour très grandes images)
    # Ici simplifié : on peut ajouter tiling si nécessaire
    with torch.no_grad():
        decoded = vae.decode(latents / LATENT_SCALE).sample
    return decoded.clamp(0, 1)

# ---------------------------------------------------------------------------------------------
# Test KO - deprecated
# -------------------------------------------------------------------------------------------

def decode_latents_to_image_test(latents, vae, tile_size=128, overlap=64, device="cuda"):
    latents = latents.to(torch.float32)

    # For 3D or 4D latents
    if latents.ndim == 2:  # [H*W] ou [N, H*W]
        raise ValueError(f"Latents trop aplatis, shape={latents.shape}")
    elif latents.ndim == 3:  # [C,H,W] -> ajouter batch
        latents = latents.unsqueeze(0)  # [1,C,H,W]
    elif latents.ndim == 4:  # [B,C,H,W]
        pass
    elif latents.ndim == 5:  # [B,C,T,H,W]
        B, C, T, H, W = latents.shape
        latents = latents.reshape(B*T, C, H, W)
    else:
        raise ValueError(f"Latents must be 3D, 4D or 5D, got {latents.ndim}D")

    # Dupliquer si 1 seul canal
    if latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    # Décodage
    with torch.no_grad():
        decoded = vae.decode(latents / 0.18215).sample
        decoded = decoded.clamp(0, 1)

    return decoded


def decode_latents_to_image_SDiffusion(latents, vae, tile_size=128, overlap=64, device="cuda"):
    """
    Décodage VAE en tuiles 128x128, compatible latents 4D ou 5D.
    """
    # Convertir en float32
    latents = latents.to(torch.float32)

    # Fusion batch+time si nécessaire
    if latents.ndim == 5:
        B, C, T, H, W = latents.shape
        latents = latents.reshape(B*T, C, H, W)
    elif latents.ndim == 4:
        B, C, H, W = latents.shape
    else:
        raise ValueError(f"Latents must be 4D or 5D, got {latents.ndim}D")

    assert C == 4, f"Expected 4 channels in latents, got {C}"

    # Décodage complet (pas en tuiles pour simplifier ici)
    with torch.no_grad():
        decoded = vae.decode(latents / 0.18215).sample  # shape [B*T, 3, H, W]
        decoded = decoded.clamp(0, 1)

    # Remettre en 5D si nécessaire
    if 'T' in locals():
        decoded = decoded.reshape(B, T, 3, H, W)

    return decoded

def decode_latents_to_image_tiled4D(latents, vae, tile_size=128, overlap=64, device="cuda"):
    """
    Fonction corrigée pour décoder les latents en image via VAE, en traitant les tuiles.
    Cette version gère correctement la dimension des canaux latents (C=4) et la conversion en float32.
    """

    # Assurez-vous que latents ont la bonne forme [B, C, H, W] où C = 4
    B, C, H, W = latents.shape
    assert C == 4, f"Expected 4 channels in latents, got {C}"

    # Convertir en float32 si nécessaire (VAE attend des latents en float32)
    latents = latents.to(torch.float32)

    # Variable pour stocker les tuiles décodées
    decoded_image = torch.zeros(B, 3, H * tile_size, W * tile_size, device=device)

    # Transformation en image
    to_pil = ToPILImage()

    # Décodez les tuiles par petits morceaux
    for i in range(0, H, tile_size - overlap):
        for j in range(0, W, tile_size - overlap):
            # Extraire la tuile avec chevauchement
            y1, y2 = i, min(i + tile_size, H)
            x1, x2 = j, min(j + tile_size, W)
            latent_tile = latents[:, :, y1:y2, x1:x2]

            # Décoder la tuile
            with torch.no_grad():
                decoded_tile = vae.decode(latent_tile / 0.18215).sample

            # Ajuster les dimensions pour fusionner les tuiles
            decoded_tile = decoded_tile.clamp(0, 1)

            # Insérer la tuile dans la position correspondante de l'image finale
            decoded_image[:, :, y1 * tile_size:(y2 * tile_size), x1 * tile_size:(x2 * tile_size)] = decoded_tile

    return decoded_image



# -------------------------
# Fonction pour charger un VAE et tester son décodage
# -------------------------
def safe_load_vae(vae_path, device="cuda", fp16=False, offload=False):
    """
    Charge un VAE (FP32 ou FP16), renvoie l'objet VAE prêt à l'emploi.
    """
    try:
        # Chargement state_dict
        state_dict = load_file(vae_path, device="cpu")
        print("✅ State dict VAE chargé, clés:", list(state_dict.keys())[:5])

        # Création d'un VAE compatible SD
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"]*4,
            up_block_types=["UpDecoderBlock2D"]*4,
            block_out_channels=[128, 256, 512, 512],
            latent_channels=4,
            sample_size=32  # à adapter selon le checkpoint
        )

        # Chargement des poids
        vae.load_state_dict(state_dict, strict=False)

        # Offload si demandé
        if offload:
            vae = vae.to("cpu")
        else:
            vae = vae.to(device)
            if fp16:
                vae = vae.half()

        return vae

    except Exception as e:
        print(f"⚠ Erreur lors du chargement du VAE : {e}")
        return None


# -------------------------
# Fonction test VAE (sans affichage d'image)
# -------------------------
def test_vae_simple(vae_path, device="cuda"):
    try:
        vae = safe_load_vae(vae_path, device=device, fp16=False)
        if vae is None:
            return False

        # Tenseur aléatoire pour test
        test_latent = torch.randn(1, 4, 32, 32).to(device)
        with torch.no_grad():
            decoded_out = vae.decode(test_latent / 0.18215)
            decoded = decoded_out.sample if hasattr(decoded_out, "sample") else decoded_out

        # Vérification simple
        if decoded is not None and decoded.shape[1] == 3:
            return True
        return False

    except Exception as e:
        print(f"⚠ Test VAE échoué : {e}")
        return False


def test_vae_256(vae, image):
    """
    Test rapide du VAE 256x256.
    Vérifie encode -> decode sans afficher d'image.
    """

    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype

    transform = T.Compose([
        T.Resize(256),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
        print("🔎 Latent shape:", latents.shape)

        decoded = vae.decode(latents).sample
        print("🔎 Decoded shape:", decoded.shape)

    print("✅ Test VAE 256 OK")



def test_vae(vae_path: str, device: str = "cuda") -> bool:
    """
    Charge un VAE depuis un .safetensors et effectue un test de décodage rapide.
    Retourne True si le VAE est opérationnel, False sinon.
    """
    try:
        # Charge le state_dict depuis le fichier .safetensors
        state_dict = load_file(vae_path, device="cpu")
        print("✅ State dict chargé avec succès, clés:", list(state_dict.keys())[:5])

        # Crée un VAE standard compatible SD
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"]*4,
            up_block_types=["UpDecoderBlock2D"]*4,
            block_out_channels=[128, 256, 512, 512],
            latent_channels=4,
            sample_size=32  # adapter selon le checkpoint
        )

        # Ignore les clés manquantes ou inattendues
        vae.load_state_dict(state_dict, strict=False)
        vae = vae.to(device)
        print(f"✅ VAE chargé et déplacé sur {device}")

        # Test rapide avec un tenseur latent aléatoire
        test_latent = torch.randn(1, 4, 32, 32).to(device)
        with torch.no_grad():
            decoded_out = vae.decode(test_latent / 0.18215)
            decoded = decoded_out.sample if hasattr(decoded_out, "sample") else decoded_out

        # Vérifie juste la forme sans afficher l'image
        if decoded.shape[1] == 3:
            print(f"✅ Décodage test OK, output shape: {decoded.shape}")
            return True
        else:
            print(f"⚠ Décodage test incorrect, output shape: {decoded.shape}")
            return False

    except Exception as e:
        print("⚠ Erreur lors du test VAE :", e)
        return False



def safe_load_vae_safetensors(vae_path, device="cuda", fp16=False, offload=False):
    """
    Charge un VAE depuis un fichier .safetensors seul.
    """
    if not vae_path or not os.path.exists(vae_path):
        print(f"⚠ VAE non trouvé à {vae_path}")
        return None

    try:
        # Charger le state dict du fichier safetensors
        state_dict = load_file(vae_path, device="cpu")  # d'abord en CPU pour éviter OOM

        # Créer un AutoencoderKL vide (Tiny-SD 128x128)
        vae = AutoencoderKL(
            in_channels=4,
            out_channels=4,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[128, 256],
            latent_channels=4,
            sample_size=32
        )

        # Charger le state dict
        vae.load_state_dict(state_dict, strict=False)

        # Déplacer sur le device voulu et ajuster dtype
        vae = vae.to(torch.float16 if fp16 else torch.float32)
        vae = vae.to(device)

        return vae
    except Exception as e:
        print(f"⚠ Erreur lors du chargement du VAE : {e}")
        return None

def decode_latents_to_image_tiled128_old(latents, vae, tile_size=128, overlap=64, device="cuda"):
    """
    Decode latents en images RGB [0,1] avec tiles pour éviter mosaïque.

    Args:
        latents: Tensor [B, C, H, W] ou [B, C, F, H, W]
        vae: modèle VAE Stable Diffusion (FP32 recommandé)
        tile_size: taille de la tile pour le décodage
        overlap: recouvrement des tiles pour fusion lisse
        device: 'cuda' ou 'cpu'

    Returns:
        Tensor [B, 3, H, W] ou [B, F, 3, H, W] avec valeurs [0,1]
    """
    if latents.ndim == 5:  # [B, C, F, H, W]
        B, C, F, H, W = latents.shape
        output = []
        for f in range(F):
            frame = decode_latents_to_image_tiled128(
                latents[:, :, f, :, :], vae, tile_size=tile_size, overlap=overlap, device=device
            )
            output.append(frame.unsqueeze(1))  # garde dimension F
        return torch.cat(output, dim=1)  # [B, F, 3, H, W]

    B, C, H, W = latents.shape
    device = latents.device if latents.is_cuda else device
    latents = latents.to(device)

    # Scale Tiny-SD
    latents = latents / 0.18215

    # output tensor
    output_img = torch.zeros(B, 3, H, W, device=device, dtype=torch.float32)

    # Compute number of tiles
    y_steps = max((H - overlap) // (tile_size - overlap), 1)
    x_steps = max((W - overlap) // (tile_size - overlap), 1)

    for by in range(B):
        img_accum = torch.zeros(3, H, W, device=device)
        img_weight = torch.zeros(1, H, W, device=device)

        for y in range(y_steps):
            for x in range(x_steps):
                y_start = y * (tile_size - overlap)
                x_start = x * (tile_size - overlap)
                y_end = min(y_start + tile_size, H)
                x_end = min(x_start + tile_size, W)

                y_start = max(y_end - tile_size, 0)
                x_start = max(x_end - tile_size, 0)

                latent_tile = latents[by:by+1, :, y_start:y_end, x_start:x_end]

                with torch.no_grad():
                    decoded_tile = vae.decode(latent_tile).sample
                    decoded_tile = decoded_tile.clamp(0, 1)

                # Weight mask pour fusion
                weight = torch.ones(1, y_end - y_start, x_end - x_start, device=device)
                img_accum[:, y_start:y_end, x_start:x_end] += decoded_tile[0]
                img_weight[:, y_start:y_end, x_start:x_end] += weight

        output_img[by] = img_accum / img_weight
    return output_img.clamp(0, 1)


# -------------------------
# Decode latents VAE auto tile
# -------------------------
def decode_latents_frame_ai_auto(latents: torch.Tensor, vae, device="cuda"):
    """
    Decode latents en image PIL ou tensor, tile_size automatique:
      - Si max(H,W) <= 256: decode complet
      - Sinon: tiles 128x128, overlap 50%
    latents: [C,H/8,W/8] ou [B,C,H/8,W/8]
    vae: modèle VAE
    device: "cuda" ou "cpu"
    Retourne: Tensor float32 [3,H,W] en [0,1]
    """
    import torch
    latents = latents.to(device)

    # Si pas de batch dim, ajouter
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)  # [1,C,H,W]

    _, C, H_lat, W_lat = latents.shape
    H_out, W_out = H_lat*8, W_lat*8  # upscaling par VAE

    # Petites images → decode complet
    if max(H_out, W_out) <= 256:
        with torch.no_grad():
            frame_tensor = vae.decode(latents / 0.18215).sample  # Tiny-SD scale
            frame_tensor = torch.clamp(frame_tensor, -1.0, 1.0)
            frame_tensor = (frame_tensor + 1) / 2  # [-1,1] → [0,1]
            frame_tensor = frame_tensor[0]  # enlever batch dim
        return frame_tensor

    # Grandes images → decode en tiles
    tile_size = 128
    overlap = tile_size // 2
    _, C, H, W = latents.shape
    output = torch.zeros((1, 3, H*8, W*8), device=device)

    count_map = torch.zeros_like(output)

    # Générer les positions de tiles
    xs = list(range(0, W, tile_size - overlap))
    ys = list(range(0, H, tile_size - overlap))

    for y in ys:
        for x in xs:
            y1, y2 = y, min(y + tile_size, H)
            x1, x2 = x, min(x + tile_size, W)

            lat_tile = latents[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                dec_tile = vae.decode(lat_tile / 0.18215).sample
                dec_tile = torch.clamp(dec_tile, -1, 1)
                dec_tile = (dec_tile + 1) / 2  # [0,1]

            H_t, W_t = dec_tile.shape[2], dec_tile.shape[3]
            output[:, :, y1*8:y1*8+H_t*8, x1*8:x1*8+W_t*8] += dec_tile.repeat(1,1,8,8)
            count_map[:, :, y1*8:y1*8+H_t*8, x1*8:x1*8+W_t*8] += 1.0

    output /= count_map.clamp(min=1.0)
    return output[0]


# -------------------------
# Mémoire GPU utils
# -------------------------
def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        print(f"[GPU MEM] {tag} → allocated={torch.cuda.memory_allocated()/1e6:.1f}MB, "
              f"reserved={torch.cuda.memory_reserved()/1e6:.1f}MB, "
              f"max_allocated={torch.cuda.max_memory_allocated()/1e6:.1f}MB")



# -------------------------
# Encode / Decode avec logs
# -------------------------
def encode_images_to_latents_ai_test(images, vae):
    device = vae.device
    images = images.to(device=device, dtype=torch.float32)
    print(f"[VAE] Encode → device={device}, images.shape={images.shape}, dtype={images.dtype}")
    log_gpu_memory("avant encode VAE")
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
        latents = latents.unsqueeze(2)  # [B,C,1,H,W]
    print(f"[VAE] Latents shape après encode: {latents.shape}")
    log_gpu_memory("après encode VAE")
    return latents

def decode_latents_frame_ai(latents, vae):
    # --- Tile size auto ---
    _, C, H, W = latents.shape[-4:]  # [B,C,H,W]
    tile_size = min(H, W)  # tile couvre toute la latente pour éviter mosaïque
    overlap = tile_size // 2
    print(f"[VAE] Decode avec tile_size={tile_size}, overlap={overlap}, device={vae.device}, latents.shape={latents.shape}")
    log_gpu_memory("avant decode VAE")
    frame_tensor = decode_latents_to_image_tiled(latents, vae, tile_size=tile_size, overlap=overlap).clamp(0,1)
    print(f"[VAE] Frame tensor shape après decode: {frame_tensor.shape}")
    log_gpu_memory("après decode VAE")
    return frame_tensor


# -------------------------
# Encode / Decode
# -------------------------
def encode_images_to_latents(images, vae):
    device = vae.device
    images = images.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        if images.dim() == 5:  # [B,C,F,H,W]
            B, C, F, H, W = images.shape
            images_2d = images.view(B*F, C, H, W)
            latents_2d = vae.encode(images_2d).latent_dist.sample() * LATENT_SCALE
            latent_shape = latents_2d.shape
            latents = latents_2d.view(B, F, latent_shape[1], latent_shape[2], latent_shape[3])
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        else:
            latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
            latents = latents.unsqueeze(2)
    return latents

def decode_latents_to_image(latents, vae):
    latents = latents.to(vae.device).float() / LATENT_SCALE
    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0,1)
    return img

# -------------------------
# Model loaders
# -------------------------
# -------------------------
# Model loaders
# -------------------------
def safe_load_unet(model_path, device, fp16=False):
    folder = os.path.join(model_path, "unet")
    if os.path.exists(folder):
        model = UNet2DConditionModel.from_pretrained(folder)
        if fp16:
            model = model.half()
        return model.to(device)
    return None

def safe_load_vae_old(model_path, device, fp16=False, offload=False):
    folder = os.path.join(model_path, "vae")
    if os.path.exists(folder):
        model = AutoencoderKL.from_pretrained(folder)
        model = model.to("cpu" if offload else device).float()
        return model
    return None

def safe_load_scheduler(model_path):
    folder = os.path.join(model_path, "scheduler")
    if os.path.exists(folder):
        return DPMSolverMultistepScheduler.from_pretrained(folder)
    return None


# --- PIPELINE PRINCIPALE ---
def generate_5D_video_auto(pretrained_model_path, config, device='cuda'):
    print("🔄 Chargement des modèles...")
    motion_module = MotionModuleTiny(device=device)
    scheduler = init_scheduler(config)  # ta fonction existante
    vae = load_vae(pretrained_model_path, device=device)

    total_frames = config['total_frames']
    fps = config['fps']
    H_src, W_src = config['image_size']  # résolution source

    # Génère les latents initiaux
    latents = torch.randn(1, 4, H_src//8, W_src//8, device=device, dtype=torch.float16)
    print(f"[INFO] Latents initiaux shape={latents.shape}")

    video_frames = []
    for t in range(total_frames):
        try:
            latents = motion_module.step(latents, t)
            frame = decode_latents_frame_auto(latents, vae, H_src, W_src)
            video_frames.append(frame)
        except Exception as e:
            print(f"⚠ Erreur frame {t:05d} → reset léger: {e}")
            continue

    save_video(video_frames, fps, output_path=config['output_path'])
    print(f"🎬 Vidéo générée : {config['output_path']}")


def decode_latents_frame_auto(latents, vae, H_src, W_src):
    """
    Decode des latents VAE en images avec tiles 128x128, auto-adapté à la taille source.
    """
    device = vae.device
    print(f"[VAE] Decode → tile_size={tile_size}, overlap={overlap}, device={device}, latents.shape={latents.shape}")
    log_gpu_memory("avant decode VAE")

    # Assure batch 4D
    latents = latents.unsqueeze(0) if latents.dim() == 3 else latents

    # Décodage VAE en tiles
    with torch.no_grad():
        frame_tensor = decode_latents_to_image_tiled(
            latents,
            vae,
            tile_size=tile_size,
            overlap=overlap
        ).clamp(0,1)

        # Redimensionnement proportionnel à l'image source
        H_out, W_out = H_src, W_src
        if frame_tensor.shape[-2:] != (H_out, W_out):
            frame_tensor = torch.nn.functional.interpolate(
                frame_tensor,
                size=(H_out, W_out),
                mode='bicubic',
                align_corners=False
            )

    log_gpu_memory("après decode VAE")
    return frame_tensor.squeeze(0)

# ---------------------------------------------------------
# Tuilage sécurisé
# ---------------------------------------------------------
def decode_latents_to_image_tiled(latents, vae, tile_size=32, overlap=8):
    """
    Decode VAE en tuiles avec couverture complète garantie.
    - Aucun trou possible
    - Blending propre
    - Stable mathématiquement
    """

    device = vae.device
    latents = latents.to(device).float() / LATENT_SCALE

    B, C, H, W = latents.shape
    stride = tile_size - overlap

    # Dimensions image finale (scale factor VAE = 8)
    out_H = H * 8
    out_W = W * 8

    output = torch.zeros(B, 3, out_H, out_W, device=device)
    weight = torch.zeros_like(output)

    # --- positions garanties ---
    y_positions = list(range(0, H - tile_size + 1, stride))
    x_positions = list(range(0, W - tile_size + 1, stride))

    if not y_positions:
        y_positions = [0]
    if not x_positions:
        x_positions = [0]

    if y_positions[-1] != H - tile_size:
        y_positions.append(H - tile_size)

    if x_positions[-1] != W - tile_size:
        x_positions.append(W - tile_size)

    for y in y_positions:
        for x in x_positions:

            y1 = y + tile_size
            x1 = x + tile_size

            tile = latents[:, :, y:y1, x:x1]

            with torch.no_grad():
                decoded = vae.decode(tile).sample

            decoded = (decoded / 2 + 0.5).clamp(0, 1)

            iy0 = y * 8
            ix0 = x * 8
            iy1 = y1 * 8
            ix1 = x1 * 8

            output[:, :, iy0:iy1, ix0:ix1] += decoded
            weight[:, :, iy0:iy1, ix0:ix1] += 1.0

    return output / weight.clamp(min=1e-6)



# -------------------------
# Encode / Decode FP16 safe
# -------------------------
def encode_images_to_latents_safe(images, vae):
    device = vae.device
    dtype = next(vae.parameters()).dtype  # prend fp16 si le VAE est en FP16
    images = images.to(device=device, dtype=dtype)

    with torch.no_grad():
        if images.dim() == 5:  # [B,C,F,H,W]
            B, C, F, H, W = images.shape
            images_2d = images.view(B*F, C, H, W)
            latents_2d = vae.encode(images_2d).latent_dist.sample() * LATENT_SCALE
            latent_shape = latents_2d.shape
            latents = latents_2d.view(B, F, latent_shape[1], latent_shape[2], latent_shape[3])
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        else:
            latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
            latents = latents.unsqueeze(2)
    return latents


def decode_latents_to_image_safe(latents, vae):
    dtype = next(vae.parameters()).dtype
    latents = latents.to(vae.device).to(dtype) / LATENT_SCALE
    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0,1)
    return img

# ------------------------------
def encode_images_to_latents_half(images, vae):
    # récupère dtype réel du VAE
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    images = images.to(device=vae_device, dtype=vae_dtype)

    with torch.no_grad():

        if images.dim() == 5:  # [B,C,F,H,W]
            B, C, F, H, W = images.shape
            images_2d = images.view(B * F, C, H, W)

            latents_2d = vae.encode(images_2d).latent_dist.sample()
            latents_2d = latents_2d * LATENT_SCALE

            latents = latents_2d.view(
                B, F,
                latents_2d.shape[1],
                latents_2d.shape[2],
                latents_2d.shape[3]
            )

            latents = latents.permute(0, 2, 1, 3, 4).contiguous()

        else:
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * LATENT_SCALE
            latents = latents.unsqueeze(2)

    return latents

def decode_latents_to_image_vae(latents, vae):

    # Récupère device + dtype réel du VAE
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    # Aligne le dtype sur celui du VAE
    latents = latents.to(device=vae_device, dtype=vae_dtype)

    latents = latents / LATENT_SCALE

    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)

    # On repasse en float32 pour sauvegarde PNG
    return img.float()
# -------------------------
# Encode / Decode
# -------------------------
def encode_images_to_latents_ori(images, vae):
    device = vae.device
    images = images.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        if images.dim() == 5:  # [B,C,F,H,W]
            B, C, F, H, W = images.shape
            images_2d = images.view(B*F, C, H, W)
            latents_2d = vae.encode(images_2d).latent_dist.sample() * LATENT_SCALE
            latent_shape = latents_2d.shape
            latents = latents_2d.view(B, F, latent_shape[1], latent_shape[2], latent_shape[3])
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        else:
            latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
            latents = latents.unsqueeze(2)
    return latents

def decode_latents_to_image_ori(latents, vae):
    latents = latents.to(vae.device).float() / LATENT_SCALE
    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0,1)
    return img


# --------------------------------------------------------
# | Mode        | VAE     | Images  | Latents | Résultat  |
# | ----------- | ------- | ------- | ------- | --------  |
# | fp32        | float32 | float32 | float32 | ✅        |
# | fp16        | float16 | float16 | float16 | ✅        |
# | offload CPU | float32 | float32 | float32 | ✅        |
# ------------------------- ci dessous:
# -------------------------
# Encode / Decode corrigé FP16 safe
# -------------------------
def encode_images_to_latents(images, vae):
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype  # on aligne avec le VAE
    images = images.to(device=device, dtype=dtype)
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
    return latents

def decode_latents_to_image(latents, vae):
    # On force latents à avoir le même dtype et device que le VAE
    vae_dtype = next(vae.parameters()).dtype
    vae_device = next(vae.parameters()).device
    latents = latents.to(device=vae_device, dtype=vae_dtype) / LATENT_SCALE

    with torch.no_grad():
        img = vae.decode(latents).sample

    # Normalisation sûre vers 0-1
    img = (img / 2 + 0.5).clamp(0, 1)

    # Si FP16 → convertir en float32 pour torchvision save_image
    if img.dtype == torch.float16:
        img = img.float()

    return img

# NEW
#
#
# ---------------------------------------------------------
# Decode latents to image avec logs et sécurité
# ---------------------------------------------------------
def decode_latents_to_image_2(latents, vae, latent_scale=0.18215):
    """
    latents: [B, C, F, H, W] ou [B, C, 1, H, W] pour frame unique
    vae: VAE pour décodage
    """
    try:
        print(f"🔹 decode_latents_to_image_2 | input shape: {latents.shape}, dtype: {latents.dtype}, device: {latents.device}")

        # Si latents a une dimension de frame singleton, la squeeze
        if latents.shape[2] == 1:
            latents = latents.squeeze(2)
            print(f"🔹 Squeeze frame dimension → shape: {latents.shape}")

        # Assurer dtype et device compatible VAE
        vae_dtype = next(vae.parameters()).dtype
        vae_device = next(vae.parameters()).device
        latents = latents.to(device=vae_device, dtype=vae_dtype) / latent_scale

        # Check NaN avant VAE
        print(f"🔹 Latents before VAE decode | min: {latents.min()}, max: {latents.max()}, dtype: {latents.dtype}")
        if torch.isnan(latents).any():
            print("❌ Warning: NaN detected in latents before VAE decode!")

        with torch.no_grad():
            img = vae.decode(latents).sample

        # Check NaN après décodage
        print(f"🔹 Image after VAE decode | min: {img.min()}, max: {img.max()}, dtype: {img.dtype}")
        if torch.isnan(img).any():
            print("❌ Warning: NaN detected in decoded image!")

        # Normalisation safe vers 0-1
        img = (img / 2 + 0.5).clamp(0, 1)
        print(f"🔹 Image final | min: {img.min()}, max: {img.max()}, dtype: {img.dtype}, shape: {img.shape}")

        # Conversion FP16 -> FP32 si nécessaire
        if img.dtype == torch.float16:
            img = img.float()

        return img

    except Exception as e:
        print(f"❌ Exception in decode_latents_to_image_2: {e}")
        # Retourne une image noire safe si VAE échoue
        B, C, H, W = latents.shape[:4]
        return torch.zeros(B, 3, H*8, W*8, device=latents.device)  # scale approx 8x pour SD VAE
# -------------------------
# Encode / Decode corrigé
# -------------------------
# -------------------------

def decode_latents_to_image_old(latents, vae):
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(device=vae_device, dtype=vae_dtype)
    latents = latents / LATENT_SCALE
    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
    return img.float()  # on repasse en float32 pour PNG
