# n3rApparenceModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime
from .tools_utils import ensure_4_channels, log_debug, sanitize_latents




def compute_high_freq_energy(
    latents,
    kernel_size=3,
    normalize=True,
    per_channel=False
):
    """
    Mesure l'énergie haute fréquence des latents.

    Args:
        latents: Tensor [B,C,H,W]
        kernel_size: taille du blur
        normalize: normalise par l'énergie globale
        per_channel: retourne une mesure par canal

    Returns:
        Tensor shape:
            [B]              si per_channel=False
            [B,C]            si per_channel=True
    """

    latents = latents.float()

    # blur basse fréquence
    blur = F.avg_pool2d(
        latents,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )

    # composante haute fréquence
    high_freq = latents - blur

    # énergie RMS
    hf_energy = torch.sqrt(
        high_freq.pow(2).mean(dim=(2, 3)) + 1e-8
    )

    if normalize:
        base_energy = torch.sqrt(
            latents.pow(2).mean(dim=(2, 3)) + 1e-8
        )

        hf_energy = hf_energy / (base_energy + 1e-6)

    if per_channel:
        return hf_energy

    # moyenne sur canaux
    return hf_energy.mean(dim=1)

def stabilize_latents(latents, target_std=1.0):

    current_std = latents.std(dim=(1,2,3), keepdim=True)

    scale = target_std / (current_std + 1e-6)

    scale = torch.clamp(scale, 0.7, 1.3)

    latents = latents * scale

    return latents

# =========================================================
# SAVE / LOAD
# =========================================================

def save_appearance_model(
    model,
    optimizer=None,
    epoch=None,
    loss=None,
    path="models/appearance_model.pt",
    latest_path="models/appearance_model_latest.pt"
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": getattr(model, "config", {}),
        "timestamp": datetime.datetime.now().isoformat()
    }

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if loss is not None:
        checkpoint["loss"] = loss

    torch.save(checkpoint, path)
    torch.save(checkpoint, latest_path)
    print(f"[INFO] AppearanceModel saved -> {path}")


def load_appearance_model(
    model_class,
    path="models/appearance_model_latest.pt",
    optimizer=None,
    device="cuda"
):
    if not os.path.exists(path):
        print("[WARN] No checkpoint found.")
        model = model_class().to(device)
        return model, None

    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("model_config", {})
    model = model_class(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(
        f"[INFO] Loaded AppearanceModel | "
        f"epoch={checkpoint.get('epoch')} | "
        f"loss={checkpoint.get('loss')} | "
        f"time={checkpoint.get('timestamp')}"
    )

    return model, checkpoint


# =========================================================
# MODEL
# =========================================================

class AppearanceModel(nn.Module):
    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()
        self.config = {"in_channels": in_channels, "base_channels": base_channels}

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

        # contrôle global (important pour éviter explosion esthétique)
        self.scale = nn.Parameter(torch.tensor(0.01)) # 0.01 ou 0.005

    def forward(self, x):
        h = self.encoder(x)
        delta = self.decoder(h)
        delta = torch.tanh(delta)  # bornage esthétique
        return self.scale * delta


# instance par défaut pour ton pipeline
appearance_model = AppearanceModel().cuda()


# =========================================================
# APPLY FUNCTION
# =========================================================
def apply_appearance(
    latents,
    appearance_model,
    strength=0.1,
    device="cuda",
    debug=False
):
    """
    Injection esthétique stable pour latents vidéo.

    Objectifs :
    - zéro dérive couleur (brun/orange supprimé)
    - stabilité temporelle
    - pas de saturation progressive
    - injection contrôlée adaptative
    """

    appearance_model.to(device).eval()
    latents = latents.to(device)

    with torch.no_grad():

        # =========================================================
        # 1. DELTA BRUT
        # =========================================================
        delta = appearance_model(latents)

        # =========================================================
        # 2. NEUTRALISATION COULEUR (CRUCIAL)
        # =========================================================

        # suppression biais spatial global
        delta = delta - delta.mean(dim=(2, 3), keepdim=True)

        # suppression biais inter-canaux (anti "warm tint")
        delta = delta - delta.mean(dim=1, keepdim=True)

        # =========================================================
        # 3. NETTOYAGE HAUTES FRÉQUENCES
        # =========================================================

        blur = F.avg_pool2d(delta, kernel_size=3, stride=1, padding=1)
        delta = delta - blur

        # =========================================================
        # 4. NORMALISATION ÉNERGIE STABLE
        # =========================================================

        delta_std = delta.std(dim=(2, 3), keepdim=True)
        delta = delta / (delta_std + 1e-6)

        # amplitude contrôlée (important pour éviter saturation)
        delta = delta * 0.01

        # =========================================================
        # 5. STRENGTH DYNAMIQUE (ANTI SUR-EXCITATION)
        # =========================================================

        hf = compute_high_freq_energy(latents)

        k = 1.5
        dynamic_strength = strength * torch.exp(-k * hf)

        dynamic_strength = torch.clamp(
            dynamic_strength,
            min=0.005,
            max=strength
        )

        # =========================================================
        # 6. INJECTION
        # =========================================================

        out = latents + dynamic_strength * delta

        # clamp doux (évite explosion sans détruire distribution)
        out = torch.clamp(out, -4.0, 4.0)

        # stabilisation finale (non destructive)
        latents = stabilize_latents(out)

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print(
            f"[Appearance] "
            f"delta_std={delta.std().item():.4f} | "
            f"hf={hf.mean().item():.4f} | "
            f"strength={dynamic_strength.mean().item():.4f}"
        )

    return latents


def apply_appearance_simple(latents, appearance_model, strength=0.1, device="cuda", debug=False):
    appearance_model.to(device).eval()
    latents = latents.to(device)

    with torch.no_grad():
        delta = appearance_model(latents)
        # centrer par canal
        delta = delta - delta.mean(dim=(2,3), keepdim=True)
        # high-pass pour ne garder que les détails fins
        delta = delta - F.avg_pool2d(delta, kernel_size=3, stride=1, padding=1)

    out = latents + strength * delta
    out = torch.clamp(out, -1.0, 1.0)
    latents = sanitize_latents(out).to(device)

    if debug:
        print(f"[Appearance] delta_std={delta.std().item():.4f} | strength={strength}")

    return latents
