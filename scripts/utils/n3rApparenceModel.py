# n3rApparenceModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime


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
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        h = self.encoder(x)
        delta = self.decoder(h)
        delta = torch.tanh(delta)  # bornage esthétique
        return self.scale * delta


# instance par défaut pour ton pipeline
appearance_model = AppearanceModel().cuda()

# =========================================================
# LATENT SANITIZE
# =========================================================

def sanitize_latents(latents):
    """
    Normalisation par sample.
    """

    latents = latents.float()

    mean = latents.mean(dim=(1, 2, 3), keepdim=True)
    std = latents.std(dim=(1, 2, 3), keepdim=True)

    latents = (latents - mean) / (std + 1e-6)
    latents = latents.clamp(-4.0, 4.0)

    return latents

# =========================================================
# APPLY FUNCTION
# =========================================================
def apply_appearance(latents, appearance_model, strength=0.1, device="cuda", debug=False):
    """
    Injection esthétique contrôlée avec stabilisation automatique.
    Évite toute teinte globale ou dérive de couleur.
    """

    appearance_model.to(device).eval()
    latents = latents.to(device)

    with torch.no_grad():
        delta = appearance_model(latents)

        # --- 1️⃣ centrer par canal ---
        delta = delta - delta.mean(dim=(2, 3), keepdim=True)

        # --- 2️⃣ high-pass : garder seulement les détails fins ---
        delta = delta - F.avg_pool2d(delta, kernel_size=3, stride=1, padding=1)

        # --- 3️⃣ ajustement automatique de la force ---
        # limiter la std du delta pour qu'elle reste proportionnelle à strength
        delta_std = delta.std(dim=(1,2,3), keepdim=True)
        max_std = 0.1  # valeur sécurisée pour ne pas saturer
        scale_factor = torch.clamp(max_std / (delta_std + 1e-6), max=1.0)
        delta = delta * scale_factor

    # injection résiduelle contrôlée
    out = latents + strength * delta

    # bornage
    out = torch.clamp(out, -1.0, 1.0)

    # sanitize pour stabiliser la distribution des latents
    latents = sanitize_latents(out).to(device)

    if debug:
        print(f"[Appearance AUTO] delta_std={delta.std().item():.4f} | strength={strength}")

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
