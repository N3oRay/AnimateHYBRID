# n3rApparenceModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime
import torch.optim as optim
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

scale_params = [appearance_model.scale]
other_params = [p for n, p in appearance_model.named_parameters() if n != "scale"]

optimizer_apparence = optim.AdamW([
    {"params": other_params, "lr": 1e-4},
    {"params": scale_params, "lr": 5e-5}  # très important: plus lent
],
betas=(0.9, 0.99),
weight_decay=1e-5)

criterion_apparence = torch.nn.MSELoss()


# =========================================================
# APPLY FUNCTION
# =========================================================
import torch
import torch.nn.functional as F
import os


def apply_appearance(
    latents,
    appearance_model,
    optimizer=None,
    criterion=None,
    train=False,
    device="cuda",
    strength=0.1,
    debug=False,
    new_image=False,
    frame_counter=0,
    max_epochs_up=5,
    model_path="models/appearance_model_latest.pt",
    ema_prev_latents=None,
    ema_alpha=0.3
):
    """
    Appearance consistency system (temporal-style architecture)

    Features:
    - Train / Eval modes
    - Dynamic training schedule
    - EMA smoothing
    - Residual latent injection
    - Stable HF-aware modulation
    """

    # =====================================================
    # DEVICE
    # =====================================================
    device = latents.device
    appearance_model.to(device)

    print(f"[Appearance] device={device}")

    latents = latents.to(device)

    # =====================================================
    # FEATURE ANALYSIS (style complexity)
    # =====================================================
    latents_fp = latents.float()

    hf = compute_high_freq_energy(latents_fp)

    texture_complexity = hf  # simple but stable proxy

    print(f"[Appearance] hf={hf.mean().item():.4f}")

    # =====================================================
    # PREP LATENTS
    # =====================================================
    latents_train = latents.clone().detach()
    latents_train.requires_grad_(False)

    model_exists = os.path.exists(model_path)

    # =====================================================
    # TRAIN MODE
    # =====================================================
    if train:

        print("[Appearance] Dynamic training mode")

        appearance_model.train()

        # -------------------------------------------------
        # DYNAMIC EPOCHS (like temporal)
        # -------------------------------------------------
        if frame_counter == 0:
            max_epochs = max_epochs_up
        else:
            max_epochs = int(
                max_epochs_up * (0.5 + 0.5 * texture_complexity.mean().item())
            )

        max_epochs = max(1, max_epochs)

        print(f"[Appearance] max_epochs={max_epochs}")

        # -------------------------------------------------
        # ENABLE GRAD
        # -------------------------------------------------
        for p in appearance_model.parameters():
            p.requires_grad = True

        # -------------------------------------------------
        # TRAIN LOOP
        # -------------------------------------------------
        for epoch in range(max_epochs):

            optimizer.zero_grad()

            with torch.enable_grad():

                delta = appearance_model(latents_train)

                # anti global drift (safe)
                delta = delta - delta.mean(dim=(2,3), keepdim=True)

                # prediction
                pred = latents_train + strength * delta

                target = latents

                loss = criterion(pred, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                appearance_model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            print(
                f"[Appearance] Epoch [{epoch+1}/{max_epochs}] "
                f"Loss={loss.item():.6f}"
            )

            if debug:
                print(f"[Appearance DEBUG] pred_std={pred.std().item():.4f}")

        # -------------------------------------------------
        # SAVE MODEL
        # -------------------------------------------------
        save_appearance_model(
            appearance_model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=loss.item(),
            path=model_path
        )

    # =====================================================
    # EVAL MODE
    # =====================================================
    else:

        print("[Appearance] Eval mode")

        if model_exists:
            appearance_model, checkpoint = load_appearance_model(
                type(appearance_model),
                path=model_path,
                optimizer=optimizer,
                device=device
            )
        else:
            print("[WARN] No appearance model found")

        appearance_model.eval()

    # =====================================================
    # INFERENCE
    # =====================================================
    with torch.no_grad():

        delta = appearance_model(latents)

        # -------------------------------------------------
        # STABLE NORMALIZATION (non destructive)
        # -------------------------------------------------
        delta = delta - delta.mean(dim=(2,3), keepdim=True)

        delta_std = delta.std(dim=(2,3), keepdim=True)
        delta = delta / (delta_std + 1e-6)

        delta = torch.tanh(delta)

    # =====================================================
    # DYNAMIC STRENGTH (HF aware)
    # =====================================================
    hf = compute_high_freq_energy(latents)

    dynamic_strength = strength / (1.0 + 1.5 * hf)

    dynamic_strength = torch.clamp(
        dynamic_strength,
        min=0.02,
        max=strength
    )

    print(f"[Appearance] strength={dynamic_strength.mean().item():.4f}")

    # =====================================================
    # INJECTION
    # =====================================================
    out = latents + dynamic_strength * appearance_model.scale * delta

    # =====================================================
    # EMA SMOOTHING
    # =====================================================
    if ema_prev_latents is not None and new_image is False:

        print("[Appearance] EMA applied")

        if ema_prev_latents.device != out.device:
            ema_prev_latents = ema_prev_latents.to(out.device)

        out = (
            ema_alpha * out +
            (1.0 - ema_alpha) * ema_prev_latents
        )

    # =====================================================
    # FINAL SAFETY CLAMP
    # =====================================================
    out = torch.clamp(out, -3.0, 3.0)

    # =====================================================
    # DEBUG
    # =====================================================
    if debug:
        print(
            f"[Appearance FINAL] "
            f"hf={hf.mean().item():.4f} | "
            f"delta_std={delta.std().item():.4f}"
        )

    return out



def apply_appearance_simple(
    latents,
    appearance_model,
    strength=0.1,
    device="cuda",
    debug=True
):
    """
    Stable appearance injection (diffusion-grade)

    Principes :
    - pas de re-normalisation globale
    - pas de whitening destructif
    - correction uniquement directionnelle
    - conservation du manifold latent
    """

    appearance_model.to(device).eval()
    latents = latents.to(device)

    with torch.no_grad():

        # =========================================================
        # 1. DELTA BRUT
        # =========================================================
        delta = appearance_model(latents)

        # =========================================================
        # 2. ANTI COLOR DRIFT (minimal, safe)
        # =========================================================
        # suppression biais couleur global uniquement
        delta = delta - delta.mean(dim=1, keepdim=True)

        # =========================================================
        # 3. NORMALISATION "SOFT" (PAS DE WHITENING)
        # =========================================================
        delta_std = delta.std(dim=(1,2,3), keepdim=True)

        delta = delta / (delta_std + 1e-6)

        # gain contrôlé (important pour préserver contraste)
        delta = delta * 0.03

        # =========================================================
        # 4. STRENGTH DYNAMIQUE STABLE
        # =========================================================
        hf = compute_high_freq_energy(latents)

        # courbe douce (évite instabilité exponentielle)
        dynamic_strength = strength / (1.0 + 1.5 * hf)

        dynamic_strength = torch.clamp(
            dynamic_strength,
            min=0.02,
            max=strength
        )

        # =========================================================
        # 5. INJECTION (SANS POST-PROCESS GLOBAL)
        # =========================================================
        out = latents + dynamic_strength * delta

        # clamp léger uniquement pour sécurité numérique
        out = torch.clamp(out, -3.0, 3.0)

    # =========================================================
    # 6. STABILISATION ULTRA LIGHT (OPTIONNEL)
    # =========================================================
    # uniquement si dérive longue durée
    latents = out  # PAS de stabilize_latents ici

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print(
            f"[Appearance PRO] "
            f"delta_std={delta.std().item():.4f} | "
            f"hf={hf.mean().item():.4f} | "
            f"strength={dynamic_strength.mean().item():.4f}"
        )

    return latents






