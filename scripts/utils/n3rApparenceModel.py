# n3rApparenceModel.py
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .tools_utils import ensure_4_channels, log_debug, sanitize_latents




def compute_high_freq_energy(latents, kernel_size=3, normalize=True, per_channel=False):

    latents = latents.float()

    blur = F.avg_pool2d(
        latents,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )

    high_freq = latents - blur

    hf_energy = torch.sqrt(high_freq.pow(2).mean(dim=(2, 3)) + 1e-8)

    if normalize:
        base = torch.sqrt(latents.pow(2).mean(dim=(2, 3)) + 1e-8)
        hf_energy = hf_energy / (base + 1e-6)

    if per_channel:
        return hf_energy

    return hf_energy.mean(dim=1)

def stabilize_latents(latents, target_std=1.0):

    std = latents.std(dim=(1,2,3), keepdim=True)
    scale = target_std / (std + 1e-6)
    scale = torch.clamp(scale, 0.7, 1.3)

    return latents * scale

# =========================================================
# SAVE / LOAD
# =========================================================

def save_appearance_model(model, optimizer=None, epoch=None, loss=None,
                          path="models/appearance_model.pt",
                          latest_path="models/appearance_model_latest.pt"):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": getattr(model, "config", {}),
        "model_version": getattr(model, "version", "v2"),
        "timestamp": datetime.datetime.now().isoformat()
    }

    if optimizer:
        checkpoint["optimizer_state"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if loss is not None:
        checkpoint["loss"] = float(loss)

    torch.save(checkpoint, path)
    torch.save(checkpoint, latest_path)

    print(f"[INFO] Saved AppearanceModel -> {path}")


# =========================================================
# LOAD ( V2 COMPATIBLE)
# =========================================================
def load_appearance_model(model_class,
                          path="models/appearance_model_latest.pt",
                          optimizer=None,
                          device="cuda"):

    if not os.path.exists(path):
        print("[WARN] No checkpoint found.")
        return model_class().to(device), None

    checkpoint = torch.load(path, map_location=device)

    model = model_class(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)

    if optimizer and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as e:
            print(f"[WARN] optimizer not loaded: {e}")

    print(
        f"[INFO] Loaded AppearanceModel | "
        f"version={checkpoint.get('model_version')} | "
        f"epoch={checkpoint.get('epoch')} | "
        f"loss={checkpoint.get('loss')} | "
        f"time={checkpoint.get('timestamp')}"
    )

    return model, checkpoint


# =========================================================
# MODEL
# =========================================================

class AppearanceModel(nn.Module):
    """
    V2 — Photometric Renderer (Exposure / Tone Mapping)
    """

    def __init__(self, in_channels=4, base_channels=32, prompt_dim=768):

        super().__init__()

        self.config = {
            "in_channels": in_channels,
            "base_channels": base_channels,
            "prompt_dim": prompt_dim
        }

        self.version = "v2"

        # -------------------------------------------------
        # ENCODER
        # -------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        # -------------------------------------------------
        # PROMPT
        # -------------------------------------------------
        self.prompt_proj = nn.Linear(prompt_dim, base_channels)

        # -------------------------------------------------
        # FUSION
        # -------------------------------------------------
        self.fusion = nn.Conv2d(base_channels * 2, base_channels, 1)

        # -------------------------------------------------
        # GLOBAL HEAD
        # -------------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d(1)

        hidden = base_channels

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )

        # -------------------------------------------------
        # RENDER PARAMETERS
        # -------------------------------------------------
        self.exposure = nn.Linear(hidden, 1)
        self.gamma    = nn.Linear(hidden, 1)
        self.contrast = nn.Linear(hidden, 1)
        self.micro    = nn.Linear(hidden, 1)

    def forward(self, x, prompt_emb):

        h = self.encoder(x)

        # prompt
        if prompt_emb.dim() == 3:
            prompt_emb = prompt_emb[:, 0, :]

        p = self.prompt_proj(prompt_emb)
        p = p[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])

        # fusion
        h = self.fusion(torch.cat([h, p], dim=1))

        # global
        g = self.pool(h).squeeze(-1).squeeze(-1)
        g = self.head(g)

        return {
            "exposure": self.exposure(g),
            "gamma": self.gamma(g),
            "contrast": self.contrast(g),
            "micro": self.micro(g)
        }

# instance par défaut pour ton pipeline
appearance_model = AppearanceModel().cuda()

# séparer scale (très lent) et reste du réseau
scale_params = []
other_params = []

for name, p in appearance_model.named_parameters():
    if "scale" in name:
        scale_params.append(p)
    else:
        other_params.append(p)

optimizer_apparence = optim.AdamW(
    [
        {"params": other_params, "lr": 1e-4},
        {"params": scale_params, "lr": 3e-5},  # très stable, très lent
    ],
    betas=(0.9, 0.99),
    weight_decay=1e-5
)

criterion_apparence = torch.nn.MSELoss()


# =========================================================
# APPLY FUNCTION
# =========================================================
def apply_appearance(
    latents,
    style_prompt_embedding,
    appearance_model,
    optimizer=None,
    criterion=None,
    train=False,
    strength=0.1,
    device="cuda",
    frame_counter=0,
    max_epochs_up=3,
    model_path="models/appearance_model_latest.pt",
    ema_prev_latents=None,
    ema_alpha=0.3,
    new_image=False,
    debug=False
):

    device = latents.device
    appearance_model.to(device)

    x = latents.to(device)
    x0 = x.detach()

    print(f"[Appearance V2] device={device}")

    hf = compute_high_freq_energy(x0)
    print(f"[Appearance V2] hf={hf.mean().item():.4f}")

    model_exists = os.path.exists(model_path)

    # =====================================================
    # TRAIN
    # =====================================================
    if train and optimizer and criterion:

        appearance_model.train()

        max_epochs = max(1, max_epochs_up)

        print("[Appearance V2] Training photometric renderer")

        for epoch in range(max_epochs):

            optimizer.zero_grad()
            with torch.enable_grad():

                pred = appearance_model(x0, style_prompt_embedding)

                exposure = pred["exposure"].view(-1,1,1,1)
                gamma    = pred["gamma"].view(-1,1,1,1)
                contrast = pred["contrast"].view(-1,1,1,1)
                micro    = pred["micro"].view(-1,1,1,1)

                out = x0

                # exposure
                out = out * (1.0 + 0.5 * torch.tanh(exposure))

                # gamma
                out = torch.sign(out) * (torch.abs(out) ** (1.0 + 0.3 * torch.tanh(gamma)))

                # contrast
                m = out.mean(dim=(2,3), keepdim=True)
                out = (out - m) * (1.0 + torch.tanh(contrast)) + m

                # micro
                out = out + 0.05 * torch.tanh(micro)

                # loss
                loss = F.l1_loss(out, x0) + 0.01 * out.pow(2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(appearance_model.parameters(), 1.0)
            optimizer.step()

            print(f"[Appearance V2] Epoch {epoch+1}/{max_epochs} | Loss={loss.item():.6f}")

            should_save = (frame_counter % 10 == 0) and (epoch == max_epochs - 1)

            if should_save:
                save_appearance_model(
                    appearance_model,
                    optimizer=optimizer,
                    epoch=frame_counter,
                    loss=loss.item(),
                    path=model_path
                )

    # =====================================================
    # LOAD
    # =====================================================
    else:

        appearance_model.eval()

        if model_exists:
            appearance_model, _ = load_appearance_model(
                type(appearance_model),
                path=model_path,
                optimizer=optimizer,
                device=device
            )

    # =====================================================
    # INFERENCE
    # =====================================================
    with torch.no_grad():

        pred = appearance_model(x, style_prompt_embedding)

        exposure = torch.tanh(pred["exposure"])
        gamma    = torch.tanh(pred["gamma"])
        contrast = torch.tanh(pred["contrast"])
        micro    = torch.tanh(pred["micro"])

        out = x

        out = out * (1.0 + 0.4 * exposure)

        out = torch.sign(out) * (torch.abs(out) ** (1.0 + 0.25 * gamma))

        m = out.mean(dim=(2,3), keepdim=True)
        out = (out - m) * (1.0 + contrast) + m

        out = out + 0.05 * micro

    # =====================================================
    # STRENGTH
    # =====================================================
    hf = compute_high_freq_energy(x)
    strength_map = strength / (1.0 + 2.5 * hf)
    strength_map = strength_map.clamp(0.01, strength)

    out = x + strength_map * (out - x)

    # =====================================================
    # EMA
    # =====================================================
    if ema_prev_latents is not None and not new_image:

        alpha = ema_alpha * (1.0 - 0.2 * hf.mean())
        alpha = alpha.clamp(0.05, 0.3)

        out = alpha * out + (1.0 - alpha) * ema_prev_latents

    return torch.clamp(out, -3.0, 3.0)










