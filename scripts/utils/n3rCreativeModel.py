# n3rCreativeModel.py
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .tools_utils import ensure_4_channels, log_debug, sanitize_latents



# =========================================================
# Compute high freq
# =========================================================

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

def save_creative_model(
    model,
    optimizer=None,
    epoch=None,
    loss=None,
    path="models/creative_model.pt",
    latest_path="models/creative_model_latest.pt",

    # creative metadata
    style_name=None,
    prompt_signature=None,
    hf_mean=None,
    training_mode="semantic_decorator",

    # optional temporal state
    ema_state=None,

    # extra debug infos
    extra_metadata=None
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    checkpoint = {

        # =================================================
        # MODEL
        # =================================================

        "model_state": model.state_dict(),

        "model_config": getattr(
            model,
            "config",
            {}
        ),

        "model_version": getattr(
            model,
            "version",
            "v1_creative"
        ),

        # =================================================
        # TRAINING
        # =================================================

        "epoch": epoch,
        "loss": float(loss) if loss is not None else None,

        # =================================================
        # CREATIVE METADATA
        # =================================================

        "style_name": style_name,
        "prompt_signature": prompt_signature,
        "hf_mean": hf_mean,
        "training_mode": training_mode,

        # =================================================
        # TEMPORAL
        # =================================================

        "ema_state": ema_state,

        # =================================================
        # DEBUG / CUSTOM
        # =================================================

        "extra_metadata": extra_metadata or {},

        # =================================================
        # TIMESTAMP
        # =================================================

        "timestamp": datetime.datetime.now().isoformat()
    }

    # =====================================================
    # OPTIMIZER
    # =====================================================

    if optimizer is not None:

        checkpoint["optimizer_state"] = (
            optimizer.state_dict()
        )

    # =====================================================
    # SAVE
    # =====================================================

    torch.save(checkpoint, path)

    torch.save(checkpoint, latest_path)

    # =====================================================
    # LOG
    # =====================================================

    print(
        f"[INFO] Saved CreativeModel | "
        f"version={checkpoint['model_version']} | "
        f"epoch={epoch} | "
        f"loss={loss} | "
        f"path={path}"
    )

    if style_name is not None:

        print(
            f"[Creative] style={style_name}"
        )

    if hf_mean is not None:

        print(
            f"[Creative] hf_mean={hf_mean:.4f}"
        )

# =========================================================
# LOAD ( V1 COMPATIBLE)
# =========================================================
def load_creative_model(
    model_class,
    path="models/creative_model_latest.pt",
    optimizer=None,
    device="cuda",

    # compatibility
    strict=False,

    # restore extras
    load_optimizer=True,
    load_ema=True,

    # debug
    verbose=True
):

    # =====================================================
    # CHECKPOINT EXISTS
    # =====================================================

    if not os.path.exists(path):

        print(
            "[WARN] No creative checkpoint found."
        )

        model = model_class().to(device)

        return model, None

    # =====================================================
    # LOAD CHECKPOINT
    # =====================================================

    checkpoint = torch.load(
        path,
        map_location=device
    )

    # =====================================================
    # BUILD MODEL
    # =====================================================

    model_config = checkpoint.get(
        "model_config",
        {}
    )

    model = model_class(
        **model_config
    ).to(device)

    # =====================================================
    # LOAD WEIGHTS
    # =====================================================

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state"],
        strict=strict
    )

    # =====================================================
    # LOAD OPTIMIZER
    # =====================================================

    if (
        optimizer is not None
        and
        load_optimizer
        and
        "optimizer_state" in checkpoint
    ):

        try:

            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )

        except Exception as e:

            print(
                f"[WARN] optimizer not loaded: {e}"
            )

    # =====================================================
    # EXTRAS
    # =====================================================

    ema_state = None

    if load_ema:

        ema_state = checkpoint.get(
            "ema_state",
            None
        )

    # =====================================================
    # METADATA
    # =====================================================

    model_version = checkpoint.get(
        "model_version",
        "unknown"
    )

    style_name = checkpoint.get(
        "style_name",
        None
    )

    prompt_signature = checkpoint.get(
        "prompt_signature",
        None
    )

    hf_mean = checkpoint.get(
        "hf_mean",
        None
    )

    training_mode = checkpoint.get(
        "training_mode",
        None
    )

    # =====================================================
    # LOG
    # =====================================================

    if verbose:

        print(
            f"[INFO] Loaded CreativeModel | "
            f"version={model_version} | "
            f"epoch={checkpoint.get('epoch')} | "
            f"loss={checkpoint.get('loss')} | "
            f"time={checkpoint.get('timestamp')}"
        )

        if style_name is not None:

            print(
                f"[Creative] style={style_name}"
            )

        if prompt_signature is not None:

            print(
                f"[Creative] prompt='{prompt_signature}'"
            )

        if hf_mean is not None:

            print(
                f"[Creative] hf_mean={hf_mean:.4f}"
            )

        if training_mode is not None:

            print(
                f"[Creative] mode={training_mode}"
            )

        # partial compatibility infos

        if len(missing_keys) > 0:

            print(
                f"[WARN] Missing keys: "
                f"{len(missing_keys)}"
            )

        if len(unexpected_keys) > 0:

            print(
                f"[WARN] Unexpected keys: "
                f"{len(unexpected_keys)}"
            )

    # =====================================================
    # RETURN
    # =====================================================

    checkpoint["_ema_state_loaded"] = ema_state

    return model, checkpoint


# =========================================================
# MODEL
# =========================================================

class CreativeDecoratorModel(nn.Module):
    """
    Creative semantic latent decorator
    """

    def __init__(
        self,
        in_channels=4,
        base_channels=32,
        prompt_dim=768
    ):

        super().__init__()

        self.config = {
            "in_channels": in_channels,
            "base_channels": base_channels,
            "prompt_dim": prompt_dim
        }

        self.version = "v1_creative"

        # -------------------------------------------------
        # LATENT ENCODER
        # -------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        # -------------------------------------------------
        # PROMPT ENCODER
        # -------------------------------------------------

        self.prompt_proj = nn.Linear(prompt_dim, base_channels)

        # -------------------------------------------------
        # FUSION
        # -------------------------------------------------

        self.fusion = nn.Conv2d(
            base_channels * 2,
            base_channels,
            1
        )

        # -------------------------------------------------
        # GLOBAL
        # -------------------------------------------------

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.SiLU()
        )

        # -------------------------------------------------
        # CREATIVE HEADS
        # -------------------------------------------------

        self.structure = nn.Linear(base_channels, 1)
        self.texture   = nn.Linear(base_channels, 1)
        self.style     = nn.Linear(base_channels, 1)
        self.chaos     = nn.Linear(base_channels, 1)
        self.rhythm    = nn.Linear(base_channels, 1)

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, x, prompt_emb):

        h = self.encoder(x)

        if prompt_emb.dim() == 3:
            prompt_emb = prompt_emb[:, 0, :]

        p = self.prompt_proj(prompt_emb)

        p = p[:, :, None, None].expand(
            -1,
            -1,
            h.shape[2],
            h.shape[3]
        )

        h = self.fusion(torch.cat([h, p], dim=1))

        g = self.pool(h).squeeze(-1).squeeze(-1)

        g = self.head(g)

        return {
            "structure": self.structure(g),
            "texture": self.texture(g),
            "style": self.style(g),
            "chaos": self.chaos(g),
            "rhythm": self.rhythm(g)
        }


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

creative_model = CreativeDecoratorModel().to(device)

optimizer_creative = optim.AdamW(

    creative_model.parameters(),

    lr=2e-4,

    betas=(0.9, 0.99),

    weight_decay=1e-5
)

criterion_creative = torch.nn.L1Loss()


# =========================================================
# APPLY FUNCTION
# =========================================================
def apply_creative_decoration(
    latents,
    style_prompt_embedding,
    creative_model,
    optimizer=None,
    criterion=None,
    train=False,
    strength=0.12,
    device="cuda",
    frame_counter=0,
    max_epochs_up=8,
    model_path="models/creative_model_latest.pt",
    ema_prev_latents=None,
    ema_alpha=0.3,
    new_image=False,
    debug=False
):

    device = latents.device

    creative_model.to(device)

    x = latents.to(device)
    x0 = x.detach()

    print(f"[Creative V1] device={device}")

    hf = compute_high_freq_energy(x0)

    print(f"[Creative V1] hf={hf.mean().item():.4f}")

    model_exists = os.path.exists(model_path)

    # =====================================================
    # TRAIN
    # =====================================================

    if train and optimizer and criterion:

        creative_model.train()

        max_epochs = max(1, max_epochs_up)

        print("[Creative V1] Training semantic decorator")

        for epoch in range(max_epochs):

            optimizer.zero_grad()

            with torch.enable_grad():

                pred = creative_model(
                    x0,
                    style_prompt_embedding
                )

                structure = pred["structure"].view(-1,1,1,1)
                texture   = pred["texture"].view(-1,1,1,1)
                style     = pred["style"].view(-1,1,1,1)
                chaos     = pred["chaos"].view(-1,1,1,1)
                rhythm    = pred["rhythm"].view(-1,1,1,1)

                out = x0

                # =================================================
                # STRUCTURE
                # =================================================

                low = F.avg_pool2d(out, 5, 1, 2)

                out = out + 0.12 * torch.tanh(structure) * (low - out)

                # =================================================
                # TEXTURE
                # =================================================

                detail = out - F.avg_pool2d(out, 3, 1, 1)

                out = out + 0.18 * torch.tanh(texture) * detail

                # =================================================
                # STYLE
                # =================================================

                style_map = torch.sin(out * 3.1415)

                out = out + 0.06 * torch.tanh(style) * style_map

                # =================================================
                # CHAOS
                # =================================================

                noise = torch.randn_like(out)

                out = out + 0.025 * torch.tanh(chaos) * noise

                # =================================================
                # RHYTHM
                # =================================================

                wave = torch.sin(out * 6.0)

                out = out + 0.04 * torch.tanh(rhythm) * wave

                # =================================================
                # LOSSES
                # =================================================

                # preserve identity
                loss_id = 0.08 * F.l1_loss(out, x0)

                # preserve high frequency structure
                loss_detail = 0.05 * F.l1_loss(
                    compute_high_freq_energy(out),
                    compute_high_freq_energy(x0)
                )

                # avoid latent explosion
                loss_energy = 0.005 * out.pow(2).mean()

                # stylistic coherence
                out_mean = out.mean(dim=(2,3))
                in_mean  = x0.mean(dim=(2,3))

                loss_style = 0.02 * F.l1_loss(
                    out_mean,
                    in_mean
                )

                loss = (
                    loss_id
                    + loss_detail
                    + loss_energy
                    + loss_style
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                creative_model.parameters(),
                1.0
            )

            optimizer.step()

            print(
                f"[Creative V1] "
                f"Epoch {epoch+1}/{max_epochs} | "
                f"Loss={loss.item():.6f}"
            )

            should_save = (
                (frame_counter % 10 == 0)
                and
                (epoch == max_epochs - 1)
            )

            if should_save:

                save_appearance_model(
                    creative_model,
                    optimizer=optimizer,
                    epoch=frame_counter,
                    loss=loss.item(),
                    path=model_path
                )

    # =====================================================
    # LOAD
    # =====================================================

    else:

        creative_model.eval()

        if model_exists:

            creative_model, _ = load_appearance_model(
                type(creative_model),
                path=model_path,
                optimizer=optimizer,
                device=device
            )

    # =====================================================
    # INFERENCE
    # =====================================================

    with torch.no_grad():

        pred = creative_model(
            x,
            style_prompt_embedding
        )

        structure = 0.20 * torch.tanh(pred["structure"])
        texture   = 0.25 * torch.tanh(pred["texture"])
        style     = 0.15 * torch.tanh(pred["style"])
        chaos     = 0.08 * torch.tanh(pred["chaos"])
        rhythm    = 0.10 * torch.tanh(pred["rhythm"])

        print(
            f"[Creative V1] "
            f"struct={structure.mean().item():.4f} | "
            f"texture={texture.mean().item():.4f} | "
            f"style={style.mean().item():.4f} | "
            f"chaos={chaos.mean().item():.4f} | "
            f"rhythm={rhythm.mean().item():.4f}"
        )

        out = x

        # =================================================
        # STRUCTURE
        # =================================================

        low = F.avg_pool2d(out, 5, 1, 2)

        out = out + structure * (low - out)

        # =================================================
        # TEXTURE
        # =================================================

        detail = out - F.avg_pool2d(out, 3, 1, 1)

        out = out + texture * detail

        # =================================================
        # STYLE
        # =================================================

        style_map = torch.sin(out * 3.1415)

        out = out + style * style_map

        # =================================================
        # CHAOS
        # =================================================

        noise = torch.randn_like(out)

        out = out + chaos * noise

        # =================================================
        # RHYTHM
        # =================================================

        wave = torch.sin(out * 6.0)

        out = out + rhythm * wave

    # =====================================================
    # ADAPTIVE STRENGTH
    # =====================================================

    hf = compute_high_freq_energy(x)

    strength_map = strength / (1.0 + 2.0 * hf)

    strength_map = strength_map.clamp(
        0.02,
        strength
    )

    out = x + strength_map * (out - x)

    # =====================================================
    # STABILIZATION
    # =====================================================

    out = stabilize_latents(out)

    # =====================================================
    # EMA TEMPORAL
    # =====================================================

    if ema_prev_latents is not None and not train:

        prev = ema_prev_latents.to(out.device)

        out_low  = F.avg_pool2d(out, 3, 1, 1)
        prev_low = F.avg_pool2d(prev, 3, 1, 1)

        out_high  = out - out_low
        prev_high = prev - prev_low

        motion_factor = float(hf.mean().item())

        alpha_low  = 0.07 + 0.08 * motion_factor
        alpha_high = 0.30 + 0.20 * motion_factor

        alpha_low  = float(torch.clamp(
            torch.tensor(alpha_low),
            0.05,
            0.16
        ))

        alpha_high = float(torch.clamp(
            torch.tensor(alpha_high),
            0.20,
            0.55
        ))

        low_ema = (
            alpha_low * out_low
            +
            (1.0 - alpha_low) * prev_low
        )

        high_ema = (
            alpha_high * out_high
            +
            (1.0 - alpha_high) * prev_high
        )

        out = low_ema + high_ema

    return out
