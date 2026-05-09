import torch
import torch.nn as nn
import torch.nn.init as init
import os
import datetime

import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# SAVE / LOAD POUR STYLE INJECTOR
# =========================================================

def save_style_model(
    model,
    optimizer=None,
    epoch=None,
    loss=None,
    path="models/style_injector.pt",
    latest_path="models/style_injector_latest.pt"
):
    """Sauvegarde le modèle StyleInjector avec son optimiseur et métriques."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model.config,
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

    print(f"[INFO] StyleInjector saved -> {path}")


def load_style_model(
    model_class,
    path="models/style_injector_latest.pt",
    optimizer=None,
    device=device
):
    """Charge le modèle StyleInjector à partir d'un checkpoint."""
    if not os.path.exists(path):
        print("[WARN] No checkpoint found.")
        return model_class().to(device), None

    checkpoint = torch.load(path, map_location=device)

    config = checkpoint.get("model_config", {})

    model = model_class(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(
        f"[INFO] Loaded StyleInjector | "
        f"epoch={checkpoint.get('epoch')} | "
        f"loss={checkpoint.get('loss')} | "
        f"time={checkpoint.get('timestamp')}"
    )

    return model, checkpoint

# =========================================================
# WEIGHTS INIT
# =========================================================

def weights_init(m):
    """
    Initialisation des poids pour les couches du réseau.

    Conv2d / ConvTranspose2d : He/Kaiming normal
    Linear : Xavier normal
    Bias : initialisé à 0
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # He/Kaiming normal pour les convolutions
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        # Xavier normal pour les fully connected
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        # BatchNorm : gamma=1, beta=0
        init.ones_(m.weight)
        init.zeros_(m.bias)


def weights_init_v1(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight, nonlinearity="linear")
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)



# =========================================================
# STYLE LOSS
# =========================================================

class StyleLoss(nn.Module):
    """
    Critère pour l'entraînement du StyleInjector.
    Compare les latents injectés aux latents cibles stylisés.
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.l1 = nn.L1Loss(reduction=reduction)
        # Optionnel : tu peux ajouter d'autres composants de perte ici
        # ex: perceptual loss, cosine similarity, etc.

    def forward(self, pred_latents, target_latents):
        """
        Args:
            pred_latents (Tensor): latents stylisés par le modèle
            target_latents (Tensor): latents cibles
        Returns:
            Tensor: valeur de la perte
        """
        loss = self.l1(pred_latents, target_latents)
        return loss

# =========================================================
# RESIDUAL BLOCK
# =========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Bloc résiduel simple pour StyleInjector"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return identity + out

class StyleInjector(nn.Module):
    """
    Injecteur de style léger pour transformer des latents existants.
    """
    def __init__(self, latent_channels=4, hidden=64, num_blocks=4, prompt_dim=768):
        super().__init__()

         # ⚠️ Réintégrer config pour compatibilité sauvegarde
        self.config = {
            "latent_channels": latent_channels,
            "hidden": hidden,
            "num_blocks": num_blocks,
            "prompt_dim": prompt_dim
        }

        self.latent_channels = latent_channels
        self.hidden = hidden
        self.prompt_dim = prompt_dim
        self.num_blocks = num_blocks

        # input_proj sera créé dynamiquement lors du premier forward
        self.input_proj = None

        # Bloc résiduel
        self.resblocks = nn.Sequential(
            *[ResidualBlock(hidden) for _ in range(num_blocks)]
        )

        # Projection du style prompt → hidden
        self.prompt_proj = nn.Linear(prompt_dim, hidden)

        # Projection finale
        self.output_proj = nn.Conv2d(hidden, latent_channels, 3, padding=1)

    def forward(self, latents, style_prompt_embedding):
        """
        Args:
            latents (Tensor): (B, C, H, W)
            style_prompt_embedding (Tensor): (B, prompt_dim)
        Returns:
            Tensor: latents transformés
        """
        B, C, H, W = latents.shape

        # Projection du prompt → style_feat (B, hidden, H, W)
        style_feat = self.prompt_proj(style_prompt_embedding)
        style_feat = style_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)

        # Concaténation latents + style_feat
        x = torch.cat([latents, style_feat], dim=1)

        # Création dynamique de input_proj si nécessaire
        if self.input_proj is None:
            in_channels = x.shape[1]  # latents + style_feat
            self.input_proj = nn.Sequential(
                nn.Conv2d(in_channels, self.hidden, 3, padding=1),
                nn.SiLU()
            ).to(x.device)

        x = self.input_proj(x)
        x = self.resblocks(x)
        delta = self.output_proj(x)

        # Injection résiduelle
        out_latents = latents + delta
        return out_latents



# =========================================================
# SANITIZE LATENTS
# =========================================================

def sanitize_latents(latents):
    latents = latents.float()
    latents = (latents - latents.mean()) / (latents.std() + 1e-6)
    latents = latents.clamp(-4.0, 4.0)
    return latents

# =========================================================
# TRAIN STEP
# =========================================================

def style_train_step(
    model,
    optimizer,
    criterion,
    latents,
    style_prompt_embedding,
    target_latents,
    device=device,
    debug=False
):
    """
    Passe d'entrainement pour le StyleInjector.
    """
    model.train()

    latents = sanitize_latents(latents).to(device)
    target_latents = sanitize_latents(target_latents).to(device)
    style_prompt_embedding = style_prompt_embedding.to(device)

    optimizer.zero_grad()

    pred_latents = model(latents, style_prompt_embedding)

    loss = criterion(pred_latents, target_latents)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if debug:
        print(f"Step loss: {loss.item():.6f}")

    return pred_latents.detach(), loss.item()

# =========================================================
# TRAIN LOOP
# =========================================================

def train_style_model(
    model,
    optimizer,
    criterion,
    style_dataset,
    epochs=10,
    device=device,
    save_every=1
):
    """
    Boucle d'entrainement pour StyleInjector.
    style_dataset doit retourner un dict avec :
        - "latents": Tensor des latents actuels
        - "prompt": Tensor embedding du prompt de style
        - "target": Tensor des latents stylisés cibles
    """
    for epoch in range(epochs):
        epoch_loss = 0.0

        for step, batch in enumerate(style_dataset):
            latents = batch["latents"]
            prompt_embedding = batch["prompt"]
            target_latents = batch["target"]

            _, loss = style_train_step(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                latents=latents,
                style_prompt_embedding=prompt_embedding,
                target_latents=target_latents,
                device=device
            )

            epoch_loss += loss

            print(f"[Epoch {epoch+1}] [Step {step+1}] Loss={loss:.6f}")

        avg_loss = epoch_loss / len(style_dataset)
        print(f"\n[Epoch {epoch+1}] Average Loss={avg_loss:.6f}\n")

        if (epoch + 1) % save_every == 0:
            save_style_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss
            )







# =========================================================
# UTILISATION
# =========================================================

if __name__ == "__main__":
    B, C, H, W = 1, 4, 64, 64
    latent = torch.randn(B, C, H, W)
    prompt_embedding = torch.randn(B, 768)

    style_model = StyleInjector(latent_channels=C).to(latent.device)
    style_model.apply(weights_init)

    out = style_model(latent, prompt_embedding)
    print("Latents après injection de style:", out.shape)
