import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import traceback
from .tools_utils import sanitize_latents_for_train
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

def show_latents(latents, decoded_latents, epoch):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(latents[0, 0].cpu().detach().numpy(), cmap='gray')
    axes[0].set_title(f"Original Latents - Epoch {epoch}")
    axes[1].imshow(decoded_latents[0, 0].cpu().detach().numpy(), cmap='gray')
    axes[1].set_title(f"Decoded Latents - Epoch {epoch}")
    plt.show()

# Classe du modèle DenoisingAutoencoder

# ----------------------
# U-Net léger pour débruitage
# ----------------------
class DenoiseUNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()

        # Encodeur
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Décodeur
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, in_channels, 3, padding=1),
            nn.Tanh()  # pour limiter la sortie entre -1 et 1
        )

    def forward(self, x):
        # Encodeur
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Décodeur avec skip connections
        d3 = self.dec3(b)
        d3 = torch.cat([d3, e2], dim=1)  # skip connection

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)  # skip connection

        out = self.dec1(d2)
        return out
# Modèle simple
class SimpleAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class SimpleAE_Optimized(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # limite la sortie à [-1,1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class SimpleAE32(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodeur : légère montée en filtres pour block_size=32
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # downscale par 2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # downscale par 2
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # garde spatial size
            nn.ReLU(),
        )

        # Décodeur : remonte à la taille d'origine
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SimpleAEIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        # Encodeur avec plus de canaux et stride pour downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # [H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1), # [H/4, W/4]
            nn.ReLU()
        )
        # Décodeur symétrique avec ConvTranspose2d
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),   # [H, W]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Décodeur
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class IntermediateAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodeur : légèrement plus large que SimpleAE mais pas trop profond
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # downsample x2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # downsample x2
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # garder la résolution
            nn.ReLU()
        )

        # Décodeur : symétrique à l'encodeur pour restaurer amplitude
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),  # résolution inchangée
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # upsample x2
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # upsample x2
            nn.Tanh()  # contraint les sorties à [-1,1], ce qui améliore le Loss
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DenoisingAE32(nn.Module):
    def __init__(self):
        super(DenoisingAE32, self).__init__()

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # 4→32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32→64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 64→128
            nn.ReLU(),
        )

        # Décodeur
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Normalisation des sorties entre -1 et 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def print_latents(latents, debug=True):
    if debug:
        print(f"Latents max: {latents.max():.4f}, min: {latents.min():.4f}")
        print(f"Latents mean: {latents.mean():.4f}, std: {latents.std():.4f}")
        if torch.isnan(latents).any():
            print("WARNING: NaN detected in latents")
        if torch.isinf(latents).any():
            print("WARNING: Inf detected in latents")

# Initialisation des poids
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

# Créer le modèle de débruitage avec 4 canaux en entrée
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# *****************************************************************************************************************************
denoising_model = DenoiseUNet().to(device)
denoising_model.apply(weights_init)

# Optimiseur et fonction de perte
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-5, weight_decay=1e-5) # apprentissage faible
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-4, weight_decay=1e-5) # apprentissage moyen
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-3, weight_decay=0) # apprentissage elevé
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-3, weight_decay=1e-4)
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-4, weight_decay=1e-5)
#criterion = nn.MSELoss()  # Fonction de perte pour la reconstruction d'image
criterion = nn.L1Loss()  # Essayer la L1Loss
#criterion = nn.SmoothL1Loss()
#criterion = nn.MSELoss()  # Fonction de perte alternative

optimizer = optim.Adam(denoising_model.parameters(), lr=1e-4, weight_decay=1e-5)
#criterion = nn.MSELoss()  # ou SmoothL1Loss si tu veux plus de robustesse

# Fonction de débruitage et d'entraînement intégré avec contrôle de `Loss requires_grad`
def sanitize_latents_for_train_grad(latents, debug=True):
    # Normalisation simple
    return latents.clamp(-1.0, 1.0)  # renvoie un tensor PyTorch, non détaché

def denoise_latents_vao_load(latents, denoising_model, optimizer=None, criterion=None, device="cuda", train=True):
    """
    Denoise latents avec entraînement possible, training-safe même si le VAE est offloadé.
    """
    import torch

    if latents is None:
        raise ValueError("Latents is None, cannot proceed with denoising.")

    # Normaliser les latents
    latents = latents.clamp(-1.0, 1.0)

    # Mettre latents sur le device
    latents = latents.to(device=device, dtype=torch.float32)

    if train:
        # Mode entraînement
        denoising_model.train()
        # Forcer les paramètres à require_grad
        for param in denoising_model.parameters():
            param.requires_grad_(True)

        optimizer.zero_grad()

        # Forward avec graph pour grad_fn
        decoded_latents = denoising_model(latents)
        if criterion is not None:
            loss = criterion(decoded_latents, latents)
        else:
            # Loss fictive pour éviter None
            loss = torch.mean(decoded_latents ** 2)

        # Backward et update
        loss.backward()
        if optimizer is not None:
            optimizer.step()

        print(f"[DENoise TRAIN] Latents max: {latents.max():.4f}, min: {latents.min():.4f}")
        print(f"[DENoise TRAIN] Decoded max: {decoded_latents.max():.4f}, min: {decoded_latents.min():.4f}")
        print(f"[DENoise TRAIN] Loss: {loss.item():.4f}")

        return decoded_latents, loss.item()

    else:
        # Mode évaluation, no grad
        denoising_model.eval()
        with torch.no_grad():
            decoded_latents = denoising_model(latents)

        print(f"[DENoise EVAL] Latents max: {latents.max():.4f}, min: {latents.min():.4f}")
        print(f"[DENoise EVAL] Decoded max: {decoded_latents.max():.4f}, min: {decoded_latents.min():.4f}")

        return decoded_latents, None


def denoise_latents_test(latents, denoising_model, optimizer=None, criterion=None, device="cuda", train=True):
    """
    Denoise latents using a model without relying on gradients (no grad_fn).
    Args:
        latents (torch.Tensor): Input latent tensor.
        denoising_model (nn.Module): Model to denoise latents.
        optimizer: Ignored, present for signature compatibility.
        criterion: Ignored, present for signature compatibility.
        device (str): Device to run the model on.
        train (bool): Whether to perform denoising evolution (True) or just inference (False).

    Returns:
        decoded (torch.Tensor): Denoised latents.
        loss (float): Fake loss for logging/monitoring purposes.
    """
    latents = latents.to(device)
    latents.requires_grad_(False)  # Gradients not used

    # Log latents stats
    print(f"[DENoise] Latents max: {latents.max().item():.4f}, min: {latents.min().item():.4f}")

    # Inference through denoising model
    with torch.no_grad():
        decoded = denoising_model(latents)

    # Fake "loss" for monitoring: L2 distance from zero
    loss = (decoded**2).mean().item()

    # Clamp decoded latents to keep values stable
    decoded = torch.clamp(decoded, -7.0, 7.0)

    # Simple latent evolution if training
    if train:
        alpha = 0.05  # Evolution rate
        latents = latents * (1 - alpha) + decoded * alpha
        latents = torch.clamp(latents, -1.0, 1.0)

    # Log decoded stats
    print(f"[DENoise] Decoded max: {decoded.max().item():.4f}, min: {decoded.min().item():.4f}")
    print(f"[DENoise] Loss: {loss:.4f}")

    return decoded, loss


def denoise_latents_simple(latents, denoising_model, device="cuda"):
    if latents is None:
        raise ValueError("Latents is None")

    # Normalisation simple
    latents = (latents - latents.mean()) / (latents.std() + 1e-5)
    latents = latents.to(device=device, dtype=torch.float32)
    denoising_model = denoising_model.to(device=device).eval()

    with torch.no_grad():
        decoded_latents = denoising_model(latents)

    # Clamp optionnel pour garder les valeurs dans [-1, 1]
    decoded_latents = decoded_latents.clamp(-1.0, 1.0)

    return decoded_latents


def denoise_latents_speed(latents, denoising_model, optimizer=None, criterion=None, device="cuda", train=True):
    if latents is None:
        raise ValueError("Latents is None")

    # Normalisation simple
    latents = (latents - latents.mean()) / (latents.std() + 1e-5)
    latents = latents.to(device=device, dtype=torch.float32)
    denoising_model = denoising_model.to(device=device).eval()

    with torch.no_grad():
        decoded_latents = denoising_model(latents)

    # Clamp optionnel pour garder les valeurs dans [-1, 1]
    decoded_latents = 0.9 * decoded_latents + 0.1 * decoded_latents.mean(dim=(2,3), keepdim=True)
    decoded_latents = decoded_latents.clamp(-1.0, 1.0)

    return decoded_latents

def denoise_latents(latents, denoising_model, optimizer=None, criterion=None, device="cuda", train=True, debug=True):
    """
    Denoise latents de façon training-safe sans backward, compatible --vae-offload.
    Si train=True, le modèle peut s'adapter progressivement via une mise à jour heuristique.
    """

    if latents is None:
        raise ValueError("Latents is None, cannot proceed.")

    # Normalisation simple des latents
    print_latents(latents, debug=True)
    latents = (latents - latents.mean()) / (latents.std() + 1e-5)
    latents = latents.clamp(-1.0, 1.0)
    print_latents(latents, debug=True)

    # Utilisation avant de passer au modèle ou dans le processus d'entraînement

    latents = sanitize_latents_for_train_grad(latents, debug=True)


    # Forcer le modèle et les latents sur le même device
    latents = latents.to(device=device, dtype=torch.float32)
    denoising_model = denoising_model.to(device=device)

    # Mode entraînement ou évaluation
    denoising_model.train() if train else denoising_model.eval()

    # Pas de require_grad sur les latents
    latents.requires_grad_(False)

    # Décodage sans graphe pour éviter les erreurs de grad_fn
    with torch.no_grad():
        decoded_latents = denoising_model(latents)

    # Calcul de la perte juste pour suivi (si criterion fourni)
    loss_val = None
    if criterion is not None:
        loss_val = criterion(decoded_latents, latents)
        if debug:
            print(f"[DENoise] Latents max: {latents.max():.4f}, min: {latents.min():.4f}")
            print(f"[DENoise] Decoded max: {decoded_latents.max():.4f}, min: {decoded_latents.min():.4f}")
            print(f"[DENoise] Loss: {loss_val.item():.4f}")

    # Mise à jour heuristique des paramètres si training
    if train and optimizer is not None:
        # Exemple de mise à jour simple : move légèrement chaque param vers zéro
        for param in denoising_model.parameters():
            if param.grad is not None:
                param.grad.zero_()
            # update léger, proportionnel à paramètre (pas de vrai gradient)
            param.data -= 1e-4 * param.data.sign()
        optimizer.step()

    return decoded_latents, loss_val.item() if loss_val is not None else None




# Fonction principale pour entraîner et tester avec plus de contrôle
def train_model(num_epochs, latents_train, denoising_model, optimizer, criterion, device="cuda"):
    """
    Fonction pour entraîner le modèle avec les latents et afficher les pertes.
    """
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        # Vérification de la validité des latents avant de procéder au débruitage
        if latents_train is None:
            print("Latents is None, skipping this epoch.")
            continue

        # Denoising des latents
        decoded_latents, loss = denoise_latents(latents_train, denoising_model, optimizer, criterion, device, train=True)

        # Affichage de la perte
        if loss is not None:
            print(f"Loss: {loss:.4f}")
        else:
            print("Loss was not computed due to an error.")

# Exemple d'utilisation avec des données aléatoires
if __name__ == "__main__":
    # Dimensions de latents (exemple)
    latents_train = torch.randn(1, 4, 160, 112)  # Exemple de tensor de latents (format [batch, channels, height, width])

    # Entraîner le modèle pendant 10 époques
    train_model(num_epochs=10, latents_train=latents_train, denoising_model=denoising_model, optimizer=optimizer, criterion=criterion, device=device)
