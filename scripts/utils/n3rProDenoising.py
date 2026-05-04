import torch
import torch.nn as nn
import torch.nn.init as init

# TEST création de model !
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),  # Accepter 4 canaux d'entrée
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )

        # Décodeur
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4 canaux en sortie
            nn.Sigmoid(),  # Sortie entre 0 et 1 (pour des images normalisées entre [0,1])
        )

    def forward(self, x):
        # Passer par l'encodeur et le décodeur
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialisation des poids
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

# Créer le modèle de débruitage avec 3 canaux en entrée
denoising_model = DenoisingAutoencoder().to(device="cuda")
# Appliquer l'initialisation des poids
denoising_model.apply(weights_init)


def denoise_latents(latents, denoising_model, device="cuda"):
    """
    Applique un Denoising Autoencoder aux latents.
    """
    latents = latents.to(device).to(torch.float32)  # Convertir en float32 avant de passer au modèle

    # Passer les latents dans le Denoising Autoencoder
    latents = denoising_model(latents)
    return latents

