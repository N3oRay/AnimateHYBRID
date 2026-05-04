import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import traceback
from .tools_utils import sanitize_latents_for_train
import matplotlib.pyplot as plt

def show_latents(latents, decoded_latents, epoch):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(latents[0, 0].cpu().detach().numpy(), cmap='gray')
    axes[0].set_title(f"Original Latents - Epoch {epoch}")
    axes[1].imshow(decoded_latents[0, 0].cpu().detach().numpy(), cmap='gray')
    axes[1].set_title(f"Decoded Latents - Epoch {epoch}")
    plt.show()

# Classe du modèle DenoisingAutoencoder
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
denoising_model = DenoisingAutoencoder().to(device)
denoising_model.apply(weights_init)

# Optimiseur et fonction de perte
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-5, weight_decay=1e-5) # apprentissage faible
optimizer = optim.Adam(denoising_model.parameters(), lr=1e-4, weight_decay=1e-5) # apprentissage moyen
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-3, weight_decay=0) # apprentissage elevé
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-3, weight_decay=1e-4)
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.MSELoss()  # Fonction de perte pour la reconstruction d'image
#criterion = nn.L1Loss()  # Essayer la L1Loss
#criterion = nn.SmoothL1Loss()
#criterion = nn.MSELoss()  # Fonction de perte alternative

# Fonction de débruitage et d'entraînement intégré avec contrôle de `Loss requires_grad`
def sanitize_latents_for_train_grad(latents, debug=True):
    # Normalisation simple
    return latents.clamp(-1.0, 1.0)  # renvoie un tensor PyTorch, non détaché



def denoise_latents(latents, denoising_model, optimizer, criterion, device="cuda", train=True):
    """
    Applique un Denoising Autoencoder sur les latents avec possibilité d'entraîner le modèle.
    """
    # Vérification explicite que latents n'est pas None
    if latents is None:
        raise ValueError("Latents is None, cannot proceed with denoising.")

    # Appliquer sanitize_latents pour normaliser les latents    latents = sanitize_latents(latents)

    # Utilisation avant de passer au modèle ou dans le processus d'entraînement
    print_latents(latents, debug=True)
    latents = sanitize_latents_for_train_grad(latents, debug=True)
    print_latents(latents, debug=True)
    latents = latents.to(device=device, dtype=torch.float32)

    # Conversion des latents en float32 avant de les passer au modèle
    latents = latents.to(device=device, dtype=torch.float32).clone()

    # Vérification du type des latents
    print(f"Latents dtype before model: {latents.dtype}")

    latents.requires_grad_()  # Activation explicite des gradients sur les latents
    print(f"Latents requires_grad before model: {latents.requires_grad}")

    if train:
        # Mode entraînement : mettre le modèle en mode entraînement
        denoising_model.train()

        # Assurez-vous que tous les paramètres du modèle ont des gradients
        for name, param in denoising_model.named_parameters():
            param.requires_grad = True
            print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

        # Zero gradients
        optimizer.zero_grad()

        try:
            # Passer les latents dans le Denoising Autoencoder
            decoded_latents = denoising_model(latents)

            # Vérification des gradients de la sortie du modèle
            print(f"Decoded latents shape: {decoded_latents.shape}")
            print(f"Decoded latents requires_grad: {decoded_latents.requires_grad}")

            # Activation explicite des gradients sur la sortie du modèle
            decoded_latents.requires_grad_()  # Forcer les gradients sur decoded_latents
            print(f"Decoded latents requires_grad (after force): {decoded_latents.requires_grad}")

            # Calcul de la perte
            loss = criterion(decoded_latents, latents)  # Comparaison avec l'entrée (image bruitée et image cible)

            # Vérification de la perte
            print(f"Loss requires_grad: {loss.requires_grad}")
            print(f"Loss: {loss.item()}")

            # Forcer les gradients sur la perte si nécessaire
            loss.requires_grad_()  # Activation explicite des gradients sur la perte
            print(f"Loss requires_grad (after force): {loss.requires_grad}")

            # Assurer que la perte a des gradients avant de procéder à la rétropropagation
            if loss.requires_grad:
                print("Loss has gradients, proceeding with backward pass.")
                loss.backward()  # Calcul de la rétropropagation
                # Vérification explicite des gradients dans les paramètres du modèle après la rétropropagation
                for name, param in denoising_model.named_parameters():
                    if param.grad is None:
                        print(f"WARNING: {name} does not have gradients.")
                    else:
                        print(f"Gradient for {name} has shape: {param.grad.shape}")
            else:
                raise ValueError("Loss does not have gradients, cannot perform backward pass.")

            # Mettre à jour les poids
            optimizer.step()

            return decoded_latents, loss.item()  # Retourner les latents débruités et la perte pour suivre l'entraînement

        except Exception as e:
            print(f"Error during training step: {str(e)}")
            print("Stacktrace:")
            traceback.print_exc()  # Afficher la stacktrace pour plus de détails
            return None, None

    else:
        # Mode évaluation, pas de rétropropagation
        denoising_model.eval()
        with torch.no_grad():
            decoded_latents = denoising_model(latents)

        # Assurez-vous que les gradients sont activés avant de retourner les latents décryptés
        decoded_latents.requires_grad_()  # On active les gradients ici aussi si nécessaire

        return decoded_latents


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
