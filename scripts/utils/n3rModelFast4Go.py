import torch
import torch.nn as nn

class N3RModelFast4GB(nn.Module):
    def __init__(self, L=4, N_samples=8, tile_size=64, cpu_offload=False):
        """
        L         : nombre de fréquences pour le positional encoding
        N_samples : échantillons par pixel
        tile_size : taille des tiles pour le forward
        cpu_offload: True pour décharger les tiles sur CPU pour VRAM limitée
        """
        super().__init__()
        self.L = L
        self.N_samples = N_samples
        self.tile_size = tile_size
        self.cpu_offload = cpu_offload

        # Dimensions d'entrée : 3 coords + 2*3*L (sin/cos)
        input_dim = 3 + 2 * 3 * self.L
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # RGB + density
        ).half()  # FP16

    def positional_encoding(self, x):
        """
        Encode input coordinates with reduced frequencies
        x: (N,3) en FP16
        """
        x = x.half()
        enc = [x]
        for i in range(self.L):
            for fn in [torch.sin, torch.cos]:
                enc.append(fn((2.0 ** i) * x))
        return torch.cat(enc, dim=-1)

    def forward_tile(self, coords):
        """
        Forward pass pour un tile
        coords: (N_tile, 3) en FP16
        """
        x_enc = self.positional_encoding(coords)
        out = self.mlp(x_enc)
        return out

    def forward(self, coords, H, W):
        """
        coords : tensor (H*W*N_samples, 3)
        H, W   : dimensions de l'image
        """
        device = coords.device
        output = torch.zeros((H*W*self.N_samples, 4), device=device, dtype=torch.float16)

        ts = self.tile_size
        for y in range(0, H, ts):
            for x in range(0, W, ts):
                y_end = min(y + ts, H)
                x_end = min(x + ts, W)

                # Meshgrid pour générer les indices
                y_range = torch.arange(y, y_end, device=device)
                x_range = torch.arange(x, x_end, device=device)
                s_range = torch.arange(self.N_samples, device=device)
                yy, xx, s = torch.meshgrid(y_range, x_range, s_range, indexing='ij')
                idx_tile = (yy * W * self.N_samples + xx * self.N_samples + s).reshape(-1)

                coords_tile = coords[idx_tile]
                output_tile = self.forward_tile(coords_tile)

                # Option CPU offload pour VRAM limitée
                if self.cpu_offload:
                    output[idx_tile.cpu()] = output_tile.cpu()
                else:
                    output[idx_tile] = output_tile

        return output
