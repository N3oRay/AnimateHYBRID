# ------------------------------------------------------------------
# lora_utils.py - utilitaires pour appliquer LoRA / n3oray sur UNet
# ------------------------------------------------------------------
import torch
from safetensors.torch import load_file


def apply_lora(unet, lora_path, alpha=0.8, device=None):
    """
    Applique un modèle LoRA/n3oray sur l'UNet.
    """
    device = device or next(unet.parameters()).device
    print(f"📌 Chargement LoRA depuis {lora_path} sur {device}")

    # ⚡ Charger d'abord sur CPU
    lora_state = load_file(lora_path, device="cpu")

    # Déplacer sur le device souhaité et appliquer alpha
    for name, param in unet.named_parameters():
        if name in lora_state:
            lora_param = lora_state[name].to(device=device, dtype=param.dtype)
            param.data = param.data * (1 - alpha) + lora_param * alpha

    print(f"✅ LoRA appliqué avec alpha={alpha}")
    return unet


def list_lora_parameters(lora_path):
    """
    Liste les paramètres disponibles dans le fichier LoRA/n3oray.
    Utile pour debug.
    """
    from safetensors.torch import load_file
    lora_state = load_file(lora_path, device="cpu")
    print(f"📌 Paramètres dans {lora_path}:")
    for k in lora_state.keys():
        print(" -", k)
    return list(lora_state.keys())
