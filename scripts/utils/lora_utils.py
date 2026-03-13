# ------------------------------------------------------------------
# lora_utils.py - utilitaires robustes pour appliquer LoRA / n3oray
# ------------------------------------------------------------------
import torch
from safetensors.torch import load_file


def detect_unet_type(unet):
    """
    Détecte le type de UNet selon cross_attention_dim
    """

    dim = None

    if hasattr(unet, "config") and hasattr(unet.config, "cross_attention_dim"):
        dim = unet.config.cross_attention_dim

    if dim == 768:
        model_type = "SD1.x compatible"
    elif dim == 1024:
        model_type = "SD2.x compatible"
    elif dim == 2048:
        model_type = "SDXL compatible"
    else:
        model_type = "UNet custom"

    return model_type, dim


def apply_lora(unet, lora_path, alpha=0.8, device=None, verbose=True):
    """
    Applique un modèle LoRA / n3oray sur UNet
    """

    device = device or next(unet.parameters()).device

    # ---------------- Détection UNet ----------------
    model_type, cross_dim = detect_unet_type(unet)

    print("🧠 Détection modèle UNet")
    print(f"   type : {model_type}")
    print(f"   cross_attention_dim : {cross_dim}")

    print(f"📌 Chargement LoRA : {lora_path}")

    # Charger LoRA sur CPU
    lora_state = load_file(lora_path, device="cpu")

    unet_state = dict(unet.named_parameters())

    applied = 0
    skipped = 0
    missing = 0

    for name, lora_param in lora_state.items():

        if name not in unet_state:
            missing += 1
            continue

        param = unet_state[name]

        # vérification dimension
        if param.shape != lora_param.shape:

            if verbose:
                print(
                    f"[LoRA SKIP] {name} "
                    f"{tuple(lora_param.shape)} != {tuple(param.shape)}"
                )

            skipped += 1
            continue

        lora_param = lora_param.to(device=device, dtype=param.dtype)

        # mélange des poids
        param.data.mul_(1 - alpha).add_(lora_param, alpha=alpha)

        applied += 1

    print("✅ LoRA résumé")
    print(f"   couches appliquées : {applied}")
    print(f"   couches ignorées   : {skipped}")
    print(f"   couches absentes   : {missing}")

    return unet


def list_lora_parameters(lora_path):
    """
    Liste les paramètres contenus dans un LoRA
    """

    lora_state = load_file(lora_path, device="cpu")

    print(f"📌 Paramètres dans {lora_path}")

    for k, v in lora_state.items():
        print(f"{k} -> {tuple(v.shape)}")

    return list(lora_state.keys())
