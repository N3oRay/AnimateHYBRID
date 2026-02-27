# motion_utils.py
import torch
import os
import importlib.util
from pathlib import Path
import safetensors.torch

# -------------------------
# Default motion module
# -------------------------
class DefaultMotionModule(torch.nn.Module):
    def forward(self, latents):
        return latents

default_motion_module = DefaultMotionModule()


# ---------------------------------------------------------
# Diffusion FONCTIONNE PARFAITEMENT
# images_latents: [B,4,T,H,W]
# apply_motion_module = generate_latent
# ---------------------------------------------------------
def apply_motion_module(latents, pos_embeds, neg_embeds, unet, scheduler, motion_module=None, device="cuda", dtype=torch.float16, guidance_scale=7.5, init_image_scale=2.0, seed=42):
    """
    latents: [B,4,T,H,W] (déjà encodés et scalés) init_image_scale: poids de l'image initiale
    """
    torch.manual_seed(seed)
    B, C, T, H, W = latents.shape
    latents = latents.to(device=device, dtype=dtype)
    latents = latents.permute(0,2,1,3,4).reshape(B*T, C, H, W).contiguous()
    # ⚡ on garde une copie des latents initiaux
    init_latents = latents.clone()
    for t in scheduler.timesteps:
        if motion_module is not None:
            latents = motion_module(latents)

        # classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        embeds = torch.cat([neg_embeds, pos_embeds])

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # ⚡ appliquer init_image_scale pour garder l’influence de l’image initiale
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents + init_image_scale * (init_latents - latents)

    latents = latents.reshape(B, T, C, H, W).permute(0,2,1,3,4).contiguous()

    return latents

#---------------------------------------------------------
# -------------------------
# Génération de latents par bloc OK
# def generate_latents(latents, pos_embeds, neg_embeds, unet, scheduler, motion_module=None, device="cuda", dtype=torch.float16, guidance_scale=7.5, init_image_scale=2.0, seed=42,
# -------------------------
def generate_latents_1(latents, pos_embeds, neg_embeds, unet, scheduler, motion_module=None, device="cuda", dtype=torch.float16, guidance_scale=4.5, init_image_scale=0.85):
    """
    latents: [B, C, F, H, W]
    pos_embeds / neg_embeds: [B, L, D]
    """
    """
    latents: [B,4,T,H,W] (déjà encodés et scalés) init_image_scale: poids de l'image initiale
    """
    torch.manual_seed(42)
    B, C, T, H, W = latents.shape
    latents = latents.to(device=device, dtype=dtype)
    latents = latents.permute(0,2,1,3,4).reshape(B*T, C, H, W).contiguous()
    # ⚡ on garde une copie des latents initiaux
    init_latents = latents.clone()
    for t in scheduler.timesteps:
        if motion_module is not None:
            latents = motion_module(latents)

        # classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        embeds = torch.cat([neg_embeds, pos_embeds])

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # ⚡ appliquer init_image_scale pour garder l’influence de l’image initiale
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents + init_image_scale * (init_latents - latents)

    latents = latents.reshape(B, T, C, H, W).permute(0,2,1,3,4).contiguous()

    return latents

# -------------------------
# Load motion module
# -------------------------
def load_motion_module(module_path: str, device: str = "cuda", fp16: bool = True):
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Motion module not found: {module_path}")

    dtype = torch.float16 if fp16 else torch.float32

    if module_path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("motion_module", module_path)
        mm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mm)

        # Cherche une classe MotionModule dans le module
        cls_candidates = [v for v in mm.__dict__.values() if isinstance(v, type) and issubclass(v, torch.nn.Module)]
        if len(cls_candidates) == 0:
            raise ValueError(f"No nn.Module subclass found in {module_path}")
        motion_module = cls_candidates[0]()   # instancier
        motion_module.to(device=device, dtype=dtype)
        motion_module.eval()
        print(f"✅ Motion module (Python) loaded and instantiated: {module_path}")
        return motion_module

    elif module_path.endswith(".ckpt") or module_path.endswith(".safetensors"):
        try:
            state_dict = torch.load(module_path, map_location="cpu")
        except Exception:
            state_dict = safetensors.torch.load_file(module_path, device="cpu")

        class MotionModule(torch.nn.Module):
            def __init__(self, sd):
                super().__init__()
                self.sd = sd
            def forward(self, x):
                return x

        motion_module = MotionModule(state_dict)
        motion_module.to(device=device, dtype=dtype)
        motion_module.eval()
        print(f"✅ Motion module (checkpoint) loaded: {module_path}")
        return motion_module

    else:
        raise ValueError("Unsupported motion module file type: must be .py, .ckpt, or .safetensors")
