# n3r_utils.py
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from torchvision.utils import save_image
import ffmpeg

import torch.nn as nn
import math
from tqdm import tqdm
from diffusers import UNet2DConditionModel, AutoencoderKL, DPMSolverMultistepScheduler
import os
import numpy as np
import yaml
import torch.nn.functional as F
LATENT_SCALE = 0.18215

import torchvision.transforms as T
from transformers import CLIPTokenizer, CLIPTextModel


import torch, math

def generate_latents_ai_5D_stable(
    latent_frame,         # [B,4,H,W] ou [1,4,H,W]
    pos_embeds,           # [1,77,768]
    neg_embeds,           # [1,77,768]
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=7.5,
    init_image_scale=0.7,
    creative_noise=0.03,
    seed=42,
    steps=40
):
    """
    Génération de latents ultra-stable avec :
      - réinjection du latent initial pour stabilité
      - creative_noise modéré
      - support batch [B,4,H,W] ou frame unique [1,4,H,W]
    Sortie : latents [B,4,H,W]
    """

    torch.manual_seed(seed)
    B = latent_frame.shape[0]

    latents = latent_frame.to(device=device, dtype=dtype)
    init_latents = latents.clone()

    scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:
        # 🔹 Motion module
        if motion_module is not None:
            latents = motion_module(latents)

        # 🔹 Creative noise
        if creative_noise > 0:
            latents = latents + torch.randn_like(latents) * creative_noise

        # 🔹 Classifier-Free Guidance
        latent_model_input = torch.cat([latents, latents], dim=0)
        embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # 🔹 Scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 🔹 Réinjection du latent initial (stabilité couleur / contenu)
        latents = latents + init_image_scale * (init_latents - latents)

        # 🔹 Sécurité NaN / inf
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            latents = latents + torch.randn_like(latents) * 1e-3

        # 🔹 Debug
        mean_val = latents.abs().mean().item()
        if mean_val < 1e-5:
            print(f"⚠ Latent trop petit à timestep {t}, mean={mean_val:.6f}")

    return latents



def generate_latents_ai_5D_optimized(
    latent_frame,         # [1,4,H,W]
    pos_embeds,           # [1,77,768]
    neg_embeds,           # [1,77,768]
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=7.5,          # aligné robuste
    init_image_scale=2.0,        # aligné robuste
    creative_noise=0.0,
    seed=42,
    steps=40
):
    """
    Version équivalente à generate_latents_robuste mais pour une seule frame.
    Sortie: [1,4,H,W]
    """

    torch.manual_seed(seed)

    # ---- Setup ----
    latents = latent_frame.to(device=device, dtype=dtype)
    init_latents = latents.clone()

    scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:

        # 🔹 Motion module
        if motion_module is not None:
            latents = motion_module(latents)

        # 🔹 Creative noise (même endroit que robuste)
        if creative_noise > 0:
            latents = latents + torch.randn_like(latents) * creative_noise

        # 🔹 Classifier-Free Guidance
        latent_model_input = torch.cat([latents, latents], dim=0)
        embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # 🔹 Scheduler step (IMPORTANT: batch normal, pas latents[:1])
        latents = scheduler.step(
            noise_pred,
            t,
            latents
        ).prev_sample

        # 🔹 Réinjection identique à robuste
        latents = latents + init_image_scale * (init_latents - latents)

        # 🔹 Sécurité NaN / inf
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            print(f"⚠ NaN/inf détecté à timestep {t}, correction légère")
            latents = latents.clone()
            latents = latents + torch.randn_like(latents) * 1e-3

        # 🔹 Debug stabilité
        mean_val = latents.abs().mean().item()
        if math.isnan(mean_val) or mean_val < 1e-5:
            print(f"⚠ Latent trop petit à timestep {t}, mean={mean_val:.6f}")

    return latents



# -------------------------
# 🔹 Génération des latents
# -------------------------

def generate_latents_4Go(
    latent_frame,          # [1,4,H,W]
    pos_embeds,            # [1,77,768]
    neg_embeds,            # [1,77,768]
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.0,
    init_image_scale=0.9,
    steps=20,
    seed=1234
):
    """
    Génération de latents optimisée pour AnimateDiff avec Classifier-Free Guidance.
    Corrige le mismatch batch entre UNet et embeddings.
    """
    torch.manual_seed(seed)

    # 🔹 Mettre le latent sur le device
    latents = latent_frame.to(device=device, dtype=dtype)

    batch_size = latents.shape[0]  # normalement 1

    # 🔹 Répliquer embeddings selon batch size
    neg_embeds = neg_embeds.repeat(batch_size, 1, 1)
    pos_embeds = pos_embeds.repeat(batch_size, 1, 1)

    # 🔹 Embeddings C.F.G
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)
    # batch = 2 * batch_size

    # 🔹 Scheduler
    scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:

        # 🔹 Input latent doublé pour C.F.G
        latent_input = torch.cat([latents, latents], dim=0)

        # 🔹 Motion module si utilisé
        if motion_module is not None:
            latent_input = motion_module(latent_input)

        # 🔹 Blend avec latent original
        latent_input = latent_input * init_image_scale + torch.cat([latents, latents], dim=0) * (1 - init_image_scale)

        # 🔹 UNet forward
        noise_pred = unet(latent_input, t, encoder_hidden_states=embeds).sample

        # 🔹 Classifier-Free Guidance
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # 🔹 Scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 🔹 Clamp sécurité fp16
        latents = latents.clamp(-1.5, 1.5)

    return latents

# -------------------------
# 🔹 Génération des texts
# -------------------------

def encode_text_embeddings(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    negative_prompt: str = "",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    max_length: int = 77
):
    """
    Encode le texte en embeddings pour Classifier-Free Guidance.

    Args:
        tokenizer (CLIPTokenizer): tokenizer du modèle.
        text_encoder (CLIPTextModel): text encoder du modèle.
        prompt (str): texte positif.
        negative_prompt (str): texte négatif (CFG).
        device (str): "cuda" ou "cpu".
        dtype (torch.dtype): torch.float16 ou torch.float32.
        max_length (int): longueur max du tokenizer (77 pour SD).

    Returns:
        tuple: (pos_embeds, neg_embeds) chacun [1, max_length, 768]
    """
    # 🔹 Tokenisation texte positif
    pos_tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    # 🔹 Tokenisation texte négatif
    neg_tokens = tokenizer(
        negative_prompt if negative_prompt else "",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        # 🔹 Encodage en embeddings
        pos_embeds = text_encoder(**pos_tokens).last_hidden_state.to(dtype)
        neg_embeds = text_encoder(**neg_tokens).last_hidden_state.to(dtype)

    return pos_embeds, neg_embeds

def load_image_latent(image_path, vae, device="cuda", dtype=torch.float16, target_size=128):
    """
    Charge une image et la convertit en latent tensor pour le UNet.

    Args:
        image_path (str): Chemin vers l'image.
        vae (AutoencoderKL): VAE du modèle.
        device (str): "cuda" ou "cpu".
        dtype (torch.dtype): torch.float16 ou torch.float32.
        target_size (int): taille de l'image (carrée) pour le modèle.

    Returns:
        torch.Tensor: latent tensor [1, 4, H/8, W/8]
    """
    # 1️⃣ Charger et redimensionner
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device=device, dtype=dtype)  # [1,3,H,W]

    # 2️⃣ Normalisation pour VAE
    img_tensor = img_tensor * 2.0 - 1.0  # [-1,1]

    # 3️⃣ Encoder via VAE (no_grad pour économiser mémoire)
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()  # [1,4,H/8,W/8]

    return latent


# ------------------------------
# 🔹 Fonction latents optimisée
# ------------------------------

def generate_latents_ai_5D_light(
    latent_frame,
    pos_embeds,
    neg_embeds,
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.0,
    init_image_scale=0.9,
    creative_noise=0.0,
    seed=1234,
    steps=25
):
    torch.manual_seed(seed)

    latent_frame = latent_frame.to(device=device, dtype=dtype)
    latents = latent_frame.repeat(2,1,1,1)  # batch=2 pour CFG une seule fois

    if creative_noise > 0.0:
        latents += torch.randn_like(latents) * creative_noise

    # embeddings
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:
        latent_input = latents

        if motion_module:
            latent_input = motion_module(latent_input)

        # blend avec image initiale
        latent_input = latent_input * init_image_scale + latent_frame.repeat(2,1,1,1) * (1 - init_image_scale)

        # UNet forward
        noise_pred = unet(latent_input, t, encoder_hidden_states=embeds).sample

        # guidance
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # scheduler step
        latents_step = scheduler.step(noise_pred, t, latents[:1]).prev_sample

        # clamp sécurité
        latents_step = latents_step.clamp(-2.0, 2.0)

        # préparer pour prochaine étape (reduplication batch=2)
        latents = torch.cat([latents_step, latents_step], dim=0)

    return latents[:1]

# ------------------------------------------------------------
# generate_latents_ai_5D_optimized_test - Anomalie sortie Frame Noir
# ------------------------------------------------------------

def generate_latents_ai_5D_optimized_test(
    latent_frame,         # [1,4,H,W]
    pos_embeds,           # [1,77,768]
    neg_embeds,           # [1,77,768]
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.0,
    init_image_scale=0.9,
    creative_noise=0.0,
    seed=1234,
    steps=40
):

    torch.manual_seed(seed)

    latent_frame = latent_frame.to(device=device, dtype=dtype)

    if creative_noise > 0.0:
        latent_frame = latent_frame + torch.randn_like(latent_frame) * creative_noise

    # 🔹 Classifier-Free Guidance embeddings
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)
    # shape = [2,77,768]

    # 🔹 Duplicate latent for CFG
    latents = latent_frame.repeat(2, 1, 1, 1)  # [2,4,H,W]

    scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:

        latent_input = latents

        # 🔹 Motion module (si utilisé)
        if motion_module is not None:
            latent_input = motion_module(latent_input)

        # 🔹 Blend avec image initiale (stabilité)
        latent_input = (
            latent_input * init_image_scale +
            latent_frame.repeat(2,1,1,1) * (1 - init_image_scale)
        )

        # 🔹 UNet forward
        noise_pred = unet(
            latent_input,
            t,
            encoder_hidden_states=embeds
        ).sample

        # 🔹 Guidance
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # 🔹 Scheduler step
        latents = scheduler.step(
            noise_pred,
            t,
            latents[:1]  # on garde batch=1 pour le step
        ).prev_sample

        # 🔹 Clamp sécurité fp16
        latents = latents.clamp(-1.5, 1.5)

        # 🔹 Re-dupliquer pour prochaine itération
        latents = latents.repeat(2,1,1,1)

    # Retour final batch=1
    return latents[:1]


def generate_latents_ai_5D_batch(
    latent_frames,       # [B,C,T,H,W] tensor
    pos_embeds,          # embeddings positifs
    neg_embeds,          # embeddings négatifs
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.5,
    init_image_scale=0.85,
    creative_noise=0.0,
    seed=42,
    steps=35
):
    """
    Génération de latents animés pour plusieurs images / frames.

    latent_frames : [B,C,T,H,W]
    pos_embeds, neg_embeds : embeddings texte
    unet : modèle UNet
    scheduler : scheduler de diffusion
    motion_module : module de mouvement optionnel
    """

    torch.manual_seed(seed)

    B, C, T, H, W = latent_frames.shape
    latent_frames = latent_frames.to(device=device, dtype=dtype)

    # Ajouter bruit créatif si besoin
    if creative_noise > 0.0:
        noise = torch.randn_like(latent_frames) * creative_noise
        latent_frames = latent_frames + noise

    # Wrapper motion module
    if motion_module:
        def motion_wrapper(latents, timestep=None):
            try:
                return motion_module(latents)
            except TypeError:
                return motion_module(latents)
    else:
        motion_wrapper = lambda x, t=None: x

    # Scheduler timesteps
    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps

    # Embeddings concaténés
    total_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
    total_batch = total_embeds.shape[0]

    # Boucle par frame
    for t_idx, t in enumerate(timesteps):
        # Itérer sur chaque timestep
        # latent_frames = [B,C,T,H,W] → on traite chaque frame
        for f_idx in range(T):
            latent_frame = latent_frames[:, :, f_idx, :, :]  # [B,C,H,W]

            # Motion module
            latent_frame = motion_wrapper(latent_frame, t)

            # Préparer batch pour guidance
            latent_input = latent_frame.repeat(total_batch, 1, 1, 1)
            latent_input = latent_input * init_image_scale + latent_frame.repeat(total_batch,1,1,1) * (1-init_image_scale)

            # UNet forward
            noise_pred = unet(latent_input, t, encoder_hidden_states=total_embeds).sample

            # Guidance scale
            if total_batch > 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            step_output = scheduler.step(noise_pred, t, latent_frame)
            latent_frame = step_output.prev_sample

            # Clamp sécurité fp16
            latent_frame = latent_frame.clamp(-1.5, 1.5)

            # Remettre la frame à sa place
            latent_frames[:, :, f_idx, :, :] = latent_frame

    return latent_frames

def generate_latents_ai_5D_optimized1(
    latent_frame,         # [B,C,H,W] tensor
    pos_embeds,           # embeddings positifs
    neg_embeds,           # embeddings négatifs
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.5,
    init_image_scale=0.85,
    creative_noise=0.0,
    seed=42,
    steps=35
):
    """
    Génération de latents animés 5D (B,C,T,H,W)
    - latent_frame : [B,C,H,W]
    - pos_embeds, neg_embeds : embeddings texte
    - unet : UNet
    - scheduler : scheduler de diffusion (DDIM/LMS)
    - motion_module : module de mouvement optionnel
    """
    torch.manual_seed(seed)

    # Assurer que le latent est sur le bon device
    latent_frame = latent_frame.to(device=device, dtype=dtype)

    # Ajouter bruit créatif
    if creative_noise > 0.0:
        noise = torch.randn_like(latent_frame) * creative_noise
        latent_frame = latent_frame + noise

    # Wrapper motion module
    if motion_module:
        def motion_wrapper(latents, timestep=None):
            try:
                return motion_module(latents)
            except TypeError:
                return motion_module(latents)
    else:
        motion_wrapper = lambda x, t=None: x

    # Scheduler timesteps
    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps

    # Préparer batch pour guidance
    total_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
    batch_size = total_embeds.shape[0]

    # Répéter le latent pour correspondre au batch
    latent_model_input = latent_frame.repeat(batch_size, 1, 1, 1)

    # Boucle diffusion
    for i, t in enumerate(timesteps):
        latent_model_input = latent_model_input.to(device=device, dtype=dtype)

        # Motion module
        latent_model_input = motion_wrapper(latent_model_input, t)

        # Guidance input
        latent_model_input_in = latent_model_input.clone()
        latent_model_input_in = latent_model_input_in * init_image_scale + latent_frame.repeat(batch_size,1,1,1) * (1-init_image_scale)

        # UNet forward
        noise_pred = unet(latent_model_input_in, t, encoder_hidden_states=total_embeds).sample

        # Guidance scale
        if batch_size > 1:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = noise_pred

        # Scheduler step
        step_output = scheduler.step(noise_pred, t, latent_model_input)
        latent_model_input = step_output.prev_sample

        # Clamp pour sécurité fp16
        latent_model_input = latent_model_input.clamp(-1.5, 1.5)

    return latent_model_input


# -------------------------
# generate_latents_ai_5D_optimized (fixed)
# -------------------------


def generate_latents_ai_5D_fixed(
    latent_frame,
    pos_embeds,
    neg_embeds,
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.5,
    init_image_scale=0.85,
    creative_noise=0.0,
    seed=42,
    steps=35,
    clamp_max=1.0
):
    """
    Génération de latents animés 5D (B,C,H,W)
    """

    torch.manual_seed(seed)

    latent_model_input = latent_frame.clone()

    # Bruit créatif
    if creative_noise > 0.0:
        noise = torch.randn_like(latent_model_input) * creative_noise
        latent_model_input += noise

    # Motion module
    if motion_module:
        def motion_wrapper(latents, timestep=None):
            try:
                return motion_module(latents)
            except TypeError:
                return motion_module(latents)
    else:
        motion_wrapper = lambda x, t=None: x

    # Scheduler
    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps

    for t in timesteps:
        latent_model_input = latent_model_input.to(device=device, dtype=dtype)

        # Motion module
        latent_model_input = motion_wrapper(latent_model_input, t)

        # Init image scale mix
        if init_image_scale < 1.0:
            latent_model_input = latent_model_input * init_image_scale + latent_frame * (1 - init_image_scale)

        # Guidance batch
        batch_size = pos_embeds.shape[0]
        latent_model_input_in = latent_model_input.repeat(batch_size * 2, 1, 1, 1)
        embeds = torch.cat([pos_embeds, neg_embeds], dim=0)

        # UNet
        noise_pred = unet(latent_model_input_in, t, encoder_hidden_states=embeds).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Scheduler step
        latent_model_input = scheduler.step(noise_pred, t, latent_model_input).prev_sample

        # Clamp
        latent_model_input = latent_model_input.clamp(-clamp_max, clamp_max)

    return latent_model_input

def generate_latents_ai_5D_256(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.75,
    creative_noise=0.1, seed=0, steps=20
):
    """
    Génère des latents animés optimisés pour AnimateDiff avec logs détaillés.
    Correctifs pour éviter fonds blancs et contours néons.
    """
    torch.manual_seed(seed)

    # Scheduler
    scheduler.set_timesteps(steps, device=device)

    # Déplacer latents et embeddings
    latents = latents.to(device=device, dtype=dtype)
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    # Ajout de bruit initial avec minimum 0.1 pour contraste
    noise = torch.randn_like(latents) * max(creative_noise, 0.1)
    latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0:1])
    print(f"[Init Noise] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    print(f"🔥 Seed: {seed}, Steps: {steps}, Guidance: {guidance_scale}, Init_scale: {init_image_scale}, Noise: {creative_noise}")
    print(f"[Init] Latents shape: {latents.shape}, min: {latents.min():.4f}, max: {latents.max():.4f}")
    print(f"Embeddings shape: {embeds.shape}, batch_size: {embeds.shape[0]}")

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for i, t in enumerate(scheduler.timesteps):

            # Motion module
            if motion_module is not None:
                if latents.dim() == 4:
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)
                print(f"[Step {i} Motion] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Guidance
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Scheduler step (clip_sample désactivé pour éviter fonds blancs)
            if "clip_sample" in scheduler.step.__code__.co_varnames:
                latents = scheduler.step(noise_pred, t, latents, clip_sample=False).prev_sample
            else:
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Logs avant clamp
            print(f"[Step {i} pre-clamp] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Clamp léger pour éviter overflow fp16
            LATENT_CLAMP = 1.5
            latents = latents.clamp(-LATENT_CLAMP, LATENT_CLAMP)

            # Logs après clamp
            print(f"[Step {i} post-clamp] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    # Amplification finale pour contraste avant VAE
    latents = latents * 1.2
    latents = latents.clamp(-LATENT_CLAMP, LATENT_CLAMP)

    return latents

def generate_latents_ai_5D_optimized_v1(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85,
    creative_noise=0.0, seed=0, steps=20
):
    """
    Génère des latents animés optimisés pour AnimateDiff avec logs détaillés.
    ✅ Modification : aucun clamp ou tanh sur les latents pour éviter fonds blancs et néons.
    """
    torch.manual_seed(seed)

    # Scheduler
    scheduler.set_timesteps(steps, device=device)

    # Déplacer latents et embeddings
    latents = latents.to(device=device, dtype=dtype)
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    # Ajout de bruit initial si demandé
    noise = torch.randn_like(latents) * creative_noise if creative_noise > 0 else torch.zeros_like(latents)
    latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0:1])
    print(f"[Init Noise] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    print(f"🔥 Seed: {seed}, Steps: {steps}, Guidance: {guidance_scale}, Init_scale: {init_image_scale}, Noise: {creative_noise}")
    print(f"[Init] Latents shape: {latents.shape}, min: {latents.min():.4f}, max: {latents.max():.4f}")
    print(f"Embeddings shape: {embeds.shape}, batch_size: {embeds.shape[0]}")

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for i, t in enumerate(scheduler.timesteps):

            # Motion module
            if motion_module is not None:
                if latents.dim() == 4:
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)
                print(f"[Step {i} Motion] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Guidance
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Logs uniquement (pas de clamp)
            print(f"[Step {i}] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    return latents


def decode_latents_correct(latents, vae, latent_scale=0.18215):

    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    # Move + rescale latents
    latents = latents.to(device=vae_device, dtype=vae_dtype) / latent_scale

    with torch.no_grad():
        img = vae.decode(latents).sample

        # ✅ Conversion SD correcte
        img = (img / 2 + 0.5).clamp(0.0, 1.0)

    return img.cpu()


def decode_latents_correct_v1(latents, vae, latent_scale=0.18215):
    """
    Décodage correct des latents AnimateDiff.
    - latents: sortie UNet brute
    - vae: modèle VAE chargé
    - latent_scale: scale exact du VAE (souvent 0.18215 pour SD/miniSD)
    """
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    # Move latents sur le device et dtype du VAE
    if vae_device.type == "cpu":
        latents = latents.to(device=vae_device, dtype=torch.float32) / latent_scale
    else:
        latents = latents.to(device=vae_device, dtype=vae_dtype) / latent_scale

    with torch.no_grad():
        img = vae.decode(latents).sample
        img = img.clamp(0.0, 1.0)

    return img.cpu()


def decode_latents_safe_ai(latents, vae, tile_size=128, overlap=64):
    """
    Décodage sécurisé des latents pour AnimateDiff.
    - Compatible fp16/fp32 et VAE-offload.
    - Évite l’effet néon / fonds saturés.
    - Clamp léger pour prévenir overflow, sans écraser la dynamique.
    """
    vae_device = next(vae.parameters()).device

    # 1️⃣ Assurer le bon device et dtype
    dtype = torch.float16 if vae_device.type == "cuda" else torch.float32
    latents = latents.to(device=vae_device, dtype=dtype)

    # 2️⃣ Éviter NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=10.0, neginf=-10.0)

    # 3️⃣ Clamp large pour fp16 (prévention overflow)
    LATENT_CLAMP = 5.0
    latents = latents.clamp(-LATENT_CLAMP, LATENT_CLAMP)

    # 4️⃣ Décodage VAE
    with torch.no_grad():
        # Décode le latent
        frame_tensor = vae.decode(latents).sample

        # Clamp final entre 0 et 1 pour image
        frame_tensor = frame_tensor.clamp(0.0, 1.0)

    return frame_tensor.cpu()



def decode_latents_safe_torch(latent_frame, vae, tile_size=128, overlap=64, latent_scale=LATENT_SCALE):
    vae_device = next(vae.parameters()).device
    dtype = torch.float16 if vae_device.type == "cuda" else torch.float32
    latents = latent_frame.to(device=vae_device, dtype=dtype) / latent_scale

    # Éviter NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=10.0, neginf=-10.0)

    # Optionnel : clamp large pour fp16
    # latents = latents.clamp(-10.0, 10.0)

    with torch.no_grad():
        frame_tensor = vae.decode(latents).sample
        frame_tensor = frame_tensor.clamp(0.0, 1.0)

    return frame_tensor.cpu()


def decode_latents_safe_clamp(latent_frame, vae, tile_size=128, overlap=64, latent_scale=LATENT_SCALE):
    """
    Décodage sécurisé des latents en tensor float16/float32, compatible fp16/fp32 et VAE-offload.
    Découpe en tiles pour limiter l’usage VRAM.
    """
    vae_device = next(vae.parameters()).device

    # 1️⃣ Forcer dtype float16 si GPU, sinon float32 CPU
    dtype = torch.float16 if vae_device.type == "cuda" else torch.float32
    latents = latent_frame.to(device=vae_device, dtype=dtype) / latent_scale

    # 2️⃣ Éviter NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)

    # 3️⃣ Clamp pour éviter overflow fp16
    LATENT_CLAMP = 1.5
    latents = latents.clamp(-LATENT_CLAMP, LATENT_CLAMP)

    # 4️⃣ Décodage tile par tile si nécessaire (ici on simplifie pour petits latents)
    B, C, H, W = latents.shape
    with torch.no_grad():
        # Décode et force sortie entre 0 et 1
        frame_tensor = vae.decode(latents).sample
        frame_tensor = frame_tensor.clamp(0.0, 1.0)

    return frame_tensor.cpu()



def decode_latents_safe_ai1(latent_frame, vae, tile_size=128, overlap=64, latent_scale=LATENT_SCALE):
    """
    Décodage sécurisé des latents en tensor float32, compatible fp16/fp32 et VAE-offload.
    Découpe en tiles pour limiter l’usage VRAM.
    """
    vae_device = next(vae.parameters()).device
    latents = latent_frame.to(device=vae_device, dtype=torch.float32) / latent_scale  # float32 obligatoire pour VAE

    # Éviter NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=0.0)

    # Décodage tile par tile si nécessaire
    B, C, H, W = latents.shape
    frame_tensor = torch.zeros(B, 3, H*16, W*16, device=vae_device)  # dimension finale approximative
    # Pour AnimateDiff on peut simplifier avec une seule passe si pas très grand
    with torch.no_grad():
        frame_tensor = vae.decode(latents).sample.clamp(0, 1)

    return frame_tensor.cpu()

# -------------------------
# Encode image en latents
# -------------------------
def encode_image_latents(image_tensor, vae, scale=0.18215, dtype=torch.float16):
    """
    Encode une image en latents avec VAE.
    """
    vae_device = next(vae.parameters()).device
    img = image_tensor.to(device=vae_device, dtype=torch.float32 if vae_device.type=="cpu" else dtype)
    with torch.no_grad():
        latents = vae.encode(img).latent_dist.sample() * scale
    print(f"[Encode] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")
    return latents.unsqueeze(2)  # [B,C,1,H,W]
# -------------------------
# Génération latents animés optimisés
# -------------------------
def generate_latents_ai_5D_optimized_old(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85,
    creative_noise=0.0, seed=0, steps=20
):
    """
    Génère des latents animés optimisés pour AnimateDiff avec logs détaillés.
    """
    torch.manual_seed(seed)

    # Scheduler
    scheduler.set_timesteps(steps, device=device)

    # Déplacer latents et embeddings
    latents = latents.to(device=device, dtype=dtype)
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    # Ajout de bruit initial
    noise = torch.randn_like(latents) * creative_noise if creative_noise > 0 else torch.zeros_like(latents)
    latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0:1])
    print(f"[Init Noise] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    print(f"🔥 Seed: {seed}, Steps: {steps}, Guidance: {guidance_scale}, Init_scale: {init_image_scale}, Noise: {creative_noise}")
    print(f"[Init] Latents shape: {latents.shape}, min: {latents.min():.4f}, max: {latents.max():.4f}")
    print(f"Embeddings shape: {embeds.shape}, batch_size: {embeds.shape[0]}")

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for i, t in enumerate(scheduler.timesteps):

            # Motion module
            if motion_module is not None:
                if latents.dim() == 4:
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)
                print(f"[Step {i} Motion] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Guidance
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Logs avant clamp
            print(f"[Step {i} pre-clamp] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Clamp progressif pour éviter overflow
            #clamp_val = max(1.0, init_image_scale * 10)
            #latents = latents.clamp(-clamp_val, clamp_val)

            # Clamp fixe à ±1.0
            #CLAMP_MAX = 1.0
            #latents = latents.clamp(-CLAMP_MAX, CLAMP_MAX)

            # Calculer max absolu actuel
            #max_abs = latents.abs().max()
            #if max_abs > 1.0:
            #    latents = latents / max_abs  # ramène les latents dans [-1, 1]

            latents = torch.tanh(latents)  # comprime automatiquement dans [-1, 1]

            # Logs après clamp
            print(f"[Step {i} post-clamp] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    return latents

# -------------------------
# Décodage sécurisé
# -------------------------
def decode_latents_safe(latents, vae, device="cuda", tile_size=128, overlap=64):
    """
    Décodage sécurisé des latents en tensor float32 sur CPU.
    - Compatible avec vae_offload (CPU) et latents GPU
    """
    vae_device = next(vae.parameters()).device
    latents = latents.to(vae_device).float()

    # Nettoyage NaN/Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=0.0)
    print(f"[Decode start] Latents shape: {latents.shape}, min: {latents.min():.4f}, max: {latents.max():.4f}")

    # Décodage en tiles (pour VRAM limitée)
    frame_tensor = decode_latents_to_image_tiled128(
        latents,
        vae,
        tile_size=tile_size,
        overlap=overlap,
        device=vae_device
    ).clamp(0, 1)

    print(f"[Decode end] Frame tensor min: {frame_tensor.min():.4f}, max: {frame_tensor.max():.4f}")

    return frame_tensor.cpu()

def generate_latents_ai_5D_debug_opti(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85,
    creative_noise=0.0, seed=0, steps=20
):
    """
    Génère des latents animés optimisés pour AnimateDiff.
    """
    import torch

    torch.manual_seed(seed)

    # Configure le scheduler
    scheduler.set_timesteps(steps, device=device)

    # Déplace latents et embeddings sur device
    latents = latents.to(device=device, dtype=dtype)
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    # Ajout de bruit initial contrôlé par creative_noise
    if creative_noise > 0.0:
        noise = torch.randn_like(latents) * creative_noise
    else:
        noise = torch.zeros_like(latents)
    latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0:1])

    print(f"🔥 Seed: {seed}, Steps: {steps}, Guidance scale: {guidance_scale}, Init scale: {init_image_scale}, Creative noise: {creative_noise}")
    print(f"[Init] Latents shape: {latents.shape}, min: {latents.min():.4f}, max: {latents.max():.4f}")
    print(f"Embeddings shape: {embeds.shape}, batch_size: {embeds.shape[0]}")

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for i, t in enumerate(scheduler.timesteps):
            # Motion module (optionnel)
            if motion_module is not None:
                if latents.dim() == 4:
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)
            print(f"[Step {i}] t: {t:.4f} | [Motion] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Guidance
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Step scheduler et scaling progressif pour éviter la compression
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Scaling progressif pour garder amplitude cohérente
            clamp_val = max(1.0, init_image_scale * 10)
            latents = latents.clamp(-clamp_val, clamp_val)

            print(f"[Step {i} post-step] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    return latents

def generate_latents_ai_5D_brouillard(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85,
    creative_noise=0.0, seed=0, steps=20
):
    """
    Génère des latents animés 5D à partir d'une image de base avec logs.
    """

    torch.manual_seed(seed)
    print(f"🔥 Seed: {seed}, Steps: {steps}, Guidance scale: {guidance_scale}, Init scale: {init_image_scale}, Creative noise: {creative_noise}")

    # -------------------------------------
    # Scheduler et timesteps
    # -------------------------------------
    scheduler.set_timesteps(steps, device=device)
    print(f"Scheduler timesteps: {scheduler.timesteps}")

    # -------------------------------------
    # Initialisation des latents
    # -------------------------------------
    latents = latents.to(device=device, dtype=torch.float32)
    noise = torch.randn_like(latents) * (creative_noise if creative_noise > 0 else 1.0)
    latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0:1])
    print(f"[Init] Latents shape: {latents.shape}, min: {latents.min():.4f}, max: {latents.max():.4f}")

    # -------------------------------------
    # Concat embeddings une seule fois
    # -------------------------------------
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)
    batch_size = embeds.shape[0]
    print(f"Embeddings shape: {embeds.shape}, batch_size: {batch_size}")

    # -------------------------------------
    # Boucle principale
    # -------------------------------------
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for step_idx, t in enumerate(scheduler.timesteps):
            print(f"\n[Step {step_idx}] t: {t}")

            # Motion module si disponible
            if motion_module is not None:
                if latents.dim() == 4:  # [B, C, H, W] -> ajouter F=1
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)
                print(f"[Motion] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Répéter latents pour correspondre au batch d'embeddings
            if latents.shape[0] != batch_size:
                repeats = batch_size // latents.shape[0]
                latent_model_input = latents.repeat(repeats, 1, 1, 1)
            else:
                latent_model_input = latents

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # UNet
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

            # CFG : split batch
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Step scheduler
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            latents = latents.clamp(-20, 20)  # pour stabilité

            print(f"[Step {step_idx} post-step] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    return latents.to(dtype)

def generate_latents_ai_5D_testcreative(
    latents,
    pos_embeds,
    neg_embeds,
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=5.0,
    init_image_scale=0.6,
    creative_noise=0.0,   # <-- AJOUTE ÇA
    seed=0,
    steps=12
):
    torch.manual_seed(seed)

    scheduler.set_timesteps(steps, device=device)

    latents = latents.to(device=device, dtype=dtype)

    # ---------------- IMG2IMG strength ----------------
    t_start = int(steps * init_image_scale)
    t_start = min(t_start, steps - 1)

    timesteps = scheduler.timesteps[t_start:]
    t_noise = timesteps[0:1]

    noise = torch.randn_like(latents)

    latents = scheduler.add_noise(
        latents,
        noise,
        t_noise
    )

    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):

        for t in timesteps:

            if motion_module is not None:
                if latents.dim() == 4:
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)

            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

            noise_uncond, noise_text = noise_pred.chunk(2)

            noise_pred = noise_uncond + guidance_scale * (
                noise_text - noise_uncond
            )

            latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents



def generate_latents_ai_5D_optitest(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85,
    creative_noise=0.0, seed=0, steps=20
):
    torch.manual_seed(seed)

    # 🔥 IMPORTANT : config réelle des steps
    scheduler.set_timesteps(steps, device=device)

    # Utilise dtype natif (fp16 si cuda)
    latents = latents.to(device=device, dtype=dtype)
    #latents = latents * scheduler.init_noise_sigma
    noise = torch.randn_like(latents)
    latents = scheduler.add_noise(
        latents,
        noise,
        scheduler.timesteps[0:1]
    )
    # -----------------------------------------------------
    # 🔥 concat embeddings UNE SEULE FOIS
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):

        for t in scheduler.timesteps:

            if motion_module:
                #latents = motion_module.apply(latents, t)
                if motion_module is not None:
                    # Si latents = [B,C,H,W], on ajoute F=1
                    if latents.dim() == 4:
                        latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                    else:
                        latents = motion_module(latents)

            # CFG batching propre
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

            noise_uncond, noise_text = noise_pred.chunk(2)

            noise_pred = noise_uncond + guidance_scale * (
                noise_text - noise_uncond
            )

            latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


# -------------------------
# Décodage hybride safe VAE
# -------------------------
def decode_latents_hybrid(vae, latents, scale=LATENT_SCALE):
    """
    Décodage sécurisé des latents pour VAE fp32 + latents fp16.
    latents : Tensor GPU (fp16 ou fp32)
    vae : modèle VAE (fp32 ou offload)
    scale : facteur d'échelle des latents
    """
    # Sauvegarder dtype des latents
    lat_dtype = latents.dtype

    # Conversion en fp32 pour VAE
    latents_fp32 = latents.to(torch.float32)

    with torch.no_grad():
        decoded = vae.decode(latents_fp32 / scale).sample

    # Clamp pour éviter valeurs extrêmes
    decoded = decoded.clamp(-1, 1)

    # Retourner au dtype original pour GPU/fp16 si besoin
    return decoded.to(lat_dtype)


# --- PIPELINE PRINCIPALE ---
def generate_5D_video_auto(pretrained_model_path, config, device='cuda'):
    print("🔄 Chargement des modèles...")
    motion_module = MotionModuleTiny(device=device)
    scheduler = init_scheduler(config)  # ta fonction existante
    vae = load_vae(pretrained_model_path, device=device)

    total_frames = config['total_frames']
    fps = config['fps']
    H_src, W_src = config['image_size']  # résolution source

    # Génère les latents initiaux
    latents = torch.randn(1, 4, H_src//8, W_src//8, device=device, dtype=torch.float16)
    print(f"[INFO] Latents initiaux shape={latents.shape}")

    video_frames = []
    for t in range(total_frames):
        try:
            latents = motion_module.step(latents, t)
            frame = decode_latents_frame_auto(latents, vae, H_src, W_src)
            video_frames.append(frame)
        except Exception as e:
            print(f"⚠ Erreur frame {t:05d} → reset léger: {e}")
            continue

    save_video(video_frames, fps, output_path=config['output_path'])
    print(f"🎬 Vidéo générée : {config['output_path']}")

# -------------------------
# Load images utility
# -------------------------
def load_images(paths, W, H, device, dtype):
    all_tensors = []
    for p in paths:
        t = load_image_file(p, W, H, device, dtype)
        print(f"✅ Image chargée : {p}, shape={t.shape}, dtype={t.dtype}, device={t.device}")
        all_tensors.append(t)
    return torch.stack(all_tensors, dim=0)

# -------------------------
# Mémoire GPU utils
# -------------------------
def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        print(f"[GPU MEM] {tag} → allocated={torch.cuda.memory_allocated()/1e6:.1f}MB, "
              f"reserved={torch.cuda.memory_reserved()/1e6:.1f}MB, "
              f"max_allocated={torch.cuda.max_memory_allocated()/1e6:.1f}MB")

def decode_latents_frame_auto(latents, vae, H_src, W_src):
    """
    Decode des latents VAE en images avec tiles 128x128, auto-adapté à la taille source.
    """
    device = vae.device
    print(f"[VAE] Decode → tile_size={tile_size}, overlap={overlap}, device={device}, latents.shape={latents.shape}")
    log_gpu_memory("avant decode VAE")

    # Assure batch 4D
    latents = latents.unsqueeze(0) if latents.dim() == 3 else latents

    # Décodage VAE en tiles
    with torch.no_grad():
        frame_tensor = decode_latents_to_image_tiled(
            latents,
            vae,
            tile_size=tile_size,
            overlap=overlap
        ).clamp(0,1)

        # Redimensionnement proportionnel à l'image source
        H_out, W_out = H_src, W_src
        if frame_tensor.shape[-2:] != (H_out, W_out):
            frame_tensor = torch.nn.functional.interpolate(
                frame_tensor,
                size=(H_out, W_out),
                mode='bicubic',
                align_corners=False
            )

    log_gpu_memory("après decode VAE")
    return frame_tensor.squeeze(0)
# -------------------------------------------------------------------
# ------------- PATCH Stable Diffusion --------
# ------------------------------------------------------------------

def generate_latents_ai_5D_std(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85, creative_noise=0.0, seed=0
):
    torch.manual_seed(seed)

    # Toujours float32 pour stabilité du scheduler
    latents = latents.to(device=device, dtype=torch.float32)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:

        # Sécurité NaN / Inf
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            latents = torch.randn_like(latents) * 0.1

        # Si Motion Module est actif
        if motion_module:
            #latents = motion_module.apply(latents, t)
            if motion_module is not None:
                # Si latents = [B,C,H,W], on ajoute F=1
                if latents.dim() == 4:
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)
        # Motion Module fin

        # 🔥 Préparer batch pour guidance
        batch_size = pos_embeds.shape[0] + neg_embeds.shape[0]
        if latents.shape[0] != batch_size:
            # Répéter latents pour matcher embeddings
            repeats = batch_size // latents.shape[0]
            latent_model_input = latents.repeat(repeats, 1, 1, 1)
        else:
            latent_model_input = latents

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = latent_model_input.to(dtype=dtype)

        embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(dtype=dtype)

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

        # Guidance : séparer le batch
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond.float() + guidance_scale * (
            noise_text.float() - noise_uncond.float()
        )

        # Step scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents.clamp(-20, 20)

    # Retour dtype original
    return latents.to(dtype)

# -------------------------------------------------------------------
# ------------- 5 D Original  --------
# ------------------------------------------------------------------

def generate_latents_ai_5D(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85, creative_noise=0.0, seed=0
):
    torch.manual_seed(seed)

    latents = latents.to(device=device, dtype=torch.float32)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:

        if torch.isnan(latents).any() or torch.isinf(latents).any():
            latents = torch.randn_like(latents) * 0.1

        # 🔥 DUPLICATION POUR GUIDANCE
        latent_model_input = torch.cat([latents, latents], dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = latent_model_input.to(dtype=dtype)

        embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(dtype=dtype)

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)

        noise_pred = noise_uncond.float() + guidance_scale * (
            noise_text.float() - noise_uncond.float()
        )

        latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents.clamp(-20, 20)

    return latents.to(dtype)


def generate_latents_ai_5D_v1(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85, creative_noise=0.0, seed=0
):
    torch.manual_seed(seed)
    latents = latents.to(device=device, dtype=torch.float32)  # scheduler en float32
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            latents = torch.randn_like(latents) * 0.1

        # préparation input UNet 5D
        latent_model_input = scheduler.scale_model_input(latents, t)
        latent_model_input = latent_model_input.to(dtype=dtype)  # UNet en fp16 si demandé

        # concat embeddings pour guidance
        embeds = torch.cat([neg_embeds, pos_embeds])
        embeds = embeds.to(dtype=dtype)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample

        # guidance
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond.float() + guidance_scale * (noise_text.float() - noise_uncond.float())

        # scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # clamp anti-explosion
        latents = latents.clamp(-20, 20)

    return latents.to(dtype)


# -------------------------
# UNet 3D latents generation
# -------------------------
def generate_latents_3d(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85, creative_noise=0.0, seed=0
):
    torch.manual_seed(seed)

    # Toujours float32 pour scheduler
    latents = latents.to(device=device, dtype=torch.float32)

    if init_image_scale < 1.0:
        noise = torch.randn_like(latents)
        latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0])
    else:
        latents = latents * scheduler.init_noise_sigma

    latents = latents.clamp(-10, 10)

    for t in scheduler.timesteps:

        if torch.isnan(latents).any() or torch.isinf(latents).any():
            print(f"⚠ NaN détecté à timestep {int(t)} → reset léger")
            latents = torch.randn_like(latents) * 0.1

        # 3D CFG concat
        latent_model_input = torch.cat([latents]*2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        embeds = torch.cat([neg_embeds, pos_embeds])

        if motion_module:
            latent_model_input = motion_module.apply(latent_model_input)

        with torch.no_grad():
            noise_pred = unet(latent_model_input.to(dtype), t, encoder_hidden_states=embeds).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond.float() + guidance_scale * (noise_text.float() - noise_uncond.float())

        latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents.clamp(-20,20)

        mean_val = latents.abs().mean().item()
        if math.isnan(mean_val) or mean_val < 1e-6:
            print(f"⚠ Latent instable à timestep {int(t)} → reset")
            latents = torch.randn_like(latents) * 0.05

    return latents.to(dtype)

# ---------------------------------------------------------
# Génération de latents AI compatible Fp16  et Fp32
# ---------------------------------------------------------

def generate_latents_ai(
    latents,
    pos_embeds,
    neg_embeds,
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.5,
    init_image_scale=0.85,
    creative_noise=0.0,
    seed=0,
):
    torch.manual_seed(seed)

    use_fp16 = dtype == torch.float16 and device == "cuda"

    # ------------------------------------------------
    # Toujours garder les latents scheduler en float32
    # ------------------------------------------------
    latents = latents.to(device=device, dtype=torch.float32)

    # ------------------------------------------------
    # Initialisation correcte (image vs text mode)
    # ------------------------------------------------
    if init_image_scale < 1.0:
        noise = torch.randn_like(latents)
        latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0])
    else:
        latents = latents * scheduler.init_noise_sigma

    latents = latents.clamp(-10, 10)

    for t in scheduler.timesteps:

        # ------------------------------------------------
        # Sécurité anti-NaN
        # ------------------------------------------------
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            print(f"⚠ NaN détecté à timestep {int(t)} → reset léger")
            latents = torch.randn_like(latents) * 0.1

        # ------------------------------------------------
        # CFG concat (plus stable et plus rapide)
        # ------------------------------------------------
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        embeds = torch.cat([neg_embeds, pos_embeds])

        # UNet en fp16 si activé
        model_input = latent_model_input.to(dtype if use_fp16 else torch.float32)

        with torch.no_grad():
            noise_pred = unet(
                model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)

        # ------------------------------------------------
        # Guidance toujours en float32 pour stabilité
        # ------------------------------------------------
        noise_pred = noise_uncond.float() + guidance_scale * (
            noise_text.float() - noise_uncond.float()
        )

        # ------------------------------------------------
        # Step scheduler en float32
        # ------------------------------------------------
        latents = scheduler.step(
            noise_pred,
            t,
            latents
        ).prev_sample

        latents = latents.clamp(-20, 20)

        mean_val = latents.abs().mean().item()
        if math.isnan(mean_val) or mean_val < 1e-6:
            print(f"⚠ Latent instable à timestep {int(t)} → reset")
            latents = torch.randn_like(latents) * 0.05

    # Retour au dtype demandé
    return latents.to(dtype if use_fp16 else torch.float32)



# ---------------------------------------------------------
# Génération de latents par bloc SAFE avec logs
# ---------------------------------------------------------
def generate_latents_2(latents, pos_embeds, neg_embeds, unet, scheduler, motion_module=None,
                       device="cuda", dtype=torch.float16, guidance_scale=4.5, init_image_scale=0.85):
    """
    latents: [B, C, F, H, W]
    pos_embeds / neg_embeds: [B, L, D]
    """
    torch.manual_seed(42)
    B, C, F, H, W = latents.shape

    # ⚡ Assurer dtype/device compatibilité UNet
    unet_dtype = next(unet.parameters()).dtype
    latents = latents.to(device=device, dtype=unet_dtype)
    pos_embeds = pos_embeds.to(device=device, dtype=unet_dtype)
    if neg_embeds is not None:
        neg_embeds = neg_embeds.to(device=device, dtype=unet_dtype)

    motion_module = motion_module.to(device=device, dtype=unet_dtype) if motion_module else None

    # Reshape pour timesteps
    latents = latents.permute(0, 2, 1, 3, 4).reshape(B*F, C, H, W).contiguous()
    init_latents = latents.clone()

    for t in scheduler.timesteps:
        try:
            if motion_module:
                latents = motion_module(latents)

            # classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            embeds = torch.cat([neg_embeds, pos_embeds]) if neg_embeds is not None else pos_embeds

            # Vérification NaN avant UNet
            if torch.isnan(latents).any():
                print(f"❌ Warning: NaN detected in latents before UNet | t={t}")

            with torch.no_grad():
                noise_pred = unet(latent_model_input, timestep=t, encoder_hidden_states=embeds).sample

            if neg_embeds is not None:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # ⚡ conserver influence de l'image initiale
            latents = latents + init_image_scale * (init_latents - latents)

            # Logs min/max
            print(f"🔹 Step t={t} | latents min: {latents.min():.6f}, max: {latents.max():.6f}")

            # Stop si NaN
            if torch.isnan(latents).any():
                raise RuntimeError(f"NaN detected in latents after UNet at timestep {t}")

        except Exception as e:
            print(f"❌ Erreur UNet/scheduler à t={t}: {e}")
            # On peut remplacer par latents initiaux pour continuer
            latents = init_latents.clone()
            torch.cuda.empty_cache()

    # Reshape final
    latents = latents.reshape(B, F, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
    return latents


# ---------------------------------------------------------
# Tuilage sécurisé
# ---------------------------------------------------------
def decode_latents_to_image_tiled(latents, vae, tile_size=32, overlap=8):
    """
    Decode VAE en tuiles avec couverture complète garantie.
    - Aucun trou possible
    - Blending propre
    - Stable mathématiquement
    """

    device = vae.device
    latents = latents.to(device).float() / LATENT_SCALE

    B, C, H, W = latents.shape
    stride = tile_size - overlap

    # Dimensions image finale (scale factor VAE = 8)
    out_H = H * 8
    out_W = W * 8

    output = torch.zeros(B, 3, out_H, out_W, device=device)
    weight = torch.zeros_like(output)

    # --- positions garanties ---
    y_positions = list(range(0, H - tile_size + 1, stride))
    x_positions = list(range(0, W - tile_size + 1, stride))

    if not y_positions:
        y_positions = [0]
    if not x_positions:
        x_positions = [0]

    if y_positions[-1] != H - tile_size:
        y_positions.append(H - tile_size)

    if x_positions[-1] != W - tile_size:
        x_positions.append(W - tile_size)

    for y in y_positions:
        for x in x_positions:

            y1 = y + tile_size
            x1 = x + tile_size

            tile = latents[:, :, y:y1, x:x1]

            with torch.no_grad():
                decoded = vae.decode(tile).sample

            decoded = (decoded / 2 + 0.5).clamp(0, 1)

            iy0 = y * 8
            ix0 = x * 8
            iy1 = y1 * 8
            ix1 = x1 * 8

            output[:, :, iy0:iy1, ix0:ix1] += decoded
            weight[:, :, iy0:iy1, ix0:ix1] += 1.0

    return output / weight.clamp(min=1e-6)
# -------------------------
# Génération tuilée 128x128 ultra safe VRAM
# -------------------------
def decode_latents_to_image_tiled_ori(latents, vae, tile_size=32, overlap=8):
    """
    Decode VAE en tuiles côté latent.
    Stable, sans toucher au scheduler.
    latents: [B, 4, H, W]
    """

    device = vae.device
    latents = latents.to(device).float() / LATENT_SCALE

    B, C, H, W = latents.shape
    stride = tile_size - overlap

    # Taille image finale (VAE scale factor = 8)
    out_H = H * 8
    out_W = W * 8

    output = torch.zeros(B, 3, out_H, out_W, device="cpu")
    weight = torch.zeros_like(output)

    for y in range(0, H, stride):
        for x in range(0, W, stride):

            y1 = min(y + tile_size, H)
            x1 = min(x + tile_size, W)

            tile = latents[:, :, y:y1, x:x1]

            with torch.no_grad():
                decoded = vae.decode(tile).sample

            decoded = (decoded / 2 + 0.5).clamp(0, 1)

            # coordonnées en image space
            iy0 = y * 8
            ix0 = x * 8
            iy1 = y1 * 8
            ix1 = x1 * 8

            output[:, :, iy0:iy1, ix0:ix1] += decoded.cpu()
            weight[:, :, iy0:iy1, ix0:ix1] += 1

    return (output / weight.clamp(min=1e-6)).to(device)


# ---------------------------------------------------------
# Chargement image unique → [1,C,1,H,W]
# ---------------------------------------------------------
def load_input_image(image_path, W, H, device, dtype):
    img = Image.open(image_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    img = preprocess(img).unsqueeze(0).unsqueeze(2)  # [1,C,1,H,W]
    return img.to(device=device, dtype=dtype)


# ---------------------------------------------------------
# Encode images → latents
# input: [B,C,T,H,W]
# output: [B,4,T,H_lat,W_lat]
# ---------------------------------------------------------
def encode_images(input_images, vae):
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype

    B, C, T, H, W = input_images.shape
    input_images = input_images.to(device=device, dtype=dtype)

    latents_list = []

    with torch.no_grad():
        for t in range(T):
            imgs_2d = input_images[:, :, t, :, :]  # [B,C,H,W]
            latent = vae.encode(imgs_2d).latent_dist.sample()
            latent = latent * LATENT_SCALE
            latents_list.append(latent)

    latents = torch.stack(latents_list, dim=2)  # [B,4,T,H_lat,W_lat]
    return latents

#---------------------------------------------------------
# VERSION ROBUSTE
# # Ajouter un peu de bruit créatif initial si demandé
# # Reshape pour traitement UNet [B*T, C, H, W]
#---------------------------------------------------------
import torch
import math


#-------------------------------------------------------------------------------
# VERY STABLE
#-------------------------------------------------------------------------------
def generate_latents_robuste(latents, pos_embeds, neg_embeds, unet, scheduler,
                             motion_module=None, device="cuda", dtype=torch.float16,
                             guidance_scale=7.5, init_image_scale=2.0,
                             creative_noise=0.0, seed=42):
    """
    Génère des latents pour une frame ou un batch de frames avec robustesse.

    latents : [B,C,T,H,W] latents encodés et scalés
    pos_embeds / neg_embeds : embeddings textuels
    creative_noise : float, bruit supplémentaire pour variation frame à frame
    """
    torch.manual_seed(seed)
    B, C, T, H, W = latents.shape
    latents = latents.to(device=device, dtype=dtype)
    latents = latents.permute(0,2,1,3,4).reshape(B*T, C, H, W).contiguous()
    init_latents = latents.clone()  # copie pour init_image_scale

    for t_step in scheduler.timesteps:
        # Motion module (optionnel)
        if motion_module is not None:
            latents = motion_module(latents)

        # Ajouter un peu de noise créatif si demandé
        if creative_noise > 0:
            latents = latents + torch.randn_like(latents) * creative_noise

        # Classifier-free guidance
        latent_model_input = torch.cat([latents, latents], dim=0)
        embeds = torch.cat([neg_embeds, pos_embeds], dim=0)
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t_step, encoder_hidden_states=embeds).sample
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # Scheduler step
        latents = scheduler.step(noise_pred, t_step, latents).prev_sample

        # Réinjection de l'image initiale
        latents = latents + init_image_scale * (init_latents - latents)

        # Vérification NaN / inf
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            print(f"⚠ NaN/inf détecté à timestep {t_step}, réinitialisation avec petit bruit")
            latents = latents.clone()
            latents = latents + torch.randn_like(latents) * 1e-3

        # Log de debug
        mean_val = latents.abs().mean().item()
        std_val = latents.std().item()
        if math.isnan(mean_val) or mean_val < 1e-5:
            print(f"⚠ Latent trop petit à timestep {t_step}, mean={mean_val:.6f}")

    # Remettre la forme [B,C,T,H,W]
    latents = latents.reshape(B, T, C, H, W).permute(0,2,1,3,4).contiguous()
    return latents
# --------------------- V1 ----------------------------------------
def generate_latents_robuste_v1(
    latents,
    pos_embeds,
    neg_embeds,
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=7.5,
    init_image_scale=2.0,
    creative_noise=0.0,
    seed=42
):
    """
    Génère des latents animés à partir d'une séquence initiale.

    latents: [B, 4, T, H, W] (déjà encodés et scalés)
    pos_embeds / neg_embeds: embeddings texte pour guidance
    guidance_scale: poids de guidance classifier-free
    init_image_scale: poids de l'image initiale
    creative_noise: bruit aléatoire ajouté avant chaque étape pour variation
    """
    torch.manual_seed(seed)

    # Vérification dimensions
    if latents.ndim != 5 or latents.shape[1] != 4:
        raise ValueError(f"Latents attendus en [B,4,T,H,W], got {latents.shape}")

    B, C, T, H, W = latents.shape
    latents = latents.to(device=device, dtype=dtype)

    # Ajouter un peu de bruit créatif initial si demandé
    if creative_noise > 0:
        latents = latents + torch.randn_like(latents) * creative_noise

    # Reshape pour traitement UNet [B*T, C, H, W]
    latents = latents.permute(0,2,1,3,4).reshape(B*T, C, H, W).contiguous()
    init_latents = latents.clone()

    for t_idx, t in enumerate(scheduler.timesteps):

        # Appliquer motion module si présent
        if motion_module is not None:
            latents = motion_module(latents)

        # Classifier-free guidance
        latent_model_input = torch.cat([latents, latents], dim=0)
        embeds = torch.cat([neg_embeds, pos_embeds], dim=0)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample

        noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Réappliquer init_image_scale pour garder influence image initiale
        latents = latents + init_image_scale * (init_latents - latents)

        # Log pour vérifier valeurs anormales
        mean_val = latents.abs().mean().item()
        if math.isnan(mean_val) or mean_val < 1e-5:
            print(f"⚠ Step {t_idx}/{len(scheduler.timesteps)}: mean latent {mean_val:.6f}, reset avec petit bruit")
            latents = init_latents + torch.randn_like(init_latents) * 0.01

    # Repasser en [B, C, T, H, W]
    latents = latents.reshape(B, T, C, H, W).permute(0,2,1,3,4).contiguous()

    return latents


# ---------------------------------------------------------
# Diffusion FONCTIONNE PARFAITEMENT
# images_latents: [B,4,T,H,W]
# ---------------------------------------------------------
def generate_latents(latents, pos_embeds, neg_embeds, unet, scheduler, motion_module=None, device="cuda", dtype=torch.float16, guidance_scale=7.5, init_image_scale=2.0, seed=42):
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
# Génération de latents tuilés (block_size + overlap)
# -------------------------
def generate_tiled(input_latents, pos_embeds, neg_embeds, unet, scheduler, motion_module,
                   device, dtype, guidance_scale=4.5, init_image_scale=0.85,
                   block_size=128, overlap=16):
    """
    input_latents: [B, C, F, H, W]
    Retourne latents [B, C, F, H, W]
    """
    B, C, F, H, W = input_latents.shape
    output_latents = torch.zeros_like(input_latents)

    h_blocks = math.ceil(H / (block_size - overlap))
    w_blocks = math.ceil(W / (block_size - overlap))

    for hi in range(h_blocks):
        for wi in range(w_blocks):
            h_start = hi * (block_size - overlap)
            w_start = wi * (block_size - overlap)
            h_end = min(h_start + block_size, H)
            w_end = min(w_start + block_size, W)
            h_start = max(h_end - block_size, 0)
            w_start = max(w_end - block_size, 0)

            block = input_latents[:, :, :, h_start:h_end, w_start:w_end]

            # Génération latents sur le bloc
            block_out = generate_latents_2(
                block, pos_embeds, neg_embeds, unet, scheduler, motion_module,
                device, dtype, guidance_scale, init_image_scale
            )

            output_latents[:, :, :, h_start:h_end, w_start:w_end] = block_out

    return output_latents


# ---------------------------------------------------------
# Decode latents → images
# latents: [B,4,T,H,W]
# ---------------------------------------------------------
def decode_latents(latents, vae):

    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype

    B, C, T, H, W = latents.shape
    latents = latents.to(device=device, dtype=dtype)

    frames = []

    with torch.no_grad():
        for t in range(T):
            latent = latents[:, :, t, :, :]

            # 🔥 INVERSE DU SCALE (CORRECTION MAJEURE)
            latent = latent / LATENT_SCALE

            img = vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)

            frames.append(img.float())

    images = torch.stack(frames, dim=2)  # [B,3,T,H,W]
    return images


# ---------------------------------------------------------
# Sauvegarde vidéo
# ---------------------------------------------------------
def create_video_from_latents(latents, vae, output_dir, fps=12):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = decode_latents(latents, vae)

    images = images.squeeze(0)  # [3,T,H,W]
    images = images.permute(1,0,2,3)  # [T,3,H,W]

    for i, img in enumerate(images):
        save_image(img, output_dir / f"frame_{i:04d}.png")

    (
        ffmpeg
        .input(f"{output_dir}/frame_%04d.png", framerate=fps)
        .output(str(output_dir / "output.mp4"),
                vcodec="libx264",
                pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )

    print(f"🎬 Vidéo générée : {output_dir / 'output.mp4'}")


# ---------- NEW FONCTION ----------------------------------
# safe_load_unet, safe_load_vae, safe_load_scheduler, encode_images_to_latents, decode_latents_to_image, load_images, save_frames_as_video

# -------------------------
# Load models safe
# -------------------------

def safe_load_unet(model_path, device, fp16=True):
    folder = os.path.join(model_path, "unet")
    if os.path.exists(folder):
        model = UNet2DConditionModel.from_pretrained(folder)
        if fp16:
            model = model.half()  # réduit la VRAM de moitié
        return model.to(device)
    return None

#def safe_load_unet(model_path, device, fp16=False):
#    model = UNet2DConditionModel.from_pretrained(os.path.join(model_path,"unet"))
#    if fp16: model = model.half()
#    return model.to(device)

def safe_load_vae(model_path, device, fp16=False, offload=False):
    model = AutoencoderKL.from_pretrained(os.path.join(model_path,"vae"))
    model = model.to("cpu" if offload else device)
    if fp16: model = model.half()
    return model

def safe_load_scheduler(model_path):
    return DPMSolverMultistepScheduler.from_pretrained(os.path.join(model_path,"scheduler"))

# -------------------------
# Encode / Decode FP16 safe
# -------------------------
def encode_images_to_latents_safe(images, vae):
    device = vae.device
    dtype = next(vae.parameters()).dtype  # prend fp16 si le VAE est en FP16
    images = images.to(device=device, dtype=dtype)

    with torch.no_grad():
        if images.dim() == 5:  # [B,C,F,H,W]
            B, C, F, H, W = images.shape
            images_2d = images.view(B*F, C, H, W)
            latents_2d = vae.encode(images_2d).latent_dist.sample() * LATENT_SCALE
            latent_shape = latents_2d.shape
            latents = latents_2d.view(B, F, latent_shape[1], latent_shape[2], latent_shape[3])
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        else:
            latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
            latents = latents.unsqueeze(2)
    return latents


def decode_latents_to_image_safe(latents, vae):
    dtype = next(vae.parameters()).dtype
    latents = latents.to(vae.device).to(dtype) / LATENT_SCALE
    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0,1)
    return img

# ------------------------------
def encode_images_to_latents_half(images, vae):
    # récupère dtype réel du VAE
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    images = images.to(device=vae_device, dtype=vae_dtype)

    with torch.no_grad():

        if images.dim() == 5:  # [B,C,F,H,W]
            B, C, F, H, W = images.shape
            images_2d = images.view(B * F, C, H, W)

            latents_2d = vae.encode(images_2d).latent_dist.sample()
            latents_2d = latents_2d * LATENT_SCALE

            latents = latents_2d.view(
                B, F,
                latents_2d.shape[1],
                latents_2d.shape[2],
                latents_2d.shape[3]
            )

            latents = latents.permute(0, 2, 1, 3, 4).contiguous()

        else:
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * LATENT_SCALE
            latents = latents.unsqueeze(2)

    return latents

def decode_latents_to_image_vae(latents, vae):

    # Récupère device + dtype réel du VAE
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    # Aligne le dtype sur celui du VAE
    latents = latents.to(device=vae_device, dtype=vae_dtype)

    latents = latents / LATENT_SCALE

    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)

    # On repasse en float32 pour sauvegarde PNG
    return img.float()
# -------------------------
# Encode / Decode
# -------------------------
def encode_images_to_latents_ori(images, vae):
    device = vae.device
    images = images.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        if images.dim() == 5:  # [B,C,F,H,W]
            B, C, F, H, W = images.shape
            images_2d = images.view(B*F, C, H, W)
            latents_2d = vae.encode(images_2d).latent_dist.sample() * LATENT_SCALE
            latent_shape = latents_2d.shape
            latents = latents_2d.view(B, F, latent_shape[1], latent_shape[2], latent_shape[3])
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        else:
            latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
            latents = latents.unsqueeze(2)
    return latents

def decode_latents_to_image_ori(latents, vae):
    latents = latents.to(vae.device).float() / LATENT_SCALE
    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0,1)
    return img


# --------------------------------------------------------
# | Mode        | VAE     | Images  | Latents | Résultat  |
# | ----------- | ------- | ------- | ------- | --------  |
# | fp32        | float32 | float32 | float32 | ✅        |
# | fp16        | float16 | float16 | float16 | ✅        |
# | offload CPU | float32 | float32 | float32 | ✅        |
# ------------------------- ci dessous:
# -------------------------
# Encode / Decode corrigé FP16 safe
# -------------------------
def encode_images_to_latents(images, vae):
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype  # on aligne avec le VAE
    images = images.to(device=device, dtype=dtype)
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
    return latents

def decode_latents_to_image(latents, vae):
    # On force latents à avoir le même dtype et device que le VAE
    vae_dtype = next(vae.parameters()).dtype
    vae_device = next(vae.parameters()).device
    latents = latents.to(device=vae_device, dtype=vae_dtype) / LATENT_SCALE

    with torch.no_grad():
        img = vae.decode(latents).sample

    # Normalisation sûre vers 0-1
    img = (img / 2 + 0.5).clamp(0, 1)

    # Si FP16 → convertir en float32 pour torchvision save_image
    if img.dtype == torch.float16:
        img = img.float()

    return img

# NEW
#
#
# ---------------------------------------------------------
# Decode latents to image avec logs et sécurité
# ---------------------------------------------------------
def decode_latents_to_image_2(latents, vae, latent_scale=0.18215):
    """
    latents: [B, C, F, H, W] ou [B, C, 1, H, W] pour frame unique
    vae: VAE pour décodage
    """
    try:
        print(f"🔹 decode_latents_to_image_2 | input shape: {latents.shape}, dtype: {latents.dtype}, device: {latents.device}")

        # Si latents a une dimension de frame singleton, la squeeze
        if latents.shape[2] == 1:
            latents = latents.squeeze(2)
            print(f"🔹 Squeeze frame dimension → shape: {latents.shape}")

        # Assurer dtype et device compatible VAE
        vae_dtype = next(vae.parameters()).dtype
        vae_device = next(vae.parameters()).device
        latents = latents.to(device=vae_device, dtype=vae_dtype) / latent_scale

        # Check NaN avant VAE
        print(f"🔹 Latents before VAE decode | min: {latents.min()}, max: {latents.max()}, dtype: {latents.dtype}")
        if torch.isnan(latents).any():
            print("❌ Warning: NaN detected in latents before VAE decode!")

        with torch.no_grad():
            img = vae.decode(latents).sample

        # Check NaN après décodage
        print(f"🔹 Image after VAE decode | min: {img.min()}, max: {img.max()}, dtype: {img.dtype}")
        if torch.isnan(img).any():
            print("❌ Warning: NaN detected in decoded image!")

        # Normalisation safe vers 0-1
        img = (img / 2 + 0.5).clamp(0, 1)
        print(f"🔹 Image final | min: {img.min()}, max: {img.max()}, dtype: {img.dtype}, shape: {img.shape}")

        # Conversion FP16 -> FP32 si nécessaire
        if img.dtype == torch.float16:
            img = img.float()

        return img

    except Exception as e:
        print(f"❌ Exception in decode_latents_to_image_2: {e}")
        # Retourne une image noire safe si VAE échoue
        B, C, H, W = latents.shape[:4]
        return torch.zeros(B, 3, H*8, W*8, device=latents.device)  # scale approx 8x pour SD VAE
# -------------------------
# Encode / Decode corrigé
# -------------------------
# -------------------------

def decode_latents_to_image_old(latents, vae):
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(device=vae_device, dtype=vae_dtype)
    latents = latents / LATENT_SCALE
    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
    return img.float()  # on repasse en float32 pour PNG

# -------------------------
# Image utilities
# -------------------------
def load_image_file(path, W, H, device, dtype):
    img = Image.open(path).convert("RGB")
    img = img.resize((W,H), Image.LANCZOS)
    img_tensor = torch.tensor(np.array(img)).permute(2,0,1).to(device=device, dtype=dtype)/127.5 - 1.0
    return img_tensor

# -------------------------
# Image utilities
# -------------------------
def load_images(paths, W, H, device, dtype):
    all_tensors = []
    for p in paths:
        if p.lower().endswith(".gif"):
            img = Image.open(p)
            frames = [torch.tensor(np.array(f)).permute(2,0,1).to(device=device, dtype=dtype)/127.5 - 1.0
                      for f in ImageSequence.Iterator(img)]
            print(f"✅ GIF chargé : {p} avec {len(frames)} frames")
            all_tensors.extend(frames)
        else:
            t = load_image_file(p, W, H, device, dtype)
            print(f"✅ Image chargée : {p}")
            all_tensors.append(t)
    return torch.stack(all_tensors, dim=0)



def load_images_s(paths, W, H, device, dtype):
    all_tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((W,H), Image.LANCZOS)
        t = torch.tensor(np.array(img)).permute(2,0,1).to(device=device,dtype=dtype)/127.5 - 1.0
        all_tensors.append(t)
    return torch.stack(all_tensors, dim=0)

def load_images_all(paths, W, H, device, dtype):
    all_tensors = []
    for p in paths:
        if p.lower().endswith(".gif"):
            img = Image.open(p)
            frames = [torch.tensor(np.array(f)).permute(2,0,1).to(device=device, dtype=dtype)/127.5 - 1.0
                      for f in ImageSequence.Iterator(img)]
            print(f"✅ GIF chargé : {p} avec {len(frames)} frames")
            all_tensors.extend(frames)
        else:
            t = load_image_file(p, W, H, device, dtype)
            print(f"✅ Image chargée : {p}")
            all_tensors.append(t)
    return torch.stack(all_tensors, dim=0)

# -------------------------
# Save video
# -------------------------
def save_frames_as_video_rmtmp(frames, output_path, fps=12):
    temp_dir = Path("temp_frames")
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    for idx, frame in enumerate(frames):
        frame.save(temp_dir / f"frame_{idx:05d}.png")
    (
        ffmpeg.input(f"{temp_dir}/frame_%05d.png", framerate=fps)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
    shutil.rmtree(temp_dir)
# -------------------------
# Video utilities
# -------------------------
def save_frames_as_video(frames, output_path, fps=12):
    temp_dir = Path("temp_frames")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    for idx, frame in enumerate(frames):
        frame.save(temp_dir / f"frame_{idx:05d}.png")

    (
        ffmpeg.input(f"{temp_dir}/frame_%05d.png", framerate=fps)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
    shutil.rmtree(temp_dir)
# -----------------------------
# -------- MOTION -------------
# -----------------------------
def default_motion_module(latents: torch.Tensor, frame_idx: int = 0, total_frames: int = 1) -> torch.Tensor:
    """
    Motion module par défaut avec effets aléatoires pour Tiny-SD
    - Translation aléatoire subtile
    - Oscillation douce
    - Zoom subtil (ne change pas la taille)
    - Bruit léger sur latents
    """
    # --- Zoom subtil ---
    zoom_factor = random.uniform(0.98, 1.02)  # ±2%
    latents = latents * zoom_factor  # scale des valeurs des latents seulement
    # (⚠️ Ne change pas H/W, safe pour le scheduler)

    B, C, H, W = latents.shape

    # --- Translation aléatoire subtile (pan/tilt) ---
    dx = random.randint(-1, 1)
    dy = random.randint(-1, 1)

    # --- Oscillation douce ---
    osc_amp = 1
    dx_osc = int(osc_amp * math.sin(2 * math.pi * frame_idx / max(total_frames,1)))
    dy_osc = int(osc_amp * math.cos(2 * math.pi * frame_idx / max(total_frames,1)))

    # Pan/tilt safe avec torch.roll
    latents = torch.roll(latents, shifts=(dy+dy_osc, dx+dx_osc), dims=(2,3))

    # --- Noise léger ---
    noise_sigma = 0.003
    latents = latents + torch.randn_like(latents) * noise_sigma

    return latents


def default_motion_module_test(latents: torch.Tensor, frame_idx: int = 0, total_frames: int = 1) -> torch.Tensor:
    # Zoom subtil
    zoom_factor = 0.01
    factor = 1.0 + zoom_factor * frame_idx / max(total_frames,1)
    B, C, H, W = latents.shape
    latents = F.interpolate(latents, scale_factor=factor, mode='bilinear', align_corners=False)
    latents = latents[:, :, :H, :W]  # recadrer si nécessaire

    # Oscillation subtile
    dx = int(1 * math.sin(2 * math.pi * frame_idx / max(total_frames,1)))
    dy = int(1 * math.cos(2 * math.pi * frame_idx / max(total_frames,1)))
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid_x = torch.clamp(grid_x - dx, 0, W-1)
    grid_y = torch.clamp(grid_y - dy, 0, H-1)
    latents = latents[:, :, grid_y, grid_x]

    return latents

# -------------------------
# Model loaders
# -------------------------


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# -------------------------
# Upscale vidéo avec Real-ESRGAN
# -------------------------
def upscale_video_with_realesrgan(video_path, output_path, device="cuda", scale=2, fps=12):
    """
    Upscale une vidéo avec Real-ESRGAN (torch)
    - video_path : chemin vidéo d'entrée
    - output_path : chemin vidéo sortie
    - scale : facteur d'upscale (2, 4, etc.)
    - fps : framerate de sortie
    """
    try:
        from realesrgan import RealESRGAN
    except ImportError:
        print("❌ Module Real-ESRGAN non installé. pip install realesrgan")
        return

    import tempfile
    from PIL import Image
    import ffmpeg

    temp_dir = tempfile.mkdtemp()
    temp_out_dir = tempfile.mkdtemp()

    # 1️⃣ Extraire frames
    (
        ffmpeg
        .input(str(video_path))
        .output(f"{temp_dir}/frame_%05d.png")
        .overwrite_output()
        .run(quiet=True)
    )

    # 2️⃣ Charger modèle
    model = RealESRGAN(device, scale=scale)
    model.load_weights(f"RealESRGAN_x{scale}.pth", download=True)  # si pas déjà présent

    # 3️⃣ Upscale frame par frame
    frame_paths = sorted(Path(temp_dir).glob("frame_*.png"))
    for idx, fpath in enumerate(frame_paths):
        img = Image.open(fpath).convert("RGB")
        upscaled = model.predict(img)
        upscaled.save(Path(temp_out_dir) / f"frame_{idx:05d}.png")

    # 4️⃣ Recomposer la vidéo
    (
        ffmpeg
        .input(f"{temp_out_dir}/frame_%05d.png", framerate=fps)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p", crf=18)
        .overwrite_output()
        .run(quiet=True)
    )

    # 5️⃣ Cleanup
    shutil.rmtree(temp_dir)
    shutil.rmtree(temp_out_dir)
    print(f"✅ Vidéo finale Real-ESRGAN x{scale} générée : {output_path}")

# -------------------------
# Upscale cinématique (sans torch)
# -------------------------
def upscale_video_cinematic_smooth(input_video_path, output_video_path, scale=4, fps=12, interp_frames=1):
    """
    Upscale x4 avec interpolation de frames pour smooth cinematic style.
    - input_video_path : vidéo 128x128
    - output_video_path : vidéo finale upscalée
    - scale : facteur d'upscale (par défaut 4)
    - interp_frames : nombre de frames interpolées entre deux frames (smooth)
    """
    input_video_path = Path(input_video_path)
    temp_dir = input_video_path.parent / "temp_frames_smooth"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # Extraction frames
    (
        ffmpeg
        .input(str(input_video_path))
        .output(str(temp_dir / "frame_%05d.png"))
        .overwrite_output()
        .run(quiet=True)
    )

    # Charger frames existantes
    frame_paths = sorted(temp_dir.glob("frame_*.png"))
    upscaled_frames = []

    for idx in range(len(frame_paths)):
        frame = Image.open(frame_paths[idx])
        W, H = frame.size
        frame_up = frame.resize((W*scale, H*scale), Image.LANCZOS)

        upscaled_frames.append(frame_up)

        # Interpolation linéaire entre frames pour smooth
        if idx < len(frame_paths) - 1 and interp_frames > 0:
            next_frame = Image.open(frame_paths[idx+1]).resize((W*scale, H*scale), Image.LANCZOS)
            for t in range(1, interp_frames+1):
                alpha = t / (interp_frames+1)
                interp_frame = Image.blend(frame_up, next_frame, alpha)
                upscaled_frames.append(interp_frame)

    # Ré-encodage vidéo
    temp_up_dir = input_video_path.parent / "temp_frames_upscaled_smooth"
    if temp_up_dir.exists():
        shutil.rmtree(temp_up_dir)
    temp_up_dir.mkdir()

    for i, f in enumerate(upscaled_frames):
        f.save(temp_up_dir / f"frame_{i:05d}.png")

    (
        ffmpeg
        .input(str(temp_up_dir / "frame_%05d.png"), framerate=fps*(interp_frames+1))
        .output(str(output_video_path), vcodec="libx264", pix_fmt="yuv420p", crf=18)
        .overwrite_output()
        .run(quiet=True)
    )

    shutil.rmtree(temp_dir)
    shutil.rmtree(temp_up_dir)
    print(f"✅ Vidéo finale x{scale} smooth cinematic générée : {output_video_path}")
