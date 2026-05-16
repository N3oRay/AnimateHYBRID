#n3rMotionHair.py
import torch
import torch.nn.functional as F
import os
import numpy as np
from .n3rMotionPose_tools import save_impact_map
from .n3rMotionPoseClass import Pose


#-------------------------------------------- Gestion du vent ------------------------------------------------

def apply_hair_motion_cycle(
    latents,
    mask_hair,
    grid,
    H,
    W,
    frame_counter,
    device,
    delta_px=None,
    prev_hair_field=None,
    strength=1.5,
    debug=False,
    debug_dir=None
):
    """
    Alternance de 3 styles de mouvement des cheveux :
    0 → apply_hair_motion_vent
    1 → apply_hair_motion_3D (cinéma)
    2 → apply_hair_motion_extreme
    """

    mode = frame_counter % 4  # cycle 0,1,2,3

    if mode == 0:
        latents_hair, hair_delta = apply_hair_motion_vent(
            latents, mask_hair, grid, H, W,
            frame_counter, device,
            delta_px=delta_px,
            prev_hair_field=prev_hair_field,
            strength=strength,
            debug=debug,
            debug_dir=debug_dir
        )
    elif mode == 1:
        latents_hair, hair_delta = apply_hair_motion_3D(
            latents, mask_hair, grid, H, W,
            frame_counter, device,
            delta_px=delta_px,
            prev_hair_field=prev_hair_field,
            strength=strength,
            debug=debug,
            debug_dir=debug_dir
        )
    elif mode == 2:
        latents_hair, hair_delta = apply_hair_motion_cinema(
            latents, mask_hair, grid, H, W,
            frame_counter, device,
            delta_px=delta_px,
            prev_hair_field=prev_hair_field,
            strength=strength,
            debug=debug,
            debug_dir=debug_dir
        )
    else:  # mode == 3
        latents_hair, hair_delta = apply_hair_motion_extreme(
            latents, mask_hair, grid, H, W,
            frame_counter, device,
            delta_px=delta_px,
            prev_hair_field=prev_hair_field,
            strength=strength,
            debug=debug,
            debug_dir=debug_dir
        )

        if debug:
            print("[DEBUG] apply_hair_motion_cycle")
            print("  - hair_delta mean px:", hair_delta.abs().mean().item())
            print("  - hair_delta max px:", hair_delta.abs().max().item())

    return latents_hair, hair_delta
#----------------------------------------------------------------------------------------------
def apply_hair_motion_3D(
    latents,
    mask_hair,
    grid,
    H,
    W,
    frame_counter,
    device,
    delta_px=None,
    prev_hair_field=None,
    strength=1.0,   # 🔥 Nouveau paramètre
    debug=False,
    debug_dir=None
):
    """
    Hair motion 3D amplifiée avec contrôle de force global via `strength`
    """
    B = latents.shape[0]
    t = torch.tensor(frame_counter, device=device, dtype=torch.float32)
    t_wind1 = t / 15.0
    t_wind2 = t / 60.0

    def multi_noise(grid, t, scales=[0.05,0.15,0.3], weights=[1.0,0.5,0.25]):
        val = 0
        for s, w in zip(scales, weights):
            val += w * smooth_noise(grid, t, scale=s)
        return val

    noise_x = multi_noise(grid, t)
    noise_y = multi_noise(grid, t + 123, scales=[0.08,0.2,0.4], weights=[1.0,0.5,0.25])

    hair_delta_field = torch.zeros((1, H, W, 2), device=device)
    hair_delta_field[...,0] = 0.06 * noise_x * strength
    hair_delta_field[...,1] = 0.10 * noise_y * strength

    wind_dir = torch.tensor([[1.0,0.2],[0.3,0.1]], device=device).mean(dim=0).view(1,1,1,2)
    wind_strength = (0.04 + 0.02 * torch.sin(t_wind1) + 0.01 * torch.sin(t_wind2)) * strength
    wind_delta = wind_dir * wind_strength

    gravity_delta = torch.zeros_like(hair_delta_field)
    gravity_delta[...,1] = 0.008 * strength

    if delta_px is not None:
        speed = torch.norm(delta_px, dim=-1, keepdim=True)
        hair_delta_field = hair_delta_field * (1.0 + 3.5 * speed)
        wind_delta = wind_delta * (1.0 + 2.0 * speed)
        gravity_delta = gravity_delta * (1.0 + 0.8 * speed)

    inertia = 0.7
    if prev_hair_field is not None:
        hair_delta_field = inertia * prev_hair_field + (1 - inertia) * hair_delta_field

    hair_delta_field = hair_delta_field.expand(B, H, W, 2).clone()
    wind_delta = wind_delta.expand(B, H, W, 2).clone()
    gravity_delta = gravity_delta.expand(B, H, W, 2).clone()

    mask_hair_expand = mask_hair.permute(0,2,3,1)
    yy = torch.linspace(0,1,H,device=device).view(1,H,1,1)
    smooth_falloff = yy**2.5 * (3 - 2*yy**1.5)
    mask_hair_expand = mask_hair_expand * smooth_falloff

    spring = 0.006 * torch.sin(t*0.5 + grid[...,1:2]*3.0) * strength
    hair_delta_field[...,1:2] += spring.expand(B,H,W,1)

    micro_noise = 0.002 * (torch.rand_like(hair_delta_field)-0.5) * strength
    hair_delta_field += micro_noise

    grid_hair = grid + hair_delta_field * mask_hair_expand
    grid_hair += wind_delta * mask_hair_expand
    grid_hair += gravity_delta * mask_hair_expand

    grid_hair[...,0] = 2.0 * grid_hair[...,0] / (W-1) - 1.0
    grid_hair[...,1] = 2.0 * grid_hair[...,1] / (H-1) - 1.0

    latents_out = F.grid_sample(latents, grid_hair, align_corners=True)

    if debug:
        print(f"[DEBUG] Hair motion 3D applied with strength={strength:.2f}")
        if debug_dir is not None:
            debug_save_mask_and_wind(mask=mask_hair, wind_delta=wind_delta, H=H, W=W,
                                     debug_dir=debug_dir, frame_counter=frame_counter)

    return latents_out, hair_delta_field



def apply_hair_motion_extreme(
    latents,
    mask_hair,
    grid,
    H,
    W,
    frame_counter,
    device,
    delta_px=None,
    prev_hair_field=None,
    strength=1.0,  # 🔥 Nouveau paramètre
    debug=False,
    debug_dir=None
):
    """
    Hair motion version CINÉMA EXTRÊME avec contrôle global `strength`.
    """
    B = latents.shape[0]
    t = torch.tensor(frame_counter, device=device, dtype=torch.float32)
    t_wind1 = t / 10.0
    t_wind2 = t / 40.0
    t_wind3 = t / 7.0

    def multi_noise(grid, t, scales=[0.05,0.15,0.3], weights=[1.0,0.5,0.25]):
        val = 0
        for s, w in zip(scales, weights):
            val += w * smooth_noise(grid, t, scale=s)
        return val

    noise_x = multi_noise(grid, t)
    noise_y = multi_noise(grid, t + 123, scales=[0.08,0.2,0.4], weights=[1.0,0.5,0.25])

    hair_delta_field = torch.zeros_like(grid)
    hair_delta_field[...,0] = 0.12 * noise_x * strength
    hair_delta_field[...,1] = 0.18 * noise_y * strength

    wind_dir = torch.tensor([[1.0,0.3],[0.5,0.2]], device=device).mean(dim=0).view(1,1,1,2)
    wind_strength = (0.12 + 0.06*torch.sin(t_wind1) + 0.03*torch.sin(t_wind2) + 0.02*torch.sin(t_wind3)) * strength
    wind_delta = wind_dir * wind_strength

    gravity_delta = torch.zeros_like(grid)
    gravity_delta[...,1] = 0.015 * strength

    if delta_px is not None:
        speed = torch.norm(delta_px, dim=-1, keepdim=True)
        hair_delta_field = hair_delta_field * (1.0 + 5.0 * speed)
        wind_delta = wind_delta * (1.0 + 3.0 * speed)
        gravity_delta = gravity_delta * (1.0 + 1.5 * speed)

    inertia = 0.5
    if prev_hair_field is not None:
        hair_delta_field = inertia * prev_hair_field + (1-inertia) * hair_delta_field

    mask_hair_expand = mask_hair.permute(0,2,3,1)
    yy = torch.linspace(0,1,H,device=device).view(1,H,1,1)
    extreme_falloff = yy**3 * (3 - 2*yy**1.5)
    mask_hair_expand = mask_hair_expand * extreme_falloff

    spring = 0.01 * torch.sin(frame_counter*0.8 + grid[...,1:2]*5.0) * strength
    hair_delta_field[...,1:2] += spring

    micro_noise = 0.003 * (torch.rand_like(hair_delta_field)-0.5) * strength
    hair_delta_field += micro_noise

    grid_hair = grid + hair_delta_field * mask_hair_expand
    grid_hair += wind_delta * mask_hair_expand
    grid_hair += gravity_delta * mask_hair_expand

    grid_hair[...,0] = 2.0 * grid_hair[...,0] / (W-1) - 1.0
    grid_hair[...,1] = 2.0 * grid_hair[...,1] / (H-1) - 1.0

    latents_out = F.grid_sample(latents, grid_hair, align_corners=True)

    if debug:
        print(f"[DEBUG] Hair motion EXTREME applied with strength={strength:.2f}")

    return latents_out, hair_delta_field

def apply_hair_motion_vent(
    latents,
    mask_hair,
    grid,
    H,
    W,
    frame_counter,
    device,
    delta_px=None,
    prev_hair_field=None,
    strength=1.0,  # 🔹 Nouveau paramètre
    debug=False,
    debug_dir=None
):
    """
    Hair motion VENT amélioré avec contrôle global `strength`.
    """
    B = latents.shape[0]

    # 🔹 Temps (Tensor SAFE)
    t_wind1 = torch.tensor(frame_counter / 10.0, device=device)
    t_wind2 = torch.tensor(frame_counter / 40.0, device=device)

    # 🔹 Multi-noise
    def multi_noise(grid, t, scales=[0.05,0.15,0.3], weights=[1.0,0.5,0.25]):
        val = 0
        for s, w in zip(scales, weights):
            val += w * smooth_noise(grid, t, scale=s)
        return val

    noise_x = multi_noise(grid, frame_counter)
    noise_y = multi_noise(grid, frame_counter + 123, scales=[0.08,0.2,0.4])

    # 🔹 Base motion
    hair_delta_field = torch.zeros_like(grid)
    hair_delta_field[...,0] = 0.10 * noise_x * strength
    hair_delta_field[...,1] = 0.14 * noise_y * strength

    # 🔹 WIND STRENGTH
    wind_strength = (
        0.08
        + 0.04 * torch.sin(t_wind1)
        + 0.02 * torch.sin(t_wind2)
        + 0.03 * torch.cos(t_wind1 * 1.3)
        + 0.015 * torch.cos(t_wind2 * 0.7)
    ) * strength

    # 🔹 Direction dynamique
    angle = 0.5 * torch.sin(t_wind2) + 0.3 * torch.cos(t_wind1)
    wind_dir = torch.stack([
        torch.cos(angle),
        torch.sin(angle) * 0.5
    ], dim=-1).view(1,1,1,2)
    wind_delta = wind_dir * wind_strength
    wind_delta = wind_delta.expand(B, H, W, 2).clone()

    # 🔹 Gravité
    gravity_delta = torch.zeros((B, H, W, 2), device=device)
    gravity_delta[...,1] = 0.012 * strength

    # 🔹 Influence torse
    if delta_px is not None:
        speed = torch.norm(delta_px, dim=-1, keepdim=True).view(B,1,1,1)
        hair_delta_field *= (1.0 + 4.0 * speed)
        wind_delta = wind_delta * (1.0 + 2.5 * speed)
        gravity_delta = gravity_delta * (1.0 + 1.2 * speed)

    # 🔹 Inertie
    inertia = 0.6
    if prev_hair_field is not None:
        hair_delta_field = inertia * prev_hair_field + (1 - inertia) * hair_delta_field

    # 🔹 Masque + falloff
    mask_hair_expand = mask_hair.permute(0,2,3,1)
    yy = torch.linspace(0,1,H,device=device).view(1,H,1,1)
    falloff = yy**2.8
    mask_hair_expand = mask_hair_expand * falloff

    # 🔹 Micro mouvement vertical régulier
    vertical_wave = 0.01 * torch.sin(t_wind1 + grid[...,1:2] * 0.05) * strength
    hair_delta_field[...,1:2] += vertical_wave

    # 🔹 Micro noise
    hair_delta_field += 0.002 * (torch.rand_like(hair_delta_field) - 0.5) * strength

    # 🔹 Application
    grid_hair = grid + hair_delta_field * mask_hair_expand
    grid_hair += wind_delta * mask_hair_expand
    grid_hair += gravity_delta * mask_hair_expand

    # 🔹 Normalisation
    grid_hair[...,0] = 2.0 * grid_hair[...,0] / (W-1) - 1.0
    grid_hair[...,1] = 2.0 * grid_hair[...,1] / (H-1) - 1.0

    # 🔹 Sampling
    latents_out = F.grid_sample(latents, grid_hair, align_corners=True)

    # 🔹 Debug
    if debug:
        print(f"[DEBUG] Hair motion Vent applied with strength={strength:.2f}")
    if frame_counter % 2 == 0:
        if debug and debug_dir is not None:
            debug_save_mask_and_wind(
                mask=mask_hair,
                wind_delta=wind_delta,
                H=H,
                W=W,
                debug_dir=debug_dir,
                frame_counter=frame_counter
            )

    return latents_out, hair_delta_field
#------------ version cinema -----------------
def apply_hair_motion_cinema(
    latents,
    mask_hair,
    grid,
    H,
    W,
    frame_counter,
    device,
    delta_px=None,
    prev_hair_field=None,
    strength: float = 1.0,   # 🔥 NOUVEAU
    debug=False,
    debug_dir=None
):
    B = latents.shape[0]

    # 🔹 Clamp sécurité
    strength = max(0.0, min(strength, 5.0))

    # -------------------- Temps --------------------
    t = frame_counter
    t_wind1 = torch.tensor(t / 15.0, device=device)
    t_wind2 = torch.tensor(t / 60.0, device=device)

    # -------------------- Multi-échelle bruit --------------------
    def multi_noise(grid, t, scales=[0.05,0.15,0.3], weights=[1.0,0.5,0.25]):
        val = 0
        for s,w in zip(scales, weights):
            val += w * smooth_noise(grid, t, scale=s)
        return val

    noise_x = multi_noise(grid, t)
    noise_y = multi_noise(grid, t + 123, scales=[0.08,0.2,0.4], weights=[1.0,0.5,0.25])

    # -------------------- Champ delta de base --------------------
    hair_delta_field = torch.zeros_like(grid)
    hair_delta_field[...,0] = 0.03 * noise_x
    hair_delta_field[...,1] = 0.05 * noise_y

    # -------------------- Vent dynamique --------------------
    wind_dir = torch.tensor([[1.0,0.2],[0.3,0.1]], device=device).mean(dim=0).view(1,1,1,2)
    wind_strength = 0.02 + 0.01 * torch.sin(t_wind1) + 0.005 * torch.sin(t_wind2)
    wind_delta = wind_dir * wind_strength

    # -------------------- Gravité --------------------
    gravity_delta = torch.zeros_like(grid)
    gravity_delta[...,1] = 0.004

    # -------------------- Influence du torse --------------------
    if delta_px is not None:
        speed = torch.norm(delta_px, dim=-1, keepdim=True)
        hair_delta_field *= (1.0 + 2.5 * speed)
        wind_delta *= (1.0 + 1.5 * speed)
        gravity_delta *= (1.0 + 0.5 * speed)

    # -------------------- Inertie --------------------
    inertia = 0.85
    if prev_hair_field is not None:
        hair_delta_field = inertia * prev_hair_field + (1 - inertia) * hair_delta_field

    # -------------------- Masque + falloff --------------------
    mask_hair_expand = mask_hair.permute(0,2,3,1)
    yy = torch.linspace(0,1,H,device=device).view(1,H,1,1)
    smooth_falloff = yy**2 * (3 - 2*yy)
    mask_hair_expand = mask_hair_expand * smooth_falloff

    # -------------------- Micro-souplesse --------------------
    spring = 0.003 * torch.sin(t*0.5 + grid[...,1:2]*3.0)
    hair_delta_field[...,1:2] += spring

    # -------------------- Micro noise --------------------
    micro_noise = 0.001 * (torch.rand_like(hair_delta_field)-0.5)
    hair_delta_field += micro_noise

    # =========================
    # 🔥 APPLICATION DU STRENGTH (AU BON ENDROIT)
    # =========================
    hair_delta_field = hair_delta_field * strength
    wind_delta = wind_delta * strength
    gravity_delta = gravity_delta * strength

    # -------------------- Application --------------------
    grid_hair = grid + hair_delta_field * mask_hair_expand
    grid_hair += wind_delta * mask_hair_expand
    grid_hair += gravity_delta * mask_hair_expand

    # -------------------- Normalisation --------------------
    grid_hair[...,0] = 2.0 * grid_hair[...,0] / (W-1) - 1.0
    grid_hair[...,1] = 2.0 * grid_hair[...,1] / (H-1) - 1.0

    # -------------------- Sampling --------------------
    latents_out = F.grid_sample(latents, grid_hair, align_corners=True)

    if debug:
        print(f"[DEBUG] Hair motion cinema applied | strength={strength:.2f}")

    if frame_counter % 2 == 0:
        if debug and debug_dir is not None:
            debug_save_mask_and_wind(
                mask=mask_hair,
                wind_delta=wind_delta,
                H=H,
                W=W,
                debug_dir=debug_dir,
                frame_counter=frame_counter
            )

    return latents_out, hair_delta_field
