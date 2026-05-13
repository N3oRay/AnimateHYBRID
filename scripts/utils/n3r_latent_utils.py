import time
import uuid
from datetime import datetime
from pathlib import Path

import torch

#mini node custom ComfyUI :
def load_latent(path, map_location="cpu"):
    data = torch.load(path, map_location=map_location)

    return {
        "samples": data["samples"],
        "metadata": data.get("metadata", {}),
    }


def export_latents_file(latents, frame_counter, output_dir="exports"):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"latents_{frame_counter:05d}.pt"

    torch.save(
        {
            "latents": latents.detach().float().cpu(),
            "frame": frame_counter,
            "shape": list(latents.shape),
        },
        save_path
    )

    print(f"💾 Latents exported: {save_path}")



"""

{
    "samples": latents,

    # dimensions image réelles
    "width": 896,
    "height": 1280,

    # infos latent
    "latent_shape": [1,4,160,112],
    "latent_scale_factor": 8,

    # modèle
    "model": "SDXL",
    "vae": "sdxl_vae",
    "dtype": "float32",

    # génération
    "seed": seed,
    "steps": steps,
    "cfg": cfg,
    "sampler": sampler_name,
    "scheduler": scheduler_name,

    # animation
    "frame": frame_counter,
    "fps": 24,

    # ton moteur
    "engine": "AnimateHybrid",
    "motion_noise": float(motion_noise),
    "temporal_strength": float(temporal_strength),

    # debug
    "timestamp": time.time(),
}
"""



def generate_sequence_id(prefix="ah"):
    date_str = datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:8]

    return f"{prefix}_{date_str}_{short_uuid}"

def build_latent_metadata(
    latents,
    frame_counter=0,
    sequence_id="AnimateHybrid",
    scale=8,
    fps=24,

    # pipeline
    denoise=True,
    temporal_consistency=True,
    style_injection=True,
    appearance=True,
    creative=True,

    # render
    sharpen_mode="both",
    gamma_boost=1.0,

    # training
    train=False,

    # temporal
    ema_prev_latents=None,

    # misc
    engine="AnimateHybrid",
):
    B, C, H, W = latents.shape

    return {

        # ------------------------------------------------
        # latent info
        # ------------------------------------------------
        "latent_shape": [B, C, H, W],
        "channels": C,

        # image size
        "width": W * scale,
        "height": H * scale,
        "latent_scale_factor": scale,

        "latent_mean": float(latents.mean().item()),
        "latent_std": float(latents.std().item()),
        "is_video": True,


        "format_version": 1,

        # frame
        "frame_index": frame_counter,
        "sequence_id": sequence_id,
        "fps": fps,

        # dtype/device
        "dtype": str(latents.dtype),
        "device": str(latents.device),

        # ------------------------------------------------
        # pipeline modules
        # ------------------------------------------------
        "denoise": denoise,
        "temporal_consistency": temporal_consistency,
        "style_injection": style_injection,
        "appearance": appearance,
        "creative": creative,

        # ------------------------------------------------
        # render settings
        # ------------------------------------------------
        "sharpen_mode": sharpen_mode,
        "gamma_boost": gamma_boost,

        # ------------------------------------------------
        # temporal
        # ------------------------------------------------
        "has_ema": ema_prev_latents is not None,

        # ------------------------------------------------
        # train/debug
        # ------------------------------------------------
        "train": train,

        # ------------------------------------------------
        # engine
        # ------------------------------------------------
        "engine": engine,

        "vae_scaling_factor": 0.18215,
        "base_model": "SD1.5",

        # ------------------------------------------------
        # timestamp
        # ------------------------------------------------
        "timestamp": time.time(),
    }

def save_latents_for_comfy(latents, frame_counter, output_dir="exports"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "samples": latents.detach().float().cpu(),
    }

    path = output_dir / f"latent_{frame_counter:05d}.latent.pt"

    torch.save(data, path)

    print(f"💾 Comfy Latent exported: {path}")

def export_latents_for_comfy_meta(
    latents,
    frame_counter,
    output_dir="exports",
    metadata=None,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "samples": latents.detach().float().cpu(),
        "metadata": metadata or {},
    }

    path = output_dir / f"latent_{frame_counter:05d}.latent.pt"

    torch.save(data, path)

    print(f"💾 Latents exported: {path}")
