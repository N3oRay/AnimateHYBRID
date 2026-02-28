# ==========================================
# scripts/n3rHYBRID13_4GB_fixed.py
# ==========================================
import torch
import imageio
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from scripts.modules.motion_module_tiny import MotionModuleTiny
from scripts.utils.n3r_utils import load_image_latent, encode_text_embeddings, generate_latents_4Go

# -------------------------
# 🔹 Config générale
# -------------------------
PRETRAINED_MODEL_PATH = "/mnt/62G/huggingface/miniSD"
DEVICE = "cuda"  # GPU
DTYPE = torch.float16

# -------------------------
# 🔹 Charger le VAE
# -------------------------
vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="vae")
vae.enable_slicing()  # réduit l’usage mémoire
vae.to(DEVICE).to(DTYPE)

# -------------------------
# 🔹 Charger le UNet
# -------------------------
unet = UNet2DConditionModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="unet")
unet.to(DEVICE).to(DTYPE)
unet.half()  # FP16

# -------------------------
# 🔹 Scheduler
# -------------------------
scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

# -------------------------
# 🔹 Motion module
# -------------------------
motion_module = MotionModuleTiny()

# -------------------------
# 🔹 Charger l'image initiale
# -------------------------
latent_frame = load_image_latent(
    "input/128x0.png",
    vae=vae,
    device=DEVICE,
    dtype=DTYPE
)

# -------------------------
# 🔹 Text embeddings
# -------------------------
# ⚠️ Il faut passer tokenizer et text_encoder
from transformers import CLIPTokenizer, CLIPTextModel

tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="text_encoder").to(DEVICE).to(DTYPE)

pos_embeds, neg_embeds = encode_text_embeddings(
    prompt="une illustration colorée",
    negative_prompt="flou",
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    device=DEVICE,
    dtype=DTYPE
)

# -------------------------
# 🔹 Génération finale
# -------------------------
final_latent = generate_latents_4Go(
    latent_frame, pos_embeds, neg_embeds,
    unet, scheduler, motion_module=motion_module,
    steps=20
)

# -------------------------
# 🔹 Décodage final avec VAE
# -------------------------
with torch.no_grad():
    image = vae.decode(final_latent).sample

image = (image.clamp(0,1) * 255).cpu().numpy().astype("uint8")

# Sauvegarde
imageio.imwrite("output/final_frame.png", image[0].transpose(1,2,0))
print("✅ Frame générée et sauvegardée dans output/final_frame.png")
