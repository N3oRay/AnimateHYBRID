n3rRealControl.py - Description et Options

Le script n3rRealControl.py permet de générer des vidéos et images animées à partir de modèles de diffusion avec des ajustements avancés. Il offre une flexibilité maximale via des options de personnalisation pour la qualité, la vitesse, et l’utilisation de la VRAM.

Principales Options :
use_mini_gpu : Utiliser un GPU à faible VRAM pour des générations rapides (2 Go VRAM).
verbose : Activer les logs détaillés pour le débogage.
latent_injection : Contrôler l'injection de latents pour ajuster la qualité.
fps et upscale_factor : Définir la fréquence d'images et la mise à l'échelle.
steps : Nombre d'étapes pour la diffusion, affecte la qualité.
guidance_scale : Intégrer la guidance de l'algorithme pour mieux contrôler la génération.
use_n3r_model et use_n3r_pro_net : Activer des modèles pour améliorer la qualité et les détails.
use_openpose : Utiliser OpenPose pour contrôler les poses humaines.
controlnet_scale et control_strength : Ajuster l’intensité de ControlNet pour une meilleure gestion des poses.
Fonctionnalités supplémentaires :
Motion Module : Applique des animations basées sur des données de mouvement.
LoRA : Charge des modèles LoRA pour personnaliser les résultats.
VAE & Tokenizer : Gère les embeddings texte pour une meilleure cohérence avec les prompts.


python -m scripts.n3rRealControl \
                      --pretrained-model-path "/mnt/62G/huggingface/miniSD" \
                      --config "configs/prompts/0_n3r/512-c.yaml" \
                      --device "cuda" \
                      --vae-offload \
                      --fp16
