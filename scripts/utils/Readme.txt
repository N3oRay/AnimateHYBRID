All fonction utils

Sample run:
python -m scripts.n3rHYBRID10 \
                         --pretrained-model-path "/mnt/62G/huggingface/miniSD" \
                         --config configs/prompts/2_animate/128.yaml \
                         --device cuda
📌 Paramètres : fps=12, frames/image=12, steps=12, seed=1234
⏱ Durée totale estimée : 5.0s
🔄 Chargement tokenizer et text_encoder
✅ Text encoder OK
✅ State dict VAE chargé, clés: ['decoder.conv_in.bias', 'decoder.conv_in.weight', 'decoder.conv_out.bias', 'decoder.conv_out.weight', 'decoder.mid.attn_1.k.bias']
🔎 Latent shape: torch.Size([1, 4, 32, 32])
🔎 Decoded shape: torch.Size([1, 3, 256, 256])
✅ Test VAE 256 OK
✅ VAE OK
✅ UNet + Scheduler OK
✅ Motion module (Python) loaded and instantiated: scripts/modules/motion_module_tiny.py
✅ Image chargée : input/image_128x0.png
✅ Image chargée : input/image_128x1.png
✅ Image chargée : input/image_128x2.png
✅ Image chargée : input/image_128x3.png
✅ Image chargée : input/image_128x4.png
✅ Génération terminée.



python -m scripts.n3rHYBRID11 \
                         --pretrained-model-path "/mnt/62G/huggingface/miniSD" \
                         --config configs/prompts/2_animate/256_quality.yaml \
                         --device cuda \
                         --vae-offload \
                         --fp16

