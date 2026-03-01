#!/bin/bash
set -e

MODEL_PATH="/mnt/62G/huggingface/miniSD"
DEVICE="cuda"

CONFIGS128=("configs/prompts/0_animate/128.yaml")
CONFIGS128P=("configs/prompts/2_animate/128p.yaml")
CONFIGS256P=("configs/prompts/1_animate/256p.yaml")
CONFIGS256=("configs/prompts/2_animate/256.yaml")
CONFIGS512=("configs/prompts/2_animate/512.yaml")

SCRIPTS1=("n3rspeed") # 128 256 512 ok
SCRIPTS2=("n3rcreative") # 128 256 512 ok
SCRIPTS3=("n3rHYBRID21") # 128 256 512 ok
SCRIPTS4=("n3rHYBRID22") # 128 256 512 ok
SCRIPTS5=("n3rHYBRID26") #128 ok 256 ko
SCRIPTS6=("n3rHYBRID14") #512 KO


run_batch () {
    local -n CONFIG_ARRAY=$1
    local -n SCRIPT_ARRAY=$2

    for CONFIG_PATH in "${CONFIG_ARRAY[@]}"; do
        echo "========================================"
        echo "Config : $CONFIG_PATH"
        echo "========================================"

        for script in "${SCRIPT_ARRAY[@]}"; do
            echo "Lancement : $script"

            python -m scripts.$script \
                --pretrained-model-path "$MODEL_PATH" \
                --config "$CONFIG_PATH" \
                --device "$DEVICE" \
                --vae-offload \
                --fp16

            echo "$script terminé."

            # Nettoyage CUDA
            python - <<EOF
import torch
torch.cuda.empty_cache()
EOF

            nvidia-smi
            echo "========================================"
        done
    done
}


# CONFIGS2 → SCRIPTS2 256 OK
#run_batch CONFIGS2 SCRIPTS1

# CONFIGS2 → SCRIPTS2 512 OK
run_batch CONFIGS512 SCRIPTS1

# CONFIGS2 → SCRIPTS2 256
#run_batch CONFIGS256 SCRIPTS2

# CONFIGS3 → SCRIPTS1 (le "sinon") 512
#run_batch CONFIGS1 SCRIPTS3

echo "Tous les scripts ont été exécutés."
