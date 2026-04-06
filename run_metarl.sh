#!/bin/bash
# Meta-RL Training - GPU 7
set -e

export CONDA_DEFAULT_ENV=rl4vlm_clean
eval "$(conda shell.bash hook)"
conda activate rl4vlm_clean

REPO="/mnt/raid/rl_gaming/RL4VLM"
export PYTHONPATH="${REPO}/gym-cards:${REPO}/VLM_PPO:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="7"

cd "${REPO}/VLM_PPO/scripts"

echo "=========================================="
echo "  Meta-RL (Reptile+PPO) Training - Points24"
echo "  GPU: 7 | Env: rl4vlm_clean"
echo "  Repo: ${REPO}"
echo "  Log dir: ${REPO}/rl_logs/points24_metarl"
echo "  Start: $(date)"
echo "=========================================="

accelerate launch \
    --config_file config_zero2.yaml \
    --main_process_port 29385 \
    ../main_metarl.py \
    --env-name gym_cards/Points24-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 15000 \
    --num-steps 512 \
    --grad-accum-steps 64 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path /mnt/raid/rl_gaming/RL4VLM/checkpoints/points24_sft_1epoch \
    --use-lora \
    --train-vision all \
    --meta-lr 1e-4 \
    --inner-steps 5 \
    --meta-batch-size 3 \
    --meta-strategy reptile \
    --log-dir /mnt/raid/rl_gaming/RL4VLM/rl_logs/points24_metarl

echo "Meta-RL training complete at $(date)"
