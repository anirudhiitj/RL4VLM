#!/bin/bash
# DoRA RL Training - GPU 6
set -e

export CONDA_DEFAULT_ENV=rl4vlm_clean
eval "$(conda shell.bash hook)"
conda activate rl4vlm_clean

REPO="/mnt/raid/rl_gaming/RL4VLM"
export PYTHONPATH="${REPO}/gym-cards:${REPO}/VLM_PPO:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="6"

cd "${REPO}/VLM_PPO/scripts"

echo "=========================================="
echo "  DoRA+PPO RL Training - Points24"
echo "  GPU: 6 | Env: rl4vlm_clean"
echo "  Repo: ${REPO}"
echo "  Log dir: ${REPO}/rl_logs/points24_dora"
echo "  Start: $(date)"
echo "=========================================="

accelerate launch \
    --config_file config_zero2.yaml \
    --main_process_port 29384 \
    ../main.py \
    --env-name gym_cards/Points24-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 15000 \
    --num-steps 1024 \
    --grad-accum-steps 128 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path /mnt/raid/rl_gaming/RL4VLM/checkpoints/points24_sft_1epoch \
    --use-lora \
    --use-dora \
    --train-vision all \
    --log-dir /mnt/raid/rl_gaming/RL4VLM/rl_logs/points24_dora

echo "DoRA training complete at $(date)"
