#!/bin/bash
# GRPO (Group Relative Policy Optimization) training for Points24
# GPU 6 | LoRA | SFT 1-epoch checkpoint | Group size 8
# Same config as PPO run but using GRPO algorithm

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="6"

accelerate launch \
    --config_file config_zero2.yaml \
    --main_process_port 29385 \
    ../main_grpo.py \
    --env-name gym_cards/Points24-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 15000 \
    --num-steps 800 \
    --grad-accum-steps 100 \
    --max-new-tokens 192 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 8 \
    --model-path /mnt/raid/rl_gaming/RL4VLM/checkpoints/points24_sft_1epoch \
    --use-lora \
    --train-vision all \
    --group-size 6 \
    --grpo-kl-coef 0.01 \
    --log-dir /mnt/raid/rl_gaming/RL4VLM/rl_logs/points24_grpo
