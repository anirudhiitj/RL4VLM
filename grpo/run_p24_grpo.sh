#!/bin/bash
# GRPO (Group Relative Policy Optimization) training for Points24
# No value network — uses group-relative advantages from G=8 samples per state

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="0" accelerate launch \
    --config_file config_zero2.yaml \
    --main_process_port 29383 \
    ../main_grpo.py \
    --env-name gym_cards/Points24-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 15000 \
    --num-steps 128 \
    --grad-accum-steps 16 \
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
    --group-size 8 \
    --grpo-kl-coef 0.01 \
    --log-dir ../rl_logs/points24_grpo
