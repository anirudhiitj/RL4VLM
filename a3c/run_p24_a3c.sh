#!/bin/bash
# A3C (Asynchronous Advantage Actor-Critic) training for Points24
# Multiple workers on single GPU with shared base model
# Each worker has its own env, collects n-step rollouts

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="0" accelerate launch \
    --config_file config_zero2.yaml \
    --main_process_port 29384 \
    ../main_a3c.py \
    --env-name gym_cards/Points24-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 100 \
    --eval-num-per-episode 200 \
    --num-env-steps 15000 \
    --num-steps 256 \
    --grad-accum-steps 8 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 1 \
    --mini-batch-size 1 \
    --model-path /mnt/raid/rl_gaming/RL4VLM/checkpoints/points24_sft_1epoch \
    --use-lora \
    --train-vision all \
    --num-workers 4 \
    --a3c-n-step 5 \
    --a3c-entropy-coef 0.01 \
    --log-dir ../rl_logs/points24_a3c
