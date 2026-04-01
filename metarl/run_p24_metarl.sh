#!/bin/bash
# Meta-RL (Reptile + PPO) training for Points24
# Outer loop: Reptile meta-update across difficulty-stratified tasks
# Inner loop: K steps of PPO on each task

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="0" accelerate launch \
    --config_file config_zero2.yaml \
    --main_process_port 29382 \
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
    --log-dir ../rl_logs/points24_metarl
