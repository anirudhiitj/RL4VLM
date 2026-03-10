#!/bin/bash
# RL (PPO) training for NumberLine on GPU 7 only
# Paper hyperparameters from Table 5 / Appendix C.1:
#   init-lr=1e-5, end-lr=1e-9, lr_max_steps=25
#   num-env-steps=15000, num-steps=512, grad-accum=128
#   thought-prob-coef=0.5, ppo-epoch=4, temperature=0.2
#   max-new-tokens=256, gamma=0.9, gae-lambda=0.95
#   LoRA r=128, alpha=256, dropout=0.05
#   eval-num-per-episode=200 (deterministic task)

export CUDA_VISIBLE_DEVICES=6

cd /mnt/raid/rl_gaming/RL4VLM/VLM_PPO

TOKENIZERS_PARALLELISM=false \
PYTHONWARNINGS=ignore \
DS_LOG_LEVEL=error \
DEEPSPEED_LOG_LEVEL=error \
accelerate launch \
    --config_file /mnt/raid/rl_gaming/RL4VLM/scripts/config_zero2_single.yaml \
    --main_process_port 29509 \
    main.py \
    --env-name gym_cards/NumberLine-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 15000 \
    --num-steps 512 \
    --grad-accum-steps 128 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path /mnt/raid/rl_gaming/RL4VLM/checkpoints/sft_numberline \
    --use-lora \
    --train-vision all
