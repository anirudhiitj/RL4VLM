#!/bin/bash
# SFT for NumberLine on GPU 7 only
# Paper: 1 epoch SFT on task-specific data, lr=2e-5, cosine schedule
# Starting from liuhaotian/llava-v1.6-mistral-7b

export CUDA_VISIBLE_DEVICES=7

cd /mnt/raid/rl_gaming/RL4VLM/LLaVA

deepspeed --include localhost:0 --master_port 29510 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.6-mistral-7b \
    --version v1 \
    --data_path /mnt/raid/rl_gaming/RL4VLM/data/sft-data/numberline.json \
    --image_folder /mnt/raid/rl_gaming/RL4VLM/data/sft-data/numberline_images_test_v2 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /mnt/raid/rl_gaming/RL4VLM/checkpoints/sft_numberline \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
