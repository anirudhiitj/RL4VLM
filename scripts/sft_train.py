#!/usr/bin/env python
"""
SFT for NumberLine - Single GPU, no DeepSpeed.
Uses HuggingFace Trainer directly on GPU 7.
Paper: 1 epoch, lr=2e-5, cosine, bf16, batch=16 effective.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
sys.path.insert(0, "/mnt/raid/rl_gaming/RL4VLM/LLaVA")

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
