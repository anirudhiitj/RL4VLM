"""Evaluate SFT checkpoint on NumberLine environment (standalone, no RL)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'VLM_PPO'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gym-cards'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaVA'))

from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()

import torch
import numpy as np
import gymnasium as gym
import gym_cards
import json
import re
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from transformers import AutoTokenizer

from a2c_ppo_acktr.rl_utils import get_prompt
from a2c_ppo_acktr.llava_interface import init_pretrained_model


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base = LlavaMistralForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    base, tokenizer = init_pretrained_model(base, tokenizer)
    image_processor = base.get_vision_tower().image_processor
    base = base.cuda().eval()
    return base, tokenizer, image_processor


def generate_action(model, tokenizer, image_processor, obs, conv_mode="mistral_instruct"):
    qs = get_prompt("gym_cards/NumberLine-v0", False)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    input_ids[input_ids == 0] = 259

    image = Image.fromarray(obs)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16).cuda()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=256,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Parse action from the generated text
    action_idx = text.find('"action"')
    if action_idx == -1:
        action_idx = text.find("'action'")
    if action_idx >= 0:
        action_str = text[action_idx:]
        if "-" in action_str:
            return 0, text  # Move left
        elif "+" in action_str:
            return 1, text  # Move right
    # Fallback: random
    return np.random.randint(0, 2), text


def evaluate(model_path, num_episodes=200):
    print(f"Loading model from {model_path}...")
    model, tokenizer, image_processor = load_model(model_path)
    print("Model loaded. Starting evaluation...")

    env = gym.make("gym_cards/NumberLine-v0")

    successes = 0
    total_rewards = []

    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            action, text = generate_action(model, tokenizer, image_processor, obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        if episode_reward > 0:
            successes += 1

        if ep < 5 or (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}: reward={episode_reward:.1f}, success={'YES' if episode_reward > 0 else 'NO'}")
            if ep < 5:
                print(f"    Last response: {text[:200]}...")

    success_rate = successes / num_episodes
    mean_reward = np.mean(total_rewards)
    print(f"\n{'='*50}")
    print(f"SFT Evaluation Results ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Success Rate: {success_rate*100:.1f}% ({successes}/{num_episodes})")
    print(f"Mean Reward:  {mean_reward:.3f}")
    print(f"{'='*50}")
    return success_rate


if __name__ == "__main__":
    model_path = "/mnt/raid/rl_gaming/RL4VLM/checkpoints/sft_numberline"
    num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    evaluate(model_path, num_episodes)
