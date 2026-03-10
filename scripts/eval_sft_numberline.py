"""
Standalone evaluation of SFT checkpoint on NumberLine-v0.
Runs N episodes, reports success rate.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'VLM_PPO'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gym-cards'))

from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()

import torch
import numpy as np
import random
import json
import gymnasium as gym
import gym_cards
from tqdm import tqdm
from functools import partial

from transformers import AutoTokenizer
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token

from a2c_ppo_acktr.llava_interface.utils import init_pretrained_model
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection

NUM_EPISODES = 200
MAX_STEPS_PER_EPISODE = 30  # max_position * 2 is at most ~20
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 256
SEED = 42

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model_path = "/mnt/raid/rl_gaming/RL4VLM/checkpoints/sft_numberline"
    print(f"Loading model from {model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base = LlavaMistralForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2"
    )
    base, tokenizer = init_pretrained_model(base, tokenizer)
    base.eval()

    image_processor = base.get_vision_tower().image_processor
    device = base.device

    # Build prompt
    qs = get_prompt("gym_cards/NumberLine-v0", action_only=False)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(f"Prompt:\n{prompt}\n")

    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259

    projection_f = partial(text_projection, env_name="gym_cards/NumberLine-v0")

    # Create environment
    env = gym.make("gym_cards/NumberLine-v0")

    successes = 0
    total_rewards = []
    parse_failures = 0

    for ep in tqdm(range(NUM_EPISODES), desc="Evaluating"):
        obs, info = env.reset(seed=SEED + ep)
        episode_reward = 0
        done = False
        truncated = False

        for step_i in range(MAX_STEPS_PER_EPISODE):
            if done or truncated:
                break

            # Process observation image
            img_tensor = image_processor.preprocess(
                obs, return_tensors='pt'
            )['pixel_values'].to(device=device, dtype=base.dtype)

            # Prepare embeddings
            input_ids = INPUT_IDS.to(device)
            _, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, img_tensor
            )
            inputs_embeds = inputs_embeds.to(device=device, dtype=base.dtype)

            # Generate
            with torch.inference_mode():
                outputs = base.generate(
                    inputs_embeds=inputs_embeds,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse action
            action_tensor = projection_f([text_output])
            action_int = action_tensor.item()

            obs, reward, done, truncated, info = env.step(action_int)
            episode_reward += reward

        if episode_reward > 0:
            successes += 1
        total_rewards.append(episode_reward)

        if (ep + 1) % 50 == 0:
            print(f"  After {ep+1} episodes: success_rate={successes/(ep+1):.3f}, "
                  f"avg_reward={np.mean(total_rewards):.3f}")

    env.close()

    success_rate = successes / NUM_EPISODES
    avg_reward = np.mean(total_rewards)

    print(f"\n{'='*50}")
    print(f"SFT Evaluation Results on NumberLine-v0")
    print(f"{'='*50}")
    print(f"Episodes:     {NUM_EPISODES}")
    print(f"Successes:    {successes}")
    print(f"Success Rate: {success_rate:.4f} ({success_rate*100:.1f}%)")
    print(f"Avg Reward:   {avg_reward:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
