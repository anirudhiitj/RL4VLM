"""
GRPO (Group Relative Policy Optimization) training script for Points24.

Key differences from PPO:
- No value/critic network — advantages computed from group-relative rewards
- For each state, generates G outputs, gets rewards, computes group advantage
- PPO-clipped surrogate with group-relative advantages
- Optional KL penalty against SFT reference policy

Pipeline: SFT (full-weight, 1 epoch) → GRPO (LoRA)
"""
from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()
print("using xformers")

import copy
import glob
import os
import sys
import time
import json
import csv
import logging
from datetime import datetime
from collections import deque

import gymnasium as gym
import gym_cards
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import utils, rl_utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage_grpo import GRPOStorage
from a2c_ppo_acktr.algo.grpo import GRPO
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

import math
import random
from functools import partial
from typing import List, Optional
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoImageProcessor
import transformers

from tqdm import tqdm

import accelerate
from accelerate.state import AcceleratorState

import warnings
warnings.filterwarnings("ignore")


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    logger = logging.getLogger("grpo_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_fmt = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_config(args, log_dir):
    config_path = os.path.join(log_dir, "config.json")
    config_dict = vars(args)
    serializable = {}
    for k, v in config_dict.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, ValueError):
            serializable[k] = str(v)
    serializable["start_time"] = datetime.now().isoformat()
    serializable["method"] = "grpo"
    with open(config_path, 'w') as f:
        json.dump(serializable, f, indent=2)


class StepLogger:
    def __init__(self, log_dir):
        self.filepath = os.path.join(log_dir, "step_details.jsonl")
        self.f = open(self.filepath, 'a')

    def log_step(self, iteration, state_idx, group_idx, info, text_action, action, reward, done):
        entry = {
            "iteration": iteration,
            "state_idx": state_idx,
            "group_idx": group_idx,
            "observation_info": str(info),
            "cot_output": text_action,
            "action": int(action) if isinstance(action, (int, np.integer)) else action.item() if hasattr(action, 'item') else str(action),
            "reward": float(reward),
            "done": bool(done),
            "timestamp": datetime.now().isoformat()
        }
        self.f.write(json.dumps(entry) + '\n')

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()


class IterationLogger:
    def __init__(self, log_dir):
        self.filepath = os.path.join(log_dir, "iteration_summary.csv")
        self.fieldnames = [
            "iteration", "timesteps", "fps", "elapsed_time_min",
            "mean_reward", "median_reward", "min_reward", "max_reward",
            "success_rate", "action_loss", "kl_loss",
            "learning_rate", "group_size",
            "group_reward_mean", "group_reward_std",
            "best_in_group_rate",
            "timestamp"
        ]
        self.f = open(self.filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.f, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.f.flush()

    def log_iteration(self, row_dict):
        row_dict["timestamp"] = datetime.now().isoformat()
        self.writer.writerow(row_dict)
        self.f.flush()

    def close(self):
        self.f.close()


def save_checkpoint(actor_critic, optimizer, lr_scheduler, iteration, log_dir, logger):
    ckpt_dir = os.path.join(log_dir, "checkpoints", f"iter_{iteration:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    try:
        model = actor_critic.module if hasattr(actor_critic, 'module') else actor_critic
        value_model = model.value_model
        base_model = value_model.base if hasattr(value_model, 'base') else value_model
        if hasattr(base_model, 'save_pretrained'):
            base_model.save_pretrained(os.path.join(ckpt_dir, "lora_adapters"))
            logger.info(f"  Saved LoRA adapters to {ckpt_dir}/lora_adapters")
        if hasattr(value_model, 'value_head'):
            torch.save(value_model.value_head.state_dict(), os.path.join(ckpt_dir, "value_head.pt"))
        torch.save({
            'iteration': iteration,
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        }, os.path.join(ckpt_dir, "training_state.pt"))
        logger.info(f"  Checkpoint saved at iteration {iteration}")
    except Exception as e:
        logger.info(f"  Checkpoint save failed: {e}")


def generate_group_outputs(actor_critic, obs, tokenizer, INPUT_IDS, args, group_size):
    """
    Generate G different outputs for the same observation using temperature sampling.

    Returns:
        group_output_ids: (G, max_output_len) tensor of padded output ids
        group_log_probs: (G,) tensor of log probs for each generation
        group_text_actions: list of G decoded text strings
        group_actions: (G, 1) tensor of parsed discrete actions
    """
    group_output_ids = []
    group_log_probs = []
    group_text_actions = []
    group_actions = []
    group_action_tokens_log_probs = []

    projection_f = actor_critic.module.projection_f if hasattr(actor_critic, 'module') else actor_critic.projection_f

    for g in range(group_size):
        with torch.no_grad():
            # act() returns (value, output_ids, action, action_log_prob, action_tokens_log_prob)
            # where action is already the projected discrete action (integer tensor)
            value, output_id, action, action_log_prob, action_tokens_log_prob = actor_critic.act(
                obs, INPUT_IDS=INPUT_IDS)

        group_output_ids.append(output_id)
        group_log_probs.append(action_log_prob)
        # Decode output_ids to get the text for logging
        text = tokenizer.decode(list(filter(lambda num: num != 0, output_id[0].tolist())))
        group_text_actions.append(text)
        # action is already the discrete action from act() — no need to call projection_f again
        group_actions.append(action)
        group_action_tokens_log_probs.append(action_tokens_log_prob)

    # Stack
    group_output_ids = torch.cat(group_output_ids, dim=0)  # (G, max_output_len)
    group_log_probs = torch.cat(group_log_probs, dim=0)    # (G,)
    group_actions_tensor = torch.cat(group_actions, dim=0)  # (G, 1)

    return group_output_ids, group_log_probs, group_text_actions, group_actions_tensor, group_action_tokens_log_probs


def evaluate_group_in_env(envs, group_actions, group_text_actions, args):
    """
    Evaluate each of G actions in the environment by stepping the underlying env directly
    and recording rewards. We save/restore env state for each group member so all G actions
    are evaluated from the same state.

    Returns:
        group_rewards: (G,) tensor of rewards
        group_dones: (G,) list of done flags
        group_infos: list of G info dicts
    """
    raw_env = envs.envs[0] if hasattr(envs, 'envs') else envs.venv.envs[0]
    underlying = raw_env
    while hasattr(underlying, 'env'):
        underlying = underlying.env

    # Save env state (deep copy mutable lists)
    saved_cards_num = list(underlying.cards_num)
    saved_cards = list(underlying.cards)
    saved_formula = list(underlying.formula)
    saved_used_cards = list(underlying.used_cards)
    saved_card_imgs = list(underlying.card_imgs)

    group_rewards = []
    group_dones = []
    group_infos = []

    for g in range(len(group_actions)):
        # Restore env state before each evaluation
        underlying.cards_num = list(saved_cards_num)
        underlying.cards = list(saved_cards)
        underlying.formula = list(saved_formula)
        underlying.used_cards = list(saved_used_cards)
        underlying.card_imgs = list(saved_card_imgs)

        # Step the underlying env directly (not through wrappers)
        action_int = group_actions[g].item() if group_actions[g].dim() == 0 else group_actions[g][0].item()
        obs_np, reward, terminated, truncated, info = underlying.step(action_int)
        done = terminated or truncated

        group_rewards.append(float(reward))
        group_dones.append(done)
        group_infos.append(info)

    # Restore env state so the episode can continue from the original state
    underlying.cards_num = list(saved_cards_num)
    underlying.cards = list(saved_cards)
    underlying.formula = list(saved_formula)
    underlying.used_cards = list(saved_used_cards)
    underlying.card_imgs = list(saved_card_imgs)

    return torch.tensor(group_rewards, dtype=torch.float32), group_dones, group_infos


def main():
    args = get_args()

    # GRPO-specific args
    group_size = getattr(args, 'group_size', 8)
    grpo_kl_coef = getattr(args, 'grpo_kl_coef', 0.01)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = getattr(args, 'log_dir', None) or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rl_logs", f"grpo_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logging(log_dir)
    step_logger = StepLogger(log_dir)
    iter_logger = IterationLogger(log_dir)

    logger.info("=" * 80)
    logger.info("  GRPO (Group Relative Policy Optimization) Training for Points24")
    logger.info(f"  Log directory: {log_dir}")
    logger.info(f"  Group size (G): {group_size}")
    logger.info(f"  KL coefficient: {grpo_kl_coef}")
    logger.info(f"  Start time: {datetime.now().isoformat()}")
    logger.info("=" * 80)

    save_config(args, log_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device
    model_device = device

    # Load model
    model_path = args.model_path
    cache_dir = args.cache_dir
    logger.info(f"Model path: {model_path}")

    if "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if args.q8:
            base = LlavaMistralForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
        elif args.q4:
            q4_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
            base = LlavaMistralForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
        else:
            if 'mistral' in model_path.lower():
                base = LlavaMistralForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)

    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        base.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    base.config.max_length = 1024
    base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter=args.pretrain_mm_adapter)
    image_processor = base.get_vision_tower().image_processor

    # Apply LoRA
    base_lora_config = LoraConfig(
        r=128, lora_alpha=256,
        target_modules=find_all_linear_names(base, args.train_vision),
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    if args.use_lora:
        base = get_peft_model(base, base_lora_config)

    # GRPO: still use VLMValue for the base model, but not for advantage computation
    value_model = VLMValue(base)
    value_model = value_model.to(model_device)

    # Create env
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, None, device, False, 1)

    # Build prompt
    obs = envs.reset()
    infos = None
    qs = get_prompt(args.env_name, args.action_only_prompt, infos)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    logger.info(f"Prompt:\n{prompt}")

    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259

    projection_f = partial(text_projection, env_name=args.env_name)

    actor_critic = VLMPolicy(tokenizer=tokenizer, image_processor=image_processor,
                             value_model=value_model, projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS, args=args)

    # Only optimize LoRA params (no value head needed for GRPO advantages)
    optimizer = optim.Adam(actor_critic.value_model.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)

    # GRPO agent
    grpo_agent = GRPO(
        actor_critic=actor_critic,
        optimizer=optimizer,
        accelerator=accelerator,
        clip_param=args.clip_param,
        grpo_epoch=args.ppo_epoch,
        mini_batch_size=args.mini_batch_size,
        group_size=group_size,
        kl_coef=grpo_kl_coef,
        max_grad_norm=args.max_grad_norm,
        ref_model=None,  # No reference model KL penalty by default
    )

    # Tracking
    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    all_group_rewards = []
    start = time.time()

    # Each iteration: collect num_steps states, each with G group samples
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    total_timesteps = 0

    logger.info(f"\n{'='*80}")
    logger.info(f"  GRPO Configuration")
    logger.info(f"  Group size (G): {group_size}")
    logger.info(f"  KL coef: {grpo_kl_coef}")
    logger.info(f"  Num updates: {num_updates}")
    logger.info(f"  States per iteration: {args.num_steps}")
    logger.info(f"  GRPO epochs: {args.ppo_epoch}")
    logger.info(f"  Init LR: {args.init_lr}, End LR: {args.end_lr}")
    logger.info(f"  Use LoRA: {args.use_lora}")
    logger.info(f"{'='*80}\n")

    running_episode_rewards = torch.zeros(args.num_processes).flatten()

    pbar = tqdm(range(num_updates), desc="GRPO Training", ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    # Initial env reset + prompt
    obs = envs.reset()
    infos = None
    qs = get_prompt(args.env_name, args.action_only_prompt, infos)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    for j in pbar:
        # GRPO storage for this iteration
        grpo_storage = GRPOStorage(
            num_states=args.num_steps,
            group_size=group_size,
            obs_shape=envs.observation_space.shape,
            max_output_len=2 * args.max_new_tokens
        )
        grpo_storage.to(device)

        iter_group_rewards = []
        iter_best_in_group = 0

        step_pbar = tqdm(range(args.num_steps), desc=f"  Iter {j} group sampling",
                        ncols=100, leave=False,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for step in step_pbar:
            # Build prompt from current env state
            INPUT_IDS_step = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            INPUT_IDS_step[INPUT_IDS_step == 0] = 259

            # Generate G outputs from the current state
            group_output_ids, group_log_probs, group_text_actions, group_actions, _ = \
                generate_group_outputs(actor_critic, obs, tokenizer, INPUT_IDS_step, args, group_size)

            # Evaluate each action in env (save/restore underlying state)
            group_rewards, group_dones, group_infos = evaluate_group_in_env(
                envs, group_actions, group_text_actions, args)

            # Store in GRPO storage
            grpo_storage.insert(obs, group_output_ids, group_log_probs, group_rewards)

            # Track metrics
            iter_group_rewards.append(group_rewards.numpy())
            best_reward = group_rewards.max().item()
            if best_reward > 0:
                iter_best_in_group += 1

            # Log step details
            for g in range(group_size):
                step_logger.log_step(
                    iteration=j, state_idx=step, group_idx=g,
                    info=group_infos[g] if g < len(group_infos) else {},
                    text_action=group_text_actions[g],
                    action=group_actions[g].item() if group_actions[g].dim() == 0 else group_actions[g][0].item(),
                    reward=group_rewards[g].item(),
                    done=group_dones[g]
                )

            # Advance the episode using the best action from the group
            best_idx = group_rewards.argmax().item()
            best_action = group_actions[best_idx:best_idx+1]
            obs, reward, done, infos = envs.step(best_action)

            actual_reward = reward.flatten()[0].item() if isinstance(reward, torch.Tensor) else float(reward)
            running_episode_rewards += actual_reward

            # Update prompt for next step
            qs = get_prompt(args.env_name, args.action_only_prompt, infos)
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            done_flag = done[0] if hasattr(done, '__getitem__') else done
            if done_flag:
                episode_rewards.append(running_episode_rewards[0].item())
                episode_success_rate.append(1 if running_episode_rewards[0] > 0 else 0)
                running_episode_rewards[0] = 0
                # Env auto-resets on done; update prompt for new episode
                qs = get_prompt(args.env_name, args.action_only_prompt, infos)
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

            total_timesteps += group_size  # G forward passes per state

        step_pbar.close()

        # GRPO update
        action_loss, kl_loss = grpo_agent.update(grpo_storage)
        lr_scheduler.step()

        current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else args.init_lr

        # Metrics
        all_iter_rewards = np.concatenate(iter_group_rewards) if iter_group_rewards else np.array([0.0])
        group_reward_mean = all_iter_rewards.mean()
        group_reward_std = all_iter_rewards.std()
        best_in_group_rate = iter_best_in_group / max(args.num_steps, 1)

        if len(episode_rewards) > 1:
            end = time.time()
            elapsed_min = (end - start) / 60.0
            fps = int(total_timesteps / (end - start))

            mean_reward = np.mean(episode_rewards)
            success_rate = np.mean(episode_success_rate)

            logger.info(f"\n{'='*60}")
            logger.info(f"  GRPO ITERATION {j}/{num_updates}")
            logger.info(f"  Timesteps: {total_timesteps} | FPS: {fps} | Elapsed: {elapsed_min:.1f} min")
            logger.info(f"  Reward mean: {mean_reward:.3f} | Success: {success_rate*100:.1f}%")
            logger.info(f"  Action Loss: {action_loss:.6f} | KL Loss: {kl_loss:.6f}")
            logger.info(f"  Group reward mean: {group_reward_mean:.3f} | std: {group_reward_std:.3f}")
            logger.info(f"  Best-in-group rate: {best_in_group_rate:.3f}")
            logger.info(f"  LR: {current_lr:.2e}")

            pbar.set_postfix({
                'succ': f'{success_rate*100:.1f}%',
                'rew': f'{mean_reward:.2f}',
                'a_loss': f'{action_loss:.4f}',
            })

            iter_logger.log_iteration({
                "iteration": j,
                "timesteps": total_timesteps,
                "fps": fps,
                "elapsed_time_min": round(elapsed_min, 2),
                "mean_reward": round(mean_reward, 4),
                "median_reward": round(np.median(episode_rewards), 4),
                "min_reward": round(np.min(episode_rewards), 4),
                "max_reward": round(np.max(episode_rewards), 4),
                "success_rate": round(success_rate, 4),
                "action_loss": round(action_loss, 6),
                "kl_loss": round(kl_loss, 6),
                "learning_rate": current_lr,
                "group_size": group_size,
                "group_reward_mean": round(group_reward_mean, 4),
                "group_reward_std": round(group_reward_std, 4),
                "best_in_group_rate": round(best_in_group_rate, 4),
            })

        step_logger.flush()

        # Checkpoint
        if (j + 1) % 5 == 0 or j == num_updates - 1:
            logger.info(f"\n  Saving checkpoint at iteration {j}...")
            save_checkpoint(actor_critic, optimizer, lr_scheduler, j, log_dir, logger)

    pbar.close()

    total_time = (time.time() - start) / 3600.0
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"  GRPO TRAINING COMPLETE!")
    logger.info(f"  Total time: {total_time:.2f} hours")
    logger.info(f"  Final success rate: {np.mean(episode_success_rate)*100:.1f}%")
    logger.info(f"  Logs saved to: {log_dir}")
    logger.info("=" * 80)

    step_logger.close()
    iter_logger.close()


if __name__ == "__main__":
    main()
