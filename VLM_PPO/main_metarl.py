"""
Meta-RL training script for Points24 using Reptile + PPO.

Outer loop: Reptile meta-update across difficulty-stratified tasks.
Inner loop: K steps of PPO on each task.

Pipeline: SFT (full-weight, 1 epoch) → Meta-RL (LoRA + Reptile-PPO)
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

from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model
from a2c_ppo_acktr.algo.meta_ppo import MetaPPO
from a2c_ppo_acktr.task_sampler import TaskSampler

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
    logger = logging.getLogger("metarl_training")
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
    serializable["method"] = "meta-rl-reptile-ppo"
    with open(config_path, 'w') as f:
        json.dump(serializable, f, indent=2)


class StepLogger:
    def __init__(self, log_dir):
        self.filepath = os.path.join(log_dir, "step_details.jsonl")
        self.f = open(self.filepath, 'a')

    def log_step(self, meta_iter, task_tier, inner_step, step, info, text_action, action, reward, done):
        entry = {
            "meta_iteration": meta_iter,
            "task_tier": task_tier,
            "inner_step": inner_step,
            "step": step,
            "observation_info": str(info),
            "cot_output": text_action,
            "action": int(action) if isinstance(action, (int, np.integer)) else action.item() if hasattr(action, 'item') else str(action),
            "reward": float(reward) if isinstance(reward, (int, float)) else float(reward.item()) if hasattr(reward, 'item') else float(reward),
            "done": bool(done) if isinstance(done, (bool, np.bool_)) else bool(done),
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
            "meta_iteration", "timesteps", "fps", "elapsed_time_min",
            "mean_reward", "median_reward", "min_reward", "max_reward",
            "success_rate", "meta_value_loss", "meta_action_loss",
            "learning_rate", "meta_lr",
            "task_tier_distribution",
            "reward_rollout_mean", "reward_rollout_std",
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
    ckpt_dir = os.path.join(log_dir, "checkpoints", f"meta_iter_{iteration:04d}")
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
            'meta_iteration': iteration,
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        }, os.path.join(ckpt_dir, "training_state.pt"))
        logger.info(f"  Checkpoint saved at meta-iteration {iteration}")
    except Exception as e:
        logger.info(f"  Checkpoint save failed: {e}")


def collect_task_rollout(actor_critic, envs, rollouts, args, tokenizer, prompt_template_fn,
                         task_cards, step_logger, meta_iter, task_tier, logger):
    """
    Collect a full rollout for one meta-task.
    Each call resets the env, which generates a fresh random card configuration
    (natural task diversity for Reptile — no need to pin specific cards).
    """
    obs = envs.reset()
    infos = None  # populated after first step; use None-safe get_prompt

    qs = get_prompt(args.env_name, args.action_only_prompt, infos)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259

    rollouts.obs[0].copy_(obs)
    running_episode_rewards = torch.zeros(args.num_processes).flatten()
    episode_rewards_local = []
    episode_success_local = []

    for step in range(args.num_steps):
        with torch.no_grad():
            INPUT_IDS_step = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            INPUT_IDS_step[INPUT_IDS_step == 0] = 259
            value, output_id, action, action_log_prob, action_tokens_log_prob = actor_critic.act(
                rollouts.obs[step], INPUT_IDS=INPUT_IDS_step)

        text_action = tokenizer.decode(list(filter(lambda num: num != 0, output_id[0].tolist())))
        obs, reward, done, infos = envs.step(action)

        qs = get_prompt(args.env_name, args.action_only_prompt, infos)
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        running_episode_rewards += reward.flatten()

        for i, d in enumerate(done):
            if d:
                episode_rewards_local.append(running_episode_rewards[i].item())
                episode_success_local.append(1 if running_episode_rewards[i] > 0 else 0)
                running_episode_rewards[i] = 0

        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        rollouts.insert(obs, output_id, action, action_log_prob, value, reward, masks, bad_masks)

        if step_logger:
            for i_proc in range(args.num_processes):
                step_logger.log_step(
                    meta_iter=meta_iter, task_tier=task_tier, inner_step=0, step=step,
                    info=infos[i_proc] if i_proc < len(infos) else {},
                    text_action=text_action,
                    action=action[i_proc] if action.dim() > 0 else action,
                    reward=reward[i_proc] if reward.dim() > 1 else reward.item(),
                    done=done[i_proc] if hasattr(done, '__getitem__') else done
                )

    # Compute returns
    with torch.no_grad():
        next_value = actor_critic.get_value(rollouts.obs[-1]).detach()
    rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

    return episode_rewards_local, episode_success_local


def main():
    args = get_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = getattr(args, 'log_dir', None) or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rl_logs", f"metarl_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logging(log_dir)
    step_logger = StepLogger(log_dir)
    iter_logger = IterationLogger(log_dir)

    logger.info("=" * 80)
    logger.info("  Meta-RL (Reptile + PPO) Training for Points24")
    logger.info(f"  Log directory: {log_dir}")
    logger.info(f"  Start time: {datetime.now().isoformat()}")
    logger.info("=" * 80)

    save_config(args, log_dir)

    # Meta-RL specific args
    meta_lr = getattr(args, 'meta_lr', 1e-4)
    inner_steps = getattr(args, 'inner_steps', 5)
    meta_batch_size = getattr(args, 'meta_batch_size', 3)
    meta_strategy = getattr(args, 'meta_strategy', 'reptile')

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
    optimizer = optim.Adam(actor_critic.value_model.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)

    # PPO agent (used in inner loop)
    ppo_agent = algo.PPO(
        actor_critic, optimizer, accelerator,
        args.clip_param, args.ppo_epoch, args.mini_batch_size,
        args.value_loss_coef, args.entropy_coef, max_grad_norm=args.max_grad_norm)

    # Meta-PPO wrapper
    meta_agent = MetaPPO(
        ppo_agent=ppo_agent, actor_critic=actor_critic,
        meta_lr=meta_lr, inner_steps=inner_steps,
        meta_batch_size=meta_batch_size, strategy=meta_strategy)

    # Task sampler: pre-compute solvable card combos per difficulty tier
    logger.info("Pre-computing solvable card configurations per difficulty tier...")
    task_sampler = TaskSampler(tiers=["easy", "medium", "hard"], precompute=True,
                                max_per_tier=300, seed=args.seed)
    for tier, combos in task_sampler.solvable_cache.items():
        logger.info(f"  {tier}: {len(combos)} solvable combos cached")

    # Tracking
    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    start = time.time()

    num_meta_updates = int(args.num_env_steps) // (args.num_steps * meta_batch_size * args.num_processes)

    logger.info(f"\n{'='*80}")
    logger.info(f"  Meta-RL Configuration")
    logger.info(f"  Strategy: {meta_strategy}")
    logger.info(f"  Meta LR (beta): {meta_lr}")
    logger.info(f"  Inner steps (K): {inner_steps}")
    logger.info(f"  Meta batch size (N tasks): {meta_batch_size}")
    logger.info(f"  Total meta updates: {num_meta_updates}")
    logger.info(f"  Steps per rollout: {args.num_steps}")
    logger.info(f"  PPO epochs per inner step: {args.ppo_epoch}")
    logger.info(f"{'='*80}\n")

    # ── Main Meta-RL training loop ──
    pbar = tqdm(range(num_meta_updates), desc="Meta-RL Training", ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    total_timesteps = 0

    for meta_iter in pbar:
        # Sample tasks: one from each tier for diversity
        task_tier_order = ["easy", "medium", "hard"][:meta_batch_size]
        if meta_batch_size > 3:
            task_tier_order = task_tier_order + [random.choice(["easy", "medium", "hard"]) for _ in range(meta_batch_size - 3)]

        task_rollouts_list = []
        meta_iter_rewards = []
        meta_iter_success = []

        logger.info(f"\n{'='*60}")
        logger.info(f"  META-ITERATION {meta_iter}/{num_meta_updates}")
        logger.info(f"{'='*60}")

        for task_idx, tier in enumerate(task_tier_order):
            task_cards = task_sampler.sample_task(tier)
            logger.info(f"  Task {task_idx}: tier={tier}, cards={task_cards}")

            rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                      envs.observation_space.shape, envs.action_space, args.max_new_tokens)
            rollouts.to(device)

            ep_rewards, ep_success = collect_task_rollout(
                actor_critic, envs, rollouts, args, tokenizer,
                lambda infos: get_prompt(args.env_name, args.action_only_prompt, infos),
                task_cards, step_logger, meta_iter, tier, logger)

            task_rollouts_list.append(rollouts)
            meta_iter_rewards.extend(ep_rewards)
            meta_iter_success.extend(ep_success)
            total_timesteps += args.num_steps * args.num_processes

        # Meta-update (Reptile)
        meta_metrics = meta_agent.meta_step(task_rollouts_list)
        lr_scheduler.step()

        # Update tracking
        for r in meta_iter_rewards:
            episode_rewards.append(r)
        for s in meta_iter_success:
            episode_success_rate.append(s)

        if len(episode_rewards) > 1:
            end = time.time()
            elapsed_min = (end - start) / 60.0
            fps = int(total_timesteps / (end - start))

            mean_reward = np.mean(episode_rewards)
            success_rate = np.mean(episode_success_rate)

            logger.info(f"\n  METRICS (Meta-Iter {meta_iter})")
            logger.info(f"  Timesteps: {total_timesteps} | FPS: {fps} | Elapsed: {elapsed_min:.1f} min")
            logger.info(f"  Reward mean: {mean_reward:.3f} | Success Rate: {success_rate:.4f} ({success_rate*100:.1f}%)")
            logger.info(f"  Meta inner value_loss: {meta_metrics['inner_value_loss']:.6f}")
            logger.info(f"  Meta inner action_loss: {meta_metrics['inner_action_loss']:.6f}")

            pbar.set_postfix({
                'succ': f'{success_rate*100:.1f}%',
                'rew': f'{mean_reward:.2f}',
            })

            current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else args.init_lr
            iter_logger.log_iteration({
                "meta_iteration": meta_iter,
                "timesteps": total_timesteps,
                "fps": fps,
                "elapsed_time_min": round(elapsed_min, 2),
                "mean_reward": round(mean_reward, 4),
                "median_reward": round(np.median(episode_rewards), 4),
                "min_reward": round(np.min(episode_rewards), 4),
                "max_reward": round(np.max(episode_rewards), 4),
                "success_rate": round(success_rate, 4),
                "meta_value_loss": round(meta_metrics['inner_value_loss'], 6),
                "meta_action_loss": round(meta_metrics['inner_action_loss'], 6),
                "learning_rate": current_lr,
                "meta_lr": meta_lr,
                "task_tier_distribution": str(task_tier_order),
                "reward_rollout_mean": round(np.mean(meta_iter_rewards) if meta_iter_rewards else 0, 4),
                "reward_rollout_std": round(np.std(meta_iter_rewards) if meta_iter_rewards else 0, 4),
            })

        step_logger.flush()

        # Checkpoint
        if (meta_iter + 1) % 5 == 0 or meta_iter == num_meta_updates - 1:
            logger.info(f"\n  Saving checkpoint at meta-iteration {meta_iter}...")
            save_checkpoint(actor_critic, optimizer, lr_scheduler, meta_iter, log_dir, logger)

    pbar.close()

    total_time = (time.time() - start) / 3600.0
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"  META-RL TRAINING COMPLETE!")
    logger.info(f"  Total time: {total_time:.2f} hours")
    logger.info(f"  Final success rate: {np.mean(episode_success_rate)*100:.1f}%")
    logger.info(f"  Logs saved to: {log_dir}")
    logger.info("=" * 80)

    step_logger.close()
    iter_logger.close()


if __name__ == "__main__":
    main()
