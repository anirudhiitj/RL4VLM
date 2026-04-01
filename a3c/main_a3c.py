"""
A3C (Asynchronous Advantage Actor-Critic) training script for Points24.

Implementation: Multiple workers on single GPU with shared base VLM.
Each worker has its own environment and collects n-step rollouts.
Workers alternate: collect short rollout → compute gradients → apply to shared model.

This is a "serial A3C" / multi-environment advantage actor-critic that captures
the key A3C benefits (diverse exploration from multiple envs, n-step returns,
no replay buffer) while staying on a single GPU.

For true async multi-GPU, each worker would run in a separate process.

Pipeline: SFT (full-weight, 1 epoch) → A3C (LoRA)
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
from a2c_ppo_acktr.algo.a3c import A3CUpdate, compute_n_step_returns
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
    logger = logging.getLogger("a3c_training")
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
    serializable["method"] = "a3c"
    with open(config_path, 'w') as f:
        json.dump(serializable, f, indent=2)


class StepLogger:
    def __init__(self, log_dir):
        self.filepath = os.path.join(log_dir, "step_details.jsonl")
        self.f = open(self.filepath, 'a')

    def log_step(self, iteration, worker_id, step, info, text_action, action, reward, done):
        entry = {
            "iteration": iteration,
            "worker_id": worker_id,
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
            "iteration", "timesteps", "fps", "elapsed_time_min",
            "mean_reward", "median_reward", "min_reward", "max_reward",
            "success_rate", "policy_loss", "value_loss",
            "learning_rate", "num_workers",
            "worker_success_rates",
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


class A3CWorker:
    """
    A single A3C worker with its own environment instance.
    Collects n-step rollouts and computes local gradients.
    """

    def __init__(self, worker_id, env, args, device):
        self.worker_id = worker_id
        self.env = env
        self.args = args
        self.device = device

        # Per-worker tracking
        self.obs = None
        self.infos = None
        self.prompt = None
        self.running_reward = 0.0
        self.episode_rewards = deque(maxlen=100)
        self.episode_success = deque(maxlen=100)

    def reset(self):
        self.obs = self.env.reset()
        if isinstance(self.obs, tuple):
            self.obs, info = self.obs
            self.infos = [info]
        else:
            self.infos = None
        self.running_reward = 0.0

    def collect_n_step_rollout(self, actor_critic, tokenizer, n_steps, args):
        """
        Collect an n-step rollout from this worker's environment.

        Returns:
            step_data: list of dicts with obs, output_ids, action, action_log_prob,
                      value, reward, done, text_action
        """
        step_data = []

        for step in range(n_steps):
            # Build prompt for current state
            qs = get_prompt(args.env_name, args.action_only_prompt, self.infos)
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            INPUT_IDS[INPUT_IDS == 0] = 259

            # Get action from policy
            with torch.no_grad():
                value, output_id, action, action_log_prob, action_tokens_log_prob = actor_critic.act(
                    self.obs, INPUT_IDS=INPUT_IDS)

            text_action = tokenizer.decode(list(filter(lambda num: num != 0, output_id[0].tolist())))

            # Step environment
            obs_new, reward, done, infos = self.env.step(action)

            self.running_reward += reward.item() if isinstance(reward, torch.Tensor) else float(reward)

            done_flag = done[0] if hasattr(done, '__getitem__') else done
            if done_flag:
                self.episode_rewards.append(self.running_reward)
                self.episode_success.append(1 if self.running_reward > 0 else 0)
                self.running_reward = 0.0

            step_data.append({
                'obs': self.obs.clone() if isinstance(self.obs, torch.Tensor) else torch.from_numpy(self.obs).float(),
                'output_ids': output_id,
                'action': action,
                'action_log_prob': action_log_prob,
                'value': value,
                'reward': reward.item() if isinstance(reward, torch.Tensor) else float(reward),
                'done': bool(done_flag),
                'text_action': text_action,
                'infos': infos,
            })

            self.obs = obs_new
            self.infos = infos

        return step_data


def a3c_worker_update(actor_critic, optimizer, a3c_update, worker_step_data, args, accelerator):
    """
    Compute A3C loss from one worker's n-step rollout and apply gradients.

    Args:
        actor_critic: Shared VLMPolicy model
        optimizer: Shared optimizer
        a3c_update: A3CUpdate instance
        worker_step_data: List of step dicts from worker's rollout
        args: Training arguments
        accelerator: Accelerate instance

    Returns:
        policy_loss, value_loss (floats)
    """
    n = len(worker_step_data)
    if n == 0:
        return 0.0, 0.0

    rewards = [s['reward'] for s in worker_step_data]
    values = [s['value'].item() for s in worker_step_data]
    dones = [s['done'] for s in worker_step_data]

    # Compute next value for bootstrap
    # If last step was done, next_value is 0 (no future reward from terminal state)
    if worker_step_data[-1]['done']:
        next_value = 0.0
    else:
        with torch.no_grad():
            last_obs = worker_step_data[-1]['obs']
            if last_obs.dim() == 3:
                last_obs = last_obs.unsqueeze(0)
            next_value = actor_critic.get_value(last_obs).item()

    # Compute n-step returns and advantages
    returns, advantages = compute_n_step_returns(rewards, values, dones, next_value, args.gamma)

    # Re-evaluate actions to get gradients
    action_log_probs_list = []
    values_list = []

    for s in worker_step_data:
        obs = s['obs']
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        output_ids = s['output_ids']

        v, alp = actor_critic.evaluate_actions(obs, output_ids)
        action_log_probs_list.append(alp)
        values_list.append(v)

    action_log_probs = torch.cat(action_log_probs_list, dim=0).squeeze()
    values_tensor = torch.cat(values_list, dim=0).squeeze()

    # Compute A3C loss
    total_loss, policy_loss, value_loss = a3c_update.compute_loss(
        action_log_probs, values_tensor, returns, advantages)

    # Backward and step
    accelerator.backward(total_loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    return policy_loss, value_loss


def main():
    args = get_args()

    # A3C specific args
    num_workers = getattr(args, 'num_workers', 4)
    n_step = getattr(args, 'a3c_n_step', 5)
    a3c_entropy_coef = getattr(args, 'a3c_entropy_coef', 0.01)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = getattr(args, 'log_dir', None) or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rl_logs", f"a3c_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logging(log_dir)
    step_logger = StepLogger(log_dir)
    iter_logger = IterationLogger(log_dir)

    logger.info("=" * 80)
    logger.info("  A3C (Asynchronous Advantage Actor-Critic) Training for Points24")
    logger.info(f"  Log directory: {log_dir}")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  N-step returns: {n_step}")
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

    value_model = VLMValue(base)
    value_model = value_model.to(model_device)

    # Create multiple environments (one per worker)
    worker_envs = []
    for w in range(num_workers):
        env = make_vec_envs(args.env_name, args.seed + w, 1,
                            args.gamma, None, device, False, 1)
        worker_envs.append(env)
    logger.info(f"Created {num_workers} worker environments")

    # Build prompt (initial)
    obs = worker_envs[0].reset()
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

    # Shared actor-critic (all workers use the same model)
    actor_critic = VLMPolicy(tokenizer=tokenizer, image_processor=image_processor,
                             value_model=value_model, projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS, args=args)
    optimizer = optim.Adam(actor_critic.value_model.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)

    # A3C update logic
    a3c_update = A3CUpdate(
        value_loss_coef=args.value_loss_coef,
        entropy_coef=a3c_entropy_coef,
        max_grad_norm=args.max_grad_norm
    )

    # Create workers
    workers = []
    for w in range(num_workers):
        worker = A3CWorker(
            worker_id=w, env=worker_envs[w],
            args=args, device=device)
        worker.reset()
        workers.append(worker)

    # Tracking
    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    # Track how many episodes have been synced from each worker
    worker_episode_synced = [0] * num_workers
    start = time.time()
    total_timesteps = 0

    # Number of iterations: total steps / (n_step * num_workers)
    num_updates = int(args.num_env_steps) // (n_step * num_workers)

    logger.info(f"\n{'='*80}")
    logger.info(f"  A3C Configuration")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  N-step: {n_step}")
    logger.info(f"  Entropy coef: {a3c_entropy_coef}")
    logger.info(f"  Value loss coef: {args.value_loss_coef}")
    logger.info(f"  Num updates: {num_updates}")
    logger.info(f"  Init LR: {args.init_lr}, End LR: {args.end_lr}")
    logger.info(f"  Use LoRA: {args.use_lora}")
    logger.info(f"{'='*80}\n")

    pbar = tqdm(range(num_updates), desc="A3C Training", ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for iteration in pbar:
        iter_policy_loss = 0.0
        iter_value_loss = 0.0
        worker_updates = 0

        # Each worker collects an n-step rollout and updates the shared model
        for w, worker in enumerate(workers):
            # Collect n-step rollout
            step_data = worker.collect_n_step_rollout(
                actor_critic, tokenizer, n_step, args)

            # Log steps
            for s_idx, s in enumerate(step_data):
                step_logger.log_step(
                    iteration=iteration, worker_id=w, step=s_idx,
                    info=s['infos'][0] if isinstance(s['infos'], list) and len(s['infos']) > 0 else s['infos'],
                    text_action=s['text_action'],
                    action=s['action'][0].item() if s['action'].dim() > 0 else s['action'].item(),
                    reward=s['reward'],
                    done=s['done']
                )

            # A3C update: compute grad from this worker's rollout and apply to shared model
            if len(step_data) > 0:
                policy_loss, value_loss = a3c_worker_update(
                    actor_critic, optimizer, a3c_update,
                    step_data, args, accelerator)
                iter_policy_loss += policy_loss
                iter_value_loss += value_loss
                worker_updates += 1

            # Collect episode stats from this worker — only new episodes since last sync
            num_new = len(worker.episode_rewards) - worker_episode_synced[w]
            if num_new > 0:
                # Get the last num_new entries from the worker's deque
                new_rewards = list(worker.episode_rewards)[-num_new:]
                new_success = list(worker.episode_success)[-num_new:]
                for r in new_rewards:
                    episode_rewards.append(r)
                for s in new_success:
                    episode_success_rate.append(s)
                worker_episode_synced[w] = len(worker.episode_rewards)

            total_timesteps += n_step

        if worker_updates > 0:
            iter_policy_loss /= worker_updates
            iter_value_loss /= worker_updates

        # LR step once per iteration (after all workers update)
        if (iteration + 1) % num_workers == 0:
            lr_scheduler.step()

        current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else args.init_lr

        # Log every few iterations
        if len(episode_rewards) > 1 and (iteration + 1) % 10 == 0:
            end = time.time()
            elapsed_min = (end - start) / 60.0
            fps = int(total_timesteps / (end - start))

            mean_reward = np.mean(episode_rewards)
            success_rate = np.mean(episode_success_rate)

            # Per-worker success rates
            worker_success_rates = []
            for worker in workers:
                ws = np.mean(worker.episode_success) if len(worker.episode_success) > 0 else 0.0
                worker_success_rates.append(round(ws, 4))

            logger.info(f"\n{'='*60}")
            logger.info(f"  A3C ITERATION {iteration}/{num_updates}")
            logger.info(f"  Timesteps: {total_timesteps} | FPS: {fps} | Elapsed: {elapsed_min:.1f} min")
            logger.info(f"  Reward mean: {mean_reward:.3f} | Success: {success_rate*100:.1f}%")
            logger.info(f"  Policy Loss: {iter_policy_loss:.6f} | Value Loss: {iter_value_loss:.6f}")
            logger.info(f"  Worker success rates: {worker_success_rates}")
            logger.info(f"  LR: {current_lr:.2e}")

            pbar.set_postfix({
                'succ': f'{success_rate*100:.1f}%',
                'rew': f'{mean_reward:.2f}',
                'p_loss': f'{iter_policy_loss:.4f}',
            })

            iter_logger.log_iteration({
                "iteration": iteration,
                "timesteps": total_timesteps,
                "fps": fps,
                "elapsed_time_min": round(elapsed_min, 2),
                "mean_reward": round(mean_reward, 4),
                "median_reward": round(np.median(episode_rewards), 4),
                "min_reward": round(np.min(episode_rewards), 4),
                "max_reward": round(np.max(episode_rewards), 4),
                "success_rate": round(success_rate, 4),
                "policy_loss": round(iter_policy_loss, 6),
                "value_loss": round(iter_value_loss, 6),
                "learning_rate": current_lr,
                "num_workers": num_workers,
                "worker_success_rates": str(worker_success_rates),
            })

        step_logger.flush()

        # Checkpoint
        if (iteration + 1) % 50 == 0 or iteration == num_updates - 1:
            logger.info(f"\n  Saving checkpoint at iteration {iteration}...")
            save_checkpoint(actor_critic, optimizer, lr_scheduler, iteration, log_dir, logger)

    pbar.close()

    total_time = (time.time() - start) / 3600.0
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"  A3C TRAINING COMPLETE!")
    logger.info(f"  Total time: {total_time:.2f} hours")
    logger.info(f"  Final success rate: {np.mean(episode_success_rate)*100:.1f}%")
    logger.info(f"  Logs saved to: {log_dir}")
    logger.info("=" * 80)

    step_logger.close()
    iter_logger.close()


if __name__ == "__main__":
    main()
