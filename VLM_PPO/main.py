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
    """Setup dual logging: file + terminal (stdout)."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "training.log")
    
    # Create a custom logger
    logger = logging.getLogger("rl_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler - logs everything to file
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_fmt)
    
    # Stream handler - logs everything to terminal
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_fmt = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_fmt)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger


def save_config(args, log_dir):
    """Save all training arguments as JSON."""
    config_path = os.path.join(log_dir, "config.json")
    config_dict = vars(args)
    # Convert non-serializable types
    serializable = {}
    for k, v in config_dict.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, ValueError):
            serializable[k] = str(v)
    serializable["start_time"] = datetime.now().isoformat()
    with open(config_path, 'w') as f:
        json.dump(serializable, f, indent=2)


class StepLogger:
    """Logs per-step details (CoT outputs, actions, rewards) to JSONL."""
    
    def __init__(self, log_dir):
        self.filepath = os.path.join(log_dir, "step_details.jsonl")
        self.f = open(self.filepath, 'a')
    
    def log_step(self, iteration, step, info, text_action, action, reward, done):
        entry = {
            "iteration": iteration,
            "step": step,
            "observation_info": str(info),
            "cot_output": text_action,
            "action": int(action) if isinstance(action, (int, np.integer)) else action.item() if hasattr(action, 'item') else str(action),
            "reward": float(reward) if hasattr(reward, 'item') else float(reward.item()) if hasattr(reward, 'item') else float(reward),
            "done": bool(done) if isinstance(done, (bool, np.bool_)) else bool(done),
            "timestamp": datetime.now().isoformat()
        }
        self.f.write(json.dumps(entry) + '\n')
    
    def flush(self):
        self.f.flush()
    
    def close(self):
        self.f.close()


class IterationLogger:
    """Logs per-iteration summary metrics to CSV."""
    
    def __init__(self, log_dir):
        self.filepath = os.path.join(log_dir, "iteration_summary.csv")
        self.fieldnames = [
            "iteration", "timesteps", "fps", "elapsed_time_min",
            "mean_reward", "median_reward", "min_reward", "max_reward",
            "success_rate", "value_loss", "action_loss",
            "learning_rate", "mean_action_tokens_log_prob",
            "reward_rollout_mean", "reward_rollout_std",
            "return_mean", "return_std",
            "value_pred_mean", "value_pred_std",
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
    """Save model checkpoint."""
    ckpt_dir = os.path.join(log_dir, "checkpoints", f"iter_{iteration:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save LoRA weights if applicable
    try:
        model = actor_critic.module if hasattr(actor_critic, 'module') else actor_critic
        value_model = model.value_model
        base_model = value_model.base if hasattr(value_model, 'base') else value_model
        
        # Save the base model's LoRA adapters
        if hasattr(base_model, 'save_pretrained'):
            base_model.save_pretrained(os.path.join(ckpt_dir, "lora_adapters"))
            logger.info(f"  Saved LoRA adapters to {ckpt_dir}/lora_adapters")
        
        # Save value head separately
        if hasattr(value_model, 'value_head'):
            torch.save(value_model.value_head.state_dict(), os.path.join(ckpt_dir, "value_head.pt"))
            logger.info(f"  Saved value head to {ckpt_dir}/value_head.pt")
        
        # Save training state
        torch.save({
            'iteration': iteration,
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        }, os.path.join(ckpt_dir, "training_state.pt"))
        
        logger.info(f"  ✅ Checkpoint saved at iteration {iteration}")
    except Exception as e:
        logger.info(f"  ⚠️ Checkpoint save failed: {e}")


def main():
    args = get_args()

    # ── Setup log directory ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = getattr(args, 'log_dir', None) or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rl_logs", f"numberline_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)
    
    # ── Setup logging ──
    logger = setup_logging(log_dir)
    step_logger = StepLogger(log_dir)
    iter_logger = IterationLogger(log_dir)
    
    logger.info("=" * 80)
    logger.info("  RL4VLM - NumberLine RL Training")
    logger.info(f"  Log directory: {log_dir}")
    logger.info(f"  Start time: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    # Save config
    save_config(args, log_dir)
    logger.info(f"Training config saved to {log_dir}/config.json")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device
    ## environment interaction device is cpu
    model_device = device

    #initialization of llava
    model_path = args.model_path
    cache_dir = args.cache_dir

    logger.info(f"Model path: {model_path}")
    #load_pretrained_model(model_path, model_path, model_path)
    if "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
        if args.q8 or args.q4:
            raise ValueError("Lora model does not support 8bit or 4bit quantization")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if args.q8:
            logger.info("8bit quantization")
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
        elif args.q4:
            q4_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                    )
            logger.info("4bit quantization")
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
        else:
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)

    use_grad_ckpt = True
    if use_grad_ckpt:
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    base.config.max_length = 1024
    logger.info("Model max context length:{}".format(base.config.max_length))
    base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter = args.pretrain_mm_adapter)
    image_processor = base.get_vision_tower().image_processor

    base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_all_linear_names(base,args.train_vision),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.use_lora:
        base = get_peft_model(base, base_lora_config)
    value_model = VLMValue(base)
    value_model = value_model.to(model_device)

    if "gym_cards" in args.env_name.lower():
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                             args.gamma, None, device, False, 1)
    else:
        logger.info("Environment not supported")
        exit(1)


    obs = envs.reset()
    infos = None
    ## Inputing Prompt here
    qs = get_prompt(args.env_name, args.action_only_prompt, infos)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    logger.info(f"Prompt:\n{prompt}")

    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace

    projection_f = partial(text_projection, env_name=args.env_name)

    actor_critic = VLMPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             value_model=value_model,
                             projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS,
                             args=args)
    optimizer = optim.Adam(actor_critic.value_model.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)

    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)

    agent = algo.PPO(
            actor_critic,
            optimizer,
            accelerator,
            args.clip_param,
            args.ppo_epoch,
            args.mini_batch_size,
            args.value_loss_coef,
            args.entropy_coef,
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, args.max_new_tokens)

    _, output_ids, action, action_log_prob, action_tokens_log_prob = actor_critic.act(obs, INPUT_IDS = INPUT_IDS)
    logger.info("action:{}".format(action))
    logger.info("action_log_prob:{}".format(action_log_prob))
    logger.info("action_tokens_log_prob:{}".format(action_tokens_log_prob))

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_action_tokens_log_prob = deque(maxlen=args.eval_num_per_episode)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    if args.use_wandb:
        import wandb
        run_name = args.wandb_run + "-" + args.env_name
        wandb.init(project=args.wandb_project, name=run_name, group=run_name, config=args)

    logger.info(f"\n{'='*80}")
    logger.info(f"  Training Configuration Summary")
    logger.info(f"  Env: {args.env_name}")
    logger.info(f"  Total env steps: {args.num_env_steps}")
    logger.info(f"  Steps per update: {args.num_steps}")
    logger.info(f"  Num updates: {num_updates}")
    logger.info(f"  PPO epochs: {args.ppo_epoch}")
    logger.info(f"  Mini batch size: {args.mini_batch_size}")
    logger.info(f"  Grad accum steps: {args.grad_accum_steps}")
    logger.info(f"  Init LR: {args.init_lr}, End LR: {args.end_lr}")
    logger.info(f"  Thought prob coef (lambda): {args.thought_prob_coef}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Use LoRA: {args.use_lora}")
    logger.info(f"  Action only prompt: {args.action_only_prompt}")
    logger.info(f"{'='*80}\n")
    
    logger.info(qs)
    running_episode_rewards = torch.zeros(args.num_processes).flatten()

    num_explore = int(args.explore_portion*num_updates)
    prev_infos = []
    infos = []
    
    # ── Main training loop with tqdm progress bar ──
    pbar = tqdm(range(num_updates), desc="RL Training", ncols=120, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for j in pbar:

        # ── Inner rollout loop with step-level progress ──
        step_pbar = tqdm(range(args.num_steps), desc=f"  Iter {j} rollout", 
                        ncols=100, leave=False,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for step in step_pbar:
            # Sample actions
            with torch.no_grad():
                INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace
                value, output_id, action, action_log_prob, action_tokens_log_prob = actor_critic.act(
                        rollouts.obs[step], INPUT_IDS = INPUT_IDS)
            text_action = tokenizer.decode(list(filter(lambda num: num != 0, output_id[0].tolist())))
            prev_infos = copy.deepcopy(infos)
            obs, reward, done, infos = envs.step(action)

            qs = get_prompt(args.env_name, args.action_only_prompt, infos)
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            running_episode_rewards += reward.flatten()
            for i, d, r in zip(range(args.num_processes), done, reward):
                if d:
                    episode_rewards.append(running_episode_rewards[i].item())
                    if running_episode_rewards[i] > 0:
                        episode_success_rate.append(1)
                    else:
                        episode_success_rate.append(0)
                    episode_action_tokens_log_prob.append(action_tokens_log_prob[i].item())
                    running_episode_rewards[i] = 0
            # bad_mask is a legacy implementation of the storage.py file
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, output_id, action,
                            action_log_prob, value, reward, masks, bad_masks)
            
            # ── Log per-step details to JSONL ──
            for i_proc in range(args.num_processes):
                step_logger.log_step(
                    iteration=j,
                    step=step,
                    info=infos[i_proc] if i_proc < len(infos) else {},
                    text_action=text_action,
                    action=action[i_proc] if action.dim() > 0 else action,
                    reward=reward[i_proc] if reward.dim() > 1 else reward.item(),
                    done=done[i_proc] if hasattr(done, '__getitem__') else done
                )
        
        step_pbar.close()
        
        # ── End of rollout: print iteration details ──
        logger.info("")
        logger.info(f"{'='*60}")
        logger.info(f"  ITERATION {j}/{num_updates}")
        logger.info(f"{'='*60}")
        logger.info(f"prompt: {prompt}")
        logger.info(f"text_action (CoT output): {text_action}")
        logger.info(f"current observation: {prev_infos}")
        logger.info(f"ground truth: {infos}")
        logger.info(f"action log prob: {action_log_prob}")
        logger.info(f"action tokens log prob: {action_tokens_log_prob}")
        
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        lr_scheduler.step()
        
        # Get current learning rate
        current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else args.init_lr

        rollouts.after_update()
        if len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            elapsed_min = (end - start) / 60.0
            fps = int(total_num_steps / (end - start))
            
            mean_reward = np.mean(episode_rewards)
            median_reward = np.median(episode_rewards)
            min_reward = np.min(episode_rewards)
            max_reward = np.max(episode_rewards)
            success_rate = np.mean(episode_success_rate)
            mean_atp = np.mean(episode_action_tokens_log_prob)

            # ── Terminal + file log ──
            logger.info("")
            logger.info(f"  📊 METRICS (Iteration {j})")
            logger.info(f"  ├─ Timesteps: {total_num_steps} | FPS: {fps} | Elapsed: {elapsed_min:.1f} min")
            logger.info(f"  ├─ Reward  → mean: {mean_reward:.3f} | median: {median_reward:.3f} | min: {min_reward:.3f} | max: {max_reward:.3f}")
            logger.info(f"  ├─ Success Rate: {success_rate:.4f} ({success_rate*100:.1f}%)")
            logger.info(f"  ├─ Value Loss: {value_loss:.6f} | Action Loss: {action_loss:.6f}")
            logger.info(f"  ├─ LR: {current_lr:.2e}")
            logger.info(f"  ├─ Rollout rewards → mean: {rollouts.rewards.mean().item():.4f} | std: {rollouts.rewards.std().item():.4f}")
            logger.info(f"  ├─ Returns → mean: {rollouts.returns.mean().item():.4f} | std: {rollouts.returns.std().item():.4f}")
            logger.info(f"  └─ Value preds → mean: {rollouts.value_preds.mean().item():.4f} | std: {rollouts.value_preds.std().item():.4f}")
            logger.info("")
            
            # ── Update tqdm progress bar description ──
            pbar.set_postfix({
                'succ': f'{success_rate*100:.1f}%',
                'rew': f'{mean_reward:.2f}',
                'v_loss': f'{value_loss:.4f}',
            })
            
            # ── Log to iteration CSV ──
            iter_logger.log_iteration({
                "iteration": j,
                "timesteps": total_num_steps,
                "fps": fps,
                "elapsed_time_min": round(elapsed_min, 2),
                "mean_reward": round(mean_reward, 4),
                "median_reward": round(median_reward, 4),
                "min_reward": round(min_reward, 4),
                "max_reward": round(max_reward, 4),
                "success_rate": round(success_rate, 4),
                "value_loss": round(value_loss, 6),
                "action_loss": round(action_loss, 6),
                "learning_rate": current_lr,
                "mean_action_tokens_log_prob": round(mean_atp, 4),
                "reward_rollout_mean": round(rollouts.rewards.mean().item(), 4),
                "reward_rollout_std": round(rollouts.rewards.std().item(), 4),
                "return_mean": round(rollouts.returns.mean().item(), 4),
                "return_std": round(rollouts.returns.std().item(), 4),
                "value_pred_mean": round(rollouts.value_preds.mean().item(), 4),
                "value_pred_std": round(rollouts.value_preds.std().item(), 4),
            })
            
            if args.use_wandb:
                wandb.log({"iteration": j,
                        "num_timesteps": total_num_steps,
                        "FPS": fps,
                        "episode_reward.mean": mean_reward,
                        "episode_reward.median": median_reward,
                        "episode_reward.min": min_reward,
                        "episode_reward.max": max_reward,
                        "episode_success_rate.mean": success_rate,
                        "episode_action_tokens_log_prob.mean": mean_atp,
                        "distribution_entropy": dist_entropy,
                        "value.loss": value_loss,
                        "action.loss": action_loss,
                        "reward.max": rollouts.rewards.max().item(),
                        "reward.min": rollouts.rewards.min().item(),
                        "reward.mean": rollouts.rewards.mean().item(),
                        "reward.std": rollouts.rewards.std().item(),
                        "reward.median": rollouts.rewards.median().item(),
                        "return.max": rollouts.returns.max().item(),
                        "return.min": rollouts.returns.min().item(),
                        "return.mean": rollouts.returns.mean().item(),
                        "return.std": rollouts.returns.std().item(),
                        "value.max": rollouts.value_preds.max().item(),
                        "value.min": rollouts.value_preds.min().item(),
                        "value.mean": rollouts.value_preds.mean().item(),
                        "value.std": rollouts.value_preds.std().item(),})
        
        # ── Flush step logs after each iteration ──
        step_logger.flush()
        
        # ── Save checkpoint every 5 iterations ──
        if (j + 1) % 5 == 0 or j == num_updates - 1:
            logger.info(f"\n  💾 Saving checkpoint at iteration {j}...")
            save_checkpoint(actor_critic, optimizer, lr_scheduler, j, log_dir, logger)
    
    pbar.close()
    
    # ── Training complete ──
    total_time = (time.time() - start) / 3600.0
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"  🎉 TRAINING COMPLETE!")
    logger.info(f"  Total time: {total_time:.2f} hours")
    logger.info(f"  Final success rate: {np.mean(episode_success_rate)*100:.1f}%")
    logger.info(f"  Logs saved to: {log_dir}")
    logger.info("=" * 80)
    
    # Cleanup
    step_logger.close()
    iter_logger.close()

if __name__ == "__main__":
    main()
