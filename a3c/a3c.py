"""
A3C (Asynchronous Advantage Actor-Critic) algorithm for VLM RL training.

Each worker:
  1. Collects a short n-step rollout in its own environment.
  2. Computes n-step returns and advantages.
  3. Computes gradients locally.
  4. Pushes gradients to the shared global model.
  5. Pulls the latest global parameters.

Key difference from PPO: no replay buffer, no clipping — uses vanilla
policy gradient with entropy bonus + value loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_n_step_returns(rewards, values, dones, next_value, gamma=0.99):
    """
    Compute n-step returns and advantages for a short rollout.

    Args:
        rewards: list of n rewards (floats)
        values: list of n value estimates (floats)
        dones: list of n done flags (bools)
        next_value: V(s_{n+1}) — value of final state
        gamma: discount factor

    Returns:
        returns: list of n discounted returns
        advantages: list of n advantages = return - value
    """
    n = len(rewards)
    returns = [0.0] * n
    R = next_value

    for i in reversed(range(n)):
        R = rewards[i] + gamma * R * (1.0 - float(dones[i]))
        returns[i] = R

    advantages = [returns[i] - values[i] for i in range(n)]
    return returns, advantages


class A3CUpdate:
    """
    Stateless A3C update logic. Used by each worker to compute loss
    and push gradients to the shared model.
    """

    def __init__(self, value_loss_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def compute_loss(self, action_log_probs, values, returns, advantages):
        """
        Compute A3C loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy.

        Args:
            action_log_probs: (n,) tensor of log probs for taken actions
            values: (n,) tensor of value predictions
            returns: (n,) tensor of n-step returns
            advantages: (n,) tensor of advantages

        Returns:
            total_loss, policy_loss, value_loss
        """
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=action_log_probs.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=values.device)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Policy loss (vanilla policy gradient)
        policy_loss = -(action_log_probs * advantages_t.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns_t)

        # Total loss (no entropy term since LLM generation doesn't have a simple entropy)
        total_loss = policy_loss + self.value_loss_coef * value_loss

        return total_loss, policy_loss.item(), value_loss.item()


def sync_local_to_global(local_model, global_model):
    """
    Push local model gradients to global model parameters.
    (Hogwild-style: no locking, async gradient application.)

    Only syncs trainable parameters (LoRA + value_head).
    """
    local_m = local_model.module if hasattr(local_model, 'module') else local_model
    global_m = global_model.module if hasattr(global_model, 'module') else global_model

    for local_param, global_param in zip(local_m.parameters(), global_m.parameters()):
        if global_param.requires_grad and local_param.grad is not None:
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad.copy_(local_param.grad)


def sync_global_to_local(global_model, local_model):
    """
    Pull latest global parameters into local model.
    Only syncs trainable parameters.
    """
    local_m = local_model.module if hasattr(local_model, 'module') else local_model
    global_m = global_model.module if hasattr(global_model, 'module') else global_model

    for local_param, global_param in zip(local_m.parameters(), global_m.parameters()):
        if global_param.requires_grad:
            local_param.data.copy_(global_param.data)
