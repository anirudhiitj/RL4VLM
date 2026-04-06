"""
Meta-PPO: Reptile/FOMAML meta-learning wrapper around PPO.

Implements the meta-RL outer loop:
  - Reptile: θ ← θ + β * (1/N) * Σ(θ'_i - θ)
  - FOMAML: Uses first-order gradients through inner loop

Only LoRA adapter parameters are meta-learned. Base model weights are frozen.
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


def _clear_optimizer_state(optimizer):
    """
    Clear Adam/AdamW momentum & variance state between inner-loop tasks.
    Works through DeepSpeed ZeRO-2 wrapped optimizers.
    Avoids copy.deepcopy which fails on non-leaf ZeRO-2 tensors.
    """
    # DeepSpeed wraps the real optimizer under .optimizer
    actual_opt = getattr(optimizer, 'optimizer', optimizer)
    actual_opt.state.clear()


class MetaPPO:
    """
    Meta-learning wrapper around PPO using Reptile algorithm.

    The inner loop runs K steps of standard PPO on a sampled task.
    The outer loop aggregates inner-loop adapted parameters via Reptile update.
    """

    def __init__(self,
                 ppo_agent,
                 actor_critic,
                 meta_lr: float = 1e-4,
                 inner_steps: int = 5,
                 meta_batch_size: int = 3,
                 strategy: str = "reptile"):
        """
        Args:
            ppo_agent: The PPO agent (a2c_ppo_acktr.algo.PPO instance).
            actor_critic: VLMPolicy with LoRA adapters.
            meta_lr: Outer loop learning rate (β).
            inner_steps: Number of PPO update steps per inner loop (K).
            meta_batch_size: Number of tasks per meta-batch (N).
            strategy: "reptile" or "fomaml".
        """
        self.ppo_agent = ppo_agent
        self.actor_critic = actor_critic
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size
        self.strategy = strategy

    def get_trainable_state_dict(self) -> OrderedDict:
        """Get state dict of only trainable (LoRA + value_head) parameters."""
        model = self.actor_critic.module if hasattr(self.actor_critic, 'module') else self.actor_critic
        trainable = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable[name] = param.data.clone()
        return trainable

    def set_trainable_state_dict(self, state_dict: OrderedDict):
        """Restore trainable parameters from a state dict."""
        model = self.actor_critic.module if hasattr(self.actor_critic, 'module') else self.actor_critic
        for name, param in model.named_parameters():
            if param.requires_grad and name in state_dict:
                param.data.copy_(state_dict[name])

    def inner_loop(self, rollouts) -> Tuple[OrderedDict, Dict[str, float]]:
        """
        Run K steps of PPO on a single task's rollout data.

        Each inner step runs 1 PPO epoch (not the full ppo_epoch count)
        to avoid overfitting on the same rollout data across K steps.

        Args:
            rollouts: RolloutStorage filled with data from one task.

        Returns:
            adapted_params: State dict of adapted LoRA + value_head params.
            metrics: Dict with aggregated inner loop losses.
        """
        total_value_loss = 0.0
        total_action_loss = 0.0

        # Save original ppo_epoch and temporarily set to 1 for inner steps
        original_ppo_epoch = self.ppo_agent.ppo_epoch
        self.ppo_agent.ppo_epoch = 1

        for k in range(self.inner_steps):
            value_loss, action_loss, dist_entropy = self.ppo_agent.update(rollouts)
            total_value_loss += value_loss
            total_action_loss += action_loss

        # Restore original ppo_epoch
        self.ppo_agent.ppo_epoch = original_ppo_epoch

        adapted_params = self.get_trainable_state_dict()

        metrics = {
            "inner_value_loss": total_value_loss / self.inner_steps,
            "inner_action_loss": total_action_loss / self.inner_steps,
        }
        return adapted_params, metrics

    def meta_update(self, meta_params: OrderedDict,
                    adapted_params_list: List[OrderedDict]) -> OrderedDict:
        """
        Reptile meta-update:
            θ ← θ + β * (1/N) * Σ(θ'_i - θ)

        Args:
            meta_params: The meta-parameters before inner loops.
            adapted_params_list: List of adapted parameters from each task.

        Returns:
            Updated meta-parameters.
        """
        n_tasks = len(adapted_params_list)
        updated_params = OrderedDict()

        for key in meta_params:
            # Compute average displacement
            displacement = torch.zeros_like(meta_params[key])
            for adapted in adapted_params_list:
                displacement += (adapted[key] - meta_params[key])
            displacement /= n_tasks

            # Reptile update
            updated_params[key] = meta_params[key] + self.meta_lr * displacement

        return updated_params

    def meta_step(self, task_rollouts_list: List) -> Dict[str, float]:
        """
        Execute one full meta-learning step:
          1. Save current meta-params
          2. For each task: clone params → run inner loop → collect adapted params
          3. Reptile meta-update

        Args:
            task_rollouts_list: List of RolloutStorage objects, one per task.

        Returns:
            Aggregated metrics from all inner loops.
        """
        # 1. Save meta-parameters
        meta_params = self.get_trainable_state_dict()

        optimizer = self.ppo_agent.optimizer
        adapted_params_list = []
        all_metrics = {"inner_value_loss": 0.0, "inner_action_loss": 0.0}

        # 2. Inner loop for each task
        for task_rollouts in task_rollouts_list:
            # Restore to meta-params before each inner loop
            self.set_trainable_state_dict(meta_params)
            # Clear Adam momentum state so each task starts fresh (safe for DeepSpeed)
            _clear_optimizer_state(optimizer)

            # Run inner loop
            adapted_params, metrics = self.inner_loop(task_rollouts)
            adapted_params_list.append(adapted_params)

            for k, v in metrics.items():
                all_metrics[k] += v

        # Average metrics
        n = len(task_rollouts_list)
        for k in all_metrics:
            all_metrics[k] /= n

        # 3. Meta-update
        updated_params = self.meta_update(meta_params, adapted_params_list)
        self.set_trainable_state_dict(updated_params)

        # Clear optimizer state after meta-update so next outer step starts fresh
        _clear_optimizer_state(optimizer)

        return all_metrics
