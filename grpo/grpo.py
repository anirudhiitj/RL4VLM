"""
GRPO: Group Relative Policy Optimization for VLM RL training.

Eliminates the value/critic network. Instead:
1. For each state, samples G outputs from the current policy.
2. Computes reward for each output.
3. Uses relative ranking within the group as advantage:
   A_i = (r_i - mean(r_1,...,r_G)) / (std(r_1,...,r_G) + eps)
4. Applies PPO-style clipped surrogate with group-relative advantages.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import accelerate


class GRPO:
    def __init__(self,
                 actor_critic,
                 optimizer,
                 accelerator,
                 clip_param,
                 grpo_epoch,
                 mini_batch_size,
                 group_size,
                 kl_coef=0.01,
                 max_grad_norm=None,
                 ref_model=None):
        """
        Args:
            actor_critic: VLMPolicy (actor only, no value network used for advantage).
            optimizer: Optimizer for policy parameters.
            accelerator: Accelerate instance.
            clip_param: PPO clipping parameter.
            grpo_epoch: Number of update epochs per batch.
            mini_batch_size: Mini-batch size for updates.
            group_size: G — number of samples per state for group advantage.
            kl_coef: Coefficient for KL penalty against reference policy.
            max_grad_norm: Max gradient norm for clipping.
            ref_model: Reference model (SFT checkpoint) for KL penalty. Optional.
        """
        self.actor_critic = actor_critic
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.clip_param = clip_param
        self.grpo_epoch = grpo_epoch
        self.mini_batch_size = mini_batch_size
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        self.ref_model = ref_model

    def compute_group_advantages(self, group_rewards):
        """
        Compute group-relative advantages.

        Args:
            group_rewards: Tensor of shape (num_states, group_size) — rewards for each
                          sample in each group.

        Returns:
            advantages: Tensor of shape (num_states, group_size) — normalized advantages.
        """
        mean_r = group_rewards.mean(dim=1, keepdim=True)
        std_r = group_rewards.std(dim=1, keepdim=True)
        advantages = (group_rewards - mean_r) / (std_r + 1e-8)
        return advantages

    def update(self, grpo_storage):
        """
        GRPO policy update using group-relative advantages.

        Args:
            grpo_storage: GRPOStorage with collected group rollout data.

        Returns:
            action_loss_epoch: Mean action loss across all updates.
            kl_loss_epoch: Mean KL penalty across all updates.
        """
        # Compute group advantages
        advantages = self.compute_group_advantages(grpo_storage.group_rewards)
        # Flatten: (num_states * group_size,)
        flat_advantages = advantages.view(-1)

        action_loss_epoch = 0
        kl_loss_epoch = 0
        grad_step = 0

        self.actor_critic.train()

        for e in range(self.grpo_epoch):
            data_generator = grpo_storage.feed_forward_generator(
                flat_advantages, self.mini_batch_size)

            for sample in data_generator:
                grad_step += 1
                obs_batch, output_ids_batch, old_action_log_probs_batch, adv_targ = sample

                # Compute current log probs
                _, action_log_probs = self.actor_critic.evaluate_actions(
                    obs_batch, output_ids_batch)

                if torch.isnan(action_log_probs).any():
                    continue

                old_action_log_probs_batch = old_action_log_probs_batch.to(action_log_probs.device).view(-1)
                adv_targ = adv_targ.to(action_log_probs.device)

                # PPO clipped surrogate
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ

                if torch.any(ratio > 10):
                    action_loss = -surr2.mean()
                else:
                    action_loss = -torch.min(surr1, surr2).mean()

                # Optional KL penalty against reference (SFT) model
                kl_loss = torch.tensor(0.0, device=action_log_probs.device)
                if self.ref_model is not None and self.kl_coef > 0:
                    with torch.no_grad():
                        _, ref_log_probs = self.ref_model.evaluate_actions(
                            obs_batch, output_ids_batch)
                    # KL(π || π_ref) ≈ exp(log π - log π_ref) * (log π - log π_ref) - (exp(log π - log π_ref) - 1)
                    log_ratio = action_log_probs - ref_log_probs.to(action_log_probs.device)
                    kl_loss = (torch.exp(log_ratio) * log_ratio - (torch.exp(log_ratio) - 1)).mean()

                loss = action_loss + self.kl_coef * kl_loss

                try:
                    assert not torch.isnan(loss), "loss is nan"
                except AssertionError:
                    print("GRPO loss is nan")
                    exit(1)

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.actor_critic.parameters(),
                        self.max_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

                action_loss_epoch += action_loss.item()
                kl_loss_epoch += kl_loss.item()

        if grad_step > 0:
            action_loss_epoch /= grad_step
            kl_loss_epoch /= grad_step

        return action_loss_epoch, kl_loss_epoch
