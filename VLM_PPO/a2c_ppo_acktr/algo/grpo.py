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
        mean_r = group_rewards.mean(dim=1, keepdim=True)
        std_r = group_rewards.std(dim=1, keepdim=True)
        advantages = (group_rewards - mean_r) / (std_r + 1e-8)
        return advantages

    def update(self, grpo_storage):
        advantages = self.compute_group_advantages(grpo_storage.group_rewards)
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

                _, action_log_probs = self.actor_critic.evaluate_actions(
                    obs_batch, output_ids_batch)

                if torch.isnan(action_log_probs).any() or torch.isinf(action_log_probs).any():
                    self.optimizer.zero_grad()
                    continue

                old_action_log_probs_batch = old_action_log_probs_batch.to(action_log_probs.device).view(-1)
                adv_targ = adv_targ.to(action_log_probs.device)

                # Clamp log-prob difference before exp to prevent ratio overflow
                log_diff = torch.clamp(action_log_probs - old_action_log_probs_batch, -10.0, 10.0)
                ratio = torch.exp(log_diff)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                kl_loss = torch.tensor(0.0, device=action_log_probs.device)
                if self.ref_model is not None and self.kl_coef > 0:
                    with torch.no_grad():
                        _, ref_log_probs = self.ref_model.evaluate_actions(
                            obs_batch, output_ids_batch)
                    log_ratio = torch.clamp(action_log_probs - ref_log_probs.to(action_log_probs.device), -10.0, 10.0)
                    kl_loss = (torch.exp(log_ratio) * log_ratio - (torch.exp(log_ratio) - 1)).mean()

                loss = action_loss + self.kl_coef * kl_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print("GRPO loss is nan/inf, skipping batch")
                    self.optimizer.zero_grad()
                    continue

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
