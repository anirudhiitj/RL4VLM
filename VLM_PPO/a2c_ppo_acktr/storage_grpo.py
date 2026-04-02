"""
GRPO-specific rollout storage.
Stores groups of G outputs per state, their rewards, and computed advantages.
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class GRPOStorage:
    """
    Storage for GRPO rollouts.

    For each env step, we store G generated outputs and their rewards.

    Shape conventions:
        obs:                (num_states, *obs_shape)
        group_output_ids:   (num_states, group_size, max_output_len)
        group_log_probs:    (num_states, group_size)
        group_rewards:      (num_states, group_size)
    """

    def __init__(self, num_states, group_size, obs_shape, max_output_len):
        self.num_states = num_states
        self.group_size = group_size
        self.max_output_len = max_output_len

        self.obs = torch.zeros(num_states, *obs_shape)
        self.group_output_ids = torch.zeros(num_states, group_size, max_output_len).long()
        self.group_log_probs = torch.zeros(num_states, group_size)
        self.group_rewards = torch.zeros(num_states, group_size)

        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.group_output_ids = self.group_output_ids.to(device)
        self.group_log_probs = self.group_log_probs.to(device)
        self.group_rewards = self.group_rewards.to(device)

    def insert(self, obs, group_output_ids, group_log_probs, group_rewards):
        if obs.dim() == len(self.obs.shape):
            self.obs[self.step].copy_(obs[0] if obs.shape[0] == 1 else obs)
        else:
            self.obs[self.step].copy_(obs)
        self.group_output_ids[self.step].copy_(group_output_ids)
        self.group_log_probs[self.step].copy_(group_log_probs)
        self.group_rewards[self.step].copy_(group_rewards)
        self.step = (self.step + 1) % self.num_states

    def after_update(self):
        self.step = 0

    def feed_forward_generator(self, flat_advantages, mini_batch_size):
        total_samples = self.num_states * self.group_size

        flat_obs = self.obs.unsqueeze(1).expand(-1, self.group_size, *self.obs.shape[1:]).reshape(total_samples, *self.obs.shape[1:])
        flat_output_ids = self.group_output_ids.view(total_samples, -1)
        flat_log_probs = self.group_log_probs.view(total_samples)

        sampler = BatchSampler(
            SubsetRandomSampler(range(total_samples)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:
            obs_batch = flat_obs[indices]
            output_ids_batch = flat_output_ids[indices]
            old_log_probs_batch = flat_log_probs[indices]
            adv_targ = flat_advantages[indices]

            yield obs_batch, output_ids_batch, old_log_probs_batch, adv_targ
