from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.full = False

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        max_idx = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_idx, size=batch_size)

        obs = torch.as_tensor(self.obs[idxs], device=self.device).float() / 255.0
        next_obs = torch.as_tensor(self.next_obs[idxs], device=self.device).float() / 255.0
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        dones = torch.as_tensor(self.dones[idxs], device=self.device)
        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
        }
