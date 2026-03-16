from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int, log_std_min: float, log_std_max: float) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(z)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(z)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()

        def q_net() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feature_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = q_net()
        self.q2 = q_net()

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([z, action], dim=-1)
        return self.q1(h), self.q2(h)
