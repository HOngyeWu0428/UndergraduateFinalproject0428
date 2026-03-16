from __future__ import annotations

import torch
import torch.nn as nn


class PixelEncoder(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], feature_dim: int) -> None:
        super().__init__()
        c, h, w = obs_shape
        assert h == 84 and w == 84, "This baseline expects 84x84 observations"
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c, h, w)).view(1, -1).shape[1]

        self.fc = nn.Sequential(nn.Linear(n_flat, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        return self.fc(x)
