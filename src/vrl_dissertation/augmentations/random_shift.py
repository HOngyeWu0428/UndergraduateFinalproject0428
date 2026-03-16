from __future__ import annotations

import torch
import torch.nn.functional as F

from vrl_dissertation.augmentations.base import Augmentation


class RandomShiftAugmentation(Augmentation):
    """Random shift augmentation from DrQ-style visual RL training.

    Input tensor format: (N, C, H, W), values usually in [0, 1].
    """

    def __init__(self, pad: int = 4) -> None:
        super().__init__()
        if pad <= 0:
            raise ValueError(f"pad must be > 0 for random shift, got {pad}")
        self.pad = pad

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(f"RandomShift expects (N,C,H,W), got shape={tuple(obs.shape)}")

        n, _, h, w = obs.shape
        padded = F.pad(obs, (self.pad, self.pad, self.pad, self.pad), mode="replicate")

        eps_h = 2.0 / (h + 2 * self.pad - 1)
        eps_w = 2.0 / (w + 2 * self.pad - 1)

        base_y = torch.linspace(-1.0 + eps_h * self.pad, 1.0 - eps_h * self.pad, h, device=obs.device)
        base_x = torch.linspace(-1.0 + eps_w * self.pad, 1.0 - eps_w * self.pad, w, device=obs.device)
        grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(n, 1, 1, 1)

        shift_x = torch.randint(-self.pad, self.pad + 1, (n, 1, 1, 1), device=obs.device, dtype=torch.float32)
        shift_y = torch.randint(-self.pad, self.pad + 1, (n, 1, 1, 1), device=obs.device, dtype=torch.float32)

        shift_x = shift_x * (2.0 / (w + 2 * self.pad - 1))
        shift_y = shift_y * (2.0 / (h + 2 * self.pad - 1))

        grid = base_grid.clone()
        grid[..., 0] = grid[..., 0] + shift_x.squeeze(-1)
        grid[..., 1] = grid[..., 1] + shift_y.squeeze(-1)

        return F.grid_sample(
            padded,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
