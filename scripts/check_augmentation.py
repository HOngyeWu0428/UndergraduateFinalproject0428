#!/usr/bin/env python3
from __future__ import annotations

import torch

from vrl_dissertation.augmentations.random_shift import RandomShiftAugmentation


def main() -> None:
    torch.manual_seed(0)
    aug = RandomShiftAugmentation(pad=4)

    # Construct deterministic toy input with visible spatial pattern.
    obs = torch.arange(0, 84 * 84, dtype=torch.float32).view(1, 1, 84, 84) / (84 * 84)
    obs = obs.repeat(2, 3, 1, 1)

    out1 = aug(obs)
    out2 = aug(obs)

    print("input_shape:", tuple(obs.shape))
    print("output_shape:", tuple(out1.shape))
    print("mean_abs_diff_input_vs_out1:", float((obs - out1).abs().mean().item()))
    print("mean_abs_diff_out1_vs_out2:", float((out1 - out2).abs().mean().item()))

    if out1.shape != obs.shape:
        raise SystemExit("FAILED: output shape mismatch")
    if torch.allclose(obs, out1):
        raise SystemExit("FAILED: augmentation made no change")
    if torch.allclose(out1, out2):
        raise SystemExit("FAILED: augmentation is not stochastic")

    print("PASSED: random shift augmentation is active and stochastic")


if __name__ == "__main__":
    main()
