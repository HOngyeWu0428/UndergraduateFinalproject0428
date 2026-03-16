#!/usr/bin/env python3
from __future__ import annotations

import argparse

from vrl_dissertation.baseline_config import load_baseline_config
from vrl_dissertation.trainers.sac_trainer import SACTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pixel SAC baseline without augmentation")
    parser.add_argument("--config", required=True, help="Path to baseline JSON config")
    args = parser.parse_args()

    cfg = load_baseline_config(args.config)
    trainer = SACTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
