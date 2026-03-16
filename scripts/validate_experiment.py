#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from vrl_dissertation.config import load_experiment_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dissertation experiment config")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args()

    try:
        cfg = load_experiment_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"[INVALID] {args.config}: {exc}")
        return 1

    print("[VALID] Config parsed successfully")
    print(json.dumps(cfg.__dict__, indent=2, default=lambda o: o.__dict__))
    return 0


if __name__ == "__main__":
    sys.exit(main())
