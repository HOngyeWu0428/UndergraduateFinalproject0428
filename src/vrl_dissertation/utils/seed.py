from __future__ import annotations

import importlib.util
import random


def set_seed(seed: int) -> None:
    random.seed(seed)

    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(seed)

    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
