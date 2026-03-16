from __future__ import annotations

import numpy as np


class TemporalInterruption:
    """Stateful temporal corruption handler.

    Supported modes:
    - frame_freeze: keep old frame with probability p
    - blackout: replace frame by zeros with probability p
    - frame_delay: output frame from k steps ago
    """

    def __init__(self, mode: str, params: dict, rng: np.random.Generator) -> None:
        self.mode = mode
        self.params = params
        self.rng = rng
        self._prev: np.ndarray | None = None
        self._buffer: list[np.ndarray] = []

    def reset(self) -> None:
        self._prev = None
        self._buffer.clear()

    def apply(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "frame_freeze":
            p = float(self.params["freeze_prob"])
            if self._prev is not None and self.rng.random() < p:
                return self._prev.copy()
            self._prev = obs.copy()
            return obs

        if self.mode == "blackout":
            p = float(self.params["blackout_prob"])
            if self.rng.random() < p:
                return np.zeros_like(obs)
            return obs

        if self.mode == "frame_delay":
            k = int(self.params["delay_k"])
            self._buffer.append(obs.copy())
            if len(self._buffer) <= k:
                return self._buffer[0].copy()
            delayed = self._buffer[-(k + 1)].copy()
            if len(self._buffer) > (k + 2):
                self._buffer.pop(0)
            return delayed

        raise ValueError(f"Unknown temporal mode: {self.mode}")
