"""Boundary smoothing."""

from __future__ import annotations

import numpy as np


def ema_smooth(b: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    if b.size == 0:
        return b
    out = np.zeros_like(b, dtype=np.float32)
    out[0] = b[0]
    for t in range(1, b.shape[0]):
        out[t] = alpha * b[t] + (1.0 - alpha) * out[t - 1]
    return out
