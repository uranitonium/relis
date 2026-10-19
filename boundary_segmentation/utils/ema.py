"""EMA helpers."""

from __future__ import annotations

import torch


def ema_smooth(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Simple EMA over last dimension (time)."""
    if x.numel() == 0:
        return x
    out = torch.zeros_like(x)
    out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = alpha * x[:, t] + (1.0 - alpha) * out[:, t - 1]
    return out
