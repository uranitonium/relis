"""Mask-aware rolling window ops."""

from __future__ import annotations

from typing import Tuple

import torch


def rolling_shift(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return x
    pad = torch.zeros_like(x[:, :k])
    return torch.cat([pad, x[:, :-k]], dim=1)


def _window_sum(x: torch.Tensor, mask: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.float()
    while mask_f.dim() < x.dim():
        mask_f = mask_f.unsqueeze(-1)
    x_masked = x * mask_f
    csum = torch.cumsum(x_masked, dim=1)
    msum = torch.cumsum(mask_f, dim=1)
    if k <= 0:
        return csum, msum
    csum_shift = rolling_shift(csum, k)
    msum_shift = rolling_shift(msum, k)
    return csum - csum_shift, msum - msum_shift


def rolling_mean(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Window is [t-k+1, t]."""
    sum_x, count = _window_sum(x, mask, k)
    denom = count.clamp_min(1.0)
    return sum_x / denom


def rolling_var(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Variance over window, averaged over feature dim."""
    sum_x, count = _window_sum(x, mask, k)
    sum_x2, _ = _window_sum(x ** 2, mask, k)
    denom = count.clamp_min(1.0)
    mean = sum_x / denom
    mean2 = sum_x2 / denom
    var = (mean2 - mean ** 2).clamp_min(0.0)
    return var.mean(dim=-1)
