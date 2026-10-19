"""Regime consistency loss."""

from __future__ import annotations

import torch

from features.rolling import rolling_mean


def consistency_loss(z: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    w = (1.0 - b).clamp(0.0, 1.0)
    mu = rolling_mean(z, mask, k)
    diff = (z - mu) ** 2
    diff = diff.sum(dim=-1)
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (w * diff * mask_f).sum() / denom
