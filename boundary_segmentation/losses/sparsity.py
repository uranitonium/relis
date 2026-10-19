"""Boundary sparsity loss."""

from __future__ import annotations

import torch


def sparsity_loss(b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (b * mask_f).sum() / denom
