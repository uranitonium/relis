"""Boundary variance regularizer."""

from __future__ import annotations

import torch


def b_variance_floor_loss(b: torch.Tensor, mask: torch.Tensor, target_var: float) -> torch.Tensor:
    """Penalize variance of b below a target to encourage non-flat boundaries."""
    if target_var <= 0.0:
        return torch.tensor(0.0, device=b.device)
    mask_f = mask.float()
    mean_b = (b * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    var_b = ((b - mean_b) ** 2 * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return torch.relu(target_var - var_b)
