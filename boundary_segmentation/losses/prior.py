"""Boundary mass prior loss."""

from __future__ import annotations

import torch


def boundary_mass_prior_loss(b: torch.Tensor, mask: torch.Tensor, target: float) -> torch.Tensor:
    """Keep mean(b) near a target to avoid trivial collapse."""
    if target <= 0.0:
        return torch.tensor(0.0, device=b.device)
    mask_f = mask.float()
    mean_b = (b * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return (mean_b - target) ** 2
