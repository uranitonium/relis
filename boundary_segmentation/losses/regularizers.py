"""Auxiliary regularizers to prevent representation collapse."""

from __future__ import annotations

import torch


def z_var_floor_loss(z: torch.Tensor, mask: torch.Tensor, target_var: float) -> torch.Tensor:
    """Penalize variance below a target (per feature dim)."""
    if target_var <= 0:
        return torch.tensor(0.0, device=z.device)
    mask_f = mask.float().unsqueeze(-1)
    denom = mask_f.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (z * mask_f).sum(dim=(0, 1)) / denom
    var = ((z - mean) ** 2 * mask_f).sum(dim=(0, 1)) / denom
    loss = torch.relu(target_var - var).mean()
    return loss
