"""Alignment loss between boundary score and phi magnitude."""

from __future__ import annotations

import torch


def boundary_phi_align_loss(
    b: torch.Tensor,
    phi: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    detach_target: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Encourage b to track normalized phi magnitude (unsupervised)."""
    mask_f = mask.float()
    phi_mag = phi.mean(dim=-1)
    # normalize per sequence over valid frames
    phi_mag = phi_mag * mask_f
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mu = phi_mag.sum(dim=1, keepdim=True) / denom
    var = ((phi_mag - mu) ** 2 * mask_f).sum(dim=1, keepdim=True) / denom
    std = (var + eps).sqrt()
    phi_norm = (phi_mag - mu) / std
    target = torch.sigmoid(temperature * phi_norm)
    if detach_target:
        target = target.detach()
    diff = (b - target) ** 2 * mask_f
    return diff.sum() / mask_f.sum().clamp_min(1.0)
