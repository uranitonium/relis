"""Mask-aware reductions."""

from __future__ import annotations

import torch


def _safe_div(numer: torch.Tensor, denom: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return numer / (denom.clamp_min(eps))


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.float()
    while mask_f.dim() < x.dim():
        mask_f = mask_f.unsqueeze(-1)
    numer = (x * mask_f).sum()
    denom = mask_f.sum()
    return _safe_div(numer, denom, eps)


def masked_mean_dim(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.float()
    while mask_f.dim() < x.dim():
        mask_f = mask_f.unsqueeze(-1)
    numer = (x * mask_f).sum(dim=dim)
    denom = mask_f.sum(dim=dim)
    return _safe_div(numer, denom, eps)


def masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    while mask_f.dim() < x.dim():
        mask_f = mask_f.unsqueeze(-1)
    return (x * mask_f).sum()


def masked_var(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    """Masked variance along dim."""
    mask_f = mask.float()
    while mask_f.dim() < x.dim():
        mask_f = mask_f.unsqueeze(-1)
    denom = mask_f.sum(dim=dim).clamp_min(eps)
    mean = (x * mask_f).sum(dim=dim) / denom
    mean = mean.unsqueeze(dim)
    var = ((x - mean) ** 2 * mask_f).sum(dim=dim) / denom
    return var
