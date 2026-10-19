"""Boundary smoothness loss."""

from __future__ import annotations

import torch


def smoothness_loss(b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if b.shape[1] < 2:
        return torch.tensor(0.0, device=b.device)
    b_prev = b[:, :-1]
    b_next = b[:, 1:]
    pair_mask = mask[:, :-1] & mask[:, 1:]
    denom = pair_mask.float().sum().clamp_min(1.0)
    diff = (b_next - b_prev).abs()
    return (diff * pair_mask.float()).sum() / denom
