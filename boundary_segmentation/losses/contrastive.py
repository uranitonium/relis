"""Optional contrastive stability loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss(z: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if z.shape[1] < 2:
        return torch.tensor(0.0, device=z.device)
    w = (1.0 - b[:, :-1]) * (1.0 - b[:, 1:])
    valid = mask[:, :-1] & mask[:, 1:] & (w > 0)
    if valid.sum() < 2:
        return torch.tensor(0.0, device=z.device)
    anchors = z[:, :-1][valid]
    positives = z[:, 1:][valid]
    weights = w[valid]
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    logits = anchors @ positives.t()
    logits = logits / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)
