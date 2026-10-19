"""Projection head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, l2_norm: bool = False) -> None:
        super().__init__()
        self.l2_norm = l2_norm
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)
        if self.l2_norm:
            z = F.normalize(z, dim=-1)
        return z
