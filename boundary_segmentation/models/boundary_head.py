"""Boundary head."""

from __future__ import annotations

import torch
import torch.nn as nn


class BoundaryHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 5,
        init_bias: float | None = None,
        skip_alpha: float = 0.5,
        temperature: float = 3.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.skip_alpha = skip_alpha
        self.temperature = temperature
        self.skip = nn.Linear(in_dim, 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        if init_bias is not None:
            nn.init.constant_(self.net[-1].bias, init_bias)
            nn.init.constant_(self.skip.bias, init_bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        phi_n = self.norm(phi)
        phi_centered = phi - phi.mean(dim=1, keepdim=True)
        out = self.net(phi_n) + self.skip_alpha * self.skip(phi_centered)
        out = out * self.temperature
        return torch.sigmoid(out).squeeze(-1)
