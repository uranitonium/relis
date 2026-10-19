"""Slow variable head."""

from __future__ import annotations

import torch
import torch.nn as nn


class SlowVarsHead(nn.Module):
    def __init__(self, hidden_dim: int, slow_dim: int = 3) -> None:
        super().__init__()
        self.slow_dim = int(slow_dim)
        if self.slow_dim > 0:
            self.linear = nn.Linear(hidden_dim, self.slow_dim)
        else:
            self.linear = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            return h.new_zeros((*h.shape[:2], 0))
        return self.linear(h)
