"""Temporal encoders."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.pos(x)
        key_padding_mask = ~mask.bool()
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        h = h * mask.float().unsqueeze(-1)
        return h


class LSTMEncoder(nn.Module):
    """Offline bidirectional LSTM encoder with padding-aware recurrence."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        directions = 2 if bidirectional else 1
        if bidirectional and hidden_dim % directions:
            raise ValueError("hidden_dim must be even for a bidirectional LSTM")
        recurrent_dim = hidden_dim // directions
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=recurrent_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Dataset masks are valid prefixes followed by batch padding. Packing
        # prevents the reverse LSTM from observing padded suffixes.
        lengths = mask.long().sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_h, batch_first=True, total_length=x.shape[1]
        )
        h = self.norm(h)
        return h * mask.float().unsqueeze(-1)


class CameraDualTransformerEncoder(nn.Module):
    """Encode raw/calibrated views separately, then fuse camera residual evidence."""

    def __init__(
        self,
        input_dim: int,
        view_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if input_dim != 2 * view_dim:
            raise ValueError(
                f"camera-dual input_dim must be twice view_dim ({2 * view_dim}), "
                f"got {input_dim}"
            )
        self.view_dim = int(view_dim)
        self.shared = TransformerEncoder(
            input_dim=self.view_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_len=max_len,
        )
        self.camera_residual = nn.Sequential(
            nn.Linear(self.view_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.constant_(self.gate[-2].bias, gate_bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raw = x[..., : self.view_dim]
        calibrated = x[..., self.view_dim :]
        raw_h = self.shared(raw, mask)
        calibrated_h = self.shared(calibrated, mask)
        difference = calibrated_h - raw_h
        gate = self.gate(torch.cat([raw_h, calibrated_h, difference.abs()], dim=-1))
        camera_h = self.camera_residual(calibrated - raw)
        h = (1.0 - gate) * raw_h + gate * calibrated_h
        h = h + torch.tanh(self.residual_scale) * camera_h
        h = self.norm(h)
        return h * mask.float().unsqueeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.act(self.conv2(x)))
        return x + res


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        kernel_size: int,
        dilation_base: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length")
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        blocks = []
        for i in range(layers):
            dilation = dilation_base ** i
            blocks.append(TCNBlock(hidden_dim, kernel_size, dilation, dropout))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x * mask.float().unsqueeze(-1)
        h = self.in_proj(x)
        h = h.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)
        h = h * mask.float().unsqueeze(-1)
        return h


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length")
        self.register_buffer("adjacency", adjacency)
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        agg = torch.einsum("vw,bctw->bctv", self.adjacency, x)
        out = self.spatial(agg)
        out = self.temporal(out)
        return out + self.residual(x)


def two_skeleton_adjacency(num_fencers: int = 2, num_joints: int = 12) -> torch.Tensor:
    if num_fencers != 2 or num_joints != 12:
        raise ValueError("The paper ST-GCN baseline expects two 12-joint skeletons")
    edges = [
        (0, 1),
        (0, 2),
        (2, 4),
        (1, 3),
        (3, 5),
        (6, 7),
        (6, 8),
        (8, 10),
        (7, 9),
        (9, 11),
        (0, 6),
        (1, 7),
    ]
    total = num_fencers * num_joints
    a = torch.eye(total, dtype=torch.float32)
    for fencer in range(num_fencers):
        offset = fencer * num_joints
        for i, j in edges:
            a[offset + i, offset + j] = 1.0
            a[offset + j, offset + i] = 1.0
    for joint in range(num_joints):
        a[joint, num_joints + joint] = 1.0
        a[num_joints + joint, joint] = 1.0
    deg = a.sum(dim=1).clamp_min(1.0)
    d_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    return d_inv_sqrt @ a @ d_inv_sqrt


class STGCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        num_fencers: int = 2,
        num_joints: int = 12,
        coord_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_fencers = int(num_fencers)
        self.num_joints = int(num_joints)
        self.coord_dim = int(coord_dim)
        expected = self.num_fencers * self.num_joints * self.coord_dim
        if input_dim != expected:
            raise ValueError(f"ST-GCN input_dim must be {expected}, got {input_dim}")
        adjacency = two_skeleton_adjacency(self.num_fencers, self.num_joints)
        blocks = []
        in_channels = self.coord_dim
        for _ in range(layers):
            blocks.append(STGCNBlock(in_channels, hidden_dim, adjacency, kernel_size, dropout))
            in_channels = hidden_dim
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, t, _ = x.shape
        x = x.reshape(bsz, t, self.num_fencers * self.num_joints, self.coord_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x * mask.float().view(bsz, 1, t, 1)
        for block in self.blocks:
            x = block(x)
            x = x * mask.float().view(bsz, 1, t, 1)
        h = x.mean(dim=-1).transpose(1, 2).contiguous()
        h = self.norm(h)
        h = h * mask.float().unsqueeze(-1)
        return h
