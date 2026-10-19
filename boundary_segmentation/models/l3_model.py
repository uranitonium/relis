"""Full L3 boundary model."""

from __future__ import annotations

from typing import Dict

import math
import torch
import torch.nn as nn

from features.phi import compute_phi
from models.boundary_head import BoundaryHead
from models.encoder import (
    CameraDualTransformerEncoder,
    LSTMEncoder,
    STGCNEncoder,
    TCNEncoder,
    TransformerEncoder,
)
from models.projection import ProjectionHead
from models.slow_vars import SlowVarsHead


class L3BoundaryModel(nn.Module):
    def __init__(self, input_dim: int, cfg: Dict) -> None:
        super().__init__()
        enc_cfg = cfg["model"]
        hidden_dim = enc_cfg["hidden_dim"]
        camera_residual_cfg = enc_cfg.get("camera_residual_fusion", {})
        self.camera_residual_fusion = bool(camera_residual_cfg.get("enabled", False))
        self.camera_view_dim = int(camera_residual_cfg.get("view_dim", input_dim // 2))
        if self.camera_residual_fusion and enc_cfg["encoder_type"] == "camera_dual_transformer":
            raise ValueError(
                "camera_residual_fusion and camera_dual_transformer are separate fusion modes"
            )
        encoder_input_dim = input_dim
        if self.camera_residual_fusion:
            if input_dim != 2 * self.camera_view_dim:
                raise ValueError(
                    "camera-residual input_dim must be twice view_dim "
                    f"({2 * self.camera_view_dim}), got {input_dim}"
                )
            encoder_input_dim = self.camera_view_dim
            initial_mix = float(camera_residual_cfg.get("initial_mix", 0.01))
            initial_mix = min(max(initial_mix, 1e-6), 1.0 - 1e-6)
            initial_logit = math.log(initial_mix / (1.0 - initial_mix))
            self.camera_feature_logits = nn.Parameter(
                torch.full((self.camera_view_dim,), initial_logit)
            )
            self.camera_context = nn.Linear(self.camera_view_dim, self.camera_view_dim)
            nn.init.zeros_(self.camera_context.weight)
            nn.init.zeros_(self.camera_context.bias)
        if enc_cfg["encoder_type"] == "transformer":
            tcfg = enc_cfg["transformer"]
            self.encoder = TransformerEncoder(
                input_dim=encoder_input_dim,
                hidden_dim=hidden_dim,
                n_layers=tcfg["n_layers"],
                n_heads=tcfg["n_heads"],
                ff_dim=tcfg["ff_dim"],
                dropout=tcfg["dropout"],
                max_len=tcfg["max_len"],
            )
        elif enc_cfg["encoder_type"] == "lstm":
            lcfg = enc_cfg.get("lstm", {})
            self.encoder = LSTMEncoder(
                input_dim=encoder_input_dim,
                hidden_dim=hidden_dim,
                n_layers=int(lcfg.get("n_layers", 3)),
                dropout=float(lcfg.get("dropout", 0.1)),
                bidirectional=bool(lcfg.get("bidirectional", True)),
            )
        elif enc_cfg["encoder_type"] == "camera_dual_transformer":
            tcfg = enc_cfg["transformer"]
            camera_cfg = enc_cfg.get("camera_dual", {})
            self.encoder = CameraDualTransformerEncoder(
                input_dim=input_dim,
                view_dim=int(camera_cfg.get("view_dim", input_dim // 2)),
                hidden_dim=hidden_dim,
                n_layers=tcfg["n_layers"],
                n_heads=tcfg["n_heads"],
                ff_dim=tcfg["ff_dim"],
                dropout=tcfg["dropout"],
                max_len=tcfg["max_len"],
                gate_bias=float(camera_cfg.get("gate_bias", -2.0)),
            )
        elif enc_cfg["encoder_type"] == "tcn":
            tcfg = enc_cfg["tcn"]
            self.encoder = TCNEncoder(
                input_dim=encoder_input_dim,
                hidden_dim=hidden_dim,
                layers=tcfg["layers"],
                kernel_size=tcfg["kernel_size"],
                dilation_base=tcfg["dilation_base"],
                dropout=tcfg["dropout"],
            )
        elif enc_cfg["encoder_type"] == "stgcn":
            scfg = enc_cfg.get("stgcn", {})
            self.encoder = STGCNEncoder(
                input_dim=encoder_input_dim,
                hidden_dim=hidden_dim,
                layers=scfg.get("layers", 3),
                kernel_size=scfg.get("kernel_size", 5),
                dropout=scfg.get("dropout", 0.1),
                num_fencers=scfg.get("num_fencers", 2),
                num_joints=scfg.get("num_joints", 12),
                coord_dim=scfg.get("coord_dim", 2),
            )
        else:
            raise ValueError(f"Unknown encoder_type: {enc_cfg['encoder_type']}")

        self.proj = ProjectionHead(hidden_dim, enc_cfg["latent_dim"], enc_cfg["proj"]["l2_norm"])
        self.slow_dim = int(enc_cfg.get("slow_dim", 3))
        self.slow = SlowVarsHead(hidden_dim, self.slow_dim)
        self.input_zero_indices = [int(i) for i in enc_cfg.get("input_zero_indices", [])]
        init_bias = cfg["model"]["boundary"].get("init_bias", None)
        skip_alpha = cfg["model"]["boundary"].get("skip_alpha", 0.5)
        temperature = cfg["model"]["boundary"].get("temperature", 3.0)
        self.use_critical = cfg["model"]["boundary"].get("use_critical", False)
        self.critical_index = cfg["model"]["boundary"].get("critical_index", None)
        phi_dim = 2 + min(self.slow_dim, 3) + max(0, min(self.slow_dim, 5) - 3) + (3 if self.use_critical else 0)
        self.boundary = BoundaryHead(
            in_dim=phi_dim,
            init_bias=init_bias,
            skip_alpha=skip_alpha,
            temperature=temperature,
        )
        self.k = cfg["model"]["boundary"]["k"]

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x_model = self.fuse_camera_views(x) if self.camera_residual_fusion else x
        if self.input_zero_indices:
            x_model = x_model.clone()
            d = int(x_model.shape[-1])
            for idx in self.input_zero_indices:
                if 0 <= int(idx) < d:
                    x_model[..., int(idx)] = 0.0
        h = self.encoder(x_model, mask)
        z = self.proj(h)
        slow = self.slow(h)
        critical_ratio = None
        if self.use_critical:
            if self.critical_index is None:
                raise ValueError("critical_index must be set when use_critical is True")
            critical_ratio = x_model[..., int(self.critical_index)]
        phi = compute_phi(z, slow, mask, self.k, critical_ratio=critical_ratio, use_critical=self.use_critical)
        b = self.boundary(phi) * mask.float()
        return h, z, slow, phi, b

    def fuse_camera_views(self, x: torch.Tensor) -> torch.Tensor:
        if not self.camera_residual_fusion:
            return x
        raw = x[..., : self.camera_view_dim]
        calibrated = x[..., self.camera_view_dim :]
        residual = calibrated - raw
        context = self.camera_context(residual.abs())
        mix = torch.sigmoid(self.camera_feature_logits + context)
        return raw + mix * residual
