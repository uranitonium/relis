from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
L3 = ROOT / "boundary_segmentation"
SCRIPTS = ROOT / "paper_experiments" / "scripts"
for path in (L3, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_camera_hybrid_l2 import build_hybrid  # noqa: E402
from models.encoder import CameraDualTransformerEncoder  # noqa: E402
from models.l3_model import L3BoundaryModel  # noqa: E402


def test_camera_hybrid_replaces_only_declared_features() -> None:
    raw = {
        "l2": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "l2_mask": [True, True],
        "frames": [10, 11],
        "feature_names": ["geometry", "motion", "critical"],
    }
    calibrated = {
        "l2": [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        "l2_mask": [True, False],
        "frames": [10, 11],
        "feature_names": raw["feature_names"],
    }

    hybrid = build_hybrid(raw, calibrated, ["motion"])

    np.testing.assert_allclose(hybrid["l2"], [[1.0, 20.0, 3.0], [0.0, 0.0, 0.0]])
    assert hybrid["l2_mask"] == [True, False]


def test_camera_dual_encoder_fuses_aligned_views() -> None:
    encoder = CameraDualTransformerEncoder(
        input_dim=8,
        view_dim=4,
        hidden_dim=16,
        n_layers=1,
        n_heads=4,
        ff_dim=32,
        dropout=0.0,
        max_len=20,
    )
    x = torch.randn(2, 6, 8)
    mask = torch.tensor([[True] * 6, [True, True, True, False, False, False]])

    output = encoder(x, mask)

    assert output.shape == (2, 6, 16)
    assert torch.count_nonzero(output[1, 3:]) == 0


def test_camera_dual_encoder_rejects_misaligned_dimensions() -> None:
    with pytest.raises(ValueError, match="twice view_dim"):
        CameraDualTransformerEncoder(
            input_dim=7,
            view_dim=4,
            hidden_dim=16,
            n_layers=1,
            n_heads=4,
            ff_dim=32,
            dropout=0.0,
            max_len=20,
        )


def test_camera_residual_fusion_starts_near_raw_view() -> None:
    cfg = {
        "model": {
            "encoder_type": "transformer",
            "hidden_dim": 16,
            "latent_dim": 8,
            "camera_residual_fusion": {
                "enabled": True,
                "view_dim": 3,
                "initial_mix": 0.001,
            },
            "boundary": {"k": 2, "use_critical": False},
            "proj": {"l2_norm": True},
            "transformer": {
                "n_layers": 1,
                "n_heads": 4,
                "ff_dim": 32,
                "dropout": 0.0,
                "max_len": 20,
            },
            "input_zero_indices": [],
        }
    }
    model = L3BoundaryModel(input_dim=6, cfg=cfg)
    raw = torch.randn(2, 5, 3)
    calibrated = raw + 10.0

    fused = model.fuse_camera_views(torch.cat([raw, calibrated], dim=-1))

    torch.testing.assert_close(fused, raw + 0.01, atol=1e-5, rtol=1e-5)


def test_camera_residual_fusion_applies_feature_zeroing_after_fusion() -> None:
    cfg = {
        "model": {
            "encoder_type": "transformer",
            "hidden_dim": 16,
            "latent_dim": 8,
            "camera_residual_fusion": {
                "enabled": True,
                "view_dim": 3,
                "initial_mix": 0.001,
            },
            "boundary": {"k": 2, "use_critical": True, "critical_index": 1},
            "proj": {"l2_norm": True},
            "transformer": {
                "n_layers": 1,
                "n_heads": 4,
                "ff_dim": 32,
                "dropout": 0.0,
                "max_len": 20,
            },
            "input_zero_indices": [0],
        }
    }
    model = L3BoundaryModel(input_dim=6, cfg=cfg)
    x = torch.randn(2, 5, 6)
    mask = torch.ones(2, 5, dtype=torch.bool)

    _, _, _, _, boundary = model(x, mask)

    assert boundary.shape == (2, 5)
