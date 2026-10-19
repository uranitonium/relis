from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "boundary_segmentation"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from models.encoder import LSTMEncoder  # noqa: E402


def test_lstm_encoder_ignores_padded_suffix() -> None:
    torch.manual_seed(7)
    encoder = LSTMEncoder(
        input_dim=5,
        hidden_dim=12,
        n_layers=3,
        dropout=0.0,
        bidirectional=True,
    ).eval()
    mask = torch.tensor(
        [[True, True, True, True, True, True], [True, True, True, False, False, False]]
    )
    x = torch.randn(2, 6, 5)
    altered = x.clone()
    altered[1, 3:] = 1000.0

    with torch.no_grad():
        original_h = encoder(x, mask)
        altered_h = encoder(altered, mask)

    assert original_h.shape == (2, 6, 12)
    assert torch.equal(original_h[1, 3:], torch.zeros_like(original_h[1, 3:]))
    assert torch.allclose(original_h[1, :3], altered_h[1, :3], atol=1e-7)
