from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


L3 = Path(__file__).resolve().parents[2] / "boundary_segmentation"
if str(L3) not in sys.path:
    sys.path.insert(0, str(L3))

from data.collate import collate_pad_pairs  # noqa: E402
from data.dataset import PairedL2SequenceDataset  # noqa: E402


def write_l2(
    path: Path,
    rows: list[list[float]],
    mask: list[bool],
    frames: list[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "l2": rows,
                "l2_mask": mask,
                "frames": frames if frames is not None else list(range(len(rows))),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_paired_dataset_aligns_views_and_intersects_masks(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    stable_dir = tmp_path / "stable"
    write_l2(raw_dir / "clip_l2.json", [[1.0, 2.0], [3.0, 4.0]], [True, True])
    write_l2(stable_dir / "clip_l2.json", [[0.5, 2.0], [2.5, 4.0]], [True, False])

    dataset = PairedL2SequenceDataset(str(raw_dir), str(stable_dir))
    item = dataset[0]

    np.testing.assert_allclose(item["x_pair"], [[0.5, 2.0], [2.5, 4.0]])
    np.testing.assert_array_equal(item["mask"], [True, False])
    batch = collate_pad_pairs([item])
    assert tuple(batch["x"].shape) == (1, 2, 2)
    assert tuple(batch["x_pair"].shape) == (1, 2, 2)
    assert batch["mask_pair"].tolist() == [[True, False]]


def test_paired_dataset_rejects_missing_view(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    stable_dir = tmp_path / "stable"
    stable_dir.mkdir()
    write_l2(raw_dir / "clip_l2.json", [[1.0]], [True])

    try:
        PairedL2SequenceDataset(str(raw_dir), str(stable_dir))
    except FileNotFoundError as error:
        assert "clip_l2.json" in str(error)
    else:
        raise AssertionError("missing paired sequence should fail fast")


def test_paired_dataset_rejects_shifted_frames(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    stable_dir = tmp_path / "stable"
    write_l2(raw_dir / "clip_l2.json", [[1.0], [2.0]], [True, True], [10, 11])
    write_l2(stable_dir / "clip_l2.json", [[1.0], [2.0]], [True, True], [11, 12])

    dataset = PairedL2SequenceDataset(str(raw_dir), str(stable_dir))
    try:
        dataset[0]
    except ValueError as error:
        assert "frames differ" in str(error)
    else:
        raise AssertionError("shifted paired sequence should fail fast")
