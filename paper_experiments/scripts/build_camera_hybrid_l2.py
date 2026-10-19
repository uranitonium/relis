#!/usr/bin/env python3
"""Build L2 inputs with raw geometry and camera-calibrated motion features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


DEFAULT_CALIBRATED_FEATURES = [
    "vL_toward",
    "vR_toward",
    "vL_lateral",
    "vR_lateral",
    "net_closing",
    "hand_lead_L",
    "hand_lead_R",
    "stop_rate_L",
    "stop_rate_R",
    "stopgo_transitions_L",
    "stopgo_transitions_R",
    "hand_forward_L",
    "hand_forward_R",
    "corr_speed",
    "corr_hand",
]


def load_l2(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_hybrid(raw: Dict[str, Any], calibrated: Dict[str, Any], features: List[str]) -> Dict[str, Any]:
    raw_names = list(raw["feature_names"])
    calibrated_names = list(calibrated["feature_names"])
    if raw_names != calibrated_names:
        raise ValueError("Raw and calibrated feature names differ")
    unknown = sorted(set(features) - set(raw_names))
    if unknown:
        raise ValueError(f"Unknown calibrated features: {', '.join(unknown)}")

    raw_l2 = np.asarray(raw["l2"], dtype=np.float32)
    calibrated_l2 = np.asarray(calibrated["l2"], dtype=np.float32)
    if raw_l2.shape != calibrated_l2.shape:
        raise ValueError(f"Raw/calibrated L2 shape mismatch: {raw_l2.shape} != {calibrated_l2.shape}")
    if list(raw.get("frames", [])) != list(calibrated.get("frames", [])):
        raise ValueError("Raw and calibrated frames differ")

    hybrid_l2 = raw_l2.copy()
    calibrated_indices = [raw_names.index(name) for name in features]
    hybrid_l2[:, calibrated_indices] = calibrated_l2[:, calibrated_indices]
    mask = np.asarray(raw["l2_mask"], dtype=bool) & np.asarray(
        calibrated["l2_mask"], dtype=bool
    )
    hybrid_l2[~mask] = 0.0
    output = dict(raw)
    output["l2"] = hybrid_l2.tolist()
    output["l2_mask"] = mask.tolist()
    output["meta"] = dict(raw.get("meta", {}))
    output["meta"]["camera_hybrid"] = {
        "calibrated_features": features,
        "raw_features": [name for name in raw_names if name not in features],
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--calibrated-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--calibrated-feature",
        action="append",
        default=[],
        help="Feature replaced by its calibrated value; repeat as needed.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    calibrated_dir = Path(args.calibrated_dir)
    out_dir = Path(args.out_dir)
    features = args.calibrated_feature or DEFAULT_CALIBRATED_FEATURES
    raw_paths = sorted(raw_dir.glob("*_l2.json"))
    if not raw_paths:
        raise RuntimeError(f"No L2 inputs found in {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in raw_paths:
        calibrated_path = calibrated_dir / raw_path.name
        if not calibrated_path.exists():
            raise FileNotFoundError(calibrated_path)
        try:
            output = build_hybrid(load_l2(raw_path), load_l2(calibrated_path), features)
        except ValueError as error:
            raise ValueError(f"{raw_path.name}: {error}") from error
        with (out_dir / raw_path.name).open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
            handle.write("\n")
    print(
        json.dumps(
            {"clips": len(raw_paths), "calibrated_features": features, "out_dir": str(out_dir)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
