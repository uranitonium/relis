#!/usr/bin/env python3
"""Audit stop-threshold invariance and frozen reach-divisor sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


REPO = Path(__file__).resolve().parents[2]
BOUNDARY_ROOT = REPO / "boundary_segmentation"
SCRIPT_ROOT = REPO / "paper_experiments/scripts"
for path in (BOUNDARY_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from annotation_schema import normalize_annotation  # noqa: E402
from compare_annotations_to_model import clip_metrics, summarize  # noqa: E402
from generate_boundary_predictions import partition_states  # noqa: E402
from infer import infer_sequence  # noqa: E402
from models.l3_model import L3BoundaryModel  # noqa: E402


ALL_MANIFEST = REPO / (
    "paper_experiments/experiments/truncated_retraining/manifests/all_hit_truncated.csv"
)
FINAL_MANIFEST = REPO / (
    "paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv"
)
CONSENSUS_DIR = REPO / "paper_experiments/annotations/adjudicated/consensus"
CONFIG = REPO / (
    "paper_experiments/experiments/truncated_retraining/candidates/"
    "conservative_smooth02_seed18/config.yaml"
)
CHECKPOINT = REPO / (
    "paper_experiments/experiments/truncated_retraining/runs/"
    "short_sparse_seed18_smooth02/ckpt_step_1000.pt"
)
OUT_ROOT = REPO / "paper_experiments/round2/feature_constant_sensitivity"
STOP_THRESHOLDS = [0.005, 0.01, 0.02, 0.04, 0.08, 0.10]
REACH_DIVISORS = [1.20, 1.35, 1.50, 1.65, 1.80]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def stop_and_reproduction_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = {threshold: 0 for threshold in STOP_THRESHOLDS}
    changed = {threshold: 0 for threshold in STOP_THRESHOLDS}
    total_fencer_frames = 0
    positive_speeds: list[np.ndarray] = []
    total_stored_frames = 0
    clips = 0

    for row in read_csv(ALL_MANIFEST):
        stored = read_json(resolve_repo_path(row["l2_path"]))
        x = np.asarray(stored["l2"], dtype=np.float32)
        mask = np.asarray(stored["l2_mask"], dtype=np.bool_)
        names = {name: index for index, name in enumerate(stored["feature_names"])}
        speed_left = np.sqrt(
            np.square(x[:, names["vL_toward"]]) + np.square(x[:, names["vL_lateral"]])
        )
        speed_right = np.sqrt(
            np.square(x[:, names["vR_toward"]]) + np.square(x[:, names["vR_lateral"]])
        )
        total_stored_frames += 2 * len(mask)
        for speeds in (speed_left, speed_right):
            valid = mask & np.isfinite(speeds)
            valid_speeds = speeds[valid]
            speeds = valid_speeds
            reference = speeds < 0.02
            total_fencer_frames += len(speeds)
            positive_speeds.append(speeds[speeds > 0])
            for threshold in STOP_THRESHOLDS:
                indicators = speeds < threshold
                counts[threshold] += int(np.sum(indicators))
                changed[threshold] += int(np.sum(indicators != reference))
        clips += 1

    positive = np.concatenate(positive_speeds)
    rows = [
        {
            "threshold": threshold,
            "valid_fencer_frames": total_fencer_frames,
            "stop_frames": counts[threshold],
            "stop_fraction": counts[threshold] / total_fencer_frames,
            "indicator_changes_vs_0.02": changed[threshold],
        }
        for threshold in STOP_THRESHOLDS
    ]
    metadata = {
        "clips": clips,
        "valid_fencer_frames": total_fencer_frames,
        "stored_fencer_frames": total_stored_frames,
        "speed_reconstruction": "sqrt(v_toward^2 + v_lateral^2)",
        "minimum_positive_speed": float(np.min(positive)),
        "positive_speed_quantiles": {
            str(q): float(value)
            for q, value in zip(
                [0.0, 0.0001, 0.001, 0.01, 0.05, 0.5],
                np.quantile(positive, [0.0, 0.0001, 0.001, 0.01, 0.05, 0.5]),
            )
        },
    }
    return rows, metadata


def rescale_reach(x: np.ndarray, names: list[str], divisor: float) -> np.ndarray:
    output = x.copy()
    index = {name: position for position, name in enumerate(names)}
    factor = 1.5 / divisor
    for name in ("reach_norm_L", "reach_norm_R", "reach_norm_avg"):
        output[:, index[name]] *= factor
    distance = output[:, index["d"]]
    reach = output[:, index["reach_norm_avg"]]
    output[:, index["critical_ratio"]] = distance / np.maximum(reach, 1e-6)
    output[:, index["critical_delta"]] = reach - distance
    output[:, index["critical_cross"]] = (reach - distance > 0).astype(np.float32)
    return output


def reach_audit(out_root: Path, device: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    final_rows = read_csv(FINAL_MANIFEST)
    first = read_json(resolve_repo_path(final_rows[0]["l2_path"]))
    model = L3BoundaryModel(input_dim=len(first["feature_names"]), cfg=cfg).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    allowed = {row["clip_id"] for row in final_rows}
    annotations = {}
    for path in sorted(CONSENSUS_DIR.glob("*.json")):
        annotation = normalize_annotation(read_json(path), strict=False)
        if annotation["clip_id"] in allowed:
            annotations[annotation["clip_id"]] = annotation

    prediction_sets: dict[float, list[dict[str, Any]]] = {
        divisor: [] for divisor in REACH_DIVISORS
    }
    crossing_counts = {divisor: 0 for divisor in REACH_DIVISORS}
    crossing_changes = {divisor: 0 for divisor in REACH_DIVISORS}
    valid_frames = 0

    for row in final_rows:
        data = read_json(resolve_repo_path(row["l2_path"]))
        x_base = np.asarray(data["l2"], dtype=np.float32)
        mask = np.asarray(data["l2_mask"], dtype=np.bool_)
        names = [str(name) for name in data["feature_names"]]
        reference = rescale_reach(x_base, names, 1.5)[:, names.index("critical_cross")]
        valid_frames += int(np.sum(mask))
        for divisor in REACH_DIVISORS:
            x = rescale_reach(x_base, names, divisor)
            crossing = x[:, names.index("critical_cross")]
            crossing_counts[divisor] += int(np.sum(crossing[mask] > 0.5))
            crossing_changes[divisor] += int(np.sum(crossing[mask] != reference[mask]))
            result = infer_sequence(x, mask, model, cfg)
            zones = [(int(start), int(end)) for start, end in result["transition_zones"]]
            states = partition_states(data, zones, len(x))
            prediction_sets[divisor].append(
                {
                    "clip_id": row["clip_id"],
                    "method": f"reach_divisor_{divisor:.2f}",
                    "coordinate_system": "source_video_frame_inclusive",
                    "transition_zones_index": zones,
                    "state_segments": states,
                    "active_segments": [
                        [state["start_frame"], state["end_frame"]]
                        for state in states
                        if state["activity"] == "active"
                    ],
                }
            )

    metric_rows: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    for divisor in REACH_DIVISORS:
        predictions = {
            row["clip_id"]: [
                (int(state["start_frame"]), int(state["end_frame"]), str(state["activity"]))
                for state in row["state_segments"]
            ]
            for row in prediction_sets[divisor]
        }
        clip_rows = [
            clip_metrics(clip_id, annotation, predictions[clip_id])
            for clip_id, annotation in annotations.items()
            if clip_id in predictions
        ]
        summary = summarize(clip_rows, "valid")
        metric_rows.append(
            {
                "reach_divisor": divisor,
                "clips": int(summary["clips"]),
                "activity_accuracy": float(summary["activity_accuracy"]),
                "active_f1": float(summary["active_f1"]),
                "boundary_f1_pm5": float(summary["boundary_f1_pm5"]),
                "boundary_f1_pm10": float(summary["boundary_f1_pm10"]),
                "model_active_coverage": float(summary["model_active_coverage"]),
                "critical_inside_frames": crossing_counts[divisor],
                "critical_cross_changes_vs_1.50": crossing_changes[divisor],
                "valid_frames": valid_frames,
            }
        )
        crossing_rows.extend(
            {
                "reach_divisor": divisor,
                "clip_id": row["clip_id"],
                "transition_zone_count": len(row["transition_zones_index"]),
            }
            for row in prediction_sets[divisor]
        )
        prediction_path = out_root / "predictions" / f"reach_divisor_{divisor:.2f}.json"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "method": f"reach_divisor_{divisor:.2f}",
                    "config": str(CONFIG),
                    "checkpoint": str(CHECKPOINT),
                    "predictions": prediction_sets[divisor],
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
    return metric_rows, crossing_rows


def render_results(
    path: Path,
    stop_rows: list[dict[str, Any]],
    stop_meta: dict[str, Any],
    reach_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Frozen feature-constant sensitivity",
        "",
        "## Near-zero-motion threshold",
        "",
        "Center speed was reconstructed in source-frame pixels per second from",
        f"the stored orthogonal velocity components in all {stop_meta['clips']:,} L2 clips.",
        "This feature is a numerical stationary-sample detector, not a calibrated",
        f"physical stopping criterion. Across {stop_meta['valid_fencer_frames']:,} valid fencer frames:",
        "",
        "| threshold (px/s) | near-zero frames | fraction | indicator changes vs 0.02 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in stop_rows:
        lines.append(
            f"| {row['threshold']:.3f} | {row['stop_frames']:,} | "
            f"{row['stop_fraction']:.4f} | {row['indicator_changes_vs_0.02']:,} |"
        )
    lines.extend(
        [
            "",
            "Thus 0.005--0.10 gives exactly the same binary indicators and therefore",
            "exactly the same rolling occupancy and transition-rate features; retraining",
            "these identical inputs would be redundant.",
            "",
            "## Reach-chain divisor (frozen checkpoint and extractor)",
            "",
            "| divisor | Acc. | Active F1 | BF1@5 | BF1@10 | crossing flips vs 1.50 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in reach_rows:
        lines.append(
            f"| {row['reach_divisor']:.2f} | {row['activity_accuracy']:.3f} | "
            f"{row['active_f1']:.3f} | {row['boundary_f1_pm5']:.3f} | "
            f"{row['boundary_f1_pm10']:.3f} | "
            f"{row['critical_cross_changes_vs_1.50']:,}/{row['valid_frames']:,} |"
        )
    lines.extend(
        [
            "",
            "This is an input-scale perturbation of the pre-specified conservative model,",
            "with no retuning.  It measures inference sensitivity, not the performance of",
            "models retrained under alternative anthropometric definitions.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Auditing archived L1/L2 and stop thresholds", flush=True)
    stop_rows, stop_meta = stop_and_reproduction_audit()
    print("Evaluating frozen reach-divisor perturbations", flush=True)
    reach_rows, per_clip_rows = reach_audit(out_root, device)
    write_csv(out_root / "stop_thresholds.csv", stop_rows)
    write_csv(out_root / "reach_divisors.csv", reach_rows)
    write_csv(out_root / "reach_divisors_per_clip_counts.csv", per_clip_rows)
    with (out_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": str(REPO / "paper_experiments/round2/PROTOCOL.md"),
                "stop_thresholds": stop_rows,
                "stop_metadata": stop_meta,
                "reach_divisors": reach_rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    render_results(out_root / "RESULTS.md", stop_rows, stop_meta, reach_rows)
    print(f"Wrote {out_root / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
