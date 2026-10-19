#!/usr/bin/env python3
"""Stress-test the frozen foil boundary model under L2 feature perturbations."""

from __future__ import annotations

import os
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
OPENMMLAB_PY = Path(os.environ.get("RELIS_PYTHON", sys.executable))
MANIFEST_FINAL = REPO / "paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv"
PRIMARY_CONFIG = REPO / "paper_experiments/experiments/truncated_retraining/candidates/optimistic_smooth02_seed18/config.yaml"
TUNED_CKPT = REPO / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_seed18_smooth02/ckpt_step_1000.pt"
CONSENSUS_DIR = REPO / "paper_experiments/annotations/adjudicated/consensus"


CONDITIONS = [
    {"name": "clean_l2", "jitter": 0.0, "hold": 0.0},
    {"name": "jitter_5pct", "jitter": 0.05, "hold": 0.0},
    {"name": "jitter_10pct", "jitter": 0.10, "hold": 0.0},
    {"name": "hold_10pct", "jitter": 0.0, "hold": 0.10},
    {"name": "jitter_10pct_hold_10pct", "jitter": 0.10, "hold": 0.10},
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def perturb_l2(src: Path, dst: Path, jitter: float, hold: float, rng: np.random.Generator) -> None:
    data = read_json(src)
    x = np.asarray(data.get("l2", []), dtype=np.float32)
    mask = np.asarray(data.get("l2_mask", [True] * len(x)), dtype=bool)
    if x.ndim != 2 or x.size == 0:
        write_json(dst, data)
        return

    y = x.copy()
    valid = np.where(mask)[0]
    if valid.size:
        std = np.nanstd(y[valid], axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
        if jitter > 0:
            noise = rng.normal(0.0, jitter, size=y.shape).astype(np.float32) * std[None, :]
            y[valid] = y[valid] + noise[valid]
        if hold > 0:
            drop = valid[rng.random(valid.size) < hold]
            for idx in drop:
                prev = valid[valid < idx]
                nxt = valid[valid > idx]
                if prev.size:
                    y[idx] = y[prev[-1]]
                elif nxt.size:
                    y[idx] = y[nxt[0]]

    data["l2"] = y.astype(np.float32).tolist()
    meta = dict(data.get("meta", {}))
    meta["paper_robustness_perturbation"] = {"jitter_std_fraction": jitter, "frame_hold_probability": hold}
    data["meta"] = meta
    write_json(dst, data)


def build_condition_manifest(rows: list[dict[str, str]], condition: dict[str, float | str], root: Path, seed: int) -> Path:
    name = str(condition["name"])
    l2_dir = root / "l2" / name
    manifest_path = root / "manifests" / f"{name}.csv"
    name_offset = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + name_offset)
    out_rows = []
    for row in rows:
        src = (REPO / row["l2_path"]).resolve()
        dst = l2_dir / src.name
        perturb_l2(src, dst, float(condition["jitter"]), float(condition["hold"]), rng)
        out = dict(row)
        out["l2_path"] = str(dst.relative_to(REPO))
        out_rows.append(out)
    write_csv(manifest_path, out_rows)
    return manifest_path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def run_condition(name: str, manifest: Path, root: Path) -> dict[str, str]:
    pred_dir = root / "predictions" / name
    pred_path = pred_dir / "full_model.json"
    py = OPENMMLAB_PY if OPENMMLAB_PY.exists() else Path(PY)
    run(
        [
            str(py),
            "paper_experiments/scripts/generate_boundary_predictions.py",
            "--manifest",
            str(manifest),
            "--method",
            "full_model",
            "--out-dir",
            str(pred_dir),
            "--config",
            str(PRIMARY_CONFIG),
            "--ckpt",
            str(TUNED_CKPT),
            "--python-bin",
            str(py),
            "--seed",
            "20270618",
        ]
    )
    result_dir = root / "results" / name
    summary_path = result_dir / "summary.csv"
    run(
        [
            PY,
            "paper_experiments/scripts/compare_annotations_to_model.py",
            "--owner-dir",
            str(CONSENSUS_DIR),
            "--manifest",
            str(MANIFEST_FINAL),
            "--predictions",
            str(pred_path),
            "--model-annotations-dir",
            str(result_dir / "model_annotations"),
            "--summary-out",
            str(summary_path),
            "--by-clip-out",
            str(result_dir / "by_clip.csv"),
            "--viewer-metrics-out",
            str(result_dir / "viewer_metrics.csv"),
        ]
    )
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid = next(row for row in rows if row["cohort"] == "valid")
    return valid


def markdown_table(rows: list[dict[str, Any]]) -> str:
    cols = ["condition", "clips", "activity_accuracy", "active_f1", "boundary_f1_pm5", "boundary_f1_pm10"]
    labels = {
        "condition": "Condition",
        "clips": "Clips",
        "activity_accuracy": "Acc.",
        "active_f1": "Active F1",
        "boundary_f1_pm5": "BF1 +/-5",
        "boundary_f1_pm10": "BF1 +/-10",
    }
    out = ["| " + " | ".join(labels[c] for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default="paper_experiments/experiments/pose_feature_robustness")
    parser.add_argument("--seed", type=int, default=20270618)
    args = parser.parse_args()

    root = REPO / args.out_root
    root.mkdir(parents=True, exist_ok=True)
    rows = read_csv(MANIFEST_FINAL)
    summary_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        manifest = build_condition_manifest(rows, condition, root, args.seed)
        valid = run_condition(str(condition["name"]), manifest, root)
        summary_rows.append(
            {
                "condition": condition["name"],
                "jitter_std_fraction": condition["jitter"],
                "frame_hold_probability": condition["hold"],
                "clips": int(float(valid["clips"])),
                "activity_accuracy": f"{float(valid['activity_accuracy']):.3f}",
                "active_f1": f"{float(valid['active_f1']):.3f}",
                "boundary_f1_pm5": f"{float(valid['boundary_f1_pm5']):.3f}",
                "boundary_f1_pm10": f"{float(valid['boundary_f1_pm10']):.3f}",
            }
        )
    write_csv(root / "pose_feature_robustness.csv", summary_rows)
    (root / "pose_feature_robustness.md").write_text(markdown_table(summary_rows), encoding="utf-8")
    print(f"Wrote {root / 'pose_feature_robustness.csv'}")


if __name__ == "__main__":
    main()
