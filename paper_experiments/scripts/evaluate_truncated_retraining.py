#!/usr/bin/env python3
"""Run and summarize development evaluation for truncated retraining variants."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(command: List[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="paper_experiments/experiments/truncated_retraining")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--owner-dir", default="paper_experiments/annotations/raw/owner")
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    workspace = (repo / args.workspace).resolve()
    manifest = workspace / "manifests" / "paper_development.csv"
    variants = {
        "deployed_checkpoint": (
            repo / "deployment/checkpoints/legacy_auto/config.yaml",
            repo / "deployment/checkpoints/legacy_auto/ckpt_best_ema.pt",
        ),
        "retrained_deployed_loss": (
            workspace / "configs/deployed_loss.yaml",
            workspace / "runs/deployed_loss/ckpt_best_ema.pt",
        ),
        "retrained_short_sparse": (
            workspace / "configs/short_sparse.yaml",
            workspace / "runs/short_sparse/ckpt_best_ema.pt",
        ),
        "retrained_short_conservative": (
            workspace / "configs/short_conservative.yaml",
            workspace / "runs/short_conservative/ckpt_best_ema.pt",
        ),
        "finetuned_deployed_loss": (
            workspace / "configs/finetune_deployed_loss.yaml",
            workspace / "runs/finetune_deployed_loss/ckpt_best_ema.pt",
        ),
        "finetuned_short_loss": (
            workspace / "configs/finetune_short_loss.yaml",
            workspace / "runs/finetune_short_loss/ckpt_best_ema.pt",
        ),
        "conservative_deployed_pp": (
            workspace / "configs/short_conservative_deployed_pp.yaml",
            workspace / "runs/short_conservative/ckpt_best_ema.pt",
        ),
        "conservative_balanced_pp": (
            workspace / "configs/short_conservative_balanced.yaml",
            workspace / "runs/short_conservative/ckpt_best_ema.pt",
        ),
        "conservative_recall_pp": (
            workspace / "configs/short_conservative_recall.yaml",
            workspace / "runs/short_conservative/ckpt_best_ema.pt",
        ),
        "sparse_deployed_pp": (
            workspace / "configs/short_sparse_deployed_pp.yaml",
            workspace / "runs/short_sparse/ckpt_best_ema.pt",
        ),
        "sparse_balanced_pp": (
            workspace / "configs/short_sparse_balanced.yaml",
            workspace / "runs/short_sparse/ckpt_best_ema.pt",
        ),
    }

    combined: List[Dict[str, object]] = []
    metadata = {}
    for name, (config, checkpoint) in variants.items():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        pred_dir = workspace / "predictions" / name
        result_dir = workspace / "results" / name
        pred_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                args.python_bin,
                "paper_experiments/scripts/generate_boundary_predictions.py",
                "--manifest",
                str(manifest),
                "--method",
                "full_model",
                "--out-dir",
                str(pred_dir),
                "--config",
                str(config),
                "--ckpt",
                str(checkpoint),
                "--python-bin",
                args.python_bin,
            ],
            repo,
        )
        run(
            [
                args.python_bin,
                "paper_experiments/scripts/compare_annotations_to_model.py",
                "--owner-dir",
                args.owner_dir,
                "--predictions",
                str(pred_dir / "full_model.json"),
                "--model-annotations-dir",
                str(result_dir / "model_annotations"),
                "--summary-out",
                str(result_dir / "summary.csv"),
                "--by-clip-out",
                str(result_dir / "by_clip.csv"),
                "--viewer-metrics-out",
                str(result_dir / "viewer_metrics.csv"),
            ],
            repo,
        )
        checkpoint_data = __import__("torch").load(checkpoint, map_location="cpu")
        metadata[name] = {
            "config": str(config),
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(checkpoint_data.get("step", -1)),
            "best_ema_step": int(checkpoint_data.get("best_ema_step", -1)),
        }
        for row in read_csv(result_dir / "summary.csv"):
            combined.append({"variant": name, **row})

    write_csv(workspace / "results" / "development_comparison.csv", combined)
    usable = [row for row in combined if row["cohort"] == "usable"]
    ranked = sorted(
        usable,
        key=lambda row: (
            float(row["activity_macro_f1"]),
            float(row["boundary_f1_pm5"]),
            -abs(float(row["total_state_count_ratio"]) - 1.0),
        ),
        reverse=True,
    )
    selection = {
        "selection_data": "40 owner-labeled paper-development clips; not final test data",
        "ranking_rule": "activity_macro_f1, then boundary_f1_pm5, then state-count-ratio proximity to 1",
        "ranking": [row["variant"] for row in ranked],
        "recommended_variant": ranked[0]["variant"] if ranked else None,
        "checkpoints": metadata,
    }
    (workspace / "results" / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
