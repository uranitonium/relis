#!/usr/bin/env python3
"""Tune and evaluate the kernelized L2 change-point baseline."""

from __future__ import annotations

import os
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from common import read_csv, write_csv, write_json


REPO = Path(__file__).resolve().parents[2]
PY_OPENMMLAB = Path(os.environ.get("RELIS_PYTHON", sys.executable))
MANIFEST_DEV = REPO / "paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv"
MANIFEST_FINAL = REPO / "paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv"
OWNER_DIR = REPO / "paper_experiments/annotations/raw/owner"
CONSENSUS_DIR = REPO / "paper_experiments/annotations/adjudicated/consensus"
BOUNDARY_ROOT = REPO / "paper_experiments/experiments/final_boundary_evaluation/boundary_suite"


def run(cmd: List[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def read_summary(path: Path, cohort: str = "valid") -> dict[str, str]:
    for row in read_csv(path):
        if row["cohort"] == cohort:
            return row
    raise KeyError(f"{cohort} not found in {path}")


def generate(manifest: Path, out_dir: Path, zone_count: int, zone_len: int, python_bin: Path) -> Path:
    run(
        [
            str(python_bin),
            "paper_experiments/scripts/generate_boundary_predictions.py",
            "--manifest",
            str(manifest),
            "--method",
            "kernel_changepoint",
            "--out-dir",
            str(out_dir),
            "--zone-count",
            str(zone_count),
            "--zone-len",
            str(zone_len),
            "--seed",
            "20270726",
        ]
    )
    return out_dir / "kernel_changepoint.json"


def compare(pred: Path, annotation_dir: Path, manifest: Path, out_dir: Path) -> Path:
    run(
        [
            sys.executable,
            "paper_experiments/scripts/compare_annotations_to_model.py",
            "--owner-dir",
            str(annotation_dir),
            "--manifest",
            str(manifest),
            "--predictions",
            str(pred),
            "--model-annotations-dir",
            str(out_dir / "model_annotations"),
            "--summary-out",
            str(out_dir / "summary.csv"),
            "--by-clip-out",
            str(out_dir / "by_clip.csv"),
            "--viewer-metrics-out",
            str(out_dir / "viewer_metrics.csv"),
        ]
    )
    return out_dir / "summary.csv"


def suite_rows(reference: str, summary: Path) -> List[Dict[str, str]]:
    rows = []
    for cohort in ("valid", "clean"):
        row = read_summary(summary, cohort)
        rows.append(
            {
                "reference": reference,
                "method": "kernel_changepoint",
                "cohort": cohort,
                "clips": row["clips"],
                "activity_accuracy": row["activity_accuracy"],
                "activity_macro_f1": row["activity_macro_f1"],
                "active_f1": row["active_f1"],
                "boundary_f1_pm3": row["boundary_f1_pm3"],
                "boundary_f1_pm5": row["boundary_f1_pm5"],
                "boundary_f1_pm10": row["boundary_f1_pm10"],
                "human_states": row["human_states"],
                "model_states": row["model_states"],
                "human_active_coverage": row["human_active_coverage"],
                "model_active_coverage": row["model_active_coverage"],
            }
        )
    return rows


def merge_suite_rows(root: Path, new_rows: List[Dict[str, str]]) -> None:
    suite = root / "boundary_suite_results.csv"
    rows = read_csv(suite) if suite.exists() else []
    rows = [row for row in rows if row.get("method") != "kernel_changepoint"]
    rows.extend(new_rows)
    write_csv(suite, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(BOUNDARY_ROOT))
    parser.add_argument("--python-bin", default=str(PY_OPENMMLAB if PY_OPENMMLAB.exists() else sys.executable))
    args = parser.parse_args()

    root = Path(args.out_root)
    python_bin = Path(args.python_bin)
    tuning_rows: List[Dict[str, object]] = []
    best: Dict[str, object] | None = None
    for count in range(1, 7):
        for zone_len in range(2, 15, 2):
            pred = generate(
                MANIFEST_DEV,
                root / "dev_predictions" / f"kernel_changepoint_c{count}_l{zone_len}",
                count,
                zone_len,
                python_bin,
            )
            summary = compare(
                pred,
                OWNER_DIR,
                MANIFEST_DEV,
                root / "dev_results" / f"kernel_changepoint_c{count}_l{zone_len}",
            )
            valid = read_summary(summary, "valid")
            objective = float(valid["boundary_f1_pm5"]) + 0.5 * float(valid["active_f1"])
            row = {
                "method": "kernel_changepoint",
                "zone_count": count,
                "zone_len": zone_len,
                "objective": objective,
                "boundary_f1_pm5": float(valid["boundary_f1_pm5"]),
                "boundary_f1_pm10": float(valid["boundary_f1_pm10"]),
                "active_f1": float(valid["active_f1"]),
                "activity_accuracy": float(valid["activity_accuracy"]),
                "clips": int(float(valid["clips"])),
            }
            tuning_rows.append(row)
            if best is None or objective > float(best["objective"]):
                best = row

    assert best is not None
    write_csv(root / "baseline_tuning" / "kernel_changepoint.csv", tuning_rows)
    write_json(root / "selected_kernel_changepoint.json", best)

    pred = generate(
        MANIFEST_FINAL,
        root / "predictions" / "kernel_changepoint",
        int(best["zone_count"]),
        int(best["zone_len"]),
        python_bin,
    )
    consensus_summary = compare(pred, CONSENSUS_DIR, MANIFEST_FINAL, root / "results" / "consensus" / "kernel_changepoint")
    collaborator_summary = compare(pred, COLLAB_DIR, MANIFEST_FINAL, root / "results" / "collaborator" / "kernel_changepoint")
    rows = suite_rows("consensus", consensus_summary) + suite_rows("collaborator", collaborator_summary)
    write_csv(root / "kernel_changepoint_results.csv", rows)
    merge_suite_rows(root, rows)
    print(f"Best kernel CPD setting: count={best['zone_count']} zone_len={best['zone_len']}")
    print(f"Wrote {root / 'kernel_changepoint_results.csv'}")


if __name__ == "__main__":
    main()
