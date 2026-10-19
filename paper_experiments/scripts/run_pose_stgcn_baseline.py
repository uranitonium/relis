#!/usr/bin/env python3
"""Train and evaluate the two-skeleton ST-GCN boundary baseline."""

from __future__ import annotations

import os
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
OPENMMLAB_PY = Path(os.environ.get("RELIS_PYTHON", sys.executable))
OUT_ROOT = REPO / "paper_experiments/experiments/pose_stgcn_baseline"
BOUNDARY_ROOT = REPO / "paper_experiments/experiments/final_boundary_evaluation/boundary_suite"
BASE_CONFIG = REPO / "paper_experiments/experiments/truncated_retraining/candidates/optimistic_smooth02_seed18/config.yaml"
OWNER_DIR = REPO / "paper_experiments/annotations/raw/owner"
COLLAB_DIR = REPO / "paper_experiments/annotations/raw/collaborator_a"
CONSENSUS_DIR = REPO / "paper_experiments/annotations/adjudicated/consensus"


def run(cmd: list[str], cwd: Path = REPO) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_summary(path: Path, cohort: str) -> dict[str, str]:
    for row in read_csv(path):
        if row["cohort"] == cohort:
            return row
    raise KeyError(f"{cohort} not found in {path}")


def build_config(out_root: Path, max_steps: int) -> Path:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["data"]["data_dir"] = str((out_root / "features/train").resolve())
    cfg["data"]["file_glob"] = "*_pose_stgcn.npz"
    cfg["model"]["encoder_type"] = "stgcn"
    cfg["model"]["hidden_dim"] = 128
    cfg["model"]["latent_dim"] = 64
    cfg["model"]["slow_dim"] = 3
    cfg["model"]["boundary"]["use_critical"] = False
    cfg["model"]["boundary"]["critical_index"] = None
    cfg["model"]["boundary"]["k"] = 8
    cfg["model"]["stgcn"] = {
        "layers": 3,
        "kernel_size": 5,
        "dropout": 0.1,
        "num_fencers": 2,
        "num_joints": 12,
        "coord_dim": 2,
    }
    cfg["train"]["max_steps"] = max_steps
    cfg["train"]["warmup_steps"] = min(600, max(10, max_steps // 2))
    cfg["train"]["ckpt_every"] = max_steps
    cfg["train"]["out_dir"] = str((out_root / "runs/pose_stgcn_l3_seed20270618").resolve())
    cfg["train"]["early_stop"]["enabled"] = False
    cfg["seed"] = 20270618
    config_path = out_root / "configs/pose_stgcn_l3.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def tune_config(config_path: Path, ckpt_path: Path, out_root: Path, trials: int) -> Path:
    search_dir = out_root / "postprocess_search"
    best_path = search_dir / "pose_stgcn-b_best.json"
    if not best_path.exists():
        run(
            [
                PY,
                "paper_experiments/scripts/search_boundary_postprocess.py",
                "--manifest",
                str(out_root / "manifests/dev_pose_stgcn.csv"),
                "--owner-dir",
                str(OWNER_DIR),
                "--model",
                f"pose_stgcn={config_path}={ckpt_path}",
                "--score-source",
                "b",
                "--trials",
                str(trials),
                "--seed",
                "20270618",
                "--top-k",
                "100",
                "--out-dir",
                str(search_dir),
            ]
        )
    best = json.loads(best_path.read_text(encoding="utf-8"))["best"]["postprocess"]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["postprocess"] = best
    tuned_path = out_root / "configs/pose_stgcn_l3_tuned.yaml"
    tuned_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return tuned_path


def compare(pred: Path, annotation_dir: Path, manifest: Path, out_dir: Path) -> Path:
    run(
        [
            PY,
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


def append_boundary_rows(boundary_root: Path) -> None:
    result_rows = []
    for reference_name, method_dir in (("consensus", CONSENSUS_DIR), ("collaborator", COLLAB_DIR)):
        summary_path = boundary_root / "results" / reference_name / "pose_stgcn" / "summary.csv"
        for cohort in ("valid", "clean"):
            row = read_summary(summary_path, cohort)
            result_rows.append(
                {
                    "reference": reference_name,
                    "method": "pose_stgcn",
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

    path = boundary_root / "boundary_suite_results.csv"
    rows = read_csv(path)
    rows = [row for row in rows if row.get("method") != "pose_stgcn"]
    rows.extend(result_rows)
    write_csv(path, rows)

    rgb_path = boundary_root / "boundary_suite_results_with_rgb.csv"
    if rgb_path.exists():
        rows = read_csv(rgb_path)
        rows = [row for row in rows if row.get("method") != "pose_stgcn"]
        rows.extend(result_rows)
        write_csv(rgb_path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--boundary-root", default=str(BOUNDARY_ROOT))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=3500)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-feature-extract", action="store_true")
    args = parser.parse_args()

    out_root = (REPO / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)
    boundary_root = (REPO / args.boundary_root).resolve() if not Path(args.boundary_root).is_absolute() else Path(args.boundary_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_feature_extract:
        cmd = [PY, "paper_experiments/scripts/extract_pose_stgcn_features.py", "--out-root", str(out_root)]
        if args.force_features:
            cmd.append("--force")
        run(cmd)

    config_path = build_config(out_root, args.max_steps)
    ckpt_path = out_root / "runs/pose_stgcn_l3_seed20270618" / f"ckpt_step_{args.max_steps}.pt"
    if not args.skip_train or not ckpt_path.exists():
        run([PY, "boundary_segmentation/train.py", "--config", str(config_path), "--device", "auto"])
    if not ckpt_path.exists():
        final_path = out_root / "runs/pose_stgcn_l3_seed20270618/ckpt_final.pt"
        if final_path.exists():
            ckpt_path = final_path
        else:
            raise FileNotFoundError(ckpt_path)

    tuned_config = tune_config(config_path, ckpt_path, out_root, args.trials)
    pred_dir = boundary_root / "predictions/pose_stgcn"
    run(
        [
            PY,
            "paper_experiments/scripts/generate_boundary_predictions.py",
            "--manifest",
            str(out_root / "manifests/final_pose_stgcn.csv"),
            "--method",
            "full_model",
            "--out-dir",
            str(pred_dir),
            "--config",
            str(tuned_config),
            "--ckpt",
            str(ckpt_path),
            "--seed",
            "20270726",
        ]
    )
    pred_path = pred_dir / "full_model.json"
    alias_path = pred_dir / "pose_stgcn.json"
    shutil.copy2(pred_path, alias_path)

    for reference_name, annotation_dir in (("consensus", CONSENSUS_DIR), ("collaborator", COLLAB_DIR)):
        compare(
            pred_path,
            annotation_dir,
            out_root / "manifests/final_pose_stgcn.csv",
            boundary_root / "results" / reference_name / "pose_stgcn",
        )
    append_boundary_rows(boundary_root)
    print(f"[pose_stgcn] complete: {boundary_root / 'results'}", flush=True)


if __name__ == "__main__":
    main()
