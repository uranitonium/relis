#!/usr/bin/env python3
"""Prepare and run a small bout-held-out foil generalization sanity check.

The main paper split is clip-disjoint. This script creates a separate
experiment in which all clips from selected source-bout numbers are removed
import os
from the self-supervised training pool and evaluated only on consensus
annotations from those unseen bout numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = ROOT / "paper_experiments" / "experiments" / "bout_heldout_sanity"
DEFAULT_OPENMMLAB_PY = Path(os.environ.get("RELIS_PYTHON", sys.executable))
DEFAULT_SOURCE_WORKSPACE = ROOT / "paper_experiments" / "experiments" / "truncated_retraining"
DEFAULT_CONFIG = (
    DEFAULT_SOURCE_WORKSPACE
    / "candidates"
    / "optimistic_smooth02_seed18"
    / "config.yaml"
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["clip_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clip_bout_id(clip_id: str) -> str:
    parts = clip_id.split("-")
    if len(parts) < 3:
        raise ValueError(f"clip_id does not contain a middle bout field: {clip_id}")
    return parts[1]


def is_usable(annotation: dict) -> bool:
    flags = annotation.get("quality_flags", {})
    return (
        bool(annotation.get("annotation_complete"))
        and bool(flags.get("valid", True))
        and not flags.get("not_exchange")
    )


def is_clean(annotation: dict) -> bool:
    flags = annotation.get("quality_flags", {})
    bad = [name for name, enabled in flags.items() if enabled and name != "valid"]
    return is_usable(annotation) and not bad


def clean_symlink_dir(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in destination.glob("*_l2.json"):
        if path.is_symlink():
            path.unlink()
        else:
            raise RuntimeError(f"Refusing to remove non-symlink training file: {path}")


def link_l2(rows: List[Dict[str, str]], destination: Path) -> None:
    clean_symlink_dir(destination)
    for row in rows:
        source = (ROOT / row["l2_path"]).resolve() if not Path(row["l2_path"]).is_absolute() else Path(row["l2_path"]).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        target = destination / source.name
        target.symlink_to(source)


def select_test_bouts(
    final_rows: List[Dict[str, str]],
    dev_rows: List[Dict[str, str]],
    annotations: Dict[str, dict],
    target_valid: int,
    min_clips_per_bout: int,
) -> List[str]:
    dev_bouts = {clip_bout_id(row["clip_id"]) for row in dev_rows}
    by_bout: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in final_rows:
        clip_id = row["clip_id"]
        if clip_id not in annotations:
            continue
        if clip_bout_id(clip_id) in dev_bouts:
            continue
        if not is_usable(annotations[clip_id]):
            continue
        by_bout[clip_bout_id(clip_id)].append(row)

    selected: List[str] = []
    for bout_id, rows in sorted(
        by_bout.items(),
        key=lambda item: (-len(item[1]), int(item[0]) if item[0].isdigit() else 10**9),
    ):
        if len(rows) < min_clips_per_bout:
            continue
        selected.append(bout_id)
        if sum(len(by_bout[b]) for b in selected) >= target_valid:
            break
    if not selected:
        raise RuntimeError("No held-out bout candidates were selected")
    return selected


def prepare(args: argparse.Namespace) -> dict:
    workspace = Path(args.workspace)
    source_workspace = Path(args.source_workspace)
    train_rows = read_csv(source_workspace / "manifests" / "train.csv")
    dev_rows = read_csv(source_workspace / "manifests" / "paper_development.csv")
    final_rows = read_csv(source_workspace / "manifests" / "paper_final_candidates.csv")
    annotations = {
        path.stem: read_json(path)
        for path in Path(args.consensus_dir).glob("*.json")
    }

    selected_bouts = (
        [str(item) for item in args.test_bouts.split(",") if item.strip()]
        if args.test_bouts
        else select_test_bouts(
            final_rows,
            dev_rows,
            annotations,
            target_valid=args.target_valid,
            min_clips_per_bout=args.min_clips_per_bout,
        )
    )
    selected_set = set(selected_bouts)
    dev_bouts = {clip_bout_id(row["clip_id"]) for row in dev_rows}
    overlap_with_dev = sorted(selected_set & dev_bouts)
    if overlap_with_dev:
        raise RuntimeError(f"Selected test bouts overlap development bouts: {overlap_with_dev}")

    train_bout_heldout = [
        row for row in train_rows if clip_bout_id(row["clip_id"]) not in selected_set
    ]
    excluded_train = [
        row for row in train_rows if clip_bout_id(row["clip_id"]) in selected_set
    ]
    test_rows = [
        row
        for row in final_rows
        if clip_bout_id(row["clip_id"]) in selected_set
        and row["clip_id"] in annotations
        and is_usable(annotations[row["clip_id"]])
    ]
    if not test_rows:
        raise RuntimeError("No consensus valid rows matched the selected held-out bouts")

    manifests = workspace / "manifests"
    write_csv(manifests / "train.csv", train_bout_heldout)
    write_csv(manifests / "excluded_train_same_bouts.csv", excluded_train)
    write_csv(manifests / "test_consensus.csv", test_rows)
    link_l2(train_bout_heldout, workspace / "data" / "train")

    with Path(args.base_config).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg.pop("_config_path", None)
    cfg["data"]["data_dir"] = str((workspace / "data" / "train").resolve())
    cfg["train"]["out_dir"] = str((workspace / "runs" / "primary").resolve())
    cfg["train"]["max_steps"] = int(args.max_steps)
    cfg["seed"] = int(args.seed)
    config_path = workspace / "configs" / "primary.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    test_outcomes = Counter(row["outcome_label"] for row in test_rows)
    train_outcomes = Counter(row["outcome_label"] for row in train_bout_heldout)
    report = {
        "workspace": str(workspace),
        "source_workspace": str(source_workspace),
        "base_config": str(Path(args.base_config)),
        "config": str(config_path),
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "selected_test_bouts": selected_bouts,
        "selection_rule": {
            "candidate_pool": "valid final-consensus clips whose middle clip-id field is not in development bouts",
            "target_valid": int(args.target_valid),
            "min_clips_per_bout": int(args.min_clips_per_bout),
        },
        "counts": {
            "train_clips": len(train_bout_heldout),
            "train_bouts": len({clip_bout_id(row["clip_id"]) for row in train_bout_heldout}),
            "excluded_train_same_bout_clips": len(excluded_train),
            "test_clips_valid": len(test_rows),
            "test_clips_clean": sum(is_clean(annotations[row["clip_id"]]) for row in test_rows),
            "test_bouts": len(selected_bouts),
            "development_bouts": len(dev_bouts),
        },
        "outcomes": {
            "train": dict(sorted(train_outcomes.items())),
            "test": dict(sorted(test_outcomes.items())),
        },
        "test_clip_ids": [row["clip_id"] for row in test_rows],
        "notes": [
            "The main paper split remains clip-disjoint.",
            "This sanity check retrains after removing all training clips from the selected source-bout numbers.",
            "The selected test bout numbers are also absent from the original development split used for postprocess calibration.",
        ],
    }
    write_json(workspace / "split_report.json", report)
    print(json.dumps(report, indent=2))
    return report


def run_train(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace)
    config = workspace / "configs" / "primary.yaml"
    cmd = [
        str(Path(args.openmmlab_python)),
        "boundary_segmentation/train.py",
        "--config",
        str(config),
        "--device",
        "auto",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_predictions(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace)
    config = workspace / "configs" / "primary.yaml"
    ckpt = workspace / "runs" / "primary" / f"ckpt_step_{args.max_steps}.pt"
    if not ckpt.exists():
        ckpt = workspace / "runs" / "primary" / "ckpt_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained checkpoint found in {workspace / 'runs' / 'primary'}")
    cmd = [
        str(Path(args.openmmlab_python)),
        "paper_experiments/scripts/generate_boundary_predictions.py",
        "--manifest",
        str(workspace / "manifests" / "test_consensus.csv"),
        "--out-dir",
        str(workspace / "predictions"),
        "--config",
        str(config),
        "--ckpt",
        str(ckpt),
        "--python-bin",
        str(Path(args.openmmlab_python)),
        "--method",
        "engineered_l2_instability",
        "--method",
        "raw_phi_l2",
        "--method",
        "full_model",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_eval(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace)
    eval_dir = workspace / "evaluation"
    rows = []
    for method in ["engineered_l2_instability", "raw_phi_l2", "full_model"]:
        pred = workspace / "predictions" / f"{method}.json"
        summary = eval_dir / f"{method}_summary.csv"
        by_clip = eval_dir / f"{method}_by_clip.csv"
        viewer = eval_dir / f"{method}_viewer_metrics.csv"
        model_annotations = eval_dir / f"{method}_annotations"
        cmd = [
            sys.executable,
            "paper_experiments/scripts/compare_annotations_to_model.py",
            "--owner-dir",
            str(Path(args.consensus_dir)),
            "--manifest",
            str(workspace / "manifests" / "test_consensus.csv"),
            "--predictions",
            str(pred),
            "--summary-out",
            str(summary),
            "--by-clip-out",
            str(by_clip),
            "--viewer-metrics-out",
            str(viewer),
            "--model-annotations-dir",
            str(model_annotations),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        for row in read_csv(summary):
            row = dict(row)
            row["method"] = method
            rows.append(row)
    metric_order = {
        "engineered_l2_instability": "Engineered L2 instability",
        "raw_phi_l2": "Learned instability magnitude",
        "full_model": "Full learned head",
    }
    compact = []
    for row in rows:
        if row["cohort"] != "valid":
            continue
        compact.append(
            {
                "method": metric_order.get(row["method"], row["method"]),
                "clips": row["clips"],
                "activity_accuracy": row["activity_accuracy"],
                "active_f1": row["active_f1"],
                "boundary_f1_pm5": row["boundary_f1_pm5"],
                "boundary_f1_pm10": row["boundary_f1_pm10"],
                "model_active_coverage": row["model_active_coverage"],
                "human_active_coverage": row["human_active_coverage"],
            }
        )
    write_csv(workspace / "summary_valid.csv", compact)
    paper_ready = ROOT / "paper_experiments" / "paper_ready" / "foil" / "bout_heldout_sanity.csv"
    write_csv(paper_ready, compact)
    print(f"Wrote {workspace / 'summary_valid.csv'}")
    print(f"Wrote {paper_ready}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--source-workspace", default=str(DEFAULT_SOURCE_WORKSPACE))
    parser.add_argument("--consensus-dir", default=str(ROOT / "paper_experiments" / "annotations" / "adjudicated" / "consensus"))
    parser.add_argument("--base-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--openmmlab-python", default=str(DEFAULT_OPENMMLAB_PY))
    parser.add_argument("--seed", type=int, default=20270618)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--target-valid", type=int, default=36)
    parser.add_argument("--min-clips-per-bout", type=int, default=2)
    parser.add_argument("--test-bouts", default="", help="Comma-separated middle clip-id fields; overrides automatic selection")
    parser.add_argument(
        "--stage",
        action="append",
        choices=["prepare", "train", "predict", "eval", "all"],
        default=[],
    )
    args = parser.parse_args()

    stages = args.stage or ["all"]
    if "all" in stages:
        stages = ["prepare", "train", "predict", "eval"]
    if "prepare" in stages:
        prepare(args)
    if "train" in stages:
        run_train(args)
    if "predict" in stages:
        run_predictions(args)
    if "eval" in stages:
        run_eval(args)


if __name__ == "__main__":
    main()
