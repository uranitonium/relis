#!/usr/bin/env python3
"""Run frozen foil boundary baselines, ablations, and result exports."""

from __future__ import annotations

import os
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
OPENMMLAB_PY = Path(os.environ.get("RELIS_PYTHON", sys.executable))
MANIFEST_DEV = REPO / "paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv"
MANIFEST_FINAL = REPO / "paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv"
CONSERVATIVE_CONFIG = REPO / "paper_experiments/experiments/truncated_retraining/candidates/conservative_smooth02_seed18/config.yaml"
OPTIMISTIC_CONFIG = REPO / "paper_experiments/experiments/truncated_retraining/candidates/optimistic_smooth02_seed18/config.yaml"
TUNED_CKPT = REPO / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_seed18_smooth02/ckpt_step_1000.pt"
OWNER_DIR = REPO / "paper_experiments/annotations/raw/owner"
COLLAB_DIR = REPO / "paper_experiments/annotations/raw/collaborator_a"
CONSENSUS_DIR = REPO / "paper_experiments/annotations/adjudicated/consensus"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def read_summary(path: Path, cohort: str = "valid") -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["cohort"] == cohort:
            return row
    raise KeyError(f"{cohort} not found in {path}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_result_rows(root: Path) -> list[dict[str, object]]:
    """Collect all method summaries already present under results/.

    Some trained baselines, such as RGB/R3D/VideoMAE/ST-GCN, are produced by
    separate scripts. Scanning the result directories preserves those rows when
    the boundary-suite runner is rerun for the lighter baselines.
    """
    rows: list[dict[str, object]] = []
    for reference_dir in sorted((root / "results").glob("*")):
        if not reference_dir.is_dir():
            continue
        reference = reference_dir.name
        for method_dir in sorted(reference_dir.glob("*")):
            summary_path = method_dir / "summary.csv"
            if not summary_path.exists():
                continue
            with summary_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows.append(
                        {
                            "reference": reference,
                            "method": method_dir.name,
                            "cohort": row["cohort"],
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


def generate(method: str, manifest: Path, out_dir: Path, zone_count: int = 3, zone_len: int = 4, reference: Path | None = None, config: Path | None = None) -> Path:
    python_bin = OPENMMLAB_PY if method in {"full_model", "no_critical", "raw_phi_l2", "random_matched", "kernel_changepoint"} and OPENMMLAB_PY.exists() else Path(PY)
    cmd = [
        str(python_bin),
        "paper_experiments/scripts/generate_boundary_predictions.py",
        "--manifest",
        str(manifest),
        "--method",
        method,
        "--out-dir",
        str(out_dir),
        "--zone-count",
        str(zone_count),
        "--zone-len",
        str(zone_len),
        "--seed",
        "20270726",
    ]
    if reference is not None:
        cmd.extend(["--reference-pred", str(reference)])
    if config is not None:
        cmd.extend(["--config", str(config), "--ckpt", str(TUNED_CKPT), "--python-bin", str(python_bin)])
    run(cmd)
    return out_dir / f"{method}.json"


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


def tune_simple_baseline(method: str, root: Path) -> dict[str, object]:
    rows = []
    best: dict[str, object] | None = None
    for count in range(1, 7):
        for zone_len in range(2, 15, 2):
            out_dir = root / "dev_predictions" / f"{method}_c{count}_l{zone_len}"
            pred = generate(method, MANIFEST_DEV, out_dir, count, zone_len)
            summary = compare(pred, OWNER_DIR, MANIFEST_DEV, root / "dev_results" / f"{method}_c{count}_l{zone_len}")
            valid = read_summary(summary, "valid")
            objective = float(valid["boundary_f1_pm5"]) + 0.5 * float(valid["active_f1"])
            row = {
                "method": method,
                "zone_count": count,
                "zone_len": zone_len,
                "objective": objective,
                "boundary_f1_pm5": float(valid["boundary_f1_pm5"]),
                "boundary_f1_pm10": float(valid["boundary_f1_pm10"]),
                "active_f1": float(valid["active_f1"]),
                "activity_accuracy": float(valid["activity_accuracy"]),
                "clips": int(float(valid["clips"])),
            }
            rows.append(row)
            if best is None or objective > float(best["objective"]):
                best = row
    write_csv(root / "baseline_tuning" / f"{method}.csv", rows)
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default="paper_experiments/experiments/final_boundary_evaluation/boundary_suite")
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()
    root = REPO / args.out_root
    root.mkdir(parents=True, exist_ok=True)

    selected_path = root / "selected_baselines.json"
    if args.skip_tuning and selected_path.exists():
        selected = json.loads(selected_path.read_text())
    else:
        selected = {method: tune_simple_baseline(method, root) for method in ("uniform", "motion", "changepoint", "kernel_changepoint", "selfsim")}
        selected_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")

    prediction_specs: dict[str, Path] = {}
    for method, cfg in selected.items():
        prediction_specs[method] = generate(
            method,
            MANIFEST_FINAL,
            root / "predictions" / method,
            int(cfg["zone_count"]),
            int(cfg["zone_len"]),
        )

    conservative_pred = REPO / "paper_experiments/experiments/final_boundary_evaluation/predictions/conservative/full_model.json"
    optimistic_pred = REPO / "paper_experiments/experiments/final_boundary_evaluation/predictions/optimistic/full_model.json"
    prediction_specs["random_matched_conservative"] = generate(
        "random_matched",
        MANIFEST_FINAL,
        root / "predictions" / "random_matched_conservative",
        reference=conservative_pred,
        config=CONSERVATIVE_CONFIG,
    )
    prediction_specs["engineered_l2_instability"] = generate(
        "engineered_l2_instability",
        MANIFEST_FINAL,
        root / "predictions" / "engineered_l2_instability",
        config=CONSERVATIVE_CONFIG,
    )
    prediction_specs["raw_phi_l2_conservative"] = generate(
        "raw_phi_l2",
        MANIFEST_FINAL,
        root / "predictions" / "raw_phi_l2_conservative",
        config=CONSERVATIVE_CONFIG,
    )
    prediction_specs["no_critical_conservative"] = generate(
        "no_critical",
        MANIFEST_FINAL,
        root / "predictions" / "no_critical_conservative",
        config=CONSERVATIVE_CONFIG,
    )
    prediction_specs["no_critical_optimistic"] = generate(
        "no_critical",
        MANIFEST_FINAL,
        root / "predictions" / "no_critical_optimistic",
        config=OPTIMISTIC_CONFIG,
    )
    prediction_specs["conservative"] = conservative_pred
    prediction_specs["optimistic"] = optimistic_pred

    for reference_name, annotation_dir in (("consensus", CONSENSUS_DIR), ("collaborator", COLLAB_DIR)):
        for method_name, pred in prediction_specs.items():
            compare(pred, annotation_dir, MANIFEST_FINAL, root / "results" / reference_name / method_name)
    result_rows = collect_result_rows(root)
    write_csv(root / "boundary_suite_results.csv", result_rows)
    print(f"Wrote {root / 'boundary_suite_results.csv'}")


if __name__ == "__main__":
    main()
