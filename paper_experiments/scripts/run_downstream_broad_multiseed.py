#!/usr/bin/env python3
"""Five-seed broad-state classifier comparison on fixed ReLIS segments."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_downstream_broad_sequence_search as broad  # noqa: E402
import run_downstream_state_model_search as core  # noqa: E402


OUT = ROOT / "paper_experiments/round2/downstream_state_model_search/broad_multiseed"
SEEDS = (20260813, 20260814, 20260815, 20260816, 20260817)
CONFIGS = {
    "balanced_reference": {
        "footwork": {
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "max_features": 0.45,
            "class_weight": "balanced_subsample",
        },
        "state": {
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "max_features": 0.45,
            "class_weight": "balanced_subsample",
        },
    },
    "task_specific_candidate": {
        "footwork": {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "max_features": 0.20,
            "class_weight": None,
        },
        "state": {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "max_features": 0.45,
            "class_weight": None,
        },
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    return (
        float(accuracy_score(y, prediction)),
        float(f1_score(y, prediction, average="macro", zero_division=0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--annotations", type=Path, default=core.DEFAULT_ANNOTATIONS)
    parser.add_argument("--raw-context", type=Path, default=core.DEFAULT_RAW_CONTEXT)
    parser.add_argument("--stabilized-context", type=Path, default=core.DEFAULT_STABILIZED_CONTEXT)
    parser.add_argument("--raw-l2", type=Path, default=core.DEFAULT_RAW_L2)
    parser.add_argument("--stabilized-l2", type=Path, default=core.DEFAULT_STABILIZED_L2)
    parser.add_argument("--base-model", type=Path, default=core.DEFAULT_BASE_MODEL)
    args = parser.parse_args()
    started = time.perf_counter()
    samples, views, metadata = broad.load_data(args)
    groups = np.asarray([str(sample["source_group"]) for sample in samples])
    group_names = sorted(set(groups))
    targets = {head: broad.target(samples, head) for head in broad.HEADS}
    clip_ids = sorted({str(sample["clip_id"]) for sample in samples})
    if len(group_names) != 10 or len(clip_ids) != 50:
        raise ValueError("Expected 10 source groups and 50 clips")
    clips_by_group = {
        group: {
            str(sample["clip_id"])
            for sample in samples
            if str(sample["source_group"]) == group
        }
        for group in group_names
    }
    if any(len(clips) != 5 for clips in clips_by_group.values()):
        raise ValueError(f"Expected five clips per source group: {clips_by_group}")
    expected_counts = {
        "footwork": {"advance": 363, "retreat": 213, "hold/reset": 76},
        "state": {"pressure": 340, "attack": 100, "defend": 199, "other": 13},
    }
    for head in broad.HEADS:
        observed = dict(
            sorted(
                Counter(
                    broad.NAMES[head][int(value)] for value in targets[head]
                ).items()
            )
        )
        if observed != expected_counts[head]:
            raise ValueError(f"Unexpected {head} counts: {observed}")
    prediction_sets: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    rows = []
    prediction_rows = []
    for method, by_head in CONFIGS.items():
        for seed_index, seed in enumerate(SEEDS):
            print(f"{method} seed {seed_index + 1}/{len(SEEDS)}", flush=True)
            predictions = {
                head: np.full(len(samples), -1, dtype=np.int64) for head in broad.HEADS
            }
            for outer_index, outer_group in enumerate(group_names):
                train = groups != outer_group
                test = groups == outer_group
                for head_index, head in enumerate(broad.HEADS):
                    probabilities = broad.fit_predict(
                        samples,
                        views[head]["engineered_dual"],
                        targets[head],
                        head,
                        by_head[head],
                        train,
                        test,
                        seed
                        + outer_index * 100
                        + head_index * 10,
                    )
                    predictions[head][test] = np.argmax(probabilities, axis=1)
            joint = np.ones(len(samples), dtype=bool)
            foot_accuracy, foot_macro = metrics(
                targets["footwork"], predictions["footwork"]
            )
            state_accuracy, state_macro = metrics(targets["state"], predictions["state"])
            for head in broad.HEADS:
                joint &= predictions[head] == targets[head]
            row = {
                "method": method,
                "seed": seed,
                "footwork_accuracy": foot_accuracy,
                "footwork_macro_f1": foot_macro,
                "state_accuracy": state_accuracy,
                "state_macro_f1": state_macro,
                "joint_exact_accuracy": float(np.mean(joint)),
            }
            rows.append(row)
            prediction_sets[(method, seed)] = predictions
            for index, sample in enumerate(samples):
                prediction_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "clip_id": sample["clip_id"],
                        "source_group": sample["source_group"],
                        "segment_idx": sample["segment_idx"],
                        "side": sample["side"],
                        "footwork_truth": broad.NAMES["footwork"][targets["footwork"][index]],
                        "footwork_prediction": broad.NAMES["footwork"][predictions["footwork"][index]],
                        "state_truth": broad.NAMES["state"][targets["state"][index]],
                        "state_prediction": broad.NAMES["state"][predictions["state"][index]],
                    }
                )
    aggregate = {}
    metric_names = (
        "footwork_accuracy",
        "footwork_macro_f1",
        "state_accuracy",
        "state_macro_f1",
        "joint_exact_accuracy",
    )
    for method in CONFIGS:
        selected = [row for row in rows if row["method"] == method]
        aggregate[method] = {}
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in selected])
            aggregate[method][metric_name] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    selected_seed = SEEDS[0]
    inference = {}
    for head_index, head in enumerate(broad.HEADS):
        truth = targets[head]
        inference[head] = core.paired_accuracy_inference(
            samples,
            prediction_sets[("balanced_reference", selected_seed)][head] == truth,
            prediction_sets[("task_specific_candidate", selected_seed)][head] == truth,
            20263000 + head_index,
        )
    baseline_joint = np.ones(len(samples), dtype=bool)
    candidate_joint = np.ones(len(samples), dtype=bool)
    for head in broad.HEADS:
        truth = targets[head]
        baseline_joint &= prediction_sets[("balanced_reference", selected_seed)][head] == truth
        candidate_joint &= prediction_sets[("task_specific_candidate", selected_seed)][head] == truth
    inference["joint_exact"] = core.paired_accuracy_inference(
        samples, baseline_joint, candidate_joint, 20263002
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_seed_metrics.csv", rows)
    write_csv(args.out_dir / "predictions.csv", prediction_rows)
    report = {
        "schema_version": "downstream_broad_multiseed_v1",
        "status": "retrospective_exploratory_screened_configuration",
        "protocol": "../BROAD_MULTISEED_PROTOCOL.md",
        "dataset": {"exchanges": 50, "sources": 10, "segments": 326, "samples": 652},
        "configs": CONFIGS,
        "seeds": list(SEEDS),
        "seed_schedule": {
            "formula": "classifier_seed + 100 * outer_fold_index + 10 * head_index",
            "footwork_right_head_offset": 1,
            "matched_between_methods": True,
        },
        "per_seed": rows,
        "aggregate": aggregate,
        "selected_seed_equal_source_inference": inference,
        "metadata": metadata,
        "inputs": {
            "annotations": str(args.annotations),
            "annotation_manifest": broad.manifest_digest(
                list(args.annotations.glob("*.json"))
            ),
            "raw_context": str(args.raw_context),
            "raw_context_sha256": core.sha256(args.raw_context),
            "stabilized_context": str(args.stabilized_context),
            "stabilized_context_sha256": core.sha256(args.stabilized_context),
            "raw_l2": str(args.raw_l2),
            "raw_l2_manifest": broad.manifest_digest(
                [args.raw_l2 / f"{clip_id}_l2.json" for clip_id in clip_ids]
            ),
            "stabilized_l2": str(args.stabilized_l2),
            "stabilized_l2_manifest": broad.manifest_digest(
                [
                    args.stabilized_l2 / f"{clip_id}_l2.json"
                    for clip_id in clip_ids
                ]
            ),
            "base_model": str(args.base_model),
            "base_model_sha256": core.sha256(args.base_model),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "runtime_sec": float(time.perf_counter() - started),
    }
    write_json(args.out_dir / "summary.json", report)
    lines = [
        "# Broad-State Multi-Seed Classifier Study",
        "",
        "| Method | Footwork acc. | Footwork macro F1 | State acc. | State macro F1 | Joint exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in CONFIGS:
        row = aggregate[method]
        lines.append(
            f"| {method.replace('_', ' ')} | "
            f"{row['footwork_accuracy']['mean']:.4f} $\\pm$ {row['footwork_accuracy']['sample_sd']:.4f} | "
            f"{row['footwork_macro_f1']['mean']:.4f} $\\pm$ {row['footwork_macro_f1']['sample_sd']:.4f} | "
            f"{row['state_accuracy']['mean']:.4f} $\\pm$ {row['state_accuracy']['sample_sd']:.4f} | "
            f"{row['state_macro_f1']['mean']:.4f} $\\pm$ {row['state_macro_f1']['sample_sd']:.4f} | "
            f"{row['joint_exact_accuracy']['mean']:.4f} $\\pm$ {row['joint_exact_accuracy']['sample_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The candidate has higher footwork, interaction-state, and joint raw",
            "accuracy in all five matched fits, but its footwork macro F1 is lower.",
            "Neither method correctly predicts the 13-sample `other` class, and all",
            "designated-seed source-level confidence intervals include zero.",
            "",
            "The task-specific configuration was identified by an exploratory screen on",
            "this same dataset; these repeated runs quantify estimator stability, not an",
            "independent selection-adjusted test.",
            "",
        ]
    )
    (args.out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
