#!/usr/bin/env python3
"""Paired bootstrap comparison of truncated boundary-model variants."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List


METRICS = [
    "activity_accuracy",
    "activity_macro_f1",
    "active_f1",
    "boundary_f1_pm5",
    "boundary_f1_pm10",
    "model_active_coverage",
    "state_count_ratio",
    "active_coverage_abs_error",
    "state_count_abs_error",
]


def read_usable(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        if row.get("usable", "").lower() != "true":
            continue
        values = {metric: float(row[metric]) for metric in METRICS[:7]}
        values["active_coverage_abs_error"] = abs(
            float(row["model_active_coverage"]) - float(row["human_active_coverage"])
        )
        values["state_count_abs_error"] = abs(float(row["model_states"]) - float(row["human_states"]))
        output[row["clip_id"]] = values
    return output


def percentile(values: List[float], q: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="paper_experiments/experiments/truncated_retraining")
    parser.add_argument("--reference", default="deployed_checkpoint")
    parser.add_argument("--candidate", default="finetuned_short_loss")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20270614)
    args = parser.parse_args()

    root = Path(args.workspace)
    reference = read_usable(root / "results" / args.reference / "by_clip.csv")
    candidate = read_usable(root / "results" / args.candidate / "by_clip.csv")
    clip_ids = sorted(reference.keys() & candidate.keys())
    if not clip_ids:
        raise RuntimeError("No common usable clips")

    rng = random.Random(args.seed)
    report = {
        "reference": args.reference,
        "candidate": args.candidate,
        "clips": len(clip_ids),
        "bootstrap_samples": args.samples,
        "difference": "candidate minus reference",
        "metrics": {},
    }
    for metric in METRICS:
        observed = mean(candidate[c][metric] - reference[c][metric] for c in clip_ids)
        samples = []
        for _ in range(args.samples):
            draw = [rng.choice(clip_ids) for _ in clip_ids]
            samples.append(mean(candidate[c][metric] - reference[c][metric] for c in draw))
        lower_is_better = metric in {"active_coverage_abs_error", "state_count_abs_error"}
        report["metrics"][metric] = {
            "observed_difference": observed,
            "ci95_low": percentile(samples, 0.025),
            "ci95_high": percentile(samples, 0.975),
            "lower_is_better": lower_is_better,
            "probability_candidate_better": sum(
                value < 0 if lower_is_better else value > 0 for value in samples
            )
            / len(samples),
        }

    output = root / "results" / f"bootstrap_{args.candidate}_vs_{args.reference}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
