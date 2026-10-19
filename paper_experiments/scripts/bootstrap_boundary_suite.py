#!/usr/bin/env python3
"""Bootstrap paired differences for the foil boundary suite."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from common import read_csv, write_csv, write_json


DEFAULT_ROOT = Path("paper_experiments/experiments/final_boundary_evaluation/boundary_suite")
METRICS = [
    "activity_accuracy",
    "activity_macro_f1",
    "active_f1",
    "boundary_f1_pm3",
    "boundary_f1_pm5",
    "boundary_f1_pm10",
]
DEFAULT_PAIRS = [
    ("optimistic", "conservative"),
    ("optimistic", "random_matched_conservative"),
    ("optimistic", "motion"),
    ("optimistic", "selfsim"),
    ("optimistic", "pose_stgcn"),
    ("optimistic", "rgb_resnet"),
    ("optimistic", "video_r3d"),
    ("optimistic", "video_videomae"),
    ("optimistic", "changepoint"),
    ("optimistic", "engineered_l2_instability"),
    ("optimistic", "raw_phi_l2_conservative"),
    ("raw_phi_l2_conservative", "engineered_l2_instability"),
    ("conservative", "raw_phi_l2_conservative"),
    ("optimistic", "no_critical_optimistic"),
    ("conservative", "no_critical_conservative"),
]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def load_metric_rows(root: Path, reference: str, method: str, cohort: str) -> dict[str, dict[str, float]]:
    rows = read_csv(root / "results" / reference / method / "by_clip.csv")
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if not truthy(row.get(cohort, False)):
            continue
        out[row["clip_id"]] = {metric: float(row[metric]) for metric in METRICS}
    return out


def paired_bootstrap(
    candidate_rows: dict[str, dict[str, float]],
    baseline_rows: dict[str, dict[str, float]],
    metric: str,
    iterations: int,
    rng: random.Random,
    cluster_by_bout: bool = False,
) -> dict[str, float | int]:
    clip_ids = sorted(set(candidate_rows) & set(baseline_rows))
    if not clip_ids:
        return {"n": 0, "mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_gt_zero": 0.0}
    diffs_by_clip = {
        clip_id: candidate_rows[clip_id][metric] - baseline_rows[clip_id][metric]
        for clip_id in clip_ids
    }
    diffs = list(diffs_by_clip.values())
    mean_diff = sum(diffs) / len(diffs)
    samples = []
    clusters = 0
    if cluster_by_bout:
        # Clip IDs are PREFIX-BOUT-EXCHANGE; the middle field identifies the
        # source bout/video, as documented by the dataset owner.
        by_bout: dict[str, list[float]] = {}
        for clip_id, diff in diffs_by_clip.items():
            parts = clip_id.split("-")
            if len(parts) < 3:
                raise ValueError(f"clip_id lacks PREFIX-BOUT-EXCHANGE fields: {clip_id}")
            by_bout.setdefault(parts[1], []).append(diff)
        bout_ids = sorted(by_bout)
        clusters = len(bout_ids)
        for _ in range(iterations):
            sampled = [bout_ids[rng.randrange(clusters)] for _ in bout_ids]
            values = [diff for bout_id in sampled for diff in by_bout[bout_id]]
            samples.append(sum(values) / len(values))
    else:
        for _ in range(iterations):
            sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
            samples.append(sum(sample) / len(sample))
    result: dict[str, float | int] = {
        "n": len(clip_ids),
        "mean_diff": mean_diff,
        "ci_low": quantile(samples, 0.025),
        "ci_high": quantile(samples, 0.975),
        "p_gt_zero": sum(value > 0 for value in samples) / len(samples),
    }
    if cluster_by_bout:
        result["clusters"] = clusters
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20270726)
    parser.add_argument(
        "--cluster-by-bout",
        action="store_true",
        help="Resample source-bout groups using the middle PREFIX-BOUT-EXCHANGE field.",
    )
    parser.add_argument("--reference", action="append", default=["consensus", "collaborator"])
    parser.add_argument("--cohort", action="append", default=["valid", "clean"])
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Candidate=baseline. May be repeated. Defaults to tuned-vs-baseline comparisons.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    pairs = []
    for spec in args.pair:
        if "=" not in spec:
            raise SystemExit("--pair must be Candidate=baseline")
        candidate, baseline = spec.split("=", 1)
        pairs.append((candidate, baseline))
    if not pairs:
        pairs = DEFAULT_PAIRS

    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    nested: dict[str, object] = {
        "root": str(root),
        "iterations": args.iterations,
        "seed": args.seed,
        "cluster_by_bout": args.cluster_by_bout,
        "pairs": [],
    }
    for reference in args.reference:
        for cohort in args.cohort:
            for candidate, baseline in pairs:
                candidate_rows = load_metric_rows(root, reference, candidate, cohort)
                baseline_rows = load_metric_rows(root, reference, baseline, cohort)
                for metric in METRICS:
                    stats = paired_bootstrap(
                        candidate_rows,
                        baseline_rows,
                        metric,
                        args.iterations,
                        rng,
                        cluster_by_bout=args.cluster_by_bout,
                    )
                    row = {
                        "reference": reference,
                        "cohort": cohort,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        **stats,
                    }
                    rows.append(row)
                    nested["pairs"].append(row)

    out_csv = Path(args.out_csv) if args.out_csv else root / "bootstrap_pairwise_differences.csv"
    out_json = Path(args.out_json) if args.out_json else root / "bootstrap_pairwise_differences.json"
    write_csv(out_csv, rows)
    write_json(out_json, nested)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
