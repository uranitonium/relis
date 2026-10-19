#!/usr/bin/env python3
"""Evaluate an explicitly provisional owner-as-consensus benchmark.

This analysis never edits annotation records or official consensus results.  It
combines already-computed, frozen-model per-clip metrics for the official final
consensus cohort and the new owner-only extension, treating the latter as if it
were consensus solely for diagnostic forecasting.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRICS = (
    "activity_accuracy",
    "activity_macro_f1",
    "active_precision",
    "active_recall",
    "active_f1",
    "boundary_f1_pm3",
    "boundary_f1_pm5",
    "boundary_f1_pm10",
    "model_active_coverage",
    "human_active_coverage",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def source_id(clip_id: str) -> str:
    parts = clip_id.split("-")
    if len(parts) != 3:
        raise ValueError(f"unexpected clip ID: {clip_id}")
    return parts[1]


def selected(rows: list[dict[str, str]], field: str) -> list[dict[str, str]]:
    return [row for row in rows if truthy(row[field])]


def metric_mean(rows: list[dict[str, str]], metric: str) -> float:
    return mean(float(row[metric]) for row in rows)


def summarize(
    calibration: str, cohort: str, rows: list[dict[str, str]]
) -> dict[str, object]:
    result: dict[str, object] = {
        "calibration": calibration,
        "cohort": cohort,
        "clips": len(rows),
        "source_ids": len({source_id(row["clip_id"]) for row in rows}),
        "human_states": sum(int(row["human_states"]) for row in rows),
        "model_states": sum(int(row["model_states"]) for row in rows),
    }
    result["state_count_ratio"] = result["model_states"] / result["human_states"]
    for metric in METRICS:
        result[metric] = metric_mean(rows, metric)
    return result


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def clustered_bootstrap_means(
    rows: list[dict[str, str]], metric: str, repeats: int, seed: int
) -> list[float]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[source_id(row["clip_id"])].append(row)
    clusters = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(repeats):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values = [float(row[metric]) for cluster in sampled for row in grouped[cluster]]
        estimates.append(mean(values))
    return estimates


def two_cohort_clustered_differences(
    candidate: list[dict[str, str]],
    reference: list[dict[str, str]],
    metric: str,
    repeats: int,
    seed: int,
) -> list[float]:
    candidate_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    reference_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate:
        candidate_by_source[source_id(row["clip_id"])].append(row)
    for row in reference:
        reference_by_source[source_id(row["clip_id"])].append(row)
    clusters = sorted(set(candidate_by_source) | set(reference_by_source))
    rng = random.Random(seed)
    differences = []
    for _ in range(repeats):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        candidate_values = [
            float(row[metric]) for cluster in sampled for row in candidate_by_source[cluster]
        ]
        reference_values = [
            float(row[metric]) for cluster in sampled for row in reference_by_source[cluster]
        ]
        if candidate_values and reference_values:
            differences.append(mean(candidate_values) - mean(reference_values))
    return differences


def paired_calibration_differences(
    candidate: list[dict[str, str]],
    reference: list[dict[str, str]],
    metric: str,
    repeats: int,
    seed: int,
) -> list[float]:
    candidate_map = {row["clip_id"]: row for row in candidate}
    reference_map = {row["clip_id"]: row for row in reference}
    if set(candidate_map) != set(reference_map):
        raise ValueError("calibration cohorts do not contain identical clip IDs")
    differences_by_source: dict[str, list[float]] = defaultdict(list)
    for clip_id in sorted(candidate_map):
        differences_by_source[source_id(clip_id)].append(
            float(candidate_map[clip_id][metric]) - float(reference_map[clip_id][metric])
        )
    clusters = sorted(differences_by_source)
    rng = random.Random(seed)
    differences = []
    for _ in range(repeats):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values = [value for cluster in sampled for value in differences_by_source[cluster]]
        differences.append(mean(values))
    return differences


def interval_row(
    comparison: str,
    metric: str,
    estimate: float,
    samples: list[float],
    clusters: int,
    repeats: int,
) -> dict[str, object]:
    return {
        "comparison": comparison,
        "metric": metric,
        "estimate": estimate,
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "bootstrap_probability_gt_zero": sum(value > 0 for value in samples) / len(samples),
        "source_clusters": clusters,
        "bootstrap_repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("paper_experiments/round2/annotation_extension")
    official = root / "returns/consensus_evaluation"
    owner = root / "owner_full_quickcheck"
    parser.add_argument("--official-dir", type=Path, default=official)
    parser.add_argument("--owner-dir", type=Path, default=owner)
    parser.add_argument(
        "--output-dir", type=Path, default=root / "provisional_owner_as_consensus"
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    calibrations = ("conservative", "optimistic")
    inputs: dict[str, dict[str, list[dict[str, str]]]] = {}
    for calibration in calibrations:
        inputs[calibration] = {
            "official": read_csv(
                args.official_dir
                / f"internal_{calibration}_final_consensus_110_by_clip.csv"
            ),
            "owner": read_csv(args.owner_dir / f"internal_{calibration}_by_clip.csv"),
        }

    official_ids = {row["clip_id"] for row in inputs["conservative"]["official"]}
    owner_ids = {row["clip_id"] for row in inputs["conservative"]["owner"]}
    overlap = sorted(official_ids & owner_ids)
    if overlap:
        raise ValueError(f"official and owner cohorts overlap: {overlap}")

    summaries: list[dict[str, object]] = []
    combined_valid: dict[str, list[dict[str, str]]] = {}
    for calibration in calibrations:
        official_valid = selected(inputs[calibration]["official"], "usable")
        owner_valid = selected(inputs[calibration]["owner"], "usable")
        combined_valid[calibration] = official_valid + owner_valid
        combined_clean = selected(inputs[calibration]["official"], "clean") + selected(
            inputs[calibration]["owner"], "clean"
        )
        summaries.extend(
            [
                summarize(calibration, "official_consensus_valid", official_valid),
                summarize(calibration, "owner_as_consensus_valid", owner_valid),
                summarize(calibration, "combined_provisional_valid", combined_valid[calibration]),
                summarize(calibration, "combined_provisional_clean", combined_clean),
            ]
        )

    key_metrics = ("activity_accuracy", "active_f1", "boundary_f1_pm5", "boundary_f1_pm10")
    bootstrap_rows: list[dict[str, object]] = []
    for calibration in calibrations:
        rows = combined_valid[calibration]
        cluster_count = len({source_id(row["clip_id"]) for row in rows})
        for offset, metric in enumerate(key_metrics):
            samples = clustered_bootstrap_means(
                rows, metric, args.bootstrap_repeats, args.seed + offset
            )
            bootstrap_rows.append(
                interval_row(
                    f"{calibration}_combined_provisional",
                    metric,
                    metric_mean(rows, metric),
                    samples,
                    cluster_count,
                    args.bootstrap_repeats,
                )
            )

    primary_official = selected(inputs["conservative"]["official"], "usable")
    primary_owner = selected(inputs["conservative"]["owner"], "usable")
    union_clusters = len(
        {source_id(row["clip_id"]) for row in primary_official + primary_owner}
    )
    for offset, metric in enumerate(key_metrics):
        samples = two_cohort_clustered_differences(
            primary_owner,
            primary_official,
            metric,
            args.bootstrap_repeats,
            args.seed + 100 + offset,
        )
        bootstrap_rows.append(
            interval_row(
                "conservative_owner_minus_official",
                metric,
                metric_mean(primary_owner, metric) - metric_mean(primary_official, metric),
                samples,
                union_clusters,
                args.bootstrap_repeats,
            )
        )

    for offset, metric in enumerate(key_metrics):
        samples = paired_calibration_differences(
            combined_valid["optimistic"],
            combined_valid["conservative"],
            metric,
            args.bootstrap_repeats,
            args.seed + 200 + offset,
        )
        bootstrap_rows.append(
            interval_row(
                "combined_optimistic_minus_conservative",
                metric,
                metric_mean(combined_valid["optimistic"], metric)
                - metric_mean(combined_valid["conservative"], metric),
                samples,
                union_clusters,
                args.bootstrap_repeats,
            )
        )

    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "bootstrap_intervals.csv", bootstrap_rows)
    audit = {
        "status": "diagnostic_only_not_consensus",
        "description": "Owner labels are provisionally treated as consensus only for forecasting.",
        "official_clip_records": len(official_ids),
        "owner_clip_records": len(owner_ids),
        "exact_clip_overlap": overlap,
        "official_usable": len(primary_official),
        "owner_usable": len(primary_owner),
        "combined_usable": len(primary_official) + len(primary_owner),
        "combined_source_ids": union_clusters,
        "source_id_definition": "middle field of PREFIX-SOURCE-INDEX clip ID",
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
        "official_inputs": str(args.official_dir),
        "owner_inputs": str(args.owner_dir),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    print(f"Wrote provisional analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
