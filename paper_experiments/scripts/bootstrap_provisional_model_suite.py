#!/usr/bin/env python3
"""Paired source-cluster bootstrap for provisional model comparisons."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[2]
EXTENSION = REPO / "paper_experiments/round2/annotation_extension"
ROOT = EXTENSION / "provisional_owner_as_consensus/other_models"
METRICS = (
    "activity_accuracy",
    "active_f1",
    "boundary_f1_pm5",
    "boundary_f1_pm10",
)
REPEATS = 10_000
SEED = 20260810


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def valid_map(official: Path, owner: Path) -> dict[str, dict[str, str]]:
    rows = [
        row
        for path in (official, owner)
        for row in read_csv(path)
        if truthy(row["usable"])
    ]
    output = {row["clip_id"]: row for row in rows}
    if len(output) != 222:
        raise ValueError(f"expected 222 unique usable clips, found {len(output)}")
    return output


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_difference(
    reference: dict[str, dict[str, str]],
    candidate: dict[str, dict[str, str]],
    metric: str,
    seed: int,
) -> tuple[float, list[float], int]:
    if set(reference) != set(candidate):
        raise ValueError("reference and candidate clip IDs differ")
    by_source: dict[str, list[float]] = defaultdict(list)
    for clip_id in sorted(reference):
        source = clip_id.split("-")[1]
        by_source[source].append(
            float(reference[clip_id][metric]) - float(candidate[clip_id][metric])
        )
    clusters = sorted(by_source)
    estimate = mean(value for values in by_source.values() for value in values)
    rng = random.Random(seed)
    samples = []
    for _ in range(REPEATS):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        samples.append(mean(value for cluster in sampled for value in by_source[cluster]))
    return estimate, samples, len(clusters)


def main() -> None:
    consensus = EXTENSION / "returns/consensus_evaluation"
    owner = EXTENSION / "owner_full_quickcheck"
    reference = valid_map(
        consensus / "internal_conservative_final_consensus_110_by_clip.csv",
        owner / "internal_conservative_by_clip.csv",
    )

    catalogs: dict[str, tuple[Path, Path]] = {}
    for method in (
        "random_matched",
        "motion",
        "selfsim",
        "changepoint",
        "kernel_changepoint",
        "engineered_l2_instability",
        "raw_phi_l2",
        "no_critical",
    ):
        catalogs[method] = (
            ROOT / f"official_results/{method}/by_clip.csv",
            ROOT / f"owner_results/{method}/by_clip.csv",
        )
    for method in (
        "rgb_resnet",
        "video_r3d",
        "video_videomae",
        "video_swin",
        "pose_stgcn",
    ):
        catalogs[method] = (
            ROOT / f"main_learned/official_results/{method}/by_clip.csv",
            ROOT / f"main_learned/owner_results/{method}/by_clip.csv",
        )

    output: list[dict[str, object]] = []
    for method_index, (method, paths) in enumerate(catalogs.items()):
        candidate = valid_map(*paths)
        for metric_index, metric in enumerate(METRICS):
            estimate, samples, clusters = bootstrap_difference(
                reference,
                candidate,
                metric,
                SEED + 100 * method_index + metric_index,
            )
            output.append(
                {
                    "reference": "relis_conservative",
                    "candidate": method,
                    "metric": metric,
                    "reference_minus_candidate": estimate,
                    "ci95_low": percentile(samples, 0.025),
                    "ci95_high": percentile(samples, 0.975),
                    "bootstrap_probability_gt_zero": sum(value > 0 for value in samples)
                    / len(samples),
                    "clips": len(reference),
                    "source_clusters": clusters,
                    "bootstrap_repeats": REPEATS,
                    "seed": SEED + 100 * method_index + metric_index,
                }
            )

    path = ROOT / "pairwise_bootstrap_vs_relis.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} paired tests to {path}")


if __name__ == "__main__":
    main()
