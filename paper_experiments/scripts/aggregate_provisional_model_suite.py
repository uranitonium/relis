#!/usr/bin/env python3
"""Aggregate provisional owner-as-consensus evaluations for frozen models."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


REPO = Path(__file__).resolve().parents[2]
EXTENSION = REPO / "paper_experiments/round2/annotation_extension"
ROOT = EXTENSION / "provisional_owner_as_consensus/other_models"
METRICS = (
    "activity_accuracy",
    "activity_macro_f1",
    "active_f1",
    "boundary_f1_pm3",
    "boundary_f1_pm5",
    "boundary_f1_pm10",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def valid_rows(path: Path) -> list[dict[str, str]]:
    return [row for row in read_csv(path) if truthy(row["usable"])]


def cohort_row(
    group: str,
    method: str,
    seed: str,
    cohort: str,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    result: dict[str, object] = {
        "group": group,
        "method": method,
        "seed": seed,
        "cohort": cohort,
        "clips": len(rows),
        "source_ids": len({row["clip_id"].split("-")[1] for row in rows}),
    }
    for metric in METRICS:
        result[metric] = mean(float(row[metric]) for row in rows)
    return result


def add_run(
    output: list[dict[str, object]],
    group: str,
    method: str,
    seed: str,
    official_path: Path,
    owner_path: Path,
) -> None:
    official = valid_rows(official_path)
    owner = valid_rows(owner_path)
    official_ids = {row["clip_id"] for row in official}
    owner_ids = {row["clip_id"] for row in owner}
    overlap = official_ids & owner_ids
    if overlap:
        raise ValueError(f"{method} has overlapping official/owner clips: {sorted(overlap)}")
    if len(official) != 105 or len(owner) != 117:
        raise ValueError(
            f"{method} expected 105 official and 117 owner clips; "
            f"found {len(official)} and {len(owner)}"
        )
    output.extend(
        [
            cohort_row(group, method, seed, "official_consensus", official),
            cohort_row(group, method, seed, "owner_as_consensus", owner),
            cohort_row(group, method, seed, "combined_provisional", official + owner),
        ]
    )


def main() -> None:
    per_run: list[dict[str, object]] = []

    consensus_eval = EXTENSION / "returns/consensus_evaluation"
    owner_eval = EXTENSION / "owner_full_quickcheck"
    for calibration in ("conservative", "optimistic"):
        add_run(
            per_run,
            "primary",
            f"relis_{calibration}",
            "20270618",
            consensus_eval / f"internal_{calibration}_final_consensus_110_by_clip.csv",
            owner_eval / f"internal_{calibration}_by_clip.csv",
        )

    generic_methods = (
        "uniform",
        "random_matched",
        "motion",
        "selfsim",
        "changepoint",
        "kernel_changepoint",
        "engineered_l2_instability",
        "raw_phi_l2",
        "no_critical",
    )
    for method in generic_methods:
        add_run(
            per_run,
            "main_baseline",
            method,
            "fixed",
            ROOT / f"official_results/{method}/by_clip.csv",
            ROOT / f"owner_results/{method}/by_clip.csv",
        )

    main_learned_root = ROOT / "main_learned"
    for method in (
        "rgb_resnet",
        "video_r3d",
        "video_videomae",
        "video_swin",
        "pose_stgcn",
    ):
        add_run(
            per_run,
            "main_learned",
            method,
            "20270618",
            main_learned_root / f"official_results/{method}/by_clip.csv",
            main_learned_root / f"owner_results/{method}/by_clip.csv",
        )

    architecture_root = ROOT / "architecture"
    architecture_metrics = read_csv(
        REPO / "paper_experiments/round2/architecture_ablations/per_seed_metrics.csv"
    )
    for row in architecture_metrics:
        name = f"{row['variant']}_seed{row['seed']}"
        add_run(
            per_run,
            "architecture",
            row["variant"],
            row["seed"],
            architecture_root / f"official_results/{name}/by_clip.csv",
            architecture_root / f"owner_results/{name}/by_clip.csv",
        )

    representation_root = ROOT / "representation"
    representation_metrics = (
        REPO / "paper_experiments/round2/representation_baseline_equalized/per_seed_metrics.csv"
    )
    if representation_root.exists():
        for row in read_csv(representation_metrics):
            if row["method"] == "relis":
                continue
            name = f"{row['method']}_seed{row['seed']}"
            official = representation_root / f"official_results/{name}/by_clip.csv"
            owner = representation_root / f"owner_results/{name}/by_clip.csv"
            if official.exists() and owner.exists():
                add_run(
                    per_run,
                    "representation",
                    row["method"],
                    row["seed"],
                    official,
                    owner,
                )

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_run:
        grouped[(str(row["group"]), str(row["method"]), str(row["cohort"]))].append(row)
    aggregate: list[dict[str, object]] = []
    for (group, method, cohort), rows in sorted(grouped.items()):
        result: dict[str, object] = {
            "group": group,
            "method": method,
            "cohort": cohort,
            "seeds": len(rows),
            "clips_per_seed": rows[0]["clips"],
            "source_ids": rows[0]["source_ids"],
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
        aggregate.append(result)

    write_csv(ROOT / "per_run_results.csv", per_run)
    write_csv(ROOT / "aggregate_results.csv", aggregate)
    print(f"Wrote {len(per_run)} per-run cohort rows and {len(aggregate)} aggregates")


if __name__ == "__main__":
    main()
