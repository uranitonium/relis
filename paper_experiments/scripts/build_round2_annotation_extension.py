#!/usr/bin/env python3
"""Build the frozen Round 2 annotation manifests and coverage audit.

The primary extension contains every usable foil exchange whose exact clip ID
is absent from training, development, and all existing human annotations.  No
model output, score, or prediction is used for selection.  The resulting cohort
is exchange-disjoint, but not bout-disjoint: the current training manifest
already contains at least one exchange from every available middle-field bout
ID.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_OUT = Path("paper_experiments/round2/annotation_extension")
OUTCOMES = {"Left", "Right", "None"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def source_id(clip_id: str) -> str:
    """Return the bout/source-video ID encoded in the middle clip-ID field."""
    parts = clip_id.split("-")
    if len(parts) != 3 or not parts[1]:
        raise ValueError(f"clip ID does not have PREFIX-SOURCE-INDEX form: {clip_id}")
    return parts[1]


def annotation_ids(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.json")} if directory.exists() else set()


def manifest_ids(path: Path) -> set[str]:
    return {row["clip_id"] for row in read_csv(path)}


def flags_text(payload: dict) -> str:
    flags = payload.get("quality_flags") or {}
    return ";".join(name for name, value in flags.items() if value and name != "valid")


def valid_complete(payload: dict) -> bool:
    flags = payload.get("quality_flags") or {}
    return bool(payload.get("annotation_complete")) and bool(flags.get("valid", True)) and not bool(flags.get("not_exchange", False))


def distribute(rows: list[dict[str, str]], parts: int, seed: int) -> list[list[dict[str, str]]]:
    """Distribute all rows while balancing source repetition and outcomes."""
    if parts < 1:
        raise ValueError("parts must be positive")
    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[source_id(row["clip_id"])].append(row)
    groups = list(by_source.items())
    for _, group in groups:
        rng.shuffle(group)
    rng.shuffle(groups)
    groups.sort(key=lambda item: -len(item[1]))

    buckets: list[list[dict[str, str]]] = [[] for _ in range(parts)]
    base, remainder = divmod(len(rows), parts)
    capacities = [base + (1 if idx < remainder else 0) for idx in range(parts)]
    outcome_counts: list[Counter[str]] = [Counter() for _ in range(parts)]
    source_counts: list[Counter[str]] = [Counter() for _ in range(parts)]
    for sid, group in groups:
        for row in group:
            outcome = row["outcome_label"]
            available = [idx for idx in range(parts) if len(buckets[idx]) < capacities[idx]]
            choice = min(
                available,
                key=lambda idx: (
                    source_counts[idx][sid],
                    outcome_counts[idx][outcome],
                    len(buckets[idx]),
                    idx,
                ),
            )
            buckets[choice].append(row)
            source_counts[choice][sid] += 1
            outcome_counts[choice][outcome] += 1
    for bucket in buckets:
        bucket.sort(key=lambda row: row["clip_id"])
    return buckets


def agreement_queue(
    agreement_path: Path,
    eligible_ids: set[str],
    raw_root: Path,
) -> list[dict[str, object]]:
    rows = {row["clip_id"]: row for row in read_csv(agreement_path)}
    queue: list[dict[str, object]] = []
    for clip_id in sorted(eligible_ids):
        if clip_id not in rows:
            continue
        row: dict[str, object] = dict(rows[clip_id])
        collaborator = read_json(raw_root / "collaborator_a" / f"{clip_id}.json")
        owner = read_json(raw_root / "owner" / f"{clip_id}.json")
        # Human-only disagreement priority used by the earlier adjudication:
        # one minus frame agreement plus one minus BF1@5.
        priority = (1.0 - float(row["activity_frame_agreement"])) + (1.0 - float(row["boundary_f1_pm5"]))
        row["collaborator_flags"] = flags_text(collaborator)
        row["owner_flags"] = flags_text(owner)
        row["adjudication_priority"] = priority
        queue.append(row)
    queue.sort(key=lambda row: (-float(row["adjudication_priority"]), str(row["clip_id"])))
    return queue


def numeric_sort(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="paper_experiments/manifests/foil_all_available.csv")
    parser.add_argument("--train", default="paper_experiments/experiments/truncated_retraining/manifests/train.csv")
    parser.add_argument("--development", default="paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv")
    parser.add_argument("--raw-root", default="paper_experiments/annotations/raw")
    parser.add_argument("--consensus-dir", default="paper_experiments/annotations/adjudicated/consensus")
    parser.add_argument("--agreement", default="paper_experiments/annotations/agreement_report.csv")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--parts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    source_path = Path(args.source)
    train_path = Path(args.train)
    development_path = Path(args.development)
    raw_root = Path(args.raw_root)
    consensus_dir = Path(args.consensus_dir)
    out_root = Path(args.out_root)
    manifests_dir = out_root / "manifests"
    reports_dir = out_root / "reports"

    source_rows = [
        row
        for row in read_csv(source_path)
        if row.get("outcome_label") in OUTCOMES and not row["clip_id"].startswith("D")
    ]
    if not source_rows:
        raise RuntimeError("no usable source rows")
    source_by_id = {row["clip_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise RuntimeError("source manifest contains duplicate clip IDs")
    missing_videos = [row["clip_id"] for row in source_rows if not Path(row["video_path"]).exists()]
    if missing_videos:
        raise FileNotFoundError(f"{len(missing_videos)} source videos are missing; first={missing_videos[0]}")

    train_ids = manifest_ids(train_path)
    development_ids = manifest_ids(development_path)
    owner_ids = annotation_ids(raw_root / "owner")
    collaborator_ids = annotation_ids(raw_root / "collaborator_a")
    consensus_ids = annotation_ids(consensus_dir)
    existing_annotation_ids = owner_ids | collaborator_ids | consensus_ids
    excluded_ids = train_ids | development_ids | existing_annotation_ids

    extension = sorted(
        (row for row in source_rows if row["clip_id"] not in excluded_ids),
        key=lambda row: row["clip_id"],
    )
    fields = list(source_rows[0].keys())
    master_path = manifests_dir / "foil_round2_exchange_disjoint_all_124.csv"
    if len(extension) != 124:
        raise RuntimeError(
            f"frozen extension expected 124 clips but found {len(extension)}; "
            "audit changed inputs before regenerating packages"
        )
    write_csv(master_path, extension, fields)

    buckets = distribute(extension, args.parts, args.seed)
    part_paths: list[str] = []
    for index, bucket in enumerate(buckets, 1):
        path = manifests_dir / f"foil_round2_exchange_disjoint_part_{index:02d}.csv"
        write_csv(path, bucket, fields)
        part_paths.append(str(path))

    all_source_ids = {source_id(row["clip_id"]) for row in source_rows}
    train_source_ids = {source_id(clip_id) for clip_id in train_ids}
    development_source_ids = {source_id(clip_id) for clip_id in development_ids}
    existing_source_ids = {source_id(clip_id) for clip_id in existing_annotation_ids}
    extension_source_ids = {source_id(row["clip_id"]) for row in extension}
    covered_source_ids = existing_source_ids | extension_source_ids
    unavailable_for_exchange_disjoint = all_source_ids - covered_source_ids
    observed_numeric = {int(value) for value in all_source_ids if value.isdigit()}
    missing_numeric = sorted(set(range(100)) - observed_numeric)

    overlap_checks = {
        "extension_vs_train": len({row["clip_id"] for row in extension} & train_ids),
        "extension_vs_development": len({row["clip_id"] for row in extension} & development_ids),
        "extension_vs_existing_annotations": len({row["clip_id"] for row in extension} & existing_annotation_ids),
        "extension_source_ids_also_in_train": len(extension_source_ids & train_source_ids),
    }
    report = {
        "frozen_protocol_date": "2026-08-10",
        "selection": "all eligible clips; no model output or score used",
        "selection_seed_only_used_for_part_assignment": args.seed,
        "source_id_definition": "middle field of PREFIX-SOURCE-INDEX clip ID",
        "evaluation_unit": "exchange/clip",
        "valid_claim": "exact-clip/exchange-disjoint from training and development",
        "invalid_claim": "bout/source-video-disjoint",
        "reason_not_bout_disjoint": "the training manifest contains exchanges from every currently available source ID",
        "source_manifest": str(source_path),
        "train_manifest": str(train_path),
        "development_manifest": str(development_path),
        "master_manifest": str(master_path),
        "part_manifests": part_paths,
        "inventory": {
            "usable_exchange_clips": len(source_rows),
            "available_source_ids": len(all_source_ids),
            "available_source_id_values": numeric_sort(all_source_ids),
            "missing_ids_within_expected_0_to_99": missing_numeric,
            "train_clips": len(train_ids),
            "train_source_ids": len(train_source_ids),
            "development_clips": len(development_ids),
            "development_source_ids": len(development_source_ids),
            "owner_annotation_files": len(owner_ids),
            "collaborator_annotation_files": len(collaborator_ids),
            "consensus_files": len(consensus_ids),
            "existing_annotation_source_ids": len(existing_source_ids),
        },
        "extension": {
            "clips": len(extension),
            "source_ids": len(extension_source_ids),
            "outcome_counts": dict(sorted(Counter(row["outcome_label"] for row in extension).items())),
            "part_sizes": [len(bucket) for bucket in buckets],
            "part_source_id_counts": [len({source_id(row["clip_id"]) for row in bucket}) for bucket in buckets],
            "part_outcome_counts": [dict(sorted(Counter(row["outcome_label"] for row in bucket).items())) for bucket in buckets],
        },
        "combined_annotation_coverage_after_extension": {
            "source_ids": len(covered_source_ids),
            "of_available_source_ids": len(all_source_ids),
            "source_id_values": numeric_sort(covered_source_ids),
            "uncovered_source_ids": numeric_sort(unavailable_for_exchange_disjoint),
            "uncovered_reason": "all usable exchanges for these IDs are already in training/development and none was previously annotated",
        },
        "overlap_checks": overlap_checks,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "coverage_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    double_ids = owner_ids & collaborator_ids
    unadjudicated = double_ids - consensus_ids
    completed_pairs: set[str] = set()
    incomplete_pairs: set[str] = set()
    pair_status: list[dict[str, object]] = []
    for clip_id in sorted(unadjudicated):
        owner = read_json(raw_root / "owner" / f"{clip_id}.json")
        collaborator = read_json(raw_root / "collaborator_a" / f"{clip_id}.json")
        both_complete = bool(owner.get("annotation_complete")) and bool(collaborator.get("annotation_complete"))
        (completed_pairs if both_complete else incomplete_pairs).add(clip_id)
        pair_status.append(
            {
                "clip_id": clip_id,
                "source_id": source_id(clip_id),
                "owner_complete": bool(owner.get("annotation_complete")),
                "collaborator_complete": bool(collaborator.get("annotation_complete")),
                "owner_valid_complete": valid_complete(owner),
                "collaborator_valid_complete": valid_complete(collaborator),
                "owner_flags": flags_text(owner),
                "collaborator_flags": flags_text(collaborator),
                "next_action": "adjudicate" if both_complete else "finish_blind_annotation_then_adjudicate",
            }
        )

    queue = agreement_queue(Path(args.agreement), completed_pairs, raw_root)
    if len(queue) != len(completed_pairs):
        raise RuntimeError(f"agreement report covers {len(queue)}/{len(completed_pairs)} completed unadjudicated pairs")
    queue_fields = list(queue[0].keys())
    queue_path = out_root / "adjudication" / "existing_completed_pairs_42_queue.csv"
    write_csv(queue_path, queue, queue_fields)
    write_csv(
        reports_dir / "existing_pair_status.csv",
        pair_status,
        [
            "clip_id",
            "source_id",
            "owner_complete",
            "collaborator_complete",
            "owner_valid_complete",
            "collaborator_valid_complete",
            "owner_flags",
            "collaborator_flags",
            "next_action",
        ],
    )

    catchup = [source_by_id[clip_id] for clip_id in sorted(incomplete_pairs)]
    catchup_path = manifests_dir / "foil_round2_collaborator_catchup_1.csv"
    write_csv(catchup_path, catchup, fields)
    adjudication_report = {
        "double_labeled_not_yet_adjudicated": len(unadjudicated),
        "ready_for_adjudication": len(completed_pairs),
        "requires_blind_completion_first": len(incomplete_pairs),
        "requires_blind_completion_clip_ids": sorted(incomplete_pairs),
        "queue": str(queue_path),
        "catchup_manifest": str(catchup_path),
        "priority_formula": "(1 - activity_frame_agreement) + (1 - boundary_f1_pm5)",
        "model_predictions_used": False,
    }
    (reports_dir / "adjudication_report.json").write_text(json.dumps(adjudication_report, indent=2) + "\n", encoding="utf-8")

    if any(overlap_checks[key] for key in ("extension_vs_train", "extension_vs_development", "extension_vs_existing_annotations")):
        raise RuntimeError(f"nonzero exact-clip overlap: {overlap_checks}")
    if set().union(*(set(row["clip_id"] for row in bucket) for bucket in buckets)) != {row["clip_id"] for row in extension}:
        raise RuntimeError("part manifests do not cover the master extension")
    if sum(len(bucket) for bucket in buckets) != len(extension):
        raise RuntimeError("part manifests contain duplicate clips")

    print(json.dumps({"coverage": report, "adjudication": adjudication_report}, indent=2))


if __name__ == "__main__":
    main()
