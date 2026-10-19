#!/usr/bin/env python3
"""Prepare leakage-resistant data splits for hit-truncated boundary retraining."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


FIELDS = [
    "clip_id",
    "weapon",
    "outcome_label",
    "prefix",
    "video_path",
    "l2_path",
    "fps",
    "frame_start",
    "frame_end",
    "num_frames",
    "hit_frame",
    "evaluation_end_source",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else FIELDS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stratified_split(rows: List[Dict[str, str]], dev_fraction: float, seed: int):
    rng = random.Random(seed)
    by_outcome: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_outcome.setdefault(row["outcome_label"], []).append(row)
    train: List[Dict[str, str]] = []
    audit: List[Dict[str, str]] = []
    for outcome, group in sorted(by_outcome.items()):
        rng.shuffle(group)
        count = max(1, round(len(group) * dev_fraction))
        audit.extend(group[:count])
        train.extend(group[count:])
    return sorted(train, key=lambda row: row["clip_id"]), sorted(audit, key=lambda row: row["clip_id"])


def link_l2(rows: List[Dict[str, str]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    expected = set()
    for row in rows:
        source = Path(row["l2_path"]).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        target = destination / source.name
        expected.add(target.name)
        if target.is_symlink() and target.resolve() == source:
            continue
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"Refusing to replace unexpected data link: {target}")
        target.symlink_to(source)
    extras = [path for path in destination.glob("*_l2.json") if path.name not in expected]
    if extras:
        raise RuntimeError(f"Unexpected existing links in {destination}: {extras[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-manifest", required=True)
    parser.add_argument("--paper-manifest", default="paper_experiments/manifests/foil_main.csv")
    parser.add_argument("--owner-dir", default="paper_experiments/annotations/raw/owner")
    parser.add_argument("--workspace", default="paper_experiments/experiments/truncated_retraining")
    parser.add_argument("--audit-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20270614)
    args = parser.parse_args()

    root = Path(args.workspace)
    manifests = root / "manifests"
    all_rows = read_csv(Path(args.all_manifest))
    paper_rows = read_csv(Path(args.paper_manifest))
    paper_ids = {row["clip_id"] for row in paper_rows}
    available = {row["clip_id"]: row for row in all_rows}
    missing = sorted(paper_ids - available.keys())
    if missing:
        raise RuntimeError(f"Paper clips absent from truncated data: {missing[:10]}")

    nonpaper = [row for row in all_rows if row["clip_id"] not in paper_ids]
    train, audit = stratified_split(nonpaper, args.audit_fraction, args.seed)
    paper = [available[row["clip_id"]] for row in paper_rows]

    owner_ids = {path.stem for path in Path(args.owner_dir).glob("*.json")}
    development = [row for row in paper if row["clip_id"] in owner_ids]
    final_candidates = [row for row in paper if row["clip_id"] not in owner_ids]

    split_rows = {
        "train": train,
        "unlabeled_audit": audit,
        "paper_all_held_out": paper,
        "paper_development": development,
        "paper_final_candidates": final_candidates,
    }
    for name, rows in split_rows.items():
        write_csv(manifests / f"{name}.csv", rows)

    link_l2(train, root / "data" / "train")
    link_l2(audit, root / "data" / "unlabeled_audit")

    train_ids = {row["clip_id"] for row in train}
    audit_ids = {row["clip_id"] for row in audit}
    assert not train_ids & audit_ids
    assert not train_ids & paper_ids
    assert not audit_ids & paper_ids

    report = {
        "seed": args.seed,
        "audit_fraction": args.audit_fraction,
        "source_manifest": args.all_manifest,
        "paper_manifest": args.paper_manifest,
        "splits": {
            name: {
                "clips": len(rows),
                "outcomes": dict(sorted(Counter(row["outcome_label"] for row in rows).items())),
            }
            for name, rows in split_rows.items()
        },
        "leakage_checks": {
            "train_audit_overlap": 0,
            "train_paper_overlap": 0,
            "audit_paper_overlap": 0,
        },
        "notes": [
            "All 150 paper clips are excluded from unsupervised training.",
            "The current 40 owner-labeled clips are development data only.",
            "The remaining paper clips are reserved as final-test candidates.",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "split_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
