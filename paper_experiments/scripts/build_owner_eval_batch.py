#!/usr/bin/env python3
"""Select a balanced owner annotation batch from untouched paper candidates."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, str]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def family(clip_id: str) -> str:
    return clip_id.split("-", 1)[0]


def round_robin_families(rows: List[Dict[str, str]], count: int, rng: random.Random) -> List[Dict[str, str]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[family(row["clip_id"])].append(row)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected: List[Dict[str, str]] = []
    while len(selected) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv",
    )
    parser.add_argument("--owner-dir", default="paper_experiments/annotations/raw/owner")
    parser.add_argument("--existing-manifest", action="append", default=[])
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--out", default="paper_experiments/manifests/foil_owner_eval_part_01.csv")
    parser.add_argument(
        "--report",
        default="paper_experiments/manifests/foil_owner_eval_part_01_report.json",
    )
    args = parser.parse_args()

    rows = read_csv(Path(args.source))
    excluded = {path.stem for path in Path(args.owner_dir).glob("*.json")}
    for manifest in args.existing_manifest:
        excluded.update(row["clip_id"] for row in read_csv(Path(manifest)))
    rows = [row for row in rows if row["clip_id"] not in excluded]

    quotas = {"Left": args.count // 3, "None": args.count // 3, "Right": args.count // 3}
    for outcome in ("Left", "None", "Right")[: args.count % 3]:
        quotas[outcome] += 1

    rng = random.Random(args.seed)
    selected: List[Dict[str, str]] = []
    for outcome in ("Left", "None", "Right"):
        candidates = [row for row in rows if row["outcome_label"] == outcome]
        if outcome == "None":
            # Preserve the rare T-prefix examples in the evaluation set.
            t_rows = [row for row in candidates if row.get("prefix") == "T"]
            selected.extend(t_rows[: quotas[outcome]])
            candidates = [row for row in candidates if row not in t_rows]
            needed = quotas[outcome] - min(len(t_rows), quotas[outcome])
        else:
            needed = quotas[outcome]
        selected.extend(round_robin_families(candidates, needed, rng))

    selected.sort(key=lambda row: row["clip_id"])
    if len(selected) != args.count:
        raise RuntimeError(f"Requested {args.count} clips but selected {len(selected)}")
    fields = list(rows[0].keys())
    write_csv(Path(args.out), selected, fields)

    outcome_counts = {outcome: sum(row["outcome_label"] == outcome for row in selected) for outcome in quotas}
    family_counts: Dict[str, int] = defaultdict(int)
    for row in selected:
        family_counts[family(row["clip_id"])] += 1
    report = {
        "purpose": "blind held-out evaluation annotation; never use for model or threshold tuning",
        "source_manifest": args.source,
        "output_manifest": args.out,
        "seed": args.seed,
        "clips": len(selected),
        "outcomes": outcome_counts,
        "families": dict(sorted(family_counts.items())),
        "existing_owner_annotations_excluded": len(excluded),
        "clip_ids": [row["clip_id"] for row in selected],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
