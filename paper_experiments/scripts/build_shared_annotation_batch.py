#!/usr/bin/env python3
"""Select a balanced shared annotation batch from previously unlabeled clips."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_annotation_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {p.stem for p in path.glob("*.json")}


def read_manifest_ids(paths: Iterable[str]) -> Set[str]:
    ids: Set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        ids.update(row["clip_id"] for row in read_csv(path))
    return ids


def bout_id(clip_id: str) -> str:
    parts = clip_id.split("-")
    return parts[1] if len(parts) >= 3 else clip_id


def prefix_family(clip_id: str) -> str:
    return clip_id.split("-", 1)[0]


def choose_balanced(rows: List[Dict[str, str]], count: int, seed: int) -> List[Dict[str, str]]:
    quotas = {"Left": count // 3, "Right": count // 3, "None": count // 3}
    for outcome in ("Left", "Right", "None")[: count % 3]:
        quotas[outcome] += 1

    rng = random.Random(seed)
    selected: List[Dict[str, str]] = []
    used_bouts: Set[str] = set()
    used_families: Counter[str] = Counter()

    by_outcome: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_outcome[row["outcome_label"]].append(row)
    for candidates in by_outcome.values():
        rng.shuffle(candidates)

    for outcome in ("Left", "Right", "None"):
        candidates = by_outcome[outcome]
        picked: List[Dict[str, str]] = []

        # First pass: maximize bout diversity.
        for row in candidates:
            if len(picked) >= quotas[outcome]:
                break
            bid = bout_id(row["clip_id"])
            if bid in used_bouts:
                continue
            picked.append(row)
            used_bouts.add(bid)
            used_families[prefix_family(row["clip_id"])] += 1

        # Second pass: fill shortages, preferring underrepresented prefix families.
        if len(picked) < quotas[outcome]:
            remaining = [row for row in candidates if row not in picked]
            remaining.sort(key=lambda row: (used_families[prefix_family(row["clip_id"])], row["clip_id"]))
            for row in remaining:
                if len(picked) >= quotas[outcome]:
                    break
                picked.append(row)
                used_bouts.add(bout_id(row["clip_id"]))
                used_families[prefix_family(row["clip_id"])] += 1

        selected.extend(picked)

    if len(selected) != count:
        raise RuntimeError(f"Requested {count} clips but selected {len(selected)}")
    selected.sort(key=lambda row: row["clip_id"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="paper_experiments/manifests/foil_all_available.csv")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--out", default="paper_experiments/manifests/foil_shared_extension_50.csv")
    parser.add_argument("--report", default="paper_experiments/manifests/foil_shared_extension_50_report.json")
    parser.add_argument("--owner-dir", default="paper_experiments/annotations/raw/owner")
    parser.add_argument("--collaborator-dir", default="paper_experiments/annotations/raw/collaborator_a")
    parser.add_argument("--consensus-dir", default="paper_experiments/annotations/adjudicated/consensus")
    parser.add_argument("--exclude-manifest", action="append", default=[])
    args = parser.parse_args()

    rows = read_csv(Path(args.source))
    excluded = set()
    excluded.update(read_annotation_ids(Path(args.owner_dir)))
    excluded.update(read_annotation_ids(Path(args.collaborator_dir)))
    excluded.update(read_annotation_ids(Path(args.consensus_dir)))
    excluded.update(read_manifest_ids(args.exclude_manifest))

    candidates = [
        row
        for row in rows
        if row["clip_id"] not in excluded
        and row.get("outcome_label") in {"Left", "Right", "None"}
        and not row["clip_id"].startswith("D")
    ]
    selected = choose_balanced(candidates, args.count, args.seed)

    fields = list(rows[0].keys())
    write_csv(Path(args.out), selected, fields)

    report = {
        "purpose": "shared blind extension batch for additional consensus foil annotations",
        "source_manifest": args.source,
        "output_manifest": args.out,
        "seed": args.seed,
        "requested_clips": args.count,
        "selected_clips": len(selected),
        "excluded_existing_clip_ids": len(excluded),
        "candidate_clips_after_exclusion": len(candidates),
        "outcome_counts": dict(Counter(row["outcome_label"] for row in selected)),
        "prefix_counts": dict(sorted(Counter(prefix_family(row["clip_id"]) for row in selected).items())),
        "unique_bout_ids": len({bout_id(row["clip_id"]) for row in selected}),
        "bout_ids": sorted({bout_id(row["clip_id"]) for row in selected}),
        "clip_ids": [row["clip_id"] for row in selected],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
