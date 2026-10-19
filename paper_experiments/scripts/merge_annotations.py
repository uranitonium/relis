#!/usr/bin/env python3
"""Create state-partition gold labels and human-agreement reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from annotation_schema import normalize_annotation
from common import precision_recall_f1, read_json, write_csv, write_json


def load_raw(root: Path, include_incomplete: bool = False) -> Dict[str, Dict[str, dict]]:
    by_clip: Dict[str, Dict[str, dict]] = {}
    for file in sorted(root.glob("*/*.json")):
        ann = normalize_annotation(read_json(file), strict=False)
        if not include_incomplete and not ann["annotation_complete"]:
            continue
        by_clip.setdefault(ann["clip_id"], {})[ann["annotator_id"]] = ann
    return by_clip


def evaluation_range(ann: dict) -> Tuple[int, int]:
    value = ann.get("evaluation_range") or ann.get("analyzed_range") or {}
    return int(value.get("start_frame", 0)), int(value.get("end_frame", 0))


def boundaries(ann: dict, lo: int, hi: int) -> List[int]:
    return [
        int(segment["start_frame"])
        for segment in ann.get("segments", [])[1:]
        if lo < int(segment["start_frame"]) <= hi
    ]


def match_boundaries(a: List[int], b: List[int], tolerance: int) -> Tuple[int, List[int]]:
    candidates = sorted((abs(x - y), ai, bi) for ai, x in enumerate(a) for bi, y in enumerate(b) if abs(x - y) <= tolerance)
    used_a, used_b, distances = set(), set(), []
    for distance, ai, bi in candidates:
        if ai in used_a or bi in used_b:
            continue
        used_a.add(ai)
        used_b.add(bi)
        distances.append(distance)
    return len(distances), distances


def activity_agreement(a: dict, b: dict) -> float:
    start_a, end_a = evaluation_range(a)
    start_b, end_b = evaluation_range(b)
    lo = max(start_a, start_b)
    hi = min(end_a, end_b)
    if hi < lo:
        return 0.0
    return sum(label_at(a, frame) == label_at(b, frame) for frame in range(lo, hi + 1)) / (hi - lo + 1)


def label_at(ann: dict, frame: int) -> str:
    for segment in ann.get("segments", []):
        if segment["start_frame"] <= frame <= segment["end_frame"]:
            return segment["activity"]
    return ""


def active_coverage(ann: dict, lo: int, hi: int) -> float:
    if hi < lo:
        return 0.0
    active = sum(label_at(ann, frame) == "active" for frame in range(lo, hi + 1))
    return active / (hi - lo + 1)


def pair_metrics(a: dict, b: dict) -> dict:
    start_a, end_a = evaluation_range(a)
    start_b, end_b = evaluation_range(b)
    lo, hi = max(start_a, start_b), min(end_a, end_b)
    ba, bb = boundaries(a, lo, hi), boundaries(b, lo, hi)
    out = {
        "states_a": len(ba) + 1,
        "states_b": len(bb) + 1,
        "exchange_end_a": end_a,
        "exchange_end_b": end_b,
        "exchange_end_distance": abs(end_a - end_b),
        "activity_frame_agreement": activity_agreement(a, b),
        "active_coverage_a": active_coverage(a, lo, hi),
        "active_coverage_b": active_coverage(b, lo, hi),
        "active_coverage_difference_b_minus_a": active_coverage(b, lo, hi) - active_coverage(a, lo, hi),
        "state_count_difference_b_minus_a": len(bb) - len(ba),
    }
    for tolerance in (3, 5, 10):
        hits, distances = match_boundaries(ba, bb, tolerance)
        precision, recall, f1 = precision_recall_f1(hits, len(ba), len(bb))
        out.update({f"boundary_precision_pm{tolerance}": precision, f"boundary_recall_pm{tolerance}": recall, f"boundary_f1_pm{tolerance}": f1, f"mean_boundary_distance_pm{tolerance}": sum(distances) / len(distances) if distances else 0.0})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge interaction-state annotations and compute agreement.")
    parser.add_argument("--raw-root", default="paper_experiments/annotations/raw")
    parser.add_argument("--primary-annotator", default="")
    parser.add_argument("--gold-out", default="paper_experiments/annotations/gold/interaction_states.json")
    parser.add_argument("--agreement-out", default="paper_experiments/annotations/agreement_report.csv")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include present but incomplete references, for explicit invalid-clip adjudication.",
    )
    args = parser.parse_args()
    by_clip = load_raw(Path(args.raw_root), include_incomplete=args.include_incomplete)
    gold = {"schema_version": 2, "annotation_task": "interaction_state_partition", "clips": []}
    agreement_rows: List[dict] = []
    for clip_id, anns in sorted(by_clip.items()):
        annotators = sorted(anns)
        if len(annotators) >= 2:
            first, second = annotators[:2]
            row = {"clip_id": clip_id, "annotator_a": first, "annotator_b": second}
            row.update(pair_metrics(anns[first], anns[second]))
            agreement_rows.append(row)
        chosen = anns.get(args.primary_annotator) if args.primary_annotator else None
        if chosen is None and annotators:
            chosen = anns[annotators[0]]
        if chosen is not None:
            gold["clips"].append(chosen)
    write_json(args.gold_out, gold)
    write_csv(args.agreement_out, agreement_rows)
    print(f"Wrote gold annotations: {args.gold_out} clips={len(gold['clips'])}")
    print(f"Wrote agreement report: {args.agreement_out} pairs={len(agreement_rows)}")


if __name__ == "__main__":
    main()
