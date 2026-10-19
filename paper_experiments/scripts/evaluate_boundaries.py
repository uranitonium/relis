#!/usr/bin/env python3
"""Evaluate predicted interaction-state partitions against expert labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from common import greedy_match_by_iou, precision_recall_f1, read_json, write_csv


Segment = Tuple[int, int, str]


def clip_segments(segments: List[Segment], lo: int, hi: int) -> List[Segment]:
    """Restrict a state partition to an inclusive evaluation interval."""
    return [
        (max(start, lo), min(end, hi), label)
        for start, end, label in segments
        if end >= lo and start <= hi
    ]


def gold_map(path: Path) -> Dict[str, List[Segment]]:
    payload = read_json(path)
    out: Dict[str, List[Segment]] = {}
    for ann in payload.get("clips", []):
        flags = ann.get("quality_flags", {})
        if not ann.get("annotation_complete") or flags.get("not_exchange") or not flags.get("valid", True):
            continue
        out[ann["clip_id"]] = [
            (int(segment["start_frame"]), int(segment["end_frame"]), str(segment["activity"]))
            for segment in ann.get("segments", [])
        ]
    return out


def pred_map(path: Path) -> Dict[str, List[Segment]]:
    payload = read_json(path)
    rows = payload.get("predictions", []) if isinstance(payload, dict) else payload
    out: Dict[str, List[Segment]] = {}
    for row in rows:
        clip_id = row.get("clip_id") or row.get("id") or row.get("file", "")
        clip_id = Path(str(clip_id)).name.replace("_l2.json", "").replace("_boundary.json", "")
        out[clip_id] = [
            (int(segment["start_frame"]), int(segment["end_frame"]), str(segment.get("activity", "stable")))
            for segment in row.get("state_segments", [])
        ]
    return out


def boundaries(segments: List[Segment]) -> List[int]:
    return [start for start, _, _ in segments[1:]]


def boundary_scores(pred: List[int], gold: List[int], tolerance: int) -> Tuple[float, float, float, float]:
    candidates = sorted((abs(p - g), pi, gi) for pi, p in enumerate(pred) for gi, g in enumerate(gold) if abs(p - g) <= tolerance)
    used_p, used_g, distances = set(), set(), []
    for distance, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        distances.append(distance)
    precision, recall, f1 = precision_recall_f1(len(distances), len(pred), len(gold))
    return precision, recall, f1, sum(distances) / len(distances) if distances else 0.0


def activity_scores(pred: List[Segment], gold: List[Segment]) -> Tuple[float, float]:
    if not gold:
        return 0.0, 0.0
    lo, hi = gold[0][0], gold[-1][1]
    def label_at(segments: List[Segment], frame: int) -> str:
        for start, end, label in segments:
            if start <= frame <= end:
                return label
        return "missing"
    pairs = [(label_at(pred, frame), label_at(gold, frame)) for frame in range(lo, hi + 1)]
    accuracy = sum(p == g for p, g in pairs) / len(pairs)
    f1s = []
    for label in ("stable", "active"):
        tp = sum(p == label and g == label for p, g in pairs)
        pred_count = sum(p == label for p, _ in pairs)
        gold_count = sum(g == label for _, g in pairs)
        f1s.append(precision_recall_f1(tp, pred_count, gold_count)[2])
    return accuracy, sum(f1s) / len(f1s)


def eval_one(method: str, preds: Dict[str, List[Segment]], golds: Dict[str, List[Segment]]) -> Dict[str, float | str]:
    rows = []
    for clip_id, gold in golds.items():
        pred = clip_segments(preds.get(clip_id, []), gold[0][0], gold[-1][1])
        row: Dict[str, float] = {}
        pred_intervals = [(start, end) for start, end, _ in pred]
        gold_intervals = [(start, end) for start, end, _ in gold]
        for threshold in (0.10, 0.25, 0.50):
            matches = greedy_match_by_iou(pred_intervals, gold_intervals, threshold)
            precision, recall, f1 = precision_recall_f1(len(matches), len(pred), len(gold))
            key = str(threshold).replace(".", "")
            row[f"segment_precision_{key}"] = precision
            row[f"segment_recall_{key}"] = recall
            row[f"segment_f1_{key}"] = f1
            row[f"mean_iou_{key}"] = sum(match[2] for match in matches) / len(matches) if matches else 0.0
        for tolerance in (3, 5, 10):
            precision, recall, f1, distance = boundary_scores(boundaries(pred), boundaries(gold), tolerance)
            row[f"boundary_precision_pm{tolerance}"] = precision
            row[f"boundary_recall_pm{tolerance}"] = recall
            row[f"boundary_f1_pm{tolerance}"] = f1
            row[f"mean_boundary_distance_pm{tolerance}"] = distance
        row["activity_frame_accuracy"], row["activity_macro_f1"] = activity_scores(pred, gold)
        row["false_states"] = max(0, len(pred) - len(gold))
        row["missed_states"] = max(0, len(gold) - len(pred))
        row["active_coverage"] = sum(end - start + 1 for start, end, label in pred if label == "active") / max(1, gold[-1][1] - gold[0][0] + 1)
        rows.append(row)
    summary: Dict[str, float | str] = {"method": method, "clips": len(rows)}
    if rows:
        for key in rows[0]:
            summary[key] = sum(row[key] for row in rows) / len(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate interaction-state predictions.")
    parser.add_argument("--gold", default="paper_experiments/annotations/gold/interaction_states.json")
    parser.add_argument("--pred", action="append", required=True, help="METHOD=path/to/predictions.json")
    parser.add_argument("--out", default="paper_experiments/paper_ready/state_partition_results.csv")
    args = parser.parse_args()
    golds = gold_map(Path(args.gold))
    summaries = []
    for spec in args.pred:
        if "=" not in spec:
            raise SystemExit("--pred must be METHOD=path")
        method, path = spec.split("=", 1)
        summaries.append(eval_one(method, pred_map(Path(path)), golds))
    write_csv(args.out, summaries)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
