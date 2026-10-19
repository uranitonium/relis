#!/usr/bin/env python3
"""Compare expert state partitions with deployed-model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from annotation_schema import normalize_annotation
from common import precision_recall_f1, read_csv, read_json, write_csv, write_json
from evaluate_boundaries import activity_scores, boundary_scores, clip_segments, pred_map


Segment = Tuple[int, int, str]


def annotation_segments(annotation: dict) -> List[Segment]:
    return [
        (int(segment["start_frame"]), int(segment["end_frame"]), str(segment["activity"]))
        for segment in annotation.get("segments", [])
    ]


def boundaries(segments: List[Segment]) -> List[int]:
    return [start for start, _, _ in segments[1:]]


def label_at(segments: List[Segment], frame: int) -> str:
    for start, end, label in segments:
        if start <= frame <= end:
            return label
    return "missing"


def activity_class_scores(pred: List[Segment], gold: List[Segment]) -> dict:
    lo, hi = gold[0][0], gold[-1][1]
    pairs = [(label_at(pred, frame), label_at(gold, frame)) for frame in range(lo, hi + 1)]
    tp = sum(p == "active" and g == "active" for p, g in pairs)
    pred_count = sum(p == "active" for p, _ in pairs)
    gold_count = sum(g == "active" for _, g in pairs)
    precision, recall, f1 = precision_recall_f1(tp, pred_count, gold_count)
    return {
        "active_precision": precision,
        "active_recall": recall,
        "active_f1": f1,
        "model_active_coverage": pred_count / len(pairs),
        "human_active_coverage": gold_count / len(pairs),
    }


def quality_labels(annotation: dict) -> List[str]:
    flags = annotation.get("quality_flags", {})
    return [name for name, enabled in flags.items() if enabled and name != "valid"]


def is_usable(annotation: dict) -> bool:
    flags = annotation.get("quality_flags", {})
    return bool(annotation.get("annotation_complete")) and bool(flags.get("valid", True)) and not flags.get("not_exchange")


def is_annotated_exchange(annotation: dict) -> bool:
    flags = annotation.get("quality_flags", {})
    return bool(annotation.get("annotation_complete")) and bool(annotation.get("segments")) and not flags.get("not_exchange")


def is_clean(annotation: dict) -> bool:
    return is_usable(annotation) and not quality_labels(annotation)


def clip_metrics(clip_id: str, annotation: dict, pred: List[Segment]) -> dict:
    gold = annotation_segments(annotation)
    full_pred = pred
    pred = clip_segments(full_pred, gold[0][0], gold[-1][1])
    analyzed_end = int(annotation.get("analyzed_range", {}).get("end_frame", gold[-1][1]))
    tail_start = gold[-1][1] + 1
    tail_frames = max(0, analyzed_end - tail_start + 1)
    tail_pred = clip_segments(full_pred, tail_start, analyzed_end) if tail_frames else []
    tail_active_frames = sum(end - start + 1 for start, end, label in tail_pred if label == "active")
    row: Dict[str, object] = {
        "clip_id": clip_id,
        "human_states": len(gold),
        "model_states": len(pred),
        "state_count_ratio": len(pred) / max(1, len(gold)),
        "all_completed": is_annotated_exchange(annotation),
        "valid": is_usable(annotation),
        "usable": is_usable(annotation),
        "clean": is_clean(annotation),
        "flags": ",".join(quality_labels(annotation)),
        "excluded_tail_frames": tail_frames,
        "post_exchange_active_frames": tail_active_frames,
        "post_exchange_active_coverage": tail_active_frames / tail_frames if tail_frames else 0.0,
        "post_exchange_states": len(tail_pred),
    }
    for tolerance in (3, 5, 10):
        precision, recall, f1, distance = boundary_scores(boundaries(pred), boundaries(gold), tolerance)
        row.update(
            {
                f"boundary_precision_pm{tolerance}": precision,
                f"boundary_recall_pm{tolerance}": recall,
                f"boundary_f1_pm{tolerance}": f1,
                f"mean_boundary_distance_pm{tolerance}": distance,
            }
        )
    accuracy, macro_f1 = activity_scores(pred, gold)
    row["activity_accuracy"] = accuracy
    row["activity_macro_f1"] = macro_f1
    row.update(activity_class_scores(pred, gold))
    return row


def summarize(rows: List[dict], cohort: str) -> dict:
    selected = [row for row in rows if row[cohort]]
    summary: Dict[str, object] = {"cohort": cohort, "clips": len(selected)}
    numeric_keys = [
        "human_states",
        "model_states",
        "state_count_ratio",
        "boundary_precision_pm3",
        "boundary_recall_pm3",
        "boundary_f1_pm3",
        "boundary_precision_pm5",
        "boundary_recall_pm5",
        "boundary_f1_pm5",
        "boundary_precision_pm10",
        "boundary_recall_pm10",
        "boundary_f1_pm10",
        "activity_accuracy",
        "activity_macro_f1",
        "active_precision",
        "active_recall",
        "active_f1",
        "model_active_coverage",
        "human_active_coverage",
        "excluded_tail_frames",
        "post_exchange_active_frames",
        "post_exchange_active_coverage",
        "post_exchange_states",
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in selected]
        summary[key] = sum(values) / len(values) if values else 0.0
    summary["total_human_states"] = sum(int(row["human_states"]) for row in selected)
    summary["total_model_states"] = sum(int(row["model_states"]) for row in selected)
    summary["total_state_count_ratio"] = summary["total_model_states"] / max(1, summary["total_human_states"])
    total_tail_frames = sum(int(row["excluded_tail_frames"]) for row in selected)
    total_tail_active = sum(int(row["post_exchange_active_frames"]) for row in selected)
    summary["total_excluded_tail_frames"] = total_tail_frames
    summary["total_post_exchange_active_frames"] = total_tail_active
    summary["frame_weighted_post_exchange_active_coverage"] = total_tail_active / total_tail_frames if total_tail_frames else 0.0
    return summary


def export_model_annotations(owner_dir: Path, predictions: Dict[str, List[Segment]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("*.json"):
        old_file.unlink()
    for owner_file in sorted(owner_dir.glob("*.json")):
        owner = normalize_annotation(read_json(owner_file), strict=False)
        segments = predictions.get(owner["clip_id"])
        if segments is None:
            continue
        gold_segments = annotation_segments(owner)
        segments = clip_segments(segments, gold_segments[0][0], gold_segments[-1][1])
        model = dict(owner)
        model["annotator_id"] = "deployed_model"
        model["segments"] = [
            {
                "start_frame": start,
                "end_frame": end,
                "start_sec": start / float(owner["fps"]),
                "end_sec": (end + 1) / float(owner["fps"]),
                "activity": activity,
                "uncertain": False,
                "note": "",
            }
            for start, end, activity in segments
        ]
        model["quality_flags"] = {
            "valid": True,
            "pose_issue": False,
            "ambiguous": False,
            "not_exchange": False,
            "needs_review": False,
        }
        model["free_text_notes"] = "Generated by the deployed legacy_auto boundary model."
        write_json(out_dir / owner_file.name, model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-dir", default="paper_experiments/annotations/raw/owner")
    parser.add_argument("--predictions", default="paper_experiments/experiments/boundary_predictions/full_model.json")
    parser.add_argument("--manifest", default="", help="Optional CSV restricting evaluation to listed clip_id values")
    parser.add_argument("--model-annotations-dir", default="paper_experiments/annotations/comparisons/deployed_model")
    parser.add_argument("--summary-out", default="paper_experiments/paper_ready/owner_vs_deployed_model.csv")
    parser.add_argument("--by-clip-out", default="paper_experiments/paper_ready/owner_vs_deployed_model_by_clip.csv")
    parser.add_argument("--viewer-metrics-out", default="paper_experiments/annotations/comparisons/owner_vs_model_metrics.csv")
    args = parser.parse_args()

    owner_dir = Path(args.owner_dir)
    predictions = pred_map(Path(args.predictions))
    allowed_clip_ids = None
    if args.manifest:
        allowed_clip_ids = {row["clip_id"] for row in read_csv(args.manifest)}
    rows = []
    for owner_file in sorted(owner_dir.glob("*.json")):
        annotation = normalize_annotation(read_json(owner_file), strict=False)
        if allowed_clip_ids is not None and annotation["clip_id"] not in allowed_clip_ids:
            continue
        pred = predictions.get(annotation["clip_id"])
        if pred is None:
            print(f"warning: no prediction for {annotation['clip_id']}")
            continue
        rows.append(clip_metrics(annotation["clip_id"], annotation, pred))

    write_csv(args.by_clip_out, sorted(rows, key=lambda row: float(row["boundary_f1_pm5"])))
    write_csv(
        args.summary_out,
        [summarize(rows, "all_completed"), summarize(rows, "valid"), summarize(rows, "clean")],
    )
    write_csv(
        args.viewer_metrics_out,
        [
            {
                "clip_id": row["clip_id"],
                "boundary_f1_pm5": row["boundary_f1_pm5"],
                "activity_frame_agreement": row["activity_accuracy"],
            }
            for row in rows
        ],
    )
    export_model_annotations(owner_dir, predictions, Path(args.model_annotations_dir))
    print(f"Compared {len(rows)} expert annotations with model predictions")
    print(f"Wrote {args.summary_out}")
    print(f"Wrote {args.by_clip_out}")


if __name__ == "__main__":
    main()
