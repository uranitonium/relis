#!/usr/bin/env python3
"""Measure post-hit activity emitted by a boundary model on untrimmed clips."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from common import read_csv, read_json, write_csv
from evaluate_boundaries import pred_map


def active_frames_in_range(segments: List[tuple[int, int, str]], start: int, end: int) -> int:
    if end < start:
        return 0
    total = 0
    for seg_start, seg_end, activity in segments:
        if activity != "active":
            continue
        overlap = max(0, min(seg_end, end) - max(seg_start, start) + 1)
        total += overlap
    return total


def count_states_in_range(segments: List[tuple[int, int, str]], start: int, end: int) -> int:
    if end < start:
        return 0
    return sum(1 for seg_start, seg_end, _ in segments if min(seg_end, end) >= max(seg_start, start))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    total_tail = sum(int(row["tail_frames"]) for row in rows)
    total_active = sum(int(row["post_hit_active_frames"]) for row in rows)
    return {
        "clips": n,
        "mean_tail_frames": sum(float(row["tail_frames"]) for row in rows) / n if n else 0.0,
        "mean_post_hit_active_frames": sum(float(row["post_hit_active_frames"]) for row in rows) / n if n else 0.0,
        "mean_post_hit_active_coverage": sum(float(row["post_hit_active_coverage"]) for row in rows) / n if n else 0.0,
        "frame_weighted_post_hit_active_coverage": total_active / total_tail if total_tail else 0.0,
        "clips_with_post_hit_activity": sum(1 for row in rows if int(row["post_hit_active_frames"]) > 0),
        "mean_post_hit_states": sum(float(row["post_hit_states"]) for row in rows) / n if n else 0.0,
        "total_tail_frames": total_tail,
        "total_post_hit_active_frames": total_active,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="paper_experiments/experiments/untrimmed_recovery/final_untrimmed.csv")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--summary-out", default="paper_experiments/experiments/untrimmed_recovery/summary.csv")
    parser.add_argument("--by-clip-out", default="paper_experiments/experiments/untrimmed_recovery/by_clip.csv")
    args = parser.parse_args()

    predictions = pred_map(Path(args.predictions))
    rows = []
    for row in read_csv(args.manifest):
        clip_id = row["clip_id"]
        pred = predictions[clip_id]
        hit = int(float(row["hit_frame"]))
        end = int(float(row["frame_end"]))
        tail_start = hit + 1
        tail_frames = max(0, end - tail_start + 1)
        active = active_frames_in_range(pred, tail_start, end)
        rows.append(
            {
                "clip_id": clip_id,
                "hit_frame": hit,
                "frame_end": end,
                "tail_frames": tail_frames,
                "post_hit_active_frames": active,
                "post_hit_active_coverage": active / tail_frames if tail_frames else 0.0,
                "post_hit_states": count_states_in_range(pred, tail_start, end),
            }
        )

    write_csv(args.by_clip_out, rows)
    write_csv(args.summary_out, [summarize(rows)])
    print(f"Wrote {args.summary_out}")
    print(f"Wrote {args.by_clip_out}")


if __name__ == "__main__":
    main()
