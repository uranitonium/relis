#!/usr/bin/env python3
"""Validation and normalization for interaction-state annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from common import clip_stem, outcome_from_clip_id, read_json, write_json


QUALITY_FLAGS = ["valid", "pose_issue", "ambiguous", "not_exchange", "needs_review"]
ACTIVITY_LABELS = {"stable", "active"}


def normalize_annotation(payload: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    if payload.get("zones") is not None and not payload.get("segments"):
        raise ValueError("legacy transition-zone annotation; restart with schema version 2")
    clip_id = str(payload.get("clip_id") or clip_stem(str(payload.get("source_video", ""))))
    if not clip_id:
        raise ValueError("missing clip_id")
    fps = float(payload.get("fps", 25.0))
    if fps <= 0:
        raise ValueError(f"{clip_id}: fps must be positive")
    outcome = payload.get("outcome_label") or outcome_from_clip_id(clip_id)
    if outcome not in {"Left", "Right", "None", None}:
        raise ValueError(f"{clip_id}: invalid outcome_label {outcome!r}")
    flags_in = payload.get("quality_flags") or {}
    flags = {name: bool(flags_in.get(name, name == "valid")) for name in QUALITY_FLAGS}
    range_in = payload.get("analyzed_range") or {}
    segments_in = payload.get("segments") or []
    if segments_in:
        fallback_start = min(int(round(float(s.get("start_frame", 0)))) for s in segments_in)
        fallback_end = max(int(round(float(s.get("end_frame", 0)))) for s in segments_in)
    else:
        fallback_start = fallback_end = 0
    range_start = int(round(float(range_in.get("start_frame", fallback_start))))
    range_end = int(round(float(range_in.get("end_frame", fallback_end))))
    if range_end < range_start:
        raise ValueError(f"{clip_id}: invalid analyzed range {range_start}-{range_end}")
    evaluation_in = payload.get("evaluation_range") or {}
    evaluation_start = int(round(float(evaluation_in.get("start_frame", range_start))))
    evaluation_end = int(round(float(evaluation_in.get("end_frame", range_end))))
    evaluation_start = max(range_start, min(range_end, evaluation_start))
    evaluation_end = max(evaluation_start, min(range_end, evaluation_end))
    exchange_end_confirmed = bool(payload.get("exchange_end_confirmed", False))
    exchange_end_source = str(payload.get("exchange_end_source", ""))

    segments: List[Dict[str, Any]] = []
    for raw in segments_in:
        start = int(round(float(raw.get("start_frame", evaluation_start))))
        end = int(round(float(raw.get("end_frame", start))))
        activity = str(raw.get("activity", "stable")).lower()
        if activity not in ACTIVITY_LABELS:
            raise ValueError(f"{clip_id}: invalid activity label {activity!r}")
        if end < start:
            start, end = end, start
        segments.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_sec": float(start / fps),
                "end_sec": float((end + 1) / fps),
                "activity": activity,
                "uncertain": bool(raw.get("uncertain", False)),
                "note": str(raw.get("note", "")),
            }
        )
    segments.sort(key=lambda row: (row["start_frame"], row["end_frame"]))

    complete = bool(payload.get("annotation_complete", False))
    excluded = bool(flags["not_exchange"] or not flags["valid"])
    if complete and not excluded:
        if strict and not exchange_end_confirmed:
            raise ValueError(f"{clip_id}: completed annotation has no confirmed exchange end")
        if not segments:
            raise ValueError(f"{clip_id}: completed annotation has no segments")
        if segments[0]["start_frame"] != evaluation_start or segments[-1]["end_frame"] != evaluation_end:
            raise ValueError(f"{clip_id}: completed segments must cover evaluation range {evaluation_start}-{evaluation_end}")
        for left, right in zip(segments, segments[1:]):
            if right["start_frame"] != left["end_frame"] + 1:
                raise ValueError(f"{clip_id}: segment gap/overlap at {left['end_frame']}->{right['start_frame']}")
    elif strict and not complete:
        raise ValueError(f"{clip_id}: annotation is not marked complete")

    return {
        "schema_version": 2,
        "annotation_task": "interaction_state_partition",
        "coordinate_system": "source_video_frame_inclusive",
        "clip_id": clip_id,
        "weapon": str(payload.get("weapon", "foil")),
        "annotator_id": str(payload.get("annotator_id", "")),
        "source_video": str(payload.get("source_video", "")),
        "fps": fps,
        "outcome_label": outcome,
        "analyzed_range": {"start_frame": range_start, "end_frame": range_end},
        "evaluation_range": {"start_frame": evaluation_start, "end_frame": evaluation_end},
        "exchange_end_confirmed": exchange_end_confirmed,
        "exchange_end_source": exchange_end_source,
        "segments": segments,
        "annotation_complete": complete,
        "quality_flags": flags,
        "free_text_notes": str(payload.get("free_text_notes", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate interaction-state annotation JSON files.")
    parser.add_argument("--input", required=True, help="Annotation JSON file or directory")
    parser.add_argument("--out", default="", help="Optional output JSON file")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    path = Path(args.input)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    normalized = [normalize_annotation(read_json(file), strict=args.strict) for file in files]
    payload: Any = normalized if path.is_dir() else normalized[0]
    if args.out:
        write_json(args.out, payload)
    else:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
