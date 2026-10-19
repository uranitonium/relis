#!/usr/bin/env python3
"""Export predictions and per-clip metrics for a postprocess-search candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_csv, read_json, write_csv, write_json  # noqa: E402
from compare_annotations_to_model import clip_metrics, summarize  # noqa: E402
from generate_boundary_predictions import partition_states  # noqa: E402
from search_boundary_postprocess import (  # noqa: E402
    cache_scores,
    load_annotations,
    load_cache,
    score_from_entry,
    zones_from_score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-json", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--postprocess-json", default="", help="Use postprocessing from another search result")
    parser.add_argument("--owner-dir", default="paper_experiments/annotations/raw/owner")
    parser.add_argument("--out-root", default="paper_experiments/experiments/truncated_retraining/results")
    args = parser.parse_args()

    search_path = Path(args.search_json)
    payload = read_json(search_path)
    manifest_path = Path(payload["manifest"])
    annotations = load_annotations(Path(args.owner_dir))
    rows = [row for row in read_csv(manifest_path) if row["clip_id"] in annotations]
    row_by_clip = {row["clip_id"]: row for row in rows}
    extension_mode = payload.get("extension_mode", "none")
    extension_length = int(payload.get("extension_length", 127))
    zero_indices = [int(index) for index in payload.get("zero_feature_indices", [])]
    out_dir = Path(args.out_root) / args.name
    cache_dir = out_dir / "score_cache"

    caches = []
    sources = []
    for index, model in enumerate(payload["models"]):
        cache_path = cache_scores(
            f"model{index}",
            Path(model["config"]),
            Path(model["checkpoint"]),
            rows,
            cache_dir,
            extension_mode=extension_mode,
            extension_length=extension_length,
            zero_feature_indices=zero_indices,
        )
        caches.append(load_cache(cache_path, annotations.keys()))
        sources.append(model.get("score_source", "b"))

    weights = [float(weight) for weight in payload["best"]["weights"]]
    pp = (
        read_json(args.postprocess_json)["best"]["postprocess"]
        if args.postprocess_json
        else payload["best"]["postprocess"]
    )
    predictions = []
    metric_rows = []
    for clip_id, annotation in annotations.items():
        entries = [cache[clip_id] for cache in caches]
        score = sum(
            weight * score_from_entry(entry, source)
            for weight, entry, source in zip(weights, entries, sources)
        )
        mask = entries[0]["mask"]
        frames = entries[0]["frames"]
        zones = zones_from_score(score, mask, pp)
        data = {
            "frames": frames.tolist(),
            "l2_mask": mask.tolist(),
        }
        states = partition_states(data, zones, len(score))
        predictions.append(
            {
                "clip_id": clip_id,
                "method": args.name,
                "coordinate_system": "source_video_frame_inclusive",
                "transition_zones_index": [[int(start), int(end)] for start, end in zones],
                "state_segments": states,
                "active_segments": [
                    [state["start_frame"], state["end_frame"]]
                    for state in states
                    if state["activity"] == "active"
                ],
                "l2_path": row_by_clip[clip_id]["l2_path"],
            }
        )
        pred = [
            (int(state["start_frame"]), int(state["end_frame"]), str(state["activity"]))
            for state in states
        ]
        metric_rows.append(clip_metrics(clip_id, annotation, pred))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "predictions.json",
        {
            "method": args.name,
            "search_json": str(search_path),
            "postprocess_json": args.postprocess_json,
            "manifest": str(manifest_path),
            "development_only": True,
            "schema_version": 2,
            "prediction_task": "interaction_state_partition",
            "coordinate_system": "source_video_frame_inclusive",
            "predictions": predictions,
        },
    )
    write_csv(out_dir / "by_clip.csv", sorted(metric_rows, key=lambda row: float(row["boundary_f1_pm5"])))
    summary = summarize(metric_rows, "usable")
    summary["objective"] = (
        float(summary["activity_macro_f1"])
        + float(summary["active_f1"])
        + float(summary["boundary_f1_pm5"])
        + float(summary["boundary_f1_pm10"])
    ) / 4.0
    write_json(out_dir / "summary.json", summary)
    print(f"Wrote {out_dir} predictions={len(predictions)} usable={summary['clips']}")


if __name__ == "__main__":
    main()
