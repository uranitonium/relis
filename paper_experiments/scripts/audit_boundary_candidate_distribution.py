#!/usr/bin/env python3
"""Audit state-count and activity-coverage distributions without human labels."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_csv, read_json, write_csv, write_json  # noqa: E402
from generate_boundary_predictions import partition_states  # noqa: E402
from search_boundary_postprocess import (  # noqa: E402
    cache_scores,
    deployed_postprocess,
    load_cache,
    score_from_entry,
    zones_from_score,
)


def quantile(values, fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round(fraction * (len(ordered) - 1))
    return float(ordered[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-json", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifest", default="paper_experiments/experiments/truncated_retraining/manifests/unlabeled_audit.csv")
    parser.add_argument("--deployed-postprocess", action="store_true")
    parser.add_argument("--postprocess-json", default="", help="Use postprocessing from another search result")
    parser.add_argument("--out-root", default="paper_experiments/experiments/truncated_retraining/unlabeled_audit")
    args = parser.parse_args()

    payload = read_json(args.search_json)
    rows = read_csv(args.manifest)
    clip_ids = [row["clip_id"] for row in rows]
    extension_mode = payload.get("extension_mode", "none")
    extension_length = int(payload.get("extension_length", 127))
    zero_indices = [int(index) for index in payload.get("zero_feature_indices", [])]
    out_dir = Path(args.out_root) / args.name
    caches = []
    sources = []
    for index, model in enumerate(payload["models"]):
        cache_path = cache_scores(
            f"model{index}",
            Path(model["config"]),
            Path(model["checkpoint"]),
            rows,
            out_dir / "score_cache",
            extension_mode=extension_mode,
            extension_length=extension_length,
            zero_feature_indices=zero_indices,
        )
        caches.append(load_cache(cache_path, clip_ids))
        sources.append(model.get("score_source", "b"))

    weights = [float(weight) for weight in payload["best"]["weights"]]
    if args.deployed_postprocess:
        pp = deployed_postprocess()
        postprocess_name = "deployed"
    elif args.postprocess_json:
        pp = read_json(args.postprocess_json)["best"]["postprocess"]
        postprocess_name = args.postprocess_json
    else:
        pp = payload["best"]["postprocess"]
        postprocess_name = "searched"
    output = []
    for clip_id in clip_ids:
        entries = [cache[clip_id] for cache in caches]
        score = sum(
            weight * score_from_entry(entry, source)
            for weight, entry, source in zip(weights, entries, sources)
        )
        mask = entries[0]["mask"]
        frames = entries[0]["frames"]
        zones = zones_from_score(score, mask, pp)
        states = partition_states({"frames": frames.tolist(), "l2_mask": mask.tolist()}, zones, len(score))
        valid_frames = max(1, int(mask.sum()))
        active_frames = sum(end - start + 1 for start, end in zones)
        output.append(
            {
                "clip_id": clip_id,
                "valid_frames": valid_frames,
                "states": len(states),
                "active_states": len(zones),
                "active_frames": active_frames,
                "active_coverage": active_frames / valid_frames,
            }
        )

    coverages = [float(row["active_coverage"]) for row in output]
    states = [int(row["states"]) for row in output]
    summary = {
        "name": args.name,
        "clips": len(output),
        "postprocess": postprocess_name,
        "active_coverage_mean": statistics.mean(coverages),
        "active_coverage_median": statistics.median(coverages),
        "active_coverage_p10": quantile(coverages, 0.10),
        "active_coverage_p90": quantile(coverages, 0.90),
        "states_mean": statistics.mean(states),
        "states_median": statistics.median(states),
        "states_p10": quantile(states, 0.10),
        "states_p90": quantile(states, 0.90),
        "clips_without_active_state": sum(row["active_states"] == 0 for row in output),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "by_clip.csv", output)
    write_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
