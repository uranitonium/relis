#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from camera_motion import estimate_camera_motion, read_json, write_json


def estimate_one(task):
    clip_id, video_path = task
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass
    return clip_id, estimate_camera_motion(Path(video_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate background camera motion for selected clips.")
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel clip workers")
    args = parser.parse_args()

    selected = read_json(Path(args.selection_json)).get("selected", [])
    clip_ids = [str(row["clip_id"]) for row in selected if isinstance(row, dict) and row.get("clip_id")]
    out_path = Path(args.out_json)
    cache = read_json(out_path) if out_path.exists() else {}
    pending = []
    for clip_id in clip_ids:
        motions = cache.get(clip_id, {}).get("motions") if clip_id in cache else None
        if motions and all("affine" in row for row in motions):
            continue
        pending.append((clip_id, str(Path(args.clips_dir) / f"{clip_id}.mp4")))
    jobs = max(1, min(int(args.jobs), os.cpu_count() or 1))
    if jobs == 1:
        results = map(estimate_one, pending)
        for index, (clip_id, result) in enumerate(results, start=1):
            cache[clip_id] = result
            write_json(out_path, cache)
            print(f"[{index}/{len(pending)}] {clip_id}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(estimate_one, task) for task in pending]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                clip_id, result = future.result()
                cache[clip_id] = result
                write_json(out_path, cache)
                print(f"[{index}/{len(pending)}] {clip_id}", flush=True)
    print(json.dumps({"clips": len(clip_ids), "estimated": len(cache)}, indent=2))


if __name__ == "__main__":
    main()
