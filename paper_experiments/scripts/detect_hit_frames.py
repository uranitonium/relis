#!/usr/bin/env python3
"""Detect the first scoring-light frame in paper clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import read_csv, write_csv, write_json


REGIONS = {
    "red": (slice(656, 662), slice(260, 440)),
    "green": (slice(656, 662), slice(831, 988)),
    "off_left": (slice(596, 600), slice(382, 494)),
    "off_right": (slice(596, 600), slice(700, 876)),
}
REFERENCE_FILES = {
    "red": "0_red_light_frame_001300.png",
    "green": "0_green_light_frame_001300.png",
    "off_left": "6_off_target_left_frame_002300.png",
    "off_right": "13_off_target_right_frame_012600.png",
}


def load_references(root: Path) -> dict[str, np.ndarray]:
    references = {name: cv2.imread(str(root / filename)) for name, filename in REFERENCE_FILES.items()}
    missing = [name for name, image in references.items() if image is None]
    if missing:
        raise FileNotFoundError(f"Missing hit-detector reference images: {', '.join(missing)}")
    return references


def detect(video_path: Path, references: dict[str, np.ndarray], threshold: int, min_frame: int) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"status": "open_failed", "hit_frame": "", "hit_type": ""}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index >= min_frame:
            flags = []
            for name, region in REGIONS.items():
                sample = frame[region]
                if sample.shape != references[name].shape:
                    capture.release()
                    return {"status": "region_mismatch", "hit_frame": "", "hit_type": ""}
                difference = int(np.sum(np.abs(sample.astype(np.int32) - references[name].astype(np.int32))))
                if difference <= threshold:
                    flags.append(name)
            if flags:
                capture.release()
                return {
                    "status": "hit_found",
                    "hit_frame": frame_index,
                    "hit_time_seconds": frame_index / fps if fps > 0 else "",
                    "hit_type": "+".join(flags),
                    "fps": fps,
                    "frame_count": frame_count,
                }
        frame_index += 1
    capture.release()
    return {
        "status": "no_hit_found",
        "hit_frame": "",
        "hit_time_seconds": "",
        "hit_type": "",
        "fps": fps,
        "frame_count": frame_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="paper_experiments/manifests/foil_main.csv")
    parser.add_argument("--reference-root", default="paper_experiments/hit_detection/reference_regions")
    parser.add_argument("--threshold", type=int, default=30000)
    parser.add_argument("--min-frame", type=int, default=60, help="Ignore stale lights near the start of the clip")
    parser.add_argument("--out", default="paper_experiments/manifests/foil_hit_frames.csv")
    args = parser.parse_args()

    references = load_references(Path(args.reference_root))
    results = []
    for row in read_csv(args.manifest):
        result = detect(Path(row["video_path"]), references, args.threshold, args.min_frame)
        results.append({"clip_id": row["clip_id"], "video_path": row["video_path"], **result})
        print(f"{row['clip_id']}: {result['status']} frame={result['hit_frame']}")
    write_csv(args.out, results)
    write_json(Path(args.out).with_suffix(".json"), {"threshold": args.threshold, "min_frame": args.min_frame, "clips": results})
    failures = [row for row in results if row["status"] != "hit_found"]
    print(f"Detected hits for {len(results) - len(failures)}/{len(results)} clips")
    if failures:
        raise SystemExit(f"Hit detection failed for: {', '.join(row['clip_id'] for row in failures)}")


if __name__ == "__main__":
    main()
