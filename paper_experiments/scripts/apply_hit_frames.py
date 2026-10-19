#!/usr/bin/env python3
"""Apply automatic scoring-light endpoints to manifests and annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import read_csv, read_json, write_csv, write_json


def clip_segments(annotation: dict, end_frame: int) -> None:
    segments = annotation.get("segments") or []
    segments = [segment for segment in segments if int(segment["start_frame"]) <= end_frame]
    if segments:
        segments[-1]["end_frame"] = end_frame
        fps = float(annotation.get("fps", 25.0))
        segments[-1]["end_sec"] = (end_frame + 1) / fps
    annotation["segments"] = segments


def update_annotation(path: Path, hit_frame: int) -> None:
    annotation = read_json(path)
    analyzed = annotation.get("analyzed_range") or {}
    start = int(analyzed.get("start_frame", 0))
    end = min(int(analyzed.get("end_frame", hit_frame)), hit_frame)
    annotation["evaluation_range"] = {"start_frame": start, "end_frame": end}
    annotation["exchange_end_confirmed"] = True
    annotation["exchange_end_source"] = "automatic_scoring_light"
    clip_segments(annotation, end)
    flags = annotation.get("quality_flags") or {}
    if flags.get("valid", True) and not flags.get("not_exchange"):
        annotation["annotation_complete"] = True
    write_json(path, annotation)


def update_package_manifest(path: Path, hit_frames: dict[str, int]) -> None:
    manifest = read_json(path)
    for clip in manifest.get("clips", []):
        clip_id = clip["clip_id"]
        if clip_id in hit_frames:
            clip["hit_frame"] = hit_frames[clip_id]
            clip["evaluation_end_source"] = "automatic_scoring_light"
    write_json(path, manifest)


def update_csv_manifest(path: Path, hit_frames: dict[str, int]) -> None:
    rows = read_csv(path)
    if not rows:
        return
    for row in rows:
        if row["clip_id"] in hit_frames:
            row["hit_frame"] = hit_frames[row["clip_id"]]
            row["evaluation_end_source"] = "automatic_scoring_light"
    fields = list(rows[0])
    for field in ("hit_frame", "evaluation_end_source"):
        if field not in fields:
            fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hit-frames", default="paper_experiments/manifests/foil_hit_frames.csv")
    parser.add_argument("--manifests-root", default="paper_experiments/manifests")
    parser.add_argument("--packages-root", default="paper_experiments/annotation_packages")
    args = parser.parse_args()

    hit_frames = {
        row["clip_id"]: int(float(row["hit_frame"]))
        for row in read_csv(args.hit_frames)
        if row.get("status") == "hit_found" and row.get("hit_frame") != ""
    }
    for path in sorted(Path(args.manifests_root).glob("foil_*.csv")):
        if path.name == Path(args.hit_frames).name:
            continue
        update_csv_manifest(path, hit_frames)
    for manifest_path in sorted(Path(args.packages_root).glob("foil_*/manifest.json")):
        update_package_manifest(manifest_path, hit_frames)
        annotation_dir = manifest_path.parent / "annotations"
        for annotation_path in sorted(annotation_dir.glob("*.json")):
            clip_id = annotation_path.stem
            if clip_id in hit_frames:
                update_annotation(annotation_path, hit_frames[clip_id])
    print(f"Applied automatic hit frames for {len(hit_frames)} clips")


if __name__ == "__main__":
    main()
