#!/usr/bin/env python3
"""Prepare a varied, non-paper batch of scoring-light-trimmed clips for inspection."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def spread_sample(rows: List[Dict[str, str]], count: int) -> List[Dict[str, str]]:
    if count >= len(rows):
        return rows
    ordered = sorted(rows, key=lambda row: (int(row["states"]), float(row["active_coverage"]), row["clip_id"]))
    indexes = [round(i * (len(ordered) - 1) / max(1, count - 1)) for i in range(count)]
    return [ordered[index] for index in indexes]


def trim_video(source: Path, destination: Path, end_sec: float, force: bool) -> None:
    if destination.exists() and not force and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-t",
        f"{end_sec:.6f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="paper_experiments/experiments/truncated_retraining/manifests/unlabeled_audit.csv",
    )
    parser.add_argument(
        "--audit",
        default="paper_experiments/experiments/truncated_retraining/unlabeled_audit/smooth02_seed18_shared_pp/by_clip.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="paper_experiments/paper_ready/tuned_trimmed_boundary_inspector",
    )
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = {row["clip_id"]: row for row in read_csv(Path(args.manifest))}
    audit_rows = read_csv(Path(args.audit))
    by_outcome: Dict[str, List[Dict[str, str]]] = {}
    for row in audit_rows:
        metadata = manifest.get(row["clip_id"])
        if metadata is None:
            continue
        row = dict(row)
        row.update(metadata)
        by_outcome.setdefault(metadata["outcome_label"], []).append(row)

    outcomes = [outcome for outcome in ("Left", "None", "Right") if outcome in by_outcome]
    base = args.count // len(outcomes)
    remainder = args.count % len(outcomes)
    selected: List[Dict[str, str]] = []
    for index, outcome in enumerate(outcomes):
        selected.extend(spread_sample(by_outcome[outcome], base + (1 if index < remainder else 0)))
    selected.sort(key=lambda row: row["clip_id"])

    out_dir = Path(args.out_dir)
    l2_dir = out_dir / "l2"
    video_dir = out_dir / "videos"
    l2_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    selection = []
    for index, row in enumerate(selected, start=1):
        source_l2 = Path(row["l2_path"])
        source_video = Path(row["video_path"])
        if not source_l2.exists():
            raise FileNotFoundError(source_l2)
        if not source_video.exists():
            raise FileNotFoundError(source_video)
        destination_l2 = l2_dir / source_l2.name
        destination_video = video_dir / source_video.name
        shutil.copy2(source_l2, destination_l2)
        fps = float(row["fps"])
        hit_frame = int(float(row["hit_frame"]))
        end_sec = (hit_frame + 1) / fps
        trim_video(source_video, destination_video, end_sec, args.force)
        selection.append(
            {
                "clip_id": row["clip_id"],
                "outcome_label": row["outcome_label"],
                "hit_frame": hit_frame,
                "fps": fps,
                "trim_end_sec_exclusive": end_sec,
                "audit_states": int(row["states"]),
                "audit_active_coverage": float(row["active_coverage"]),
                "source_video": str(source_video),
                "trimmed_video": str(destination_video),
                "l2_path": str(destination_l2),
            }
        )
        print(f"[{index:02d}/{len(selected):02d}] {row['clip_id']} -> {end_sec:.2f}s")

    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Tuned Trimmed Boundary Inspector\n\n"
        "This batch contains non-paper audit clips only. Videos end immediately after the first "
        "automatic scoring-light frame. The selection is stratified by outcome and spread across "
        "the conservative model's state-count/activity-coverage distribution.\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(selection)} clips in {out_dir}")


if __name__ == "__main__":
    main()
