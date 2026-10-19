#!/usr/bin/env python3
"""Build stratified clip manifests for annotation and experiments."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from common import clip_stem, load_l2_meta, outcome_from_clip_id, stratified_sample, write_csv, write_json


REPO = Path(__file__).resolve().parents[2]
V2_DATA_ROOT = Path(os.getenv("FENCING_V2_DATA_ROOT", str(REPO.parent / "V2"))).resolve()


def find_video(video_root: Path, stem: str) -> str:
    candidates = [
        video_root / f"{stem}_pose_h264.mp4",
        video_root / f"{stem}_pose.mp4",
        video_root / f"{stem}.mp4",
        video_root / f"{stem}_com_overlay_h264.mp4",
        video_root / f"{stem}_com_overlay.mp4",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    hits = sorted(video_root.glob(f"{stem}*.mp4")) if video_root.exists() else []
    return str(hits[0]) if hits else ""


def scan_rows(l2_dir: Path, video_root: Path, weapon: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for l2_path in sorted(l2_dir.glob("*_l2.json")):
        stem = clip_stem(l2_path)
        outcome = outcome_from_clip_id(stem)
        if outcome is None:
            continue
        video_path = find_video(video_root, stem)
        if not video_path:
            continue
        meta = load_l2_meta(l2_path)
        rows.append(
            {
                "clip_id": stem,
                "weapon": weapon,
                "outcome_label": outcome,
                "prefix": stem[0].upper(),
                "video_path": video_path,
                "l2_path": str(l2_path),
                "fps": meta["fps"],
                "frame_start": meta["frame_start"],
                "frame_end": meta["frame_end"],
                "num_frames": meta["num_frames"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper experiment manifests.")
    parser.add_argument("--l2-dir", default="paper_experiments/experiments/truncated_retraining/l2_all")
    parser.add_argument("--video-root", default=str(V2_DATA_ROOT / "training_highfps_h264"))
    parser.add_argument("--out-dir", default="paper_experiments/manifests")
    parser.add_argument("--weapon", default="foil")
    parser.add_argument("--main-size", type=int, default=150)
    parser.add_argument("--double-size", type=int, default=40)
    parser.add_argument("--calibration-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20270726)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = scan_rows(Path(args.l2_dir), Path(args.video_root), args.weapon)
    if not rows:
        raise SystemExit("No usable clips found. Check --l2-dir and --video-root.")

    main_rows = stratified_sample(rows, min(args.main_size, len(rows)), "outcome_label", args.seed)
    rng = random.Random(args.seed + 1)
    rng.shuffle(main_rows)
    calib = main_rows[: min(args.calibration_size, len(main_rows))]
    double = main_rows[: min(args.double_size, len(main_rows))]
    remaining = main_rows[min(args.calibration_size, len(main_rows)) :]

    all_fields = [
        "clip_id",
        "weapon",
        "outcome_label",
        "prefix",
        "video_path",
        "l2_path",
        "fps",
        "frame_start",
        "frame_end",
        "num_frames",
    ]
    write_csv(out_dir / "foil_all_available.csv", rows, all_fields)
    write_csv(out_dir / "foil_main.csv", main_rows, all_fields)
    write_csv(out_dir / "foil_calibration.csv", calib, all_fields)
    write_csv(out_dir / "foil_double_label.csv", double, all_fields)
    write_csv(out_dir / "foil_collaborator_main.csv", remaining, all_fields)
    for part_index, start in enumerate(range(0, len(remaining), 52), start=1):
        write_csv(
            out_dir / f"foil_collaborator_main_part_{part_index:02d}.csv",
            remaining[start : start + 52],
            all_fields,
        )
    write_json(
        out_dir / "manifest_summary.json",
        {
            "source_l2_dir": str(Path(args.l2_dir)),
            "source_video_root": str(Path(args.video_root)),
            "weapon": args.weapon,
            "seed": args.seed,
            "available": len(rows),
            "main": len(main_rows),
            "calibration": len(calib),
            "double_label": len(double),
            "collaborator_main": len(remaining),
            "outcome_counts": {
                label: sum(1 for row in rows if row["outcome_label"] == label)
                for label in ["Left", "Right", "None"]
            },
        },
    )
    print(f"Wrote manifests to {out_dir}")
    print(f"available={len(rows)} main={len(main_rows)} calibration={len(calib)} double={len(double)}")


if __name__ == "__main__":
    main()
