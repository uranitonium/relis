#!/usr/bin/env python3
"""Extract two-skeleton pose features for a learned ST-GCN boundary baseline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from common import read_csv, write_csv


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = {
    "train": REPO / "paper_experiments/experiments/truncated_retraining/manifests/train.csv",
    "dev": REPO / "paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv",
    "final": REPO / "paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv",
}
TORSO_JOINTS = np.asarray([0, 1, 6, 7], dtype=np.int64)
TORSO_LINKS = ((0, 6), (1, 7))
FEATURE_DIM = 2 * 12 * 2


def parse_split(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("split must be NAME=CSV")
    name, path = value.split("=", 1)
    return name, Path(path)


def load_processed(path: Path) -> Dict[int, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, np.ndarray] = {}
    for frame in data.get("frame_data", []):
        try:
            frame_number = int(frame["frame_number"])
            keypoints = np.asarray(frame["keypoints"], dtype=np.float32)
        except Exception:
            continue
        if keypoints.shape == (12, 2) and np.isfinite(keypoints).all():
            out[frame_number] = keypoints
    return out


def torso_scale(raw: np.ndarray, valid: np.ndarray) -> float:
    values: List[float] = []
    for t in range(raw.shape[0]):
        if not valid[t]:
            continue
        for fencer in range(2):
            for a, b in TORSO_LINKS:
                pa = raw[t, fencer, a]
                pb = raw[t, fencer, b]
                if np.isfinite(pa).all() and np.isfinite(pb).all():
                    d = float(np.linalg.norm(pa - pb))
                    if d > 1.0:
                        values.append(d)
    if not values:
        return 1.0
    return max(float(np.median(values)), 1.0)


def clip_center(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    points = []
    for t in range(raw.shape[0]):
        if valid[t]:
            points.append(raw[t, :, TORSO_JOINTS, :].reshape(-1, 2))
    if not points:
        return np.zeros((2,), dtype=np.float32)
    return np.concatenate(points, axis=0).mean(axis=0).astype(np.float32)


def extract_clip(row: Dict[str, str], processed_root: Path, out_path: Path, force: bool = False) -> None:
    start = int(float(row.get("frame_start", 0)))
    end = int(float(row.get("frame_end", row.get("hit_frame", start))))
    if end < start:
        start, end = end, start
    frames = np.arange(start, end + 1, dtype=np.int32)
    if not force and out_path.exists():
        try:
            data = np.load(out_path)
            if int(data["x"].shape[0]) == len(frames) and int(data["x"].shape[1]) == FEATURE_DIM:
                return
        except Exception:
            pass

    clip_id = row["clip_id"]
    left_path = processed_root / "left" / f"{clip_id}_pose_processed.json"
    right_path = processed_root / "right" / f"{clip_id}_pose_processed.json"
    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError(f"Missing processed pose for {clip_id}: {left_path} / {right_path}")
    left = load_processed(left_path)
    right = load_processed(right_path)

    raw = np.zeros((len(frames), 2, 12, 2), dtype=np.float32)
    valid = np.zeros((len(frames),), dtype=np.bool_)
    for i, frame in enumerate(frames.tolist()):
        lkp = left.get(frame)
        rkp = right.get(frame)
        if lkp is None or rkp is None:
            continue
        raw[i, 0] = lkp
        raw[i, 1] = rkp
        valid[i] = True

    center = clip_center(raw, valid)
    scale = torso_scale(raw, valid)
    x = (raw - center.reshape(1, 1, 1, 2)) / scale
    x[~valid] = 0.0
    x = x.reshape(len(frames), FEATURE_DIM).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=x,
        mask=valid,
        frames=frames,
        fps=np.asarray(float(row.get("fps", 25.0)), dtype=np.float32),
        center=center.astype(np.float32),
        scale=np.asarray(scale, dtype=np.float32),
    )


def link_train_features(train_rows: List[Dict[str, str]], all_dir: Path, train_dir: Path) -> None:
    train_dir.mkdir(parents=True, exist_ok=True)
    for row in train_rows:
        src = all_dir / f"{row['clip_id']}_pose_stgcn.npz"
        dst = train_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def write_manifests(splits: Dict[str, List[Dict[str, str]]], all_dir: Path, manifest_dir: Path) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        output = []
        for row in rows:
            item = dict(row)
            item["l2_path"] = str(all_dir / f"{row['clip_id']}_pose_stgcn.npz")
            output.append(item)
        write_csv(manifest_dir / f"{name}_pose_stgcn.csv", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", default="processed_joints_v3")
    parser.add_argument("--out-root", default="paper_experiments/experiments/pose_stgcn_baseline")
    parser.add_argument("--split", action="append", type=parse_split, default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    processed_root = (REPO / args.processed_root).resolve()
    out_root = (REPO / args.out_root).resolve()
    all_dir = out_root / "features" / "all"
    train_dir = out_root / "features" / "train"
    manifest_dir = out_root / "manifests"
    split_specs = dict(args.split) if args.split else DEFAULT_SPLITS
    splits = {name: read_csv(path) for name, path in split_specs.items()}

    rows_by_clip: Dict[str, Dict[str, str]] = {}
    for rows in splits.values():
        for row in rows:
            rows_by_clip.setdefault(row["clip_id"], row)
    rows = list(rows_by_clip.values())
    if args.limit:
        rows = rows[: args.limit]

    print(f"[pose_stgcn] clips={len(rows)} out={all_dir}", flush=True)
    for idx, row in enumerate(rows, 1):
        extract_clip(row, processed_root, all_dir / f"{row['clip_id']}_pose_stgcn.npz", force=args.force)
        if idx % 100 == 0 or idx == len(rows):
            print(f"[pose_stgcn] {idx}/{len(rows)}", flush=True)

    if "train" in splits and not args.limit:
        link_train_features(splits["train"], all_dir, train_dir)
    if not args.limit:
        write_manifests(splits, all_dir, manifest_dir)
    print(f"[pose_stgcn] wrote manifests to {manifest_dir}", flush=True)


if __name__ == "__main__":
    main()
