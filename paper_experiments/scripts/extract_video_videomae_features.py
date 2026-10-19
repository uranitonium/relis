#!/usr/bin/env python3
"""Extract VideoMAE frame-centered video embeddings from no-overlay clips."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

from common import read_csv, write_csv


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
DEFAULT_SPLITS = {
    "train": REPO / "paper_experiments/experiments/truncated_retraining/manifests/train.csv",
    "dev": REPO / "paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv",
    "final": REPO / "paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv",
}


def parse_split(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("split must be NAME=CSV")
    name, path = value.split("=", 1)
    return name, Path(path)


def load_video(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {path}")
    return frames


def valid_existing(path: Path, expected_len: int, expected_dim: int = 768) -> bool:
    if not path.exists():
        return False
    try:
        data = np.load(path)
        return int(data["x"].shape[0]) == expected_len and int(data["x"].shape[1]) == expected_dim
    except Exception:
        return False


def window_frames(frames: List[np.ndarray], center: int, window: int) -> List[np.ndarray]:
    half = window // 2
    result = []
    for offset in range(window):
        index = int(center) + offset - half
        index = max(0, min(index, len(frames) - 1))
        result.append(frames[index])
    return result


def encode_batch(
    videos: List[List[np.ndarray]],
    processor: VideoMAEImageProcessor,
    model: VideoMAEModel,
    device: str,
) -> np.ndarray:
    inputs = processor(videos, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().cpu().numpy().astype(np.float32)


def extract_clip(
    row: Dict[str, str],
    video_root: Path,
    out_path: Path,
    processor: VideoMAEImageProcessor,
    model: VideoMAEModel,
    device: str,
    batch_size: int,
    window: int,
    feature_stride: int,
    force: bool = False,
) -> None:
    start = int(float(row.get("frame_start", 0)))
    end = int(float(row.get("frame_end", row.get("hit_frame", start))))
    if end < start:
        start, end = end, start
    centers = np.arange(start, end + 1, dtype=np.int32)
    if not force and valid_existing(out_path, len(centers)):
        return

    frames = load_video(video_root / f"{row['clip_id']}.mp4")
    sample_centers = centers[:: max(1, feature_stride)]
    if len(sample_centers) == 0 or sample_centers[-1] != centers[-1]:
        sample_centers = np.concatenate([sample_centers, centers[-1:]])

    outputs = []
    pending: List[List[np.ndarray]] = []
    for center in sample_centers:
        pending.append(window_frames(frames, int(center), window))
        if len(pending) >= batch_size:
            outputs.append(encode_batch(pending, processor, model, device))
            pending = []
    if pending:
        outputs.append(encode_batch(pending, processor, model, device))

    sampled = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, model.config.hidden_size), dtype=np.float32)
    if feature_stride > 1 and len(sampled):
        nearest = []
        sample_centers_f = sample_centers.astype(np.float32)
        for center in centers:
            nearest.append(int(np.argmin(np.abs(sample_centers_f - float(center)))))
        x = sampled[np.asarray(nearest, dtype=np.int32)]
    else:
        x = sampled

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=x,
        mask=np.ones((len(centers),), dtype=np.bool_),
        frames=centers,
        fps=np.asarray(float(row.get("fps", 25.0)), dtype=np.float32),
    )


def link_train_features(train_rows: List[Dict[str, str]], all_dir: Path, train_dir: Path) -> None:
    train_dir.mkdir(parents=True, exist_ok=True)
    for row in train_rows:
        src = all_dir / f"{row['clip_id']}_videomae.npz"
        dst = train_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            import shutil

            shutil.copy2(src, dst)


def write_manifests(splits: Dict[str, List[Dict[str, str]]], all_dir: Path, manifest_dir: Path, video_root: Path) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        output = []
        for row in rows:
            item = dict(row)
            item["video_path"] = str(video_root / f"{row['clip_id']}.mp4")
            item["l2_path"] = str(all_dir / f"{row['clip_id']}_videomae.npz")
            output.append(item)
        write_csv(manifest_dir / f"{name}_videomae.csv", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--video-root", default="training_no_overlay")
    parser.add_argument("--out-root", default="paper_experiments/experiments/video_videomae_baseline")
    parser.add_argument("--split", action="append", type=parse_split, default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--feature-stride", type=int, default=8, help="Run VideoMAE every N evaluated frames and repeat nearest features")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    video_root = (REPO / args.video_root).resolve()
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = VideoMAEImageProcessor.from_pretrained(args.model_name)
    model = VideoMAEModel.from_pretrained(args.model_name).eval().to(device)
    print(f"[videomae] device={device} clips={len(rows)} out={all_dir}", flush=True)
    for idx, row in enumerate(rows, 1):
        extract_clip(
            row,
            video_root,
            all_dir / f"{row['clip_id']}_videomae.npz",
            processor,
            model,
            device,
            args.batch_size,
            args.window,
            args.feature_stride,
            force=args.force,
        )
        if idx % 10 == 0 or idx == len(rows):
            print(f"[videomae] {idx}/{len(rows)}", flush=True)

    if "train" in splits and not args.limit:
        link_train_features(splits["train"], all_dir, train_dir)
    if not args.limit:
        write_manifests(splits, all_dir, manifest_dir, video_root)
    print(f"[videomae] wrote manifests to {manifest_dir}", flush=True)


if __name__ == "__main__":
    main()
