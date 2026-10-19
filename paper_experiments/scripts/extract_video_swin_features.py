#!/usr/bin/env python3
"""Extract Kinetics-pretrained Swin3D-T frame-centered embeddings."""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision.models.video import Swin3D_T_Weights, swin3d_t

from common import read_csv, write_csv


REPO = Path(__file__).resolve().parents[2]
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


def load_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {path}")
    return frames


def valid_existing(path: Path, length: int) -> bool:
    if not path.exists():
        return False
    try:
        data = np.load(path)
        return data["x"].shape == (length, 768)
    except Exception:
        return False


def encode_windows(
    windows: list[torch.Tensor],
    transform: torch.nn.Module,
    model: torch.nn.Module,
    device: str,
) -> np.ndarray:
    batch = transform(torch.stack(windows, dim=0)).to(device)
    with torch.no_grad():
        output = model(batch)
    return output.detach().cpu().numpy().astype(np.float32)


def extract_clip(
    row: dict[str, str],
    video_root: Path,
    output: Path,
    transform: torch.nn.Module,
    model: torch.nn.Module,
    device: str,
    batch_size: int,
    window: int,
    stride: int,
) -> None:
    start = int(float(row.get("frame_start", 0)))
    end = int(float(row.get("frame_end", row.get("hit_frame", start))))
    if end < start:
        start, end = end, start
    centers = np.arange(start, end + 1, dtype=np.int32)
    if valid_existing(output, len(centers)):
        return
    frames = load_video(video_root / f"{row['clip_id']}.mp4")
    sample_centers = centers[:: max(1, stride)]
    if sample_centers[-1] != centers[-1]:
        sample_centers = np.concatenate([sample_centers, centers[-1:]])
    half = window // 2
    pending = []
    outputs = []
    for center in sample_centers:
        indexes = [
            max(0, min(int(center) + offset - half, len(frames) - 1))
            for offset in range(window)
        ]
        clip = torch.stack(
            [torch.from_numpy(frames[index]).permute(2, 0, 1) for index in indexes],
            dim=0,
        )
        pending.append(clip)
        if len(pending) == batch_size:
            outputs.append(encode_windows(pending, transform, model, device))
            pending = []
    if pending:
        outputs.append(encode_windows(pending, transform, model, device))
    sampled = np.concatenate(outputs, axis=0)
    nearest = np.asarray(
        [
            int(np.argmin(np.abs(sample_centers.astype(float) - float(center))))
            for center in centers
        ]
    )
    x = sampled[nearest]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x,
        mask=np.ones(len(centers), dtype=np.bool_),
        frames=centers,
        fps=np.asarray(float(row.get("fps", 25.0)), dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", default=os.environ.get("RELIS_VIDEO_ROOT", "data/videos"))
    parser.add_argument("--out-root", default="paper_experiments/round2/video_swin_baseline")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--feature-stride", type=int, default=8)
    parser.add_argument("--split", action="append", type=parse_split, default=[])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_root = (REPO / args.out_root).resolve()
    video_root = Path(args.video_root).resolve()
    split_specs = dict(args.split) if args.split else DEFAULT_SPLITS
    splits = {name: read_csv(path) for name, path in split_specs.items()}
    rows_by_clip = {}
    for rows in splits.values():
        for row in rows:
            rows_by_clip.setdefault(row["clip_id"], row)
    rows = list(rows_by_clip.values())
    if args.limit:
        rows = rows[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Swin3D_T_Weights.DEFAULT
    transform = weights.transforms()
    model = swin3d_t(weights=weights)
    model.head = torch.nn.Identity()
    model.eval().to(device)
    all_dir = output_root / "features/all"
    print(f"[video_swin] device={device} clips={len(rows)}", flush=True)
    for index, row in enumerate(rows, 1):
        extract_clip(
            row,
            video_root,
            all_dir / f"{row['clip_id']}_video_swin.npz",
            transform,
            model,
            device,
            args.batch_size,
            args.window,
            args.feature_stride,
        )
        if index % 10 == 0 or index == len(rows):
            print(f"[video_swin] {index}/{len(rows)}", flush=True)
    if not args.limit and "train" in splits:
        train_dir = output_root / "features/train"
        train_dir.mkdir(parents=True, exist_ok=True)
        for row in splits["train"]:
            source = all_dir / f"{row['clip_id']}_video_swin.npz"
            target = train_dir / source.name
            if not target.exists():
                target.symlink_to(source.resolve())
    if not args.limit:
        manifest_dir = output_root / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        for name, split_rows in splits.items():
            output_rows = []
            for row in split_rows:
                item = dict(row)
                item["video_path"] = str(video_root / f"{row['clip_id']}.mp4")
                item["l2_path"] = str(all_dir / f"{row['clip_id']}_video_swin.npz")
                output_rows.append(item)
            write_csv(manifest_dir / f"{name}_video_swin.csv", output_rows)


if __name__ == "__main__":
    main()
