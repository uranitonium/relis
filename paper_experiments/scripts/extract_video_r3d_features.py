#!/usr/bin/env python3
"""Extract Kinetics-pretrained R3D frame-centered video embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18

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


def build_encoder(device: str) -> torch.nn.Module:
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


def load_video(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {path}")
    return frames


def preprocess_window(frames: List[np.ndarray], indexes: List[int]) -> torch.Tensor:
    clips = []
    for index in indexes:
        frame = frames[max(0, min(index, len(frames) - 1))]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = 128.0 / min(h, w)
        resized = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        rh, rw = resized.shape[:2]
        y0 = max(0, (rh - 112) // 2)
        x0 = max(0, (rw - 112) // 2)
        crop = resized[y0 : y0 + 112, x0 : x0 + 112]
        if crop.shape[0] != 112 or crop.shape[1] != 112:
            crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        clips.append(torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0)
    clip = torch.stack(clips, dim=1)
    mean = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(3, 1, 1, 1)
    return (clip - mean) / std


def valid_existing(path: Path, expected_len: int) -> bool:
    if not path.exists():
        return False
    try:
        data = np.load(path)
        return int(data["x"].shape[0]) == expected_len and int(data["x"].shape[1]) == 512
    except Exception:
        return False


def extract_clip(
    row: Dict[str, str],
    video_root: Path,
    out_path: Path,
    model: torch.nn.Module,
    device: str,
    batch_size: int,
    window: int,
    feature_stride: int,
    force: bool = False,
) -> None:
    clip_id = row["clip_id"]
    start = int(float(row.get("frame_start", 0)))
    end = int(float(row.get("frame_end", row.get("hit_frame", start))))
    if end < start:
        start, end = end, start
    centers = np.arange(start, end + 1, dtype=np.int32)
    if not force and valid_existing(out_path, len(centers)):
        return

    frames = load_video(video_root / f"{clip_id}.mp4")
    half = window // 2
    sample_centers = centers[:: max(1, feature_stride)]
    if len(sample_centers) == 0 or sample_centers[-1] != centers[-1]:
        sample_centers = np.concatenate([sample_centers, centers[-1:]])
    outputs = []
    pending = []
    for center in sample_centers:
        indexes = [int(center) + offset - half for offset in range(window)]
        pending.append(preprocess_window(frames, indexes))
        if len(pending) >= batch_size:
            with torch.no_grad():
                emb = model(torch.stack(pending, dim=0).to(device)).detach().cpu().numpy().astype(np.float32)
            outputs.append(emb)
            pending = []
    if pending:
        with torch.no_grad():
            emb = model(torch.stack(pending, dim=0).to(device)).detach().cpu().numpy().astype(np.float32)
        outputs.append(emb)

    sampled = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 512), dtype=np.float32)
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
        src = all_dir / f"{row['clip_id']}_r3d.npz"
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
            item["l2_path"] = str(all_dir / f"{row['clip_id']}_r3d.npz")
            output.append(item)
        write_csv(manifest_dir / f"{name}_r3d.csv", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", default="training_no_overlay")
    parser.add_argument("--out-root", default="paper_experiments/experiments/video_r3d_baseline")
    parser.add_argument("--split", action="append", type=parse_split, default=[])
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--feature-stride", type=int, default=1, help="Run R3D every N evaluated frames and repeat nearest features")
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
    model = build_encoder(device)
    print(f"[r3d] device={device} clips={len(rows)} out={all_dir}", flush=True)
    for idx, row in enumerate(rows, 1):
        extract_clip(
            row,
            video_root,
            all_dir / f"{row['clip_id']}_r3d.npz",
            model,
            device,
            args.batch_size,
            args.window,
            args.feature_stride,
            force=args.force,
        )
        if idx % 10 == 0 or idx == len(rows):
            print(f"[r3d] {idx}/{len(rows)}", flush=True)

    if "train" in splits and not args.limit:
        link_train_features(splits["train"], all_dir, train_dir)
    if not args.limit:
        write_manifests(splits, all_dir, manifest_dir, video_root)
    print(f"[r3d] wrote manifests to {manifest_dir}", flush=True)


if __name__ == "__main__":
    main()
