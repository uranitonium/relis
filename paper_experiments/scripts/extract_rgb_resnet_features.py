#!/usr/bin/env python3
"""Extract no-overlay RGB frame embeddings for a learned RGB boundary baseline."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

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
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


def preprocess(frames: List[np.ndarray], device: str) -> torch.Tensor:
    batch = []
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        batch.append((tensor - mean) / std)
    return torch.stack(batch, dim=0).to(device)


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
    force: bool = False,
) -> None:
    clip_id = row["clip_id"]
    start = int(float(row.get("frame_start", 0)))
    end = int(float(row.get("frame_end", row.get("hit_frame", start))))
    if end < start:
        start, end = end, start
    frames = np.arange(start, end + 1, dtype=np.int32)
    if not force and valid_existing(out_path, len(frames)):
        return

    video_path = video_root / f"{clip_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    features: List[np.ndarray] = []
    mask: List[bool] = []
    pending: List[np.ndarray] = []
    pending_ok: List[bool] = []

    def flush() -> None:
        nonlocal pending, pending_ok, features, mask
        if not pending:
            return
        ok_indices = [i for i, ok in enumerate(pending_ok) if ok]
        out = np.zeros((len(pending), 512), dtype=np.float32)
        if ok_indices:
            with torch.no_grad():
                tensor = preprocess([pending[i] for i in ok_indices], device)
                emb = model(tensor).detach().cpu().numpy().astype(np.float32)
            for dst_idx, emb_row in zip(ok_indices, emb):
                out[dst_idx] = emb_row
        features.extend(out)
        mask.extend(pending_ok)
        pending = []
        pending_ok = []

    for _frame_idx in frames:
        ok, frame = cap.read()
        if ok and frame is not None:
            pending.append(frame)
            pending_ok.append(True)
        else:
            pending.append(np.zeros((224, 224, 3), dtype=np.uint8))
            pending_ok.append(False)
        if len(pending) >= batch_size:
            flush()
    flush()
    cap.release()

    x = np.asarray(features, dtype=np.float32)
    m = np.asarray(mask, dtype=np.bool_)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=x,
        mask=m,
        frames=frames,
        fps=np.asarray(float(row.get("fps", 25.0)), dtype=np.float32),
    )


def link_train_features(train_rows: List[Dict[str, str]], all_dir: Path, train_dir: Path) -> None:
    train_dir.mkdir(parents=True, exist_ok=True)
    for row in train_rows:
        src = all_dir / f"{row['clip_id']}_rgb.npz"
        dst = train_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            import shutil

            shutil.copy2(src, dst)


def write_rgb_manifests(splits: Dict[str, List[Dict[str, str]]], all_dir: Path, manifest_dir: Path, video_root: Path) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        output = []
        for row in rows:
            item = dict(row)
            item["video_path"] = str(video_root / f"{row['clip_id']}.mp4")
            item["l2_path"] = str(all_dir / f"{row['clip_id']}_rgb.npz")
            output.append(item)
        write_csv(manifest_dir / f"{name}_rgb.csv", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", default="training_no_overlay")
    parser.add_argument("--out-root", default="paper_experiments/experiments/rgb_resnet_baseline")
    parser.add_argument("--split", action="append", type=parse_split, default=[])
    parser.add_argument("--batch-size", type=int, default=96)
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
    print(f"[rgb] device={device} clips={len(rows)} out={all_dir}", flush=True)
    for idx, row in enumerate(rows, 1):
        out_path = all_dir / f"{row['clip_id']}_rgb.npz"
        extract_clip(row, video_root, out_path, model, device, args.batch_size, force=args.force)
        if idx % 25 == 0 or idx == len(rows):
            print(f"[rgb] {idx}/{len(rows)}", flush=True)

    if "train" in splits and not args.limit:
        link_train_features(splits["train"], all_dir, train_dir)
    if not args.limit:
        write_rgb_manifests(splits, all_dir, manifest_dir, video_root)
    print(f"[rgb] wrote manifests to {manifest_dir}", flush=True)


if __name__ == "__main__":
    main()
