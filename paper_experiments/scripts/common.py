#!/usr/bin/env python3
"""Shared utilities for paper experiment tooling."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Interval = Tuple[int, int]


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_sequence(path: str | Path) -> Dict[str, Any]:
    """Read a temporal feature sequence in the L2-compatible schema.

    JSON files use the deployed pose-feature names, while NPZ files are used by
    paper baselines such as RGB frame embeddings. Both return ``l2``,
    ``l2_mask``, and ``frames`` keys so downstream boundary tooling can stay
    modality-agnostic.
    """
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        x = np.asarray(data["x"], dtype=np.float32)
        mask = np.asarray(data["mask"], dtype=np.bool_)
        frames = np.asarray(data["frames"], dtype=np.int32) if "frames" in data else np.arange(x.shape[0], dtype=np.int32)
        feature_names = [f"f{i}" for i in range(x.shape[1] if x.ndim == 2 else 0)]
        return {
            "l2": x,
            "l2_mask": mask,
            "frames": frames,
            "feature_names": feature_names,
            "meta": {"fps": float(data["fps"]) if "fps" in data else 25.0},
        }
    return read_json(path)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def outcome_from_clip_id(clip_id: str) -> Optional[str]:
    name = Path(clip_id).name
    if name.endswith("_pose.mp4"):
        name = name[: -len("_pose.mp4")]
    if name.endswith("_l2.json"):
        name = name[: -len("_l2.json")]
    if not name:
        return None
    prefix = name[0].upper()
    if prefix == "L":
        return "Left"
    if prefix == "R":
        return "Right"
    if prefix in {"N", "T"}:
        return "None"
    if prefix == "D":
        return None
    return None


def clip_stem(path_or_name: str | Path) -> str:
    name = Path(path_or_name).name
    for suffix in ("_pose_h264.mp4", "_pose.mp4", "_boundary.json", "_l2.json", "_rgb.npz", ".mp4", ".json", ".npz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def load_l2_meta(path: str | Path) -> Dict[str, Any]:
    data = read_sequence(path)
    frames = data.get("frames", [])
    if hasattr(frames, "tolist"):
        frames = frames.tolist()
    fps = float(data.get("meta", {}).get("fps", 25.0))
    return {
        "fps": fps,
        "frame_start": int(frames[0]) if frames else 0,
        "frame_end": int(frames[-1]) if frames else max(0, len(data.get("l2", [])) - 1),
        "num_frames": len(data.get("l2", [])),
        "feature_names": data.get("feature_names", []),
    }


def normalize_intervals(intervals: Iterable[Sequence[Any]], min_frame: int = 0, max_frame: Optional[int] = None) -> List[Interval]:
    out: List[Interval] = []
    for raw in intervals:
        if len(raw) < 2:
            continue
        s = int(round(float(raw[0])))
        e = int(round(float(raw[1])))
        if max_frame is not None:
            s = min(max_frame, s)
            e = min(max_frame, e)
        s = max(min_frame, s)
        e = max(min_frame, e)
        if e < s:
            s, e = e, s
        out.append((s, e))
    return merge_intervals(out, gap=0)


def merge_intervals(intervals: Iterable[Interval], gap: int = 0) -> List[Interval]:
    clean = sorted((int(s), int(e)) for s, e in intervals if int(e) >= int(s))
    if not clean:
        return []
    merged = [clean[0]]
    for s, e in clean[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1 + gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def interval_len(interval: Interval) -> int:
    return max(0, int(interval[1]) - int(interval[0]) + 1)


def interval_iou(a: Interval, b: Interval) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = interval_len(a) + interval_len(b) - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_match_by_iou(pred: Sequence[Interval], gold: Sequence[Interval], threshold: float) -> List[Tuple[int, int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gold):
            iou = interval_iou(p, g)
            if iou >= threshold:
                candidates.append((iou, pi, gi))
    candidates.sort(reverse=True)
    used_p = set()
    used_g = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, iou))
    return matches


def precision_recall_f1(tp: int, pred_count: int, gold_count: int) -> Tuple[float, float, float]:
    precision = tp / pred_count if pred_count else (1.0 if gold_count == 0 else 0.0)
    recall = tp / gold_count if gold_count else (1.0 if pred_count == 0 else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def boundary_center(interval: Interval) -> float:
    return 0.5 * (interval[0] + interval[1])


def copy_or_link(src: Path, dst: Path, mode: str = "copy") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def stratified_sample(rows: Sequence[Dict[str, Any]], total: int, key: str, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(key, "")), []).append(dict(row))
    for vals in buckets.values():
        rng.shuffle(vals)
    labels = sorted(buckets)
    if not labels or total <= 0:
        return []
    base = total // len(labels)
    rem = total % len(labels)
    picked: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []
    for i, label in enumerate(labels):
        n = base + (1 if i < rem else 0)
        bucket = buckets[label]
        picked.extend(bucket[:n])
        leftovers.extend(bucket[n:])
    if len(picked) < total:
        rng.shuffle(leftovers)
        picked.extend(leftovers[: total - len(picked)])
    rng.shuffle(picked)
    return picked[:total]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out
