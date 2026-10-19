#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def clip_id(path: Path) -> str:
    suffix = "_pose_processed.json"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def interpolate_parameter(values: np.ndarray, valid: np.ndarray, fallback: float) -> np.ndarray:
    indices = np.arange(len(values), dtype=float)
    if not np.any(valid):
        return np.full_like(values, fallback, dtype=float)
    return np.interp(indices, indices[valid], values[valid]).astype(float)


def smooth(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0 or len(values) < 3:
        return values
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + 2 * radius + 1]) for index in range(len(values))],
        dtype=float,
    )


def pair_transforms(camera: dict[str, Any], mode: str) -> list[np.ndarray]:
    motions = camera.get("motions") or []
    count = int(camera.get("frame_count", len(motions)))
    source_width = float(camera.get("source_width", 1) or 1)
    source_height = float(camera.get("source_height", 1) or 1)
    analysis_width = float(camera.get("analysis_width", source_width) or source_width)
    analysis_height = float(camera.get("analysis_height", source_height) or source_height)

    tx = np.zeros(count, dtype=float)
    ty = np.zeros(count, dtype=float)
    scale = np.ones(count, dtype=float)
    angle = np.zeros(count, dtype=float)
    valid = np.zeros(count, dtype=bool)
    valid[0] = True
    for index in range(1, min(count, len(motions))):
        row = motions[index]
        affine = np.asarray(row.get("affine"), dtype=float)
        if not row.get("valid") or affine.shape != (2, 3) or not np.isfinite(affine).all():
            continue
        a = float(affine[0, 0])
        b = float(affine[0, 1])
        frame_scale = math.sqrt(a * a + b * b)
        frame_angle = math.atan2(b, a)
        # Reject camera estimates inconsistent with a continuous broadcast shot.
        if abs(frame_scale - 1.0) > 0.03 or abs(float(affine[0, 2])) > analysis_width * 0.08:
            continue
        tx[index] = float(affine[0, 2]) * source_width / analysis_width
        ty[index] = float(affine[1, 2]) * source_height / analysis_height
        scale[index] = frame_scale
        angle[index] = frame_angle
        valid[index] = True

    tx = interpolate_parameter(tx, valid, 0.0)
    ty = interpolate_parameter(ty, valid, 0.0)
    scale = interpolate_parameter(scale, valid, 1.0)
    angle = interpolate_parameter(angle, valid, 0.0)
    # Frame zero has no preceding transform. Exclude its forced identity from
    # smoothing so it cannot suppress real camera motion in the first frames.
    tx[1:] = smooth(tx[1:])
    ty[1:] = smooth(ty[1:])
    scale[1:] = smooth(scale[1:])
    angle[1:] = smooth(angle[1:])
    if mode == "translation":
        scale[:] = 1.0
        angle[:] = 0.0

    transforms: list[np.ndarray] = []
    for index in range(count):
        cosine = math.cos(float(angle[index])) * float(scale[index])
        sine = math.sin(float(angle[index])) * float(scale[index])
        transforms.append(
            np.asarray(
                [
                    [cosine, sine, float(tx[index])],
                    [-sine, cosine, float(ty[index])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        )
    transforms[0] = np.eye(3, dtype=float)
    return transforms


def cumulative_transforms(camera: dict[str, Any], mode: str) -> list[np.ndarray]:
    pair = pair_transforms(camera, mode)
    cumulative = [np.eye(3, dtype=float)]
    for index in range(1, len(pair)):
        cumulative.append(pair[index] @ cumulative[-1])
    return cumulative


def stabilize_payload(payload: dict[str, Any], cumulative: list[np.ndarray], mode: str) -> dict[str, Any]:
    output = copy.deepcopy(payload)
    frames = output.get("frame_data") or []
    if not frames:
        return output
    reference_frame = int(frames[0].get("frame_number", 0))
    if reference_frame < 0 or reference_frame >= len(cumulative):
        raise ValueError(f"Reference frame {reference_frame} is outside camera motion range")
    inverse_reference = np.linalg.inv(cumulative[reference_frame])
    for position, frame in enumerate(frames):
        frame_number = int(frame.get("frame_number", position))
        if frame_number < 0 or frame_number >= len(cumulative):
            continue
        relative = cumulative[frame_number] @ inverse_reference
        inverse_relative = np.linalg.inv(relative)
        keypoints = np.asarray(frame.get("keypoints"), dtype=float)
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            continue
        homogeneous = np.concatenate([keypoints, np.ones((len(keypoints), 1), dtype=float)], axis=1)
        stabilized = (inverse_relative @ homogeneous.T).T[:, :2]
        frame["keypoints"] = stabilized.astype(float).tolist()
    output["camera_stabilization"] = {
        "method": f"background_{mode}_inverse_v1",
        "reference_frame": reference_frame,
        "smoothed_pair_transform_radius": 2,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Stabilize processed pose keypoints using background motion.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--camera-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--subdirs", default="left,right,left_from_right")
    parser.add_argument("--mode", choices=("translation", "partial_affine"), default="partial_affine")
    args = parser.parse_args()

    camera_cache = read_json(Path(args.camera_cache))
    processed_root = Path(args.processed_dir)
    out_root = Path(args.out_dir)
    cumulative_cache: dict[str, list[np.ndarray]] = {}
    processed = 0
    skipped = 0
    for subdir in [value.strip() for value in args.subdirs.split(",") if value.strip()]:
        for path in sorted((processed_root / subdir).glob("*_pose_processed.json")):
            cid = clip_id(path)
            camera = camera_cache.get(cid)
            if not camera:
                skipped += 1
                continue
            cumulative = cumulative_cache.setdefault(cid, cumulative_transforms(camera, args.mode))
            payload = stabilize_payload(read_json(path), cumulative, args.mode)
            write_json(out_root / subdir / path.name, payload)
            processed += 1
    print(json.dumps({"processed": processed, "skipped": skipped, "clips": len(cumulative_cache)}, indent=2))


if __name__ == "__main__":
    main()
