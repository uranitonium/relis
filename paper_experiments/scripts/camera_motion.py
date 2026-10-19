"""Background-camera motion estimation shared by boundary experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ALGORITHM_VERSION = "background_translation_inverse_v1"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _resize_gray(frame: np.ndarray, analysis_width: int) -> tuple[np.ndarray, int, int]:
    source_height, source_width = frame.shape[:2]
    scale = min(1.0, analysis_width / float(source_width))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (int(round(source_width * scale)), int(round(source_height * scale))),
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), source_width, source_height


def _identity_motion(frame: int) -> dict[str, Any]:
    return {
        "frame": frame,
        "valid": False,
        "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "pan_x_norm": 0.0,
        "pan_y_norm": 0.0,
        "zoom_delta": 0.0,
        "inliers": 0,
        "tracked": 0,
    }


def estimate_camera_motion(video_path: Path, analysis_width: int = 640) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    ok, first = capture.read()
    if not ok:
        capture.release()
        return {"frame_count": 0, "motions": []}

    previous, source_width, source_height = _resize_gray(first, analysis_width)
    height, width = previous.shape
    roi_mask = np.zeros_like(previous, dtype=np.uint8)
    roi_mask[: max(1, int(round(height * 0.52))), :] = 255
    motions: list[dict[str, Any]] = [_identity_motion(0)]
    frame_index = 1

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        current, source_width, source_height = _resize_gray(frame, analysis_width)
        row = _identity_motion(frame_index)
        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=700,
            qualityLevel=0.01,
            minDistance=6,
            mask=roi_mask,
            blockSize=7,
        )
        if points is not None and len(points) >= 12:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                previous,
                current,
                points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if next_points is not None and status is not None:
                tracked = status.reshape(-1).astype(bool)
                old = points.reshape(-1, 2)[tracked]
                new = next_points.reshape(-1, 2)[tracked]
                finite = np.isfinite(old).all(axis=1) & np.isfinite(new).all(axis=1)
                old = old[finite]
                new = new[finite]
                row["tracked"] = int(len(old))
                if len(old) >= 12:
                    affine, inlier_mask = cv2.estimateAffinePartial2D(
                        old,
                        new,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=2.5,
                        maxIters=2000,
                        confidence=0.99,
                        refineIters=10,
                    )
                    inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
                    if affine is not None and inliers >= 10:
                        a, b, tx = [float(value) for value in affine[0]]
                        ty = float(affine[1, 2])
                        scale = math.sqrt(a * a + b * b)
                        row.update(
                            {
                                "valid": True,
                                "affine": affine.astype(float).tolist(),
                                "pan_x_norm": tx / float(width),
                                "pan_y_norm": ty / float(height),
                                "zoom_delta": scale - 1.0,
                                "inliers": inliers,
                                "inlier_ratio": round(inliers / max(1, len(old)), 4),
                            }
                        )
        motions.append(row)
        previous = current
        frame_index += 1
    capture.release()
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "frame_count": len(motions),
        "source_width": source_width,
        "source_height": source_height,
        "analysis_width": width,
        "analysis_height": height,
        "motions": motions,
    }
