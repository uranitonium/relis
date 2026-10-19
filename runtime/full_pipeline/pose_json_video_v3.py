#!/usr/bin/env python3
"""
Process every subfolder of ROOT_PARENT, write outputs to mirrored structure.
Tracks both left and right fencers using appearance+pose cues with grace voting.
Left is drawn red and first in JSON; right is green and second.
Referees are filtered by large, low boxes; background movers are de-prioritized.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent

_POSE_RUNTIME_LOADED = False
torch = None
InstanceData = Any
init_detector = None
inference_detector = None
init_model = None
inference_topdown = None
adapt_mmdet_pipeline = None
Detection = None
Tracker = None

# -------- paths/configs ----------------------------------------------------
DEFAULT_ROOT_PARENT = HERE / "preprocessedvideos"
DEFAULT_LIST_DIR = HERE / "processedvideoslist"
HIGH_LIST_NAME = "processed_videos_highfps.txt"
LOW_LIST_NAMES = [
    "processed_videos_lowfps.txt",
    "processed_videos_lowfpsflip.txt",
]
DEFAULT_OUT_PARENT = HERE / "working" / "pose_overlay_v3"

DEFAULT_DETECTOR_CFG = HERE / "mmdetection" / "configs" / "rtmdet" / "rtmdet_tiny_8xb32-300e_coco.py"
DEFAULT_DETECTOR_WTS = HERE / "weights" / "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
DEFAULT_POSE_CFG = HERE / "mmpose" / "configs" / "body_2d_keypoint" / "rtmpose" / "coco" / "rtmpose-m_8xb256-420e_coco-256x192.py"
DEFAULT_POSE_WTS = HERE / "weights" / "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"
FALLBACK_DETECTOR_WTS = Path("weights/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth")
FALLBACK_POSE_WTS = Path("weights/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth")

# BGR colors for pose overlays: left=red, right=green
COLOR_LEFT = (0, 0, 255)
COLOR_RIGHT = (0, 255, 0)
MIN_CONF = 0.2
MAX_DETS = 8
IOU_THR = 0.5
STARTUP_GRACE_FRAMES = 15
SWITCH_COOLDOWN = 8
RECLAIM_FRAMES = 20
APPEARANCE_GATE = 0.35
REF_AREA_RATIO = 0.18  # referee filter: area ratio threshold
REF_BOTTOM_RATIO = 0.82  # referee filter: bottom close to camera

IGNORE_JOINTS = {0, 1, 2, 3, 4}
SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]

TORSO_KEYPOINTS = (5, 6, 11, 12)
LINE_FAR = (148, 154, 164)
LINE_NEAR = (0, 226, 255)
LINE_INSIDE = (0, 64, 255)


def _load_pose_runtime() -> None:
    global _POSE_RUNTIME_LOADED
    global torch, InstanceData, init_detector, inference_detector
    global init_model, inference_topdown, adapt_mmdet_pipeline
    global Detection, Tracker
    if _POSE_RUNTIME_LOADED:
        return
    # Prefer vendored repos in full/ for a self-contained pipeline.
    for repo_dir in (HERE / "mmdetection", HERE / "mmpose"):
        if repo_dir.exists() and str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
    import torch as torch_mod
    from mmengine.structures import InstanceData as InstanceDataCls
    from mmdet.apis.inference import init_detector as init_detector_fn, inference_detector as inference_detector_fn
    from mmpose.apis.inference import init_model as init_model_fn, inference_topdown as inference_topdown_fn
    from mmpose.utils import adapt_mmdet_pipeline as adapt_mmdet_pipeline_fn
    from norfair import Detection as DetectionCls, Tracker as TrackerCls

    torch = torch_mod
    InstanceData = InstanceDataCls
    init_detector = init_detector_fn
    inference_detector = inference_detector_fn
    init_model = init_model_fn
    inference_topdown = inference_topdown_fn
    adapt_mmdet_pipeline = adapt_mmdet_pipeline_fn
    Detection = DetectionCls
    Tracker = TrackerCls
    _POSE_RUNTIME_LOADED = True


def read_list_file(path: Path) -> List[str]:
    if not path.exists():
        logging.warning("List file missing: %s", path)
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def list_videos_from_txt(root_parent: Path, list_dir: Path) -> List[Path]:
    """Build video list from highfps + lowfps text files (low lists merged, deduped)."""
    names_high = read_list_file(list_dir / HIGH_LIST_NAME)
    names_low = read_list_file(list_dir / LOW_LIST_NAMES[0]) + read_list_file(list_dir / LOW_LIST_NAMES[1])
    # Dedup lowfps among themselves
    low_unique = []
    seen_low = set()
    for n in names_low:
        if n not in seen_low:
            seen_low.add(n)
            low_unique.append(n)
    # Combine high + low, dedup overall to avoid double work
    combined = []
    seen_all = set()
    for n in names_high + low_unique:
        if n not in seen_all:
            seen_all.add(n)
            combined.append(n)

    # Build an index of available videos under ROOT_PARENT
    index = {p.name: p for p in root_parent.rglob("*.mp4")}
    vids, missing = [], []
    for name in combined:
        if name in index:
            vids.append(index[name])
        else:
            missing.append(name)
    if missing:
        logging.warning("Missing %d listed videos (not found under %s): %s", len(missing), root_parent, missing)
    return vids


def _resolve_file(path_str: str, fallback: Optional[Path] = None, label: str = "file") -> str:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        # Resolve relative paths against current working dir first, then script dir.
        p_cwd = Path.cwd() / p
        if p_cwd.exists():
            return str(p_cwd.resolve())
        p_here = HERE / p
        if p_here.exists():
            return str(p_here.resolve())
    if p.exists():
        return str(p.resolve())
    if fallback is not None and fallback.exists():
        logging.warning("%s not found at %s, using fallback %s", label, p, fallback)
        return str(fallback.resolve())
    raise FileNotFoundError(f"{label} not found: {p}")


def bbox_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - inter
        + 1e-6
    )
    return inter / union


def compute_hist(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def pose_to_vec(inst: Optional[InstanceData]) -> Optional[np.ndarray]:
    if inst is None or not hasattr(inst, "keypoints") or len(inst.keypoints) == 0:
        return None
    pts = inst.keypoints[0][:, :2].astype(np.float32)
    conf = inst.keypoint_scores[0]
    valid = conf >= MIN_CONF
    if not np.any(valid):
        return None
    pts[~valid] = np.nan
    centre = np.nanmedian(pts, axis=0)
    pts = pts - centre
    scale = np.nanmax(np.linalg.norm(pts, axis=1))
    if not scale or np.isnan(scale):
        return None
    return pts / scale


def appearance_similarity(h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> float:
    if h1 is None or h2 is None:
        return 0.0
    dist = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    return max(0.0, 1.0 - min(dist, 1.0))


def pose_similarity(p1: Optional[np.ndarray], p2: Optional[np.ndarray]) -> float:
    if p1 is None or p2 is None:
        return 0.0
    mask = ~np.isnan(p1[:, 0]) & ~np.isnan(p2[:, 0])
    if not np.any(mask):
        return 0.0
    diff = p1[mask] - p2[mask]
    mse = np.mean(np.sum(diff * diff, axis=1))
    return max(0.0, 1.0 - min(mse, 1.0))


def draw_pose(img, inst: InstanceData, color):
    if len(inst.keypoints) == 0:
        return
    kp = inst.keypoints[0]
    conf = inst.keypoint_scores[0]
    for idx, ((x, y), c) in enumerate(zip(kp, conf)):
        if idx in IGNORE_JOINTS or c < MIN_CONF or x <= 0 or y <= 0:
            continue
        cv2.circle(img, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
    for i, j in SKELETON:
        if conf[i] < MIN_CONF or conf[j] < MIN_CONF:
            continue
        xi, yi = map(int, kp[i])
        xj, yj = map(int, kp[j])
        cv2.line(img, (xi, yi), (xj, yj), color, 2, cv2.LINE_AA)


def _pose_bbox_from_arrays(keypoints: Optional[np.ndarray], scores: Optional[np.ndarray]) -> Optional[Tuple[float, float, float, float]]:
    if keypoints is None or keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return None
    conf = None
    if scores is not None and scores.ndim == 1 and scores.shape[0] >= keypoints.shape[0]:
        conf = scores
    xs: List[float] = []
    ys: List[float] = []
    for idx in range(keypoints.shape[0]):
        if idx in IGNORE_JOINTS:
            continue
        x = float(keypoints[idx, 0])
        y = float(keypoints[idx, 1])
        if not np.isfinite(x) or not np.isfinite(y) or x <= 0 or y <= 0:
            continue
        if conf is not None and float(conf[idx]) < MIN_CONF:
            continue
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        return None
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    pad_x = max(12.0, w * 0.16)
    pad_y = max(14.0, h * 0.18)
    return (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)


def _pose_com_from_arrays(keypoints: Optional[np.ndarray], scores: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
    if keypoints is None or keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return None
    conf = None
    if scores is not None and scores.ndim == 1 and scores.shape[0] >= keypoints.shape[0]:
        conf = scores
    points: List[Tuple[float, float]] = []
    for idx in TORSO_KEYPOINTS:
        if idx >= keypoints.shape[0]:
            continue
        x = float(keypoints[idx, 0])
        y = float(keypoints[idx, 1])
        if not np.isfinite(x) or not np.isfinite(y) or x <= 0 or y <= 0:
            continue
        if conf is not None and float(conf[idx]) < MIN_CONF:
            continue
        points.append((x, y))
    if not points:
        bbox = _pose_bbox_from_arrays(keypoints, scores)
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
    arr = np.asarray(points, dtype=np.float32)
    return (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))


def _critical_distance_line_color(
    left_bbox: Tuple[float, float, float, float],
    right_bbox: Tuple[float, float, float, float],
) -> Tuple[int, int, int]:
    lx0, ly0, lx1, ly1 = left_bbox
    rx0, ry0, rx1, ry1 = right_bbox
    gap_px = max(0.0, rx0 - lx1)
    left_h = max(1.0, ly1 - ly0)
    right_h = max(1.0, ry1 - ry0)
    mean_height = max(1.0, 0.5 * (left_h + right_h))
    gap_ratio = gap_px / mean_height
    if gap_ratio <= 0.34:
        return LINE_INSIDE
    if gap_ratio <= 0.66:
        return LINE_NEAR
    return LINE_FAR


def _draw_pair_distance_overlay(
    img: np.ndarray,
    left_keypoints: Optional[np.ndarray],
    left_scores: Optional[np.ndarray],
    right_keypoints: Optional[np.ndarray],
    right_scores: Optional[np.ndarray],
) -> None:
    left_bbox = _pose_bbox_from_arrays(left_keypoints, left_scores)
    right_bbox = _pose_bbox_from_arrays(right_keypoints, right_scores)
    left_com = _pose_com_from_arrays(left_keypoints, left_scores)
    right_com = _pose_com_from_arrays(right_keypoints, right_scores)
    if left_bbox is None or right_bbox is None or left_com is None or right_com is None:
        return
    line_color = _critical_distance_line_color(left_bbox, right_bbox)
    lcx, lcy = int(round(left_com[0])), int(round(left_com[1]))
    rcx, rcy = int(round(right_com[0])), int(round(right_com[1]))
    cv2.line(img, (lcx, lcy), (rcx, rcy), line_color, 3, cv2.LINE_AA)
    cv2.circle(img, (lcx, lcy), 4, COLOR_LEFT, -1, cv2.LINE_AA)
    cv2.circle(img, (rcx, rcy), 4, COLOR_RIGHT, -1, cv2.LINE_AA)


def filter_initial(bboxes: np.ndarray, scores: np.ndarray, w: int, h: int):
    if len(bboxes) == 0:
        return bboxes, scores
    mask = []
    for box in bboxes:
        x1, y1, x2, y2 = box
        centre_y = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        good_height = h * 0.18 < centre_y < h * 0.9
        reasonable = area > w * h * 0.02
        # referee reject: very large and near bottom
        is_ref = area > w * h * REF_AREA_RATIO and y2 > REF_BOTTOM_RATIO * h
        mask.append(good_height and reasonable and not is_ref)
    mask = np.array(mask)
    return bboxes[mask], scores[mask]


def filter_tracking(bboxes: np.ndarray, scores: np.ndarray, w: int, h: int):
    if len(bboxes) == 0:
        return bboxes, scores
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    bottoms = bboxes[:, 3]
    mask = (areas > 400) & ~((areas > w * h * REF_AREA_RATIO) & (bottoms > REF_BOTTOM_RATIO * h))
    return bboxes[mask], scores[mask]


def nms(bboxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5):
    """Simple NMS to drop duplicate detections that confuse tracking."""
    if len(bboxes) == 0:
        return bboxes, scores
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([bbox_iou(bboxes[i], bboxes[j]) for j in rest])
        order = rest[ious < iou_thr]
    return bboxes[keep], scores[keep]


def filter_dark_referees(frame: np.ndarray, bboxes: np.ndarray, scores: np.ndarray):
    """
    Drop detections that look like all-black attire (typical referee),
    using average HSV saturation/value inside the crop.
    """
    if len(bboxes) == 0:
        return bboxes, scores
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    keep = []
    for i, (x1, y1, x2, y2) in enumerate(bboxes.astype(int)):
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(hsv.shape[1] - 1, x2)
        y2 = min(hsv.shape[0] - 1, y2)
        if x2 <= x1 or y2 <= y1:
            keep.append(i)
            continue
        crop = hsv[y1:y2, x1:x2]
        if crop.size == 0:
            keep.append(i)
            continue
        mean_hsv = crop.reshape(-1, 3).mean(axis=0)
        mean_s, mean_v = mean_hsv[1], mean_hsv[2]
        # Reject very dark + low saturation boxes (black clothing)
        if mean_v < 60 and mean_s < 70:
            continue
        keep.append(i)
    if not keep:
        return bboxes, scores  # avoid dropping everything
    keep = np.array(keep, dtype=int)
    return bboxes[keep], scores[keep]


@dataclass
class Signature:
    hist: Optional[np.ndarray] = None
    pose_vec: Optional[np.ndarray] = None

    def update(self, new_hist: Optional[np.ndarray], new_pose: Optional[np.ndarray]):
        if new_hist is not None:
            if self.hist is None:
                self.hist = new_hist
            else:
                self.hist = 0.6 * self.hist + 0.4 * new_hist
        if new_pose is not None:
            if self.pose_vec is None:
                self.pose_vec = new_pose
            else:
                self.pose_vec = 0.6 * self.pose_vec + 0.4 * new_pose


def score_candidate(track, w: int, h: int, sig: Signature, prefer_left: bool, last_bbox: Optional[np.ndarray]) -> float:
    bbox = track.last_detection.data
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area < w * h * 0.01:
        return -1.0
    horiz = 1.0 - (cx / w) if prefer_left else (cx / w)
    horiz = np.clip(horiz, 0.0, 1.0)
    size_score = min(area / (w * h * 0.12), 1.0)
    appearance_score = appearance_similarity(getattr(track, "appearance", None), sig.hist)
    pose_score = pose_similarity(getattr(track, "pose_vec", None), sig.pose_vec)
    continuity = 0.0
    if last_bbox is not None:
        iou = bbox_iou(bbox, last_bbox)
        continuity = min(iou / 0.5, 1.0) * 0.8
        if iou < 0.08 and appearance_score < APPEARANCE_GATE:
            return -1.0
    vertical_pen = 0.18 if (cy < 0.12 * h or cy > 0.93 * h) else 0.0
    return (
        horiz * 1.4
        + size_score * 0.25
        + appearance_score * 1.0
        + pose_score * 0.7
        + continuity
        - vertical_pen
    )

def combined_distance(det: Detection, trk) -> float:
    """Scalar distance for Norfair versions without vectorized support."""
    x1, y1, x2, y2 = det.data
    x1b, y1b, x2b, y2b = trk.last_detection.data
    ix1, iy1 = max(x1, x1b), max(y1, y1b)
    ix2, iy2 = min(x2, x2b), min(y2, y2b)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        iou_dist = 1.0
    else:
        areaA = (x2 - x1) * (y2 - y1)
        areaB = (x2b - x1b) * (y2b - y1b)
        iou_dist = 1.0 - inter / (areaA + areaB - inter)
    cent_curr = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    cent_prev = np.array([(x1b + x2b) / 2, (y1b + y2b) / 2])
    bbox_size = max(x2 - x1, y2 - y1, x2b - x1b, y2b - y1b)
    centroid_dist = np.linalg.norm(cent_curr - cent_prev) / max(bbox_size, 1)
    app_pen = 0.0
    if hasattr(det, "hist") and hasattr(trk, "appearance"):
        app_pen = 1.0 - appearance_similarity(det.hist, trk.appearance)
    return 0.4 * iou_dist + 0.45 * min(centroid_dist, 1.0) + 0.15 * app_pen


def _select_track_by_bbox(tracks, target_bbox: Optional[np.ndarray], excluded_ids: Optional[set] = None):
    if target_bbox is None:
        return None, None
    excluded_ids = excluded_ids or set()
    best_iou = -1.0
    best_track = None
    for t in tracks:
        if t.id in excluded_ids:
            continue
        iou = bbox_iou(t.last_detection.data, target_bbox)
        if iou > best_iou:
            best_iou = iou
            best_track = t
    return best_track, best_iou


def _draw_selection_preview(
    frame: np.ndarray,
    tracks,
    left_id: Optional[int],
    right_id: Optional[int],
    frame_idx: int,
    out_dir: Path,
    video_name: str,
    extra_payload: Optional[Dict[str, Any]] = None,
    selection_assist: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save a clean frame image and let the frontend render all candidate overlays.
    preview = frame.copy()
    candidates: List[Dict[str, Any]] = []
    preview_name = f"{Path(video_name).stem}_selection_frame_{frame_idx:04d}.jpg"
    preview_path = out_dir / preview_name
    assist_roles: Dict[int, str] = {}
    if isinstance(selection_assist, dict):
        left_pt = _assist_point(selection_assist, "left", frame.shape[1], frame.shape[0])
        right_pt = _assist_point(selection_assist, "right", frame.shape[1], frame.shape[0])
        used_ids: set[int] = set()
        for role, point in (("left", left_pt), ("right", right_pt)):
            if point is None:
                continue
            best_track = None
            best_dist = None
            for t in tracks:
                if int(t.id) in used_ids:
                    continue
                dist = float(np.linalg.norm(_bbox_center(t.last_detection.data) - point))
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_track = t
            if best_track is not None:
                used_ids.add(int(best_track.id))
                assist_roles[int(best_track.id)] = role
        if left_id is None:
            left_id = next((track_id for track_id, role in assist_roles.items() if role == "left"), left_id)
        if right_id is None:
            right_id = next((track_id for track_id, role in assist_roles.items() if role == "right"), right_id)

    for i, t in enumerate(sorted(tracks, key=lambda x: float(x.id))):
        bbox = np.asarray(t.last_detection.data, dtype=np.float32)
        if t.id == left_id:
            role = "left_auto"
        elif t.id == right_id:
            role = "right_auto"
        else:
            role = "candidate"
        candidates.append(
            {
                "candidate_id": i,
                "track_id": int(t.id),
                "bbox": [float(v) for v in bbox.tolist()],
                "auto_role": role if role != "candidate" else "",
                "assist_role": assist_roles.get(int(t.id), ""),
            }
        )

    cv2.imwrite(str(preview_path), preview)
    payload = {
        "video_name": video_name,
        "frame_idx": int(frame_idx),
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
        "image_file": preview_name,
        "auto_left_track_id": int(left_id) if left_id is not None else None,
        "auto_right_track_id": int(right_id) if right_id is not None else None,
        "auto_left_candidate_id": None,
        "auto_right_candidate_id": None,
        "candidates": candidates,
    }
    if isinstance(extra_payload, dict):
        payload.update(extra_payload)
    if isinstance(selection_assist, dict):
        payload["selection_assist"] = selection_assist
    track_to_candidate = {c["track_id"]: c["candidate_id"] for c in candidates}
    if left_id is not None and int(left_id) in track_to_candidate:
        payload["auto_left_candidate_id"] = track_to_candidate[int(left_id)]
    if right_id is not None and int(right_id) in track_to_candidate:
        payload["auto_right_candidate_id"] = track_to_candidate[int(right_id)]
    meta_path = out_dir / "selection_preview.json"
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info("selection preview written to %s", meta_path)
    return payload


def _track_pose_arrays(track) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    inst = getattr(track, "pose_inst", None)
    if inst is None or not hasattr(inst, "keypoints") or len(inst.keypoints) == 0:
        return None, None
    keypoints = np.asarray(inst.keypoints[0], dtype=np.float32)
    scores = None
    if hasattr(inst, "keypoint_scores") and len(inst.keypoint_scores) > 0:
        scores = np.asarray(inst.keypoint_scores[0], dtype=np.float32)
    return keypoints, scores


def _serialize_track_snapshot(track) -> Dict[str, Any]:
    bbox = np.asarray(track.last_detection.data, dtype=np.float32)
    keypoints, scores = _track_pose_arrays(track)
    payload: Dict[str, Any] = {
        "track_id": int(track.id),
        "bbox": [float(v) for v in bbox.tolist()],
        "keypoints": None,
        "keypoint_scores": None,
    }
    if keypoints is not None:
        payload["keypoints"] = keypoints.astype(np.float32).tolist()
    if scores is not None:
        payload["keypoint_scores"] = scores.astype(np.float32).tolist()
    return payload


def _draw_pose_from_arrays(
    img: np.ndarray,
    keypoints: Optional[np.ndarray],
    scores: Optional[np.ndarray],
    color: Tuple[int, int, int],
) -> None:
    if keypoints is None:
        return
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return
    conf = None
    if scores is not None and scores.ndim == 1 and scores.shape[0] >= keypoints.shape[0]:
        conf = scores
    for idx in range(keypoints.shape[0]):
        if idx in IGNORE_JOINTS:
            continue
        x = float(keypoints[idx, 0])
        y = float(keypoints[idx, 1])
        if not np.isfinite(x) or not np.isfinite(y) or x <= 0 or y <= 0:
            continue
        if conf is not None and float(conf[idx]) < MIN_CONF:
            continue
        cv2.circle(img, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
    for i, j in SKELETON:
        if i >= keypoints.shape[0] or j >= keypoints.shape[0]:
            continue
        xi = float(keypoints[i, 0])
        yi = float(keypoints[i, 1])
        xj = float(keypoints[j, 0])
        yj = float(keypoints[j, 1])
        if not (np.isfinite(xi) and np.isfinite(yi) and np.isfinite(xj) and np.isfinite(yj)):
            continue
        if conf is not None and (float(conf[i]) < MIN_CONF or float(conf[j]) < MIN_CONF):
            continue
        cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)), color, 2, cv2.LINE_AA)


def _write_track_cache_json(
    cache_path: Path,
    *,
    vid: Path,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    frames: List[Dict[str, Any]],
) -> None:
    track_summary: Dict[int, Dict[str, int]] = {}
    for frame in frames:
        frame_idx = int(frame.get("frame_idx", 0))
        for track in frame.get("tracks", []):
            track_id = int(track.get("track_id", -1))
            if track_id < 0:
                continue
            current = track_summary.get(track_id)
            if current is None:
                track_summary[track_id] = {
                    "track_id": track_id,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "frames_seen": 1,
                }
            else:
                current["last_frame"] = frame_idx
                current["frames_seen"] += 1
    payload = {
        "version": "pose_track_cache_v1",
        "video_name": vid.name,
        "video_path": str(vid.resolve()),
        "image_width": int(width),
        "image_height": int(height),
        "fps": float(fps),
        "total_frames": int(total_frames),
        "frames": frames,
        "track_summaries": sorted(track_summary.values(), key=lambda row: int(row["track_id"])),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info("track cache written to %s", cache_path)


def _materialize_pose_from_track_cache(
    *,
    video_path: Path,
    cache_path: Path,
    left_track_id: int,
    right_track_id: int,
    out_dir: Path,
) -> bool:
    if not video_path.exists():
        logging.error("materialize failed: input video not found: %s", video_path)
        return False
    if not cache_path.exists():
        logging.error("materialize failed: track cache not found: %s", cache_path)
        return False
    with open(cache_path, "r") as f:
        cache = json.load(f)
    if not isinstance(cache, dict):
        logging.error("materialize failed: invalid cache payload in %s", cache_path)
        return False
    frames_raw = cache.get("frames")
    if not isinstance(frames_raw, list):
        logging.error("materialize failed: cache frames missing in %s", cache_path)
        return False
    frame_map: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for frame_entry in frames_raw:
        if not isinstance(frame_entry, dict):
            continue
        frame_idx = int(frame_entry.get("frame_idx", 0))
        track_rows = frame_entry.get("tracks")
        if frame_idx <= 0 or not isinstance(track_rows, list):
            continue
        per_track: Dict[int, Dict[str, Any]] = {}
        for row in track_rows:
            if not isinstance(row, dict):
                continue
            try:
                tid = int(row.get("track_id"))
            except Exception:
                continue
            per_track[tid] = row
        frame_map[frame_idx] = per_track

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error("materialize failed: unable to open %s", video_path)
        return False
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(cache.get("image_width") or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(cache.get("image_height") or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(cache.get("total_frames") or 0)
    fps = _safe_output_fps(float(cache.get("fps") or 0.0))
    if fps <= 0:
        fps = _resolve_output_fps(video_path, cv_fps_raw=cap.get(cv2.CAP_PROP_FPS) or 0.0, total_frames=total_frames)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{video_path.stem}_pose.mp4"
    out_json = out_dir / f"{video_path.stem}_pose.json"
    identity_json = out_dir / f"{video_path.stem}_identity_reliability.json"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        logging.error("materialize failed: unable to create output video %s", out_video)
        return False

    joints_log = []
    identity_stats: Dict[str, Any] = {
        "frames": 0,
        "left_missing_frames": 0,
        "right_missing_frames": 0,
        "bbox_overlap_frames": [],
        "left_right_crossing_frames": [],
    }
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        rows = frame_map.get(frame_idx, {})

        left_row = rows.get(left_track_id)
        right_row = rows.get(right_track_id)

        left_kp = None
        left_scores = None
        right_kp = None
        right_scores = None
        if isinstance(left_row, dict) and isinstance(left_row.get("keypoints"), list):
            left_kp = np.asarray(left_row.get("keypoints"), dtype=np.float32)
            if isinstance(left_row.get("keypoint_scores"), list):
                left_scores = np.asarray(left_row.get("keypoint_scores"), dtype=np.float32)
        if isinstance(right_row, dict) and isinstance(right_row.get("keypoints"), list):
            right_kp = np.asarray(right_row.get("keypoints"), dtype=np.float32)
            if isinstance(right_row.get("keypoint_scores"), list):
                right_scores = np.asarray(right_row.get("keypoint_scores"), dtype=np.float32)

        _draw_pose_from_arrays(frame, left_kp, left_scores, COLOR_LEFT)
        _draw_pose_from_arrays(frame, right_kp, right_scores, COLOR_RIGHT)
        _draw_pair_distance_overlay(frame, left_kp, left_scores, right_kp, right_scores)
        identity_stats["frames"] += 1
        if left_kp is None:
            identity_stats["left_missing_frames"] += 1
        if right_kp is None:
            identity_stats["right_missing_frames"] += 1
        if isinstance(left_row, dict) and isinstance(right_row, dict) and isinstance(left_row.get("bbox"), list) and isinstance(right_row.get("bbox"), list):
            left_bbox = np.asarray(left_row.get("bbox"), dtype=np.float32)
            right_bbox = np.asarray(right_row.get("bbox"), dtype=np.float32)
            if bbox_iou(left_bbox, right_bbox) >= 0.25:
                identity_stats["bbox_overlap_frames"].append(frame_idx)
            if _bbox_center(left_bbox)[0] > _bbox_center(right_bbox)[0]:
                identity_stats["left_right_crossing_frames"].append(frame_idx)
        joints_log.append(
            {
                "frame": frame_idx,
                "keypoints": [
                    left_kp.astype(np.float32).tolist() if left_kp is not None else None,
                    right_kp.astype(np.float32).tolist() if right_kp is not None else None,
                ],
            }
        )
        writer.write(frame)

    cap.release()
    writer.release()
    with open(out_json, "w") as f:
        json.dump(joints_log, f)
    total_identity_frames = max(1, int(identity_stats.get("frames") or 0))
    missing_ratio = max(
        float(identity_stats.get("left_missing_frames") or 0) / total_identity_frames,
        float(identity_stats.get("right_missing_frames") or 0) / total_identity_frames,
    )
    overlap_ratio = len(identity_stats.get("bbox_overlap_frames") or []) / total_identity_frames
    crossing_ratio = len(identity_stats.get("left_right_crossing_frames") or []) / total_identity_frames
    flags: List[str] = []
    if missing_ratio >= 0.35:
        flags.append("identity_uncertain")
    if overlap_ratio >= 0.12 or crossing_ratio >= 0.05:
        flags.append("possible_side_swap")
    score = max(0.0, min(1.0, 1.0 - max(missing_ratio, overlap_ratio * 1.5, crossing_ratio * 2.0)))
    with open(identity_json, "w") as f:
        json.dump(
            {
                "score": round(score, 3),
                "flags": flags,
                "missing_frame_ratio": round(missing_ratio, 3),
                "left_missing_frame_ratio": round(float(identity_stats.get("left_missing_frames") or 0) / total_identity_frames, 3),
                "right_missing_frame_ratio": round(float(identity_stats.get("right_missing_frames") or 0) / total_identity_frames, 3),
                "track_id_switches": 0,
                "left_track_switches": 0,
                "right_track_switches": 0,
                "bbox_overlap_frame_count": len(identity_stats.get("bbox_overlap_frames") or []),
                "left_right_crossing_frame_count": len(identity_stats.get("left_right_crossing_frames") or []),
                "affected_segment_idxs": [],
                "sample_frames": {
                    "bbox_overlap": list(identity_stats.get("bbox_overlap_frames") or [])[:12],
                    "left_right_crossing": list(identity_stats.get("left_right_crossing_frames") or [])[:12],
                },
            },
            f,
            indent=2,
        )
    logging.info("materialized pose outputs from cache: %s", out_video)
    return True


def _load_selection_override(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"selection override not found: {path}")
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("selection override must be a JSON object")
    return payload


def _load_selection_assist(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"selection assist not found: {path}")
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("selection assist must be a JSON object")
    return payload


def _assist_point(payload: Optional[Dict[str, Any]], role: str, frame_w: int, frame_h: int) -> Optional[np.ndarray]:
    if not isinstance(payload, dict):
        return None
    raw = payload.get(f"{role}_point")
    if not isinstance(raw, dict):
        return None
    try:
        source_w = max(1.0, float(payload.get("image_width") or frame_w))
        source_h = max(1.0, float(payload.get("image_height") or frame_h))
        x = float(raw.get("x")) * (float(frame_w) / source_w)
        y = float(raw.get("y")) * (float(frame_h) / source_h)
        return np.asarray([max(0.0, min(float(frame_w), x)), max(0.0, min(float(frame_h), y))], dtype=np.float32)
    except Exception:
        return None


def _bbox_center(bbox: Any) -> np.ndarray:
    arr = np.asarray(bbox, dtype=np.float32)
    return np.asarray([(arr[0] + arr[2]) / 2.0, (arr[1] + arr[3]) / 2.0], dtype=np.float32)


def _merge_assisted_detections(
    normal_bboxes: np.ndarray,
    normal_scores: np.ndarray,
    all_bboxes: np.ndarray,
    all_scores: np.ndarray,
    selection_assist: Optional[Dict[str, Any]],
    frame_w: int,
    frame_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(selection_assist, dict) or len(all_bboxes) == 0:
        return normal_bboxes, normal_scores
    left_pt = _assist_point(selection_assist, "left", frame_w, frame_h)
    right_pt = _assist_point(selection_assist, "right", frame_w, frame_h)
    if left_pt is None and right_pt is None:
        return normal_bboxes, normal_scores
    selected: List[int] = []
    used: set[int] = set()
    diag = max(1.0, float(np.hypot(frame_w, frame_h)))
    max_dist = diag * 0.28
    for pt in (left_pt, right_pt):
        if pt is None:
            continue
        best_idx = None
        best_dist = None
        for idx, bbox in enumerate(all_bboxes):
            if idx in used:
                continue
            try:
                score = float(all_scores[idx])
            except Exception:
                score = 0.0
            if score < 0.05:
                continue
            dist = float(np.linalg.norm(_bbox_center(bbox) - pt))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None and (best_dist is None or best_dist <= max_dist):
            used.add(best_idx)
            selected.append(best_idx)
    if not selected:
        return normal_bboxes, normal_scores
    merged_boxes = list(normal_bboxes) if len(normal_bboxes) else []
    merged_scores = list(normal_scores) if len(normal_scores) else []
    for idx in selected:
        bbox = all_bboxes[idx]
        if any(bbox_iou(np.asarray(existing, dtype=np.float32), np.asarray(bbox, dtype=np.float32)) > 0.85 for existing in merged_boxes):
            continue
        merged_boxes.append(bbox)
        merged_scores.append(all_scores[idx])
    if not merged_boxes:
        return normal_bboxes, normal_scores
    return np.asarray(merged_boxes, dtype=np.float32), np.asarray(merged_scores, dtype=np.float32)


def _safe_output_fps(raw_fps: float) -> float:
    try:
        fps = float(raw_fps)
    except Exception:
        fps = 0.0
    # Some phone/browser-recorded WebMs arrive with bogus container timing
    # like 1000 fps and no duration. That makes the rendered review clip play
    # in a fraction of a second even though frame order/count are correct.
    if not np.isfinite(fps) or fps < 1.0 or fps > 120.0:
        return 25.0
    return fps


def _probe_video_timing(path: Path) -> Dict[str, float]:
    payload: Dict[str, float] = {
        "duration_sec": 0.0,
        "avg_frame_rate": 0.0,
        "r_frame_rate": 0.0,
        "nb_frames": 0.0,
    }
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw = json.loads(proc.stdout or "{}")
    except Exception:
        return payload

    streams = raw.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = raw.get("format") or {}

    def _parse_float(value: Any) -> float:
        try:
            out = float(value)
        except Exception:
            return 0.0
        return out if np.isfinite(out) and out > 0 else 0.0

    def _parse_rate(value: Any) -> float:
        text = str(value or "").strip()
        if not text or text in {"0/0", "N/A"}:
            return 0.0
        try:
            if "/" in text:
                num_s, den_s = text.split("/", 1)
                num = float(num_s)
                den = float(den_s)
                if den == 0:
                    return 0.0
                rate = num / den
            else:
                rate = float(text)
        except Exception:
            return 0.0
        return rate if np.isfinite(rate) and rate > 0 else 0.0

    payload["duration_sec"] = _parse_float(fmt.get("duration"))
    payload["avg_frame_rate"] = _parse_rate(stream.get("avg_frame_rate"))
    payload["r_frame_rate"] = _parse_rate(stream.get("r_frame_rate"))
    payload["nb_frames"] = _parse_float(stream.get("nb_frames"))
    return payload


def _resolve_output_fps(
    vid: Path,
    *,
    cv_fps_raw: float,
    total_frames: int,
) -> float:
    cv_fps = _safe_output_fps(cv_fps_raw)
    probe = _probe_video_timing(vid)
    duration_sec = float(probe.get("duration_sec") or 0.0)
    nb_frames = int(round(float(probe.get("nb_frames") or 0.0)))
    derived_from_cv = (float(total_frames) / duration_sec) if total_frames > 0 and duration_sec > 0 else 0.0
    derived_from_probe = (float(nb_frames) / duration_sec) if nb_frames > 0 and duration_sec > 0 else 0.0
    derived_fps = _safe_output_fps(derived_from_probe or derived_from_cv or 0.0)
    avg_rate = _safe_output_fps(float(probe.get("avg_frame_rate") or 0.0))
    r_rate = _safe_output_fps(float(probe.get("r_frame_rate") or 0.0))

    chosen = cv_fps
    # Preserve playback duration when container-reported fps disagrees with
    # the actual frame_count / duration observed for browser/mobile uploads.
    if duration_sec > 0 and total_frames > 0 and derived_fps > 0:
        if abs(derived_fps - cv_fps) / max(derived_fps, cv_fps, 1.0) > 0.15:
            chosen = derived_fps
    elif avg_rate > 0:
        chosen = avg_rate
    elif r_rate > 0:
        chosen = r_rate

    logging.info(
        "video timing %s cv_fps=%.3f chosen_fps=%.3f total_frames=%d ffprobe_duration=%.3f ffprobe_avg=%.3f ffprobe_r=%.3f ffprobe_nb_frames=%d derived_cv=%.3f derived_probe=%.3f",
        vid.name,
        float(cv_fps_raw or 0.0),
        chosen,
        total_frames,
        duration_sec,
        avg_rate,
        r_rate,
        nb_frames,
        derived_from_cv,
        derived_from_probe,
    )
    return chosen


def _open_video_capture(path: Path) -> tuple[cv2.VideoCapture, Path]:
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap, path

    fallback_path = path.with_name(f"{path.stem}_opencv_fallback.mp4")
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        try:
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            logging.error("ffmpeg fallback unavailable for %s: %s", path, exc)
            cap.release()
            return cap, path
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(fallback_path),
    ]
    try:
        logging.warning("VideoCapture failed for %s; retrying via ffmpeg transcode to %s", path, fallback_path)
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if proc.stderr:
            logging.info("ffmpeg fallback stderr for %s:\n%s", path, proc.stderr.strip())
    except Exception as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            logging.error("ffmpeg fallback failed for %s:\n%s", path, detail)
        else:
            logging.error("ffmpeg fallback failed for %s: %s", path, exc)
        cap.release()
        return cap, path

    cap.release()
    retry = cv2.VideoCapture(str(fallback_path))
    if retry.isOpened():
        logging.info("VideoCapture opened ffmpeg fallback %s", fallback_path)
        return retry, fallback_path
    logging.error("VideoCapture still could not open ffmpeg fallback %s", fallback_path)
    return retry, path


def process_video(
    vid: Path,
    det,
    pose,
    out_dir: Path,
    *,
    selection_preview_only: bool = False,
    selection_frame: int = STARTUP_GRACE_FRAMES,
    selection_max_frame: Optional[int] = None,
    selection_preview_dir: Optional[Path] = None,
    selection_override: Optional[Dict[str, Any]] = None,
    selection_assist: Optional[Dict[str, Any]] = None,
    track_cache_path: Optional[Path] = None,
    track_cache_only: bool = False,
):
    _load_pose_runtime()
    cap, opened_path = _open_video_capture(vid)
    if not cap.isOpened():
        logging.error("✗ Failed to open %s", vid)
        return False
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = _resolve_output_fps(
        opened_path,
        cv_fps_raw=cap.get(cv2.CAP_PROP_FPS) or 0.0,
        total_frames=total_frames,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{vid.stem}_pose.mp4"
    out_json = out_dir / f"{vid.stem}_pose.json"
    identity_json = out_dir / f"{vid.stem}_identity_reliability.json"

    writer = None
    if not selection_preview_only and not track_cache_only:
        logging.info("using output fps %.3f for %s", fps, vid.name)
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            logging.error("✗ Failed to create output for %s", vid)
            return False

    tracker = Tracker(
        distance_function=combined_distance,
        distance_threshold=IOU_THR,
        hit_counter_max=75,
        initialization_delay=1,
        detection_threshold=0.08,
    )
    left_id = None
    right_id = None
    left_sig = Signature()
    right_sig = Signature()
    left_last_bbox = None
    right_last_bbox = None
    left_cooldown = 0
    right_cooldown = 0
    left_frames_since_loss = 0
    right_frames_since_loss = 0
    left_votes = {}
    right_votes = {}
    frame_idx = 0
    joints_log = []  # each entry: {"frame": n, "keypoints": [left_kp or None, right_kp or None]}
    identity_stats: Dict[str, Any] = {
        "frames": 0,
        "left_missing_frames": 0,
        "right_missing_frames": 0,
        "left_track_switches": 0,
        "right_track_switches": 0,
        "bbox_overlap_frames": [],
        "left_right_crossing_frames": [],
    }
    previous_left_track_id = None
    previous_right_track_id = None
    track_cache_frames: List[Dict[str, Any]] = []
    preview_saved = False
    selection_scan_start = max(1, int(selection_frame or 1))
    selection_scan_end = max(selection_scan_start, int(selection_max_frame or selection_scan_start))

    def maybe_write_selection_preview(frame: np.ndarray, tracks) -> bool:
        nonlocal preview_saved
        if preview_saved or frame_idx < selection_scan_start:
            return False
        if not selection_preview_only and selection_preview_dir is None:
            return False
        preview_dir = selection_preview_dir or out_dir
        preview_payload = _draw_selection_preview(
            frame,
            tracks,
            left_id,
            right_id,
            frame_idx,
            preview_dir,
            vid.name,
            extra_payload={
                "scan_start_frame": int(selection_scan_start),
                "scan_end_frame": int(selection_scan_end),
            },
            selection_assist=selection_assist,
        )
        candidate_count = len(preview_payload.get("candidates") or [])
        if selection_preview_only:
            if candidate_count >= 2 or frame_idx >= selection_scan_end:
                preview_saved = True
                return True
            return False
        if candidate_count >= 2 or frame_idx >= selection_scan_end:
            preview_saved = True
        return False

    override_target_frame = STARTUP_GRACE_FRAMES
    override_left_bbox = None
    override_right_bbox = None
    override_applied = False
    if selection_override:
        try:
            override_target_frame = int(selection_override.get("frame_idx", STARTUP_GRACE_FRAMES))
        except Exception:
            override_target_frame = STARTUP_GRACE_FRAMES
        if selection_override.get("left_bbox") is not None:
            override_left_bbox = np.asarray(selection_override.get("left_bbox"), dtype=np.float32)
        if selection_override.get("right_bbox") is not None:
            override_right_bbox = np.asarray(selection_override.get("right_bbox"), dtype=np.float32)
        logging.info("selection override loaded for frame %d", override_target_frame)

    if selection_preview_only and selection_scan_start > 1:
        # Source-level Auto Select Beta confirmation frames can be hundreds or
        # thousands of frames into a full bout. Selection preview does not need
        # identity history before that frame, so seek directly instead of running
        # detector/pose on every prior frame and timing out Lambda.
        seek_frame = max(0, selection_scan_start - 1)
        if cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame):
            frame_idx = seek_frame
            logging.info("selection preview seeked directly to frame %d", selection_scan_start)
        else:
            logging.warning("selection preview could not seek to frame %d; scanning from current position", selection_scan_start)

    with torch.no_grad():
        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=vid.name, ncols=78)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            pbar.update(1)

            res = inference_detector(det, frame).pred_instances.cpu().numpy()
            mask = res.labels == 0
            all_bboxes, all_scores = res.bboxes[mask], res.scores[mask]

            if left_id is None and right_id is None:
                bboxes, scores = filter_initial(all_bboxes, all_scores, w, h)
            else:
                bboxes, scores = filter_tracking(all_bboxes, all_scores, w, h)
            bboxes, scores = _merge_assisted_detections(bboxes, scores, all_bboxes, all_scores, selection_assist, w, h)

            if len(bboxes) == 0:
                tracks = tracker.update([])
                if track_cache_path is not None:
                    serialized_tracks = [_serialize_track_snapshot(t) for t in sorted(tracks, key=lambda x: float(x.id))]
                    track_cache_frames.append({"frame_idx": frame_idx, "tracks": serialized_tracks})
                if maybe_write_selection_preview(frame, tracks):
                    break
                if writer is not None:
                    writer.write(frame)
                continue

            # Remove likely referees (dark attire, low saturation)
            bboxes, scores = filter_dark_referees(frame, bboxes, scores)
            bboxes, scores = _merge_assisted_detections(bboxes, scores, all_bboxes, all_scores, selection_assist, w, h)
            if len(bboxes) == 0:
                tracks = tracker.update([])
                if track_cache_path is not None:
                    serialized_tracks = [_serialize_track_snapshot(t) for t in sorted(tracks, key=lambda x: float(x.id))]
                    track_cache_frames.append({"frame_idx": frame_idx, "tracks": serialized_tracks})
                if maybe_write_selection_preview(frame, tracks):
                    break
                if writer is not None:
                    writer.write(frame)
                continue

            # De-duplicate overlapping detections before limiting count
            bboxes, scores = nms(bboxes, scores, iou_thr=0.55)

            if len(bboxes) > MAX_DETS:
                keep = np.argsort(scores)[-MAX_DETS:]
                bboxes, scores = bboxes[keep], scores[keep]

            pad = 0.12
            padded = []
            for x1, y1, x2, y2 in bboxes:
                w_box, h_box = x2 - x1, y2 - y1
                padded.append(
                    [
                        max(0, x1 - pad * w_box),
                        max(0, y1 - pad * h_box),
                        min(w - 1, x2 + pad * w_box),
                        min(h - 1, y2 + pad * h_box),
                    ]
                )
            padded_bboxes = np.array(padded, dtype=np.float32)

            pose_results = inference_topdown(pose, frame, padded_bboxes.astype(np.float32))

            detections = []
            for bbox, score, pose_inst in zip(padded_bboxes, scores, pose_results):
                inst = pose_inst.pred_instances if hasattr(pose_inst, "pred_instances") else None
                det_obj = Detection(
                    points=np.array(
                        [
                            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                            [bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[0], bbox[3]],
                            [bbox[2], bbox[3]],
                        ],
                        dtype=np.float32,
                    ),
                    scores=np.full(5, score, dtype=np.float32),
                    data=bbox.astype(np.float32),
                )
                det_obj.hist = compute_hist(frame, bbox)  # type: ignore[attr-defined]
                det_obj.pose_inst = inst  # type: ignore[attr-defined]
                det_obj.pose_vec = pose_to_vec(inst)  # type: ignore[attr-defined]
                detections.append(det_obj)

            tracks = tracker.update(detections)

            for t in tracks:
                det_obj = t.last_detection
                if hasattr(det_obj, "hist") and det_obj.hist is not None:
                    prev = getattr(t, "appearance", None)
                    t.appearance = det_obj.hist if prev is None else 0.7 * prev + 0.3 * det_obj.hist
                if hasattr(det_obj, "pose_vec"):
                    t.pose_vec = det_obj.pose_vec
                if hasattr(det_obj, "pose_inst"):
                    t.pose_inst = det_obj.pose_inst

            if track_cache_path is not None:
                serialized_tracks = [_serialize_track_snapshot(t) for t in sorted(tracks, key=lambda x: float(x.id))]
                track_cache_frames.append({"frame_idx": frame_idx, "tracks": serialized_tracks})

            active = {t.id: t for t in tracks}

            # ---- grace voting ----
            if frame_idx <= STARTUP_GRACE_FRAMES and tracks:
                for t in tracks:
                    if t.age < 2:
                        continue
                    s_left = score_candidate(t, w, h, left_sig, True, left_last_bbox)
                    s_right = score_candidate(t, w, h, right_sig, False, right_last_bbox)
                    if s_left > 0:
                        left_votes[t.id] = left_votes.get(t.id, 0.0) + s_left
                    if s_right > 0:
                        right_votes[t.id] = right_votes.get(t.id, 0.0) + s_right
                if frame_idx == STARTUP_GRACE_FRAMES:
                    if left_votes:
                        best_l = max(left_votes, key=left_votes.get)
                        if best_l in active:
                            left_id = best_l
                            left_sig.update(getattr(active[best_l], "appearance", None), getattr(active[best_l], "pose_vec", None))
                            left_last_bbox = active[best_l].last_detection.data.copy()
                            left_cooldown = SWITCH_COOLDOWN
                    if right_votes:
                        best_r = max(right_votes, key=right_votes.get)
                        if best_r in active and best_r != left_id:
                            right_id = best_r
                            right_sig.update(getattr(active[best_r], "appearance", None), getattr(active[best_r], "pose_vec", None))
                            right_last_bbox = active[best_r].last_detection.data.copy()
                            right_cooldown = SWITCH_COOLDOWN
                    left_votes.clear()
                    right_votes.clear()

            # Manual selection override (from preview confirmation) is applied at the selection frame
            # by matching the chosen preview boxes to current tracked candidates.
            if (
                not override_applied
                and (override_left_bbox is not None or override_right_bbox is not None)
                and frame_idx >= override_target_frame
                and tracks
            ):
                used_ids = set()
                left_track, left_iou = _select_track_by_bbox(tracks, override_left_bbox, excluded_ids=used_ids)
                if left_track is not None and (left_iou or 0.0) > 0.01:
                    left_id = left_track.id
                    used_ids.add(left_id)
                    left_last_bbox = left_track.last_detection.data.copy()
                    left_sig.update(getattr(left_track, "appearance", None), getattr(left_track, "pose_vec", None))
                    left_cooldown = SWITCH_COOLDOWN
                    left_frames_since_loss = 0
                right_track, right_iou = _select_track_by_bbox(tracks, override_right_bbox, excluded_ids=used_ids)
                if right_track is not None and (right_iou or 0.0) > 0.01:
                    right_id = right_track.id
                    used_ids.add(right_id)
                    right_last_bbox = right_track.last_detection.data.copy()
                    right_sig.update(getattr(right_track, "appearance", None), getattr(right_track, "pose_vec", None))
                    right_cooldown = SWITCH_COOLDOWN
                    right_frames_since_loss = 0
                override_applied = True
                left_votes.clear()
                right_votes.clear()
                logging.info(
                    "selection override applied at frame %d (left_iou=%.3f right_iou=%.3f)",
                    frame_idx,
                    float(left_iou or 0.0),
                    float(right_iou or 0.0),
                )

            if maybe_write_selection_preview(frame, tracks):
                break

            # ---- ongoing selection ----
            if active:
                # left
                left_current = score_candidate(active[left_id], w, h, left_sig, True, left_last_bbox) if left_id in active else -1
                scored_left = sorted(
                    [(score_candidate(t, w, h, left_sig, True, left_last_bbox), t.id, t) for t in tracks],
                    key=lambda x: x[0],
                    reverse=True,
                )
                best_left_score, best_left_id, best_left_track = scored_left[0]
                if left_id is None and frame_idx > STARTUP_GRACE_FRAMES and best_left_score > 0:
                    left_id = best_left_id
                    left_last_bbox = best_left_track.last_detection.data.copy()
                    left_sig.update(getattr(best_left_track, "appearance", None), getattr(best_left_track, "pose_vec", None))
                    left_cooldown = SWITCH_COOLDOWN
                    left_frames_since_loss = 0
                elif left_id is not None and best_left_id is not None:
                    allow_switch = (
                        best_left_id != left_id
                        and best_left_score > left_current + 0.25
                        and left_cooldown == 0
                    )
                    if allow_switch and best_left_score > 0.25:
                        left_id = best_left_id
                        left_last_bbox = best_left_track.last_detection.data.copy()
                        left_cooldown = SWITCH_COOLDOWN
                        left_frames_since_loss = 0
                    elif left_id in active:
                        left_last_bbox = active[left_id].last_detection.data.copy()
                    if left_id in active:
                        left_sig.update(getattr(active[left_id], "appearance", None), getattr(active[left_id], "pose_vec", None))

                # right
                right_current = score_candidate(active[right_id], w, h, right_sig, False, right_last_bbox) if right_id in active else -1
                scored_right = sorted(
                    [(score_candidate(t, w, h, right_sig, False, right_last_bbox), t.id, t) for t in tracks if t.id != left_id],
                    key=lambda x: x[0],
                    reverse=True,
                )
                if scored_right:
                    best_right_score, best_right_id, best_right_track = scored_right[0]
                else:
                    best_right_score, best_right_id, best_right_track = -1, None, None
                if right_id is None and frame_idx > STARTUP_GRACE_FRAMES and best_right_score > 0:
                    right_id = best_right_id
                    right_last_bbox = best_right_track.last_detection.data.copy()
                    right_sig.update(getattr(best_right_track, "appearance", None), getattr(best_right_track, "pose_vec", None))
                    right_cooldown = SWITCH_COOLDOWN
                    right_frames_since_loss = 0
                elif right_id is not None and best_right_id is not None:
                    allow_switch_r = (
                        best_right_id != right_id
                        and best_right_score > right_current + 0.25
                        and right_cooldown == 0
                    )
                    if allow_switch_r and best_right_score > 0.25:
                        right_id = best_right_id
                        right_last_bbox = best_right_track.last_detection.data.copy()
                        right_cooldown = SWITCH_COOLDOWN
                        right_frames_since_loss = 0
                    elif right_id in active:
                        right_last_bbox = active[right_id].last_detection.data.copy()
                    if right_id in active:
                        right_sig.update(getattr(active[right_id], "appearance", None), getattr(active[right_id], "pose_vec", None))

            if left_cooldown > 0:
                left_cooldown -= 1
            if right_cooldown > 0:
                right_cooldown -= 1

            if left_id is not None and left_id not in active:
                left_frames_since_loss += 1
            else:
                left_frames_since_loss = 0
            if right_id is not None and right_id not in active:
                right_frames_since_loss += 1
            else:
                right_frames_since_loss = 0

            if left_id is None and active and left_sig.hist is not None and left_frames_since_loss < RECLAIM_FRAMES:
                best_app, best_tid, best_tr = 0.0, None, None
                for t in tracks:
                    sim = appearance_similarity(getattr(t, "appearance", None), left_sig.hist)
                    if sim > best_app:
                        best_app, best_tid, best_tr = sim, t.id, t
                if best_app >= APPEARANCE_GATE and best_tid is not None and best_tid != right_id:
                    left_id = best_tid
                    left_last_bbox = best_tr.last_detection.data.copy()
                    left_cooldown = SWITCH_COOLDOWN
                    left_frames_since_loss = 0

            if right_id is None and active and right_sig.hist is not None and right_frames_since_loss < RECLAIM_FRAMES:
                best_app, best_tid, best_tr = 0.0, None, None
                for t in tracks:
                    if t.id == left_id:
                        continue
                    sim = appearance_similarity(getattr(t, "appearance", None), right_sig.hist)
                    if sim > best_app:
                        best_app, best_tid, best_tr = sim, t.id, t
                if best_app >= APPEARANCE_GATE and best_tid is not None:
                    right_id = best_tid
                    right_last_bbox = best_tr.last_detection.data.copy()
                    right_cooldown = SWITCH_COOLDOWN
                    right_frames_since_loss = 0

            if left_frames_since_loss > RECLAIM_FRAMES:
                left_id = None
                left_frames_since_loss = 0
            if right_frames_since_loss > RECLAIM_FRAMES:
                right_id = None
                right_frames_since_loss = 0

            # Draw selected left/right only.
            left_pose = None
            right_pose = None
            left_bbox_for_identity = None
            right_bbox_for_identity = None
            if left_id is not None and left_id in active:
                inst = getattr(active[left_id], "pose_inst", None)
                left_bbox_for_identity = np.asarray(active[left_id].last_detection.data, dtype=np.float32)
                if inst is not None:
                    draw_pose(frame, inst, COLOR_LEFT)
                    left_kp_np, left_scores_np = _track_pose_arrays(active[left_id])
                else:
                    left_kp_np, left_scores_np = None, None
                if inst is not None:
                    left_pose = inst.keypoints[0].astype(np.float32).tolist()
            else:
                left_kp_np, left_scores_np = None, None
            if right_id is not None and right_id in active:
                inst = getattr(active[right_id], "pose_inst", None)
                right_bbox_for_identity = np.asarray(active[right_id].last_detection.data, dtype=np.float32)
                if inst is not None:
                    draw_pose(frame, inst, COLOR_RIGHT)
                    right_kp_np, right_scores_np = _track_pose_arrays(active[right_id])
                else:
                    right_kp_np, right_scores_np = None, None
                if inst is not None:
                    right_pose = inst.keypoints[0].astype(np.float32).tolist()
            else:
                right_kp_np, right_scores_np = None, None

            _draw_pair_distance_overlay(frame, left_kp_np, left_scores_np, right_kp_np, right_scores_np)

            if writer is not None:
                identity_stats["frames"] += 1
                if left_pose is None:
                    identity_stats["left_missing_frames"] += 1
                if right_pose is None:
                    identity_stats["right_missing_frames"] += 1
                if left_id is not None:
                    if previous_left_track_id is not None and int(left_id) != int(previous_left_track_id):
                        identity_stats["left_track_switches"] += 1
                    previous_left_track_id = int(left_id)
                if right_id is not None:
                    if previous_right_track_id is not None and int(right_id) != int(previous_right_track_id):
                        identity_stats["right_track_switches"] += 1
                    previous_right_track_id = int(right_id)
                if left_bbox_for_identity is not None and right_bbox_for_identity is not None:
                    if bbox_iou(left_bbox_for_identity, right_bbox_for_identity) >= 0.25:
                        identity_stats["bbox_overlap_frames"].append(frame_idx)
                    if _bbox_center(left_bbox_for_identity)[0] > _bbox_center(right_bbox_for_identity)[0]:
                        identity_stats["left_right_crossing_frames"].append(frame_idx)
                joints_log.append({"frame": frame_idx, "keypoints": [left_pose, right_pose]})
                writer.write(frame)

        pbar.close()

    cap.release()
    if writer is not None:
        writer.release()
        with open(out_json, "w") as f:
            json.dump(joints_log, f)
        total_identity_frames = max(1, int(identity_stats.get("frames") or 0))
        missing_ratio = max(
            float(identity_stats.get("left_missing_frames") or 0) / total_identity_frames,
            float(identity_stats.get("right_missing_frames") or 0) / total_identity_frames,
        )
        overlap_ratio = len(identity_stats.get("bbox_overlap_frames") or []) / total_identity_frames
        crossing_ratio = len(identity_stats.get("left_right_crossing_frames") or []) / total_identity_frames
        switches = int(identity_stats.get("left_track_switches") or 0) + int(identity_stats.get("right_track_switches") or 0)
        flags: List[str] = []
        if missing_ratio >= 0.35:
            flags.append("identity_uncertain")
        if switches > 0 or overlap_ratio >= 0.12 or crossing_ratio >= 0.05:
            flags.append("possible_side_swap")
        score = max(0.0, min(1.0, 1.0 - max(missing_ratio, overlap_ratio * 1.5, crossing_ratio * 2.0) - min(0.3, switches * 0.08)))
        identity_payload = {
            "score": round(score, 3),
            "flags": flags,
            "missing_frame_ratio": round(missing_ratio, 3),
            "left_missing_frame_ratio": round(float(identity_stats.get("left_missing_frames") or 0) / total_identity_frames, 3),
            "right_missing_frame_ratio": round(float(identity_stats.get("right_missing_frames") or 0) / total_identity_frames, 3),
            "track_id_switches": switches,
            "left_track_switches": int(identity_stats.get("left_track_switches") or 0),
            "right_track_switches": int(identity_stats.get("right_track_switches") or 0),
            "bbox_overlap_frame_count": len(identity_stats.get("bbox_overlap_frames") or []),
            "left_right_crossing_frame_count": len(identity_stats.get("left_right_crossing_frames") or []),
            "affected_segment_idxs": [],
            "sample_frames": {
                "bbox_overlap": list(identity_stats.get("bbox_overlap_frames") or [])[:12],
                "left_right_crossing": list(identity_stats.get("left_right_crossing_frames") or [])[:12],
            },
        }
        with open(identity_json, "w") as f:
            json.dump(identity_payload, f, indent=2)
        logging.info("✔ saved %s", out_video)
    elif selection_preview_only and not preview_saved:
        logging.error("✗ failed to generate selection preview for %s by frame %d", vid, selection_scan_end)
        return False
    if track_cache_path is not None:
        _write_track_cache_json(
            track_cache_path,
            vid=vid,
            width=w,
            height=h,
            fps=fps,
            total_frames=total_frames if total_frames > 0 else frame_idx,
            frames=track_cache_frames,
        )
    if track_cache_only and selection_preview_dir is not None and not preview_saved:
        logging.error("✗ track-cache run finished without a usable selection preview for %s", vid)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract and track left/right fencers from video(s).")
    parser.add_argument("--input-video", default="", help="Single input video path.")
    parser.add_argument("--input-root", default="", help="Input directory to scan recursively for *.mp4.")
    parser.add_argument("--use-list-files", action="store_true", help="Use list files under --list-dir against --root-parent.")
    parser.add_argument("--root-parent", default=str(DEFAULT_ROOT_PARENT), help="Root used with --use-list-files.")
    parser.add_argument("--list-dir", default=str(DEFAULT_LIST_DIR), help="Directory containing processed_videos_*.txt files.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT_PARENT), help="Output root for pose mp4/json.")
    parser.add_argument("--detector-cfg", default=str(DEFAULT_DETECTOR_CFG))
    parser.add_argument("--detector-wts", default=str(DEFAULT_DETECTOR_WTS))
    parser.add_argument("--pose-cfg", default=str(DEFAULT_POSE_CFG))
    parser.add_argument("--pose-wts", default=str(DEFAULT_POSE_WTS))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--selection-preview-only",
        action="store_true",
        help="Run only until --selection-frame and write bbox-only selection preview metadata/image.",
    )
    parser.add_argument(
        "--selection-frame",
        type=int,
        default=STARTUP_GRACE_FRAMES,
        help="First frame index used for auto/manual fencer selection preview scan.",
    )
    parser.add_argument(
        "--selection-max-frame",
        type=int,
        default=STARTUP_GRACE_FRAMES,
        help="Last frame index considered during selection preview scan.",
    )
    parser.add_argument(
        "--selection-preview-dir",
        default="",
        help="Output directory for selection preview image/json (defaults to output dir for each video).",
    )
    parser.add_argument(
        "--selection-override-json",
        default="",
        help="JSON file with manually confirmed left/right bboxes to force on selection frame.",
    )
    parser.add_argument(
        "--selection-assist-json",
        default="",
        help="JSON file with manual approximate fencer-center points used to prioritize selection preview detections.",
    )
    parser.add_argument(
        "--track-cache-json",
        default="",
        help="Write full per-frame tracked-person cache JSON to this path.",
    )
    parser.add_argument(
        "--track-cache-only",
        action="store_true",
        help="Run full detector/pose tracking pass but skip writing left/right pose outputs.",
    )
    parser.add_argument(
        "--materialize-from-track-cache-json",
        default="",
        help="Use an existing track cache to materialize left/right pose outputs without mmdet/mmpose inference.",
    )
    parser.add_argument(
        "--materialize-left-track-id",
        type=int,
        default=None,
        help="Track id to materialize as left fencer (requires --materialize-from-track-cache-json).",
    )
    parser.add_argument(
        "--materialize-right-track-id",
        type=int,
        default=None,
        help="Track id to materialize as right fencer (requires --materialize-from-track-cache-json).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.selection_preview_only and args.track_cache_only:
        logging.error("--selection-preview-only and --track-cache-only are mutually exclusive.")
        return

    if args.materialize_from_track_cache_json:
        if not args.input_video:
            logging.error("--materialize-from-track-cache-json requires --input-video.")
            return
        if args.materialize_left_track_id is None or args.materialize_right_track_id is None:
            logging.error("--materialize-from-track-cache-json requires both --materialize-left-track-id and --materialize-right-track-id.")
            return
        out_root = Path(args.output_root)
        out_root.mkdir(parents=True, exist_ok=True)
        ok = _materialize_pose_from_track_cache(
            video_path=Path(args.input_video),
            cache_path=Path(args.materialize_from_track_cache_json),
            left_track_id=int(args.materialize_left_track_id),
            right_track_id=int(args.materialize_right_track_id),
            out_dir=out_root,
        )
        if not ok:
            raise SystemExit(1)
        return

    _load_pose_runtime()
    dev = args.device
    if dev == "auto":
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info("device %s", dev)

    detector_cfg = _resolve_file(args.detector_cfg, label="detector config")
    detector_wts = _resolve_file(args.detector_wts, fallback=FALLBACK_DETECTOR_WTS, label="detector checkpoint")
    pose_cfg = _resolve_file(args.pose_cfg, label="pose config")
    pose_wts = _resolve_file(args.pose_wts, fallback=FALLBACK_POSE_WTS, label="pose checkpoint")

    det = init_detector(detector_cfg, detector_wts, device=dev)
    det.cfg = adapt_mmdet_pipeline(det.cfg)
    pose = init_model(pose_cfg, pose_wts, device=dev)

    processed = 0
    failed = 0
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selection_preview_root = Path(args.selection_preview_dir) if args.selection_preview_dir else None
    selection_override = _load_selection_override(Path(args.selection_override_json)) if args.selection_override_json else None
    selection_assist = _load_selection_assist(Path(args.selection_assist_json)) if args.selection_assist_json else None

    videos: List[Path] = []
    input_root: Optional[Path] = None
    if args.use_list_files:
        root_parent = Path(args.root_parent)
        list_dir = Path(args.list_dir)
        videos = list_videos_from_txt(root_parent, list_dir)
        input_root = root_parent
    elif args.input_video:
        videos = [Path(args.input_video)]
    else:
        input_root = Path(args.input_root) if args.input_root else None
        if input_root is not None and input_root.exists():
            videos = sorted(input_root.rglob("*.mp4"))

    if not videos:
        logging.error("No videos found. Provide --input-video or --input-root, or use --use-list-files.")
        return
    if args.track_cache_json and len(videos) != 1:
        logging.error("--track-cache-json currently supports exactly one input video.")
        return

    for v in videos:
        if input_root is not None:
            try:
                rel = v.parent.relative_to(input_root)
                out_dir = output_root / rel
            except ValueError:
                out_dir = output_root
        else:
            out_dir = output_root
        try:
            preview_dir = None
            if selection_preview_root is not None:
                if input_root is not None:
                    try:
                        rel = v.parent.relative_to(input_root)
                        preview_dir = selection_preview_root / rel
                    except ValueError:
                        preview_dir = selection_preview_root
                else:
                    preview_dir = selection_preview_root
            cache_path = None
            if args.track_cache_json:
                cache_path = Path(args.track_cache_json)
            elif args.track_cache_only:
                cache_path = out_dir / f"{v.stem}_track_cache.json"
            if process_video(
                v,
                det,
                pose,
                out_dir,
                selection_preview_only=args.selection_preview_only,
                selection_frame=args.selection_frame,
                selection_max_frame=args.selection_max_frame,
                selection_preview_dir=preview_dir,
                selection_override=selection_override,
                selection_assist=selection_assist,
                track_cache_path=cache_path,
                track_cache_only=args.track_cache_only,
            ):
                processed += 1
            else:
                failed += 1
        except Exception:
            failed += 1
            logging.exception("✗ Error processing %s", v)

    logging.info("Processing complete: %d ok, %d failed", processed, failed)


if __name__ == "__main__":
    main()
