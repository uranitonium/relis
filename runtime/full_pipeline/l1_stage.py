#!/usr/bin/env python3
"""L1 per-fencer physical facts from processed_joints_v3 JSONs.

Inputs:
  poses[t,2].xy[J,2], conf[J], fps, mask_frame[t]
Outputs:
  L1[t,2,D1] float32, L1_mask[t,2] bool, optional debug_signals
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class L1Config:
    fps: float
    conf_threshold: float = 0.2
    min_valid_joint_frac: float = 0.5
    interp_max_gap: int = 3
    smooth_window: int = 3  # odd
    smooth_poly: int = 2
    torso_scale_window: int = 11  # odd
    outlier_scale: float = 6.0
    weapon_side: str = "auto"  # "left", "right", or "auto"
    weapon_auto_mode: str = "legacy"  # "legacy" | "stable_center" | "first_center" | "hybrid"
    weapon_center_margin: float = 0.05
    weapon_hybrid_min_conf: float = 0.12
    weapon_ref_frames: int = 15
    use_normalized: bool = True
    interpolated_conf: float = 0.5


FEATURE_NAMES = [
    "com_x",
    "com_y",
    "torso_vec_x",
    "torso_vec_y",
    "torso_angle",
    "com_vx",
    "com_vy",
    "com_speed",
    "com_ax",
    "com_ay",
    "com_acc",
    "hand_x",
    "hand_y",
    "hand_vx",
    "hand_vy",
    "hand_speed",
    "arm_ext",
    "hand_lead",
    "hand_forward_component",
    "stance_width",
    "stance_width_dot",
    "mean_conf",
    "valid_joint_frac",
    "torso_scale_norm",
    "torso_scale_raw",
    "scale_px_per_norm",
    "reach_norm",
    "reach_norm_llm",
]


def save_l1_json(
    out_path: str,
    l1: np.ndarray,
    l1_mask: np.ndarray,
    frames: np.ndarray,
    feature_names: List[str],
    meta: Optional[dict] = None,
) -> None:
    payload = {
        "l1": l1.astype(np.float32).tolist(),
        "l1_mask": l1_mask.astype(bool).tolist(),
        "frames": frames.tolist(),
        "feature_names": feature_names,
        "meta": meta or {},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)


def load_processed_json(
    path: str,
    use_normalized: bool = True,
    interpolated_conf: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, Optional[np.ndarray]]:
    with open(path, "r") as f:
        data = json.load(f)
    names = data["body_keypoint_names"]
    frames = data["frame_data"]
    if not frames:
        raise ValueError(f"{path} has no frame_data")
    key = "normalized_keypoints" if use_normalized else "keypoints"
    if key not in frames[0]:
        raise ValueError(f"{path} missing '{key}' in frame_data[0]")
    xy = np.array([fr[key] for fr in frames], dtype=np.float32)
    raw_xy = None
    if "keypoints" in frames[0]:
        raw_xy = np.array([fr["keypoints"] for fr in frames], dtype=np.float32)
    if xy.ndim != 3:
        raise ValueError(f"{path} keypoints array has shape {xy.shape}, expected [T,J,2]")
    # Confidence is not present; use 1.0, optionally lower for interpolated frames.
    conf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
    if interpolated_conf < 1.0:
        for i, fr in enumerate(frames):
            if fr.get("interpolated"):
                conf[i, :] = interpolated_conf
    frame_numbers = np.array([fr.get("frame_number", i) for i, fr in enumerate(frames)], dtype=np.int32)
    return xy, conf, names, frame_numbers, raw_xy


def build_joint_map(names: Iterable[str]) -> Dict[str, int]:
    name_to_idx = {name: i for i, name in enumerate(names)}
    required = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_wrist",
        "right_wrist",
        "left_ankle",
        "right_ankle",
    ]
    missing = [k for k in required if k not in name_to_idx]
    if missing:
        raise ValueError(f"Missing required joints: {missing}")
    joint_map = {k: name_to_idx[k] for k in required}
    optional = [
        "left_elbow",
        "right_elbow",
        "left_knee",
        "right_knee",
    ]
    for k in optional:
        if k in name_to_idx:
            joint_map[k] = name_to_idx[k]
    return joint_map


def _savgol_coeffs(window: int, poly: int, deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be odd and >= 3")
    if deriv > poly:
        raise ValueError("deriv must be <= poly")
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    a = np.vander(x, poly + 1, increasing=True)
    b = np.zeros(poly + 1, dtype=np.float64)
    b[deriv] = math.factorial(deriv) / (delta ** deriv)
    coeffs, _, _, _ = np.linalg.lstsq(a.T, b, rcond=None)
    return coeffs.astype(np.float32)


def _apply_savgol(values: np.ndarray, window: int, poly: int, deriv: int, delta: float) -> np.ndarray:
    if window > values.shape[0]:
        return np.zeros_like(values)
    coeffs = _savgol_coeffs(window, poly, deriv, delta)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="reflect")
    out = np.convolve(padded, coeffs[::-1], mode="valid")
    return out.astype(np.float32)


def _interp_short_gaps(values: np.ndarray, valid: np.ndarray, max_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    out = values.copy()
    valid_out = valid.copy()
    n = out.shape[0]
    i = 0
    while i < n:
        if valid_out[i]:
            i += 1
            continue
        start = i
        while i < n and not valid_out[i]:
            i += 1
        end = i
        gap = end - start
        if gap <= max_gap and start > 0 and end < n:
            v0 = out[start - 1]
            v1 = out[end]
            for k in range(gap):
                alpha = (k + 1) / (gap + 1)
                out[start + k] = v0 + (v1 - v0) * alpha
            valid_out[start:end] = True
    return out, valid_out


def _fill_nan_nearest(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    idx = np.where(np.isfinite(out))[0]
    if idx.size == 0:
        return out
    # Forward fill then backward fill.
    last = idx[0]
    for i in range(last + 1, out.shape[0]):
        if np.isfinite(out[i]):
            last = i
        else:
            out[i] = out[last]
    last = idx[-1]
    for i in range(last - 1, -1, -1):
        if np.isfinite(out[i]):
            last = i
        else:
            out[i] = out[last]
    return out


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError("window must be odd")
    if window > values.shape[0]:
        return values.copy()
    half = window // 2
    out = np.zeros_like(values)
    for i in range(values.shape[0]):
        lo = max(0, i - half)
        hi = min(values.shape[0], i + half + 1)
        out[i] = np.nanmedian(values[lo:hi])
    return out


def _clamp_vectors(vec: np.ndarray, quality: np.ndarray, scale: float) -> np.ndarray:
    mag = np.linalg.norm(vec, axis=1)
    med = np.nanmedian(mag)
    mad = np.nanmedian(np.abs(mag - med)) + 1e-6
    limit = med + scale * mad
    per = limit * (0.5 + 0.5 * quality)
    factors = np.minimum(1.0, per / (mag + 1e-6))
    return vec * factors[:, None]


def _choose_weapon_side(xy: np.ndarray, valid: np.ndarray, joint_map: Dict[str, int]) -> str:
    li = joint_map["left_wrist"]
    ri = joint_map["right_wrist"]
    ls = joint_map["left_shoulder"]
    rs = joint_map["right_shoulder"]
    left_ok = valid[:, li] & valid[:, ls]
    right_ok = valid[:, ri] & valid[:, rs]
    if left_ok.any():
        left_ext = np.linalg.norm(xy[left_ok, li] - xy[left_ok, ls], axis=1)
        left_med = float(np.nanmedian(left_ext))
    else:
        left_med = -1.0
    if right_ok.any():
        right_ext = np.linalg.norm(xy[right_ok, ri] - xy[right_ok, rs], axis=1)
        right_med = float(np.nanmedian(right_ext))
    else:
        right_med = -1.0
    return "left" if left_med > right_med else "right"


def _estimate_pair_center_x(raw_pair_xy: np.ndarray) -> Optional[float]:
    if raw_pair_xy is None or raw_pair_xy.size == 0:
        return None
    xs = raw_pair_xy[..., 0]
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return None
    return float(0.5 * (np.nanmin(xs) + np.nanmax(xs)))


def _estimate_pair_center_x_from_com(
    raw_pair_xy: np.ndarray,
    joint_map: Dict[str, int],
) -> Optional[float]:
    """Estimate screen-center x from the midpoint between both fencers' COMs."""
    if raw_pair_xy is None or raw_pair_xy.ndim != 4 or raw_pair_xy.shape[1] != 2:
        return None
    ls = joint_map["left_shoulder"]
    rs = joint_map["right_shoulder"]
    lh = joint_map["left_hip"]
    rh = joint_map["right_hip"]
    try:
        sh_l = 0.5 * (raw_pair_xy[:, 0, ls, 0] + raw_pair_xy[:, 0, rs, 0])
        hp_l = 0.5 * (raw_pair_xy[:, 0, lh, 0] + raw_pair_xy[:, 0, rh, 0])
        com_l = 0.5 * (sh_l + hp_l)

        sh_r = 0.5 * (raw_pair_xy[:, 1, ls, 0] + raw_pair_xy[:, 1, rs, 0])
        hp_r = 0.5 * (raw_pair_xy[:, 1, lh, 0] + raw_pair_xy[:, 1, rh, 0])
        com_r = 0.5 * (sh_r + hp_r)

        mid = 0.5 * (com_l + com_r)
    except Exception:
        return None
    mid = mid[np.isfinite(mid)]
    if mid.size == 0:
        return None
    return float(np.nanmedian(mid))


def _build_batch_center_hint_map(
    processed_dir: str,
    config: L1Config,
    max_frames: int,
) -> Dict[str, float]:
    """Precompute one center hint per clip from left/right pair COM midpoint."""
    left_dir = os.path.join(processed_dir, "left")
    right_dir = os.path.join(processed_dir, "right")
    if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
        return {}

    left_files = [f for f in os.listdir(left_dir) if f.endswith("_pose_processed.json")]
    right_set = set(f for f in os.listdir(right_dir) if f.endswith("_pose_processed.json"))
    common = sorted([f for f in left_files if f in right_set])
    out: Dict[str, float] = {}

    for fname in common:
        left_path = os.path.join(left_dir, fname)
        right_path = os.path.join(right_dir, fname)
        try:
            left_xy, _, names, left_frames, left_raw = load_processed_json(
                left_path,
                use_normalized=config.use_normalized,
                interpolated_conf=config.interpolated_conf,
            )
            right_xy, _, names2, right_frames, right_raw = load_processed_json(
                right_path,
                use_normalized=config.use_normalized,
                interpolated_conf=config.interpolated_conf,
            )
            if names != names2 or left_raw is None or right_raw is None:
                continue
            if not np.array_equal(left_frames, right_frames):
                common_frames = sorted(set(left_frames.tolist()) & set(right_frames.tolist()))
                if not common_frames:
                    continue
                idx_l = {fn: i for i, fn in enumerate(left_frames.tolist())}
                idx_r = {fn: i for i, fn in enumerate(right_frames.tolist())}
                left_raw = np.stack([left_raw[idx_l[fn]] for fn in common_frames], axis=0)
                right_raw = np.stack([right_raw[idx_r[fn]] for fn in common_frames], axis=0)
            raw_pair = np.stack([left_raw, right_raw], axis=1)
            if max_frames and raw_pair.shape[0] > max_frames:
                raw_pair = raw_pair[:max_frames]
            joint_map = build_joint_map(names)
            hint = _estimate_pair_center_x_from_com(raw_pair, joint_map)
            if hint is None:
                hint = _estimate_pair_center_x(raw_pair)
            if hint is not None and np.isfinite(hint):
                out[fname] = float(hint)
        except Exception:
            continue
    return out


def _choose_weapon_side_stable_center(
    xy: np.ndarray,
    valid: np.ndarray,
    joint_map: Dict[str, int],
    raw_xy: Optional[np.ndarray] = None,
    center_x_hint: Optional[float] = None,
    prefer_inward_sign: int = 0,
    margin: float = 0.05,
    ref_frames: int = 0,
) -> Tuple[str, float, str]:
    li = joint_map["left_wrist"]
    ri = joint_map["right_wrist"]
    side_xy = raw_xy if raw_xy is not None and raw_xy.shape == xy.shape else xy

    wrist_ok = (
        valid[:, li]
        & valid[:, ri]
        & np.isfinite(side_xy[:, li, 0])
        & np.isfinite(side_xy[:, ri, 0])
    )

    vote_mask = wrist_ok
    if ref_frames and ref_frames > 0:
        k = min(int(ref_frames), int(vote_mask.shape[0]))
        if k > 0:
            first = np.zeros_like(vote_mask)
            first[:k] = True
            vote_first = vote_mask & first
            if vote_first.any():
                vote_mask = vote_first

    # Primary: choose wrist closer to center (stable over selected vote window).
    if center_x_hint is not None and np.isfinite(center_x_hint) and vote_mask.any():
        left_dx = np.abs(side_xy[vote_mask, li, 0] - float(center_x_hint))
        right_dx = np.abs(side_xy[vote_mask, ri, 0] - float(center_x_hint))
        left_med = float(np.nanmedian(left_dx))
        right_med = float(np.nanmedian(right_dx))
        den = max(left_med + right_med, 1e-6)
        conf = abs(left_med - right_med) / den
        if conf >= margin:
            side = "left" if left_med < right_med else "right"
            method = "center_first" if (ref_frames and ref_frames > 0) else "center"
            return side, float(conf), method

    # Secondary: known side fallback (inward direction by x).
    if prefer_inward_sign != 0 and wrist_ok.any():
        left_x = float(np.nanmedian(side_xy[wrist_ok, li, 0]))
        right_x = float(np.nanmedian(side_xy[wrist_ok, ri, 0]))
        spread = abs(left_x - right_x)
        den = max(abs(left_x) + abs(right_x), 1e-6)
        conf = spread / den
        if prefer_inward_sign > 0:
            return ("left" if left_x > right_x else "right"), float(conf), "inward"
        return ("left" if left_x < right_x else "right"), float(conf), "inward"

    # Tertiary: legacy extension fallback.
    side = _choose_weapon_side(xy, valid, joint_map)
    return side, 0.0, "legacy_fallback"


def compute_l1_for_fencer(
    xy: np.ndarray,
    conf: Optional[np.ndarray],
    fps: float,
    mask_frame: Optional[np.ndarray],
    joint_map: Dict[str, int],
    config: L1Config,
    raw_xy: Optional[np.ndarray] = None,
    center_x_hint: Optional[float] = None,
    prefer_inward_sign: int = 0,
    return_debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    t, j, _ = xy.shape
    if conf is None:
        conf = np.ones((t, j), dtype=np.float32)
    if conf.ndim == 1:
        conf = np.tile(conf[None, :], (t, 1))
    if mask_frame is None:
        mask_frame = np.ones(t, dtype=bool)

    # Validity mask per joint.
    finite = np.isfinite(xy).all(axis=-1)
    valid = finite & (conf >= config.conf_threshold)

    # Interpolate short gaps per joint and coordinate.
    xy_interp = xy.copy()
    valid_interp = valid.copy()
    for ji in range(j):
        for axis in range(2):
            vals = xy_interp[:, ji, axis]
            vals, valid_j = _interp_short_gaps(vals, valid_interp[:, ji], config.interp_max_gap)
            vals = _fill_nan_nearest(vals)
            xy_interp[:, ji, axis] = vals
            valid_interp[:, ji] = valid_j

    # Smooth positions (Savitzky-Golay, deriv=0).
    win = config.smooth_window
    poly = config.smooth_poly
    xy_smooth = xy_interp.copy()
    if win >= 3 and win % 2 == 1 and win <= t:
        for ji in range(j):
            for axis in range(2):
                xy_smooth[:, ji, axis] = _apply_savgol(
                    xy_interp[:, ji, axis], win, poly, deriv=0, delta=1.0
                )

    # Joint indices.
    ls = joint_map["left_shoulder"]
    rs = joint_map["right_shoulder"]
    lh = joint_map["left_hip"]
    rh = joint_map["right_hip"]
    la = joint_map["left_ankle"]
    ra = joint_map["right_ankle"]
    le = joint_map.get("left_elbow")
    re = joint_map.get("right_elbow")
    lk = joint_map.get("left_knee")
    rk = joint_map.get("right_knee")

    # Weapon side selection.
    weapon_side = config.weapon_side
    weapon_auto_conf = 1.0
    weapon_auto_method = "manual"
    if weapon_side == "auto":
        if config.weapon_auto_mode == "stable_center":
            weapon_side, weapon_auto_conf, weapon_auto_method = _choose_weapon_side_stable_center(
                xy_smooth,
                valid_interp,
                joint_map,
                raw_xy=raw_xy,
                center_x_hint=center_x_hint,
                prefer_inward_sign=prefer_inward_sign,
                margin=config.weapon_center_margin,
                ref_frames=0,
            )
        elif config.weapon_auto_mode == "first_center":
            weapon_side, weapon_auto_conf, weapon_auto_method = _choose_weapon_side_stable_center(
                xy_smooth,
                valid_interp,
                joint_map,
                raw_xy=raw_xy,
                center_x_hint=center_x_hint,
                prefer_inward_sign=prefer_inward_sign,
                margin=config.weapon_center_margin,
                ref_frames=config.weapon_ref_frames,
            )
        elif config.weapon_auto_mode == "hybrid":
            center_side, center_conf, center_method = _choose_weapon_side_stable_center(
                xy_smooth,
                valid_interp,
                joint_map,
                raw_xy=raw_xy,
                center_x_hint=center_x_hint,
                prefer_inward_sign=prefer_inward_sign,
                margin=config.weapon_center_margin,
                ref_frames=0,
            )
            legacy_side = _choose_weapon_side(xy_smooth, valid_interp, joint_map)
            if center_method == "center" and center_conf >= config.weapon_hybrid_min_conf:
                weapon_side = center_side
                weapon_auto_conf = center_conf
                weapon_auto_method = "hybrid:center"
            else:
                weapon_side = legacy_side
                weapon_auto_conf = 1.0 - float(center_conf)
                weapon_auto_method = f"hybrid:legacy({center_method})"
        else:
            weapon_side = _choose_weapon_side(xy_smooth, valid_interp, joint_map)
            weapon_auto_conf = 1.0
            weapon_auto_method = "legacy"
    if weapon_side not in ("left", "right"):
        raise ValueError("weapon_side must be 'left', 'right', or 'auto'")

    ws = ls if weapon_side == "left" else rs
    wh = joint_map["left_wrist"] if weapon_side == "left" else joint_map["right_wrist"]
    hs = lh if weapon_side == "left" else rh
    ha = la if weapon_side == "left" else ra
    he = le if weapon_side == "left" else re
    hk = lk if weapon_side == "left" else rk

    # Core anchors.
    mid_shoulder = 0.5 * (xy_smooth[:, ls] + xy_smooth[:, rs])
    mid_hip = 0.5 * (xy_smooth[:, lh] + xy_smooth[:, rh])
    com = 0.5 * (mid_shoulder + mid_hip)
    torso_vec = mid_shoulder - mid_hip
    torso_scale = np.linalg.norm(torso_vec, axis=1)
    shoulder_width = np.linalg.norm(xy_smooth[:, ls] - xy_smooth[:, rs], axis=1)
    torso_scale = np.where(torso_scale > 1e-6, torso_scale, shoulder_width)
    torso_scale = _rolling_median(torso_scale, config.torso_scale_window)
    torso_scale_raw = None
    if raw_xy is not None and raw_xy.shape == xy.shape:
        mid_shoulder_raw = 0.5 * (raw_xy[:, ls] + raw_xy[:, rs])
        mid_hip_raw = 0.5 * (raw_xy[:, lh] + raw_xy[:, rh])
        torso_vec_raw = mid_shoulder_raw - mid_hip_raw
        torso_scale_raw = np.linalg.norm(torso_vec_raw, axis=1)
        shoulder_width_raw = np.linalg.norm(raw_xy[:, ls] - raw_xy[:, rs], axis=1)
        torso_scale_raw = np.where(torso_scale_raw > 1e-6, torso_scale_raw, shoulder_width_raw)
        torso_scale_raw = _rolling_median(torso_scale_raw, config.torso_scale_window)
    torso_vec_unit = torso_vec / (torso_scale[:, None] + 1e-6)
    torso_angle = np.arctan2(torso_vec_unit[:, 1], torso_vec_unit[:, 0])

    # Kinematics via Savitzky-Golay derivative.
    dt = 1.0 / fps
    if win >= 3 and win % 2 == 1 and win <= t:
        com_vx = _apply_savgol(com[:, 0], win, poly, deriv=1, delta=dt)
        com_vy = _apply_savgol(com[:, 1], win, poly, deriv=1, delta=dt)
        com_ax = _apply_savgol(com[:, 0], win, poly, deriv=2, delta=dt)
        com_ay = _apply_savgol(com[:, 1], win, poly, deriv=2, delta=dt)
    else:
        com_vx = np.gradient(com[:, 0], dt).astype(np.float32)
        com_vy = np.gradient(com[:, 1], dt).astype(np.float32)
        com_ax = np.gradient(com_vx, dt).astype(np.float32)
        com_ay = np.gradient(com_vy, dt).astype(np.float32)
    com_v = np.stack([com_vx, com_vy], axis=1)
    com_a = np.stack([com_ax, com_ay], axis=1)

    hand = xy_smooth[:, wh]
    if win >= 3 and win % 2 == 1 and win <= t:
        hand_vx = _apply_savgol(hand[:, 0], win, poly, deriv=1, delta=dt)
        hand_vy = _apply_savgol(hand[:, 1], win, poly, deriv=1, delta=dt)
    else:
        hand_vx = np.gradient(hand[:, 0], dt).astype(np.float32)
        hand_vy = np.gradient(hand[:, 1], dt).astype(np.float32)
    hand_v = np.stack([hand_vx, hand_vy], axis=1)

    # Stance width (ankles).
    stance_width = np.linalg.norm(xy_smooth[:, la] - xy_smooth[:, ra], axis=1)
    stance_width = stance_width / (torso_scale + 1e-6)
    ankle_ok = valid_interp[:, la] & valid_interp[:, ra]
    stance_width = np.where(ankle_ok, stance_width, 0.0)
    if win >= 3 and win % 2 == 1 and win <= t:
        stance_width_dot = _apply_savgol(stance_width, win, poly, deriv=1, delta=dt)
    else:
        stance_width_dot = np.gradient(stance_width, dt).astype(np.float32)
    stance_width_dot = np.where(ankle_ok, stance_width_dot, 0.0)

    # Quality metrics.
    mean_conf = conf.mean(axis=1)
    valid_joint_frac = valid_interp.mean(axis=1)
    quality = mean_conf * valid_joint_frac

    # Clamp outliers using quality score.
    com_v = _clamp_vectors(com_v, quality, config.outlier_scale)
    hand_v = _clamp_vectors(hand_v, quality, config.outlier_scale)
    com_a = _clamp_vectors(com_a, quality, config.outlier_scale)

    com_speed = np.linalg.norm(com_v, axis=1)
    com_acc = np.linalg.norm(com_a, axis=1)
    hand_speed = np.linalg.norm(hand_v, axis=1)

    arm_ext = np.linalg.norm(hand - xy_smooth[:, ws], axis=1) / (torso_scale + 1e-6)
    hand_lead = hand_speed - com_speed
    hand_forward_component = np.sum(hand_v * torso_vec_unit, axis=1)

    # Core validity.
    core_valid = (
        valid_interp[:, ls]
        & valid_interp[:, rs]
        & valid_interp[:, lh]
        & valid_interp[:, rh]
        & valid_interp[:, wh]
    )
    l1_mask = mask_frame & (valid_joint_frac >= config.min_valid_joint_frac) & core_valid

    # Boundary reach estimate: older arm+leg approximation used by the trained boundary model.
    # Keep this stable so boundary features match the model's training distribution.
    def _seg_len(a: int, b: int) -> np.ndarray:
        return np.linalg.norm(xy_smooth[:, a] - xy_smooth[:, b], axis=1)

    chain_ok = (
        valid_interp[:, wh]
        & valid_interp[:, ws]
        & valid_interp[:, hs]
        & valid_interp[:, ha]
    )
    if he is not None:
        chain_ok = chain_ok & valid_interp[:, he]
        arm_len = _seg_len(ws, he) + _seg_len(he, wh)
    else:
        arm_len = _seg_len(ws, wh)
    if hk is not None:
        chain_ok = chain_ok & valid_interp[:, hk]
        leg_len = _seg_len(hs, hk) + _seg_len(hk, ha)
    else:
        leg_len = _seg_len(hs, ha)

    torso_link_len = _seg_len(ws, hs)
    reach_norm = (arm_len + leg_len) / 1.5
    reach_norm = np.where(chain_ok, reach_norm, 0.0)
    # LLM-only reach estimate: fuller body chain that better approximates critical distance.
    reach_norm_llm = 1.1 * (arm_len + torso_link_len + leg_len)
    reach_norm_llm = np.where(chain_ok, reach_norm_llm, 0.0)

    scale_px_per_norm = None
    if torso_scale_raw is not None:
        scale_px_per_norm = torso_scale_raw / (torso_scale + 1e-6)

    extra_feats = [
        torso_scale,
        torso_scale_raw if torso_scale_raw is not None else np.zeros_like(torso_scale),
        scale_px_per_norm if scale_px_per_norm is not None else np.ones_like(torso_scale),
    ]

    feats = np.stack(
        [
            com[:, 0],
            com[:, 1],
            torso_vec_unit[:, 0],
            torso_vec_unit[:, 1],
            torso_angle,
            com_v[:, 0],
            com_v[:, 1],
            com_speed,
            com_a[:, 0],
            com_a[:, 1],
            com_acc,
            hand[:, 0],
            hand[:, 1],
            hand_v[:, 0],
            hand_v[:, 1],
            hand_speed,
            arm_ext,
            hand_lead,
            hand_forward_component,
            stance_width,
            stance_width_dot,
            mean_conf,
            valid_joint_frac,
            extra_feats[0],
            extra_feats[1],
            extra_feats[2],
            reach_norm,
            reach_norm_llm,
        ],
        axis=1,
    ).astype(np.float32)

    # Zero out masked frames.
    feats[~l1_mask] = 0.0

    debug = None
    if return_debug:
        debug = {
            "com": com.astype(np.float32),
            "torso_vec_unit": torso_vec_unit.astype(np.float32),
            "torso_scale": torso_scale.astype(np.float32),
            "hand": hand.astype(np.float32),
            "stance_width": stance_width.astype(np.float32),
            "quality": quality.astype(np.float32),
            "weapon_side": np.array([weapon_side] * t),
            "weapon_auto_conf": np.array([weapon_auto_conf] * t, dtype=np.float32),
            "weapon_auto_method": np.array([weapon_auto_method] * t),
        }

    return feats, l1_mask.astype(bool), debug


def compute_l1(
    poses: np.ndarray,
    conf: Optional[np.ndarray],
    fps: float,
    mask_frame: Optional[np.ndarray],
    joint_map: Dict[str, int],
    config: L1Config,
    center_x_hint: Optional[float] = None,
    prefer_inward_signs: Tuple[int, int] = (1, -1),
    return_debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict[str, np.ndarray]]]]:
    if poses.ndim != 4:
        raise ValueError("poses must be [T,2,J,2]")
    t, f, _, _ = poses.shape
    if f != 2:
        raise ValueError("poses must have 2 fencers on axis 1")
    if conf is None:
        conf = None
    elif conf.ndim == 3:
        if conf.shape[:2] != (t, 2):
            raise ValueError("conf shape must match poses[:2]")
    else:
        raise ValueError("conf must be None or shape [T,2,J]")

    l1 = np.zeros((t, 2, len(FEATURE_NAMES)), dtype=np.float32)
    l1_mask = np.zeros((t, 2), dtype=bool)
    debug_all = [] if return_debug else None
    for fi in range(2):
        feats, mask, debug = compute_l1_for_fencer(
            poses[:, fi],
            conf[:, fi] if conf is not None else None,
            fps,
            mask_frame,
            joint_map,
            config,
            raw_xy=None,
            center_x_hint=center_x_hint,
            prefer_inward_sign=prefer_inward_signs[fi] if fi < len(prefer_inward_signs) else 0,
            return_debug=return_debug,
        )
        l1[:, fi] = feats
        l1_mask[:, fi] = mask
        if return_debug:
            debug_all.append(debug)
    return l1, l1_mask, debug_all


def load_pair_from_processed(
    left_path: str,
    right_path: str,
    config: L1Config,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, Optional[np.ndarray]]:
    left_xy, left_conf, names, left_frames, left_raw = load_processed_json(
        left_path, use_normalized=config.use_normalized, interpolated_conf=config.interpolated_conf
    )
    right_xy, right_conf, names2, right_frames, right_raw = load_processed_json(
        right_path, use_normalized=config.use_normalized, interpolated_conf=config.interpolated_conf
    )
    if names != names2:
        raise ValueError("body_keypoint_names differ between left/right")
    # Align by frame_number if needed.
    if not np.array_equal(left_frames, right_frames):
        common = sorted(set(left_frames.tolist()) & set(right_frames.tolist()))
        if not common:
            raise ValueError("No overlapping frame_numbers between left/right")
        left_idx = {fn: i for i, fn in enumerate(left_frames.tolist())}
        right_idx = {fn: i for i, fn in enumerate(right_frames.tolist())}
        left_xy = np.stack([left_xy[left_idx[fn]] for fn in common], axis=0)
        right_xy = np.stack([right_xy[right_idx[fn]] for fn in common], axis=0)
        left_conf = np.stack([left_conf[left_idx[fn]] for fn in common], axis=0)
        right_conf = np.stack([right_conf[right_idx[fn]] for fn in common], axis=0)
        if left_raw is not None and right_raw is not None:
            left_raw = np.stack([left_raw[left_idx[fn]] for fn in common], axis=0)
            right_raw = np.stack([right_raw[right_idx[fn]] for fn in common], axis=0)
        frames = np.array(common, dtype=np.int32)
    else:
        frames = left_frames
    poses = np.stack([left_xy, right_xy], axis=1)
    conf = np.stack([left_conf, right_conf], axis=1)
    if left_raw is not None and right_raw is not None:
        raw = np.stack([left_raw, right_raw], axis=1)
    else:
        raw = None
    return poses, conf, names, frames, raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute L1 features from processed_joints_v3 JSONs.")
    parser.add_argument("--left", help="Left fencer processed JSON path")
    parser.add_argument("--right", help="Right fencer processed JSON path")
    parser.add_argument("--fps", type=float, default=25.0, help="Frames per second (default: 25)")
    parser.add_argument("--out", help="Output .json path")
    parser.add_argument(
        "--processed-dir",
        default="processed_joints_v3",
        help="Root processed_joints_v3 directory to batch process (expects left/ and right/)",
    )
    parser.add_argument(
        "--out-dir",
        default="l1_out",
        help="Output directory for batch mode (one .npz per clip)",
    )
    parser.add_argument(
        "--right-subdir",
        default="right",
        help="Right fencer subdir name under processed-dir (default: right)",
    )
    parser.add_argument(
        "--right-subdirs",
        default="right,left_from_right",
        help="(deprecated) Comma-separated right subdirs to process in batch",
    )
    parser.add_argument(
        "--subdirs",
        default="left,right,left_from_right",
        help="Comma-separated processed_joints_v3 subdirs to process in batch",
    )
    parser.add_argument("--weapon-side", default="auto", choices=["auto", "left", "right"])
    parser.add_argument(
        "--weapon-auto-mode",
        default="legacy",
        choices=["legacy", "stable_center", "first_center", "hybrid"],
        help="Auto weapon arm selection mode.",
    )
    parser.add_argument(
        "--weapon-ref-frames",
        type=int,
        default=15,
        help="For first_center mode, choose active arm from first N frames and track that arm for clip.",
    )
    parser.add_argument(
        "--weapon-hybrid-min-conf",
        type=float,
        default=0.12,
        help="For hybrid mode, minimum center confidence to trust center-based selection.",
    )
    parser.add_argument("--use-raw", action="store_true", help="Use raw keypoints instead of normalized")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=79,
        help="Max frames to keep per clip (truncate after this frame index)",
    )
    args = parser.parse_args()

    config = L1Config(
        fps=args.fps,
        weapon_side=args.weapon_side,
        weapon_auto_mode=args.weapon_auto_mode,
        weapon_hybrid_min_conf=args.weapon_hybrid_min_conf,
        weapon_ref_frames=args.weapon_ref_frames,
        use_normalized=not args.use_raw,
    )
    if args.processed_dir and not (args.left and args.right and args.out):
        os.makedirs(args.out_dir, exist_ok=True)
        center_hint_map: Dict[str, float] = {}
        if (
            config.weapon_side == "auto"
            and config.weapon_auto_mode in ("stable_center", "first_center", "hybrid")
        ):
            center_hint_map = _build_batch_center_hint_map(
                args.processed_dir,
                config,
                args.max_frames,
            )
            if center_hint_map:
                print(f"center hints prepared for {len(center_hint_map)} clips")
        subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
        for subdir in subdirs:
            in_dir = os.path.join(args.processed_dir, subdir)
            if not os.path.isdir(in_dir):
                print(f"skip subdir {subdir}: missing {in_dir}")
                continue
            out_dir = os.path.join(args.out_dir, subdir)
            os.makedirs(out_dir, exist_ok=True)
            subdir_lower = subdir.lower()
            if subdir_lower == "right":
                prefer_inward_sign = -1
            elif subdir_lower in ("left", "left_from_right"):
                prefer_inward_sign = 1
            else:
                prefer_inward_sign = 0
            files = [f for f in os.listdir(in_dir) if f.endswith("_pose_processed.json")]
            files.sort()
            processed = 0
            skipped = 0
            for fname in files:
                in_path = os.path.join(in_dir, fname)
                out_name = fname.replace("_pose_processed.json", "_l1.json")
                out_path = os.path.join(out_dir, out_name)
                try:
                    xy, conf, names, frames, raw_xy = load_processed_json(
                        in_path,
                        use_normalized=config.use_normalized,
                        interpolated_conf=config.interpolated_conf,
                    )
                    if args.max_frames and xy.shape[0] > args.max_frames:
                        xy = xy[: args.max_frames]
                        conf = conf[: args.max_frames]
                        frames = frames[: args.max_frames]
                        if raw_xy is not None:
                            raw_xy = raw_xy[: args.max_frames]
                    joint_map = build_joint_map(names)
                    feats, mask, _ = compute_l1_for_fencer(
                        xy,
                        conf,
                        args.fps,
                        None,
                        joint_map,
                        config,
                        raw_xy=raw_xy,
                        center_x_hint=center_hint_map.get(fname),
                        prefer_inward_sign=prefer_inward_sign,
                        return_debug=False,
                    )
                    l1 = feats[:, None, :]
                    l1_mask = mask[:, None]
                    save_l1_json(
                        out_path,
                        l1,
                        l1_mask,
                        frames,
                        FEATURE_NAMES,
                        meta={"fps": args.fps, "source": in_path},
                    )
                    processed += 1
                except Exception as exc:
                    skipped += 1
                    print(f"skip {subdir}/{fname}: {exc}")
            print(f"batch done ({subdir}): processed={processed} skipped={skipped} out_dir={out_dir}")
        return

    if not (args.left and args.right and args.out):
        raise SystemExit("--left, --right, and --out are required for single-pair mode")
    poses, conf, names, frames, raw = load_pair_from_processed(args.left, args.right, config)
    if args.max_frames and poses.shape[0] > args.max_frames:
        poses = poses[: args.max_frames]
        conf = conf[: args.max_frames]
        frames = frames[: args.max_frames]
        if raw is not None:
            raw = raw[: args.max_frames]
    joint_map = build_joint_map(names)
    if raw is None:
        l1, l1_mask, _ = compute_l1(
            poses,
            conf,
            args.fps,
            None,
            joint_map,
            config,
            center_x_hint=None,
            prefer_inward_signs=(1, -1),
            return_debug=False,
        )
    else:
        # Compute per-fencer with raw scale anchor.
        l1 = np.zeros((poses.shape[0], 2, len(FEATURE_NAMES)), dtype=np.float32)
        l1_mask = np.zeros((poses.shape[0], 2), dtype=bool)
        center_x_hint = _estimate_pair_center_x_from_com(raw, joint_map)
        if center_x_hint is None:
            center_x_hint = _estimate_pair_center_x(raw)
        for fi in range(2):
            feats, mask, _ = compute_l1_for_fencer(
                poses[:, fi],
                conf[:, fi],
                args.fps,
                None,
                joint_map,
                config,
                raw_xy=raw[:, fi],
                center_x_hint=center_x_hint,
                prefer_inward_sign=1 if fi == 0 else -1,
                return_debug=False,
            )
            l1[:, fi] = feats
            l1_mask[:, fi] = mask
    save_l1_json(
        args.out,
        l1,
        l1_mask,
        frames,
        FEATURE_NAMES,
        meta={"fps": args.fps, "left": args.left, "right": args.right},
    )


if __name__ == "__main__":
    main()
