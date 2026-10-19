#!/usr/bin/env python3
"""L2 pairwise interaction dynamics from L1 outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class L2Config:
    fps: float
    window: int = 15  # frames, odd preferred
    smooth_window: int = 3  # for d_dot, odd
    smooth_poly: int = 2
    stop_speed_thresh: float = 0.02
    toward_mode: str = "pair_axis"  # "pair_axis" | "horizontal"
    feature_profile: str = "llm"  # "llm" | "boundary"
    eps: float = 1e-6


FEATURE_NAMES_L2_LLM = [
    "d",
    "d_dot",
    "closing_speed",
    "vL_toward",
    "vR_toward",
    "net_closing",
    "hand_lead_L",
    "hand_lead_R",
    "arm_ext_L",
    "arm_ext_R",
    "stop_rate_L",
    "stop_rate_R",
    "stopgo_transitions_L",
    "stopgo_transitions_R",
    "hand_forward_L",
    "hand_forward_R",
    "corr_speed",
    "corr_hand",
    "reach_norm_L",
    "reach_norm_R",
    "reach_norm_avg",
    "critical_ratio",
    "critical_delta",
    "critical_cross",
]


FEATURE_NAMES_L2_BOUNDARY = [
    "d",
    "d_dot",
    "closing_speed",
    "vL_toward",
    "vR_toward",
    "vL_lateral",
    "vR_lateral",
    "net_closing",
    "hand_lead_L",
    "hand_lead_R",
    "arm_ext_L",
    "arm_ext_R",
    "stop_rate_L",
    "stop_rate_R",
    "stopgo_transitions_L",
    "stopgo_transitions_R",
    "hand_forward_L",
    "hand_forward_R",
    "corr_speed",
    "corr_hand",
    "reach_norm_L",
    "reach_norm_R",
    "reach_norm_avg",
    "critical_ratio",
    "critical_delta",
    "critical_cross",
]

# Backward-compatible alias used by helper scripts.
FEATURE_NAMES_L2 = FEATURE_NAMES_L2_LLM


L2_SWAP_RULES_LLM = {
    "invariant": [
        "d",
        "d_dot",
        "closing_speed",
        "net_closing",
        "corr_speed",
        "corr_hand",
        "reach_norm_avg",
        "critical_ratio",
        "critical_delta",
        "critical_cross",
    ],
    "sign_flip": [],
    "swap_pairs": [
        ("vL_toward", "vR_toward"),
        ("hand_lead_L", "hand_lead_R"),
        ("arm_ext_L", "arm_ext_R"),
        ("stop_rate_L", "stop_rate_R"),
        ("stopgo_transitions_L", "stopgo_transitions_R"),
        ("hand_forward_L", "hand_forward_R"),
        ("reach_norm_L", "reach_norm_R"),
    ],
}

L2_SWAP_RULES_BOUNDARY = {
    "invariant": [
        "d",
        "d_dot",
        "closing_speed",
        "net_closing",
        "corr_speed",
        "corr_hand",
        "reach_norm_avg",
        "critical_ratio",
        "critical_delta",
        "critical_cross",
    ],
    "sign_flip": [],
    "swap_pairs": [
        ("vL_toward", "vR_toward"),
        ("vL_lateral", "vR_lateral"),
        ("hand_lead_L", "hand_lead_R"),
        ("arm_ext_L", "arm_ext_R"),
        ("stop_rate_L", "stop_rate_R"),
        ("stopgo_transitions_L", "stopgo_transitions_R"),
        ("hand_forward_L", "hand_forward_R"),
        ("reach_norm_L", "reach_norm_R"),
    ],
}


def save_l2_json(
    out_path: str,
    l2: np.ndarray,
    l2_mask: np.ndarray,
    frames: np.ndarray,
    feature_names: List[str],
    swap_rules: Dict[str, List],
    meta: Optional[dict] = None,
) -> None:
    payload = {
        "l2": l2.astype(np.float32).tolist(),
        "l2_mask": l2_mask.astype(bool).tolist(),
        "frames": frames.tolist(),
        "feature_names": feature_names,
        "swap_rules": swap_rules,
        "meta": meta or {},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)


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


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    n = values.shape[0]
    if window <= 1 or n == 0:
        return values.astype(np.float32)
    window = min(window, n)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    n = values.shape[0]
    if window <= 1 or n == 0:
        return values.astype(np.float32)
    window = min(window, n)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    window = max(2, min(window, n))
    half = window // 2
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        xs = x[lo:hi]
        ys = y[lo:hi]
        if xs.shape[0] < 2:
            out[i] = 0.0
            continue
        mx = float(xs.mean())
        my = float(ys.mean())
        vx = float(((xs - mx) ** 2).mean())
        vy = float(((ys - my) ** 2).mean())
        if vx < 1e-8 or vy < 1e-8:
            out[i] = 0.0
            continue
        cov = float(((xs - mx) * (ys - my)).mean())
        out[i] = cov / (math.sqrt(vx) * math.sqrt(vy))
    return out


def _idx_map(feature_names: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(feature_names)}


def _stable_lr_sign(diff_x: np.ndarray, eps: float) -> np.ndarray:
    """Return stable sign of left->right direction using x only."""
    s = np.zeros(diff_x.shape[0], dtype=np.float32)
    nz = np.abs(diff_x) > eps
    s[nz] = np.sign(diff_x[nz]).astype(np.float32)
    if not np.any(nz):
        s[:] = 1.0
        return s
    first = int(np.argmax(nz))
    s[:first] = s[first]
    for i in range(first + 1, s.shape[0]):
        if s[i] == 0.0:
            s[i] = s[i - 1]
    return s


def compute_l2(
    l1: np.ndarray,
    l1_mask: np.ndarray,
    fps: float,
    mask_frame: Optional[np.ndarray],
    l1_feature_names: List[str],
    config: L2Config,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if l1.ndim != 3 or l1.shape[1] != 2:
        raise ValueError("l1 must be [T,2,D1]")
    t = l1.shape[0]
    if mask_frame is None:
        mask_frame = np.ones(t, dtype=bool)
    idx = _idx_map(l1_feature_names)

    com_L = l1[:, 0, [idx["com_x"], idx["com_y"]]]
    com_R = l1[:, 1, [idx["com_x"], idx["com_y"]]]
    vL = l1[:, 0, [idx["com_vx"], idx["com_vy"]]]
    vR = l1[:, 1, [idx["com_vx"], idx["com_vy"]]]
    com_speed_L = l1[:, 0, idx["com_speed"]]
    com_speed_R = l1[:, 1, idx["com_speed"]]
    hand_speed_L = l1[:, 0, idx["hand_speed"]]
    hand_speed_R = l1[:, 1, idx["hand_speed"]]
    hand_lead_L = l1[:, 0, idx["hand_lead"]]
    hand_lead_R = l1[:, 1, idx["hand_lead"]]
    arm_ext_L = l1[:, 0, idx["arm_ext"]]
    arm_ext_R = l1[:, 1, idx["arm_ext"]]
    reach_L = l1[:, 0, idx["reach_norm"]] if "reach_norm" in idx else np.zeros(t, dtype=np.float32)
    reach_R = l1[:, 1, idx["reach_norm"]] if "reach_norm" in idx else np.zeros(t, dtype=np.float32)

    diff = com_R - com_L
    d = np.linalg.norm(diff, axis=1)
    u = diff / (d[:, None] + config.eps)
    u_perp = np.stack([-u[:, 1], u[:, 0]], axis=1)

    dt = 1.0 / fps
    if config.smooth_window >= 3 and config.smooth_window % 2 == 1 and config.smooth_window <= t:
        d_dot = _apply_savgol(d, config.smooth_window, config.smooth_poly, deriv=1, delta=dt)
    else:
        d_dot = np.gradient(d, dt).astype(np.float32)

    closing_speed = np.maximum(0.0, -d_dot)
    if config.toward_mode == "horizontal":
        lr_sign = _stable_lr_sign(diff[:, 0], config.eps)
        vL_toward = vL[:, 0] * lr_sign
        vR_toward = -vR[:, 0] * lr_sign
        vL_lateral = vL[:, 1]
        vR_lateral = -vR[:, 1]
    else:
        vL_toward = np.sum(vL * u, axis=1)
        vR_toward = np.sum(vR * (-u), axis=1)
        vL_lateral = np.sum(vL * u_perp, axis=1)
        vR_lateral = np.sum(vR * (-u_perp), axis=1)
    net_closing = vL_toward + vR_toward

    hand_lead_L_ind = hand_lead_L
    hand_lead_R_ind = hand_lead_R
    arm_ext_L_ind = arm_ext_L
    arm_ext_R_ind = arm_ext_R

    stop_L = com_speed_L < config.stop_speed_thresh
    stop_R = com_speed_R < config.stop_speed_thresh
    stop_rate_L = _rolling_mean(stop_L.astype(np.float32), config.window)
    stop_rate_R = _rolling_mean(stop_R.astype(np.float32), config.window)

    trans_L = (stop_L[1:] != stop_L[:-1]).astype(np.float32)
    trans_R = (stop_R[1:] != stop_R[:-1]).astype(np.float32)
    trans_L = np.concatenate([[0.0], trans_L])
    trans_R = np.concatenate([[0.0], trans_R])
    trans_L = _rolling_sum(trans_L, config.window)
    trans_R = _rolling_sum(trans_R, config.window)
    window_sec = max(1, min(config.window, t)) / fps
    stopgo_trans_L = trans_L / window_sec
    stopgo_trans_R = trans_R / window_sec

    corr_speed = _rolling_corr(com_speed_L, com_speed_R, config.window)
    corr_hand = _rolling_corr(hand_speed_L, hand_speed_R, config.window)

    if config.toward_mode == "horizontal":
        lr_sign = _stable_lr_sign(diff[:, 0], config.eps)
        hand_forward_L = l1[:, 0, idx["hand_vx"]] * lr_sign
        hand_forward_R = -l1[:, 1, idx["hand_vx"]] * lr_sign
    else:
        hand_forward_L = np.sum(l1[:, 0, [idx["hand_vx"], idx["hand_vy"]]] * u, axis=1)
        hand_forward_R = np.sum(l1[:, 1, [idx["hand_vx"], idx["hand_vy"]]] * (-u), axis=1)

    reach_avg = 0.5 * (reach_L + reach_R)
    reach_avg = np.where(reach_avg > config.eps, reach_avg, config.eps)
    critical_ratio = d / reach_avg
    critical_delta = reach_avg - d
    critical_cross = (critical_delta > 0).astype(np.float32)

    if config.feature_profile == "boundary":
        feature_names = FEATURE_NAMES_L2_BOUNDARY
        l2 = np.stack(
            [
                d,
                d_dot,
                closing_speed,
                vL_toward,
                vR_toward,
                vL_lateral,
                vR_lateral,
                net_closing,
                hand_lead_L_ind,
                hand_lead_R_ind,
                arm_ext_L_ind,
                arm_ext_R_ind,
                stop_rate_L,
                stop_rate_R,
                stopgo_trans_L,
                stopgo_trans_R,
                hand_forward_L,
                hand_forward_R,
                corr_speed,
                corr_hand,
                reach_L,
                reach_R,
                reach_avg,
                critical_ratio,
                critical_delta,
                critical_cross,
            ],
            axis=1,
        ).astype(np.float32)
    else:
        feature_names = FEATURE_NAMES_L2_LLM
        l2 = np.stack(
            [
                d,
                d_dot,
                closing_speed,
                vL_toward,
                vR_toward,
                net_closing,
                hand_lead_L_ind,
                hand_lead_R_ind,
                arm_ext_L_ind,
                arm_ext_R_ind,
                stop_rate_L,
                stop_rate_R,
                stopgo_trans_L,
                stopgo_trans_R,
                hand_forward_L,
                hand_forward_R,
                corr_speed,
                corr_hand,
                reach_L,
                reach_R,
                reach_avg,
                critical_ratio,
                critical_delta,
                critical_cross,
            ],
            axis=1,
        ).astype(np.float32)

    l2_mask = mask_frame & l1_mask[:, 0] & l1_mask[:, 1] & np.isfinite(d)
    l2[~l2_mask] = 0.0
    return l2, l2_mask, feature_names


def _load_l1(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        l1 = np.array(data["l1"], dtype=np.float32)
        l1_mask = np.array(data["l1_mask"], dtype=bool)
        frames = np.array(data["frames"], dtype=np.int32)
        feature_names = list(data["feature_names"])
    else:
        data = np.load(path, allow_pickle=True)
        l1 = data["l1"]
        l1_mask = data["l1_mask"]
        frames = data["frames"]
        feature_names = list(data["feature_names"])
    if l1.ndim == 2:
        l1 = l1[:, None, :]
    if l1_mask.ndim == 1:
        l1_mask = l1_mask[:, None]
    return l1.astype(np.float32), l1_mask.astype(bool), feature_names, frames


def _align_by_frames(
    left: Tuple[np.ndarray, np.ndarray, List[str], np.ndarray],
    right: Tuple[np.ndarray, np.ndarray, List[str], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    l1_L, mask_L, feat_L, frames_L = left
    l1_R, mask_R, feat_R, frames_R = right
    if feat_L != feat_R:
        raise ValueError("L1 feature_names mismatch")
    if np.array_equal(frames_L, frames_R):
        l1 = np.concatenate([l1_L[:, :1, :], l1_R[:, :1, :]], axis=1)
        l1_mask = np.concatenate([mask_L[:, :1], mask_R[:, :1]], axis=1)
        return l1, l1_mask, feat_L, frames_L
    common = sorted(set(frames_L.tolist()) & set(frames_R.tolist()))
    if not common:
        raise ValueError("No overlapping frames")
    idx_L = {fn: i for i, fn in enumerate(frames_L.tolist())}
    idx_R = {fn: i for i, fn in enumerate(frames_R.tolist())}
    l1 = np.stack([l1_L[idx_L[fn], 0] for fn in common], axis=0)
    r1 = np.stack([l1_R[idx_R[fn], 0] for fn in common], axis=0)
    l1 = np.stack([l1, r1], axis=1)
    mL = np.array([mask_L[idx_L[fn], 0] for fn in common], dtype=bool)
    mR = np.array([mask_R[idx_R[fn], 0] for fn in common], dtype=bool)
    l1_mask = np.stack([mL, mR], axis=1)
    frames = np.array(common, dtype=np.int32)
    return l1, l1_mask, feat_L, frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute L2 features from L1 json outputs.")
    parser.add_argument("--left-dir", default="l1_out/left", help="Left L1 folder")
    parser.add_argument("--right-dir", default="l1_out/right", help="Right L1 folder")
    parser.add_argument("--out-dir", default="l2_out", help="Output L2 folder")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--stop-speed", type=float, default=0.02)
    parser.add_argument(
        "--toward-mode",
        default="pair_axis",
        choices=["pair_axis", "horizontal"],
        help="How toward/lateral projections are computed. horizontal ignores y-axis drift.",
    )
    parser.add_argument(
        "--feature-profile",
        default="llm",
        choices=["llm", "boundary"],
        help="llm: reduced features for downstream reasoning. boundary: legacy full feature set for boundary model.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=79,
        help="Max frames to keep per clip (truncate after this frame index)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    left_files = [f for f in os.listdir(args.left_dir) if f.endswith("_l1.json")]
    left_files.sort()

    cfg = L2Config(
        fps=args.fps,
        window=args.window,
        stop_speed_thresh=args.stop_speed,
        toward_mode=args.toward_mode,
        feature_profile=args.feature_profile,
    )

    processed = 0
    skipped = 0
    for fname in left_files:
        left_path = os.path.join(args.left_dir, fname)
        right_path = os.path.join(args.right_dir, fname)
        if not os.path.exists(right_path):
            skipped += 1
            continue
        out_name = fname.replace("_l1.json", "_l2.json")
        out_path = os.path.join(args.out_dir, out_name)
        try:
            left = _load_l1(left_path)
            right = _load_l1(right_path)
            l1, l1_mask, feat_names, frames = _align_by_frames(left, right)
            if args.max_frames and l1.shape[0] > args.max_frames:
                l1 = l1[: args.max_frames]
                l1_mask = l1_mask[: args.max_frames]
                frames = frames[: args.max_frames]
            l2, l2_mask, feature_names = compute_l2(l1, l1_mask, args.fps, None, feat_names, cfg)
            swap_rules = L2_SWAP_RULES_BOUNDARY if args.feature_profile == "boundary" else L2_SWAP_RULES_LLM
            save_l2_json(
                out_path,
                l2,
                l2_mask,
                frames,
                feature_names,
                swap_rules,
                meta={
                    "fps": args.fps,
                    "left": left_path,
                    "right": right_path,
                    "toward_mode": args.toward_mode,
                    "feature_profile": args.feature_profile,
                },
            )
            processed += 1
        except Exception as exc:
            skipped += 1
            print(f"skip {fname}: {exc}")
    print(f"batch done: processed={processed} skipped={skipped} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
