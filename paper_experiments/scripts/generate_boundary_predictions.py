#!/usr/bin/env python3
"""Generate boundary predictions for baselines and deployed model variants."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from common import clip_stem, merge_intervals, read_csv, read_json, read_sequence, write_json


Interval = Tuple[int, int]
_MODEL_CACHE: Dict[Tuple[str, str, int, str], Tuple[Any, Dict[str, Any]]] = {}


def load_l2(path: str | Path) -> Dict[str, Any]:
    return read_sequence(path)


def seq_len(row: Dict[str, str], data: Dict[str, Any]) -> int:
    x = data.get("l2")
    if x is not None:
        return len(x)
    return int(float(row.get("num_frames", 0)))


def source_frames(data: Dict[str, Any], n: int) -> List[int]:
    raw_frames = data.get("frames", [])
    if hasattr(raw_frames, "tolist"):
        raw_frames = raw_frames.tolist()
    frames = [int(frame) for frame in raw_frames]
    if len(frames) >= n:
        return frames[:n]
    return list(range(n))


def source_intervals(data: Dict[str, Any], intervals: List[Interval], n: int) -> List[Interval]:
    frames = source_frames(data, n)
    if not frames:
        return []
    out: List[Interval] = []
    for start, end in intervals:
        start = max(0, min(int(start), len(frames) - 1))
        end = max(start, min(int(end), len(frames) - 1))
        out.append((frames[start], frames[end]))
    return out


def partition_states(data: Dict[str, Any], zones: List[Interval], n: int) -> List[Dict[str, Any]]:
    frames = source_frames(data, n)
    if not frames:
        return []
    mask = data.get("l2_mask")
    if mask is None:
        mask = [True] * n
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    valid = [i for i in range(min(n, len(mask))) if bool(mask[i])]
    if not valid:
        return []
    lo, hi = valid[0], valid[-1]
    states: List[Tuple[int, int, str]] = []
    cursor = lo
    for raw_start, raw_end in sorted(zones):
        start = max(lo, int(raw_start))
        end = min(hi, int(raw_end))
        if end < start:
            continue
        if start > cursor:
            states.append((cursor, start - 1, "stable"))
        # Do not merge adjacent active intervals: the deployed valley split
        # intentionally represents consecutive interaction states.
        states.append((start, end, "active"))
        cursor = max(cursor, end + 1)
    if cursor <= hi:
        states.append((cursor, hi, "stable"))
    rows = [
        {"start_frame": frames[start], "end_frame": frames[end], "activity": activity}
        for start, end, activity in states
    ]
    # Human annotations cover the full source-video interval. If L2 skipped a
    # few source frames, assign that gap to the preceding state so prediction
    # segments still form a complete, non-overlapping partition.
    for index in range(len(rows) - 1):
        rows[index]["end_frame"] = rows[index + 1]["start_frame"] - 1
    rows[-1]["end_frame"] = frames[hi]
    return rows


def uniform_zones(n: int, count: int = 3, zone_len: int = 4) -> List[Interval]:
    if n <= 0 or count <= 0:
        return []
    out: List[Interval] = []
    for i in range(1, count + 1):
        center = int(round(i * n / (count + 1)))
        s = max(0, center - zone_len // 2)
        e = min(n - 1, s + zone_len - 1)
        out.append((s, e))
    return merge_intervals(out)


def motion_energy_zones(data: Dict[str, Any], count: int = 3, zone_len: int = 4) -> List[Interval]:
    x = np.asarray(data.get("l2", []), dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 3:
        return []
    dx = np.abs(np.diff(x, axis=0))
    energy = np.nanmean(dx, axis=1)
    energy = np.concatenate([[energy[0]], energy])
    picks = []
    radius = max(2, zone_len)
    order = np.argsort(-energy)
    for idx in order:
        i = int(idx)
        if any(abs(i - p) <= radius for p in picks):
            continue
        picks.append(i)
        if len(picks) >= count:
            break
    zones = []
    for c in sorted(picks):
        s = max(0, c - zone_len // 2)
        e = min(x.shape[0] - 1, s + zone_len - 1)
        zones.append((s, e))
    return merge_intervals(zones)


def changepoint_zones(data: Dict[str, Any], count: int = 3, zone_len: int = 4) -> List[Interval]:
    x = np.asarray(data.get("l2", []), dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 4:
        return []
    try:
        import ruptures as rpt  # type: ignore

        model = rpt.Binseg(model="l2").fit(np.nan_to_num(x))
        cps = model.predict(n_bkps=max(1, count))[:-1]
    except Exception:
        # Dependency-free fallback: large shifts in rolling mean.
        w = min(8, max(2, x.shape[0] // 8))
        scores = []
        for i in range(w, x.shape[0] - w):
            left = np.nanmean(x[i - w : i], axis=0)
            right = np.nanmean(x[i : i + w], axis=0)
            scores.append((float(np.linalg.norm(right - left)), i))
        cps = [i for _, i in sorted(scores, reverse=True)[:count]]
    zones = []
    for c in sorted(cps[:count]):
        s = max(0, int(c) - zone_len // 2)
        e = min(x.shape[0] - 1, s + zone_len - 1)
        zones.append((s, e))
    return merge_intervals(zones)


def kernel_changepoint_zones(data: Dict[str, Any], count: int = 3, zone_len: int = 4) -> List[Interval]:
    """Kernelized offline change-point baseline on normalized L2 features."""
    x = np.asarray(data.get("l2", []), dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 4:
        return []
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / np.maximum(std, 1e-6)
    try:
        import ruptures as rpt  # type: ignore

        model = rpt.KernelCPD(kernel="rbf", min_size=max(2, zone_len // 2), jump=1).fit(x)
        cps = model.predict(n_bkps=max(1, count))[:-1]
    except Exception:
        return changepoint_zones(data, count=count, zone_len=zone_len)
    zones = []
    for c in sorted(cps[:count]):
        s = max(0, int(c) - zone_len // 2)
        e = min(x.shape[0] - 1, s + zone_len - 1)
        zones.append((s, e))
    return merge_intervals(zones)


def self_similarity_zones(data: Dict[str, Any], count: int = 3, zone_len: int = 4) -> List[Interval]:
    """Boundary proposal from pose-feature temporal self-similarity novelty.

    This is a lightweight GEBD/UBoCo-style baseline: it does not train a
    boundary-contrastive model, but it uses the same generic premise that event
    boundaries appear as temporal self-similarity discontinuities. Scores are
    high when the feature window before a candidate frame is internally
    coherent, the window after is internally coherent, and the two windows are
    mutually dissimilar.
    """
    x = np.asarray(data.get("l2", []), dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 8:
        return []
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / np.maximum(std, 1e-6)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-6)

    n = x.shape[0]
    window = max(4, min(14, n // 6))
    scores = np.full(n, -np.inf, dtype=np.float32)
    for i in range(window, n - window):
        left = x[i - window : i]
        right = x[i : i + window]
        ll = left @ left.T
        rr = right @ right.T
        lr = left @ right.T
        within = 0.5 * (float(ll.mean()) + float(rr.mean()))
        cross = float(lr.mean())
        scores[i] = within - cross

    picks = []
    radius = max(2, zone_len)
    for idx in np.argsort(-scores):
        i = int(idx)
        if not np.isfinite(scores[i]):
            continue
        if any(abs(i - p) <= radius for p in picks):
            continue
        picks.append(i)
        if len(picks) >= count:
            break
    zones = []
    for c in sorted(picks):
        s = max(0, c - zone_len // 2)
        e = min(n - 1, s + zone_len - 1)
        zones.append((s, e))
    return merge_intervals(zones)


def random_matched_zones(n: int, ref: List[Interval], rng: random.Random) -> List[Interval]:
    out = []
    for s, e in ref:
        length = max(1, e - s + 1)
        if n <= length:
            out.append((0, max(0, n - 1)))
        else:
            rs = rng.randint(0, n - length)
        out.append((rs, rs + length - 1))
    return merge_intervals(out)


def _robust_feature_z(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-clip robust feature normalization for fixed engineered scores."""
    out = np.zeros_like(x, dtype=np.float32)
    valid = mask.astype(bool)
    if x.ndim != 2 or not np.any(valid):
        return out
    vals = x[valid]
    med = np.nanmedian(vals, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(vals - med), axis=0, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-6)
    out = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - med) / scale
    return np.clip(out, -8.0, 8.0).astype(np.float32)


def _idxs(feature_names: List[str], names: List[str]) -> List[int]:
    by_name = {name: i for i, name in enumerate(feature_names)}
    return [by_name[name] for name in names if name in by_name]


def _shift(arr: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(arr)
    if arr.shape[0] > k:
        out[k:] = arr[:-k]
    return out


def _rolling_var_np(x: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    out = np.zeros((x.shape[0],), dtype=np.float32)
    half = max(1, window // 2)
    valid = mask.astype(bool)
    for i in range(x.shape[0]):
        lo = max(0, i - half)
        hi = min(x.shape[0], i + half + 1)
        local_mask = valid[lo:hi]
        if not np.any(local_mask):
            continue
        local = x[lo:hi][local_mask]
        out[i] = float(np.mean(np.var(local, axis=0)))
    return out


def _positive_robust_unit(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map a nonnegative component to roughly [0,1] using valid-frame robust z."""
    out = np.zeros_like(x, dtype=np.float32)
    valid = mask.astype(bool)
    vals = x[valid]
    if vals.size == 0:
        return out
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = max(1.4826 * mad, 1e-6)
    z = (x - med) / scale
    out = np.clip(z, 0.0, 5.0) / 5.0
    out[~valid] = 0.0
    return out.astype(np.float32)


def engineered_l2_instability_score(data: Dict[str, Any], k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed no-model relational-instability score from the 26-D L2 stream.

    This intentionally avoids learned coordinates: no encoder, projection head,
    auxiliary variables, or boundary head are used. The score combines robustly
    normalized feature deltas, local feature variance, stop-go changes, hand and
    reach changes, and critical-distance changes/crossings.
    """
    x = np.asarray(data.get("l2", []), dtype=np.float32)
    if x.ndim != 2:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)
    mask = np.asarray(data.get("l2_mask", [True] * x.shape[0]), dtype=np.bool_)
    if mask.shape[0] != x.shape[0]:
        mask = np.ones((x.shape[0],), dtype=np.bool_)
    feature_names = [str(name) for name in data.get("feature_names", [])]
    if not feature_names or len(feature_names) != x.shape[1]:
        feature_names = [f"f{i}" for i in range(x.shape[1])]
    z = _robust_feature_z(x, mask)

    broad = _idxs(
        feature_names,
        [
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
        ],
    )
    hand_reach = _idxs(
        feature_names,
        ["hand_lead_L", "hand_lead_R", "arm_ext_L", "arm_ext_R", "hand_forward_L", "hand_forward_R", "reach_norm_avg"],
    )
    stopgo = _idxs(feature_names, ["stop_rate_L", "stop_rate_R", "stopgo_transitions_L", "stopgo_transitions_R"])
    pressure = _idxs(feature_names, ["closing_speed", "net_closing", "vL_toward", "vR_toward"])
    critical_ratio_idx = _idxs(feature_names, ["critical_ratio"])
    critical_cross_idx = _idxs(feature_names, ["critical_cross"])

    components: List[np.ndarray] = []
    valid_pair = mask.copy()
    valid_pair[1:] &= mask[:-1]
    valid_k = mask.copy()
    if k > 0:
        valid_k[:k] = False
        valid_k[k:] &= mask[:-k]

    if broad:
        delta = np.zeros((x.shape[0],), dtype=np.float32)
        dz = np.diff(z[:, broad], axis=0, prepend=z[:1, broad])
        delta = np.linalg.norm(dz, axis=1).astype(np.float32)
        delta[~valid_pair] = 0.0
        components.append(delta)
        components.append(_rolling_var_np(z[:, broad], mask, k))

    if hand_reach:
        d = z[:, hand_reach] - _shift(z[:, hand_reach], k)
        comp = np.linalg.norm(d, axis=1).astype(np.float32)
        comp[~valid_k] = 0.0
        components.append(comp)

    if stopgo:
        stop_level = np.maximum(z[:, stopgo], 0.0).mean(axis=1).astype(np.float32)
        stop_delta = np.linalg.norm(z[:, stopgo] - _shift(z[:, stopgo], k), axis=1).astype(np.float32)
        stop_delta[~valid_k] = 0.0
        components.extend([stop_level, stop_delta])

    if pressure:
        # Positive approach/pressure values are activity evidence, not their sign.
        components.append(np.maximum(z[:, pressure], 0.0).mean(axis=1).astype(np.float32))

    if critical_ratio_idx:
        c = z[:, critical_ratio_idx[0]]
        dc = np.abs(c - _shift(c, k)).astype(np.float32)
        dc[~valid_k] = 0.0
        components.append(dc)

    if critical_cross_idx:
        raw_cross = x[:, critical_cross_idx[0]]
        shifted = _shift(raw_cross, k)
        flip = ((raw_cross > 0.5) != (shifted > 0.5)).astype(np.float32)
        flip[~valid_k] = 0.0
        components.append(flip)

    if not components:
        return np.zeros((x.shape[0],), dtype=np.float32), mask
    normalized = [_positive_robust_unit(np.nan_to_num(component, nan=0.0), mask) for component in components]
    score = np.mean(np.stack(normalized, axis=1), axis=1).astype(np.float32)
    score[~mask] = 0.0
    return score, mask


def engineered_l2_instability_zones(data: Dict[str, Any], config: str) -> List[Interval]:
    import yaml  # type: ignore

    l3_dir = Path("boundary_segmentation").resolve()
    if str(l3_dir) not in sys.path:
        sys.path.insert(0, str(l3_dir))
    from postprocess.zones import extract_transition_zones  # type: ignore

    with open(config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    score, mask = engineered_l2_instability_score(data, k=int(cfg["model"]["boundary"].get("k", 8)))
    pp = cfg["postprocess"]
    return [
        (int(start), int(end))
        for start, end in extract_transition_zones(
            score,
            mask,
            alpha=pp["ema_alpha"],
            thr_on=pp["thr_on"],
            min_len=pp["min_len"],
            merge_gap=pp["merge_gap"],
            mode=pp.get("mode", "robust_hysteresis"),
            thr_off=pp.get("thr_off"),
            abs_on=pp.get("abs_on", 0.0),
            abs_off=pp.get("abs_off"),
            robust_z_on=pp.get("robust_z_on", 1.0),
            robust_z_off=pp.get("robust_z_off", 0.4),
            min_prominence=pp.get("min_prominence", 0.0),
            min_peak_height=pp.get("min_peak_height", 0.0),
            boundary_frac_min=pp.get("boundary_frac_min", 0.0),
            boundary_frac_max=pp.get("boundary_frac_max", 1.0),
            budget_step=pp.get("budget_step", 0.15),
            max_budget_iters=pp.get("max_budget_iters", 3),
            onset_backtrack=pp.get("onset_backtrack", 0),
            split_min_zone_len=pp.get("split_min_zone_len", 0),
            split_min_rel_drop=pp.get("split_min_rel_drop", 0.0),
            split_max_splits=pp.get("split_max_splits", 0),
            micro_merge_zone_len=pp.get("micro_merge_zone_len", 0),
            micro_merge_gap=pp.get("micro_merge_gap", 0),
            micro_merge_max_span=pp.get("micro_merge_max_span", 0),
        )
    ]


def run_model_infer_in_process(l2_path: str, config: str, ckpt: str, zero_critical: bool = False) -> List[Interval]:
    import torch  # type: ignore
    import yaml  # type: ignore

    l3_dir = Path("boundary_segmentation").resolve()
    if str(l3_dir) not in sys.path:
        sys.path.insert(0, str(l3_dir))
    from infer import infer_sequence  # type: ignore
    from models.l3_model import L3BoundaryModel  # type: ignore

    data = read_sequence(l2_path)
    x = np.asarray(data.get("l2", []), dtype=np.float32)
    mask = np.asarray(data.get("l2_mask", []), dtype=np.bool_)
    if zero_critical:
        feature_names = data.get("feature_names", [])
        indexes = [i for i, name in enumerate(feature_names) if name in {"critical_ratio", "critical_delta", "critical_cross"}]
        x = x.copy()
        for index in indexes:
            if index < x.shape[1]:
                x[:, index] = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (str(Path(config).resolve()), str(Path(ckpt).resolve()), int(x.shape[1]), device)
    cached = _MODEL_CACHE.get(key)
    if cached is None:
        with open(config, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        model = L3BoundaryModel(input_dim=x.shape[1], cfg=cfg).to(device)
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        cached = (model, cfg)
        _MODEL_CACHE[key] = cached
    model, cfg = cached
    result = infer_sequence(x, mask, model, cfg)
    return [(int(start), int(end)) for start, end in result.get("transition_zones", [])]


def run_model_score_infer_in_process(
    l2_path: str,
    config: str,
    ckpt: str,
    score_source: str,
    zero_critical: bool = False,
) -> List[Interval]:
    import torch  # type: ignore
    import yaml  # type: ignore

    l3_dir = Path("boundary_segmentation").resolve()
    if str(l3_dir) not in sys.path:
        sys.path.insert(0, str(l3_dir))
    from models.l3_model import L3BoundaryModel  # type: ignore
    from postprocess.zones import extract_transition_zones  # type: ignore

    data = read_sequence(l2_path)
    x = np.asarray(data.get("l2", []), dtype=np.float32)
    mask = np.asarray(data.get("l2_mask", []), dtype=np.bool_)
    if zero_critical:
        feature_names = data.get("feature_names", [])
        indexes = [i for i, name in enumerate(feature_names) if name in {"critical_ratio", "critical_delta", "critical_cross"}]
        x = x.copy()
        for index in indexes:
            if index < x.shape[1]:
                x[:, index] = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (str(Path(config).resolve()), str(Path(ckpt).resolve()), int(x.shape[1]), device)
    cached = _MODEL_CACHE.get(key)
    if cached is None:
        with open(config, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        model = L3BoundaryModel(input_dim=x.shape[1], cfg=cfg).to(device)
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        cached = (model, cfg)
        _MODEL_CACHE[key] = cached
    model, cfg = cached
    with torch.no_grad():
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask.astype(np.bool_)).unsqueeze(0).to(device)
        _, _, _, phi, b = model(x_t, mask_t)
    phi_np = phi.squeeze(0).cpu().numpy()
    b_np = b.squeeze(0).cpu().numpy()
    if score_source == "raw_phi_l2":
        score = np.linalg.norm(phi_np, axis=-1)
    elif score_source == "raw_phi_mean":
        score = phi_np.mean(axis=-1)
    elif score_source == "learned_b":
        score = b_np
    else:
        raise ValueError(f"unknown score_source {score_source}")
    pp = cfg["postprocess"]
    return [
        (int(start), int(end))
        for start, end in extract_transition_zones(
            score,
            mask,
            alpha=pp["ema_alpha"],
            thr_on=pp["thr_on"],
            min_len=pp["min_len"],
            merge_gap=pp["merge_gap"],
            mode=pp.get("mode", "robust_hysteresis"),
            thr_off=pp.get("thr_off"),
            abs_on=pp.get("abs_on", 0.0),
            abs_off=pp.get("abs_off"),
            robust_z_on=pp.get("robust_z_on", 1.0),
            robust_z_off=pp.get("robust_z_off", 0.4),
            min_prominence=pp.get("min_prominence", 0.0),
            min_peak_height=pp.get("min_peak_height", 0.0),
            boundary_frac_min=pp.get("boundary_frac_min", 0.0),
            boundary_frac_max=pp.get("boundary_frac_max", 1.0),
            budget_step=pp.get("budget_step", 0.15),
            max_budget_iters=pp.get("max_budget_iters", 3),
            onset_backtrack=pp.get("onset_backtrack", 0),
            split_min_zone_len=pp.get("split_min_zone_len", 0),
            split_min_rel_drop=pp.get("split_min_rel_drop", 0.0),
            split_max_splits=pp.get("split_max_splits", 0),
            micro_merge_zone_len=pp.get("micro_merge_zone_len", 0),
            micro_merge_gap=pp.get("micro_merge_gap", 0),
            micro_merge_max_span=pp.get("micro_merge_max_span", 0),
        )
    ]


def run_model_infer(l2_path: str, config: str, ckpt: str, python_bin: str, zero_critical: bool = False) -> List[Interval]:
    if Path(python_bin).resolve() == Path(sys.executable).resolve():
        return run_model_infer_in_process(l2_path, config, ckpt, zero_critical=zero_critical)
    inp = Path(l2_path)
    use_path = inp
    tmp_dir_obj = None
    if zero_critical:
        data = read_sequence(inp)
        feature_names = data.get("feature_names", [])
        idxs = [i for i, n in enumerate(feature_names) if n in {"critical_ratio", "critical_delta", "critical_cross"}]
        for row in data.get("l2", []):
            for i in idxs:
                if i < len(row):
                    row[i] = 0.0
        tmp_dir_obj = tempfile.TemporaryDirectory()
        use_path = Path(tmp_dir_obj.name) / inp.name
        write_json(use_path, data)
    out_path = Path(tempfile.mkdtemp(prefix="boundary_pred_")) / "out.json"
    cmd = [
        python_bin,
        "infer.py",
        "--config",
        str(Path(config).resolve()),
        "--ckpt",
        str(Path(ckpt).resolve()),
        "--input",
        str(use_path.resolve()),
        "--output",
        str(out_path),
        "--device",
        "auto",
    ]
    subprocess.run(cmd, cwd=str(Path("boundary_segmentation").resolve()), check=True)
    payload = read_json(out_path)
    if tmp_dir_obj is not None:
        tmp_dir_obj.cleanup()
    return [(int(s), int(e)) for s, e in payload.get("transition_zones", [])]


def predictions_for_method(args, method: str, rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    rng = random.Random(args.seed)
    full_cache: Dict[str, List[Interval]] = {}
    if args.reference_pred:
        payload = read_json(args.reference_pred)
        for pred in payload.get("predictions", []):
            raw_zones = pred.get("transition_zones_index") or pred.get("transition_zones") or []
            full_cache[str(pred.get("clip_id"))] = [(int(s), int(e)) for s, e in raw_zones]
    preds: List[Dict[str, Any]] = []
    for row in rows:
        clip_id = row["clip_id"]
        data = load_l2(row["l2_path"])
        n = seq_len(row, data)
        if method == "uniform":
            zones = uniform_zones(n, args.zone_count, args.zone_len)
        elif method == "motion":
            zones = motion_energy_zones(data, args.zone_count, args.zone_len)
        elif method == "changepoint":
            zones = changepoint_zones(data, args.zone_count, args.zone_len)
        elif method == "kernel_changepoint":
            zones = kernel_changepoint_zones(data, args.zone_count, args.zone_len)
        elif method == "selfsim":
            zones = self_similarity_zones(data, args.zone_count, args.zone_len)
        elif method == "engineered_l2_instability":
            zones = engineered_l2_instability_zones(data, args.config)
        elif method == "random_matched":
            ref = full_cache.get(clip_id)
            if ref is None:
                ref = run_model_infer(row["l2_path"], args.config, args.ckpt, args.python_bin)
                full_cache[clip_id] = ref
            zones = random_matched_zones(n, ref, rng)
        elif method == "full_model":
            zones = run_model_infer(row["l2_path"], args.config, args.ckpt, args.python_bin)
        elif method == "no_critical":
            zones = run_model_infer(row["l2_path"], args.config, args.ckpt, args.python_bin, zero_critical=True)
        elif method == "raw_phi_l2":
            zones = run_model_score_infer_in_process(row["l2_path"], args.config, args.ckpt, "raw_phi_l2")
        else:
            raise ValueError(f"unknown method {method}")
        preds.append(
            {
                "clip_id": clip_id,
                "method": method,
                "coordinate_system": "source_video_frame_inclusive",
                "transition_zones_index": zones,
                "state_segments": partition_states(data, zones, n),
            }
        )
        preds[-1]["active_segments"] = [
            [state["start_frame"], state["end_frame"]]
            for state in preds[-1]["state_segments"]
            if state["activity"] == "active"
        ]
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate boundary predictions.")
    parser.add_argument("--manifest", default="paper_experiments/manifests/foil_main.csv")
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        choices=[
            "uniform",
            "motion",
            "changepoint",
            "kernel_changepoint",
            "selfsim",
            "engineered_l2_instability",
            "random_matched",
            "full_model",
            "no_critical",
            "raw_phi_l2",
        ],
    )
    parser.add_argument("--out-dir", default="paper_experiments/experiments/boundary_predictions")
    parser.add_argument("--config", default="deployment/checkpoints/legacy_auto/config.yaml")
    parser.add_argument("--ckpt", default="deployment/checkpoints/legacy_auto/ckpt_best_ema.pt")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--zone-count", type=int, default=3)
    parser.add_argument("--zone-len", type=int, default=4)
    parser.add_argument("--reference-pred", default="", help="Existing prediction JSON for random_matched reference zones")
    parser.add_argument("--seed", type=int, default=20270726)
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in args.method:
        preds = predictions_for_method(args, method, rows)
        out = out_dir / f"{method}.json"
        write_json(
            out,
            {
                "method": method,
                "manifest": args.manifest,
                "config": args.config,
                "ckpt": args.ckpt,
                "reference_pred": args.reference_pred,
                "seed": args.seed,
                "schema_version": 2,
                "prediction_task": "interaction_state_partition",
                "coordinate_system": "source_video_frame_inclusive",
                "predictions": preds,
            },
        )
        print(f"Wrote {out} predictions={len(preds)}")


if __name__ == "__main__":
    main()
