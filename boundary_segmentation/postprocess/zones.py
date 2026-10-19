"""Transition zone extraction."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from postprocess.smooth import ema_smooth


def _extract_intervals(active: np.ndarray) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    if active.size == 0:
        return intervals
    in_seg = False
    start = 0
    for i, v in enumerate(active):
        if v and not in_seg:
            in_seg = True
            start = i
        elif not v and in_seg:
            intervals.append((start, i - 1))
            in_seg = False
    if in_seg:
        intervals.append((start, active.size - 1))
    return intervals


def _drop_short(intervals: List[Tuple[int, int]], min_len: int) -> List[Tuple[int, int]]:
    return [(s, e) for s, e in intervals if (e - s + 1) >= min_len]


def _merge_gaps(intervals: List[Tuple[int, int]], merge_gap: int) -> List[Tuple[int, int]]:
    if not intervals:
        return intervals
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e - 1 <= merge_gap:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged


def _backtrack_onsets(
    intervals: List[Tuple[int, int]],
    valid: np.ndarray,
    backtrack: int,
    ref: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    if backtrack <= 0 or not intervals:
        return intervals
    n = valid.size
    out: List[Tuple[int, int]] = []
    for s, e in intervals:
        lo = max(0, s - backtrack)
        hi = s + 1
        s2 = lo
        # Prefer the local valley before the detected onset to reduce lag.
        if ref is not None:
            cand = [i for i in range(lo, hi) if i < n and valid[i]]
            if cand:
                s2 = min(cand, key=lambda i: float(ref[i]))
        # Fallback: avoid starting in invalid region if masks are discontinuous.
        while s2 < s and (s2 >= n or not valid[s2]):
            s2 += 1
        if s2 > e:
            s2 = s
        out.append((s2, e))
    return out


def _build_hysteresis_active(on: np.ndarray, off: np.ndarray, valid: np.ndarray) -> np.ndarray:
    active = np.zeros_like(valid, dtype=bool)
    in_seg = False
    for i in range(valid.size):
        if not valid[i]:
            in_seg = False
            continue
        if not in_seg:
            if on[i]:
                in_seg = True
                active[i] = True
        else:
            if off[i]:
                active[i] = True
            else:
                in_seg = False
    return active


def _robust_z(x: np.ndarray, valid: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    z = np.zeros_like(x, dtype=np.float32)
    vals = x[valid]
    if vals.size == 0:
        return z
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = max(1.4826 * mad, eps)
    z = (x - med) / scale
    return z.astype(np.float32)


def _drop_low_prominence(
    x: np.ndarray,
    intervals: List[Tuple[int, int]],
    min_prominence: float,
    min_peak_height: float,
    window: int,
) -> List[Tuple[int, int]]:
    if not intervals:
        return intervals
    if min_prominence <= 0.0 and min_peak_height <= 0.0:
        return intervals
    out: List[Tuple[int, int]] = []
    n = x.shape[0]
    for s, e in intervals:
        seg = x[s : e + 1]
        if seg.size == 0:
            continue
        peak = float(np.max(seg))
        if peak < min_peak_height:
            continue
        ls = max(0, s - window)
        le = s
        rs = e + 1
        re = min(n, e + 1 + window)
        left_min = float(np.min(x[ls:le])) if le > ls else float(x[s])
        right_min = float(np.min(x[rs:re])) if re > rs else float(x[e])
        baseline = max(left_min, right_min)
        prominence = peak - baseline
        if prominence >= min_prominence:
            out.append((s, e))
    return out


def _coverage(intervals: List[Tuple[int, int]], valid_count: int) -> float:
    if valid_count <= 0:
        return 0.0
    total = 0
    for s, e in intervals:
        total += max(0, e - s + 1)
    return float(total / valid_count)


def _split_by_internal_valleys(
    x: np.ndarray,
    intervals: List[Tuple[int, int]],
    min_len: int,
    split_min_zone_len: int,
    split_min_rel_drop: float,
    split_max_splits: int,
) -> List[Tuple[int, int]]:
    """Split long intervals at deep internal valleys between two peaks.

    rel_drop = (min(left_peak, right_peak) - valley) / (seg_max - seg_min + eps)
    """
    if not intervals or split_max_splits <= 0 or split_min_rel_drop <= 0.0:
        return intervals

    split_min_zone_len = max(split_min_zone_len, min_len)
    out: List[Tuple[int, int]] = []
    eps = 1e-6

    for s0, e0 in intervals:
        parts: List[Tuple[int, int]] = [(s0, e0)]
        splits_done = 0
        while splits_done < split_max_splits:
            best_j = -1
            best_i = -1
            best_score = split_min_rel_drop
            for j, (s, e) in enumerate(parts):
                if (e - s + 1) < (2 * split_min_zone_len + 1):
                    continue
                lo = s + split_min_zone_len
                hi = e - split_min_zone_len
                if lo > hi:
                    continue
                seg = x[s : e + 1]
                seg_max = float(np.max(seg))
                seg_min = float(np.min(seg))
                seg_range = max(seg_max - seg_min, eps)
                for i in range(lo, hi + 1):
                    valley = float(x[i])
                    left_peak = float(np.max(x[s : i + 1]))
                    right_peak = float(np.max(x[i : e + 1]))
                    shoulder = min(left_peak, right_peak)
                    rel_drop = (shoulder - valley) / seg_range
                    if rel_drop > best_score:
                        best_score = rel_drop
                        best_j = j
                        best_i = i
            if best_j < 0 or best_i < 0:
                break
            s, e = parts[best_j]
            left = (s, best_i)
            right = (best_i + 1, e)
            if (left[1] - left[0] + 1) < split_min_zone_len or (right[1] - right[0] + 1) < split_min_zone_len:
                break
            parts = parts[:best_j] + [left, right] + parts[best_j + 1 :]
            splits_done += 1
        out.extend(parts)
    return _drop_short(sorted(out), min_len)


def _merge_micro_successions(
    intervals: List[Tuple[int, int]],
    micro_max_len: int,
    micro_gap: int,
    micro_max_span: int,
) -> List[Tuple[int, int]]:
    """Merge chains of tiny adjacent transition zones.

    This is meant to clean up successions like:
    transition / tiny stable gap / transition / tiny stable gap / transition
    where each individual transition is too short to support a reliable
    standalone state read.
    """
    if not intervals or micro_max_len <= 0 or micro_gap < 0 or micro_max_span <= 0:
        return intervals

    merged: List[Tuple[int, int]] = []
    i = 0
    while i < len(intervals):
        s0, e0 = intervals[i]
        if (e0 - s0 + 1) > micro_max_len:
            merged.append((s0, e0))
            i += 1
            continue

        chain_start = s0
        chain_end = e0
        chain_count = 1
        j = i + 1
        while j < len(intervals):
            s1, e1 = intervals[j]
            gap = s1 - chain_end - 1
            zone_len = e1 - s1 + 1
            span = e1 - chain_start + 1
            if gap > micro_gap or zone_len > micro_max_len or span > micro_max_span:
                break
            chain_end = e1
            chain_count += 1
            j += 1

        if chain_count >= 2:
            merged.append((chain_start, chain_end))
            i = j
        else:
            merged.append((s0, e0))
            i += 1
    return merged


def _extract_with_params(
    b_s: np.ndarray,
    valid: np.ndarray,
    mode: str,
    thr_on: float,
    thr_off: float,
    abs_on: float,
    abs_off: float,
    robust_z_on: float,
    robust_z_off: float,
    min_len: int,
    merge_gap: int,
    min_prominence: float,
    min_peak_height: float,
    onset_backtrack: int,
    split_min_zone_len: int,
    split_min_rel_drop: float,
    split_max_splits: int,
    micro_merge_zone_len: int,
    micro_merge_gap: int,
    micro_merge_max_span: int,
) -> List[Tuple[int, int]]:
    if mode == "robust_hysteresis":
        z = _robust_z(b_s, valid)
        on = (z >= robust_z_on) & (b_s >= abs_on)
        off = (z >= robust_z_off) & (b_s >= abs_off)
        active = _build_hysteresis_active(on, off, valid)
        prom_series = z
    else:
        on = b_s >= thr_on
        off = b_s >= thr_off
        active = _build_hysteresis_active(on, off, valid)
        prom_series = b_s
    intervals = _extract_intervals(active)
    intervals = _drop_short(intervals, min_len)
    intervals = _merge_gaps(intervals, merge_gap)
    intervals = _drop_low_prominence(
        prom_series,
        intervals,
        min_prominence=min_prominence,
        min_peak_height=min_peak_height,
        window=max(6, 2 * min_len),
    )
    intervals = _backtrack_onsets(intervals, valid, onset_backtrack, ref=b_s)
    intervals = _merge_gaps(intervals, merge_gap)
    intervals = _split_by_internal_valleys(
        b_s,
        intervals,
        min_len=min_len,
        split_min_zone_len=split_min_zone_len,
        split_min_rel_drop=split_min_rel_drop,
        split_max_splits=split_max_splits,
    )
    intervals = _merge_micro_successions(
        intervals,
        micro_max_len=micro_merge_zone_len,
        micro_gap=micro_merge_gap,
        micro_max_span=micro_merge_max_span,
    )
    return intervals


def extract_transition_zones(
    b: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
    thr_on: float = 0.6,
    min_len: int = 5,
    merge_gap: int = 3,
    mode: str = "legacy",
    thr_off: Optional[float] = None,
    abs_on: float = 0.0,
    abs_off: Optional[float] = None,
    robust_z_on: float = 1.0,
    robust_z_off: float = 0.4,
    min_prominence: float = 0.0,
    min_peak_height: float = 0.0,
    boundary_frac_min: float = 0.0,
    boundary_frac_max: float = 1.0,
    budget_step: float = 0.15,
    max_budget_iters: int = 3,
    onset_backtrack: int = 0,
    split_min_zone_len: int = 0,
    split_min_rel_drop: float = 0.0,
    split_max_splits: int = 0,
    micro_merge_zone_len: int = 0,
    micro_merge_gap: int = 0,
    micro_merge_max_span: int = 0,
) -> List[Tuple[int, int]]:
    b_s = ema_smooth(b.astype(np.float32), alpha=alpha)
    valid = mask.astype(bool) if mask is not None else np.ones_like(b_s, dtype=bool)
    if thr_off is None:
        thr_off = max(0.0, 0.6 * thr_on)
    if abs_off is None:
        abs_off = max(0.0, 0.7 * abs_on)
    intervals = _extract_with_params(
        b_s=b_s,
        valid=valid,
        mode=mode,
        thr_on=thr_on,
        thr_off=thr_off,
        abs_on=abs_on,
        abs_off=abs_off,
        robust_z_on=robust_z_on,
        robust_z_off=robust_z_off,
        min_len=min_len,
        merge_gap=merge_gap,
        min_prominence=min_prominence,
        min_peak_height=min_peak_height,
        onset_backtrack=onset_backtrack,
        split_min_zone_len=split_min_zone_len,
        split_min_rel_drop=split_min_rel_drop,
        split_max_splits=split_max_splits,
        micro_merge_zone_len=micro_merge_zone_len,
        micro_merge_gap=micro_merge_gap,
        micro_merge_max_span=micro_merge_max_span,
    )
    # Optional budget clamp to avoid too sparse/too dense boundary coverage.
    valid_count = int(np.sum(valid))
    if valid_count <= 0:
        return intervals
    for _ in range(max_budget_iters):
        frac = _coverage(intervals, valid_count)
        changed = False
        if frac < boundary_frac_min:
            if mode == "robust_hysteresis":
                robust_z_on -= budget_step
                robust_z_off -= budget_step
            else:
                thr_on = max(0.0, thr_on - budget_step)
                thr_off = max(0.0, thr_off - budget_step)
            changed = True
        elif frac > boundary_frac_max:
            if mode == "robust_hysteresis":
                robust_z_on += budget_step
                robust_z_off += budget_step
            else:
                thr_on += budget_step
                thr_off += budget_step
            changed = True
        if not changed:
            break
        intervals = _extract_with_params(
            b_s=b_s,
            valid=valid,
            mode=mode,
            thr_on=thr_on,
            thr_off=thr_off,
            abs_on=abs_on,
            abs_off=abs_off,
            robust_z_on=robust_z_on,
            robust_z_off=robust_z_off,
            min_len=min_len,
            merge_gap=merge_gap,
            min_prominence=min_prominence,
            min_peak_height=min_peak_height,
            onset_backtrack=onset_backtrack,
            split_min_zone_len=split_min_zone_len,
            split_min_rel_drop=split_min_rel_drop,
            split_max_splits=split_max_splits,
            micro_merge_zone_len=micro_merge_zone_len,
            micro_merge_gap=micro_merge_gap,
            micro_merge_max_span=micro_merge_max_span,
        )
    return intervals
