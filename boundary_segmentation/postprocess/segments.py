"""Stable segment extraction."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _complement(intervals: List[Tuple[int, int]], start: int, end: int) -> List[Tuple[int, int]]:
    if start > end:
        return []
    if not intervals:
        return [(start, end)]
    intervals = sorted(intervals)
    segs: List[Tuple[int, int]] = []
    cur = start
    for s, e in intervals:
        if s > cur:
            segs.append((cur, s - 1))
        cur = max(cur, e + 1)
    if cur <= end:
        segs.append((cur, end))
    return segs


def stable_segments(
    mask: np.ndarray,
    transition_zones: List[Tuple[int, int]],
    min_seg_len: int = 8,
) -> List[Tuple[int, int]]:
    if mask is None or mask.size == 0:
        return []
    idx = np.where(mask.astype(bool))[0]
    if idx.size == 0:
        return []
    start = int(idx[0])
    end = int(idx[-1])
    segs = _complement(transition_zones, start, end)
    segs = [(s, e) for s, e in segs if (e - s + 1) >= min_seg_len]
    return segs
