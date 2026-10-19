"""Evaluation metrics for boundary outputs."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _interval_lengths(intervals: List[Tuple[int, int]]) -> List[int]:
    return [int(e - s + 1) for s, e in intervals]


def summarize_zones(
    b_mean: float,
    transition_zones: List[List[Tuple[int, int]]],
    stable_segments: List[List[Tuple[int, int]]],
) -> Dict[str, float]:
    zone_counts = [len(z) for z in transition_zones]
    zone_lens = [l for z in transition_zones for l in _interval_lengths(z)]
    seg_lens = [l for s in stable_segments for l in _interval_lengths(s)]

    def _avg(x: List[int]) -> float:
        return float(np.mean(x)) if x else 0.0

    return {
        "mean_boundary_mass": float(b_mean),
        "avg_zones_per_clip": float(np.mean(zone_counts)) if zone_counts else 0.0,
        "avg_zone_length": _avg(zone_lens),
        "avg_stable_length": _avg(seg_lens),
    }
