#!/usr/bin/env python3
"""Evaluate a boundary candidate by camera motion and as a gated router."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import read_csv, read_json, write_csv, write_json
from search_boundary_postprocess import (
    evaluate_rows,
    load_annotations,
    load_cache,
    objective_summary,
)


METRICS = [
    "objective",
    "activity_macro_f1",
    "active_f1",
    "boundary_f1_pm5",
    "boundary_f1_pm10",
    "total_state_count_ratio",
]
CLIP_METRICS = [
    "activity_macro_f1",
    "active_f1",
    "boundary_f1_pm5",
    "boundary_f1_pm10",
    "state_count_ratio",
]


def camera_pan_p95(camera: Dict[str, Any]) -> float:
    magnitudes = [
        math.hypot(float(row.get("pan_x_norm", 0.0)), float(row.get("pan_y_norm", 0.0)))
        for row in camera.get("motions", [])
        if row.get("valid", False)
    ]
    return float(np.quantile(magnitudes, 0.95)) if magnitudes else 0.0


def evaluate_each(
    annotations: Dict[str, dict],
    cache: Dict[str, Dict[str, np.ndarray]],
    postprocess: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        clip_id: evaluate_rows(
            {clip_id: annotation},
            [{clip_id: cache[clip_id]}],
            [1.0],
            postprocess,
        )[0]
        for clip_id, annotation in annotations.items()
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return objective_summary(rows) if rows else {"clips": 0}


def bootstrap_difference(
    candidate: List[Dict[str, Any]],
    baseline: List[Dict[str, Any]],
    iterations: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    if len(candidate) != len(baseline):
        raise ValueError("Candidate and baseline rows must be paired")
    rng = random.Random(seed)
    samples = {metric: [] for metric in METRICS}
    for _ in range(iterations):
        indices = [rng.randrange(len(candidate)) for _ in candidate]
        candidate_summary = summarize([candidate[index] for index in indices])
        baseline_summary = summarize([baseline[index] for index in indices])
        for metric in METRICS:
            samples[metric].append(
                float(candidate_summary[metric]) - float(baseline_summary[metric])
            )
    output = {}
    observed_candidate = summarize(candidate)
    observed_baseline = summarize(baseline)
    for metric, values in samples.items():
        output[metric] = {
            "difference": float(observed_candidate[metric]) - float(observed_baseline[metric]),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "p_gt_zero": sum(value > 0.0 for value in values) / len(values),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--owner-dir", required=True)
    parser.add_argument("--camera-motion", required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--baseline-result", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--pan-threshold", type=float, default=0.01)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20270713)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    manifest_ids = {row["clip_id"] for row in read_csv(args.manifest)}
    annotations = {
        clip_id: annotation
        for clip_id, annotation in load_annotations(Path(args.owner_dir)).items()
        if clip_id in manifest_ids
    }
    camera_motion = read_json(args.camera_motion)
    missing_camera = sorted(set(annotations) - set(camera_motion))
    if missing_camera:
        raise ValueError(f"Missing camera estimates for {len(missing_camera)} clips")

    baseline_cache = load_cache(Path(args.baseline_cache), annotations)
    candidate_cache = load_cache(Path(args.candidate_cache), annotations)
    baseline_pp = read_json(args.baseline_result)["best"]["postprocess"]
    candidate_pp = read_json(args.candidate_result)["best"]["postprocess"]
    baseline_rows = evaluate_each(annotations, baseline_cache, baseline_pp)
    candidate_rows = evaluate_each(annotations, candidate_cache, candidate_pp)
    pan = {clip_id: camera_pan_p95(camera_motion[clip_id]) for clip_id in annotations}
    high_ids = sorted(
        clip_id for clip_id, score in pan.items() if score >= args.pan_threshold
    )
    low_ids = sorted(set(annotations) - set(high_ids))
    all_ids = sorted(annotations)
    router_rows = {
        clip_id: candidate_rows[clip_id] if clip_id in high_ids else baseline_rows[clip_id]
        for clip_id in all_ids
    }

    cohorts = {"all": all_ids, "high_pan": high_ids, "low_pan": low_ids}
    summaries = {}
    for cohort, clip_ids in cohorts.items():
        summaries[cohort] = {
            "baseline": summarize([baseline_rows[clip_id] for clip_id in clip_ids]),
            "candidate": summarize([candidate_rows[clip_id] for clip_id in clip_ids]),
            "router": summarize([router_rows[clip_id] for clip_id in clip_ids]),
        }

    router_bootstrap = bootstrap_difference(
        [router_rows[clip_id] for clip_id in all_ids],
        [baseline_rows[clip_id] for clip_id in all_ids],
        args.bootstrap_iterations,
        args.seed,
    )
    candidate_bootstrap = bootstrap_difference(
        [candidate_rows[clip_id] for clip_id in all_ids],
        [baseline_rows[clip_id] for clip_id in all_ids],
        args.bootstrap_iterations,
        args.seed + 1,
    )
    output = {
        "manifest": str(Path(args.manifest).resolve()),
        "camera_motion": str(Path(args.camera_motion).resolve()),
        "pan_metric": "95th percentile per-frame normalized translation magnitude",
        "pan_threshold": args.pan_threshold,
        "high_pan_clips": high_ids,
        "summaries": summaries,
        "router_vs_baseline_bootstrap": router_bootstrap,
        "candidate_vs_baseline_bootstrap": candidate_bootstrap,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
    }
    write_json(args.out_json, output)

    csv_rows = []
    for clip_id in all_ids:
        row: Dict[str, Any] = {
            "clip_id": clip_id,
            "pan_p95": pan[clip_id],
            "high_pan": clip_id in high_ids,
            "router_model": "candidate" if clip_id in high_ids else "baseline",
        }
        for model_name, model_rows in (
            ("baseline", baseline_rows),
            ("candidate", candidate_rows),
            ("router", router_rows),
        ):
            for metric in CLIP_METRICS:
                row[f"{model_name}_{metric}"] = model_rows[clip_id][metric]
        csv_rows.append(row)
    write_csv(args.out_csv, csv_rows)
    print(
        f"Evaluated {len(all_ids)} clips; high_pan={len(high_ids)}; "
        f"router_objective={summaries['all']['router']['objective']:.4f}"
    )


if __name__ == "__main__":
    main()
