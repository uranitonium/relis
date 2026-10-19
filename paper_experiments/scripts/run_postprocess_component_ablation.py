#!/usr/bin/env python3
"""Evaluate postprocessing component ablations for a frozen boundary score."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_csv, write_csv  # noqa: E402
from search_boundary_postprocess import (  # noqa: E402
    cache_scores,
    evaluate_rows,
    load_annotations,
    load_cache,
    objective_summary,
)


def variant_configs(base: dict) -> list[tuple[str, dict]]:
    variants: list[tuple[str, dict]] = []
    variants.append(("full", dict(base)))

    pp = dict(base)
    pp["split_max_splits"] = 0
    pp["split_min_zone_len"] = 0
    pp["split_min_rel_drop"] = 0.0
    variants.append(("no_valley_split", pp))

    pp = dict(base)
    pp["boundary_frac_min"] = 0.0
    pp["boundary_frac_max"] = 1.0
    pp["max_budget_iters"] = 0
    variants.append(("no_coverage_clamp", pp))

    pp = dict(base)
    pp["onset_backtrack"] = 0
    variants.append(("no_onset_backtrack", pp))

    pp = dict(base)
    pp["merge_gap"] = 0
    variants.append(("no_gap_merge", pp))

    pp = dict(base)
    pp.update(
        {
            "mode": "legacy",
            "thr_on": 0.5,
            "thr_off": 0.25,
            "abs_on": 0.0,
            "abs_off": 0.0,
            "robust_z_on": 1.0,
            "robust_z_off": 0.4,
            "boundary_frac_min": 0.0,
            "boundary_frac_max": 1.0,
            "max_budget_iters": 0,
            "split_max_splits": 0,
            "onset_backtrack": 0,
        }
    )
    variants.append(("simple_legacy_hysteresis", pp))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv")
    parser.add_argument("--annotation-dir", default="paper_experiments/annotations/adjudicated/consensus")
    parser.add_argument("--config", default="paper_experiments/experiments/truncated_retraining/candidates/optimistic_smooth02_seed18/config.yaml")
    parser.add_argument("--ckpt", default="paper_experiments/experiments/truncated_retraining/runs/short_sparse_seed18_smooth02/ckpt_step_1000.pt")
    parser.add_argument("--out", default="paper_experiments/experiments/final_boundary_evaluation/postprocess_component_ablation.csv")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    annotations = load_annotations(Path(args.annotation_dir))
    rows = [row for row in read_csv(args.manifest) if row["clip_id"] in annotations]
    cache_path = cache_scores(
        "postprocess_component_ablation",
        cfg_path,
        ckpt_path,
        rows,
        Path(args.out).resolve().parent / "score_cache",
    )
    cache = load_cache(cache_path, annotations.keys())

    out_rows = []
    for name, pp in variant_configs(dict(cfg["postprocess"])):
        summary = objective_summary(evaluate_rows(annotations, [cache], [1.0], pp))
        out_rows.append(
            {
                "variant": name,
                "clips": summary["clips"],
                "activity_accuracy": summary["activity_accuracy"],
                "activity_macro_f1": summary["activity_macro_f1"],
                "active_f1": summary["active_f1"],
                "boundary_f1_pm5": summary["boundary_f1_pm5"],
                "boundary_f1_pm10": summary["boundary_f1_pm10"],
                "model_active_coverage": summary["model_active_coverage"],
                "model_states": summary["model_states"],
                "objective": summary["objective"],
            }
        )
    write_csv(args.out, out_rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
