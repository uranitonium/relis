#!/usr/bin/env python3
"""Cross-validate boundary postprocessing on the labeled development clips."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_csv, write_csv, write_json  # noqa: E402
from search_boundary_postprocess import (  # noqa: E402
    cache_scores,
    deployed_postprocess,
    evaluate_rows,
    load_annotations,
    load_cache,
    model_spec,
    objective_summary,
    sample_config,
)


def make_folds(annotations: Dict[str, dict], count: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    by_outcome: Dict[str, List[str]] = defaultdict(list)
    for clip_id, annotation in annotations.items():
        by_outcome[str(annotation.get("outcome_label", "Unknown"))].append(clip_id)
    folds: List[List[str]] = [[] for _ in range(count)]
    for clip_ids in by_outcome.values():
        rng.shuffle(clip_ids)
        for index, clip_id in enumerate(clip_ids):
            folds[index % count].append(clip_id)
    return [sorted(fold) for fold in folds]


def subset(rows: List[dict], clip_ids: set) -> List[dict]:
    return [row for row in rows if row["clip_id"] in clip_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=model_spec, required=True)
    parser.add_argument("--manifest", default="paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv")
    parser.add_argument("--owner-dir", default="paper_experiments/annotations/raw/owner")
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20270614)
    parser.add_argument("--out-dir", default="paper_experiments/experiments/truncated_retraining/cross_validation")
    args = parser.parse_args()

    name, config, checkpoint = args.model
    annotations = load_annotations(Path(args.owner_dir))
    manifest_rows = [row for row in read_csv(args.manifest) if row["clip_id"] in annotations]
    out_dir = Path(args.out_dir) / name
    cache_path = cache_scores(name, config, checkpoint, manifest_rows, out_dir / "score_cache")
    cache = load_cache(cache_path, annotations.keys())
    folds = make_folds(annotations, args.folds, args.seed)
    rng = random.Random(args.seed)
    configs = [deployed_postprocess()] + [sample_config(rng) for _ in range(args.trials)]

    baseline_rows = evaluate_rows(annotations, [cache], [1.0], deployed_postprocess())
    best = [None] * args.folds
    all_ids = set(annotations)
    for config_index, pp in enumerate(configs):
        rows = evaluate_rows(annotations, [cache], [1.0], pp)
        for fold_index, test_list in enumerate(folds):
            test_ids = set(test_list)
            train_summary = objective_summary(subset(rows, all_ids - test_ids))
            score = float(train_summary["objective"])
            if best[fold_index] is None or score > best[fold_index]["train_objective"]:
                best[fold_index] = {
                    "config_index": config_index,
                    "train_objective": score,
                    "postprocess": pp,
                    "test_rows": subset(rows, test_ids),
                }
        if config_index % 250 == 0:
            print(f"evaluated {config_index}/{len(configs) - 1}")

    output = []
    for fold_index, selected in enumerate(best):
        test_ids = set(folds[fold_index])
        tuned_summary = objective_summary(selected["test_rows"])
        baseline_summary = objective_summary(subset(baseline_rows, test_ids))
        output.append(
            {
                "fold": fold_index,
                "test_clips": len(test_ids),
                "selected_config_index": selected["config_index"],
                "train_objective": selected["train_objective"],
                **{f"tuned_{key}": tuned_summary[key] for key in (
                    "objective", "activity_macro_f1", "active_f1", "boundary_f1_pm5", "boundary_f1_pm10"
                )},
                **{f"baseline_{key}": baseline_summary[key] for key in (
                    "objective", "activity_macro_f1", "active_f1", "boundary_f1_pm5", "boundary_f1_pm10"
                )},
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "fold_results.csv", output)
    write_json(
        out_dir / "selected_postprocess.json",
        {
            "development_only": True,
            "model": name,
            "folds": folds,
            "trials": args.trials,
            "seed": args.seed,
            "selected": [
                {
                    "fold": index,
                    "config_index": item["config_index"],
                    "train_objective": item["train_objective"],
                    "postprocess": item["postprocess"],
                }
                for index, item in enumerate(best)
            ],
        },
    )
    print(f"Wrote {out_dir / 'fold_results.csv'}")


if __name__ == "__main__":
    main()
