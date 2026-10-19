#!/usr/bin/env python3
"""Sweep shape-preserving boundary-model inference constants on development labels."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import List

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_csv, read_json, write_csv  # noqa: E402
from search_boundary_postprocess import cache_scores, evaluate, load_annotations, load_cache  # noqa: E402


def float_list(value: str) -> List[float]:
    return [float(item) for item in value.split(",")]


def int_list(value: str) -> List[int]:
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--postprocess-json", required=True)
    parser.add_argument("--manifest", default="paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv")
    parser.add_argument("--owner-dir", default="paper_experiments/annotations/raw/owner")
    parser.add_argument("--k-values", type=int_list, default=[4, 6, 8, 10, 12, 16])
    parser.add_argument("--skip-values", type=float_list, default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--temperatures", type=float_list, default=[2.0])
    parser.add_argument("--out-dir", default="paper_experiments/experiments/truncated_retraining/inference_sweep")
    args = parser.parse_args()

    base_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    checkpoint = Path(args.checkpoint).resolve()
    annotations = load_annotations(Path(args.owner_dir))
    rows = [row for row in read_csv(args.manifest) if row["clip_id"] in annotations]
    pp = read_json(args.postprocess_json)["best"]["postprocess"]
    out_dir = Path(args.out_dir)
    config_dir = out_dir / "configs"
    cache_dir = out_dir / "score_cache"
    config_dir.mkdir(parents=True, exist_ok=True)
    output = []

    for k in args.k_values:
        for skip_alpha in args.skip_values:
            for temperature in args.temperatures:
                name = f"k{k}_skip{skip_alpha:g}_temp{temperature:g}".replace(".", "p")
                config = copy.deepcopy(base_config)
                config["model"]["boundary"]["k"] = k
                config["model"]["boundary"]["skip_alpha"] = skip_alpha
                config["model"]["boundary"]["temperature"] = temperature
                config_path = config_dir / f"{name}.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                cache_path = cache_scores(name, config_path.resolve(), checkpoint, rows, cache_dir)
                cache = load_cache(cache_path, annotations.keys())
                summary = evaluate(annotations, [cache], [1.0], pp)
                output.append(
                    {
                        "name": name,
                        "k": k,
                        "skip_alpha": skip_alpha,
                        "temperature": temperature,
                        **summary,
                    }
                )
                print(name, summary["objective"])

    output.sort(key=lambda row: float(row["objective"]), reverse=True)
    write_csv(out_dir / "fixed_postprocess_results.csv", output)
    print(f"Wrote {out_dir / 'fixed_postprocess_results.csv'} rows={len(output)}")


if __name__ == "__main__":
    main()
