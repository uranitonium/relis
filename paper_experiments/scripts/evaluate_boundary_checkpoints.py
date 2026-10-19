#!/usr/bin/env python3
"""Evaluate all saved checkpoints with fixed development postprocessing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_csv, read_json, write_csv  # noqa: E402
from search_boundary_postprocess import cache_scores, evaluate, load_annotations, load_cache  # noqa: E402


def checkpoint_step(path: Path) -> int:
    import torch

    return int(torch.load(path, map_location="cpu").get("step", -1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="paper_experiments/experiments/truncated_retraining")
    parser.add_argument("--manifest", default="paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv")
    parser.add_argument("--owner-dir", default="paper_experiments/annotations/raw/owner")
    parser.add_argument("--postprocess-json", required=True)
    parser.add_argument("--out", default="paper_experiments/experiments/truncated_retraining/results/checkpoint_sweep.csv")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    annotations = load_annotations(Path(args.owner_dir))
    manifest_rows = [row for row in read_csv(args.manifest) if row["clip_id"] in annotations]
    search_payload = read_json(args.postprocess_json)
    pp = search_payload["best"]["postprocess"]
    output = []

    for run_dir in sorted((workspace / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        config_path = workspace / "configs" / f"{run_dir.name}.yaml"
        if not config_path.exists():
            config_path = run_dir / "config.yaml"
        for checkpoint in sorted(run_dir.glob("ckpt*.pt")):
            name = f"{run_dir.name}__{checkpoint.stem}"
            cache_path = cache_scores(
                name,
                config_path.resolve(),
                checkpoint.resolve(),
                manifest_rows,
                workspace / "postprocess_search" / "checkpoint_cache",
            )
            cache = load_cache(cache_path, annotations.keys())
            summary = evaluate(annotations, [cache], [1.0], pp)
            output.append(
                {
                    "run": run_dir.name,
                    "checkpoint": checkpoint.name,
                    "step": checkpoint_step(checkpoint),
                    **summary,
                }
            )
            print(name, summary["objective"])

    output.sort(key=lambda row: float(row["objective"]), reverse=True)
    write_csv(args.out, output)
    print(f"Wrote {args.out} rows={len(output)}")


if __name__ == "__main__":
    main()
