#!/usr/bin/env python3
"""Collect paper-ready result files and simple dataset stats."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_csv, read_json, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dataset stats for paper-ready folder.")
    parser.add_argument("--manifest", default="paper_experiments/manifests/foil_main.csv")
    parser.add_argument("--gold", default="paper_experiments/annotations/gold/interaction_states.json")
    parser.add_argument("--out-dir", default="paper_experiments/paper_ready")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    rows = read_csv(args.manifest)
    counts = {label: sum(1 for r in rows if r.get("outcome_label") == label) for label in ["Left", "Right", "None"]}
    gold = read_json(args.gold) if Path(args.gold).exists() else {"clips": []}
    segment_counts = [len(c.get("segments", [])) for c in gold.get("clips", [])]
    stats = {
        "manifest": args.manifest,
        "clips_in_manifest": len(rows),
        "outcome_counts": counts,
        "gold_annotated_clips": len(gold.get("clips", [])),
        "mean_gold_states_per_clip": sum(segment_counts) / len(segment_counts) if segment_counts else 0.0,
        "total_gold_states": sum(segment_counts),
    }
    write_json(out_dir / "dataset_stats.json", stats)
    write_csv(out_dir / "dataset_stats.csv", [{"metric": k, "value": v} for k, v in stats.items()])
    print(f"Wrote dataset stats to {out_dir}")


if __name__ == "__main__":
    main()
