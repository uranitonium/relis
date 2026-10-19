#!/usr/bin/env python3
"""Create a frozen boundary candidate config from model and postprocess results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--postprocess-json", required=True)
    parser.add_argument("--out-root", default="paper_experiments/experiments/truncated_retraining/candidates")
    parser.add_argument("--status", choices=["conservative", "optimistic"], required=True)
    args = parser.parse_args()

    model_config_path = Path(args.model_config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    postprocess_path = Path(args.postprocess_json).resolve()
    config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    postprocess_payload = json.loads(postprocess_path.read_text(encoding="utf-8"))
    config["postprocess"] = postprocess_payload["best"]["postprocess"]

    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metadata = {
        "name": args.name,
        "status": args.status,
        "development_only": True,
        "model_config_source": str(model_config_path),
        "checkpoint": str(checkpoint_path),
        "postprocess_source": str(postprocess_path),
        "frozen_config": str((out_dir / "config.yaml").resolve()),
        "selection_rule": (
            "Primary candidate for held-out evaluation; selected for development metrics and unlabeled-audit stability."
            if args.status == "conservative"
            else "Secondary optimistic candidate; fully tuned on owner development labels."
        ),
    }
    (out_dir / "candidate.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
