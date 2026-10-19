#!/usr/bin/env python3
"""Run development-only slow-variable ablations for the foil boundary model."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
OUT_ROOT = REPO / "paper_experiments/experiments/slow_variable_ablation"
BASE_CONFIG = REPO / "paper_experiments/experiments/truncated_retraining/candidates/optimistic_smooth02_seed18/config.yaml"
PRIMARY_BEST = REPO / "paper_experiments/experiments/truncated_retraining/postprocess_search/sparse_seed18_smooth02-b_best.json"
OWNER_DIR = REPO / "paper_experiments/annotations/raw/owner"
DEV_MANIFEST = REPO / "paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv"


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(REPO), check=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_config(out_root: Path, slow_dim: int, max_steps: int) -> Path:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["model"]["slow_dim"] = slow_dim
    cfg["train"]["max_steps"] = max_steps
    cfg["train"]["warmup_steps"] = min(600, max(10, max_steps // 2))
    cfg["train"]["ckpt_every"] = max_steps
    cfg["train"]["early_stop"]["enabled"] = False
    cfg["train"]["out_dir"] = str((out_root / f"runs/slow{slow_dim}_seed20270618").resolve())
    cfg["seed"] = 20270618
    config_path = out_root / f"configs/slow{slow_dim}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def tune_dev(name: str, config_path: Path, ckpt_path: Path, out_root: Path, trials: int) -> dict[str, Any]:
    search_dir = out_root / "postprocess_search"
    best_path = search_dir / f"{name}-b_best.json"
    if not best_path.exists():
        run(
            [
                PY,
                "paper_experiments/scripts/search_boundary_postprocess.py",
                "--manifest",
                str(DEV_MANIFEST),
                "--owner-dir",
                str(OWNER_DIR),
                "--model",
                f"{name}={config_path}={ckpt_path}",
                "--score-source",
                "b",
                "--trials",
                str(trials),
                "--seed",
                "20270618",
                "--top-k",
                "100",
                "--out-dir",
                str(search_dir),
            ]
        )
    return json.loads(best_path.read_text(encoding="utf-8"))["best"]["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    out_root = (REPO / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    primary = json.loads(PRIMARY_BEST.read_text(encoding="utf-8"))["best"]["summary"]
    rows.append(
        {
            "slow_dim": 3,
            "source": "existing_primary",
            "clips": int(primary["clips"]),
            "activity_accuracy": primary["activity_accuracy"],
            "active_f1": primary["active_f1"],
            "boundary_f1_pm5": primary["boundary_f1_pm5"],
            "boundary_f1_pm10": primary["boundary_f1_pm10"],
            "objective": primary["objective"],
        }
    )

    for slow_dim in (0, 1, 5):
        config_path = build_config(out_root, slow_dim, args.max_steps)
        ckpt_path = out_root / f"runs/slow{slow_dim}_seed20270618/ckpt_step_{args.max_steps}.pt"
        if not args.skip_train or not ckpt_path.exists():
            run([PY, "boundary_segmentation/train.py", "--config", str(config_path), "--device", "auto"])
        if not ckpt_path.exists():
            ckpt_path = out_root / f"runs/slow{slow_dim}_seed20270618/ckpt_final.pt"
        summary = tune_dev(f"slow{slow_dim}", config_path, ckpt_path, out_root, args.trials)
        rows.append(
            {
                "slow_dim": slow_dim,
                "source": "development_retrain",
                "clips": int(summary["clips"]),
                "activity_accuracy": summary["activity_accuracy"],
                "active_f1": summary["active_f1"],
                "boundary_f1_pm5": summary["boundary_f1_pm5"],
                "boundary_f1_pm10": summary["boundary_f1_pm10"],
                "objective": summary["objective"],
            }
        )

    rows = sorted(rows, key=lambda row: int(row["slow_dim"]))
    write_csv(out_root / "slow_variable_dev.csv", rows)
    print(f"Wrote {out_root / 'slow_variable_dev.csv'}")


if __name__ == "__main__":
    main()
