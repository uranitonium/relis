#!/usr/bin/env python3
"""Generate controlled boundary-model configs for hit-truncated retraining."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def set_common(cfg: dict, root: Path, name: str) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = 20270614
    cfg["data"]["data_dir"] = str((root / "data" / "train").resolve())
    cfg["data"]["max_len"] = 0
    cfg["train"].update(
        {
            "max_steps": 10000,
            "warmup_steps": 600,
            "log_every": 100,
            "ckpt_every": 1000,
            "out_dir": str((root / "runs" / name).resolve()),
        }
    )
    cfg["train"]["early_stop"].update(
        {"enabled": True, "min_steps": 1800, "patience_steps": 900, "min_delta": 0.0001}
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="deployment/checkpoints/legacy_auto/config.yaml")
    parser.add_argument("--workspace", default="paper_experiments/experiments/truncated_retraining")
    args = parser.parse_args()

    root = Path(args.workspace)
    with Path(args.base).open(encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    variants = {}
    variants["deployed_loss"] = set_common(base, root, "deployed_loss")

    sparse = set_common(base, root, "short_sparse")
    sparse["loss"].update(
        {
            "lambda_sparse": 0.05,
            "lambda_smooth": 0.50,
            "lambda_b_prior": 1.20,
            "b_mass_target": 0.07,
            "lambda_b_var": 1.50,
            "b_var_target": 0.015,
        }
    )
    sparse["postprocess"].update(
        {
            "robust_z_on": 1.05,
            "robust_z_off": 0.45,
            "boundary_frac_min": 0.04,
            "boundary_frac_max": 0.24,
        }
    )
    variants["short_sparse"] = sparse

    conservative = set_common(base, root, "short_conservative")
    conservative["loss"].update(
        {
            "lambda_sparse": 0.06,
            "lambda_smooth": 0.60,
            "lambda_b_prior": 1.50,
            "b_mass_target": 0.06,
            "lambda_b_var": 1.00,
            "b_var_target": 0.012,
        }
    )
    conservative["postprocess"].update(
        {
            "robust_z_on": 1.10,
            "robust_z_off": 0.50,
            "min_prominence": 0.12,
            "boundary_frac_min": 0.00,
            "boundary_frac_max": 0.20,
        }
    )
    variants["short_conservative"] = conservative

    finetune_deployed = set_common(base, root, "finetune_deployed_loss")
    finetune_deployed["train"].update(
        {"lr": 0.00005, "max_steps": 4000, "warmup_steps": 100, "ckpt_every": 500}
    )
    finetune_deployed["train"]["early_stop"].update(
        {"min_steps": 500, "patience_steps": 500, "min_delta": 0.00005}
    )
    variants["finetune_deployed_loss"] = finetune_deployed

    finetune_short = set_common(base, root, "finetune_short_loss")
    finetune_short["loss"].update(
        {
            "lambda_sparse": 0.04,
            "lambda_smooth": 0.50,
            "lambda_b_prior": 1.20,
            "b_mass_target": 0.08,
            "lambda_b_var": 1.50,
            "b_var_target": 0.015,
        }
    )
    finetune_short["postprocess"].update(
        {"boundary_frac_min": 0.05, "boundary_frac_max": 0.26}
    )
    finetune_short["train"].update(
        {"lr": 0.00005, "max_steps": 4000, "warmup_steps": 100, "ckpt_every": 500}
    )
    finetune_short["train"]["early_stop"].update(
        {"min_steps": 500, "patience_steps": 500, "min_delta": 0.00005}
    )
    variants["finetune_short_loss"] = finetune_short

    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in variants.items():
        path = config_dir / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        print(path)


if __name__ == "__main__":
    main()
