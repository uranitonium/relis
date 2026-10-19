#!/usr/bin/env python3
"""Create a small declared postprocessing grid for truncated checkpoints."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path("paper_experiments/experiments/truncated_retraining")


def load(name: str) -> dict:
    with (ROOT / "configs" / f"{name}.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save(name: str, cfg: dict) -> None:
    path = ROOT / "configs" / f"{name}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    print(path)


def with_postprocess(base: dict, **updates) -> dict:
    cfg = copy.deepcopy(base)
    cfg["postprocess"].update(updates)
    return cfg


def main() -> None:
    conservative = load("short_conservative")
    sparse = load("short_sparse")
    deployed = load("deployed_loss")["postprocess"]

    save("short_conservative_deployed_pp", with_postprocess(conservative, **deployed))
    save(
        "short_conservative_balanced",
        with_postprocess(
            conservative,
            robust_z_on=0.90,
            robust_z_off=0.35,
            min_prominence=0.08,
            boundary_frac_min=0.04,
            boundary_frac_max=0.26,
        ),
    )
    save(
        "short_conservative_recall",
        with_postprocess(
            conservative,
            robust_z_on=0.80,
            robust_z_off=0.25,
            min_prominence=0.06,
            boundary_frac_min=0.06,
            boundary_frac_max=0.28,
        ),
    )
    save("short_sparse_deployed_pp", with_postprocess(sparse, **deployed))
    save(
        "short_sparse_balanced",
        with_postprocess(
            sparse,
            robust_z_on=0.90,
            robust_z_off=0.35,
            min_prominence=0.08,
            boundary_frac_min=0.05,
            boundary_frac_max=0.27,
        ),
    )


if __name__ == "__main__":
    main()
