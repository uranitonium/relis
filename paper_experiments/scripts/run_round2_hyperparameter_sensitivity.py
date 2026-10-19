#!/usr/bin/env python3
"""Run the frozen development-only Round-2 loss sensitivity study."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
PY = Path(sys.executable)
ROOT = REPO / "paper_experiments/round2/hyperparameter_sensitivity"
BASE = REPO / "paper_experiments/experiments/truncated_retraining/candidates/optimistic_smooth02_seed18/config.yaml"
BASE_CKPT = REPO / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_seed18_smooth02/ckpt_step_1000.pt"
DEV = REPO / "paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv"
OWNER = REPO / "paper_experiments/annotations/raw/owner"


GROUPS = {
    "score_shape": ["lambda_sparse", "lambda_smooth"],
    "latent_coherence": ["lambda_cons", "lambda_contrast"],
    "anti_collapse": ["lambda_z_var", "lambda_b_prior", "lambda_b_var"],
    "alignment": ["lambda_b_align"],
}
TARGETS = ["b_mass_target", "b_var_target", "z_var_target"]


def run(command: list[object]) -> None:
    values = [str(value) for value in command]
    print("+", " ".join(values), flush=True)
    subprocess.run(values, cwd=str(REPO), check=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def variants() -> list[tuple[str, str, float]]:
    values = [("base", "base", 1.0)]
    for group in GROUPS:
        for factor in (0.5, 2.0):
            values.append((f"{group}_x{factor:g}", group, factor))
    for target in TARGETS:
        for factor in (0.5, 2.0):
            values.append((f"{target}_x{factor:g}", target, factor))
    return values


def make_config(name: str, kind: str, factor: float) -> Path:
    config = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    config["seed"] = 20270618
    if kind in GROUPS:
        for key in GROUPS[kind]:
            config["loss"][key] = float(config["loss"][key]) * factor
    elif kind in TARGETS:
        config["loss"][kind] = float(config["loss"][kind]) * factor
    config["train"]["max_steps"] = 1000
    config["train"]["ckpt_every"] = 1000
    config["train"]["early_stop"]["enabled"] = False
    config["train"]["out_dir"] = str((ROOT / "runs" / name).resolve())
    path = ROOT / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def checkpoint(name: str, config: Path) -> Path:
    if name == "base":
        return BASE_CKPT
    path = ROOT / "runs" / name / "ckpt_step_1000.pt"
    if not path.exists():
        run([PY, "boundary_segmentation/train.py", "--config", config, "--device", "auto"])
    if not path.exists():
        final = path.parent / "ckpt_final.pt"
        if final.exists():
            shutil.copy2(final, path)
        else:
            raise FileNotFoundError(path)
    return path


def tune(name: str, config: Path, ckpt: Path) -> dict[str, object]:
    output = ROOT / "postprocess_search"
    best_path = output / f"{name}-b_best.json"
    if not best_path.exists():
        run(
            [
                PY,
                "paper_experiments/scripts/search_boundary_postprocess.py",
                "--manifest",
                DEV,
                "--owner-dir",
                OWNER,
                "--model",
                f"{name}={config}={ckpt}",
                "--score-source",
                "b",
                "--trials",
                1000,
                "--seed",
                20260810,
                "--top-k",
                100,
                "--out-dir",
                output,
            ]
        )
    return json.loads(best_path.read_text(encoding="utf-8"))["best"]["summary"]


def main() -> None:
    rows = []
    for name, kind, factor in variants():
        print(f"[{name}]", flush=True)
        config = make_config(name, kind, factor)
        ckpt = checkpoint(name, config)
        result = tune(name, config, ckpt)
        rows.append(
            {
                "variant": name,
                "kind": kind,
                "factor": factor,
                "clips": int(result["clips"]),
                "activity_accuracy": float(result["activity_accuracy"]),
                "active_f1": float(result["active_f1"]),
                "boundary_f1_pm5": float(result["boundary_f1_pm5"]),
                "boundary_f1_pm10": float(result["boundary_f1_pm10"]),
                "objective": float(result["objective"]),
            }
        )
        write_csv(ROOT / "metrics.csv", rows)
    base = next(row for row in rows if row["variant"] == "base")
    lines = [
        "# Frozen development-only loss sensitivity",
        "",
        "Single seed; every row uses an independent 1,000-trial development search.",
        "No consensus labels are evaluated or used.",
        "",
        "| variant | Active F1 | BF1@5 | BF1@10 | delta Active F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['active_f1']:.3f} | {row['boundary_f1_pm5']:.3f} | "
            f"{row['boundary_f1_pm10']:.3f} | {row['active_f1'] - base['active_f1']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "This is a local one-factor/group-at-a-time diagnostic rather than a full",
            "factorial analysis. It measures development sensitivity, not held-out",
            "generalization or multi-seed robustness.",
            "",
        ]
    )
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
