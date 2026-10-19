#!/usr/bin/env python3
"""Run the frozen multi-seed Round-2 architecture ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO / "boundary_segmentation"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from models.l3_model import L3BoundaryModel  # noqa: E402


PY = Path(sys.executable)
OUT_ROOT = REPO / "paper_experiments/round2/architecture_ablations"
BASE_CONFIG = REPO / (
    "paper_experiments/experiments/truncated_retraining/candidates/"
    "optimistic_smooth02_seed18/config.yaml"
)
DEV_MANIFEST = REPO / (
    "paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv"
)
FINAL_MANIFEST = REPO / (
    "paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv"
)
OWNER_DIR = REPO / "paper_experiments/annotations/raw/owner"
CONSENSUS_DIR = REPO / "paper_experiments/annotations/adjudicated/consensus"
SEEDS = [20270614, 20270615, 20270616, 20270617, 20270618]
SEARCH_SEED = 20260810


VARIANTS = {
    "transformer_aux3": {
        "encoder_type": "transformer",
        "slow_dim": 3,
        "use_critical": True,
    },
    "transformer_noaux": {
        "encoder_type": "transformer",
        "slow_dim": 0,
        "use_critical": True,
    },
    "transformer_latent_only": {
        "encoder_type": "transformer",
        "slow_dim": 0,
        "use_critical": False,
    },
    "tcn_aux3": {
        "encoder_type": "tcn",
        "slow_dim": 3,
        "use_critical": True,
    },
    "lstm_aux3": {
        "encoder_type": "lstm",
        "slow_dim": 3,
        "use_critical": True,
    },
}


def run(command: list[Any]) -> None:
    command = [str(value) for value in command]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(REPO), check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def controlled_search_is_current(
    best_path: Path,
    top_path: Path,
    trials: int,
    expected_name: str,
    expected_manifest: Path,
) -> bool:
    """Return whether cached search outputs implement the random-only protocol."""
    if not best_path.exists() or not top_path.exists():
        return False
    try:
        payload = json.loads(best_path.read_text(encoding="utf-8"))
        protocol = payload["candidate_protocol"]
        best = payload["best"]
        top = read_csv(top_path)
        first = top[0]
        model = payload["models"][0]
        expected_cache = top_path.parent / "score_cache" / f"{expected_name}.npz"
        return (
            int(payload["trials"]) == trials
            and int(payload["seed"]) == SEARCH_SEED
            and Path(payload["manifest"]).resolve() == expected_manifest.resolve()
            and model["name"] == expected_name
            and model["score_source"] == "b"
            and Path(model["score_cache"]).resolve() == expected_cache.resolve()
            and bool(protocol["random_only"])
            and not bool(protocol["include_deployed_candidate"])
            and int(protocol["random_samples"]) == trials
            and int(protocol["total_candidates"]) == trials
            and best["candidate_source"] == "random"
            and 1 <= int(best["sample_index"]) <= trials
            and int(best["config_index"]) == int(best["sample_index"]) - 1
            and len(top) == min(100, trials)
            and all(row["candidate_source"] == "random" for row in top)
            and int(first["sample_index"]) == int(best["sample_index"])
            and abs(float(first["objective"]) - float(best["summary"]["objective"])) < 1e-12
            and abs(
                float(first["balanced_objective"])
                - float(best["summary"]["balanced_objective"])
            ) < 1e-12
            and all(
                str(first[f"pp_{key}"]) == str(value)
                for key, value in best["postprocess"].items()
            )
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_yaml_if_changed(path: Path, payload: dict[str, Any]) -> bool:
    """Write YAML only when its content changed and report that change."""
    rendered = yaml.safe_dump(payload, sort_keys=False)
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return True


def existing_transformer_checkpoint(seed: int) -> Path:
    if seed == 20270618:
        return REPO / (
            "paper_experiments/experiments/truncated_retraining/runs/"
            "short_sparse_seed18_smooth02/ckpt_step_1000.pt"
        )
    return REPO / (
        "paper_experiments/experiments/truncated_retraining/runs/"
        f"short_sparse_smooth02_seed{seed}/ckpt_step_1000.pt"
    )


def make_config(out_root: Path, variant: str, seed: int) -> Path:
    spec = VARIANTS[variant]
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["seed"] = seed
    cfg["model"]["encoder_type"] = spec["encoder_type"]
    cfg["model"]["slow_dim"] = spec["slow_dim"]
    cfg["model"]["boundary"]["use_critical"] = spec["use_critical"]
    cfg["model"]["boundary"]["critical_index"] = 23 if spec["use_critical"] else None
    cfg["model"]["tcn"] = {
        "layers": 4,
        "kernel_size": 3,
        "dilation_base": 2,
        "dropout": 0.1,
    }
    cfg["model"]["lstm"] = {
        "n_layers": 3,
        "dropout": 0.1,
        "bidirectional": True,
    }
    cfg["train"]["max_steps"] = 1000
    cfg["train"]["warmup_steps"] = 600
    cfg["train"]["ckpt_every"] = 1000
    cfg["train"]["early_stop"]["enabled"] = False
    cfg["train"]["out_dir"] = str((out_root / "runs" / variant / str(seed)).resolve())
    path = out_root / "configs" / variant / f"seed{seed}.yaml"
    write_yaml_if_changed(path, cfg)
    return path


def checkpoint_for(out_root: Path, variant: str, seed: int) -> Path:
    if variant == "transformer_aux3":
        return existing_transformer_checkpoint(seed)
    return out_root / "runs" / variant / str(seed) / "ckpt_step_1000.pt"


def train_if_needed(config: Path, checkpoint: Path, variant: str) -> None:
    if variant == "transformer_aux3":
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        return
    if not checkpoint.exists():
        run([PY, "boundary_segmentation/train.py", "--config", config, "--device", "auto"])
    if not checkpoint.exists():
        final = checkpoint.parent / "ckpt_final.pt"
        if final.exists():
            shutil.copy2(final, checkpoint)
        else:
            raise FileNotFoundError(checkpoint)


def parameter_count(config: Path) -> int:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    model = L3BoundaryModel(input_dim=26, cfg=cfg)
    return sum(parameter.numel() for parameter in model.parameters())


def tune(
    out_root: Path, variant: str, seed: int, config: Path, checkpoint: Path, trials: int
) -> tuple[Path, dict[str, Any], bool]:
    name = f"{variant}_seed{seed}"
    search_dir = out_root / "postprocess_search"
    best_path = search_dir / f"{name}-b_best.json"
    top_path = search_dir / f"{name}-b_top.csv"
    if not controlled_search_is_current(
        best_path, top_path, trials, name, DEV_MANIFEST
    ):
        run(
            [
                PY,
                "paper_experiments/scripts/search_boundary_postprocess.py",
                "--manifest",
                DEV_MANIFEST,
                "--owner-dir",
                OWNER_DIR,
                "--model",
                f"{name}={config}={checkpoint}",
                "--score-source",
                "b",
                "--trials",
                trials,
                "--random-only",
                "--seed",
                SEARCH_SEED,
                "--top-k",
                100,
                "--out-dir",
                search_dir,
            ]
        )
    if not controlled_search_is_current(
        best_path, top_path, trials, name, DEV_MANIFEST
    ):
        raise RuntimeError(f"Controlled search artifacts failed validation: {best_path}")
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg["postprocess"] = payload["best"]["postprocess"]
    tuned = out_root / "configs_tuned" / variant / f"seed{seed}.yaml"
    changed = write_yaml_if_changed(tuned, cfg)
    return tuned, payload["best"]["summary"], changed


def evaluate(
    out_root: Path,
    variant: str,
    seed: int,
    config: Path,
    checkpoint: Path,
    force: bool = False,
) -> dict[str, str]:
    name = f"{variant}_seed{seed}"
    prediction_dir = out_root / "predictions" / name
    prediction_path = prediction_dir / "full_model.json"
    prediction_changed = force or not prediction_path.exists()
    if prediction_changed:
        run(
            [
                PY,
                "paper_experiments/scripts/generate_boundary_predictions.py",
                "--manifest",
                FINAL_MANIFEST,
                "--method",
                "full_model",
                "--out-dir",
                prediction_dir,
                "--config",
                config,
                "--ckpt",
                checkpoint,
                "--python-bin",
                PY,
                "--seed",
                SEARCH_SEED,
            ]
        )
    result_dir = out_root / "consensus_results" / name
    summary_path = result_dir / "summary.csv"
    if prediction_changed or not summary_path.exists():
        run(
            [
                PY,
                "paper_experiments/scripts/compare_annotations_to_model.py",
                "--owner-dir",
                CONSENSUS_DIR,
                "--manifest",
                FINAL_MANIFEST,
                "--predictions",
                prediction_path,
                "--model-annotations-dir",
                result_dir / "model_annotations",
                "--summary-out",
                summary_path,
                "--by-clip-out",
                result_dir / "by_clip.csv",
                "--viewer-metrics-out",
                result_dir / "viewer_metrics.csv",
            ]
        )
    for row in read_csv(summary_path):
        if row["cohort"] == "valid":
            return row
    raise KeyError(f"No valid row in {summary_path}")


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = [
        "dev_active_f1",
        "dev_boundary_f1_pm5",
        "dev_boundary_f1_pm10",
        "test_activity_accuracy",
        "test_active_f1",
        "test_boundary_f1_pm5",
        "test_boundary_f1_pm10",
    ]
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    for variant in VARIANTS:
        selected = by_variant[variant]
        row: dict[str, Any] = {
            "variant": variant,
            "seeds": len(selected),
            "parameters": selected[0]["parameters"],
        }
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in selected], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        output.append(row)
    return output


def render_results(path: Path, rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> None:
    display = {
        "transformer_aux3": "Transformer + 3 auxiliary",
        "transformer_noaux": "Transformer, no auxiliary",
        "transformer_latent_only": "Transformer, latent only",
        "tcn_aux3": "TCN + 3 auxiliary",
        "lstm_aux3": "BiLSTM + 3 auxiliary",
    }
    lines = [
        "# Frozen multi-seed architecture ablations",
        "",
        "All five variants use the same unlabeled training set, loss settings, 1,000",
        "steps, and five seeds. Each model/seed received the same 2,000 random",
        "development-only extraction candidates before one consensus evaluation.",
        "",
        "| variant | parameters | dev Active F1 | consensus Active F1 | consensus BF1@5 | consensus BF1@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {display[row['variant']]} | {int(row['parameters']):,} | "
            f"{row['dev_active_f1_mean']:.3f} +/- {row['dev_active_f1_sd']:.3f} | "
            f"{row['test_active_f1_mean']:.3f} +/- {row['test_active_f1_sd']:.3f} | "
            f"{row['test_boundary_f1_pm5_mean']:.3f} +/- {row['test_boundary_f1_pm5_sd']:.3f} | "
            f"{row['test_boundary_f1_pm10_mean']:.3f} +/- {row['test_boundary_f1_pm10_sd']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Ranges (all retained seeds):",
            "",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            f"- {display[row['variant']]}: consensus Active F1 "
            f"{row['test_active_f1_min']:.3f}--{row['test_active_f1_max']:.3f}; "
            f"BF1@10 {row['test_boundary_f1_pm10_min']:.3f}--"
            f"{row['test_boundary_f1_pm10_max']:.3f}."
        )
    lines.extend(
        [
            "",
            "These are architecture sensitivity analyses, not a separately selected",
            "primary result, and no seed or variant was selected from the consensus",
            "results.",
            "",
            "Per-seed values, tuned configs, development searches, predictions, and",
            "per-clip consensus metrics are retained alongside this report.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            print(f"[{variant}] seed {seed}", flush=True)
            config = make_config(out_root, variant, seed)
            checkpoint = checkpoint_for(out_root, variant, seed)
            if not args.skip_train:
                train_if_needed(config, checkpoint, variant)
            elif not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            tuned, dev, tuned_changed = tune(
                out_root, variant, seed, config, checkpoint, args.trials
            )
            test = evaluate(
                out_root, variant, seed, tuned, checkpoint, force=tuned_changed
            )
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "parameters": parameter_count(config),
                    "config": str(config),
                    "tuned_config": str(tuned),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "dev_clips": int(dev["clips"]),
                    "dev_activity_accuracy": float(dev["activity_accuracy"]),
                    "dev_active_f1": float(dev["active_f1"]),
                    "dev_boundary_f1_pm5": float(dev["boundary_f1_pm5"]),
                    "dev_boundary_f1_pm10": float(dev["boundary_f1_pm10"]),
                    "dev_objective": float(dev["objective"]),
                    "test_clips": int(float(test["clips"])),
                    "test_activity_accuracy": float(test["activity_accuracy"]),
                    "test_active_f1": float(test["active_f1"]),
                    "test_boundary_f1_pm5": float(test["boundary_f1_pm5"]),
                    "test_boundary_f1_pm10": float(test["boundary_f1_pm10"]),
                }
            )
            write_csv(out_root / "per_seed_metrics.csv", rows)

    aggregates = aggregate(rows)
    write_csv(out_root / "aggregate_metrics.csv", aggregates)
    render_results(out_root / "RESULTS.md", rows, aggregates)
    with (out_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": str(REPO / "paper_experiments/round2/PROTOCOL.md"),
                "variants": VARIANTS,
                "seeds": SEEDS,
                "postprocess_trials_per_model_seed": args.trials,
                "postprocess_search_seed": SEARCH_SEED,
                "postprocess_candidate_protocol": "random_only_common_seeded_stream",
                "rows": rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(f"Wrote {out_root / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
