#!/usr/bin/env python3
"""Run equal-budget five-seed robustness checks for representation baselines."""

from __future__ import annotations

import argparse
import copy
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
OUT_ROOT = REPO / "paper_experiments/round2/representation_baseline_equalized"
OWNER_DIR = REPO / "paper_experiments/annotations/raw/owner"
CONSENSUS_DIR = REPO / "paper_experiments/annotations/adjudicated/consensus"
RELIS_ROWS = REPO / "paper_experiments/round2/architecture_ablations/per_seed_metrics.csv"
REFERENCE_CONFIG = REPO / (
    "paper_experiments/experiments/truncated_retraining/candidates/"
    "conservative_smooth02_seed18/config.yaml"
)
SEEDS = [20270614, 20270615, 20270616, 20270617, 20270618]
SEARCH_SEED = 20260810


BASELINES = {
    "rgb_resnet": {
        "label": "ResNet-18 RGB",
        "input_dim": 512,
        "base_config": REPO / "paper_experiments/experiments/rgb_resnet_baseline/configs/rgb_resnet_l3.yaml",
        "dev_manifest": REPO / "paper_experiments/experiments/rgb_resnet_baseline/manifests/dev_rgb.csv",
        "final_manifest": REPO / "paper_experiments/experiments/rgb_resnet_baseline/manifests/final_rgb.csv",
        "existing_checkpoint": REPO / "paper_experiments/experiments/rgb_resnet_baseline/runs/rgb_resnet_l3_seed20270618/ckpt_step_1000.pt",
    },
    "video_r3d": {
        "label": "R3D-18",
        "input_dim": 512,
        "base_config": REPO / "paper_experiments/experiments/video_r3d_baseline/configs/video_r3d_l3.yaml",
        "dev_manifest": REPO / "paper_experiments/experiments/video_r3d_baseline/manifests/dev_r3d.csv",
        "final_manifest": REPO / "paper_experiments/experiments/video_r3d_baseline/manifests/final_r3d.csv",
        "existing_checkpoint": REPO / "paper_experiments/experiments/video_r3d_baseline/runs/video_r3d_l3_seed20270618/ckpt_step_1000.pt",
    },
    "video_videomae": {
        "label": "VideoMAE-B",
        "input_dim": 768,
        "base_config": REPO / "paper_experiments/experiments/video_videomae_baseline/configs/video_videomae_l3.yaml",
        "dev_manifest": REPO / "paper_experiments/experiments/video_videomae_baseline/manifests/dev_videomae.csv",
        "final_manifest": REPO / "paper_experiments/experiments/video_videomae_baseline/manifests/final_videomae.csv",
        "existing_checkpoint": REPO / "paper_experiments/experiments/video_videomae_baseline/runs/video_videomae_l3_seed20270618/ckpt_step_1000.pt",
    },
    "video_swin": {
        "label": "Swin3D-T",
        "input_dim": 768,
        "base_config": REPO / "paper_experiments/round2/video_swin_baseline/configs/video_swin.yaml",
        "dev_manifest": REPO / "paper_experiments/round2/video_swin_baseline/manifests/dev_video_swin.csv",
        "final_manifest": REPO / "paper_experiments/round2/video_swin_baseline/manifests/final_video_swin.csv",
        "existing_checkpoint": REPO / "paper_experiments/round2/video_swin_baseline/runs/video_swin_seed20270618/ckpt_step_1000.pt",
    },
    "pose_stgcn": {
        "label": "two-skeleton ST-GCN",
        "input_dim": 48,
        "base_config": REPO / "paper_experiments/experiments/pose_stgcn_baseline/configs/pose_stgcn_l3.yaml",
        "dev_manifest": REPO / "paper_experiments/experiments/pose_stgcn_baseline/manifests/dev_pose_stgcn.csv",
        "final_manifest": REPO / "paper_experiments/experiments/pose_stgcn_baseline/manifests/final_pose_stgcn.csv",
        "existing_checkpoint": REPO / "paper_experiments/experiments/pose_stgcn_baseline/runs/pose_stgcn_l3_seed20270618/ckpt_step_1000.pt",
    },
}


def run(command: list[Any]) -> None:
    values = [str(value) for value in command]
    print("+", " ".join(values), flush=True)
    subprocess.run(values, cwd=str(REPO), check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_config(out_root: Path, method: str, seed: int) -> Path:
    spec = BASELINES[method]
    cfg = yaml.safe_load(Path(spec["base_config"]).read_text(encoding="utf-8"))
    reference = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    # Preserve only the representation/input adapter from the archived
    # baseline. The objective, model interfaces, optimizer, and schedule are
    # copied from ReLIS so this study genuinely equalizes training effort.
    cfg["loss"] = copy.deepcopy(reference["loss"])
    cfg["model"]["hidden_dim"] = reference["model"]["hidden_dim"]
    cfg["model"]["latent_dim"] = reference["model"]["latent_dim"]
    cfg["model"]["slow_dim"] = reference["model"].get("slow_dim", 3)
    cfg["model"]["proj"] = copy.deepcopy(reference["model"]["proj"])
    cfg["model"]["boundary"] = copy.deepcopy(reference["model"]["boundary"])
    cfg["model"]["boundary"]["use_critical"] = False
    cfg["model"]["boundary"]["critical_index"] = None
    for key in (
        "batch_size",
        "lr",
        "weight_decay",
        "grad_clip",
        "warmup_steps",
        "log_every",
    ):
        cfg["train"][key] = reference["train"][key]
    cfg["seed"] = seed
    cfg["train"]["max_steps"] = 1000
    cfg["train"]["ckpt_every"] = 1000
    cfg["train"]["early_stop"]["enabled"] = False
    cfg["train"]["out_dir"] = str((out_root / "runs" / method / str(seed)).resolve())
    path = out_root / "configs" / method / f"seed{seed}.yaml"
    write_yaml_if_changed(path, cfg)
    return path


def checkpoint_for(out_root: Path, method: str, seed: int) -> Path:
    return out_root / "runs" / method / str(seed) / "ckpt_step_1000.pt"


def train_if_needed(config: Path, checkpoint: Path, seed: int) -> None:
    if not checkpoint.exists():
        run([PY, "boundary_segmentation/train.py", "--config", config, "--device", "auto"])
    if not checkpoint.exists():
        final = checkpoint.parent / "ckpt_final.pt"
        if not final.exists():
            raise FileNotFoundError(checkpoint)
        shutil.copy2(final, checkpoint)


def tune(
    out_root: Path,
    method: str,
    seed: int,
    config: Path,
    checkpoint: Path,
    trials: int,
) -> tuple[Path, dict[str, Any], bool]:
    spec = BASELINES[method]
    name = f"{method}_seed{seed}"
    search_root = out_root / "postprocess_search"
    best_path = search_root / f"{name}-b_best.json"
    top_path = search_root / f"{name}-b_top.csv"
    if not controlled_search_is_current(
        best_path, top_path, trials, name, Path(spec["dev_manifest"])
    ):
        run(
            [
                PY,
                "paper_experiments/scripts/search_boundary_postprocess.py",
                "--manifest",
                spec["dev_manifest"],
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
                search_root,
            ]
        )
    if not controlled_search_is_current(
        best_path, top_path, trials, name, Path(spec["dev_manifest"])
    ):
        raise RuntimeError(f"Controlled search artifacts failed validation: {best_path}")
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg["postprocess"] = payload["best"]["postprocess"]
    tuned = out_root / "configs_tuned" / method / f"seed{seed}.yaml"
    changed = write_yaml_if_changed(tuned, cfg)
    return tuned, payload["best"]["summary"], changed


def evaluate(
    out_root: Path,
    method: str,
    seed: int,
    config: Path,
    checkpoint: Path,
    force: bool = False,
) -> dict[str, str]:
    spec = BASELINES[method]
    name = f"{method}_seed{seed}"
    prediction_root = out_root / "predictions" / name
    prediction_path = prediction_root / "full_model.json"
    prediction_changed = force or not prediction_path.exists()
    if prediction_changed:
        run(
            [
                PY,
                "paper_experiments/scripts/generate_boundary_predictions.py",
                "--manifest",
                spec["final_manifest"],
                "--method",
                "full_model",
                "--out-dir",
                prediction_root,
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
    result_root = out_root / "consensus_results" / name
    summary = result_root / "summary.csv"
    if prediction_changed or not summary.exists():
        run(
            [
                PY,
                "paper_experiments/scripts/compare_annotations_to_model.py",
                "--owner-dir",
                CONSENSUS_DIR,
                "--manifest",
                spec["final_manifest"],
                "--predictions",
                prediction_path,
                "--model-annotations-dir",
                result_root / "model_annotations",
                "--summary-out",
                summary,
                "--by-clip-out",
                result_root / "by_clip.csv",
                "--viewer-metrics-out",
                result_root / "viewer_metrics.csv",
            ]
        )
    for row in read_csv(summary):
        if row["cohort"] == "valid":
            return row
    raise KeyError(f"No valid row in {summary}")


def parameter_count(config: Path, input_dim: int) -> int:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    model = L3BoundaryModel(input_dim=input_dim, cfg=cfg)
    return sum(parameter.numel() for parameter in model.parameters())


def relis_rows() -> list[dict[str, Any]]:
    if not RELIS_ROWS.exists():
        raise FileNotFoundError(
            f"Run run_round2_architecture_ablations.py first: {RELIS_ROWS}"
        )
    output: list[dict[str, Any]] = []
    for row in read_csv(RELIS_ROWS):
        if row["variant"] != "transformer_aux3":
            continue
        output.append(
            {
                "method": "relis",
                "seed": int(row["seed"]),
                "parameters": int(row["parameters"]),
                "config": row["config"],
                "tuned_config": row["tuned_config"],
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "dev_clips": int(row["dev_clips"]),
                "dev_active_f1": float(row["dev_active_f1"]),
                "dev_boundary_f1_pm5": float(row["dev_boundary_f1_pm5"]),
                "dev_boundary_f1_pm10": float(row["dev_boundary_f1_pm10"]),
                "test_clips": int(row["test_clips"]),
                "test_activity_accuracy": float(row["test_activity_accuracy"]),
                "test_active_f1": float(row["test_active_f1"]),
                "test_boundary_f1_pm5": float(row["test_boundary_f1_pm5"]),
                "test_boundary_f1_pm10": float(row["test_boundary_f1_pm10"]),
            }
        )
    if len(output) != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} ReLIS rows, found {len(output)}")
    return output


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    metrics = [
        "dev_active_f1",
        "dev_boundary_f1_pm5",
        "dev_boundary_f1_pm10",
        "test_activity_accuracy",
        "test_active_f1",
        "test_boundary_f1_pm5",
        "test_boundary_f1_pm10",
    ]
    output: list[dict[str, Any]] = []
    for method in ["relis", *BASELINES]:
        selected = by_method[method]
        row: dict[str, Any] = {
            "method": method,
            "seeds": len(selected),
            "parameters": selected[0]["parameters"],
        }
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in selected])
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        output.append(row)
    return output


def render_results(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = {"relis": "ReLIS", **{key: value["label"] for key, value in BASELINES.items()}}
    lines = [
        "# Equal-budget five-seed representation comparison",
        "",
        "Every row uses the ReLIS loss settings and optimizer schedule, five",
        "training seeds, 1,000 unsupervised training steps, and the same 2,000 random",
        "development-only extraction candidates per model/seed.",
        "The frozen visual backbones are reused; no test labels enter training or",
        "post-processing selection.",
        "",
        "| representation | parameters | dev Active F1 | consensus Acc. | consensus Active F1 | consensus BF1@5 | consensus BF1@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {labels[row['method']]} | {int(row['parameters']):,} | "
            f"{row['dev_active_f1_mean']:.3f} +/- {row['dev_active_f1_sd']:.3f} | "
            f"{row['test_activity_accuracy_mean']:.3f} +/- {row['test_activity_accuracy_sd']:.3f} | "
            f"{row['test_active_f1_mean']:.3f} +/- {row['test_active_f1_sd']:.3f} | "
            f"{row['test_boundary_f1_pm5_mean']:.3f} +/- {row['test_boundary_f1_pm5_sd']:.3f} | "
            f"{row['test_boundary_f1_pm10_mean']:.3f} +/- {row['test_boundary_f1_pm10_sd']:.3f} |"
        )
    lines.extend(
        [
            "",
            "All five seeds are retained; no test labels enter model or extractor selection.",
            "",
            "Cached features, checkpoints, development searches, tuned configs, final",
            "predictions, and per-clip evaluations are retained with this report.",
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

    rows = relis_rows()
    write_csv(out_root / "per_seed_metrics.csv", rows)
    for method, spec in BASELINES.items():
        for seed in SEEDS:
            print(f"[{method}] seed {seed}", flush=True)
            config = make_config(out_root, method, seed)
            checkpoint = checkpoint_for(out_root, method, seed)
            if not args.skip_train:
                train_if_needed(config, checkpoint, seed)
            elif not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            tuned, dev, tuned_changed = tune(
                out_root, method, seed, config, checkpoint, args.trials
            )
            test = evaluate(
                out_root, method, seed, tuned, checkpoint, force=tuned_changed
            )
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "parameters": parameter_count(config, int(spec["input_dim"])),
                    "config": str(config),
                    "tuned_config": str(tuned),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "dev_clips": int(dev["clips"]),
                    "dev_active_f1": float(dev["active_f1"]),
                    "dev_boundary_f1_pm5": float(dev["boundary_f1_pm5"]),
                    "dev_boundary_f1_pm10": float(dev["boundary_f1_pm10"]),
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
    render_results(out_root / "RESULTS.md", aggregates)
    with (out_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": str(REPO / "paper_experiments/round2/PROTOCOL.md"),
                "seeds": SEEDS,
                "postprocess_trials_per_model_seed": args.trials,
                "postprocess_search_seed": SEARCH_SEED,
                "postprocess_candidate_protocol": "random_only_common_seeded_stream",
                "equalized_reference_config": str(REFERENCE_CONFIG),
                "baselines": {
                    key: {field: str(value) for field, value in spec.items()}
                    for key, spec in BASELINES.items()
                },
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
