#!/usr/bin/env python3
"""Evaluate frozen ReLIS auxiliary summaries in the downstream state classifier."""

from __future__ import annotations

import os
import argparse
import csv
import hashlib
import itertools
import json
import pickle
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import yaml
from sklearn.metrics import f1_score


BOUNDARY_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = BOUNDARY_ROOT / "boundary_segmentation"
STATE_ROOT = Path(os.environ.get("RELIS_STATE_ROOT", "data/state"))
STATE_SCRIPTS = STATE_ROOT / "experiments/scripts"
for path in (
    BOUNDARY_ROOT,
    MODEL_ROOT,
    STATE_ROOT,
    STATE_SCRIPTS,
    Path(__file__).resolve().parent,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_camera_state_variants import METHODS, sample_key  # noqa: E402
from experiment_camera_robust_state import cross_validated_head  # noqa: E402
from experiment_primary_state_models import (  # noqa: E402
    HEAD_VALUES,
    build_samples,
    evaluate_prediction_set,
    feature_matrix,
    load_annotations,
    read_json,
)
from state_model.features import paired_feature_indices  # noqa: E402
from models.l3_model import L3BoundaryModel  # noqa: E402


DEFAULT_OUT = BOUNDARY_ROOT / "paper_experiments/round2/downstream_aux_classifier"
DEFAULT_ANNOTATIONS = STATE_ROOT / (
    "experiments/aaai_tactical_principles/state_audit_v2_linked50/annotations"
)
DEFAULT_RAW_CONTEXT = STATE_ROOT / (
    "experiments/aaai_tactical_principles/full_exchange_regions/"
    "pipeline_no_tail_refined2s_v37/l3_out/l3_llm_context.json"
)
DEFAULT_STABILIZED_CONTEXT = STATE_ROOT / (
    "experiments/aaai_tactical_principles/camera_robust_state_v1/"
    "l3_llm_context_translation_stabilized.json"
)
DEFAULT_L2 = STATE_ROOT / (
    "experiments/aaai_tactical_principles/full_exchange_regions/"
    "pipeline_legacy_v32/working/legacy_auto/l2_out"
)
DEFAULT_BASE_MODEL = STATE_ROOT / (
    "artifacts/state_models/state_model_v32_extra_trees_no_counter_state/"
    "model_mixed_per_head.pkl"
)
RNG_SEED = 20260813
BOOTSTRAP_REPEATS = 20_000
SELECTED_SEED = 20270618
SEED_SPECS = [
    (
        20270614,
        BOUNDARY_ROOT / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_smooth02_seed20270614/config.yaml",
        BOUNDARY_ROOT / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_smooth02_seed20270614/ckpt_step_1000.pt",
    ),
    (
        20270615,
        BOUNDARY_ROOT / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_smooth02_seed20270615/config.yaml",
        BOUNDARY_ROOT / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_smooth02_seed20270615/ckpt_step_1000.pt",
    ),
    (
        20270616,
        BOUNDARY_ROOT / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_smooth02_seed20270616/config.yaml",
        BOUNDARY_ROOT / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_smooth02_seed20270616/ckpt_step_1000.pt",
    ),
    (
        20270617,
        BOUNDARY_ROOT / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_smooth02_seed20270617/config.yaml",
        BOUNDARY_ROOT / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_smooth02_seed20270617/ckpt_step_1000.pt",
    ),
    (
        20270618,
        BOUNDARY_ROOT / "paper_experiments/experiments/truncated_retraining/candidates/optimistic_smooth02_seed18/config.yaml",
        BOUNDARY_ROOT / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_seed18_smooth02/ckpt_step_1000.pt",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model(config_path: Path, checkpoint_path: Path, input_dim: int, device: str):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = L3BoundaryModel(input_dim=input_dim, cfg=config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def infer_clip(model: L3BoundaryModel, l2_payload: dict[str, Any], device: str):
    values = np.asarray(l2_payload["l2"], dtype=np.float32)
    mask = np.asarray(l2_payload["l2_mask"], dtype=np.bool_)
    with torch.no_grad():
        values_t = torch.from_numpy(values).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
        _, latent, auxiliary, _, score = model(values_t, mask_t)
    return (
        values,
        mask,
        latent.squeeze(0).cpu().numpy(),
        auxiliary.squeeze(0).cpu().numpy(),
        score.squeeze(0).cpu().numpy(),
    )


def summarize_trajectory(values: np.ndarray, lag: int = 8) -> np.ndarray:
    """Return ten fixed summary statistics for each trajectory coordinate."""
    if values.ndim == 1:
        values = values[:, None]
    frame_count, dimension = values.shape
    if frame_count == 0:
        return np.zeros(10 * dimension, dtype=np.float64)
    one_difference = (
        np.abs(np.diff(values, axis=0))
        if frame_count > 1
        else np.zeros((1, dimension))
    )
    lag_difference = (
        np.abs(values[lag:] - values[:-lag])
        if frame_count > lag
        else np.zeros((1, dimension))
    )
    signs = np.sign(values)
    sign_changes = (
        np.mean(signs[1:] != signs[:-1], axis=0)
        if frame_count > 1
        else np.zeros(dimension)
    )
    statistics = np.stack(
        [
            np.mean(values, axis=0),
            np.std(values, axis=0),
            values[0],
            values[-1],
            values[-1] - values[0],
            np.max(values, axis=0) - np.min(values, axis=0),
            np.mean(one_difference, axis=0),
            np.max(one_difference, axis=0),
            np.mean(lag_difference, axis=0),
            sign_changes,
        ],
        axis=1,
    )
    return np.nan_to_num(statistics.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def context_window(rows: list[np.ndarray], index: int, radius: int) -> np.ndarray:
    if not rows:
        raise ValueError("Cannot build context from an empty sequence")
    dim = int(rows[0].shape[0])
    zero = np.zeros(dim, dtype=np.float32)
    return np.concatenate(
        [rows[index + offset] if 0 <= index + offset < len(rows) else zero for offset in range(-radius, radius + 1)]
    ).astype(np.float32, copy=False)


def clip_id(value: Any) -> str:
    name = Path(str(value or "")).name
    for suffix in ("_l2.json", ".tactical.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def context_by_clip(context: dict[str, Any], selected_clips: set[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for file_entry in context.get("files", []):
        cid = clip_id(file_entry.get("file"))
        if cid in selected_clips:
            output[cid] = file_entry
    missing = selected_clips - set(output)
    if missing:
        raise KeyError(f"Raw context is missing {len(missing)} selected clips: {sorted(missing)[:5]}")
    return output


def auxiliary_summaries(
    raw_context: dict[str, Any],
    selected_clips: set[str],
    l2_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: str,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    entries = context_by_clip(raw_context, selected_clips)
    first_clip = sorted(selected_clips)[0]
    first_l2 = read_json(l2_root / f"{first_clip}_l2.json")
    model = load_model(config_path, checkpoint_path, len(first_l2["feature_names"]), device)
    summaries: dict[tuple[str, int], np.ndarray] = {}
    segment_count = 0
    for cid in sorted(selected_clips):
        l2 = read_json(l2_root / f"{cid}_l2.json")
        frames = np.asarray(l2.get("frames", np.arange(len(l2["l2"]))), dtype=np.int64)
        _, mask, _, auxiliary, _ = infer_clip(model, l2, device)
        for position, segment in enumerate(entries[cid].get("segments", [])):
            segment_idx = int(segment.get("segment_idx", position))
            start = int(segment["frame_start"])
            end = int(segment["frame_end"])
            take = (frames >= start) & (frames <= end) & mask
            if not np.any(take):
                raise ValueError(f"No valid auxiliary frames for {cid}:{segment_idx} [{start}, {end}]")
            summary = summarize_trajectory(auxiliary[take]).astype(np.float32)
            if summary.shape != (30,) or not np.all(np.isfinite(summary)):
                raise ValueError(f"Invalid auxiliary summary for {cid}:{segment_idx}: {summary.shape}")
            summaries[(cid, segment_idx)] = summary
            segment_count += 1
    return summaries, {
        "clips": len(selected_clips),
        "segments": segment_count,
        "summary_dimensions": 30,
        "config": str(config_path.relative_to(BOUNDARY_ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(BOUNDARY_ROOT)),
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint_path),
    }


def auxiliary_matrix(
    samples: list[dict[str, Any]],
    raw_context: dict[str, Any],
    summaries: dict[tuple[str, int], np.ndarray],
    radius: int,
) -> np.ndarray:
    selected_clips = {str(sample["clip_id"]) for sample in samples}
    entries = context_by_clip(raw_context, selected_clips)
    windows: dict[tuple[str, int], np.ndarray] = {}
    for cid, entry in entries.items():
        segments = list(entry.get("segments", []))
        indices = [int(segment.get("segment_idx", pos)) for pos, segment in enumerate(segments)]
        rows = [summaries[(cid, idx)] for idx in indices]
        for position, idx in enumerate(indices):
            windows[(cid, idx)] = context_window(rows, position, radius)
    output = np.stack(
        [windows[(str(sample["clip_id"]), int(sample["segment_idx"]))] for sample in samples]
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("Auxiliary feature matrix contains non-finite values")
    return output


def predictions_for_representation(
    samples: list[dict[str, Any]],
    raw_samples: list[dict[str, Any]],
    stabilized_samples: list[dict[str, Any]],
    feature_names: list[str],
    raw_context: dict[str, Any],
    summaries: dict[tuple[str, int], np.ndarray],
    representation: str,
) -> dict[str, np.ndarray]:
    pairs = paired_feature_indices(feature_names)
    predictions: dict[str, np.ndarray] = {}
    for head, method in METHODS.items():
        raw_x = feature_matrix(raw_samples, method, len(feature_names), pairs)
        stabilized_x = feature_matrix(stabilized_samples, method, len(feature_names), pairs)
        base_x = np.concatenate([raw_x, stabilized_x], axis=1)
        aux_x = auxiliary_matrix(samples, raw_context, summaries, int(method["radius"]))
        if representation == "engineered_dual":
            x = base_x
        elif representation == "auxiliary_only":
            x = aux_x
        elif representation == "engineered_dual_plus_auxiliary":
            x = np.concatenate([base_x, aux_x], axis=1)
        else:
            raise ValueError(representation)
        predictions[head] = cross_validated_head(samples, x, head, method)
    return predictions


def metric_row(seed: int, representation: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "representation": representation,
        "samples": int(result["footwork"]["n"]),
        "footwork_accuracy": float(result["footwork"]["accuracy"]),
        "footwork_macro_f1": float(result["footwork"]["macro_f1"]),
        "interaction_state_accuracy": float(result["tactical"]["accuracy"]),
        "interaction_state_macro_f1": float(result["tactical"]["macro_f1"]),
        "joint_exact_accuracy": float(result["joint_primary_accuracy"]),
    }


def prediction_rows(
    seed: int,
    representation: str,
    samples: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for index, sample in enumerate(samples):
        rows.append(
            {
                "seed": seed,
                "representation": representation,
                "sample_index": index,
                "clip_id": sample["clip_id"],
                "source_group": sample["source_group"],
                "segment_idx": sample["segment_idx"],
                "side": sample["side"],
                "footwork_truth": HEAD_VALUES["footwork"][int(sample["footwork"])],
                "footwork_prediction": HEAD_VALUES["footwork"][int(predictions["footwork"][index])],
                "interaction_state_truth": HEAD_VALUES["tactical"][int(sample["tactical"])],
                "interaction_state_prediction": HEAD_VALUES["tactical"][int(predictions["tactical"][index])],
            }
        )
    return rows


def paired_accuracy_inference(
    samples: list[dict[str, Any]],
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    seed: int = RNG_SEED,
) -> list[dict[str, Any]]:
    groups = np.asarray([str(sample["source_group"]) for sample in samples])
    unique = np.asarray(sorted(set(groups.tolist())))
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    outputs: list[dict[str, Any]] = []
    footwork_truth = np.asarray([sample["footwork"] for sample in samples])
    state_truth = np.asarray([sample["tactical"] for sample in samples])
    pairs = {
        "footwork_accuracy": (
            baseline["footwork"] == footwork_truth,
            candidate["footwork"] == footwork_truth,
        ),
        "interaction_state_accuracy": (
            baseline["tactical"] == state_truth,
            candidate["tactical"] == state_truth,
        ),
        "joint_exact_accuracy": (
            (baseline["footwork"] == footwork_truth)
            & (baseline["tactical"] == state_truth),
            (candidate["footwork"] == footwork_truth)
            & (candidate["tactical"] == state_truth),
        ),
    }
    rng = np.random.default_rng(seed)
    for metric_name, (base_correct, candidate_correct) in pairs.items():
        delta = candidate_correct.astype(float) - base_correct.astype(float)
        observed = float(np.mean(delta))
        randomization = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(unique)):
            signed = np.concatenate(
                [delta[by_group[group]] * signs[index] for index, group in enumerate(unique)]
            )
            randomization.append(float(np.mean(signed)))
        bootstrap = np.empty(BOOTSTRAP_REPEATS, dtype=float)
        for repeat in range(BOOTSTRAP_REPEATS):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            take = np.concatenate([by_group[group] for group in chosen])
            bootstrap[repeat] = float(np.mean(delta[take]))
        outputs.append(
            {
                "metric": metric_name,
                "delta": observed,
                "source_bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "source_bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
                "source_sign_randomization_p_two_sided": float(
                    np.mean(np.abs(randomization) >= abs(observed) - 1e-12)
                ),
                "improved_sources": int(
                    sum(np.mean(delta[by_group[group]]) > 0 for group in unique)
                ),
                "regressed_sources": int(
                    sum(np.mean(delta[by_group[group]]) < 0 for group in unique)
                ),
                "tied_sources": int(
                    sum(np.mean(delta[by_group[group]]) == 0 for group in unique)
                ),
            }
        )
    return outputs


def paired_macro_f1_inference(
    samples: list[dict[str, Any]],
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    seed: int = RNG_SEED + 1,
) -> list[dict[str, Any]]:
    groups = np.asarray([str(sample["source_group"]) for sample in samples])
    unique = np.asarray(sorted(set(groups.tolist())))
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    outputs = []
    for head, metric_name in (
        ("footwork", "footwork_macro_f1"),
        ("tactical", "interaction_state_macro_f1"),
    ):
        truth = np.asarray([sample[head] for sample in samples], dtype=np.int64)
        class_indices = list(range(len(HEAD_VALUES[head])))

        def score(take: np.ndarray, prediction: np.ndarray) -> float:
            return float(
                f1_score(
                    truth[take],
                    prediction[take],
                    labels=class_indices,
                    average="macro",
                    zero_division=0,
                )
            )

        all_indices = np.arange(len(samples), dtype=np.int64)
        observed = score(all_indices, candidate[head]) - score(all_indices, baseline[head])
        bootstrap = np.empty(BOOTSTRAP_REPEATS, dtype=float)
        for repeat in range(BOOTSTRAP_REPEATS):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            take = np.concatenate([by_group[group] for group in chosen])
            bootstrap[repeat] = score(take, candidate[head]) - score(take, baseline[head])
        outputs.append(
            {
                "metric": metric_name,
                "delta": observed,
                "source_bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "source_bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
                "source_sign_randomization_p_two_sided": "",
                "improved_sources": "",
                "regressed_sources": "",
                "tied_sources": "",
            }
        )
    return outputs


def aggregate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "footwork_accuracy",
        "footwork_macro_f1",
        "interaction_state_accuracy",
        "interaction_state_macro_f1",
        "joint_exact_accuracy",
    )
    output = []
    for representation in ("auxiliary_only", "engineered_dual_plus_auxiliary"):
        selected = [row for row in rows if row["representation"] == representation]
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            output.append(
                {
                    "representation": representation,
                    "metric": metric,
                    "runs": len(values),
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--raw-context", type=Path, default=DEFAULT_RAW_CONTEXT)
    parser.add_argument("--stabilized-context", type=Path, default=DEFAULT_STABILIZED_CONTEXT)
    parser.add_argument("--l2-root", type=Path, default=DEFAULT_L2)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(args.annotations)
    raw_context = read_json(args.raw_context)
    stabilized_context = read_json(args.stabilized_context)
    with args.base_model.open("rb") as handle:
        artifact = pickle.load(handle)
    feature_names = list(artifact["feature_names"])
    raw_samples = build_samples(raw_context, annotations, feature_names)
    stabilized_unordered = build_samples(stabilized_context, annotations, feature_names)
    stabilized_by_key = {sample_key(sample): sample for sample in stabilized_unordered}
    keys = [sample_key(sample) for sample in raw_samples]
    if set(keys) != set(stabilized_by_key):
        raise ValueError("Raw and stabilized sample keys do not align")
    stabilized_samples = [stabilized_by_key[key] for key in keys]
    if len(raw_samples) != 652 or len({sample["clip_id"] for sample in raw_samples}) != 50:
        raise ValueError(f"Unexpected sample set: {len(raw_samples)}")
    selected_clips = {str(sample["clip_id"]) for sample in raw_samples}

    metric_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    prediction_sets: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    seed_provenance: dict[str, Any] = {}

    for seed, config_path, checkpoint_path in SEED_SPECS:
        summaries, provenance = auxiliary_summaries(
            raw_context,
            selected_clips,
            args.l2_root,
            config_path,
            checkpoint_path,
            args.device,
        )
        seed_provenance[str(seed)] = provenance
        representations = ["auxiliary_only", "engineered_dual_plus_auxiliary"]
        if seed == SELECTED_SEED:
            representations.insert(0, "engineered_dual")
        for representation in representations:
            predictions = predictions_for_representation(
                raw_samples,
                raw_samples,
                stabilized_samples,
                feature_names,
                raw_context,
                summaries,
                representation,
            )
            prediction_sets[(seed, representation)] = predictions
            result = evaluate_prediction_set(raw_samples, predictions)
            metric_rows.append(metric_row(seed, representation, result))
            all_prediction_rows.extend(
                prediction_rows(seed, representation, raw_samples, predictions)
            )

    selected_base = prediction_sets[(SELECTED_SEED, "engineered_dual")]
    selected_augmented = prediction_sets[(SELECTED_SEED, "engineered_dual_plus_auxiliary")]
    paired = paired_accuracy_inference(raw_samples, selected_base, selected_augmented)
    paired.extend(paired_macro_f1_inference(raw_samples, selected_base, selected_augmented))
    seed_aggregate = aggregate_seed_rows(metric_rows)

    write_csv(args.out_dir / "per_run_metrics.csv", metric_rows)
    write_csv(args.out_dir / "five_checkpoint_summary.csv", seed_aggregate)
    write_csv(args.out_dir / "selected_paired_source_inference.csv", paired)
    write_csv(args.out_dir / "predictions.csv", all_prediction_rows)

    selected_lookup = {
        row["representation"]: row
        for row in metric_rows
        if int(row["seed"]) == SELECTED_SEED
    }
    report = {
        "schema_version": "downstream_auxiliary_classifier_ablation_v1",
        "protocol": "PROTOCOL.md",
        "dataset": {
            "clips": 50,
            "source_videos": 10,
            "segments": 326,
            "per_fencer_samples": len(raw_samples),
            "label_counts": {
                head: dict(
                    sorted(Counter(HEAD_VALUES[head][int(sample[head])] for sample in raw_samples).items())
                )
                for head in HEAD_VALUES
            },
        },
        "selected_seed": SELECTED_SEED,
        "selected_results": selected_lookup,
        "paired_source_inference": paired,
        "five_checkpoint_summary": seed_aggregate,
        "inputs": {
            "annotations": str(args.annotations),
            "annotations_file_count": len(list(args.annotations.glob("*.json"))),
            "raw_context": str(args.raw_context),
            "raw_context_sha256": sha256(args.raw_context),
            "stabilized_context": str(args.stabilized_context),
            "stabilized_context_sha256": sha256(args.stabilized_context),
            "base_model": str(args.base_model),
            "base_model_sha256": sha256(args.base_model),
            "l2_root": str(args.l2_root),
        },
        "reproducibility": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": args.device,
            "random_seed": RNG_SEED,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "classifier": {
                "n_estimators": 180,
                "min_samples_leaf": 5,
                "max_features": 0.45,
                "class_weight": "balanced_subsample",
            },
            "seed_checkpoints": seed_provenance,
        },
    }
    write_json(args.out_dir / "summary.json", report)

    base = selected_lookup["engineered_dual"]
    augmented = selected_lookup["engineered_dual_plus_auxiliary"]
    auxiliary = selected_lookup["auxiliary_only"]
    lines = [
        "# Downstream ReLIS Auxiliary-Feature Ablation",
        "",
        "All results use the same 50 exchanges, 10 leave-one-source-video-out folds,",
        "326 usable segments, and 652 per-fencer samples.",
        "",
        "| Representation | Footwork acc. | Footwork macro F1 | State acc. | State macro F1 | Joint exact acc. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in (
        ("Engineered dual view", base),
        ("ReLIS auxiliary only", auxiliary),
        ("Engineered + ReLIS auxiliary", augmented),
    ):
        lines.append(
            f"| {label} | {row['footwork_accuracy']:.4f} | {row['footwork_macro_f1']:.4f} | "
            f"{row['interaction_state_accuracy']:.4f} | {row['interaction_state_macro_f1']:.4f} | "
            f"{row['joint_exact_accuracy']:.4f} |"
        )
    lines.extend(["", "## Selected-run paired source inference", ""])
    for row in paired:
        p_value = row["source_sign_randomization_p_two_sided"]
        p_text = "" if p_value == "" else f", exact source-sign p={float(p_value):.4f}"
        lines.append(
            f"- {row['metric']}: delta {row['delta']:+.4f}, 95% source-bootstrap CI "
            f"[{row['source_bootstrap_ci95_low']:+.4f}, {row['source_bootstrap_ci95_high']:+.4f}]"
            f"{p_text}"
        )
    lines.extend(
        [
            "",
            "The auxiliary features are evaluated jointly and do not identify individual",
            "coordinates as fixed semantic axes. This feature ablation does not compare",
            "alternative segmentation sources.",
            "",
        ]
    )
    (args.out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
