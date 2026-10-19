#!/usr/bin/env python3
"""Nested source-video study for state prediction on ReLIS-proposed segments."""

from __future__ import annotations

import os
import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score


BOUNDARY_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = BOUNDARY_ROOT / "boundary_segmentation"
STATE_ROOT = Path(os.environ.get("RELIS_STATE_ROOT", "data/state"))
STATE_SCRIPTS = STATE_ROOT / "experiments/scripts"
sys.path = [
    str(STATE_SCRIPTS),
    str(STATE_ROOT),
    str(BOUNDARY_ROOT),
    *[value for value in sys.path if value not in {str(STATE_SCRIPTS), str(STATE_ROOT), str(BOUNDARY_ROOT)}],
]

from experiments.scripts import experiment_camera_integrated_full_state as full_state  # noqa: E402
from experiments.scripts import experiment_primary_trace_state as primary_trace  # noqa: E402
from experiments.scripts import experiment_secondary_trace_gate as trace_gate  # noqa: E402
from state_model import labels  # noqa: E402
from state_model.features import paired_feature_indices  # noqa: E402

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
from models.l3_model import L3BoundaryModel  # noqa: E402


DEFAULT_OUT = BOUNDARY_ROOT / "paper_experiments/round2/downstream_state_model_search"
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
DEFAULT_RAW_L2 = STATE_ROOT / (
    "experiments/aaai_tactical_principles/full_exchange_regions/"
    "pipeline_legacy_v32/working/legacy_auto/l2_out"
)
DEFAULT_STABILIZED_L2 = STATE_ROOT / (
    "experiments/aaai_tactical_principles/camera_robust_state_v1/translation_l2_out_llm"
)
DEFAULT_BASE_MODEL = STATE_ROOT / (
    "artifacts/state_models/state_model_v32_extra_trees_no_counter_state/"
    "model_mixed_per_head.pkl"
)
DEFAULT_RELIS_CONFIG = BOUNDARY_ROOT / (
    "paper_experiments/experiments/truncated_retraining/"
    "candidates/optimistic_smooth02_seed18/config.yaml"
)
DEFAULT_RELIS_CHECKPOINT = BOUNDARY_ROOT / (
    "paper_experiments/experiments/truncated_retraining/"
    "runs/short_sparse_seed18_smooth02/ckpt_step_1000.pt"
)

HEADS = ("primary_footwork", "primary_tactical")
HEAD_NAMES = {
    "primary_footwork": labels.FOOTWORK,
    "primary_tactical": labels.TACTICAL,
}
VIEW_NAMES = ("engineered_dual", "l2_trace_stabilized", "relis_full")
CONFIGS = {
    "balanced_leaf5": {
        "n_estimators": 180,
        "min_samples_leaf": 5,
        "max_features": 0.45,
        "class_weight": "balanced_subsample",
    },
    "plain_leaf2": {
        "n_estimators": 180,
        "min_samples_leaf": 2,
        "max_features": 0.45,
        "class_weight": None,
    },
}
PAIR_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
FUSION_WEIGHTS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.75, 0.25, 0.0),
    (0.5, 0.5, 0.0),
    (0.25, 0.75, 0.0),
    (0.75, 0.0, 0.25),
    (0.5, 0.0, 0.5),
    (0.25, 0.0, 0.75),
    (0.0, 0.75, 0.25),
    (0.0, 0.5, 0.5),
    (0.0, 0.25, 0.75),
    (0.5, 0.25, 0.25),
    (0.25, 0.5, 0.25),
    (0.25, 0.25, 0.5),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
)
SELECTION_TOLERANCE = 0.005
CANDIDATE_SPECS = {
    "engineered_nested": (
        ((1.0, 0.0, 0.0),),
        PAIR_WEIGHTS,
    ),
    "no_relis": (
        tuple(weights for weights in FUSION_WEIGHTS if weights[2] == 0.0),
        PAIR_WEIGHTS,
    ),
    "full_no_pair": (
        FUSION_WEIGHTS,
        (0.0,),
    ),
    "full": (
        FUSION_WEIGHTS,
        PAIR_WEIGHTS,
    ),
}
RELIS_STATS = (
    "mean",
    "std",
    "minimum",
    "q25",
    "median",
    "q75",
    "maximum",
    "first_third_mean",
    "last_third_mean",
    "late_minus_early",
    "slope",
    "mean_abs_diff",
    "max_abs_diff",
    "sign_changes_per_transition",
)


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


def clip_id(value: Any) -> str:
    return full_state.clip_id(value)


def aligned_probabilities(model: Any, x: np.ndarray, classes: int) -> np.ndarray:
    output = np.zeros((len(x), classes), dtype=np.float64)
    probabilities = model.predict_proba(x)
    for position, value in enumerate(model.classes_):
        output[:, int(value)] = probabilities[:, position]
    return output


def classifier(config: dict[str, Any], seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=int(config["n_estimators"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=config["max_features"],
        class_weight=config["class_weight"],
        random_state=int(seed),
        n_jobs=-1,
    )


def trajectory_statistics(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        values = values[:, None]
    output: list[float] = []
    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        column = column[np.isfinite(column)]
        if not len(column):
            output.extend([0.0] * len(RELIS_STATS))
            continue
        third = max(1, int(math.ceil(len(column) / 3)))
        early = float(np.mean(column[:third]))
        late = float(np.mean(column[-third:]))
        positions = np.linspace(0.0, 1.0, len(column))
        centered = positions - float(np.mean(positions))
        denominator = float(np.sum(centered**2))
        slope = (
            float(np.sum(centered * (column - float(np.mean(column)))) / denominator)
            if denominator
            else 0.0
        )
        difference = np.abs(np.diff(column)) if len(column) > 1 else np.zeros(1)
        signs = np.sign(column)
        signs = signs[signs != 0]
        changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        output.extend(
            [
                float(np.mean(column)),
                float(np.std(column)),
                float(np.min(column)),
                float(np.quantile(column, 0.25)),
                float(np.median(column)),
                float(np.quantile(column, 0.75)),
                float(np.max(column)),
                early,
                late,
                late - early,
                slope,
                float(np.mean(difference)),
                float(np.max(difference)),
                changes / max(1, len(signs) - 1),
            ]
        )
    return np.nan_to_num(np.asarray(output, dtype=np.float32))


def load_relis_model(
    config_path: Path,
    checkpoint_path: Path,
    input_dim: int,
    device: str,
) -> L3BoundaryModel:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = L3BoundaryModel(input_dim=input_dim, cfg=config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def infer_relis(
    model: L3BoundaryModel,
    payload: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(payload["l2"], dtype=np.float32)
    mask = np.asarray(payload["l2_mask"], dtype=bool)
    frames = np.asarray(payload.get("frames", np.arange(len(values))), dtype=np.int64)
    with torch.no_grad():
        tensor = torch.from_numpy(values).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
        _, z, auxiliary, phi, score = model(tensor, mask_tensor)
    return (
        frames,
        mask,
        z.squeeze(0).cpu().numpy(),
        auxiliary.squeeze(0).cpu().numpy(),
        phi.squeeze(0).cpu().numpy(),
        score.squeeze(0).cpu().numpy(),
    )


def segment_bounds(segment: dict[str, Any]) -> tuple[int, int]:
    return (
        int(segment.get("frame_start", segment.get("frame_start_idx", 0))),
        int(segment.get("frame_end", segment.get("frame_end_excl_idx", 0))),
    )


def relis_segment_features(
    raw_context: dict[str, Any],
    l2_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: str,
) -> tuple[dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    files = {
        clip_id(entry.get("file")): entry
        for entry in raw_context.get("files", [])
        if isinstance(entry, dict)
    }
    first_clip = sorted(files)[0]
    first_l2 = full_state.read_json(l2_root / f"{first_clip}_l2.json")
    feature_names = list(first_l2["feature_names"])
    model = load_relis_model(
        config_path,
        checkpoint_path,
        len(first_l2["feature_names"]),
        device,
    )
    output: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    dimensions: dict[str, int] = {}
    for cid in sorted(files):
        payload = full_state.read_json(l2_root / f"{cid}_l2.json")
        if list(payload.get("feature_names", [])) != feature_names:
            raise ValueError(f"L2 feature ordering differs for {cid}")
        frames, mask, z, auxiliary, phi, score = infer_relis(model, payload, device)
        for position, segment in enumerate(files[cid].get("segments", [])):
            segment_index = int(segment.get("segment_idx", position))
            start, end = segment_bounds(segment)
            take = (frames >= start) & (frames <= end) & mask
            if not np.any(take):
                raise ValueError(f"No valid ReLIS frames for {cid}:{segment_index}")
            z_summary = trajectory_statistics(z[take])
            compact = np.concatenate(
                [
                    trajectory_statistics(auxiliary[take]),
                    trajectory_statistics(phi[take]),
                    trajectory_statistics(score[take]),
                ]
            ).astype(np.float32)
            output[(cid, segment_index)] = (compact, z_summary)
            dimensions = {
                "compact": int(len(compact)),
                "latent_summary": int(len(z_summary)),
            }
    return output, {
        "clips": len(files),
        "segments": len(output),
        **dimensions,
        "statistics_per_coordinate": len(RELIS_STATS),
    }


def segment_context_matrix(
    samples: list[dict[str, Any]],
    raw_context: dict[str, Any],
    features: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for entry in raw_context.get("files", []):
        if not isinstance(entry, dict):
            continue
        cid = clip_id(entry.get("file"))
        segments = list(entry.get("segments", []))
        indices = [int(segment.get("segment_idx", position)) for position, segment in enumerate(segments)]
        compact_rows = [features[(cid, index)][0] for index in indices]
        latent_rows = [features[(cid, index)][1] for index in indices]
        compact_zero = np.zeros_like(compact_rows[0])
        latent_zero = np.zeros_like(latent_rows[0])
        for position, index in enumerate(indices):
            compact = np.concatenate(
                [
                    compact_rows[position + offset]
                    if 0 <= position + offset < len(compact_rows)
                    else compact_zero
                    for offset in range(-radius, radius + 1)
                ]
            )
            latent = np.concatenate(
                [
                    latent_rows[position + offset]
                    if 0 <= position + offset < len(latent_rows)
                    else latent_zero
                    for offset in range(-radius, radius + 1)
                ]
            )
            windows[(cid, index)] = (compact, latent)
    compact_matrix = np.stack(
        [windows[(str(sample["clip_id"]), int(sample["segment_idx"]))][0] for sample in samples]
    )
    latent_matrix = np.stack(
        [windows[(str(sample["clip_id"]), int(sample["segment_idx"]))][1] for sample in samples]
    )
    return compact_matrix.astype(np.float32), latent_matrix.astype(np.float32)


def gate_by_side(x: np.ndarray, sides: np.ndarray) -> np.ndarray:
    left = (sides == "left").astype(np.float32)[:, None]
    right = (sides == "right").astype(np.float32)[:, None]
    return np.concatenate([x * left, x * right], axis=1)


def prepare_view_matrices(
    samples: list[dict[str, Any]],
    feature_names: list[str],
    raw_context: dict[str, Any],
    raw_l2_root: Path,
    stabilized_l2_root: Path,
    relis_features: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pairs = paired_feature_indices(feature_names)
    trace_raw, trace_stabilized, trace_names = trace_gate.build_trace_matrices(
        samples,
        trace_gate.index_l2(raw_l2_root),
        trace_gate.index_l2(stabilized_l2_root),
    )
    sides = np.asarray([sample["side"] for sample in samples])
    views: dict[str, dict[str, Any]] = {head: {} for head in HEADS}
    for head in HEADS:
        method = full_state.HEAD_METHODS[head]
        raw_x = full_state.feature_matrix(samples, "raw", method, len(feature_names), pairs)
        stable_x = full_state.feature_matrix(samples, "stabilized", method, len(feature_names), pairs)
        views[head]["engineered_dual"] = {
            "x": np.concatenate([raw_x, stable_x], axis=1).astype(np.float32),
        }
        trace_x = trace_stabilized
        if method["pool_sides"]:
            trace_x = primary_trace.canonicalize_trace_for_right(trace_x, samples, trace_names)
        views[head]["l2_trace_stabilized"] = {"x": trace_x.astype(np.float32)}
        compact, latent = segment_context_matrix(
            samples,
            raw_context,
            relis_features,
            int(method["radius"]),
        )
        if method["pool_sides"]:
            compact = gate_by_side(compact, sides)
        views[head]["relis_full"] = {
            "compact": compact,
            "latent": latent,
        }
        if method["pool_sides"]:
            views[head]["relis_full"]["_gate_sides"] = sides
    return views, {
        "engineered_dimensions": {
            head: int(views[head]["engineered_dual"]["x"].shape[1]) for head in HEADS
        },
        "trace_dimensions": int(len(trace_names)),
        "relis_dimensions": {
            head: {
                f"{key}_pre_pca" if key == "latent" else key: int(value.shape[1])
                for key, value in views[head]["relis_full"].items()
                if not key.startswith("_")
            }
            for head in HEADS
        },
        "relis_latent_pca_max_components": 32,
        "relis_latent_pca_output_dimensions": {
            head: 64 if full_state.HEAD_METHODS[head]["pool_sides"] else 32
            for head in HEADS
        },
    }


def transform_relis(
    view: dict[str, np.ndarray],
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    latent = view["latent"]
    components = min(32, int(np.sum(train)) - 1, latent.shape[1])
    if components < 1:
        raise ValueError("Insufficient training rows for ReLIS PCA")
    pca = PCA(n_components=components, svd_solver="randomized", random_state=int(seed))
    train_latent = pca.fit_transform(latent[train])
    test_latent = pca.transform(latent[test])
    if "_gate_sides" in view:
        sides = view["_gate_sides"]
        train_latent = gate_by_side(train_latent, sides[train])
        test_latent = gate_by_side(test_latent, sides[test])
    return (
        np.concatenate([view["compact"][train], train_latent], axis=1),
        np.concatenate([view["compact"][test], test_latent], axis=1),
    )


def fit_predict_view(
    samples: list[dict[str, Any]],
    view: dict[str, np.ndarray],
    view_name: str,
    head: str,
    config: dict[str, Any],
    train_group: np.ndarray,
    test_group: np.ndarray,
    seed: int,
) -> np.ndarray:
    values = HEAD_NAMES[head]
    classes = len(values)
    y = np.asarray([sample[head] for sample in samples], dtype=np.int64)
    sides = np.asarray([sample["side"] for sample in samples])
    method = full_state.HEAD_METHODS[head]
    output = np.zeros((int(np.sum(test_group)), classes), dtype=np.float64)
    test_positions = np.flatnonzero(test_group)

    def run(train: np.ndarray, test: np.ndarray, local_seed: int) -> np.ndarray:
        if view_name == "relis_full":
            train_x, test_x = transform_relis(view, train, test, local_seed)
        else:
            train_x = view["x"][train]
            test_x = view["x"][test]
        model = classifier(config, local_seed)
        model.fit(train_x, y[train])
        return aligned_probabilities(model, test_x, classes)

    if method["pool_sides"]:
        return run(train_group, test_group, seed)
    for side_index, side in enumerate(("left", "right")):
        train = train_group & (sides == side)
        test = test_group & (sides == side)
        predicted = run(train, test, seed + side_index)
        positions = np.flatnonzero(test)
        lookup = {global_index: local_index for local_index, global_index in enumerate(test_positions)}
        output[[lookup[index] for index in positions]] = predicted
    return output


def source_objective(
    samples: list[dict[str, Any]],
    head: str,
    prediction: np.ndarray,
    selected: np.ndarray,
) -> tuple[float, float, float]:
    y = np.asarray([sample[head] for sample in samples], dtype=np.int64)
    groups = np.asarray([str(sample["source_group"]) for sample in samples])
    labels_all = list(range(len(HEAD_NAMES[head])))
    rows = []
    for group in sorted(set(groups[selected])):
        take = selected & (groups == group)
        present = sorted(set(int(value) for value in y[take]))
        accuracy = float(accuracy_score(y[take], prediction[take]))
        macro = float(
            f1_score(
                y[take],
                prediction[take],
                labels=present or labels_all,
                average="macro",
                zero_division=0,
            )
        )
        rows.append((accuracy + macro) / 2.0)
    pooled_accuracy = float(accuracy_score(y[selected], prediction[selected]))
    pooled_macro = float(
        f1_score(
            y[selected],
            prediction[selected],
            labels=labels_all,
            average="macro",
            zero_division=0,
        )
    )
    return float(np.mean(rows)), pooled_macro, pooled_accuracy


def pair_prior(
    samples: list[dict[str, Any]],
    head: str,
    selected: np.ndarray,
    smoothing: float = 1.0,
) -> np.ndarray:
    y = np.asarray([sample[head] for sample in samples], dtype=np.int64)
    classes = len(HEAD_NAMES[head])
    counts = np.full((classes, classes), float(smoothing), dtype=np.float64)
    by_segment: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for index, sample in enumerate(samples):
        if selected[index]:
            by_segment[(str(sample["clip_id"]), int(sample["segment_idx"]))][str(sample["side"])] = index
    for sides in by_segment.values():
        if "left" in sides and "right" in sides:
            counts[y[sides["left"]], y[sides["right"]]] += 1.0
    return counts / float(np.sum(counts))


def paired_predictions(
    samples: list[dict[str, Any]],
    probabilities: np.ndarray,
    selected: np.ndarray,
    prior: np.ndarray,
    weight: float,
) -> np.ndarray:
    prediction = np.argmax(probabilities, axis=1).astype(np.int64)
    if weight <= 0:
        return prediction
    by_segment: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for index, sample in enumerate(samples):
        if selected[index]:
            by_segment[(str(sample["clip_id"]), int(sample["segment_idx"]))][str(sample["side"])] = index
    log_prior = np.log(np.maximum(prior, 1e-12))
    for sides in by_segment.values():
        if "left" not in sides or "right" not in sides:
            continue
        left = sides["left"]
        right = sides["right"]
        scores = (
            np.log(np.maximum(probabilities[left], 1e-12))[:, None]
            + np.log(np.maximum(probabilities[right], 1e-12))[None, :]
            + float(weight) * log_prior
        )
        left_value, right_value = np.unravel_index(int(np.argmax(scores)), scores.shape)
        prediction[left] = int(left_value)
        prediction[right] = int(right_value)
    return prediction


def combine_probabilities(probabilities: dict[str, np.ndarray], weights: tuple[float, float, float]) -> np.ndarray:
    output = np.zeros_like(probabilities[VIEW_NAMES[0]])
    for view_name, weight in zip(VIEW_NAMES, weights):
        output += float(weight) * probabilities[view_name]
    row_sum = np.sum(output, axis=1, keepdims=True)
    return np.divide(output, row_sum, out=np.zeros_like(output), where=row_sum > 0)


def choose_fusion(
    samples: list[dict[str, Any]],
    head: str,
    groups: np.ndarray,
    outer_train: np.ndarray,
    inner_groups: list[str],
    probabilities: dict[str, np.ndarray],
    weight_options: tuple[tuple[float, float, float], ...],
    pair_options: tuple[float, ...],
) -> tuple[tuple[float, float, float], float, tuple[float, float, float], list[dict[str, Any]]]:
    grid: list[dict[str, Any]] = []
    for weights in weight_options:
        fused = combine_probabilities(probabilities, weights)
        for pair_weight in pair_options:
            prediction = np.full(len(samples), -1, dtype=np.int64)
            for inner_group in inner_groups:
                inner_test = groups == inner_group
                prior = pair_prior(samples, head, outer_train & ~inner_test)
                decoded = paired_predictions(
                    samples,
                    fused,
                    inner_test,
                    prior,
                    pair_weight,
                )
                prediction[inner_test] = decoded[inner_test]
            objective = source_objective(samples, head, prediction, outer_train)
            grid.append(
                {
                    "weights": list(weights),
                    "pair_weight": float(pair_weight),
                    "objective": [float(value) for value in objective],
                }
            )
    maximum = max(float(row["objective"][0]) for row in grid)
    eligible = [
        row for row in grid if float(row["objective"][0]) >= maximum - SELECTION_TOLERANCE
    ]

    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        weights = tuple(float(value) for value in row["weights"])
        nonzero = sum(value > 0 for value in weights)
        uses_pair = float(row["pair_weight"]) > 0
        objective = tuple(float(value) for value in row["objective"])
        return (
            nonzero + int(uses_pair),
            nonzero,
            int(uses_pair),
            -weights[0],
            tuple(-value for value in objective),
            weights,
            float(row["pair_weight"]),
        )

    selected = min(eligible, key=rank)
    return (
        tuple(float(value) for value in selected["weights"]),
        float(selected["pair_weight"]),
        tuple(float(value) for value in selected["objective"]),
        grid,
    )


def broad_mapping(head: str) -> tuple[np.ndarray, list[str]]:
    if head == "primary_footwork":
        return np.asarray([0, 1, 2, 2], dtype=np.int64), ["advance", "retreat", "hold/reset"]
    return np.asarray([3, 0, 1, 2, 3], dtype=np.int64), ["pressure", "attack", "defend", "other"]


def metric_block(
    samples: list[dict[str, Any]],
    head: str,
    prediction: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray([sample[head] for sample in samples], dtype=np.int64)
    groups = np.asarray([str(sample["source_group"]) for sample in samples])
    fine_labels = list(range(len(HEAD_NAMES[head])))
    mapping, broad_names = broad_mapping(head)
    broad_y = mapping[y]
    broad_prediction = mapping[prediction]
    broad_labels = list(range(len(broad_names)))
    per_source = {}
    for group in sorted(set(groups)):
        take = groups == group
        per_source[group] = {
            "accuracy": float(accuracy_score(y[take], prediction[take])),
            "macro_f1_present": float(
                f1_score(
                    y[take],
                    prediction[take],
                    labels=sorted(set(int(value) for value in y[take])),
                    average="macro",
                    zero_division=0,
                )
            ),
            "broad_accuracy": float(
                accuracy_score(broad_y[take], broad_prediction[take])
            ),
            "broad_macro_f1_present": float(
                f1_score(
                    broad_y[take],
                    broad_prediction[take],
                    labels=sorted(set(int(value) for value in broad_y[take])),
                    average="macro",
                    zero_division=0,
                )
            ),
        }
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(
            f1_score(y, prediction, labels=fine_labels, average="macro", zero_division=0)
        ),
        "equal_source_accuracy": float(np.mean([row["accuracy"] for row in per_source.values()])),
        "equal_source_macro_f1_present": float(
            np.mean([row["macro_f1_present"] for row in per_source.values()])
        ),
        "broad_accuracy": float(accuracy_score(broad_y, broad_prediction)),
        "broad_macro_f1": float(
            f1_score(
                broad_y,
                broad_prediction,
                labels=broad_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "equal_source_broad_accuracy": float(
            np.mean([row["broad_accuracy"] for row in per_source.values()])
        ),
        "equal_source_broad_macro_f1_present": float(
            np.mean([row["broad_macro_f1_present"] for row in per_source.values()])
        ),
        "truth": dict(sorted(Counter(HEAD_NAMES[head][int(value)] for value in y).items())),
        "prediction": dict(
            sorted(Counter(HEAD_NAMES[head][int(value)] for value in prediction).items())
        ),
        "per_source": per_source,
    }


def paired_accuracy_inference(
    samples: list[dict[str, Any]],
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int,
    repeats: int = 20_000,
) -> dict[str, Any]:
    groups = np.asarray([str(sample["source_group"]) for sample in samples])
    unique = np.asarray(sorted(set(groups)))
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    difference = candidate.astype(float) - baseline.astype(float)
    group_differences = np.asarray(
        [float(np.mean(difference[by_group[group]])) for group in unique], dtype=float
    )
    observed = float(np.mean(group_differences))
    assignments = []
    for mask in range(1 << len(unique)):
        signs = np.asarray(
            [1.0 if mask & (1 << index) else -1.0 for index in range(len(unique))]
        )
        assignments.append(float(np.mean(group_differences * signs)))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        chosen = rng.integers(0, len(unique), size=len(unique))
        bootstrap[repeat] = float(np.mean(group_differences[chosen]))
    return {
        "delta": observed,
        "source_bootstrap_ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "exact_source_sign_p_two_sided": float(
            np.mean(np.abs(assignments) >= abs(observed) - 1e-12)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--raw-context", type=Path, default=DEFAULT_RAW_CONTEXT)
    parser.add_argument("--stabilized-context", type=Path, default=DEFAULT_STABILIZED_CONTEXT)
    parser.add_argument("--raw-l2", type=Path, default=DEFAULT_RAW_L2)
    parser.add_argument("--stabilized-l2", type=Path, default=DEFAULT_STABILIZED_L2)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--relis-config", type=Path, default=DEFAULT_RELIS_CONFIG)
    parser.add_argument("--relis-checkpoint", type=Path, default=DEFAULT_RELIS_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.base_model.open("rb") as handle:
        base_artifact = pickle.load(handle)
    feature_names = list(base_artifact["feature_names"])
    raw_context = full_state.read_json(args.raw_context)
    stabilized_context = full_state.read_json(args.stabilized_context)
    stabilized_clips = {
        clip_id(row.get("file"))
        for row in stabilized_context.get("files", [])
        if isinstance(row, dict)
    }
    raw_context = {
        **raw_context,
        "files": [
            row
            for row in raw_context.get("files", [])
            if isinstance(row, dict) and clip_id(row.get("file")) in stabilized_clips
        ],
    }
    samples = full_state.build_samples(
        raw_context,
        stabilized_context,
        full_state.load_annotations(args.annotations),
        feature_names,
    )
    if len(samples) != 652 or len({sample["clip_id"] for sample in samples}) != 50:
        raise ValueError(f"Unexpected sample set: {len(samples)} rows")
    sample_keys = [
        (str(sample["clip_id"]), int(sample["segment_idx"]), str(sample["side"]))
        for sample in samples
    ]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("Duplicate clip/segment/side sample keys")
    sides_by_segment: dict[tuple[str, int], set[str]] = defaultdict(set)
    for cid, segment_index, side in sample_keys:
        sides_by_segment[(cid, segment_index)].add(side)
    if any(sides != {"left", "right"} for sides in sides_by_segment.values()):
        raise ValueError("Every retained segment must have exactly one left and one right sample")
    groups = np.asarray([str(sample["source_group"]) for sample in samples])
    group_names = sorted(set(groups))
    if len(group_names) != 10:
        raise ValueError(f"Expected 10 source groups, found {group_names}")

    relis_features, relis_meta = relis_segment_features(
        raw_context,
        args.raw_l2,
        args.relis_config,
        args.relis_checkpoint,
        args.device,
    )
    views, view_meta = prepare_view_matrices(
        samples,
        feature_names,
        raw_context,
        args.raw_l2,
        args.stabilized_l2,
        relis_features,
    )

    baseline_probabilities = {
        head: np.zeros((len(samples), len(HEAD_NAMES[head])), dtype=np.float64) for head in HEADS
    }
    candidate_probabilities = {
        candidate: {
            head: np.zeros((len(samples), len(HEAD_NAMES[head])), dtype=np.float64)
            for head in HEADS
        }
        for candidate in CANDIDATE_SPECS
    }
    candidate_predictions = {
        candidate: {head: np.full(len(samples), -1, dtype=np.int64) for head in HEADS}
        for candidate in CANDIDATE_SPECS
    }
    selections: list[dict[str, Any]] = []

    for outer_index, outer_group in enumerate(group_names):
        outer_train = groups != outer_group
        outer_test = groups == outer_group
        inner_groups = [group for group in group_names if group != outer_group]
        print(f"outer {outer_index + 1}/{len(group_names)} source={outer_group}", flush=True)
        for head_index, head in enumerate(HEADS):
            inner_by_view: dict[str, np.ndarray] = {}
            outer_by_view: dict[str, np.ndarray] = {}
            selected_configs: dict[str, str] = {}
            config_scores: dict[str, dict[str, list[float]]] = {}
            for view_index, view_name in enumerate(VIEW_NAMES):
                inner_candidates: dict[str, np.ndarray] = {}
                config_scores[view_name] = {}
                config_options = (
                    {"balanced_leaf5": CONFIGS["balanced_leaf5"]}
                    if view_name == "engineered_dual"
                    else CONFIGS
                )
                for config_name, config in config_options.items():
                    inner_probabilities = np.zeros(
                        (len(samples), len(HEAD_NAMES[head])), dtype=np.float64
                    )
                    for inner_index, inner_group in enumerate(inner_groups):
                        inner_train = outer_train & (groups != inner_group)
                        inner_test = groups == inner_group
                        predicted = fit_predict_view(
                            samples,
                            views[head][view_name],
                            view_name,
                            head,
                            config,
                            inner_train,
                            inner_test,
                            100_000
                            + outer_index * 10_000
                            + head_index * 2_000
                            + view_index * 400
                            + inner_index * 4,
                        )
                        inner_probabilities[inner_test] = predicted
                    inner_prediction = np.argmax(inner_probabilities, axis=1).astype(np.int64)
                    score = source_objective(samples, head, inner_prediction, outer_train)
                    inner_candidates[config_name] = inner_probabilities
                    config_scores[view_name][config_name] = [float(value) for value in score]
                maximum_config_score = max(
                    values[0] for values in config_scores[view_name].values()
                )
                config_name = (
                    "balanced_leaf5"
                    if "balanced_leaf5" in config_options
                    and config_scores[view_name]["balanced_leaf5"][0]
                    >= maximum_config_score - SELECTION_TOLERANCE
                    else max(
                        config_options,
                        key=lambda name: tuple(config_scores[view_name][name]),
                    )
                )
                selected_configs[view_name] = config_name
                inner_by_view[view_name] = inner_candidates[config_name]
                outer_seed = (
                    5000 + outer_index
                    if head == "primary_tactical" and view_name == "engineered_dual"
                    else 6000 + outer_index * 4
                    if head == "primary_footwork" and view_name == "engineered_dual"
                    else 200_000
                    + outer_index * 10_000
                    + head_index * 2_000
                    + view_index * 100
                )
                outer_by_view[view_name] = fit_predict_view(
                    samples,
                    views[head][view_name],
                    view_name,
                    head,
                    CONFIGS[config_name],
                    outer_train,
                    outer_test,
                    outer_seed,
                )

            baseline_outer = fit_predict_view(
                samples,
                views[head]["engineered_dual"],
                "engineered_dual",
                head,
                CONFIGS["balanced_leaf5"],
                outer_train,
                outer_test,
                5000 + outer_index if full_state.HEAD_METHODS[head]["pool_sides"] else 6000 + outer_index * 4,
            )
            baseline_probabilities[head][outer_test] = baseline_outer
            candidate_selections = {}
            for candidate_name, (weight_options, pair_options) in CANDIDATE_SPECS.items():
                weights, pair_weight, objective, selection_grid = choose_fusion(
                    samples,
                    head,
                    groups,
                    outer_train,
                    inner_groups,
                    inner_by_view,
                    weight_options,
                    pair_options,
                )
                fused_outer = combine_probabilities(outer_by_view, weights)
                fused_outer_full = np.zeros(
                    (len(samples), len(HEAD_NAMES[head])), dtype=np.float64
                )
                fused_outer_full[outer_test] = fused_outer
                outer_prior = pair_prior(samples, head, outer_train)
                decoded_outer = paired_predictions(
                    samples,
                    fused_outer_full,
                    outer_test,
                    outer_prior,
                    pair_weight,
                )
                candidate_probabilities[candidate_name][head][outer_test] = fused_outer
                candidate_predictions[candidate_name][head][outer_test] = decoded_outer[outer_test]
                candidate_selections[candidate_name] = {
                    "weights": list(weights),
                    "pair_weight": pair_weight,
                    "inner_objective": list(objective),
                    "grid": selection_grid,
                }
            selections.append(
                {
                    "outer_source": outer_group,
                    "head": head,
                    "selected_configs": selected_configs,
                    "config_scores": config_scores,
                    "candidates": candidate_selections,
                }
            )

    if any(
        np.any(prediction < 0)
        for predictions in candidate_predictions.values()
        for prediction in predictions.values()
    ):
        raise RuntimeError("Nested evaluation left samples without predictions")
    baseline_predictions = {
        head: np.argmax(probabilities, axis=1).astype(np.int64)
        for head, probabilities in baseline_probabilities.items()
    }
    baseline_metrics = {
        head: metric_block(samples, head, baseline_predictions[head]) for head in HEADS
    }
    candidate_metrics = {
        candidate: {
            head: metric_block(samples, head, predictions[head]) for head in HEADS
        }
        for candidate, predictions in candidate_predictions.items()
    }
    baseline_joint = np.ones(len(samples), dtype=bool)
    candidate_joint = {
        candidate: np.ones(len(samples), dtype=bool) for candidate in CANDIDATE_SPECS
    }
    for head in HEADS:
        truth = np.asarray([sample[head] for sample in samples], dtype=np.int64)
        baseline_joint &= baseline_predictions[head] == truth
        for candidate in CANDIDATE_SPECS:
            candidate_joint[candidate] &= candidate_predictions[candidate][head] == truth
    baseline_joint_accuracy = float(np.mean(baseline_joint))
    candidate_joint_accuracy = {
        candidate: float(np.mean(values)) for candidate, values in candidate_joint.items()
    }
    selected_metrics = candidate_metrics["full"]
    selected_joint_accuracy = candidate_joint_accuracy["full"]

    inference = {}
    for candidate_index, candidate in enumerate(CANDIDATE_SPECS):
        inference[candidate] = {}
        for head_index, head in enumerate(HEADS):
            truth = np.asarray([sample[head] for sample in samples], dtype=np.int64)
            inference[candidate][head] = paired_accuracy_inference(
                samples,
                baseline_predictions[head] == truth,
                candidate_predictions[candidate][head] == truth,
                20260813 + candidate_index * 10 + head_index,
            )
        inference[candidate]["joint_exact"] = paired_accuracy_inference(
            samples,
            baseline_joint,
            candidate_joint[candidate],
            20260900 + candidate_index,
        )
    inference["full_vs_no_relis"] = {}
    for head_index, head in enumerate(HEADS):
        truth = np.asarray([sample[head] for sample in samples], dtype=np.int64)
        inference["full_vs_no_relis"][head] = paired_accuracy_inference(
            samples,
            candidate_predictions["no_relis"][head] == truth,
            candidate_predictions["full"][head] == truth,
            20261000 + head_index,
        )
    inference["full_vs_no_relis"]["joint_exact"] = paired_accuracy_inference(
        samples,
        candidate_joint["no_relis"],
        candidate_joint["full"],
        20261002,
    )

    prediction_rows = []
    probability_rows = []
    for index, sample in enumerate(samples):
        row = {
            "sample_index": index,
            "clip_id": sample["clip_id"],
            "source_group": sample["source_group"],
            "segment_idx": sample["segment_idx"],
            "side": sample["side"],
            "footwork_truth": labels.FOOTWORK[int(sample["primary_footwork"])],
            "footwork_baseline": labels.FOOTWORK[
                int(baseline_predictions["primary_footwork"][index])
            ],
            "state_truth": labels.TACTICAL[int(sample["primary_tactical"])],
            "state_baseline": labels.TACTICAL[
                int(baseline_predictions["primary_tactical"][index])
            ],
        }
        for candidate in CANDIDATE_SPECS:
            row[f"footwork_{candidate}"] = labels.FOOTWORK[
                int(candidate_predictions[candidate]["primary_footwork"][index])
            ]
            row[f"state_{candidate}"] = labels.TACTICAL[
                int(candidate_predictions[candidate]["primary_tactical"][index])
            ]
        prediction_rows.append(row)
        for model_name, by_head in {
            "baseline": baseline_probabilities,
            **candidate_probabilities,
        }.items():
            for head in HEADS:
                for class_index, class_name in enumerate(HEAD_NAMES[head]):
                    probability_rows.append(
                        {
                            "sample_index": index,
                            "model": model_name,
                            "head": head,
                            "class": class_name,
                            "probability": float(by_head[head][index, class_index]),
                        }
                    )
    write_csv(args.out_dir / "predictions.csv", prediction_rows)
    write_csv(args.out_dir / "probabilities.csv", probability_rows)
    write_json(args.out_dir / "nested_selections.json", selections)

    report = {
        "schema_version": "downstream_state_model_nested_source_v1",
        "status": "retrospective_exploratory",
        "protocol": "PROTOCOL.md",
        "dataset": {
            "exchanges": 50,
            "source_videos": 10,
            "segments": 326,
            "per_fencer_samples": len(samples),
            "source_counts": {
                str(key): int(value) for key, value in sorted(Counter(groups).items())
            },
            "label_counts": {
                head: dict(
                    sorted(Counter(HEAD_NAMES[head][int(sample[head])] for sample in samples).items())
                )
                for head in HEADS
            },
        },
        "views": view_meta,
        "relis": {
            **relis_meta,
            "config": str(args.relis_config.relative_to(BOUNDARY_ROOT)),
            "checkpoint": str(args.relis_checkpoint.relative_to(BOUNDARY_ROOT)),
            "config_sha256": sha256(args.relis_config),
            "checkpoint_sha256": sha256(args.relis_checkpoint),
        },
        "baseline": {
            **baseline_metrics,
            "joint_exact_accuracy": baseline_joint_accuracy,
        },
        "nested_candidate": {
            **selected_metrics,
            "joint_exact_accuracy": selected_joint_accuracy,
        },
        "nested_candidates": {
            candidate: {
                **candidate_metrics[candidate],
                "joint_exact_accuracy": candidate_joint_accuracy[candidate],
            }
            for candidate in CANDIDATE_SPECS
        },
        "paired_source_inference": inference,
        "selection_summary": {
            "configuration_counts": {
                head: {
                    view: dict(
                        sorted(
                            Counter(
                                row["selected_configs"][view]
                                for row in selections
                                if row["head"] == head
                            ).items()
                        )
                    )
                    for view in VIEW_NAMES
                }
                for head in HEADS
            },
            "fusion_weight_counts": {
                candidate: {
                    head: dict(
                        sorted(
                            Counter(
                                "/".join(
                                    f"{value:.3f}"
                                    for value in row["candidates"][candidate]["weights"]
                                )
                                for row in selections
                                if row["head"] == head
                            ).items()
                        )
                    )
                    for head in HEADS
                }
                for candidate in CANDIDATE_SPECS
            },
            "pair_weight_counts": {
                candidate: {
                    head: dict(
                        sorted(
                            Counter(
                                str(row["candidates"][candidate]["pair_weight"])
                                for row in selections
                                if row["head"] == head
                            ).items()
                        )
                    )
                    for head in HEADS
                }
                for candidate in CANDIDATE_SPECS
            },
        },
        "inputs": {
            "annotations": str(args.annotations),
            "raw_context": str(args.raw_context),
            "raw_context_sha256": sha256(args.raw_context),
            "stabilized_context": str(args.stabilized_context),
            "stabilized_context_sha256": sha256(args.stabilized_context),
            "raw_l2": str(args.raw_l2),
            "stabilized_l2": str(args.stabilized_l2),
            "base_model": str(args.base_model),
            "base_model_sha256": sha256(args.base_model),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": args.device,
        },
        "runtime_sec": float(time.perf_counter() - started),
    }
    write_json(args.out_dir / "summary.json", report)

    result_models = [
        ("Existing engineered dual", report["baseline"]),
        *[
            (f"Nested {candidate.replace('_', ' ')}", report["nested_candidates"][candidate])
            for candidate in CANDIDATE_SPECS
        ],
    ]
    lines = [
        "# Downstream State-Model Improvement Study",
        "",
        "Retrospective nested leave-one-source-video-out evaluation on 50 exchanges,",
        "326 ReLIS-proposed segments, and 652 per-fencer samples.",
        "",
        "| Model | FW acc. | FW F1 | FW eq-src acc. | FW eq-src F1 | State acc. | State F1 | State eq-src acc. | State eq-src F1 | Joint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {name} | {model['primary_footwork']['accuracy']:.4f} | "
            f"{model['primary_footwork']['macro_f1']:.4f} | "
            f"{model['primary_footwork']['equal_source_accuracy']:.4f} | "
            f"{model['primary_footwork']['equal_source_macro_f1_present']:.4f} | "
            f"{model['primary_tactical']['accuracy']:.4f} | "
            f"{model['primary_tactical']['macro_f1']:.4f} | "
            f"{model['primary_tactical']['equal_source_accuracy']:.4f} | "
            f"{model['primary_tactical']['equal_source_macro_f1_present']:.4f} | "
            f"{model['joint_exact_accuracy']:.4f} |"
            for name, model in result_models
        ],
        "",
        "## Broad-label diagnostic",
        "",
        "These are collapsed fine-head predictions, not separately optimized broad heads.",
        "",
        "| Model | FW acc. | FW F1 | FW eq-src acc. | FW eq-src F1 | State acc. | State F1 | State eq-src acc. | State eq-src F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {name} | {model['primary_footwork']['broad_accuracy']:.4f} | "
            f"{model['primary_footwork']['broad_macro_f1']:.4f} | "
            f"{model['primary_footwork']['equal_source_broad_accuracy']:.4f} | "
            f"{model['primary_footwork']['equal_source_broad_macro_f1_present']:.4f} | "
            f"{model['primary_tactical']['broad_accuracy']:.4f} | "
            f"{model['primary_tactical']['broad_macro_f1']:.4f} | "
            f"{model['primary_tactical']['equal_source_broad_accuracy']:.4f} | "
            f"{model['primary_tactical']['equal_source_broad_macro_f1_present']:.4f} |"
            for name, model in result_models
        ],
        "",
        "All configurations, fusion weights, and paired-label priors are selected inside",
        "the nine outer-training sources. The study remains retrospective because the",
        "ten-source semantic dataset was used in earlier state-model development.",
        "",
    ]
    (args.out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
