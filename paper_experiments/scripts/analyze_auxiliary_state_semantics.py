#!/usr/bin/env python3
"""Post-hoc semantic probe of frozen ReLIS auxiliary coordinates.

This script implements paper_experiments/round2/PROTOCOL.md.  It never changes
the boundary model or its postprocessor and it evaluates semantic probes with
leave-one-source-video-out predictions.
"""

from __future__ import annotations

import os
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO / "boundary_segmentation"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from models.l3_model import L3BoundaryModel  # noqa: E402


DEFAULT_STATE_ROOT = Path(os.environ.get("RELIS_STATE_ROOT", "data/state"))
DEFAULT_ANNOTATIONS = DEFAULT_STATE_ROOT / (
    "experiments/aaai_tactical_principles/state_audit_v2_linked50/annotations"
)
DEFAULT_CANDIDATES = DEFAULT_STATE_ROOT / (
    "experiments/aaai_tactical_principles/state_audit_v2_linked50/candidates.json"
)
DEFAULT_L2 = DEFAULT_STATE_ROOT / (
    "experiments/aaai_tactical_principles/full_exchange_regions/"
    "pipeline_legacy_v32/working/legacy_auto/l2_out"
)
DEFAULT_OUT = REPO / "paper_experiments/round2/semantic_probe"
RNG_SEED = 20260810

SEED_SPECS = [
    (
        20270614,
        REPO / "paper_experiments/experiments/truncated_retraining/runs/"
        "short_sparse_smooth02_seed20270614/config.yaml",
        REPO / "paper_experiments/experiments/truncated_retraining/runs/"
        "short_sparse_smooth02_seed20270614/ckpt_step_1000.pt",
    ),
    (
        20270615,
        REPO / "paper_experiments/experiments/truncated_retraining/runs/"
        "short_sparse_smooth02_seed20270615/config.yaml",
        REPO / "paper_experiments/experiments/truncated_retraining/runs/"
        "short_sparse_smooth02_seed20270615/ckpt_step_1000.pt",
    ),
    (
        20270616,
        REPO / "paper_experiments/experiments/truncated_retraining/runs/"
        "short_sparse_smooth02_seed20270616/config.yaml",
        REPO / "paper_experiments/experiments/truncated_retraining/runs/"
        "short_sparse_smooth02_seed20270616/ckpt_step_1000.pt",
    ),
    (
        20270617,
        REPO / "paper_experiments/experiments/truncated_retraining/runs/"
        "short_sparse_smooth02_seed20270617/config.yaml",
        REPO / "paper_experiments/experiments/truncated_retraining/runs/"
        "short_sparse_smooth02_seed20270617/ckpt_step_1000.pt",
    ),
    (
        20270618,
        REPO / "paper_experiments/experiments/truncated_retraining/candidates/"
        "optimistic_smooth02_seed18/config.yaml",
        REPO / "paper_experiments/experiments/truncated_retraining/runs/"
        "short_sparse_seed18_smooth02/ckpt_step_1000.pt",
    ),
]


@dataclass
class SegmentRecord:
    segment_id: str
    clip_id: str
    source_group: str
    start_frame: int
    end_frame: int
    feature_names: list[str]
    x: np.ndarray
    z: np.ndarray
    u: np.ndarray
    b: np.ndarray
    left_footwork: str
    right_footwork: str
    left_tactic: str
    right_tactic: str
    phase_change: str


@dataclass
class TargetData:
    name: str
    labels: np.ndarray
    groups: np.ndarray
    record_indices: np.ndarray
    sides: np.ndarray | None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_tactic(value: str) -> str:
    value = str(value or "unknown").strip().lower()
    if value == "threaten":
        return "pressure"
    if value in {"pressure", "defend", "attack"}:
        return value
    return "probe_other"


def normalize_footwork(value: str) -> str:
    value = str(value or "hold").strip().lower()
    if value in {"advance", "retreat"}:
        return value
    return "hold_reset"


def dyadic_footwork(left: str, right: str) -> str:
    values = sorted((left, right))
    if values == ["advance", "advance"]:
        return "mutual_advance"
    if values == ["advance", "retreat"]:
        return "advance_retreat"
    return "other"


def dyadic_tactic(left: str, right: str) -> str:
    values = sorted((left, right))
    if "attack" in values:
        return "any_attack"
    if values == ["pressure", "pressure"]:
        return "mutual_pressure"
    if values == ["defend", "pressure"]:
        return "pressure_defend"
    return "other"


def usable_segment(segment: dict[str, Any]) -> bool:
    if not segment.get("reviewed", False):
        return False
    if segment.get("non_fencing", False) or str(segment.get("exclusion_reason", "")).strip():
        return False
    for side in ("left", "right"):
        primary = (segment.get(side) or {}).get("primary_phase") or {}
        if not primary.get("footwork_action") or not primary.get("tactical_action"):
            return False
    return True


def candidate_groups(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    payload = read_json(path)
    groups = {
        str(item["clip_id"]): str(item["source_video_group"])
        for item in payload.get("files", [])
    }
    return groups, payload


def load_model(config_path: Path, checkpoint_path: Path, input_dim: int, device: str):
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model = L3BoundaryModel(input_dim=input_dim, cfg=cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def infer_clip(model: L3BoundaryModel, l2_payload: dict[str, Any], device: str):
    x = np.asarray(l2_payload["l2"], dtype=np.float32)
    mask = np.asarray(l2_payload["l2_mask"], dtype=np.bool_)
    with torch.no_grad():
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
        _, z, u, _, b = model(x_t, mask_t)
    return (
        x,
        mask,
        z.squeeze(0).cpu().numpy(),
        u.squeeze(0).cpu().numpy(),
        b.squeeze(0).cpu().numpy(),
    )


def collect_records(
    annotations: Path,
    candidates: Path,
    l2_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: str,
) -> tuple[list[SegmentRecord], dict[str, Any]]:
    groups, candidate_payload = candidate_groups(candidates)
    annotation_paths = sorted(annotations.glob("*.tactical.json"))
    if len(annotation_paths) != 50:
        raise ValueError(f"Expected 50 annotations, found {len(annotation_paths)}")

    first_clip = annotation_paths[0].name[: -len(".tactical.json")]
    first_l2 = read_json(l2_root / f"{first_clip}_l2.json")
    input_dim = len(first_l2["feature_names"])
    model = load_model(config_path, checkpoint_path, input_dim, device)

    records: list[SegmentRecord] = []
    excluded = Counter()
    for annotation_path in annotation_paths:
        annotation = read_json(annotation_path)
        clip_id = str(annotation["clip_id"])
        if clip_id not in groups:
            raise KeyError(f"No source group for {clip_id}")
        l2_path = l2_root / f"{clip_id}_l2.json"
        l2 = read_json(l2_path)
        frames = np.asarray(l2.get("frames", np.arange(len(l2["l2"]))), dtype=np.int64)
        x, mask, z, u, b = infer_clip(model, l2, device)
        feature_names = [str(value) for value in l2["feature_names"]]
        for segment in annotation.get("segments", []):
            if not usable_segment(segment):
                reason = str(segment.get("exclusion_reason") or "non_fencing_or_incomplete")
                excluded[reason] += 1
                continue
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            take = (frames >= start) & (frames <= end) & mask
            if not np.any(take):
                raise ValueError(f"No valid frames for {clip_id} [{start}, {end}]")
            left = segment["left"]
            right = segment["right"]
            left_primary = left["primary_phase"]
            right_primary = right["primary_phase"]
            phase_change = "yes" if left.get("secondary_phase") or right.get("secondary_phase") else "no"
            records.append(
                SegmentRecord(
                    segment_id=f"{clip_id}:{int(segment['segment_index'])}",
                    clip_id=clip_id,
                    source_group=groups[clip_id],
                    start_frame=start,
                    end_frame=end,
                    feature_names=feature_names,
                    x=x[take],
                    z=z[take],
                    u=u[take],
                    b=b[take, None],
                    left_footwork=normalize_footwork(left_primary["footwork_action"]),
                    right_footwork=normalize_footwork(right_primary["footwork_action"]),
                    left_tactic=normalize_tactic(left_primary["tactical_action"]),
                    right_tactic=normalize_tactic(right_primary["tactical_action"]),
                    phase_change=phase_change,
                )
            )
    meta = {
        "annotation_task": candidate_payload.get("annotation_task"),
        "blind_to_state_model_predictions": candidate_payload.get(
            "blind_to_state_model_predictions"
        ),
        "annotation_files": len(annotation_paths),
        "source_groups": len({record.source_group for record in records}),
        "usable_segments": len(records),
        "per_fencer_states": 2 * len(records),
        "excluded": dict(excluded),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    return records, meta


def summarize_trajectory(values: np.ndarray, lag: int = 8) -> np.ndarray:
    """Ten fixed summary statistics per coordinate."""
    if values.ndim == 1:
        values = values[:, None]
    n, dim = values.shape
    if n == 0:
        return np.zeros(10 * dim, dtype=np.float64)
    one_diff = np.abs(np.diff(values, axis=0)) if n > 1 else np.zeros((1, dim))
    lag_diff = (
        np.abs(values[lag:] - values[:-lag]) if n > lag else np.zeros((1, dim))
    )
    signs = np.sign(values)
    sign_changes = (
        np.mean(signs[1:] != signs[:-1], axis=0) if n > 1 else np.zeros(dim)
    )
    stats = np.stack(
        [
            np.mean(values, axis=0),
            np.std(values, axis=0),
            values[0],
            values[-1],
            values[-1] - values[0],
            np.max(values, axis=0) - np.min(values, axis=0),
            np.mean(one_diff, axis=0),
            np.max(one_diff, axis=0),
            np.mean(lag_diff, axis=0),
            sign_changes,
        ],
        axis=1,
    )
    return np.nan_to_num(stats.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def summarize_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    diff = np.abs(np.diff(values)) if len(values) > 1 else np.zeros(1)
    return np.asarray(
        [
            np.mean(values),
            np.std(values),
            values[0],
            values[-1],
            values[-1] - values[0],
            np.max(values) - np.min(values),
            np.mean(diff),
            np.max(diff),
        ],
        dtype=np.float64,
    )


def build_targets(records: list[SegmentRecord]) -> list[TargetData]:
    groups = np.asarray([record.source_group for record in records])
    indices = np.arange(len(records), dtype=np.int64)
    targets = [
        TargetData(
            "dyadic_footwork",
            np.asarray(
                [dyadic_footwork(r.left_footwork, r.right_footwork) for r in records]
            ),
            groups,
            indices,
            None,
        ),
        TargetData(
            "dyadic_tactic",
            np.asarray([dyadic_tactic(r.left_tactic, r.right_tactic) for r in records]),
            groups,
            indices,
            None,
        ),
        TargetData(
            "human_phase_change",
            np.asarray([r.phase_change for r in records]),
            groups,
            indices,
            None,
        ),
    ]
    per_indices = np.repeat(indices, 2)
    per_groups = np.repeat(groups, 2)
    sides = np.tile(np.asarray(["left", "right"]), len(records))
    targets.extend(
        [
            TargetData(
                "per_fencer_footwork",
                np.asarray(
                    [value for r in records for value in (r.left_footwork, r.right_footwork)]
                ),
                per_groups,
                per_indices,
                sides,
            ),
            TargetData(
                "per_fencer_tactic",
                np.asarray(
                    [value for r in records for value in (r.left_tactic, r.right_tactic)]
                ),
                per_groups,
                per_indices,
                sides,
            ),
        ]
    )
    return targets


def side_gate(features: np.ndarray, sides: np.ndarray | None) -> np.ndarray:
    if sides is None:
        return features
    left = (sides == "left").astype(np.float64)[:, None]
    right = (sides == "right").astype(np.float64)[:, None]
    return np.concatenate([features * left, features * right], axis=1)


def record_features(
    records: list[SegmentRecord],
    target: TargetData,
    representation: str,
    train_record_indices: np.ndarray | None = None,
) -> np.ndarray:
    if representation == "auxiliary":
        base = np.stack([summarize_trajectory(r.u) for r in records])
    elif representation == "activity_score":
        base = np.stack([summarize_score(r.b) for r in records])
    elif representation == "latent_pca3":
        if train_record_indices is None:
            raise ValueError("latent_pca3 requires training record indices")
        frame_bank = np.concatenate([records[int(i)].z for i in train_record_indices], axis=0)
        pca = PCA(n_components=3, random_state=RNG_SEED)
        pca.fit(frame_bank)
        base = np.stack([summarize_trajectory(pca.transform(r.z)) for r in records])
    else:
        raise ValueError(representation)
    selected = base[target.record_indices]
    return side_gate(selected, target.sides)


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=RNG_SEED,
        ),
    )
    model.fit(train_x, train_y)
    return model.predict(test_x)


def loso_predictions(
    records: list[SegmentRecord], target: TargetData, representation: str
) -> np.ndarray:
    predictions = np.empty(len(target.labels), dtype=object)
    for group in sorted(set(target.groups.tolist())):
        test = target.groups == group
        train = ~test
        if representation == "latent_pca3":
            train_record_indices = np.unique(target.record_indices[train])
            features = record_features(
                records, target, representation, train_record_indices=train_record_indices
            )
        else:
            features = record_features(records, target, representation)
        predictions[test] = fit_predict(features[train], target.labels[train], features[test])
    return predictions.astype(str)


def majority_predictions(target: TargetData) -> np.ndarray:
    predictions = np.empty(len(target.labels), dtype=object)
    for group in sorted(set(target.groups.tolist())):
        test = target.groups == group
        train = ~test
        majority = Counter(target.labels[train].tolist()).most_common(1)[0][0]
        predictions[test] = majority
    return predictions.astype(str)


def metrics(labels: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=class_names, average="macro", zero_division=0)
        ),
    }


def cluster_bootstrap(
    labels: np.ndarray,
    predictions: np.ndarray,
    majority: np.ndarray,
    groups: np.ndarray,
    class_names: list[str],
    repeats: int,
) -> dict[str, float]:
    rng = np.random.default_rng(RNG_SEED)
    unique = np.asarray(sorted(set(groups.tolist())))
    scores = []
    deltas = []
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([by_group[group] for group in sampled])
        score = f1_score(
            labels[take], predictions[take], labels=class_names, average="macro", zero_division=0
        )
        floor = f1_score(
            labels[take], majority[take], labels=class_names, average="macro", zero_division=0
        )
        scores.append(score)
        deltas.append(score - floor)
    return {
        "macro_f1_ci_low": float(np.quantile(scores, 0.025)),
        "macro_f1_ci_high": float(np.quantile(scores, 0.975)),
        "delta_vs_majority_ci_low": float(np.quantile(deltas, 0.025)),
        "delta_vs_majority_ci_high": float(np.quantile(deltas, 0.975)),
    }


def permutation_p_value(
    records: list[SegmentRecord],
    target: TargetData,
    observed_macro_f1: float,
    repeats: int,
) -> tuple[float, list[float]]:
    rng = np.random.default_rng(RNG_SEED)
    features = record_features(records, target, "auxiliary")
    scores: list[float] = []
    class_names = sorted(set(target.labels.tolist()))
    for _ in range(repeats):
        shuffled = target.labels.copy()
        for group in sorted(set(target.groups.tolist())):
            take = np.flatnonzero(target.groups == group)
            shuffled[take] = rng.permutation(shuffled[take])
        predictions = np.empty(len(shuffled), dtype=object)
        for group in sorted(set(target.groups.tolist())):
            test = target.groups == group
            train = ~test
            predictions[test] = fit_predict(features[train], shuffled[train], features[test])
        scores.append(
            float(
                f1_score(
                    shuffled,
                    predictions.astype(str),
                    labels=class_names,
                    average="macro",
                    zero_division=0,
                )
            )
        )
    p_value = (1.0 + sum(score >= observed_macro_f1 for score in scores)) / (repeats + 1.0)
    return float(p_value), scores


def probe_main(
    records: list[SegmentRecord], bootstrap_repeats: int, permutation_repeats: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    permutation_scores: dict[str, list[float]] = {}
    for target in build_targets(records):
        class_names = sorted(set(target.labels.tolist()))
        majority = majority_predictions(target)
        representation_predictions: dict[str, np.ndarray] = {"majority": majority}
        for representation in ("activity_score", "latent_pca3", "auxiliary"):
            representation_predictions[representation] = loso_predictions(
                records, target, representation
            )
        for representation, predictions in representation_predictions.items():
            values = metrics(target.labels, predictions, class_names)
            intervals = cluster_bootstrap(
                target.labels,
                predictions,
                majority,
                target.groups,
                class_names,
                bootstrap_repeats,
            )
            row: dict[str, Any] = {
                "target": target.name,
                "representation": representation,
                "n": len(target.labels),
                "classes": "|".join(class_names),
                "class_counts": json.dumps(dict(Counter(target.labels.tolist())), sort_keys=True),
                **values,
                **intervals,
                "permutation_p": "",
            }
            if representation == "auxiliary":
                p_value, null_scores = permutation_p_value(
                    records, target, values["macro_f1"], permutation_repeats
                )
                row["permutation_p"] = p_value
                permutation_scores[target.name] = null_scores
            metric_rows.append(row)
        for index, (label, group, record_index) in enumerate(
            zip(target.labels, target.groups, target.record_indices)
        ):
            record = records[int(record_index)]
            prediction_rows.append(
                {
                    "target": target.name,
                    "sample_index": index,
                    "segment_id": record.segment_id,
                    "source_group": group,
                    "side": "" if target.sides is None else target.sides[index],
                    "label": label,
                    **{
                        f"prediction_{name}": values[index]
                        for name, values in representation_predictions.items()
                    },
                }
            )
    return metric_rows, prediction_rows, permutation_scores


def seed_probe(records: list[SegmentRecord], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in build_targets(records):
        predictions = loso_predictions(records, target, "auxiliary")
        class_names = sorted(set(target.labels.tolist()))
        values = metrics(target.labels, predictions, class_names)
        rows.append({"seed": seed, "target": target.name, "n": len(target.labels), **values})
    return rows


def correlation_rows(records: list[SegmentRecord], seed: int) -> list[dict[str, Any]]:
    names = records[0].feature_names
    x_means = np.stack([np.mean(record.x, axis=0) for record in records])
    u_means = np.stack([np.mean(record.u, axis=0) for record in records])
    rows: list[dict[str, Any]] = []
    for coordinate in range(u_means.shape[1]):
        for feature_index, feature_name in enumerate(names):
            rho, p_value = spearmanr(u_means[:, coordinate], x_means[:, feature_index])
            rows.append(
                {
                    "seed": seed,
                    "coordinate": f"u{coordinate}",
                    "feature": feature_name,
                    "spearman_rho": float(rho),
                    "abs_spearman_rho": float(abs(rho)),
                    "nominal_p": float(p_value),
                }
            )
    return rows


def class_mean_rows(records: list[SegmentRecord], seed: int) -> list[dict[str, Any]]:
    u_means = np.stack([np.mean(record.u, axis=0) for record in records])
    scale = np.std(u_means, axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (u_means - np.mean(u_means, axis=0)) / scale
    label_sets = {
        "dyadic_footwork": np.asarray(
            [dyadic_footwork(r.left_footwork, r.right_footwork) for r in records]
        ),
        "dyadic_tactic": np.asarray(
            [dyadic_tactic(r.left_tactic, r.right_tactic) for r in records]
        ),
        "human_phase_change": np.asarray([r.phase_change for r in records]),
    }
    rows: list[dict[str, Any]] = []
    for target, labels in label_sets.items():
        for label in sorted(set(labels.tolist())):
            take = labels == label
            for coordinate in range(standardized.shape[1]):
                values = standardized[take, coordinate]
                rows.append(
                    {
                        "seed": seed,
                        "target": target,
                        "label": label,
                        "coordinate": f"u{coordinate}",
                        "n": int(np.sum(take)),
                        "standardized_mean": float(np.mean(values)),
                        "standardized_sd": float(np.std(values)),
                    }
                )
    return rows


def make_figure(metric_rows: list[dict[str, Any]], path: Path) -> None:
    targets = [
        "dyadic_footwork",
        "dyadic_tactic",
        "human_phase_change",
        "per_fencer_footwork",
        "per_fencer_tactic",
    ]
    reps = ["majority", "activity_score", "latent_pca3", "auxiliary"]
    labels = ["Majority", "Activity score", "Latent PCA-3", "Auxiliary $u$"]
    colors = ["#9a9a9a", "#d28e32", "#5278a8", "#2f8f68"]
    lookup = {(row["target"], row["representation"]): row for row in metric_rows}
    x = np.arange(len(targets))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    for rep_index, (rep, label, color) in enumerate(zip(reps, labels, colors)):
        values = [lookup[(target, rep)]["macro_f1"] for target in targets]
        lows = [
            values[i] - lookup[(target, rep)]["macro_f1_ci_low"]
            for i, target in enumerate(targets)
        ]
        highs = [
            lookup[(target, rep)]["macro_f1_ci_high"] - values[i]
            for i, target in enumerate(targets)
        ]
        ax.bar(
            x + (rep_index - 1.5) * width,
            values,
            width,
            label=label,
            color=color,
            yerr=np.asarray([lows, highs]),
            capsize=2,
            linewidth=0.4,
            edgecolor="white",
        )
    ax.set_ylabel("Source-held-out macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Dyadic\nfootwork", "Dyadic\ntactic", "Human phase\nchange", "Per-fencer\nfootwork", "Per-fencer\ntactic"]
    )
    ax.set_ylim(0.0, 0.75)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    if value == "" or value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def render_results(
    path: Path,
    metadata: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
) -> None:
    lookup = {(row["target"], row["representation"]): row for row in metric_rows}
    targets = [
        "dyadic_footwork",
        "dyadic_tactic",
        "human_phase_change",
        "per_fencer_footwork",
        "per_fencer_tactic",
    ]
    display = {
        "dyadic_footwork": "Dyadic footwork",
        "dyadic_tactic": "Dyadic tactic",
        "human_phase_change": "Human phase change",
        "per_fencer_footwork": "Per-fencer footwork",
        "per_fencer_tactic": "Per-fencer tactic",
    }
    lines = [
        "# Frozen ReLIS auxiliary-coordinate semantic probe",
        "",
        "This is a post-hoc diagnostic on later epee state annotations; it does not",
        "retrain ReLIS and it is not part of the foil primary evaluation.",
        "",
        "## Data and provenance",
        "",
        f"- Human-reviewed exchanges: {metadata['annotation_files']}",
        f"- Source videos: {metadata['source_groups']}",
        f"- Usable segments: {metadata['usable_segments']}",
        f"- Per-fencer states: {metadata['per_fencer_states']}",
        f"- Annotation blind to state-model predictions: {metadata['blind_to_state_model_predictions']}",
        f"- Frozen checkpoint SHA-256: `{metadata['checkpoint_sha256']}`",
        "- Important conditioning: people labeled the semantic content of ReLIS-proposed segments.",
        "",
        "## Source-video-held-out results",
        "",
        "| target | majority F1 | score F1 | latent PCA-3 F1 | auxiliary F1 [95% CI] | delta-vs-majority CI | perm. p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in targets:
        majority = lookup[(target, "majority")]
        score = lookup[(target, "activity_score")]
        latent = lookup[(target, "latent_pca3")]
        aux = lookup[(target, "auxiliary")]
        lines.append(
            f"| {display[target]} | {fmt(majority['macro_f1'])} | {fmt(score['macro_f1'])} | "
            f"{fmt(latent['macro_f1'])} | {fmt(aux['macro_f1'])} "
            f"[{fmt(aux['macro_f1_ci_low'])}, {fmt(aux['macro_f1_ci_high'])}] | "
            f"[{fmt(aux['delta_vs_majority_ci_low'])}, {fmt(aux['delta_vs_majority_ci_high'])}] | "
            f"{fmt(aux['permutation_p'])} |"
        )

    lines.extend(
        [
            "",
            "`Latent PCA-3` uses three components fitted only on each training fold.  The",
            "activity-score control summarizes the scalar boundary score.  Intervals resample",
            "the ten source videos; permutations shuffle labels within source video.",
            "",
            "## Five-seed stability",
            "",
            "| target | auxiliary macro F1, mean +/- SD | range |",
            "| --- | ---: | ---: |",
        ]
    )
    for target in targets:
        values = np.asarray(
            [row["macro_f1"] for row in seed_rows if row["target"] == target], dtype=float
        )
        lines.append(
            f"| {display[target]} | {np.mean(values):.3f} +/- {np.std(values, ddof=1):.3f} | "
            f"{np.min(values):.3f}--{np.max(values):.3f} |"
        )

    lines.extend(["", "## Strongest coordinate/input associations", ""])
    for seed in sorted({int(row["seed"]) for row in correlations}):
        lines.append(f"Seed {seed}:")
        lines.append("")
        subset = [row for row in correlations if int(row["seed"]) == seed]
        for coordinate in ("u0", "u1", "u2"):
            ranked = sorted(
                [row for row in subset if row["coordinate"] == coordinate],
                key=lambda row: row["abs_spearman_rho"],
                reverse=True,
            )[:3]
            text = ", ".join(
                f"{row['feature']} ({float(row['spearman_rho']):+.2f})" for row in ranked
            )
            lines.append(f"- {coordinate}: {text}")
        lines.append("")

    aux_rows = [row for row in metric_rows if row["representation"] == "auxiliary"]
    supported = [
        row
        for row in aux_rows
        if float(row["permutation_p"]) <= 0.05
        and float(row["delta_vs_majority_ci_low"]) > 0.0
    ]
    lines.extend(
        [
            "## Claim decision",
            "",
            (
                "The frozen criterion is met for: "
                + ", ".join(display[row["target"]] for row in supported)
                + "."
                if supported
                else "The frozen criterion is not met for any pre-defined target."
            ),
            "",
            "Even when the criterion is met, the supported wording is *low-dimensional and",
            "semantically informative in a post-hoc source-held-out probe*.  The coordinates",
            "are not uniquely identifiable tactic variables, and they received no semantic",
            "supervision during ReLIS training.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--l2-root", type=Path, default=DEFAULT_L2)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--permutation-repeats", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--main-seed-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    all_seed_rows: list[dict[str, Any]] = []
    all_correlations: list[dict[str, Any]] = []
    all_class_means: list[dict[str, Any]] = []
    main_records: list[SegmentRecord] | None = None
    main_metadata: dict[str, Any] | None = None

    specs = [SEED_SPECS[-1]] if args.main_seed_only else SEED_SPECS
    for seed, config, checkpoint in specs:
        print(f"Inferring seed {seed} from {checkpoint}", flush=True)
        records, metadata = collect_records(
            args.annotations, args.candidates, args.l2_root, config, checkpoint, device
        )
        all_seed_rows.extend(seed_probe(records, seed))
        all_correlations.extend(correlation_rows(records, seed))
        all_class_means.extend(class_mean_rows(records, seed))
        if seed == 20270618:
            main_records = records
            main_metadata = metadata

    if main_records is None or main_metadata is None:
        raise RuntimeError("Main seed 20270618 was not evaluated")

    print("Running source-held-out controls, bootstrap, and permutations", flush=True)
    metric_rows, prediction_rows, permutation_scores = probe_main(
        main_records, args.bootstrap_repeats, args.permutation_repeats
    )

    write_csv(args.out_dir / "metrics.csv", metric_rows)
    write_csv(args.out_dir / "predictions.csv", prediction_rows)
    write_csv(args.out_dir / "seed_metrics.csv", all_seed_rows)
    write_csv(args.out_dir / "coordinate_feature_correlations.csv", all_correlations)
    write_csv(args.out_dir / "coordinate_class_means.csv", all_class_means)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": str(REPO / "paper_experiments/round2/PROTOCOL.md"),
                "random_seed": RNG_SEED,
                "bootstrap_repeats": args.bootstrap_repeats,
                "permutation_repeats": args.permutation_repeats,
                "metadata": main_metadata,
                "metrics": metric_rows,
                "seed_metrics": all_seed_rows,
                "permutation_null_macro_f1": permutation_scores,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    make_figure(metric_rows, args.out_dir / "semantic_probe.png")
    render_results(
        args.out_dir / "RESULTS.md",
        main_metadata,
        metric_rows,
        all_seed_rows,
        all_correlations,
    )
    print(f"Wrote {args.out_dir / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
