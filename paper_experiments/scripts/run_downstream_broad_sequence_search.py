#!/usr/bin/env python3
"""Nested broad-state classifiers and dyadic sequence decoding."""

from __future__ import annotations

import os
import argparse
import csv
import hashlib
import json
import pickle
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
STATE_ROOT = Path(os.environ.get("RELIS_STATE_ROOT", "data/state"))
for path in (SCRIPT_ROOT, STATE_ROOT, STATE_ROOT / "experiments/scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_downstream_state_model_search as core  # noqa: E402
from experiments.scripts import experiment_camera_integrated_full_state as full_state  # noqa: E402
from experiments.scripts import experiment_primary_trace_state as primary_trace  # noqa: E402
from experiments.scripts import experiment_secondary_trace_gate as trace_gate  # noqa: E402
from state_model.features import paired_feature_indices  # noqa: E402


OUT = ROOT / "paper_experiments/round2/downstream_state_model_search/broad_sequence"
CONFIGS = {
    "balanced_leaf5": core.CONFIGS["balanced_leaf5"],
    "balanced_leaf2": {
        **core.CONFIGS["balanced_leaf5"],
        "min_samples_leaf": 2,
    },
    "plain_leaf2": core.CONFIGS["plain_leaf2"],
}
VIEWS = ("engineered_dual", "trace_stabilized", "trace_dual")
WEIGHTS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.75, 0.25, 0.0),
    (0.5, 0.5, 0.0),
    (0.25, 0.75, 0.0),
    (0.75, 0.0, 0.25),
    (0.5, 0.0, 0.5),
    (0.25, 0.0, 0.75),
)
PAIR_WEIGHTS = (0.0, 0.25, 0.5)
TRANSITION_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
TOLERANCE = 0.005
HEADS = ("footwork", "state")
NAMES = {
    "footwork": ("advance", "retreat", "hold/reset"),
    "state": ("pressure", "attack", "defend", "other"),
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest_digest(paths: list[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(core.sha256(path).encode("ascii"))
        digest.update(b"\n")
    return {"files": len(paths), "sha256": digest.hexdigest()}


def target(samples: list[dict[str, Any]], head: str) -> np.ndarray:
    if head == "footwork":
        mapping = np.asarray([0, 1, 2, 2], dtype=np.int64)
        fine = np.asarray([sample["primary_footwork"] for sample in samples], dtype=np.int64)
    else:
        mapping = np.asarray([3, 0, 1, 2, 3], dtype=np.int64)
        fine = np.asarray([sample["primary_tactical"] for sample in samples], dtype=np.int64)
    return mapping[fine]


def load_data(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    with args.base_model.open("rb") as handle:
        artifact = pickle.load(handle)
    feature_names = list(artifact["feature_names"])
    raw_context = full_state.read_json(args.raw_context)
    stable_context = full_state.read_json(args.stabilized_context)
    stable_ids = {
        core.clip_id(row.get("file"))
        for row in stable_context.get("files", [])
        if isinstance(row, dict)
    }
    raw_context = {
        **raw_context,
        "files": [
            row
            for row in raw_context.get("files", [])
            if isinstance(row, dict) and core.clip_id(row.get("file")) in stable_ids
        ],
    }
    samples = full_state.build_samples(
        raw_context,
        stable_context,
        full_state.load_annotations(args.annotations),
        feature_names,
    )
    keys = [
        (str(row["clip_id"]), int(row["segment_idx"]), str(row["side"]))
        for row in samples
    ]
    if len(samples) != 652 or len(set(keys)) != 652:
        raise ValueError("Expected 652 unique side-segment samples")
    sides_by_segment: dict[tuple[str, int], set[str]] = defaultdict(set)
    for cid, segment_index, side in keys:
        sides_by_segment[(cid, segment_index)].add(side)
    if len(sides_by_segment) != 326 or any(
        sides != {"left", "right"} for sides in sides_by_segment.values()
    ):
        raise ValueError("Expected 326 paired left/right segments")
    pairs = paired_feature_indices(feature_names)
    trace_raw, trace_stable, trace_names = trace_gate.build_trace_matrices(
        samples,
        trace_gate.index_l2(args.raw_l2),
        trace_gate.index_l2(args.stabilized_l2),
    )
    views: dict[str, dict[str, np.ndarray]] = {head: {} for head in HEADS}
    for head in HEADS:
        method_name = "primary_footwork" if head == "footwork" else "primary_tactical"
        method = full_state.HEAD_METHODS[method_name]
        raw_x = full_state.feature_matrix(samples, "raw", method, len(feature_names), pairs)
        stable_x = full_state.feature_matrix(
            samples, "stabilized", method, len(feature_names), pairs
        )
        views[head]["engineered_dual"] = np.concatenate([raw_x, stable_x], axis=1)
        raw_trace = trace_raw
        stable_trace = trace_stable
        if method["pool_sides"]:
            raw_trace = primary_trace.canonicalize_trace_for_right(
                raw_trace, samples, trace_names
            )
            stable_trace = primary_trace.canonicalize_trace_for_right(
                stable_trace, samples, trace_names
            )
        views[head]["trace_stabilized"] = stable_trace
        views[head]["trace_dual"] = np.concatenate([raw_trace, stable_trace], axis=1)
    return samples, views, {
        "feature_schema_count": len(feature_names),
        "dimensions": {
            head: {name: int(values.shape[1]) for name, values in by_view.items()}
            for head, by_view in views.items()
        },
    }


def fit_predict(
    samples: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    head: str,
    config: dict[str, Any],
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> np.ndarray:
    classes = len(NAMES[head])
    sides = np.asarray([sample["side"] for sample in samples])
    pooled = head == "state"
    output = np.zeros((int(np.sum(test)), classes), dtype=np.float64)
    test_positions = np.flatnonzero(test)
    if pooled:
        model = core.classifier(config, seed)
        model.fit(x[train], y[train])
        return core.aligned_probabilities(model, x[test], classes)
    lookup = {global_index: local_index for local_index, global_index in enumerate(test_positions)}
    for side_index, side in enumerate(("left", "right")):
        side_train = train & (sides == side)
        side_test = test & (sides == side)
        model = core.classifier(config, seed + side_index)
        model.fit(x[side_train], y[side_train])
        probabilities = core.aligned_probabilities(model, x[side_test], classes)
        positions = np.flatnonzero(side_test)
        output[[lookup[index] for index in positions]] = probabilities
    return output


def objective(
    samples: list[dict[str, Any]],
    y: np.ndarray,
    prediction: np.ndarray,
    selected: np.ndarray,
) -> tuple[float, float, float]:
    groups = np.asarray([str(sample["source_group"]) for sample in samples])
    rows = []
    for group in sorted(set(groups[selected])):
        take = selected & (groups == group)
        present = sorted(set(int(value) for value in y[take]))
        accuracy = float(accuracy_score(y[take], prediction[take]))
        macro = float(
            f1_score(
                y[take], prediction[take], labels=present, average="macro", zero_division=0
            )
        )
        rows.append((accuracy + macro) / 2.0)
    return (
        float(np.mean(rows)),
        float(f1_score(y[selected], prediction[selected], average="macro", zero_division=0)),
        float(accuracy_score(y[selected], prediction[selected])),
    )


def combine(probabilities: dict[str, np.ndarray], weights: tuple[float, float, float]) -> np.ndarray:
    output = np.zeros_like(probabilities[VIEWS[0]])
    for name, weight in zip(VIEWS, weights):
        output += float(weight) * probabilities[name]
    sums = np.sum(output, axis=1, keepdims=True)
    return np.divide(output, sums, out=np.zeros_like(output), where=sums > 0)


def priors(
    samples: list[dict[str, Any]],
    y: np.ndarray,
    selected: np.ndarray,
    classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    pair = np.ones((classes, classes), dtype=np.float64)
    transition = np.ones((classes, classes), dtype=np.float64)
    by_segment: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    by_sequence: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if not selected[index]:
            continue
        cid = str(sample["clip_id"])
        segment = int(sample["segment_idx"])
        side = str(sample["side"])
        by_segment[(cid, segment)][side] = index
        by_sequence[(cid, side)].append(index)
    for sides in by_segment.values():
        if set(sides) == {"left", "right"}:
            pair[y[sides["left"]], y[sides["right"]]] += 1.0
    pair = (pair + pair.T) / 2.0
    for indices in by_sequence.values():
        ordered = sorted(indices, key=lambda index: int(samples[index]["segment_idx"]))
        for previous, current in zip(ordered, ordered[1:]):
            if int(samples[current]["segment_idx"]) == int(samples[previous]["segment_idx"]) + 1:
                transition[y[previous], y[current]] += 1.0
    pair /= np.sum(pair)
    pair_left = np.sum(pair, axis=1, keepdims=True)
    pair_right = np.sum(pair, axis=0, keepdims=True)
    compatibility = pair / np.maximum(pair_left * pair_right, 1e-12)
    transition /= np.sum(transition, axis=1, keepdims=True)
    return np.log(np.maximum(compatibility, 1e-12)), np.log(np.maximum(transition, 1e-12))


def decode(
    samples: list[dict[str, Any]],
    probabilities: np.ndarray,
    selected: np.ndarray,
    log_pair: np.ndarray,
    log_transition: np.ndarray,
    pair_weight: float,
    transition_weight: float,
) -> np.ndarray:
    prediction = np.argmax(probabilities, axis=1).astype(np.int64)
    classes = probabilities.shape[1]
    states = [(left, right) for left in range(classes) for right in range(classes)]
    by_clip: dict[str, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for index, sample in enumerate(samples):
        if selected[index]:
            by_clip[str(sample["clip_id"])][int(sample["segment_idx"])][
                str(sample["side"])
            ] = index
    for segments in by_clip.values():
        indexed = [
            (key, segments[key])
            for key in sorted(segments)
            if set(segments[key]) == {"left", "right"}
        ]
        runs: list[list[dict[str, int]]] = []
        previous_index: int | None = None
        for segment_index, row in indexed:
            if previous_index is None or segment_index != previous_index + 1:
                runs.append([])
            runs[-1].append(row)
            previous_index = segment_index
        for ordered in runs:
            scores = np.full((len(ordered), len(states)), -np.inf, dtype=np.float64)
            back = np.zeros((len(ordered), len(states)), dtype=np.int64)
            for state_index, (left_value, right_value) in enumerate(states):
                left = ordered[0]["left"]
                right = ordered[0]["right"]
                scores[0, state_index] = (
                    np.log(max(probabilities[left, left_value], 1e-12))
                    + np.log(max(probabilities[right, right_value], 1e-12))
                    + pair_weight * log_pair[left_value, right_value]
                )
            for time_index in range(1, len(ordered)):
                left = ordered[time_index]["left"]
                right = ordered[time_index]["right"]
                for state_index, (left_value, right_value) in enumerate(states):
                    emission = (
                        np.log(max(probabilities[left, left_value], 1e-12))
                        + np.log(max(probabilities[right, right_value], 1e-12))
                        + pair_weight * log_pair[left_value, right_value]
                    )
                    candidates = np.asarray(
                        [
                            scores[time_index - 1, previous_index]
                            + transition_weight
                            * (
                                log_transition[previous_left, left_value]
                                + log_transition[previous_right, right_value]
                            )
                            for previous_index, (previous_left, previous_right) in enumerate(states)
                        ]
                    )
                    best_previous = int(np.argmax(candidates))
                    scores[time_index, state_index] = emission + candidates[best_previous]
                    back[time_index, state_index] = best_previous
            path = [int(np.argmax(scores[-1]))]
            for time_index in range(len(ordered) - 1, 0, -1):
                path.append(int(back[time_index, path[-1]]))
            path.reverse()
            for row, state_index in zip(ordered, path):
                left_value, right_value = states[state_index]
                prediction[row["left"]] = left_value
                prediction[row["right"]] = right_value
    return prediction


def select_setting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    maximum = max(float(row["objective"][0]) for row in rows)
    eligible = [row for row in rows if float(row["objective"][0]) >= maximum - TOLERANCE]

    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        weights = tuple(float(value) for value in row["weights"])
        active = sum(value > 0 for value in weights)
        decoder = int(float(row["pair_weight"]) > 0) + int(
            float(row["transition_weight"]) > 0
        )
        return (
            active,
            decoder,
            -weights[0],
            tuple(-float(value) for value in row["objective"]),
            weights,
            float(row["pair_weight"]),
            float(row["transition_weight"]),
        )

    return min(eligible, key=rank)


def metric(
    samples: list[dict[str, Any]],
    head: str,
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    groups = np.asarray([str(sample["source_group"]) for sample in samples])
    per_source = {}
    for group in sorted(set(groups)):
        take = groups == group
        present = sorted(set(int(value) for value in y[take]))
        per_source[group] = {
            "accuracy": float(accuracy_score(y[take], prediction[take])),
            "macro_f1_present": float(
                f1_score(
                    y[take], prediction[take], labels=present, average="macro", zero_division=0
                )
            ),
        }
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "equal_source_accuracy": float(np.mean([row["accuracy"] for row in per_source.values()])),
        "equal_source_macro_f1_present": float(
            np.mean([row["macro_f1_present"] for row in per_source.values()])
        ),
        "truth": dict(sorted(Counter(NAMES[head][int(value)] for value in y).items())),
        "per_source": per_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--annotations", type=Path, default=core.DEFAULT_ANNOTATIONS)
    parser.add_argument("--raw-context", type=Path, default=core.DEFAULT_RAW_CONTEXT)
    parser.add_argument("--stabilized-context", type=Path, default=core.DEFAULT_STABILIZED_CONTEXT)
    parser.add_argument("--raw-l2", type=Path, default=core.DEFAULT_RAW_L2)
    parser.add_argument("--stabilized-l2", type=Path, default=core.DEFAULT_STABILIZED_L2)
    parser.add_argument("--base-model", type=Path, default=core.DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--fine-predictions",
        type=Path,
        default=core.DEFAULT_OUT / "predictions.csv",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    samples, views, metadata = load_data(args)
    groups = np.asarray([str(sample["source_group"]) for sample in samples])
    group_names = sorted(set(groups))
    if len(group_names) != 10 or len({str(sample["clip_id"]) for sample in samples}) != 50:
        raise ValueError("Expected 10 source groups and 50 clips")
    clips_by_group = {
        group: {
            str(sample["clip_id"])
            for sample in samples
            if str(sample["source_group"]) == group
        }
        for group in group_names
    }
    if any(len(clips) != 5 for clips in clips_by_group.values()):
        raise ValueError(f"Expected five clips per source group: {clips_by_group}")
    expected_counts = {
        "footwork": {"advance": 363, "retreat": 213, "hold/reset": 76},
        "state": {"pressure": 340, "attack": 100, "defend": 199, "other": 13},
    }
    for head in HEADS:
        observed = dict(sorted(Counter(NAMES[head][int(value)] for value in target(samples, head)).items()))
        if observed != expected_counts[head]:
            raise ValueError(f"Unexpected {head} counts: {observed}")

    with args.fine_predictions.open("r", encoding="utf-8", newline="") as handle:
        fine_rows = list(csv.DictReader(handle))
    expected_keys = [
        (str(sample["clip_id"]), str(sample["segment_idx"]), str(sample["side"]))
        for sample in samples
    ]
    actual_keys = [(row["clip_id"], row["segment_idx"], row["side"]) for row in fine_rows]
    if expected_keys != actual_keys:
        raise ValueError("Fine baseline predictions do not align")
    fine_foot_names = ("advance", "retreat", "hold", "reset")
    fine_state_names = ("probe", "pressure", "attack", "defend", "unknown")
    fine_foot_truth = [fine_foot_names[int(sample["primary_footwork"])] for sample in samples]
    fine_state_truth = [fine_state_names[int(sample["primary_tactical"])] for sample in samples]
    if (
        [row["footwork_truth"] for row in fine_rows] != fine_foot_truth
        or [row["state_truth"] for row in fine_rows] != fine_state_truth
        or [row["source_group"] for row in fine_rows] != list(groups)
    ):
        raise ValueError("Fine prediction truths/source groups differ from current samples")
    foot_map = np.asarray([0, 1, 2, 2])
    state_map = np.asarray([3, 0, 1, 2, 3])
    mapped_fine = {
        "footwork": foot_map[
            np.asarray([fine_foot_names.index(row["footwork_baseline"]) for row in fine_rows])
        ],
        "state": state_map[
            np.asarray([fine_state_names.index(row["state_baseline"]) for row in fine_rows])
        ],
    }
    direct_broad = {head: np.full(len(samples), -1, dtype=np.int64) for head in HEADS}
    outputs = {
        candidate: {head: np.full(len(samples), -1, dtype=np.int64) for head in HEADS}
        for candidate in ("no_decoder", "matched_decoder", "sequence")
    }
    output_probabilities = {
        candidate: {
            head: np.zeros((len(samples), len(NAMES[head])), dtype=np.float64)
            for head in HEADS
        }
        for candidate in ("direct_broad", "no_decoder", "matched_decoder", "sequence")
    }
    selections = []
    for outer_index, outer_group in enumerate(group_names):
        print(f"outer {outer_index + 1}/10 source={outer_group}", flush=True)
        outer_train = groups != outer_group
        outer_test = groups == outer_group
        inner_groups = [group for group in group_names if group != outer_group]
        for head_index, head in enumerate(HEADS):
            y = target(samples, head)
            inner_by_view = {}
            outer_by_view = {}
            chosen_configs = {}
            config_scores = {}
            for view_index, view in enumerate(VIEWS):
                candidates = {}
                config_scores[view] = {}
                config_options = (
                    {"balanced_leaf5": CONFIGS["balanced_leaf5"]}
                    if view == "engineered_dual"
                    else CONFIGS
                )
                for config_name, config in config_options.items():
                    inner_prob = np.zeros((len(samples), len(NAMES[head])), dtype=np.float64)
                    for inner_index, inner_group in enumerate(inner_groups):
                        inner_test = groups == inner_group
                        inner_train = outer_train & ~inner_test
                        prediction = fit_predict(
                            samples,
                            views[head][view],
                            y,
                            head,
                            config,
                            inner_train,
                            inner_test,
                            100_000
                            + outer_index * 10_000
                            + head_index * 2_000
                            + view_index * 300
                            + inner_index * 4,
                        )
                        inner_prob[inner_test] = prediction
                    score = objective(
                        samples,
                        y,
                        np.argmax(inner_prob, axis=1),
                        outer_train,
                    )
                    candidates[config_name] = inner_prob
                    config_scores[view][config_name] = list(score)
                maximum = max(value[0] for value in config_scores[view].values())
                eligible_configs = [
                    name
                    for name, values in config_scores[view].items()
                    if values[0] >= maximum - TOLERANCE
                ]
                config_priority = {
                    "balanced_leaf5": 0,
                    "balanced_leaf2": 1,
                    "plain_leaf2": 2,
                }
                chosen = min(
                    eligible_configs,
                    key=lambda name: (
                        config_priority[name],
                        tuple(-float(value) for value in config_scores[view][name]),
                    ),
                )
                chosen_configs[view] = chosen
                inner_by_view[view] = candidates[chosen]
                outer_by_view[view] = fit_predict(
                    samples,
                    views[head][view],
                    y,
                    head,
                    CONFIGS[chosen],
                    outer_train,
                    outer_test,
                    5000 + outer_index
                    if head == "state" and view == "engineered_dual"
                    else 6000 + outer_index * 4
                    if head == "footwork" and view == "engineered_dual"
                    else 200_000
                    + outer_index * 10_000
                    + head_index * 2_000
                    + view_index * 100,
                )
            direct_broad[head][outer_test] = np.argmax(
                outer_by_view["engineered_dual"], axis=1
            )
            output_probabilities["direct_broad"][head][outer_test] = outer_by_view[
                "engineered_dual"
            ]
            grid = []
            for weights in WEIGHTS:
                fused = combine(inner_by_view, weights)
                for pair_weight in PAIR_WEIGHTS:
                    for transition_weight in TRANSITION_WEIGHTS:
                        prediction = np.full(len(samples), -1, dtype=np.int64)
                        for inner_group in inner_groups:
                            inner_test = groups == inner_group
                            inner_train = outer_train & ~inner_test
                            log_pair, log_transition = priors(
                                samples, y, inner_train, len(NAMES[head])
                            )
                            decoded = decode(
                                samples,
                                fused,
                                inner_test,
                                log_pair,
                                log_transition,
                                pair_weight,
                                transition_weight,
                            )
                            prediction[inner_test] = decoded[inner_test]
                        grid.append(
                            {
                                "weights": list(weights),
                                "pair_weight": pair_weight,
                                "transition_weight": transition_weight,
                                "objective": list(objective(samples, y, prediction, outer_train)),
                            }
                        )
            no_decoder = select_setting(
                [
                    row
                    for row in grid
                    if float(row["pair_weight"]) == 0
                    and float(row["transition_weight"]) == 0
                ]
            )
            no_decoder_weights = tuple(float(value) for value in no_decoder["weights"])
            selected_by_candidate = {
                "no_decoder": no_decoder,
                "matched_decoder": select_setting(
                    [
                        row
                        for row in grid
                        if tuple(float(value) for value in row["weights"])
                        == no_decoder_weights
                    ]
                ),
                "sequence": select_setting(grid),
            }
            for candidate, selected in selected_by_candidate.items():
                weights = tuple(float(value) for value in selected["weights"])
                fused_outer = combine(outer_by_view, weights)
                output_probabilities[candidate][head][outer_test] = fused_outer
                full_prob = np.zeros((len(samples), len(NAMES[head])), dtype=np.float64)
                full_prob[outer_test] = fused_outer
                log_pair, log_transition = priors(samples, y, outer_train, len(NAMES[head]))
                decoded = decode(
                    samples,
                    full_prob,
                    outer_test,
                    log_pair,
                    log_transition,
                    float(selected["pair_weight"]),
                    float(selected["transition_weight"]),
                )
                outputs[candidate][head][outer_test] = decoded[outer_test]
                selections.append(
                    {
                        "outer_source": outer_group,
                        "head": head,
                        "candidate": candidate,
                        "configs": chosen_configs,
                        "config_scores": config_scores,
                        "selected": selected,
                        "selection_grid": grid,
                    }
                )
    if any(np.any(values < 0) for candidate in outputs.values() for values in candidate.values()):
        raise RuntimeError("Missing outer predictions")
    if any(np.any(values < 0) for values in direct_broad.values()):
        raise RuntimeError("Missing direct broad baseline predictions")

    metrics = {
        "mapped_fine_reference": {},
        "direct_broad_baseline": {},
        "no_decoder": {},
        "matched_decoder": {},
        "sequence": {},
    }
    for head in HEADS:
        y = target(samples, head)
        metrics["mapped_fine_reference"][head] = metric(
            samples, head, y, mapped_fine[head]
        )
        metrics["direct_broad_baseline"][head] = metric(
            samples, head, y, direct_broad[head]
        )
        for candidate in ("no_decoder", "matched_decoder", "sequence"):
            metrics[candidate][head] = metric(
                samples, head, y, outputs[candidate][head]
            )
    correctness = {}
    for name, predictions in {
        "mapped_fine_reference": mapped_fine,
        "direct_broad_baseline": direct_broad,
        **outputs,
    }.items():
        joint = np.ones(len(samples), dtype=bool)
        for head in HEADS:
            joint &= predictions[head] == target(samples, head)
        metrics[name]["joint_exact_accuracy"] = float(np.mean(joint))
        correctness[name] = joint
    inference = {}
    for candidate_index, candidate in enumerate(("no_decoder", "matched_decoder", "sequence")):
        inference[candidate] = {}
        for head_index, head in enumerate(HEADS):
            y = target(samples, head)
            inference[candidate][head] = core.paired_accuracy_inference(
                samples,
                direct_broad[head] == y,
                outputs[candidate][head] == y,
                20262000 + candidate_index * 10 + head_index,
            )
        inference[candidate]["joint_exact"] = core.paired_accuracy_inference(
            samples,
            correctness["direct_broad_baseline"],
            correctness[candidate],
            20262020 + candidate_index,
        )
    for comparison_index, candidate in enumerate(("matched_decoder", "sequence")):
        comparison = f"{candidate}_vs_no_decoder"
        inference[comparison] = {}
        for head_index, head in enumerate(HEADS):
            y = target(samples, head)
            inference[comparison][head] = core.paired_accuracy_inference(
                samples,
                outputs["no_decoder"][head] == y,
                outputs[candidate][head] == y,
                20262100 + comparison_index * 10 + head_index,
            )
        inference[comparison]["joint_exact"] = core.paired_accuracy_inference(
            samples,
            correctness["no_decoder"],
            correctness[candidate],
            20262120 + comparison_index,
        )
    prediction_rows = []
    probability_rows = []
    for index, sample in enumerate(samples):
        prediction_rows.append(
            {
                "clip_id": sample["clip_id"],
                "source_group": sample["source_group"],
                "segment_idx": sample["segment_idx"],
                "side": sample["side"],
                "footwork_truth": NAMES["footwork"][target(samples, "footwork")[index]],
                "footwork_mapped_fine": NAMES["footwork"][mapped_fine["footwork"][index]],
                "footwork_direct_broad": NAMES["footwork"][direct_broad["footwork"][index]],
                "footwork_no_decoder": NAMES["footwork"][outputs["no_decoder"]["footwork"][index]],
                "footwork_matched_decoder": NAMES["footwork"][outputs["matched_decoder"]["footwork"][index]],
                "footwork_sequence": NAMES["footwork"][outputs["sequence"]["footwork"][index]],
                "state_truth": NAMES["state"][target(samples, "state")[index]],
                "state_mapped_fine": NAMES["state"][mapped_fine["state"][index]],
                "state_direct_broad": NAMES["state"][direct_broad["state"][index]],
                "state_no_decoder": NAMES["state"][outputs["no_decoder"]["state"][index]],
                "state_matched_decoder": NAMES["state"][outputs["matched_decoder"]["state"][index]],
                "state_sequence": NAMES["state"][outputs["sequence"]["state"][index]],
            }
        )
        for model_name, by_head in output_probabilities.items():
            for head in HEADS:
                for class_index, class_name in enumerate(NAMES[head]):
                    probability_rows.append(
                        {
                            "sample_index": index,
                            "model": model_name,
                            "stage": "predecoder_emission",
                            "head": head,
                            "class": class_name,
                            "probability": float(by_head[head][index, class_index]),
                        }
                    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "predictions.csv", prediction_rows)
    write_csv(args.out_dir / "predecoder_probabilities.csv", probability_rows)
    write_json(args.out_dir / "nested_selections.json", selections)
    report = {
        "schema_version": "downstream_broad_state_sequence_v1",
        "status": "retrospective_exploratory",
        "protocol": "../BROAD_SEQUENCE_PROTOCOL.md",
        "dataset": {
            "exchanges": 50,
            "source_videos": 10,
            "segments": 326,
            "samples": 652,
        },
        "metadata": metadata,
        "metrics": metrics,
        "paired_equal_source_inference": {
            "statistic": "equal-source mean accuracy delta",
            "comparisons": inference,
        },
        "inputs": {
            "annotations": str(args.annotations),
            "annotation_files": len(list(args.annotations.glob("*.json"))),
            "raw_context": str(args.raw_context),
            "raw_context_sha256": core.sha256(args.raw_context),
            "stabilized_context": str(args.stabilized_context),
            "stabilized_context_sha256": core.sha256(args.stabilized_context),
            "base_model": str(args.base_model),
            "base_model_sha256": core.sha256(args.base_model),
            "fine_predictions": str(args.fine_predictions),
            "fine_predictions_sha256": core.sha256(args.fine_predictions),
            "annotation_manifest": manifest_digest(list(args.annotations.glob("*.json"))),
            "raw_l2": str(args.raw_l2),
            "raw_l2_manifest": manifest_digest(
                [args.raw_l2 / f"{cid}_l2.json" for cid in sorted({str(row['clip_id']) for row in samples})]
            ),
            "stabilized_l2": str(args.stabilized_l2),
            "stabilized_l2_manifest": manifest_digest(
                [
                    args.stabilized_l2 / f"{cid}_l2.json"
                    for cid in sorted({str(row['clip_id']) for row in samples})
                ]
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "runtime_sec": float(time.perf_counter() - started),
    }
    write_json(args.out_dir / "summary.json", report)
    lines = [
        "# Broad-State Sequence Study",
        "",
        "| Model | Footwork acc. | Footwork macro F1 | State acc. | State macro F1 | Joint exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "mapped_fine_reference",
        "direct_broad_baseline",
        "no_decoder",
        "matched_decoder",
        "sequence",
    ):
        row = metrics[name]
        lines.append(
            f"| {name.replace('_', ' ')} | {row['footwork']['accuracy']:.4f} | "
            f"{row['footwork']['macro_f1']:.4f} | {row['state']['accuracy']:.4f} | "
            f"{row['state']['macro_f1']:.4f} | {row['joint_exact_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "All settings are selected inside each outer fold. These are broad-label",
            "heads with the ontology declared in BROAD_SEQUENCE_PROTOCOL.md.",
            "",
        ]
    )
    (args.out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
