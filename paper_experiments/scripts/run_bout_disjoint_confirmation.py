#!/usr/bin/env python3
"""Run a manifest-defined source-bout-held-out retraining analysis.

The rule and protocol are fixed before retraining and held-out inference.  Test
labels are read when constructing the quality-filtered evaluation manifest,
but never for source selection, training, extractor selection, or seed
selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from generate_boundary_predictions import engineered_l2_instability_score


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "paper_experiments/round2/bout_disjoint_confirmation"
SOURCE = ROOT / "paper_experiments/experiments/truncated_retraining"
TRAIN_MANIFEST = SOURCE / "manifests/train.csv"
DEV_MANIFEST = SOURCE / "manifests/paper_development.csv"
REFERENCE_MANIFEST = SOURCE / "manifests/paper_final_candidates.csv"
CONFIRMATION_MANIFEST = (
    ROOT
    / "paper_experiments/round2/annotation_extension/manifests/"
    "foil_round2_exchange_disjoint_all_124.csv"
)
CONFIRMATION_GOLD = (
    ROOT
    / "paper_experiments/round2/annotation_extension/packages/"
    "foil_round2_extension_124_consensus/consensus"
)
DEV_GOLD = ROOT / "paper_experiments/annotations/raw/owner"
BASE_CONFIG = (
    SOURCE / "candidates/conservative_smooth02_seed18/config.yaml"
)
MAIN_PER_CLIP = (
    ROOT
    / "paper_experiments/round2/annotation_extension/"
    "final_consensus_evaluation/main_per_clip.csv"
)
OPENMMLAB_PY = Path(os.environ.get("RELIS_PYTHON", sys.executable))
SEEDS = tuple(range(20270614, 20270619))
SEARCH_SEED = 20260810
SEARCH_TRIALS = 2000
EXPECTED_TEST_SOURCES = (
    "20", "24", "30", "32", "36", "51", "52", "59", "65",
    "74", "77", "87", "88", "89", "92", "97", "98",
)
KEY_METRICS = (
    "activity_accuracy",
    "active_f1",
    "boundary_f1_pm5",
    "boundary_f1_pm10",
)
FIXED_BASELINES = (
    "motion",
    "selfsim",
    "changepoint",
    "kernel_changepoint",
)
ENGINEERED_METHOD = "engineered_l2_instability"
BOOTSTRAP_SEED = 20260813


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(materialized[0])
    for row in materialized[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    """Return a repository-relative provenance path when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def hash_record(path: Path) -> dict[str, object]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def source_id(clip_id: str) -> str:
    fields = clip_id.split("-")
    if len(fields) != 3:
        raise ValueError(f"unexpected clip ID: {clip_id}")
    return fields[1]


def usable(annotation: dict[str, Any]) -> bool:
    flags = annotation.get("quality_flags", {})
    return (
        bool(annotation.get("annotation_complete"))
        and bool(flags.get("valid", True))
        and not bool(flags.get("not_exchange", False))
    )


def clean(annotation: dict[str, Any]) -> bool:
    if not usable(annotation):
        return False
    return not any(
        bool(value)
        for key, value in annotation.get("quality_flags", {}).items()
        if key != "valid"
    )


def normalize_to_model_input(row: dict[str, str]) -> dict[str, str]:
    """Align derived manifest endpoints with the hit-truncated L2 sequence."""
    normalized = dict(row)
    path = Path(row["l2_path"])
    if not path.is_absolute():
        path = ROOT / path
    sequence = read_json(path)
    frames = sequence["frames"]
    if not frames:
        raise RuntimeError(f"empty L2 frame list: {path}")
    normalized["frame_start"] = str(frames[0])
    normalized["frame_end"] = str(frames[-1])
    normalized["num_frames"] = str(len(frames))
    if int(normalized["frame_end"]) != int(normalized["hit_frame"]):
        raise RuntimeError(f"L2 endpoint does not match scoring light: {row['clip_id']}")
    return normalized


def run(command: list[object]) -> None:
    rendered = [str(item) for item in command]
    print("+", " ".join(rendered), flush=True)
    subprocess.run(rendered, cwd=ROOT, check=True)


def reset_symlink_dir(path: Path, rows: list[dict[str, str]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for existing in path.iterdir():
        if existing.is_symlink():
            existing.unlink()
        else:
            raise RuntimeError(f"refusing to remove non-symlink: {existing}")
    for row in rows:
        source = Path(row["l2_path"])
        if not source.is_absolute():
            source = ROOT / source
        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        (path / source.name).symlink_to(source)


def split_from_manifests() -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    train_rows = read_csv(TRAIN_MANIFEST)
    dev_rows = read_csv(DEV_MANIFEST)
    reference_rows = read_csv(REFERENCE_MANIFEST)
    confirmation_rows = read_csv(CONFIRMATION_MANIFEST)
    dev_sources = {source_id(row["clip_id"]) for row in dev_rows}
    reference_sources = {source_id(row["clip_id"]) for row in reference_rows}
    confirmation_sources = {source_id(row["clip_id"]) for row in confirmation_rows}
    selected = sorted(
        confirmation_sources - reference_sources - dev_sources,
        key=int,
    )
    if tuple(selected) != EXPECTED_TEST_SOURCES:
        raise RuntimeError(f"source-rule drift: {selected}")
    selected_set = set(selected)
    test_planned = [
        normalize_to_model_input(row) for row in confirmation_rows
        if source_id(row["clip_id"]) in selected_set
    ]
    train_retained = [
        row for row in train_rows
        if source_id(row["clip_id"]) not in selected_set
    ]
    return selected, train_retained, test_planned


def prepare() -> None:
    selected, train_rows, test_planned = split_from_manifests()
    annotations = {
        path.stem: read_json(path)
        for path in sorted(CONFIRMATION_GOLD.glob("*.json"))
    }
    test_valid = [
        row for row in test_planned
        if row["clip_id"] in annotations and usable(annotations[row["clip_id"]])
    ]
    if (len(selected), len(train_rows), len(test_planned), len(test_valid)) != (17, 746, 33, 31):
        raise RuntimeError(
            "unexpected split counts: "
            f"{len(selected)}, {len(train_rows)}, {len(test_planned)}, {len(test_valid)}"
        )
    dev_rows = read_csv(DEV_MANIFEST)
    train_sources = {source_id(row["clip_id"]) for row in train_rows}
    dev_sources = {source_id(row["clip_id"]) for row in dev_rows}
    test_sources = set(selected)
    if train_sources & test_sources or dev_sources & test_sources:
        raise RuntimeError("held-out source leakage into training or development")

    write_csv(WORK / "manifests/train.csv", train_rows)
    write_csv(WORK / "manifests/test_planned.csv", test_planned)
    write_csv(WORK / "manifests/test_valid.csv", test_valid)
    reset_symlink_dir(WORK / "data/train", train_rows)
    report = {
        "protocol": "manifest-membership-only confirmation-source novelty",
        "test_source_rule": (
            "confirmation source IDs minus original-reference source IDs "
            "minus original-development source IDs"
        ),
        "test_sources": selected,
        "counts": {
            "training_clips": len(train_rows),
            "training_sources": len(train_sources),
            "development_clips": len(dev_rows),
            "development_sources": len(dev_sources),
            "test_planned_clips": len(test_planned),
            "test_usable_clips": len(test_valid),
            "test_clean_clips": sum(clean(annotations[row["clip_id"]]) for row in test_planned),
            "test_sources": len(test_sources),
        },
        "integrity": {
            "train_test_source_overlap": sorted(train_sources & test_sources, key=int),
            "development_test_source_overlap": sorted(dev_sources & test_sources, key=int),
            "manifest_sha256": {
                "train_source": sha256(TRAIN_MANIFEST),
                "development_source": sha256(DEV_MANIFEST),
                "reference_source": sha256(REFERENCE_MANIFEST),
                "confirmation_source": sha256(CONFIRMATION_MANIFEST),
                "train_retained": sha256(WORK / "manifests/train.csv"),
                "test_planned": sha256(WORK / "manifests/test_planned.csv"),
                "test_valid": sha256(WORK / "manifests/test_valid.csv"),
            },
        },
        "seeds": list(SEEDS),
        "extractor_search": {
            "random_seed": SEARCH_SEED,
            "random_candidates_per_score_and_seed": SEARCH_TRIALS,
            "score_sources": ["b", "phi_l2", ENGINEERED_METHOD],
        },
        "test_clip_ids": [row["clip_id"] for row in test_valid],
    }
    write_json(WORK / "split_report.json", report)
    print(json.dumps(report, indent=2))


def base_seed_config(seed: int) -> dict[str, Any]:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg.pop("_config_path", None)
    cfg["seed"] = seed
    cfg["data"]["data_dir"] = str((WORK / "data/train").resolve())
    cfg["train"]["out_dir"] = str((WORK / f"runs/seed{seed}").resolve())
    cfg["train"]["max_steps"] = 1000
    cfg["train"]["ckpt_every"] = 1000
    cfg["train"]["early_stop"]["enabled"] = False
    return cfg


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(payload, sort_keys=False)
    if not path.exists() or path.read_text(encoding="utf-8") != rendered:
        path.write_text(rendered, encoding="utf-8")


def config_path(seed: int, tuned: str | None = None) -> Path:
    if tuned is None:
        return WORK / f"configs/seed{seed}.yaml"
    return WORK / f"configs_tuned/{tuned}/seed{seed}.yaml"


def checkpoint_path(seed: int) -> Path:
    return WORK / f"runs/seed{seed}/ckpt_step_1000.pt"


def engineered_config_path() -> Path:
    return WORK / f"configs_tuned/{ENGINEERED_METHOD}.yaml"


def engineered_prediction_path() -> Path:
    return WORK / f"predictions/{ENGINEERED_METHOD}/{ENGINEERED_METHOD}.json"


def normalized_checkpoint_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = dict(payload)
    config.pop("_config_path", None)
    return config


def verify_training_links() -> None:
    expected = {}
    for row in read_csv(WORK / "manifests/train.csv"):
        source = Path(row["l2_path"])
        if not source.is_absolute():
            source = ROOT / source
        expected[source.name] = source.resolve()
    link_dir = WORK / "data/train"
    observed = {
        path.name: path.resolve()
        for path in link_dir.iterdir()
        if path.is_symlink()
    }
    if observed != expected:
        raise RuntimeError("training symlink set does not match the retained manifest")


def make_configs() -> None:
    for seed in SEEDS:
        write_yaml(config_path(seed), base_seed_config(seed))


def train() -> None:
    make_configs()
    verify_training_links()
    for seed in SEEDS:
        checkpoint = checkpoint_path(seed)
        if checkpoint.exists():
            saved = __import__("torch").load(checkpoint, map_location="cpu")
            saved_cfg = normalized_checkpoint_config(saved.get("cfg", {}))
            expected_cfg = yaml.safe_load(config_path(seed).read_text(encoding="utf-8"))
            if int(saved.get("step", -1)) != 1000 or saved_cfg != expected_cfg:
                raise RuntimeError(
                    f"checkpoint/config mismatch; preserve or replace explicitly: {checkpoint}"
                )
            print(f"reusing {checkpoint}")
            continue
        run([
            OPENMMLAB_PY,
            "boundary_segmentation/train.py",
            "--config", config_path(seed),
            "--device", "auto",
        ])
        if not checkpoint.exists():
            final = checkpoint.parent / "ckpt_final.pt"
            if not final.exists():
                raise FileNotFoundError(checkpoint)
            shutil.copy2(final, checkpoint)
    provenance = {
        "training_manifest": hash_record(WORK / "manifests/train.csv"),
        "checkpoints": {
            str(seed): {
                "config": hash_record(config_path(seed)),
                "checkpoint": hash_record(checkpoint_path(seed)),
                "step": 1000,
            }
            for seed in SEEDS
        },
    }
    write_json(WORK / "checkpoint_provenance.json", provenance)


def search(seed: int, score: str) -> dict[str, Any]:
    name = f"relis_seed{seed}"
    search_dir = WORK / "postprocess_search"
    common: list[object] = [
        OPENMMLAB_PY,
        "paper_experiments/scripts/search_boundary_postprocess.py",
        "--manifest", DEV_MANIFEST,
        "--owner-dir", DEV_GOLD,
        "--model", f"{name}={config_path(seed)}={checkpoint_path(seed)}",
    ]
    common += [
        "--score-source", score,
        "--trials", SEARCH_TRIALS,
        "--random-only",
        "--seed", SEARCH_SEED,
        "--top-k", 100,
        "--out-dir", search_dir,
    ]
    run(common)
    best_path = search_dir / f"{name}-{score}_best.json"
    payload = read_json(best_path)
    protocol = payload["candidate_protocol"]
    if not (
        protocol["random_only"]
        and not protocol["include_deployed_candidate"]
        and int(protocol["total_candidates"]) == SEARCH_TRIALS
        and payload["best"]["candidate_source"] == "random"
    ):
        raise RuntimeError(f"invalid search provenance: {best_path}")
    return payload


def cache_engineered_scores() -> Path:
    """Materialize the fixed engineered score on development clips."""
    cache_path = WORK / f"postprocess_search/score_cache/{ENGINEERED_METHOD}.npz"
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    k = int(base["model"]["boundary"].get("k", 8))
    rows = read_csv(DEV_MANIFEST)
    arrays: dict[str, np.ndarray] = {}
    for row in rows:
        path = Path(row["l2_path"])
        if not path.is_absolute():
            path = ROOT / path
        data = read_json(path)
        score, mask = engineered_l2_instability_score(data, k=k)
        clip_id = row["clip_id"]
        frames = np.asarray(data["frames"], dtype=np.int32)
        if not (len(score) == len(mask) == len(frames)):
            raise RuntimeError(f"engineered score length mismatch: {clip_id}")
        arrays[f"b::{clip_id}"] = score.astype(np.float32)
        arrays[f"phi::{clip_id}"] = np.zeros((len(score), 1), dtype=np.float32)
        arrays[f"mask::{clip_id}"] = mask.astype(np.bool_)
        arrays[f"frames::{clip_id}"] = frames
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    write_json(
        cache_path.with_suffix(".json"),
        {
            "method": ENGINEERED_METHOD,
            "development_manifest": hash_record(DEV_MANIFEST),
            "base_config": hash_record(BASE_CONFIG),
            "clips": [row["clip_id"] for row in rows],
            "cache": hash_record(cache_path),
        },
    )
    return cache_path


def search_engineered() -> dict[str, Any]:
    cache_path = cache_engineered_scores()
    search_dir = WORK / "postprocess_search"
    run([
        OPENMMLAB_PY,
        "paper_experiments/scripts/search_boundary_postprocess.py",
        "--manifest", DEV_MANIFEST,
        "--owner-dir", DEV_GOLD,
        "--score-cache", f"{ENGINEERED_METHOD}={cache_path}",
        "--score-source", "b",
        "--trials", SEARCH_TRIALS,
        "--random-only",
        "--seed", SEARCH_SEED,
        "--top-k", 100,
        "--out-dir", search_dir,
    ])
    best_path = search_dir / f"{ENGINEERED_METHOD}-b_best.json"
    payload = read_json(best_path)
    protocol = payload["candidate_protocol"]
    if not (
        protocol["random_only"]
        and not protocol["include_deployed_candidate"]
        and int(protocol["total_candidates"]) == SEARCH_TRIALS
        and payload["best"]["candidate_source"] == "random"
    ):
        raise RuntimeError(f"invalid engineered-search provenance: {best_path}")
    return payload


def development_annotation_records() -> list[dict[str, object]]:
    records = []
    for row in read_csv(DEV_MANIFEST):
        path = DEV_GOLD / f"{row['clip_id']}.json"
        if path.exists():
            records.append(hash_record(path))
    return records


def selection_input_hashes() -> dict[str, object]:
    return {
        "training_manifest": hash_record(WORK / "manifests/train.csv"),
        "development_manifest": hash_record(DEV_MANIFEST),
        "development_annotations": development_annotation_records(),
        "engineered_baseline": {
            "score_cache": hash_record(
                WORK / f"postprocess_search/score_cache/{ENGINEERED_METHOD}.npz"
            ),
            "score_cache_metadata": hash_record(
                WORK / f"postprocess_search/score_cache/{ENGINEERED_METHOD}.json"
            ),
            "search": hash_record(
                WORK / f"postprocess_search/{ENGINEERED_METHOD}-b_best.json"
            ),
            "config": hash_record(engineered_config_path()),
        },
        "per_seed": {
            str(seed): {
                "checkpoint": hash_record(checkpoint_path(seed)),
                "base_config": hash_record(config_path(seed)),
                "full_model_search": hash_record(
                    WORK / f"postprocess_search/relis_seed{seed}-b_best.json"
                ),
                "raw_phi_l2_search": hash_record(
                    WORK / f"postprocess_search/relis_seed{seed}-phi_l2_best.json"
                ),
                "full_model_config": hash_record(config_path(seed, "full_model")),
                "raw_phi_l2_config": hash_record(config_path(seed, "raw_phi_l2")),
            }
            for seed in SEEDS
        },
    }


def tune() -> None:
    make_configs()
    selections: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        if not checkpoint_path(seed).exists():
            raise FileNotFoundError(checkpoint_path(seed))
        selections[str(seed)] = {}
        for score, label in (("b", "full_model"), ("phi_l2", "raw_phi_l2")):
            payload = search(seed, score)
            tuned = yaml.safe_load(config_path(seed).read_text(encoding="utf-8"))
            tuned["postprocess"] = payload["best"]["postprocess"]
            write_yaml(config_path(seed, label), tuned)
            selections[str(seed)][label] = {
                "sample_index": payload["best"]["sample_index"],
                "development": payload["best"]["summary"],
                "postprocess": payload["best"]["postprocess"],
            }

    engineered = search_engineered()
    engineered_config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    engineered_config.pop("_config_path", None)
    engineered_config["postprocess"] = engineered["best"]["postprocess"]
    write_yaml(engineered_config_path(), engineered_config)

    selected: dict[str, int] = {}
    for method in ("full_model", "raw_phi_l2"):
        selected[method] = max(
            SEEDS,
            key=lambda seed: (
                float(selections[str(seed)][method]["development"]["objective"]),
                float(selections[str(seed)][method]["development"]["balanced_objective"]),
                -seed,
            ),
        )
    payload = {
        "selection_uses_test_labels": False,
        "selection_rule": (
            "maximum development objective, then balanced objective, then smaller seed"
        ),
        "selected_seed": selected,
        "per_seed": selections,
        "engineered_baseline": {
            "sample_index": engineered["best"]["sample_index"],
            "development": engineered["best"]["summary"],
            "postprocess": engineered["best"]["postprocess"],
        },
        "input_sha256": selection_input_hashes(),
    }
    write_json(WORK / "model_selection.json", payload)
    print(json.dumps(payload, indent=2))


def predict() -> None:
    selection_path = WORK / "model_selection.json"
    if not selection_path.exists():
        raise RuntimeError("run tune before generating held-out predictions")
    selection = read_json(selection_path)
    if selection.get("input_sha256") != selection_input_hashes():
        raise RuntimeError("selection inputs changed; rerun tune before held-out inference")
    for seed in SEEDS:
        for method in ("full_model", "raw_phi_l2"):
            run([
                OPENMMLAB_PY,
                "paper_experiments/scripts/generate_boundary_predictions.py",
                "--manifest", WORK / "manifests/test_valid.csv",
                "--out-dir", WORK / f"predictions/seed{seed}/{method}",
                "--config", config_path(seed, method),
                "--ckpt", checkpoint_path(seed),
                "--python-bin", OPENMMLAB_PY,
                "--method", method,
            ])
    run([
        OPENMMLAB_PY,
        "paper_experiments/scripts/generate_boundary_predictions.py",
        "--manifest", WORK / "manifests/test_valid.csv",
        "--out-dir", WORK / f"predictions/{ENGINEERED_METHOD}",
        "--config", engineered_config_path(),
        "--ckpt", checkpoint_path(SEEDS[0]),
        "--python-bin", OPENMMLAB_PY,
        "--method", ENGINEERED_METHOD,
    ])
    provenance = {
        "selection": hash_record(selection_path),
        "test_manifest": hash_record(WORK / "manifests/test_valid.csv"),
        "predictions": {
            f"{seed}/{method}": hash_record(
                WORK / f"predictions/seed{seed}/{method}/{method}.json"
            )
            for seed in SEEDS
            for method in ("full_model", "raw_phi_l2")
        },
    }
    provenance["predictions"][ENGINEERED_METHOD] = hash_record(
        engineered_prediction_path()
    )
    write_json(WORK / "prediction_provenance.json", provenance)


def evaluate() -> None:
    for seed in SEEDS:
        for method in ("full_model", "raw_phi_l2"):
            pred = WORK / f"predictions/seed{seed}/{method}/{method}.json"
            if not pred.exists():
                raise FileNotFoundError(pred)
            out = WORK / f"evaluation/seed{seed}/{method}"
            run([
                sys.executable,
                "paper_experiments/scripts/compare_annotations_to_model.py",
                "--owner-dir", CONFIRMATION_GOLD,
                "--manifest", WORK / "manifests/test_valid.csv",
                "--predictions", pred,
                "--summary-out", out / "summary.csv",
                "--by-clip-out", out / "by_clip.csv",
                "--viewer-metrics-out", out / "viewer_metrics.csv",
                "--model-annotations-dir",
                WORK / f"evaluation/model_annotations/seed{seed}/{method}",
            ])
    pred = engineered_prediction_path()
    if not pred.exists():
        raise FileNotFoundError(pred)
    out = WORK / f"evaluation/{ENGINEERED_METHOD}"
    run([
        sys.executable,
        "paper_experiments/scripts/compare_annotations_to_model.py",
        "--owner-dir", CONFIRMATION_GOLD,
        "--manifest", WORK / "manifests/test_valid.csv",
        "--predictions", pred,
        "--summary-out", out / "summary.csv",
        "--by-clip-out", out / "by_clip.csv",
        "--viewer-metrics-out", out / "viewer_metrics.csv",
        "--model-annotations-dir",
        WORK / f"evaluation/model_annotations/{ENGINEERED_METHOD}",
    ])


def valid_summary(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return next(row for row in rows if row["cohort"] == "valid")


def mean_rows(rows: list[dict[str, str]], metric: str) -> float:
    return statistics.mean(float(row[metric]) for row in rows)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap(
    selected_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    repeats: int = 10_000,
) -> list[dict[str, object]]:
    selected = {row["clip_id"]: row for row in selected_rows}
    baseline = {row["clip_id"]: row for row in baseline_rows}
    if set(selected) != set(baseline):
        raise RuntimeError("bootstrap clip mismatch")
    clusters: dict[str, list[str]] = defaultdict(list)
    for clip_id in selected:
        clusters[source_id(clip_id)].append(clip_id)
    keys = sorted(clusters, key=int)
    rng = random.Random(BOOTSTRAP_SEED)
    output = []
    for metric in KEY_METRICS:
        absolute: list[float] = []
        delta: list[float] = []
        for _ in range(repeats):
            sampled = [rng.choice(keys) for _ in keys]
            ids = [clip for key in sampled for clip in clusters[key]]
            a = statistics.mean(float(selected[clip][metric]) for clip in ids)
            b = statistics.mean(float(baseline[clip][metric]) for clip in ids)
            absolute.append(a)
            delta.append(a - b)
        point = statistics.mean(float(row[metric]) for row in selected_rows)
        base = statistics.mean(float(row[metric]) for row in baseline_rows)
        output.append({
            "metric": metric,
            "selected_relis": point,
            "engineered_baseline": base,
            "delta": point - base,
            "relis_ci_low": quantile(absolute, 0.025),
            "relis_ci_high": quantile(absolute, 0.975),
            "delta_ci_low": quantile(delta, 0.025),
            "delta_ci_high": quantile(delta, 0.975),
            "bootstrap_repeats": repeats,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "source_clusters": len(keys),
        })
    return output


def aggregate() -> None:
    selection = read_json(WORK / "model_selection.json")
    per_seed: list[dict[str, object]] = []
    by_clip: dict[tuple[int, str], list[dict[str, str]]] = {}
    for seed in SEEDS:
        for method in ("full_model", "raw_phi_l2"):
            summary = valid_summary(WORK / f"evaluation/seed{seed}/{method}/summary.csv")
            rows = read_csv(WORK / f"evaluation/seed{seed}/{method}/by_clip.csv")
            expected_ids = {
                row["clip_id"] for row in read_csv(WORK / "manifests/test_valid.csv")
            }
            row_ids = [row["clip_id"] for row in rows]
            if len(row_ids) != len(set(row_ids)) or set(row_ids) != expected_ids:
                raise RuntimeError(f"evaluation clip mismatch for {seed}/{method}")
            by_clip[(seed, method)] = rows
            dev = selection["per_seed"][str(seed)][method]["development"]
            row: dict[str, object] = {
                "method": method,
                "seed": seed,
                "selected_by_development": seed == selection["selected_seed"][method],
                "development_objective": dev["objective"],
                "development_active_f1": dev["active_f1"],
                "clips": summary["clips"],
            }
            row.update({metric: summary[metric] for metric in KEY_METRICS})
            row["model_active_coverage"] = summary["model_active_coverage"]
            row["human_active_coverage"] = summary["human_active_coverage"]
            per_seed.append(row)
    write_csv(WORK / "per_seed_results.csv", per_seed)

    aggregate_rows: list[dict[str, object]] = []
    for method in ("full_model", "raw_phi_l2"):
        rows = [row for row in per_seed if row["method"] == method]
        selected = next(row for row in rows if row["selected_by_development"])
        selected_row: dict[str, object] = {
            "method": method,
            "protocol": "development_selected_seed",
            "seed": selected["seed"],
            "clips": selected["clips"],
        }
        selected_row.update({metric: selected[metric] for metric in KEY_METRICS})
        aggregate_rows.append(selected_row)
        mean_row: dict[str, object] = {
            "method": method,
            "protocol": "five_seed_mean",
            "seed": "mean_sd",
            "clips": selected["clips"],
        }
        for metric in KEY_METRICS:
            values = [float(row[metric]) for row in rows]
            mean_row[metric] = statistics.mean(values)
            mean_row[f"{metric}_sd"] = statistics.stdev(values)
            mean_row[f"{metric}_min"] = min(values)
            mean_row[f"{metric}_max"] = max(values)
        aggregate_rows.append(mean_row)

    test_ids = {
        row["clip_id"] for row in read_csv(WORK / "manifests/test_valid.csv")
    }
    main_rows = read_csv(MAIN_PER_CLIP)
    baseline_by_method: dict[str, list[dict[str, str]]] = {}
    for method in FIXED_BASELINES:
        rows = [
            row for row in main_rows
            if row["method"] == method and row["clip_id"] in test_ids
        ]
        row_ids = [row["clip_id"] for row in rows]
        if len(row_ids) != len(set(row_ids)) or set(row_ids) != test_ids:
            raise RuntimeError(f"baseline {method} has {len(rows)} test rows")
        baseline_by_method[method] = rows
        result: dict[str, object] = {
            "method": method,
            "protocol": "previously_fixed_baseline",
            "seed": "fixed",
            "clips": len(rows),
        }
        result.update({metric: mean_rows(rows, metric) for metric in KEY_METRICS})
        aggregate_rows.append(result)

    engineered_rows = read_csv(WORK / f"evaluation/{ENGINEERED_METHOD}/by_clip.csv")
    engineered_ids = [row["clip_id"] for row in engineered_rows]
    if len(engineered_ids) != len(set(engineered_ids)) or set(engineered_ids) != test_ids:
        raise RuntimeError(f"engineered baseline has {len(engineered_rows)} test rows")
    engineered_summary = valid_summary(
        WORK / f"evaluation/{ENGINEERED_METHOD}/summary.csv"
    )
    engineered_result: dict[str, object] = {
        "method": ENGINEERED_METHOD,
        "protocol": "independent_development_search",
        "seed": "fixed",
        "clips": engineered_summary["clips"],
    }
    engineered_result.update(
        {metric: engineered_summary[metric] for metric in KEY_METRICS}
    )
    aggregate_rows.append(engineered_result)
    write_csv(WORK / "aggregate_results.csv", aggregate_rows)

    selected_seed = int(selection["selected_seed"]["full_model"])
    boot = bootstrap(
        by_clip[(selected_seed, "full_model")],
        engineered_rows,
    )
    write_csv(WORK / "source_cluster_bootstrap.csv", boot)

    selected = next(
        row for row in aggregate_rows
        if row["method"] == "full_model" and row["protocol"] == "development_selected_seed"
    )
    mean = next(
        row for row in aggregate_rows
        if row["method"] == "full_model" and row["protocol"] == "five_seed_mean"
    )
    engineered = next(row for row in aggregate_rows if row["method"] == "engineered_l2_instability")
    lines = [
        "# Bout-Disjoint Confirmation Results",
        "",
        "The manifest-only rule held out 17 confirmation-manifest source bouts",
        "absent from the complete reference and development manifests. All 166",
        "training clips from those sources were removed, leaving 746 training clips;",
        "the held-out sources are also absent from development. Evaluation uses 31",
        "usable adjudicated confirmation exchanges.",
        "",
        "## ReLIS",
        "",
        f"Development selected seed {selected_seed}. Its held-out metrics are "
        f"accuracy {float(selected['activity_accuracy']):.3f}, active F1 "
        f"{float(selected['active_f1']):.3f}, BF1@5 "
        f"{float(selected['boundary_f1_pm5']):.3f}, and BF1@10 "
        f"{float(selected['boundary_f1_pm10']):.3f}.",
        "",
        f"Across all five seeds, active F1 is {float(mean['active_f1']):.3f} "
        f"+/- {float(mean['active_f1_sd']):.3f} (range "
        f"{float(mean['active_f1_min']):.3f}--{float(mean['active_f1_max']):.3f}).",
        "",
        "## Independently tuned engineered comparator",
        "",
        f"Engineered instability obtains accuracy {float(engineered['activity_accuracy']):.3f}, "
        f"active F1 {float(engineered['active_f1']):.3f}, BF1@5 "
        f"{float(engineered['boundary_f1_pm5']):.3f}, and BF1@10 "
        f"{float(engineered['boundary_f1_pm10']):.3f}.",
        "",
        "The engineered score receives its own 2,000-candidate development-only extractor search.",
        "Every seed and the source-cluster bootstrap are provided in the adjacent CSV files.",
    ]
    (WORK / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    build_artifact_audit()
    print("\n".join(lines))


def build_artifact_audit() -> None:
    train_rows = read_csv(WORK / "manifests/train.csv")
    test_rows = read_csv(WORK / "manifests/test_valid.csv")
    retained_l2 = []
    for row in train_rows:
        path = Path(row["l2_path"])
        if not path.is_absolute():
            path = ROOT / path
        retained_l2.append(hash_record(path))
    test_annotations = [
        hash_record(CONFIRMATION_GOLD / f"{row['clip_id']}.json")
        for row in test_rows
    ]
    compact_outputs = [
        WORK / "split_report.json",
        WORK / "checkpoint_provenance.json",
        WORK / "model_selection.json",
        WORK / "prediction_provenance.json",
        WORK / "per_seed_results.csv",
        WORK / "aggregate_results.csv",
        WORK / "source_cluster_bootstrap.csv",
    ]
    payload = {
        "protocol_script": hash_record(Path(__file__)),
        "evaluation_script": hash_record(
            ROOT / "paper_experiments/scripts/compare_annotations_to_model.py"
        ),
        "prediction_script": hash_record(
            ROOT / "paper_experiments/scripts/generate_boundary_predictions.py"
        ),
        "search_script": hash_record(
            ROOT / "paper_experiments/scripts/search_boundary_postprocess.py"
        ),
        "base_config": hash_record(BASE_CONFIG),
        "main_baseline_input": hash_record(MAIN_PER_CLIP),
        "engineered_score_cache": hash_record(
            WORK / f"postprocess_search/score_cache/{ENGINEERED_METHOD}.npz"
        ),
        "engineered_score_cache_metadata": hash_record(
            WORK / f"postprocess_search/score_cache/{ENGINEERED_METHOD}.json"
        ),
        "engineered_search": hash_record(
            WORK / f"postprocess_search/{ENGINEERED_METHOD}-b_best.json"
        ),
        "retained_training_features": retained_l2,
        "development_annotations": development_annotation_records(),
        "test_annotations": test_annotations,
        "checkpoints": {
            str(seed): hash_record(checkpoint_path(seed)) for seed in SEEDS
        },
        "tuned_configs": {
            f"{seed}/{method}": hash_record(config_path(seed, method))
            for seed in SEEDS
            for method in ("full_model", "raw_phi_l2")
        },
        "predictions": {
            f"{seed}/{method}": hash_record(
                WORK / f"predictions/seed{seed}/{method}/{method}.json"
            )
            for seed in SEEDS
            for method in ("full_model", "raw_phi_l2")
        },
        "compact_outputs": [hash_record(path) for path in compact_outputs],
    }
    payload["tuned_configs"][ENGINEERED_METHOD] = hash_record(engineered_config_path())
    payload["predictions"][ENGINEERED_METHOD] = hash_record(engineered_prediction_path())
    write_json(WORK / "artifact_audit.json", payload)


def main() -> None:
    global OPENMMLAB_PY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(os.environ.get("BOUNDARY_PYTHON_BIN", OPENMMLAB_PY)),
        help="Python interpreter containing the model dependencies",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=["prepare", "train", "tune", "predict", "evaluate", "aggregate", "all"],
        default=[],
    )
    args = parser.parse_args()
    OPENMMLAB_PY = args.python_bin.resolve()
    stages = args.stage or ["all"]
    if "all" in stages:
        stages = ["prepare", "train", "tune", "predict", "evaluate", "aggregate"]
    actions = {
        "prepare": prepare,
        "train": train,
        "tune": tune,
        "predict": predict,
        "evaluate": evaluate,
        "aggregate": aggregate,
    }
    for stage in stages:
        actions[stage]()


if __name__ == "__main__":
    main()
