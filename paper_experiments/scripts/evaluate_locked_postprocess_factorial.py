#!/usr/bin/env python3
"""Evaluate frozen ReLIS postprocessing configurations on Round-2 cohorts.

This is an evaluation-only runner.  It has two deliberately separate stages:

``fixed-confirmation``
    Evaluate all 16 masks of the historical primary extractor on the reference
    cohort, but evaluate only the four prospectively locked masks (VCBG, VCB-,
    ----, and -C--) on the annotation-blind confirmation cohort.  Apply the
    prespecified confirmation gates without changing the historical primary.

``locked-candidates``
    Evaluate the historical primary and the development-selected performance A
    and parsimony B configurations stored in ``final_selection.json``.  Report
    every system on reference, confirmation, pooled, and clean cohorts; do not
    select among them using held-out results.

``multiseed-structural``
    Evaluate only historical VCBG and the V-B- structural simplification
    selected by the frozen five-seed development analysis.  The structural
    lock must predate both preceding held-out evaluation stages.

The script never trains a model, changes a configuration, or writes into an
existing result directory.  It requires cached raw boundary scores and verifies
the frozen selection/configuration hashes before loading held-out annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
L3_DIR = REPO / "boundary_segmentation"
for import_dir in (SCRIPT_DIR, L3_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from annotation_schema import normalize_annotation  # noqa: E402
from common import read_csv, read_json, write_csv, write_json  # noqa: E402
from compare_annotations_to_model import clip_metrics  # noqa: E402
from generate_boundary_predictions import partition_states  # noqa: E402
from search_boundary_postprocess import zones_from_score  # noqa: E402


DEFAULT_PROTOCOL = (
    REPO
    / "paper_experiments/round2/postprocess_equal_budget_search_20260814/PROTOCOL.md"
)
DEFAULT_EVALUATION_MANIFEST = (
    REPO
    / "paper_experiments/round2/annotation_extension/final_consensus_evaluation/"
    "evaluation_manifest.csv"
)
DEFAULT_STRUCTURAL_LOCK = (
    REPO
    / "paper_experiments/round2/postprocess_multiseed_factorial_20260814/"
    "selection.json"
)
EXPECTED_STRUCTURAL_LOCK_SHA256 = (
    "2a8f46c32c942f7c839158a27f1d20fca436f437cd9876c04dcfac525f158184"
)

COMPONENTS = ("V", "C", "B", "G")
ALL_MASKS = tuple(range(16))
FIXED_CONFIRMATION_MASKS = (15, 7, 0, 2)  # VCBG, VCB-, ----, -C--
BASELINE_MASK = 15
BOOTSTRAP_REPEATS = 10_000

EXPECTED_COUNTS = {
    "reference_records": 110,
    "reference_usable": 105,
    "reference_clean": 98,
    "confirmation_records": 124,
    "confirmation_usable": 118,
    "confirmation_clean": 115,
    "pooled_usable": 223,
    "pooled_clean": 213,
    "pooled_bouts": 82,
}

SELECTION_METRICS = (
    "activity_macro_f1",
    "active_f1",
    "boundary_f1_pm5",
    "boundary_f1_pm10",
)

CLIP_MEAN_METRICS = (
    "activity_accuracy",
    "activity_macro_f1",
    "active_precision",
    "active_recall",
    "active_f1",
    "boundary_precision_pm3",
    "boundary_recall_pm3",
    "boundary_f1_pm3",
    "mean_boundary_distance_pm3",
    "boundary_precision_pm5",
    "boundary_recall_pm5",
    "boundary_f1_pm5",
    "mean_boundary_distance_pm5",
    "boundary_precision_pm10",
    "boundary_recall_pm10",
    "boundary_f1_pm10",
    "mean_boundary_distance_pm10",
    "model_active_coverage",
    "human_active_coverage",
    "active_coverage_delta",
    "model_states",
    "human_states",
    "state_count_ratio",
    "objective",
    "balanced_objective",
)

# Human-only quantities have zero paired difference, so they are summarized but
# need not produce thousands of uninformative bootstrap zeros.
BOOTSTRAP_METRICS = tuple(
    metric for metric in CLIP_MEAN_METRICS if metric not in {"human_active_coverage", "human_states"}
) + (
    "model_human_state_ratio",
    "model_human_active_coverage_ratio",
)

COHORT_ORDER = (
    "reference",
    "confirmation",
    "pooled",
    "reference_clean",
    "confirmation_clean",
    "pooled_clean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "fixed-confirmation",
            "locked-candidates",
            "multiseed-structural",
        ),
        required=True,
        help="Run exactly one prespecified held-out evaluation stage.",
    )
    parser.add_argument(
        "--lock-json",
        type=Path,
        required=True,
        help="Development-only outputs/final_selection.json from the equal-budget search.",
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=DEFAULT_EVALUATION_MANIFEST,
        help="Frozen 234-record manifest with original/extension split labels.",
    )
    parser.add_argument(
        "--structural-lock-json",
        type=Path,
        default=DEFAULT_STRUCTURAL_LOCK,
        help="Pre-confirmation multiseed selection lock; used only by multiseed-structural.",
    )
    parser.add_argument("--reference-annotation-dir", type=Path, required=True)
    parser.add_argument("--confirmation-annotation-dir", type=Path, required=True)
    parser.add_argument("--reference-score-cache", type=Path, required=True)
    parser.add_argument("--confirmation-score-cache", type=Path, required=True)
    parser.add_argument(
        "--reference-score-cache-meta",
        type=Path,
        default=None,
        help="Defaults to the reference NPZ path with a .json suffix.",
    )
    parser.add_argument(
        "--confirmation-score-cache-meta",
        type=Path,
        default=None,
        help="Defaults to the confirmation NPZ path with a .json suffix.",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=BOOTSTRAP_REPEATS,
        help="The frozen evaluation requires exactly 10,000 resamples.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Stage outputs are written to a new child directory under this path.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_files_hash(paths: Iterable[Path]) -> str:
    """Hash names and contents without making the digest path-location dependent."""

    digest = hashlib.sha256()
    resolved = sorted((path.resolve() for path in paths), key=lambda item: item.name)
    for path in resolved:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def mask_label(mask: int) -> str:
    return "".join(
        letter if mask & (1 << index) else "-"
        for index, letter in enumerate(COMPONENTS)
    )


def rule_count(mask: int) -> int:
    return int(mask).bit_count()


def bout_id(clip_id: str) -> str:
    fields = clip_id.split("-")
    if len(fields) != 3 or not fields[1]:
        raise ValueError(
            f"clip ID does not follow PREFIX-BOUT-EXCHANGE with a middle bout field: {clip_id}"
        )
    return fields[1]


def apply_mask(base: Mapping[str, Any], mask: int) -> dict[str, Any]:
    """Apply the exact V/C/B/G neutralizations used by the development runner."""

    if mask not in ALL_MASKS:
        raise ValueError(f"mask must be in [0, 15], got {mask}")
    result = dict(base)
    if not mask & 1:  # V: internal-valley split
        result.update(
            split_min_zone_len=0,
            split_min_rel_drop=0.0,
            split_max_splits=0,
        )
    if not mask & 2:  # C: active-coverage clamp
        result.update(
            boundary_frac_min=0.0,
            boundary_frac_max=1.0,
            max_budget_iters=0,
        )
    if not mask & 4:  # B: onset backtracking
        result["onset_backtrack"] = 0
    if not mask & 8:  # G: short-gap merge
        result["merge_gap"] = 0
    return result


def validate_masked_config(config: Mapping[str, Any], mask: int, label: str) -> None:
    if str(config.get("mode")) != "robust_hysteresis":
        raise ValueError(f"{label}: locked mask family must use robust_hysteresis")
    if not mask & 1 and any(
        (
            int(config.get("split_min_zone_len", 0)) != 0,
            float(config.get("split_min_rel_drop", 0.0)) != 0.0,
            int(config.get("split_max_splits", 0)) != 0,
        )
    ):
        raise ValueError(f"{label}: valley splitting is not neutralized")
    if not mask & 2 and any(
        (
            float(config.get("boundary_frac_min", 0.0)) != 0.0,
            float(config.get("boundary_frac_max", 1.0)) != 1.0,
            int(config.get("max_budget_iters", 0)) != 0,
        )
    ):
        raise ValueError(f"{label}: coverage clamp is not neutralized")
    if not mask & 4 and int(config.get("onset_backtrack", 0)) != 0:
        raise ValueError(f"{label}: onset backtracking is not neutralized")
    if not mask & 8 and int(config.get("merge_gap", 0)) != 0:
        raise ValueError(f"{label}: gap merging is not neutralized")


def validate_selection_lock(path: Path, require_candidates: bool) -> dict[str, Any]:
    """Validate hashes before any held-out annotation is read."""

    selection = read_json(path)
    if not isinstance(selection, dict):
        raise ValueError("selection lock must contain a JSON object")
    if selection.get("development_only") is not True:
        raise ValueError("selection lock is not marked development_only=true")
    if selection.get("heldout_metrics_read_by_this_runner") is not False:
        raise ValueError(
            "selection lock must state heldout_metrics_read_by_this_runner=false"
        )
    recorded_hash = str(selection.get("selection_hash", ""))
    unhashed = dict(selection)
    unhashed.pop("selection_hash", None)
    actual_hash = canonical_hash(unhashed)
    if not recorded_hash or recorded_hash != actual_hash:
        raise ValueError(
            f"selection lock hash mismatch: recorded={recorded_hash!r}, actual={actual_hash}"
        )

    historical = selection.get("historical_primary_diagnostic")
    if not isinstance(historical, dict) or not isinstance(
        historical.get("postprocess"), dict
    ):
        raise ValueError("selection lock lacks historical_primary_diagnostic.postprocess")
    historical_hash = canonical_hash(historical["postprocess"])
    if historical_hash != str(historical.get("postprocess_hash", "")):
        raise ValueError("historical primary postprocess hash mismatch")
    validate_masked_config(historical["postprocess"], BASELINE_MASK, "historical_primary")

    permitted = set(selection.get("heldout_evaluation_permitted_for", []))
    if "historical_primary" not in permitted:
        raise ValueError("selection lock does not permit historical_primary evaluation")

    if require_candidates:
        required = {"performance_A", "parsimony_B"}
        if not required <= permitted:
            raise ValueError(
                f"selection lock does not permit candidate evaluation: {sorted(required - permitted)}"
            )
        frozen = selection.get("frozen_candidates")
        if not isinstance(frozen, dict) or not required <= set(frozen):
            raise ValueError("selection lock lacks frozen performance_A/parsimony_B")
        for name in sorted(required):
            record = frozen[name]
            if not isinstance(record, dict) or not isinstance(
                record.get("postprocess"), dict
            ):
                raise ValueError(f"{name}: missing postprocess object")
            if canonical_hash(record["postprocess"]) != str(
                record.get("postprocess_hash", "")
            ):
                raise ValueError(f"{name}: postprocess hash mismatch")
            mask = int(record["mask_integer"])
            if record.get("family") != mask_label(mask):
                raise ValueError(f"{name}: family does not match mask_integer")
            if int(record.get("rules", -1)) != rule_count(mask):
                raise ValueError(f"{name}: rules count does not match mask_integer")
            validate_masked_config(record["postprocess"], mask, name)
    return selection


def timestamp_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def validate_structural_lock(
    path: Path,
    historical_postprocess: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    actual_file_hash = sha256_file(path)
    if actual_file_hash != EXPECTED_STRUCTURAL_LOCK_SHA256:
        raise ValueError(
            "structural selection lock SHA-256 mismatch: "
            f"expected={EXPECTED_STRUCTURAL_LOCK_SHA256}, actual={actual_file_hash}"
        )
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("development_only") is not True:
        raise ValueError("structural lock must be a development_only JSON object")
    for field, expected in (
        ("expected_mask", "V-B-"),
        ("selected_mask", "V-B-"),
        ("full_mask", "VCBG"),
    ):
        if payload.get(field) != expected:
            raise ValueError(
                f"structural lock {field} mismatch: expected {expected}, "
                f"observed {payload.get(field)!r}"
            )
    selected = payload.get("selected_postprocess")
    expected_selected = apply_mask(historical_postprocess, 5)
    if not isinstance(selected, dict) or selected != expected_selected:
        raise ValueError(
            "structural lock selected_postprocess is not exactly "
            "apply_mask(historical_primary, 5)"
        )
    validate_masked_config(selected, 5, "multiseed_structural_V-B-")
    matching = [
        row
        for row in payload.get("eligibility", [])
        if isinstance(row, dict) and row.get("mask") == "V-B-"
    ]
    if len(matching) != 1 or matching[0].get("eligible") is not True:
        raise ValueError("structural lock does not uniquely mark V-B- as eligible")
    stat = path.stat()
    metadata = {
        "path": str(path),
        "sha256": actual_file_hash,
        "mtime_ns": stat.st_mtime_ns,
        "mtime_utc": timestamp_utc(stat.st_mtime_ns),
        "selected_mask": "V-B-",
        "selected_mask_integer": 5,
        "selected_postprocess_hash": canonical_hash(selected),
    }
    return payload, metadata


def validate_structural_lock_predates_prior_outputs(
    out_base: Path,
    structural_lock_mtime_ns: int,
) -> dict[str, dict[str, Any]]:
    """Use filesystem metadata only; do not inspect preceding result contents."""

    output: dict[str, dict[str, Any]] = {}
    for stage in ("locked_candidates", "fixed_confirmation"):
        stage_dir = out_base.resolve() / stage
        files = sorted(path for path in stage_dir.rglob("*") if path.is_file())
        if not files:
            raise FileNotFoundError(
                f"cannot establish structural-lock ordering: no files under {stage_dir}"
            )
        mtimes = [path.stat().st_mtime_ns for path in files]
        earliest = min(mtimes)
        latest = max(mtimes)
        if structural_lock_mtime_ns >= earliest:
            raise ValueError(
                "structural selection lock does not predate the earliest "
                f"{stage} output"
            )
        output[stage] = {
            "directory": str(stage_dir),
            "files": len(files),
            "earliest_output_mtime_ns": earliest,
            "earliest_output_mtime_utc": timestamp_utc(earliest),
            "latest_output_mtime_ns": latest,
            "latest_output_mtime_utc": timestamp_utc(latest),
            "structural_lock_predates_earliest_output": True,
        }
    return output


def is_usable(annotation: Mapping[str, Any]) -> bool:
    flags = annotation.get("quality_flags", {})
    return bool(annotation.get("annotation_complete")) and bool(
        flags.get("valid", True)
    ) and not bool(flags.get("not_exchange", False))


def is_clean(annotation: Mapping[str, Any]) -> bool:
    if not is_usable(annotation):
        return False
    return not any(
        bool(value)
        for key, value in annotation.get("quality_flags", {}).items()
        if key != "valid"
    )


def load_annotation_cohort(
    directory: Path, cohort: str, allowed_ids: set[str]
) -> tuple[dict[str, dict], set[str], list[Path]]:
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"{cohort}: no JSON annotations in {directory}")
    usable: dict[str, dict] = {}
    clean_ids: set[str] = set()
    seen: set[str] = set()
    selected_paths: list[Path] = []
    for path in paths:
        annotation = normalize_annotation(read_json(path), strict=False)
        clip_id = str(annotation["clip_id"])
        if clip_id in seen:
            raise ValueError(f"{cohort}: duplicate annotation clip ID {clip_id}")
        seen.add(clip_id)
        if clip_id not in allowed_ids:
            continue
        selected_paths.append(path)
        if is_usable(annotation):
            if not annotation.get("segments"):
                raise ValueError(f"{cohort}: usable annotation has no segments: {clip_id}")
            usable[clip_id] = annotation
            if is_clean(annotation):
                clean_ids.add(clip_id)
    missing = allowed_ids - seen
    if missing:
        raise ValueError(f"{cohort}: manifest IDs lack annotations: {sorted(missing)}")
    return usable, clean_ids, selected_paths


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_evaluation_manifest(
    path: Path,
) -> tuple[set[str], set[str], list[dict[str, str]]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError("evaluation manifest is empty")
    keys = {(str(row.get("split")), str(row.get("clip_id"))) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("evaluation manifest contains duplicate split/clip rows")
    unexpected_splits = {str(row.get("split")) for row in rows} - {
        "original",
        "extension",
    }
    if unexpected_splits:
        raise ValueError(f"unexpected evaluation-manifest splits: {unexpected_splits}")
    reference_ids = {
        str(row["clip_id"]) for row in rows if row["split"] == "original"
    }
    confirmation_ids = {
        str(row["clip_id"]) for row in rows if row["split"] == "extension"
    }
    if reference_ids & confirmation_ids:
        raise ValueError("evaluation manifest assigns clip IDs to both splits")
    if len(reference_ids) != EXPECTED_COUNTS["reference_records"]:
        raise ValueError("evaluation manifest does not contain 110 original records")
    if len(confirmation_ids) != EXPECTED_COUNTS["confirmation_records"]:
        raise ValueError("evaluation manifest does not contain 124 extension records")
    for row in rows:
        clip_id = str(row["clip_id"])
        if str(row.get("source_id", "")) != bout_id(clip_id):
            raise ValueError(f"manifest source_id does not match middle clip field: {clip_id}")
    return reference_ids, confirmation_ids, rows


def validate_manifest_labels(
    rows: Sequence[Mapping[str, str]],
    reference: Mapping[str, Any],
    reference_clean: set[str],
    confirmation: Mapping[str, Any],
    confirmation_clean: set[str],
) -> None:
    usable_by_split = {
        "original": set(reference),
        "extension": set(confirmation),
    }
    clean_by_split = {
        "original": reference_clean,
        "extension": confirmation_clean,
    }
    for row in rows:
        split = str(row["split"])
        clip_id = str(row["clip_id"])
        if truthy(row.get("usable")) != (clip_id in usable_by_split[split]):
            raise ValueError(f"manifest/annotation usable mismatch: {split}/{clip_id}")
        if truthy(row.get("clean")) != (clip_id in clean_by_split[split]):
            raise ValueError(f"manifest/annotation clean mismatch: {split}/{clip_id}")


def validate_cohort_counts(
    reference: Mapping[str, Any],
    reference_clean: set[str],
    reference_files: Sequence[Path],
    confirmation: Mapping[str, Any],
    confirmation_clean: set[str],
    confirmation_files: Sequence[Path],
) -> None:
    observed = {
        "reference_records": len(reference_files),
        "reference_usable": len(reference),
        "reference_clean": len(reference_clean),
        "confirmation_records": len(confirmation_files),
        "confirmation_usable": len(confirmation),
        "confirmation_clean": len(confirmation_clean),
        "pooled_usable": len(reference) + len(confirmation),
        "pooled_clean": len(reference_clean) + len(confirmation_clean),
    }
    for name, expected in EXPECTED_COUNTS.items():
        if name == "pooled_bouts":
            continue
        if observed[name] != expected:
            raise ValueError(
                f"frozen cohort count mismatch for {name}: expected {expected}, "
                f"observed {observed[name]}"
            )
    overlap = set(reference) & set(confirmation)
    if overlap:
        raise ValueError(f"reference/confirmation clip overlap: {sorted(overlap)}")
    pooled_bouts = {bout_id(clip_id) for clip_id in set(reference) | set(confirmation)}
    if len(pooled_bouts) != EXPECTED_COUNTS["pooled_bouts"]:
        raise ValueError(
            f"expected {EXPECTED_COUNTS['pooled_bouts']} pooled middle-field bouts, "
            f"observed {len(pooled_bouts)}"
        )


def load_score_cache(
    cache_path: Path,
    metadata_path: Path,
    cohort_ids: set[str],
    required_ids: Iterable[str],
    cohort: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    metadata = read_json(metadata_path)
    cache_ids = [str(value) for value in metadata.get("clips", [])]
    if len(cache_ids) != len(set(cache_ids)):
        raise ValueError(f"{cohort}: score-cache metadata contains duplicate clip IDs")
    if set(cache_ids) != cohort_ids:
        raise ValueError(
            f"{cohort}: cache IDs do not exactly match the frozen manifest split; "
            f"cache_only={sorted(set(cache_ids) - cohort_ids)}, "
            f"manifest_only={sorted(cohort_ids - set(cache_ids))}"
        )
    missing_meta = set(required_ids) - set(cache_ids)
    if missing_meta:
        raise ValueError(
            f"{cohort}: score-cache metadata misses usable clips: {sorted(missing_meta)}"
        )

    output: dict[str, dict[str, np.ndarray]] = {}
    with np.load(cache_path, allow_pickle=False) as archive:
        for clip_id in sorted(required_ids):
            keys = {
                name: f"{name}::{clip_id}" for name in ("b", "mask", "frames")
            }
            absent = [key for key in keys.values() if key not in archive]
            if absent:
                raise ValueError(f"{cohort}: cache misses arrays {absent}")
            score = np.asarray(archive[keys["b"]], dtype=np.float32).copy()
            mask = np.asarray(archive[keys["mask"]], dtype=np.bool_).copy()
            frames = np.asarray(archive[keys["frames"]], dtype=np.int32).copy()
            if score.ndim != 1 or mask.ndim != 1 or frames.ndim != 1:
                raise ValueError(f"{cohort}/{clip_id}: cache arrays must be one-dimensional")
            if not (len(score) == len(mask) == len(frames)):
                raise ValueError(f"{cohort}/{clip_id}: cache array lengths differ")
            if not np.all(np.isfinite(score)):
                raise ValueError(f"{cohort}/{clip_id}: nonfinite cached boundary score")
            output[clip_id] = {"b": score, "mask": mask, "frames": frames}
    return output, metadata


def validate_cache_pair(
    reference_meta: Mapping[str, Any],
    confirmation_meta: Mapping[str, Any],
) -> dict[str, str]:
    """Prove that both cohorts use raw scores from the same frozen model."""

    for field in (
        "extension_mode",
        "extension_length",
        "zero_feature_indices",
    ):
        if reference_meta.get(field) != confirmation_meta.get(field):
            raise ValueError(f"reference/confirmation cache metadata differ for {field}")

    source_hashes: dict[str, str] = {}
    source_paths: dict[str, tuple[Path, Path]] = {}
    for field in ("config", "checkpoint"):
        reference_value = reference_meta.get(field)
        confirmation_value = confirmation_meta.get(field)
        if not reference_value or not confirmation_value:
            raise ValueError(f"both score-cache metadata files must record {field}")
        reference_path = Path(str(reference_value)).resolve()
        confirmation_path = Path(str(confirmation_value)).resolve()
        if not reference_path.exists() or not confirmation_path.exists():
            raise FileNotFoundError(
                f"cannot verify cached model {field}: "
                f"{reference_path}, {confirmation_path}"
            )
        source_paths[field] = (reference_path, confirmation_path)
        reference_hash = sha256_file(reference_path)
        confirmation_hash = sha256_file(confirmation_path)
        if field == "checkpoint" and reference_hash != confirmation_hash:
            raise ValueError(
                "reference/confirmation caches were produced from different checkpoints"
            )
        source_hashes[f"reference_score_source_{field}_sha256"] = reference_hash
        source_hashes[f"confirmation_score_source_{field}_sha256"] = confirmation_hash

    reference_config = yaml.safe_load(
        source_paths["config"][0].read_text(encoding="utf-8")
    )
    confirmation_config = yaml.safe_load(
        source_paths["config"][1].read_text(encoding="utf-8")
    )
    if not isinstance(reference_config, dict) or not isinstance(
        confirmation_config, dict
    ):
        raise ValueError("cached model configs must contain JSON-like objects")
    # Candidate config files intentionally carry different extraction settings.
    # Raw b scores are nevertheless comparable when all model/data/training
    # fields and the checkpoint match, so exclude postprocess alone.
    reference_core = dict(reference_config)
    confirmation_core = dict(confirmation_config)
    reference_core.pop("postprocess", None)
    confirmation_core.pop("postprocess", None)
    if canonical_hash(reference_core) != canonical_hash(confirmation_core):
        raise ValueError(
            "reference/confirmation cache configs differ outside postprocess"
        )
    source_hashes["score_source_config_without_postprocess_hash"] = canonical_hash(
        reference_core
    )
    return source_hashes


def evaluate_clip(
    method: str,
    config_hash: str,
    mask: int | str,
    origin: str,
    clip_id: str,
    annotation: dict,
    entry: Mapping[str, np.ndarray],
    postprocess: Mapping[str, Any],
) -> dict[str, Any]:
    score = entry["b"]
    valid_mask = entry["mask"]
    frames = entry["frames"]
    zones = zones_from_score(score, valid_mask, dict(postprocess))
    states = partition_states(
        {"frames": frames.tolist(), "l2_mask": valid_mask.tolist()},
        zones,
        len(score),
    )
    prediction = [
        (int(row["start_frame"]), int(row["end_frame"]), str(row["activity"]))
        for row in states
    ]
    row = clip_metrics(clip_id, annotation, prediction)
    row.update(
        {
            "method": method,
            "postprocess_hash": config_hash,
            "mask": mask_label(mask) if isinstance(mask, int) else mask,
            "mask_integer": mask,
            "rules": rule_count(mask) if isinstance(mask, int) else "",
            "origin": origin,
            "bout_id": bout_id(clip_id),
        }
    )
    row["active_coverage_delta"] = float(row["model_active_coverage"]) - float(
        row["human_active_coverage"]
    )
    human_coverage = float(row["human_active_coverage"])
    row["model_human_active_coverage_ratio"] = (
        float(row["model_active_coverage"]) / human_coverage
        if human_coverage > 0.0
        else math.nan
    )
    row["objective"] = statistics.mean(float(row[name]) for name in SELECTION_METRICS)
    row["balanced_objective"] = statistics.mean(
        (float(row["activity_macro_f1"]), float(row["boundary_f1_pm5"]))
    )
    return row


def evaluate_configs(
    configs: Mapping[str, dict[str, Any]],
    annotations: Mapping[str, dict],
    clean_ids: set[str],
    cache: Mapping[str, Mapping[str, np.ndarray]],
    origin: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, record in configs.items():
        postprocess = record["postprocess"]
        config_hash = str(record["postprocess_hash"])
        mask: int | str = record.get("mask_integer", "locked")
        for clip_id in sorted(annotations):
            row = evaluate_clip(
                method,
                config_hash,
                mask,
                origin,
                clip_id,
                annotations[clip_id],
                cache[clip_id],
                postprocess,
            )
            row["clean"] = clip_id in clean_ids
            rows.append(row)
    return rows


def cohort_subset(rows: Sequence[dict[str, Any]], cohort: str) -> list[dict[str, Any]]:
    if cohort == "reference":
        return [row for row in rows if row["origin"] == "reference"]
    if cohort == "confirmation":
        return [row for row in rows if row["origin"] == "confirmation"]
    if cohort == "pooled":
        return list(rows)
    if cohort == "reference_clean":
        return [
            row
            for row in rows
            if row["origin"] == "reference" and bool(row["clean"])
        ]
    if cohort == "confirmation_clean":
        return [
            row
            for row in rows
            if row["origin"] == "confirmation" and bool(row["clean"])
        ]
    if cohort == "pooled_clean":
        return [row for row in rows if bool(row["clean"])]
    raise ValueError(f"unknown cohort {cohort}")


def point_metric(rows: Sequence[dict[str, Any]], metric: str) -> float:
    if not rows:
        raise ValueError(f"cannot summarize empty rows for {metric}")
    if metric == "model_human_state_ratio":
        return sum(float(row["model_states"]) for row in rows) / max(
            1.0, sum(float(row["human_states"]) for row in rows)
        )
    if metric == "model_human_active_coverage_ratio":
        human = sum(float(row["human_active_coverage"]) for row in rows)
        return (
            sum(float(row["model_active_coverage"]) for row in rows) / human
            if human > 0.0
            else math.nan
        )
    return statistics.mean(float(row[metric]) for row in rows)


def summary_row(
    stage: str, method: str, cohort: str, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage": stage,
        "method": method,
        "cohort": cohort,
        "clips": len(rows),
        "bout_clusters": len({str(row["bout_id"]) for row in rows}),
        "postprocess_hash": rows[0]["postprocess_hash"],
        "mask": rows[0]["mask"],
        "mask_integer": rows[0]["mask_integer"],
        "rules": rows[0]["rules"],
        "total_human_states": sum(int(row["human_states"]) for row in rows),
        "total_model_states": sum(int(row["model_states"]) for row in rows),
    }
    for metric in CLIP_MEAN_METRICS:
        result[metric] = point_metric(rows, metric)
    result["model_human_state_ratio"] = point_metric(
        rows, "model_human_state_ratio"
    )
    result["model_human_active_coverage_ratio"] = point_metric(
        rows, "model_human_active_coverage_ratio"
    )
    return result


def summarize_methods(
    stage: str,
    rows: Sequence[dict[str, Any]],
    cohorts: Sequence[str] = COHORT_ORDER,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    for cohort in cohorts:
        selected = cohort_subset(rows, cohort)
        for method in methods:
            current = [row for row in selected if row["method"] == method]
            if current:
                output.append(summary_row(stage, method, cohort, current))
    return output


def draw_counts(bouts: int, repeats: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = np.full(bouts, 1.0 / bouts, dtype=np.float64)
    return rng.multinomial(bouts, probabilities, size=repeats).astype(np.float64)


def grouped_sums(
    rows: Mapping[str, dict[str, Any]],
    ordered_bouts: Sequence[str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.zeros(len(ordered_bouts), dtype=np.float64)
    denominator = np.zeros(len(ordered_bouts), dtype=np.float64)
    bout_index = {bout: index for index, bout in enumerate(ordered_bouts)}
    for row in rows.values():
        index = bout_index[str(row["bout_id"])]
        if metric == "model_human_state_ratio":
            numerator[index] += float(row["model_states"])
            denominator[index] += float(row["human_states"])
        elif metric == "model_human_active_coverage_ratio":
            numerator[index] += float(row["model_active_coverage"])
            denominator[index] += float(row["human_active_coverage"])
        else:
            numerator[index] += float(row[metric])
            denominator[index] += 1.0
    return numerator, denominator


def paired_bootstrap(
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    metric: str,
    counts: np.ndarray,
    ordered_bouts: Sequence[str],
) -> tuple[float, np.ndarray]:
    baseline = {str(row["clip_id"]): row for row in baseline_rows}
    candidate = {str(row["clip_id"]): row for row in candidate_rows}
    if set(baseline) != set(candidate):
        raise ValueError("paired bootstrap method clip sets differ")
    baseline_num, baseline_den = grouped_sums(baseline, ordered_bouts, metric)
    candidate_num, candidate_den = grouped_sums(candidate, ordered_bouts, metric)
    baseline_sample_den = counts @ baseline_den
    candidate_sample_den = counts @ candidate_den
    if np.any(baseline_sample_den <= 0.0) or np.any(candidate_sample_den <= 0.0):
        raise ValueError(f"bootstrap denominator is nonpositive for {metric}")
    samples = (counts @ candidate_num) / candidate_sample_den - (
        counts @ baseline_num
    ) / baseline_sample_den
    estimate = point_metric(candidate_rows, metric) - point_metric(
        baseline_rows, metric
    )
    return estimate, samples


def bootstrap_methods(
    stage: str,
    rows: Sequence[dict[str, Any]],
    baseline_method: str,
    bootstrap_seed: int,
    bootstrap_repeats: int,
    cohorts: Sequence[str] = COHORT_ORDER,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    candidates = [method for method in methods if method != baseline_method]
    for cohort in cohorts:
        selected = cohort_subset(rows, cohort)
        baseline = [row for row in selected if row["method"] == baseline_method]
        if not baseline:
            continue
        ordered_bouts = sorted({str(row["bout_id"]) for row in baseline})
        cohort_seed = bootstrap_seed + COHORT_ORDER.index(cohort)
        counts = draw_counts(
            len(ordered_bouts), bootstrap_repeats, cohort_seed
        )
        for candidate_method in candidates:
            candidate = [
                row for row in selected if row["method"] == candidate_method
            ]
            if not candidate:
                continue
            for metric in BOOTSTRAP_METRICS:
                estimate, samples = paired_bootstrap(
                    baseline, candidate, metric, counts, ordered_bouts
                )
                output.append(
                    {
                        "stage": stage,
                        "cohort": cohort,
                        "baseline": baseline_method,
                        "candidate": candidate_method,
                        "metric": metric,
                        "estimate_candidate_minus_baseline": estimate,
                        "ci95_low": float(np.quantile(samples, 0.025)),
                        "ci95_high": float(np.quantile(samples, 0.975)),
                        "bootstrap_probability_gt_zero": float(
                            np.mean(samples > 0.0)
                        ),
                        "clips": len(baseline),
                        "bout_clusters": len(ordered_bouts),
                        "bootstrap_repeats": bootstrap_repeats,
                        "bootstrap_seed": cohort_seed,
                    }
                )
    return output


def fixed_configurations(
    historical: Mapping[str, Any], masks: Iterable[int]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for mask in masks:
        config = apply_mask(historical, mask)
        label = mask_label(mask)
        validate_masked_config(config, mask, f"historical_fixed_{label}")
        output[label] = {
            "mask_integer": mask,
            "postprocess": config,
            "postprocess_hash": canonical_hash(config),
        }
    return output


def confirmation_gate_outputs(
    summaries: Sequence[dict[str, Any]],
    bootstrap_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    confirmation = {
        str(row["method"]): row
        for row in summaries
        if row["cohort"] == "confirmation"
    }
    bootstrap = {
        (str(row["candidate"]), str(row["metric"])): row
        for row in bootstrap_rows
        if row["cohort"] == "confirmation"
    }
    baseline = confirmation["VCBG"]
    gate_rows: list[dict[str, Any]] = []
    for mask in FIXED_CONFIRMATION_MASKS:
        method = mask_label(mask)
        if method == "VCBG":
            continue
        current = confirmation[method]
        active_boot = bootstrap[(method, "active_f1")]
        bf10_boot = bootstrap[(method, "boundary_f1_pm10")]
        delta_active = float(current["active_f1"]) - float(baseline["active_f1"])
        delta_bf10 = float(current["boundary_f1_pm10"]) - float(
            baseline["boundary_f1_pm10"]
        )
        gates = {
            "active_f1_delta_ge_minus_0p010": delta_active >= -0.010,
            "boundary_f1_pm10_delta_ge_minus_0p010": delta_bf10 >= -0.010,
            "active_f1_ci_low_gt_minus_0p020": float(active_boot["ci95_low"])
            > -0.020,
            "boundary_f1_pm10_ci_low_gt_minus_0p020": float(
                bf10_boot["ci95_low"]
            )
            > -0.020,
        }
        gate_rows.append(
            {
                "method": method,
                "mask_integer": mask,
                "rules": rule_count(mask),
                "confirmation_active_f1": current["active_f1"],
                "delta_active_f1": delta_active,
                "active_f1_ci95_low": active_boot["ci95_low"],
                "active_f1_ci95_high": active_boot["ci95_high"],
                "confirmation_boundary_f1_pm5": current["boundary_f1_pm5"],
                "confirmation_boundary_f1_pm10": current["boundary_f1_pm10"],
                "delta_boundary_f1_pm10": delta_bf10,
                "boundary_f1_pm10_ci95_low": bf10_boot["ci95_low"],
                "boundary_f1_pm10_ci95_high": bf10_boot["ci95_high"],
                **gates,
                "eligible": all(gates.values()),
            }
        )

    eligible = [row for row in gate_rows if bool(row["eligible"])]
    preference: str | None = None
    tolerance_pool: list[dict[str, Any]] = []
    if eligible:
        best_bf5 = max(float(row["confirmation_boundary_f1_pm5"]) for row in eligible)
        tolerance_pool = [
            row
            for row in eligible
            if float(row["confirmation_boundary_f1_pm5"]) >= best_bf5 - 0.005
        ]
        chosen = min(
            tolerance_pool,
            key=lambda row: (
                int(row["rules"]),
                -float(row["confirmation_boundary_f1_pm5"]),
                int(row["mask_integer"]),
            ),
        )
        preference = str(chosen["method"])
    decision = {
        "scope": "prespecified one-shot 118-exchange confirmation diagnostic",
        "baseline": "VCBG",
        "eligible_nonbaseline_masks": [
            str(row["method"]) for row in gate_rows if bool(row["eligible"])
        ],
        "bf1_pm5_tolerance_pool": [str(row["method"]) for row in tolerance_pool],
        "diagnostic_preference": preference,
        "historical_primary_remains_primary": True,
        "may_replace_or_redefine_primary": False,
        "gates": {
            "active_f1_delta_min": -0.010,
            "boundary_f1_pm10_delta_min": -0.010,
            "active_f1_bootstrap_ci95_low_strict_min": -0.020,
            "boundary_f1_pm10_bootstrap_ci95_low_strict_min": -0.020,
            "bf1_pm5_near_best_tolerance": 0.005,
        },
    }
    return gate_rows, decision


def ensure_new_stage_dir(base: Path, stage: str) -> Path:
    stage_dir = base.resolve() / stage.replace("-", "_")
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty evaluation directory: {stage_dir}"
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def write_stage_outputs(
    stage_dir: Path,
    outputs: Mapping[str, Sequence[dict[str, Any]] | Mapping[str, Any]],
) -> dict[str, str]:
    paths: list[Path] = []
    for name, payload in outputs.items():
        path = stage_dir / name
        if name.endswith(".csv"):
            if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
                raise TypeError(f"CSV payload must be a sequence: {name}")
            rows = list(payload)
            if not rows:
                raise ValueError(f"refusing to write empty output: {name}")
            write_csv(path, rows)
        elif name.endswith(".json"):
            write_json(path, payload)
        else:
            raise ValueError(f"unsupported output suffix: {name}")
        paths.append(path)
    return {path.name: sha256_file(path) for path in paths}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.bootstrap_repeats != BOOTSTRAP_REPEATS:
        raise ValueError(
            f"frozen evaluation requires exactly {BOOTSTRAP_REPEATS} bootstrap repeats"
        )

    lock_path = args.lock_json.resolve()
    protocol_path = args.protocol.resolve()
    evaluation_manifest_path = args.evaluation_manifest.resolve()
    structural_lock_path = args.structural_lock_json.resolve()
    reference_annotation_dir = args.reference_annotation_dir.resolve()
    confirmation_annotation_dir = args.confirmation_annotation_dir.resolve()
    reference_cache_path = args.reference_score_cache.resolve()
    confirmation_cache_path = args.confirmation_score_cache.resolve()
    reference_meta_path = (
        args.reference_score_cache_meta.resolve()
        if args.reference_score_cache_meta
        else reference_cache_path.with_suffix(".json")
    )
    confirmation_meta_path = (
        args.confirmation_score_cache_meta.resolve()
        if args.confirmation_score_cache_meta
        else confirmation_cache_path.with_suffix(".json")
    )
    required_paths = [
        lock_path,
        protocol_path,
        evaluation_manifest_path,
        reference_annotation_dir,
        confirmation_annotation_dir,
        reference_cache_path,
        confirmation_cache_path,
        reference_meta_path,
        confirmation_meta_path,
    ]
    if args.stage == "multiseed-structural":
        required_paths.append(structural_lock_path)
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen evaluation inputs: {missing}")

    # Lock and destination checks happen before any held-out annotation is read.
    selection = validate_selection_lock(
        lock_path, require_candidates=args.stage == "locked-candidates"
    )
    historical_record = selection["historical_primary_diagnostic"]
    historical = historical_record["postprocess"]
    structural_lock_payload: dict[str, Any] | None = None
    structural_lock_metadata: dict[str, Any] | None = None
    prior_evaluation_timestamps: dict[str, dict[str, Any]] | None = None
    if args.stage == "multiseed-structural":
        structural_lock_payload, structural_lock_metadata = validate_structural_lock(
            structural_lock_path, historical
        )
        prior_evaluation_timestamps = (
            validate_structural_lock_predates_prior_outputs(
                args.out_dir.resolve(), int(structural_lock_metadata["mtime_ns"])
            )
        )
    stage_dir = ensure_new_stage_dir(args.out_dir, args.stage)

    reference_ids, confirmation_ids, evaluation_manifest_rows = (
        load_evaluation_manifest(evaluation_manifest_path)
    )
    reference, reference_clean, reference_files = load_annotation_cohort(
        reference_annotation_dir, "reference", reference_ids
    )
    confirmation, confirmation_clean, confirmation_files = load_annotation_cohort(
        confirmation_annotation_dir, "confirmation", confirmation_ids
    )
    validate_cohort_counts(
        reference,
        reference_clean,
        reference_files,
        confirmation,
        confirmation_clean,
        confirmation_files,
    )
    validate_manifest_labels(
        evaluation_manifest_rows,
        reference,
        reference_clean,
        confirmation,
        confirmation_clean,
    )
    reference_cache, reference_cache_meta = load_score_cache(
        reference_cache_path,
        reference_meta_path,
        reference_ids,
        reference,
        "reference",
    )
    confirmation_cache, confirmation_cache_meta = load_score_cache(
        confirmation_cache_path,
        confirmation_meta_path,
        confirmation_ids,
        confirmation,
        "confirmation",
    )

    score_source_hashes = validate_cache_pair(
        reference_cache_meta,
        confirmation_cache_meta,
    )
    outputs: dict[str, Sequence[dict[str, Any]] | Mapping[str, Any]] = {}
    evaluated_configs: dict[str, Any]

    if args.stage == "fixed-confirmation":
        reference_configs = fixed_configurations(historical, ALL_MASKS)
        confirmation_configs = {
            label: reference_configs[label]
            for label in (mask_label(mask) for mask in FIXED_CONFIRMATION_MASKS)
        }
        reference_rows = evaluate_configs(
            reference_configs,
            reference,
            reference_clean,
            reference_cache,
            "reference",
        )
        confirmation_rows = evaluate_configs(
            confirmation_configs,
            confirmation,
            confirmation_clean,
            confirmation_cache,
            "confirmation",
        )
        reference_summaries = summarize_methods(
            args.stage,
            reference_rows,
            cohorts=("reference", "reference_clean"),
        )
        reference_bootstrap = bootstrap_methods(
            args.stage,
            reference_rows,
            "VCBG",
            args.bootstrap_seed,
            args.bootstrap_repeats,
            cohorts=("reference", "reference_clean"),
        )
        locked_reference_rows = [
            row
            for row in reference_rows
            if row["method"] in set(confirmation_configs)
        ]
        one_shot_rows = locked_reference_rows + confirmation_rows
        one_shot_summaries = summarize_methods(args.stage, one_shot_rows)
        one_shot_bootstrap = bootstrap_methods(
            args.stage,
            one_shot_rows,
            "VCBG",
            args.bootstrap_seed,
            args.bootstrap_repeats,
        )
        gate_rows, decision = confirmation_gate_outputs(
            one_shot_summaries, one_shot_bootstrap
        )
        outputs.update(
            {
                "fixed_reference_16_by_clip.csv": reference_rows,
                "fixed_reference_16_summary.csv": reference_summaries,
                "fixed_reference_16_paired_bootstrap.csv": reference_bootstrap,
                "fixed_one_shot_by_clip.csv": one_shot_rows,
                "fixed_one_shot_summary.csv": one_shot_summaries,
                "fixed_one_shot_paired_bootstrap.csv": one_shot_bootstrap,
                "fixed_one_shot_gates.csv": gate_rows,
                "fixed_one_shot_decision.json": decision,
            }
        )
        evaluated_configs = {
            "reference_fixed_16": reference_configs,
            "confirmation_fixed_4": confirmation_configs,
        }
    elif args.stage == "locked-candidates":
        configs = {
            "historical_primary": {
                "mask_integer": BASELINE_MASK,
                "postprocess": historical,
                "postprocess_hash": historical_record["postprocess_hash"],
            },
            "performance_A": selection["frozen_candidates"]["performance_A"],
            "parsimony_B": selection["frozen_candidates"]["parsimony_B"],
        }
        reference_rows = evaluate_configs(
            configs,
            reference,
            reference_clean,
            reference_cache,
            "reference",
        )
        confirmation_rows = evaluate_configs(
            configs,
            confirmation,
            confirmation_clean,
            confirmation_cache,
            "confirmation",
        )
        candidate_rows = reference_rows + confirmation_rows
        candidate_summaries = summarize_methods(args.stage, candidate_rows)
        candidate_bootstrap = bootstrap_methods(
            args.stage,
            candidate_rows,
            "historical_primary",
            args.bootstrap_seed,
            args.bootstrap_repeats,
        )
        outputs.update(
            {
                "locked_candidates_by_clip.csv": candidate_rows,
                "locked_candidates_summary.csv": candidate_summaries,
                "locked_candidates_paired_bootstrap.csv": candidate_bootstrap,
                "locked_candidates_scope.json": {
                    "systems_reported": list(configs),
                    "heldout_selection_permitted": False,
                    "all_prespecified_systems_reported": True,
                    "historical_primary_remains_prespecified_comparator": True,
                },
            }
        )
        evaluated_configs = configs
    else:
        assert structural_lock_payload is not None
        structural_config = structural_lock_payload["selected_postprocess"]
        configs = {
            "VCBG": {
                "mask_integer": BASELINE_MASK,
                "postprocess": historical,
                "postprocess_hash": historical_record["postprocess_hash"],
            },
            "V-B-": {
                "mask_integer": 5,
                "postprocess": structural_config,
                "postprocess_hash": canonical_hash(structural_config),
            },
        }
        reference_rows = evaluate_configs(
            configs,
            reference,
            reference_clean,
            reference_cache,
            "reference",
        )
        confirmation_rows = evaluate_configs(
            configs,
            confirmation,
            confirmation_clean,
            confirmation_cache,
            "confirmation",
        )
        structural_rows = reference_rows + confirmation_rows
        structural_summaries = summarize_methods(args.stage, structural_rows)
        structural_bootstrap = bootstrap_methods(
            args.stage,
            structural_rows,
            "VCBG",
            args.bootstrap_seed,
            args.bootstrap_repeats,
        )
        outputs.update(
            {
                "multiseed_structural_by_clip.csv": structural_rows,
                "multiseed_structural_summary.csv": structural_summaries,
                "multiseed_structural_paired_bootstrap.csv": structural_bootstrap,
                "multiseed_structural_scope.json": {
                    "systems_reported": ["VCBG", "V-B-"],
                    "additional_candidates_permitted": False,
                    "heldout_selection_permitted": False,
                    "selection_source": str(structural_lock_path),
                    "historical_primary_remains_prespecified_comparator": True,
                },
            }
        )
        evaluated_configs = configs

    output_hashes = write_stage_outputs(stage_dir, outputs)
    manifest = {
        "evaluation_only": True,
        "stage": args.stage,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "evaluation_manifest": str(evaluation_manifest_path),
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
        "selection_lock": str(lock_path),
        "selection_lock_sha256": sha256_file(lock_path),
        "selection_hash": selection["selection_hash"],
        "structural_lock": structural_lock_metadata,
        "prior_evaluation_output_timestamps": prior_evaluation_timestamps,
        "configuration_set_hash": canonical_hash(evaluated_configs),
        "bootstrap": {
            "cluster": "middle clip-ID field (source bout)",
            "repeats": args.bootstrap_repeats,
            "base_seed": args.bootstrap_seed,
            "interval": "two-sided 95% percentile",
            "paired_difference": "candidate minus historical baseline",
        },
        "inputs": {
            "reference_annotation_dir": str(reference_annotation_dir),
            "reference_all_annotations_aggregate_sha256": aggregate_files_hash(
                reference_files
            ),
            "confirmation_annotation_dir": str(confirmation_annotation_dir),
            "confirmation_all_annotations_aggregate_sha256": aggregate_files_hash(
                confirmation_files
            ),
            "reference_score_cache": str(reference_cache_path),
            "reference_score_cache_sha256": sha256_file(reference_cache_path),
            "reference_score_cache_meta": str(reference_meta_path),
            "reference_score_cache_meta_sha256": sha256_file(reference_meta_path),
            "reference_score_cache_metadata_hash": canonical_hash(reference_cache_meta),
            "confirmation_score_cache": str(confirmation_cache_path),
            "confirmation_score_cache_sha256": sha256_file(
                confirmation_cache_path
            ),
            "confirmation_score_cache_meta": str(confirmation_meta_path),
            "confirmation_score_cache_meta_sha256": sha256_file(
                confirmation_meta_path
            ),
            "confirmation_score_cache_metadata_hash": canonical_hash(
                confirmation_cache_meta
            ),
            **score_source_hashes,
        },
        "invariants": {
            **EXPECTED_COUNTS,
            "fixed_reference_masks": 16
            if args.stage == "fixed-confirmation"
            else None,
            "fixed_confirmation_masks": list(
                mask_label(mask) for mask in FIXED_CONFIRMATION_MASKS
            )
            if args.stage == "fixed-confirmation"
            else None,
            "locked_candidate_systems": 3
            if args.stage == "locked-candidates"
            else None,
            "multiseed_structural_systems": ["VCBG", "V-B-"]
            if args.stage == "multiseed-structural"
            else None,
            "multiseed_structural_additional_candidates": 0
            if args.stage == "multiseed-structural"
            else None,
        },
        "outputs": output_hashes,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(stage_dir / "evaluation_manifest.json", manifest)
    print(
        f"wrote frozen {args.stage} evaluation to {stage_dir}; "
        f"selection_hash={selection['selection_hash']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
