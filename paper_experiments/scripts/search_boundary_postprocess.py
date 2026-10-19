#!/usr/bin/env python3
"""Cache raw boundary scores and search postprocessing on development labels."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml


REPO = Path(__file__).resolve().parents[2]
L3_DIR = REPO / "boundary_segmentation"
if str(L3_DIR) not in sys.path:
    sys.path.insert(0, str(L3_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotation_schema import normalize_annotation  # noqa: E402
from common import read_csv, read_json, read_sequence, write_csv, write_json  # noqa: E402
from compare_annotations_to_model import clip_metrics, summarize  # noqa: E402
from generate_boundary_predictions import partition_states  # noqa: E402
from models.l3_model import L3BoundaryModel  # noqa: E402
from postprocess.zones import extract_transition_zones  # noqa: E402


SEARCH_ROOT = REPO / "paper_experiments" / "experiments" / "truncated_retraining" / "postprocess_search"
DEFAULT_CONFIG = REPO / "deployment" / "checkpoints" / "legacy_auto" / "config.yaml"
DEFAULT_CKPT = REPO / "deployment" / "checkpoints" / "legacy_auto" / "ckpt_best_ema.pt"


def model_spec(value: str) -> Tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("model must be NAME=CONFIG=CHECKPOINT")
    return parts[0], Path(parts[1]).resolve(), Path(parts[2]).resolve()


def cache_spec(value: str) -> Tuple[str, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("score cache must be NAME=NPZ")
    path = Path(parts[1]).resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"score cache does not exist: {path}")
    return parts[0], path


def load_annotations(owner_dir: Path) -> Dict[str, dict]:
    output = {}
    for path in sorted(owner_dir.glob("*.json")):
        ann = normalize_annotation(read_json(path), strict=False)
        flags = ann.get("quality_flags", {})
        if ann.get("annotation_complete") and flags.get("valid", True) and not flags.get("not_exchange"):
            output[ann["clip_id"]] = ann
    return output


def extend_sequence(
    x: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    target_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_length = x.shape[1]
    extra = max(0, target_length - original_length)
    if mode == "none" or extra == 0:
        return x, mask

    valid = x[0, mask[0]]
    if valid.numel() == 0:
        return x, mask
    if mode == "repeat_last":
        tail = valid[-1:].repeat(extra, 1)
    elif mode == "reflect":
        source = torch.flip(valid, dims=[0])
        repeats = (extra + source.shape[0] - 1) // source.shape[0]
        tail = source.repeat(repeats, 1)[:extra]
    else:
        raise ValueError(f"Unknown extension mode: {mode}")
    tail = tail.unsqueeze(0)
    tail_mask = torch.ones((1, extra), dtype=torch.bool, device=mask.device)
    return torch.cat([x, tail], dim=1), torch.cat([mask, tail_mask], dim=1)


def cache_scores(
    name: str,
    config_path: Path,
    checkpoint_path: Path,
    rows: List[Dict[str, str]],
    cache_dir: Path,
    extension_mode: str = "none",
    extension_length: int = 127,
    zero_feature_indices: List[int] | None = None,
) -> Path:
    zero_feature_indices = zero_feature_indices or []
    cache_path = cache_dir / f"{name}.npz"
    metadata_path = cache_dir / f"{name}.json"
    expected = {
        "config": str(config_path),
        "config_mtime_ns": config_path.stat().st_mtime_ns,
        "checkpoint": str(checkpoint_path),
        "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
        "clips": [row["clip_id"] for row in rows],
        "extension_mode": extension_mode,
        "extension_length": extension_length,
        "zero_feature_indices": zero_feature_indices,
    }
    if cache_path.exists() and metadata_path.exists() and read_json(metadata_path) == expected:
        return cache_path

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    first = read_sequence(rows[0]["l2_path"])
    input_dim = np.asarray(first["l2"], dtype=np.float32).shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = L3BoundaryModel(input_dim=input_dim, cfg=cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    arrays: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for row in rows:
            data = read_sequence(row["l2_path"])
            x = torch.as_tensor(data["l2"], dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.as_tensor(data["l2_mask"], dtype=torch.bool, device=device).unsqueeze(0)
            if zero_feature_indices:
                x = x.clone()
                for index in zero_feature_indices:
                    if 0 <= index < x.shape[-1]:
                        x[..., index] = 0.0
            original_length = x.shape[1]
            x_model, mask_model = extend_sequence(x, mask, extension_mode, extension_length)
            _, _, _, phi, b = model(x_model, mask_model)
            clip_id = row["clip_id"]
            arrays[f"b::{clip_id}"] = b[0, :original_length].cpu().numpy().astype(np.float32)
            arrays[f"phi::{clip_id}"] = phi[0, :original_length].cpu().numpy().astype(np.float32)
            arrays[f"mask::{clip_id}"] = mask.squeeze(0).cpu().numpy().astype(np.bool_)
            arrays[f"frames::{clip_id}"] = np.asarray(data["frames"], dtype=np.int32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    write_json(metadata_path, expected)
    return cache_path


def load_cache(path: Path, clip_ids: Iterable[str]) -> Dict[str, Dict[str, np.ndarray]]:
    data = np.load(path)
    return {
        clip_id: {
            "b": data[f"b::{clip_id}"],
            "phi": data[f"phi::{clip_id}"],
            "mask": data[f"mask::{clip_id}"],
            "frames": data[f"frames::{clip_id}"],
        }
        for clip_id in clip_ids
    }


def deployed_postprocess() -> Dict[str, Any]:
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    return dict(cfg["postprocess"])


def sample_config(rng: random.Random) -> Dict[str, Any]:
    mode = rng.choice(["robust_hysteresis"] * 4 + ["legacy"])
    min_len = rng.choice([1, 2, 3, 3, 4, 5, 6])
    split_enabled = rng.random() < 0.65
    clamp_enabled = rng.random() < 0.7
    boundary_min = rng.choice([0.0, 0.0, 0.025, 0.05, 0.075, 0.10]) if clamp_enabled else 0.0
    boundary_max = rng.choice([0.18, 0.22, 0.26, 0.30, 0.35, 0.45, 1.0]) if clamp_enabled else 1.0
    if boundary_max <= boundary_min:
        boundary_max = min(1.0, boundary_min + 0.15)
    robust_on = rng.uniform(0.35, 1.8)
    robust_off = rng.uniform(-0.15, min(1.0, robust_on))
    abs_on = rng.choice([0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.16, 0.22])
    abs_off = rng.uniform(0.0, abs_on) if abs_on else 0.0
    thr_on = rng.uniform(0.03, 0.55)
    thr_off = rng.uniform(0.0, thr_on)
    return {
        "ema_alpha": rng.uniform(0.08, 0.92),
        "thr_on": thr_on,
        "thr_off": thr_off,
        "min_len": min_len,
        "merge_gap": rng.choice([0, 0, 1, 1, 2, 3, 4]),
        "min_seg_len": 1,
        "mode": mode,
        "abs_on": abs_on,
        "abs_off": abs_off,
        "robust_z_on": robust_on,
        "robust_z_off": robust_off,
        "min_prominence": rng.choice([0.0, 0.0, 0.03, 0.06, 0.10, 0.15, 0.25, 0.40]),
        "min_peak_height": 0.0,
        "boundary_frac_min": boundary_min,
        "boundary_frac_max": boundary_max,
        "budget_step": rng.choice([0.08, 0.12, 0.15, 0.20, 0.30]),
        "max_budget_iters": rng.choice([0, 1, 2, 3, 4, 5]),
        "onset_backtrack": rng.choice([0, 0, 1, 2, 3, 4, 5, 6, 8]),
        "split_min_zone_len": rng.choice([3, 4, 5, 6, 8]) if split_enabled else 0,
        "split_min_rel_drop": rng.uniform(0.12, 0.75) if split_enabled else 0.0,
        "split_max_splits": rng.choice([1, 1, 2, 3]) if split_enabled else 0,
        "micro_merge_zone_len": 0,
        "micro_merge_gap": 0,
        "micro_merge_max_span": 0,
    }


def zones_from_score(score: np.ndarray, mask: np.ndarray, pp: Dict[str, Any]) -> List[Tuple[int, int]]:
    return extract_transition_zones(
        score,
        mask,
        alpha=float(pp["ema_alpha"]),
        thr_on=float(pp["thr_on"]),
        min_len=int(pp["min_len"]),
        merge_gap=int(pp["merge_gap"]),
        mode=str(pp["mode"]),
        thr_off=float(pp["thr_off"]),
        abs_on=float(pp["abs_on"]),
        abs_off=float(pp["abs_off"]),
        robust_z_on=float(pp["robust_z_on"]),
        robust_z_off=float(pp["robust_z_off"]),
        min_prominence=float(pp["min_prominence"]),
        min_peak_height=float(pp["min_peak_height"]),
        boundary_frac_min=float(pp["boundary_frac_min"]),
        boundary_frac_max=float(pp["boundary_frac_max"]),
        budget_step=float(pp["budget_step"]),
        max_budget_iters=int(pp["max_budget_iters"]),
        onset_backtrack=int(pp.get("onset_backtrack", 0)),
        split_min_zone_len=int(pp.get("split_min_zone_len", 0)),
        split_min_rel_drop=float(pp.get("split_min_rel_drop", 0.0)),
        split_max_splits=int(pp.get("split_max_splits", 0)),
        micro_merge_zone_len=int(pp.get("micro_merge_zone_len", 0)),
        micro_merge_gap=int(pp.get("micro_merge_gap", 0)),
        micro_merge_max_span=int(pp.get("micro_merge_max_span", 0)),
    )


def score_from_entry(entry: Dict[str, np.ndarray], source: str) -> np.ndarray:
    if source == "b":
        return entry["b"]

    phi = entry["phi"]
    if source == "phi_mean":
        return phi.mean(axis=-1)
    if source == "phi_l2":
        return np.linalg.norm(phi, axis=-1)
    if source == "phi_max":
        return phi.max(axis=-1)
    if source.startswith("phi") and source[3:].isdigit():
        index = int(source[3:])
        if index >= phi.shape[-1]:
            raise ValueError(f"{source} is unavailable for phi dimension {phi.shape[-1]}")
        return phi[:, index]
    raise ValueError(f"Unknown score source: {source}")


def evaluate_rows(
    annotations: Dict[str, dict],
    caches: List[Dict[str, Dict[str, np.ndarray]]],
    weights: List[float],
    pp: Dict[str, Any],
    sources: List[str] | None = None,
) -> List[Dict[str, Any]]:
    sources = sources or ["b"] * len(caches)
    if len(sources) != len(caches):
        raise ValueError("One score source is required per cache")
    rows = []
    for clip_id, ann in annotations.items():
        entries = [cache[clip_id] for cache in caches]
        score = sum(
            weight * score_from_entry(entry, source)
            for weight, entry, source in zip(weights, entries, sources)
        )
        mask = entries[0]["mask"]
        frames = entries[0]["frames"]
        zones = zones_from_score(score, mask, pp)
        data = {"frames": frames.tolist(), "l2_mask": mask.tolist()}
        states = partition_states(data, zones, len(score))
        pred = [(int(s["start_frame"]), int(s["end_frame"]), str(s["activity"])) for s in states]
        rows.append(clip_metrics(clip_id, ann, pred))
    return rows


def objective_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = summarize(rows, "usable")
    summary["objective"] = (
        float(summary["activity_macro_f1"])
        + float(summary["active_f1"])
        + float(summary["boundary_f1_pm5"])
        + float(summary["boundary_f1_pm10"])
    ) / 4.0
    summary["balanced_objective"] = (
        float(summary["activity_macro_f1"]) + float(summary["boundary_f1_pm5"])
    ) / 2.0
    return summary


def evaluate(
    annotations: Dict[str, dict],
    caches: List[Dict[str, Dict[str, np.ndarray]]],
    weights: List[float],
    pp: Dict[str, Any],
    sources: List[str] | None = None,
) -> Dict[str, Any]:
    return objective_summary(evaluate_rows(annotations, caches, weights, pp, sources=sources))


def flatten_result(
    rank: int,
    model_name: str,
    weights: List[float],
    summary: Dict[str, Any],
    pp: Dict[str, Any],
    config_index: int,
    candidate_source: str,
    sample_index: int | None,
) -> Dict[str, Any]:
    keys = [
        "objective", "balanced_objective", "activity_accuracy", "activity_macro_f1", "active_precision",
        "active_recall", "active_f1", "boundary_f1_pm3", "boundary_f1_pm5", "boundary_f1_pm10",
        "model_active_coverage", "human_active_coverage", "total_human_states", "total_model_states",
        "total_state_count_ratio",
    ]
    row: Dict[str, Any] = {
        "rank": rank,
        "model": model_name,
        "weights": json.dumps(weights),
        "config_index": config_index,
        "candidate_source": candidate_source,
        "sample_index": "" if sample_index is None else sample_index,
    }
    row.update({key: summary[key] for key in keys})
    row.update({f"pp_{key}": value for key, value in pp.items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv")
    parser.add_argument("--owner-dir", default="paper_experiments/annotations/raw/owner")
    parser.add_argument("--model", action="append", type=model_spec)
    parser.add_argument(
        "--score-cache",
        action="append",
        type=cache_spec,
        help="Use an existing NAME=NPZ score cache; cannot be combined with --model",
    )
    parser.add_argument(
        "--score-source",
        action="append",
        choices=["b", "phi_mean", "phi_l2", "phi_max"] + [f"phi{i}" for i in range(8)],
        help="Score source for each --model (default: learned boundary score b)",
    )
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--extension-mode", choices=["none", "repeat_last", "reflect"], default="none")
    parser.add_argument("--extension-length", type=int, default=127)
    parser.add_argument("--zero-feature-index", action="append", type=int, default=[])
    parser.add_argument("--seed", type=int, default=20270614)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--random-only",
        action="store_true",
        help="Restrict selection to the seeded random samples (used for controlled model comparisons)",
    )
    parser.add_argument("--out-dir", default=str(SEARCH_ROOT))
    args = parser.parse_args()

    if args.model and args.score_cache:
        raise SystemExit("Use either --model or --score-cache, not both")
    specs = args.model or ([] if args.score_cache else [("deployed", DEFAULT_CONFIG, DEFAULT_CKPT)])
    cache_specs = args.score_cache or []
    model_count = len(cache_specs) if cache_specs else len(specs)
    sources = args.score_source or ["b"] * model_count
    if len(sources) == 1 and model_count > 1:
        sources *= model_count
    if len(sources) != model_count:
        raise SystemExit("Provide either one --score-source or one per --model")
    rows = read_csv(args.manifest)
    annotations = load_annotations(Path(args.owner_dir))
    rows = [row for row in rows if row["clip_id"] in annotations]
    annotations = {row["clip_id"]: annotations[row["clip_id"]] for row in rows}
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "score_cache"
    if cache_specs:
        cache_paths = [path for _, path in cache_specs]
    else:
        cache_paths = [
            cache_scores(
                name,
                cfg,
                ckpt,
                rows,
                cache_dir,
                extension_mode=args.extension_mode,
                extension_length=args.extension_length,
                zero_feature_indices=args.zero_feature_index,
            )
            for name, cfg, ckpt in specs
        ]
    caches = [load_cache(path, annotations.keys()) for path in cache_paths]

    if len(caches) == 1:
        weight_sets = [[1.0]]
    elif len(caches) == 2:
        weight_sets = [[round(alpha, 2), round(1.0 - alpha, 2)] for alpha in np.linspace(0.0, 1.0, 11)]
    else:
        raise SystemExit("Search currently supports one or two models")

    rng = random.Random(args.seed)
    candidate_records: List[Tuple[str, int | None, Dict[str, Any]]] = []
    if not args.random_only:
        candidate_records.append(("deployed", None, deployed_postprocess()))
    candidate_records.extend(("random", index, sample_config(rng)) for index in range(1, args.trials + 1))
    results = []
    names = [name for name, _ in cache_specs] if cache_specs else [spec[0] for spec in specs]
    model_name = "+".join(f"{name}-{source}" for name, source in zip(names, sources))
    if args.extension_mode != "none":
        model_name += f"-{args.extension_mode}{args.extension_length}"
    if args.zero_feature_index:
        model_name += "-zero" + "_".join(str(index) for index in args.zero_feature_index)
    for config_index, (candidate_source, sample_index, pp) in enumerate(candidate_records):
        for weights in weight_sets:
            summary = evaluate(annotations, caches, weights, pp, sources=sources)
            results.append(
                (
                    summary["objective"],
                    summary,
                    weights,
                    pp,
                    config_index,
                    candidate_source,
                    sample_index,
                )
            )
    results.sort(key=lambda item: (item[0], item[1]["balanced_objective"]), reverse=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    top = results[: args.top_k]
    write_csv(
        out_dir / f"{model_name}_top.csv",
        [
            flatten_result(i + 1, model_name, w, s, pp, config_index, candidate_source, sample_index)
            for i, (_, s, w, pp, config_index, candidate_source, sample_index) in enumerate(top)
        ],
    )
    write_json(
        out_dir / f"{model_name}_best.json",
        {
            "development_only": True,
            "manifest": args.manifest,
            "models": (
                [
                    {"name": name, "score_cache": str(path), "score_source": source}
                    for (name, path), source in zip(cache_specs, sources)
                ]
                if cache_specs
                else [
                    {"name": n, "config": str(c), "checkpoint": str(k), "score_source": source}
                    for (n, c, k), source in zip(specs, sources)
                ]
            ),
            "trials": args.trials,
            "seed": args.seed,
            "candidate_protocol": {
                "random_samples": args.trials,
                "random_only": args.random_only,
                "include_deployed_candidate": not args.random_only,
                "total_candidates": len(candidate_records),
            },
            "extension_mode": args.extension_mode,
            "extension_length": args.extension_length,
            "zero_feature_indices": args.zero_feature_index,
            "best": {
                "summary": top[0][1],
                "weights": top[0][2],
                "postprocess": top[0][3],
                "config_index": top[0][4],
                "candidate_source": top[0][5],
                "sample_index": top[0][6],
            },
            "deployed_baseline": {
                "summary": evaluate(
                    annotations,
                    caches,
                    [1.0] + [0.0] * (len(caches) - 1),
                    deployed_postprocess(),
                    sources=sources,
                ),
                "postprocess": deployed_postprocess(),
            },
        },
    )
    best = top[0]
    print(json.dumps({"model": model_name, "summary": best[1], "weights": best[2], "postprocess": best[3]}, indent=2))


if __name__ == "__main__":
    main()
