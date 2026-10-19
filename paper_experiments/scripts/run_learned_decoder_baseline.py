#!/usr/bin/env python3
"""Train a small supervised decoder over frozen boundary scores on dev labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from annotation_schema import normalize_annotation  # noqa: E402
from common import read_csv, read_json, write_csv, write_json  # noqa: E402
from compare_annotations_to_model import clip_metrics, summarize  # noqa: E402
from generate_boundary_predictions import partition_states  # noqa: E402
from search_boundary_postprocess import cache_scores, load_cache, zones_from_score  # noqa: E402


PRIMARY_CONFIG = REPO / "paper_experiments/experiments/truncated_retraining/candidates/optimistic_smooth02_seed18/config.yaml"
PRIMARY_CKPT = REPO / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_seed18_smooth02/ckpt_step_1000.pt"


def load_annotations(path: Path) -> Dict[str, dict]:
    anns = {}
    for file in sorted(path.glob("*.json")):
        ann = normalize_annotation(read_json(file), strict=False)
        flags = ann.get("quality_flags", {})
        if ann.get("annotation_complete") and flags.get("valid", True) and not flags.get("not_exchange"):
            anns[ann["clip_id"]] = ann
    return anns


def label_at(segments: List[Tuple[int, int, str]], frame: int) -> int:
    for start, end, activity in segments:
        if start <= frame <= end:
            return 1 if activity == "active" else 0
    return -1


def annotation_segments(ann: dict) -> List[Tuple[int, int, str]]:
    return [(int(s["start_frame"]), int(s["end_frame"]), str(s["activity"])) for s in ann.get("segments", [])]


def feature_matrix(entry: Dict[str, np.ndarray]) -> np.ndarray:
    b = np.asarray(entry["b"], dtype=np.float32)
    phi = np.asarray(entry["phi"], dtype=np.float32)
    db = np.concatenate([[0.0], np.abs(np.diff(b))]).astype(np.float32)
    left = np.concatenate([b[:1], b[:-1]])
    right = np.concatenate([b[1:], b[-1:]])
    local_max = np.maximum.reduce([left, b, right]).astype(np.float32)
    local_min = np.minimum.reduce([left, b, right]).astype(np.float32)
    return np.concatenate([b[:, None], db[:, None], local_max[:, None], local_min[:, None], phi], axis=1)


def training_rows(cache: Dict[str, Dict[str, np.ndarray]], anns: Dict[str, dict]) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for clip_id, ann in anns.items():
        entry = cache[clip_id]
        frames = [int(f) for f in entry["frames"]]
        feats = feature_matrix(entry)
        segments = annotation_segments(ann)
        for idx, frame in enumerate(frames):
            y = label_at(segments, frame)
            if y < 0:
                continue
            xs.append(feats[idx])
            ys.append(y)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def predict_states(
    clip_id: str,
    entry: Dict[str, np.ndarray],
    model: Any,
    threshold: float,
    pp: Dict[str, Any],
) -> List[Tuple[int, int, str]]:
    feats = feature_matrix(entry)
    prob = model.predict_proba(feats)[:, 1].astype(np.float32)
    zones = zones_from_score(prob, entry["mask"], {**pp, "mode": "legacy", "thr_on": threshold, "thr_off": threshold * 0.6})
    data = {"frames": entry["frames"].tolist(), "l2_mask": entry["mask"].tolist()}
    states = partition_states(data, zones, len(prob))
    return [(int(s["start_frame"]), int(s["end_frame"]), str(s["activity"])) for s in states]


def evaluate_decoder(
    cache: Dict[str, Dict[str, np.ndarray]],
    anns: Dict[str, dict],
    model: Any,
    threshold: float,
    pp: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for clip_id, ann in anns.items():
        rows.append(clip_metrics(clip_id, ann, predict_states(clip_id, cache[clip_id], model, threshold, pp)))
    return rows, summarize(rows, "usable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-manifest", default="paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv")
    parser.add_argument("--final-manifest", default="paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv")
    parser.add_argument("--dev-annotations", default="paper_experiments/annotations/raw/owner")
    parser.add_argument("--consensus-annotations", default="paper_experiments/annotations/adjudicated/consensus")
    parser.add_argument("--single-annotations", default="paper_experiments/annotations/raw/collaborator_a")
    parser.add_argument("--out-root", default="paper_experiments/experiments/learned_decoder")
    args = parser.parse_args()

    out_root = REPO / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    dev_rows_all = read_csv(REPO / args.dev_manifest)
    final_rows = read_csv(REPO / args.final_manifest)
    dev_anns_all = load_annotations(REPO / args.dev_annotations)
    dev_rows = [row for row in dev_rows_all if row["clip_id"] in dev_anns_all]
    dev_anns = {row["clip_id"]: dev_anns_all[row["clip_id"]] for row in dev_rows}
    consensus_anns = load_annotations(REPO / args.consensus_annotations)
    single_anns = load_annotations(REPO / args.single_annotations)

    dev_cache_path = cache_scores("primary_dev", PRIMARY_CONFIG, PRIMARY_CKPT, dev_rows, out_root / "score_cache")
    final_cache_path = cache_scores("primary_final", PRIMARY_CONFIG, PRIMARY_CKPT, final_rows, out_root / "score_cache")
    dev_cache = load_cache(dev_cache_path, dev_anns.keys())
    final_clip_ids = {row["clip_id"] for row in final_rows}
    final_cache = load_cache(final_cache_path, final_clip_ids)

    x, y = training_rows(dev_cache, dev_anns)
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=20270618))
    clf.fit(x, y)

    base_pp = {
        "ema_alpha": 0.35,
        "thr_on": 0.5,
        "thr_off": 0.3,
        "min_len": 3,
        "merge_gap": 1,
        "min_seg_len": 1,
        "abs_on": 0.0,
        "abs_off": 0.0,
        "robust_z_on": 1.0,
        "robust_z_off": 0.4,
        "min_prominence": 0.0,
        "min_peak_height": 0.0,
        "boundary_frac_min": 0.0,
        "boundary_frac_max": 1.0,
        "budget_step": 0.15,
        "max_budget_iters": 0,
        "onset_backtrack": 0,
        "split_min_zone_len": 0,
        "split_min_rel_drop": 0.0,
        "split_max_splits": 0,
        "micro_merge_zone_len": 0,
        "micro_merge_gap": 0,
        "micro_merge_max_span": 0,
    }
    candidates = []
    for threshold in np.linspace(0.25, 0.75, 21):
        for alpha in (0.15, 0.25, 0.35, 0.5):
            for min_len in (2, 3, 4, 5):
                pp = dict(base_pp)
                pp["ema_alpha"] = float(alpha)
                pp["min_len"] = int(min_len)
                _, summary = evaluate_decoder(dev_cache, dev_anns, clf, float(threshold), pp)
                objective = (
                    float(summary["activity_macro_f1"])
                    + float(summary["active_f1"])
                    + float(summary["boundary_f1_pm5"])
                    + float(summary["boundary_f1_pm10"])
                ) / 4.0
                candidates.append((objective, float(threshold), pp, summary))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_objective, best_threshold, best_pp, best_dev = candidates[0]

    result_payload = {
        "development_only": True,
        "train_frames": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "best_threshold": best_threshold,
        "best_postprocess": best_pp,
        "best_dev_summary": best_dev,
    }
    write_json(out_root / "selection.json", result_payload)

    outputs = []
    for name, anns in (("consensus", consensus_anns), ("single", single_anns)):
        anns = {clip_id: ann for clip_id, ann in anns.items() if clip_id in final_cache}
        rows, summary = evaluate_decoder(final_cache, anns, clf, best_threshold, best_pp)
        write_csv(out_root / f"{name}_by_clip.csv", sorted(rows, key=lambda row: float(row["boundary_f1_pm5"])))
        write_csv(out_root / f"{name}_summary.csv", [summary])
        outputs.append({"reference": name, **summary})
    write_csv(out_root / "summary.csv", outputs)
    print(json.dumps(result_payload, indent=2))
    print(f"Wrote {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
