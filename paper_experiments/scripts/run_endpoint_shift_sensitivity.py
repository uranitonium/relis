#!/usr/bin/env python3
"""Evaluate sensitivity to small scoring-light endpoint perturbations.

The paper's target is the pre-touch exchange window. This script rebuilds L2
sequences from the untrimmed feature source with the endpoint shifted by a few
frames, reruns frozen prediction pipelines, shifts the terminal consensus state
by the same amount, and exports comparable metrics.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from annotation_schema import normalize_annotation
from common import read_csv, read_json, write_csv, write_json


REPO = Path(__file__).resolve().parents[2]
V2_DATA_ROOT = Path(os.getenv("FENCING_V2_DATA_ROOT", str(REPO.parent / "V2"))).resolve()
MANIFEST = REPO / "paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv"
CONSENSUS_DIR = REPO / "paper_experiments/annotations/adjudicated/consensus"
FULL_L2_ROOT = V2_DATA_ROOT / "old/old2/l2_out_raw_firstcenter_boundary_full"
OUT_ROOT = REPO / "paper_experiments/experiments/final_boundary_evaluation/endpoint_shift_sensitivity"
PAPER_READY_OUT = REPO / "paper_experiments/paper_ready/foil/endpoint_shift_sensitivity.csv"
OPTIMISTIC_CONFIG = REPO / "paper_experiments/experiments/truncated_retraining/candidates/optimistic_smooth02_seed18/config.yaml"
CONSERVATIVE_CONFIG = REPO / "paper_experiments/experiments/truncated_retraining/candidates/conservative_smooth02_seed18/config.yaml"
TUNED_CKPT = REPO / "paper_experiments/experiments/truncated_retraining/runs/short_sparse_seed18_smooth02/ckpt_step_1000.pt"
OPENMMLAB_PY = Path(os.environ.get("RELIS_PYTHON", sys.executable))


def run(cmd: List[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def delta_tag(delta: int) -> str:
    return f"m{abs(delta)}" if delta < 0 else f"p{delta}"


def shifted_l2_and_manifest(delta: int, manifest: Path, l2_root: Path, out_root: Path) -> Path:
    tag = delta_tag(delta)
    l2_out = out_root / f"delta_{tag}" / "l2"
    l2_out.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in read_csv(manifest):
        clip_id = row["clip_id"]
        source_path = l2_root / f"{clip_id}_l2.json"
        data = read_json(source_path)
        frames = [int(frame) for frame in data.get("frames", [])]
        endpoint = int(float(row.get("hit_frame") or row["frame_end"])) + delta
        keep = [idx for idx, frame in enumerate(frames) if frame <= endpoint]
        if not keep:
            raise ValueError(f"{clip_id}: no frames at endpoint {endpoint}")
        last = keep[-1] + 1
        shifted = dict(data)
        for key in ("l2", "l2_mask", "frames"):
            shifted[key] = data[key][:last]
        shifted["meta"] = dict(shifted.get("meta", {}))
        shifted["meta"].update(
            {
                "evaluation_end_frame": endpoint,
                "evaluation_end_source": f"scoring_light_shift_{delta:+d}",
                "original_sequence_length": len(data.get("l2", [])),
                "endpoint_shift_frames": delta,
            }
        )
        out_path = l2_out / f"{clip_id}_l2.json"
        write_json(out_path, shifted)
        out_row = dict(row)
        out_row["l2_path"] = str(out_path.relative_to(REPO))
        out_row["frame_end"] = shifted["frames"][-1]
        out_row["num_frames"] = len(shifted["frames"])
        out_row["evaluation_end_source"] = f"scoring_light_shift_{delta:+d}"
        rows.append(out_row)
    out_manifest = out_root / f"delta_{tag}" / "manifest.csv"
    write_csv(out_manifest, rows)
    return out_manifest


def shift_annotation(annotation: dict, delta: int) -> dict:
    ann = copy.deepcopy(annotation)
    segments = ann.get("segments", [])
    if not segments:
        return ann
    fps = float(ann.get("fps") or 25.0)
    first_frame = int(segments[0]["start_frame"])
    old_end = int(segments[-1]["end_frame"])
    new_end = max(first_frame, old_end + delta)
    out_segments = []
    for segment in segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if start > new_end:
            break
        updated = copy.deepcopy(segment)
        if end >= new_end:
            updated["end_frame"] = new_end
            updated["end_sec"] = (new_end + 1) / fps
            out_segments.append(updated)
            break
        out_segments.append(updated)
    if not out_segments:
        updated = copy.deepcopy(segments[0])
        updated["start_frame"] = first_frame
        updated["start_sec"] = first_frame / fps
        updated["end_frame"] = new_end
        updated["end_sec"] = (new_end + 1) / fps
        out_segments = [updated]
    elif int(out_segments[-1]["end_frame"]) < new_end:
        out_segments[-1]["end_frame"] = new_end
        out_segments[-1]["end_sec"] = (new_end + 1) / fps
    ann["segments"] = out_segments
    for key in ("evaluation_range", "analyzed_range"):
        if key in ann:
            ann[key] = dict(ann[key])
            ann[key]["end_frame"] = new_end
            ann[key]["end_sec"] = (new_end + 1) / fps
    ann["free_text_notes"] = (ann.get("free_text_notes", "") + f"\nEndpoint shifted by {delta:+d} frames for sensitivity analysis.").strip()
    return ann


def shifted_annotations(delta: int, annotation_dir: Path, out_root: Path) -> Path:
    tag = delta_tag(delta)
    out_dir = out_root / f"delta_{tag}" / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    for path in sorted(annotation_dir.glob("*.json")):
        annotation = normalize_annotation(read_json(path), strict=False)
        write_json(out_dir / path.name, shift_annotation(annotation, delta))
    return out_dir


def generate_predictions(delta_root: Path, manifest: Path, python_bin: Path) -> Dict[str, Path]:
    pred_root = delta_root / "predictions"
    specs = {
        "relis_alt_extractor": ("full_model", OPTIMISTIC_CONFIG),
        "raw_phi": ("raw_phi_l2", CONSERVATIVE_CONFIG),
    }
    outputs: Dict[str, Path] = {}
    for name, (method, config) in specs.items():
        out_dir = pred_root / name
        run(
            [
                str(python_bin),
                "paper_experiments/scripts/generate_boundary_predictions.py",
                "--manifest",
                str(manifest),
                "--method",
                method,
                "--out-dir",
                str(out_dir),
                "--config",
                str(config),
                "--ckpt",
                str(TUNED_CKPT),
                "--python-bin",
                str(python_bin),
                "--seed",
                "20270726",
            ]
        )
        outputs[name] = out_dir / f"{method}.json"
    return outputs


def compare_predictions(delta_root: Path, manifest: Path, annotation_dir: Path, predictions: Dict[str, Path]) -> List[dict]:
    rows = []
    for name, pred in predictions.items():
        result_dir = delta_root / "results" / name
        run(
            [
                sys.executable,
                "paper_experiments/scripts/compare_annotations_to_model.py",
                "--owner-dir",
                str(annotation_dir),
                "--manifest",
                str(manifest),
                "--predictions",
                str(pred),
                "--model-annotations-dir",
                str(result_dir / "model_annotations"),
                "--summary-out",
                str(result_dir / "summary.csv"),
                "--by-clip-out",
                str(result_dir / "by_clip.csv"),
                "--viewer-metrics-out",
                str(result_dir / "viewer_metrics.csv"),
            ]
        )
        for row in read_csv(result_dir / "summary.csv"):
            if row["cohort"] == "valid":
                out = {"method": name, **row}
                rows.append(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--annotation-dir", default=str(CONSENSUS_DIR))
    parser.add_argument("--full-l2-root", default=str(FULL_L2_ROOT))
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--paper-ready-out", default=str(PAPER_READY_OUT))
    parser.add_argument("--python-bin", default=str(OPENMMLAB_PY if OPENMMLAB_PY.exists() else sys.executable))
    parser.add_argument("--deltas", default="-5,-3,0,3,5")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    all_rows = []
    for delta in [int(item.strip()) for item in args.deltas.split(",") if item.strip()]:
        tag = delta_tag(delta)
        delta_root = out_root / f"delta_{tag}"
        manifest = shifted_l2_and_manifest(delta, Path(args.manifest), Path(args.full_l2_root), out_root)
        annotations = shifted_annotations(delta, Path(args.annotation_dir), out_root)
        predictions = generate_predictions(delta_root, manifest, Path(args.python_bin))
        for row in compare_predictions(delta_root, manifest, annotations, predictions):
            row["endpoint_shift_frames"] = delta
            all_rows.append(row)

    columns = [
        "method",
        "endpoint_shift_frames",
        "cohort",
        "clips",
        "activity_accuracy",
        "active_f1",
        "boundary_f1_pm5",
        "boundary_f1_pm10",
        "human_states",
        "model_states",
        "human_active_coverage",
        "model_active_coverage",
    ]
    compact_rows = [{key: row.get(key, "") for key in columns} for row in all_rows]
    write_csv(Path(args.paper_ready_out), compact_rows)
    print(f"Wrote {args.paper_ready_out}")


if __name__ == "__main__":
    main()
