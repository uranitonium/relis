#!/usr/bin/env python3
"""Train and evaluate the frozen-protocol Video Swin representation swap."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
PY = Path(sys.executable)
ROOT = REPO / "paper_experiments/round2/video_swin_baseline"
BASE_CONFIG = REPO / "paper_experiments/experiments/video_videomae_baseline/configs/video_videomae_l3.yaml"
OWNER = REPO / "paper_experiments/annotations/raw/owner"
CONSENSUS = REPO / "paper_experiments/annotations/adjudicated/consensus"


def run(command: list[object]) -> None:
    values = [str(value) for value in command]
    print("+", " ".join(values), flush=True)
    subprocess.run(values, cwd=str(REPO), check=True)


def read_valid(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["cohort"] == "valid":
                return row
    raise KeyError(f"No valid row in {path}")


def main() -> None:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config["seed"] = 20270618
    config["data"]["data_dir"] = str((ROOT / "features/train").resolve())
    config["data"]["file_glob"] = "*_video_swin.npz"
    config["data"]["synthetic_m"] = 768
    config["train"]["max_steps"] = 1000
    config["train"]["ckpt_every"] = 1000
    config["train"]["early_stop"]["enabled"] = False
    config["train"]["out_dir"] = str((ROOT / "runs/video_swin_seed20270618").resolve())
    config_path = ROOT / "configs/video_swin.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    checkpoint = ROOT / "runs/video_swin_seed20270618/ckpt_step_1000.pt"
    if not checkpoint.exists():
        run([PY, "boundary_segmentation/train.py", "--config", config_path, "--device", "auto"])

    search_root = ROOT / "postprocess_search"
    best_path = search_root / "video_swin-b_best.json"
    if not best_path.exists():
        run(
            [
                PY,
                "paper_experiments/scripts/search_boundary_postprocess.py",
                "--manifest",
                ROOT / "manifests/dev_video_swin.csv",
                "--owner-dir",
                OWNER,
                "--model",
                f"video_swin={config_path}={checkpoint}",
                "--score-source",
                "b",
                "--trials",
                3500,
                "--seed",
                20270618,
                "--top-k",
                100,
                "--out-dir",
                search_root,
            ]
        )
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    config["postprocess"] = payload["best"]["postprocess"]
    tuned_path = ROOT / "configs/video_swin_tuned.yaml"
    tuned_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    prediction_dir = ROOT / "predictions"
    prediction_path = prediction_dir / "full_model.json"
    if not prediction_path.exists():
        run(
            [
                PY,
                "paper_experiments/scripts/generate_boundary_predictions.py",
                "--manifest",
                ROOT / "manifests/final_video_swin.csv",
                "--method",
                "full_model",
                "--out-dir",
                prediction_dir,
                "--config",
                tuned_path,
                "--ckpt",
                checkpoint,
                "--seed",
                20270618,
            ]
        )
    result_root = ROOT / "consensus_results"
    summary = result_root / "summary.csv"
    if not summary.exists():
        run(
            [
                PY,
                "paper_experiments/scripts/compare_annotations_to_model.py",
                "--owner-dir",
                CONSENSUS,
                "--manifest",
                ROOT / "manifests/final_video_swin.csv",
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
    dev = payload["best"]["summary"]
    test = read_valid(summary)
    report = "\n".join(
        [
            "# Frozen Video Swin representation comparison",
            "",
            "- Backbone: Kinetics-400-pretrained Swin3D-T, frozen feature extractor",
            "- Sampling: centered 16-frame windows, stride 8, nearest repetition",
            "- Boundary learner: 1,000 steps, seed 20270618",
            "- Extractor selection: 3,500 development-only trials",
            f"- Development: Active F1 {float(dev['active_f1']):.3f}, BF1@5 {float(dev['boundary_f1_pm5']):.3f}, BF1@10 {float(dev['boundary_f1_pm10']):.3f}",
            f"- Consensus (104): Acc. {float(test['activity_accuracy']):.3f}, Active F1 {float(test['active_f1']):.3f}, BF1@5 {float(test['boundary_f1_pm5']):.3f}, BF1@10 {float(test['boundary_f1_pm10']):.3f}",
            "",
            "This reviewer-requested representation comparison followed access to the",
            "original consensus labels and uses one seed; it is not a new independent",
            "benchmark or an equally exhaustive optimization of the backbone.",
            "",
        ]
    )
    (ROOT / "RESULTS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
