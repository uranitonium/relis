#!/usr/bin/env python3
"""Export foil paper tables and notes from frozen experiment outputs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from common import read_csv, read_json, write_csv, write_json


BOUNDARY_ROOT = Path("paper_experiments/experiments/final_boundary_evaluation/boundary_suite")
MANIFEST_DEV = Path("paper_experiments/experiments/truncated_retraining/manifests/paper_development.csv")
MANIFEST_FINAL = Path("paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv")
OWNER_DIR = Path("paper_experiments/annotations/raw/owner")
COLLAB_DIR = Path("paper_experiments/annotations/raw/collaborator_a")
CONSENSUS_DIR = Path("paper_experiments/annotations/adjudicated/consensus")


METHOD_NAMES = {
    "uniform": "Uniform windows",
    "motion": "Motion-energy threshold",
    "changepoint": "L2 change-point",
    "kernel_changepoint": "Kernel change-point",
    "selfsim": "Self-similarity novelty",
    "rgb_resnet": "RGB temporal model",
    "video_r3d": "R3D video temporal model",
    "video_videomae": "VideoMAE temporal model",
    "pose_stgcn": "Two-skeleton ST-GCN",
    "random_matched_conservative": "Random matched",
    "engineered_l2_instability": "Engineered L2 instability",
    "raw_phi_l2_conservative": "Instability magnitude phi",
    "no_critical_conservative": "Conservative, no critical cues",
    "no_critical_optimistic": "Primary, no critical cues",
    "conservative": "Conservative configuration",
    "optimistic": "Primary configuration",
}


def f3(value: Any) -> str:
    return f"{float(value):.3f}"


def read_annotations(directory: Path) -> list[dict[str, Any]]:
    anns = []
    for path in sorted(directory.glob("*.json")):
        anns.append(read_json(path))
    return anns


def quality_flags(annotation: dict[str, Any]) -> list[str]:
    flags = annotation.get("quality_flags", {})
    return [name for name, enabled in flags.items() if enabled and name != "valid"]


def is_valid(annotation: dict[str, Any]) -> bool:
    flags = annotation.get("quality_flags", {})
    return bool(annotation.get("annotation_complete")) and bool(flags.get("valid", True)) and not flags.get("not_exchange")


def is_clean(annotation: dict[str, Any]) -> bool:
    return is_valid(annotation) and not quality_flags(annotation)


def active_coverage(annotation: dict[str, Any]) -> float:
    segments = annotation.get("segments", [])
    if not segments:
        return 0.0
    lo = int(segments[0]["start_frame"])
    hi = int(segments[-1]["end_frame"])
    total = max(1, hi - lo + 1)
    active = sum(
        int(segment["end_frame"]) - int(segment["start_frame"]) + 1
        for segment in segments
        if segment.get("activity") == "active"
    )
    return active / total


def annotation_stats(name: str, directory: Path) -> dict[str, object]:
    anns = read_annotations(directory)
    valid = [ann for ann in anns if is_valid(ann)]
    clean = [ann for ann in anns if is_clean(ann)]
    outcomes = Counter(str(ann.get("outcome_label", "")) for ann in valid)
    flags = Counter(flag for ann in anns for flag in quality_flags(ann))
    state_counts = [len(ann.get("segments", [])) for ann in valid]
    coverages = [active_coverage(ann) for ann in valid]
    return {
        "label_set": name,
        "json_files": len(anns),
        "valid_exchanges": len(valid),
        "clean_exchanges": len(clean),
        "left": outcomes.get("Left", 0),
        "right": outcomes.get("Right", 0),
        "none": outcomes.get("None", 0),
        "not_exchange": flags.get("not_exchange", 0),
        "pose_issue": flags.get("pose_issue", 0),
        "ambiguous": flags.get("ambiguous", 0),
        "needs_review": flags.get("needs_review", 0),
        "mean_states": sum(state_counts) / len(state_counts) if state_counts else 0.0,
        "mean_active_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
    }


def manifest_stats(name: str, path: Path) -> dict[str, object]:
    rows = read_csv(path)
    labels = Counter(row["outcome_label"] for row in rows)
    prefixes = Counter(row["prefix"] for row in rows)
    return {
        "manifest": name,
        "clips": len(rows),
        "left": labels.get("Left", 0),
        "right": labels.get("Right", 0),
        "none": labels.get("None", 0),
        "prefix_L": prefixes.get("L", 0),
        "prefix_R": prefixes.get("R", 0),
        "prefix_N": prefixes.get("N", 0),
        "prefix_T": prefixes.get("T", 0),
    }


def select_boundary_rows(rows: list[dict[str, str]], reference: str, cohort: str, methods: list[str]) -> list[dict[str, object]]:
    by_method = {
        row["method"]: row
        for row in rows
        if row["reference"] == reference and row["cohort"] == cohort
    }
    out = []
    for method in methods:
        row = by_method[method]
        out.append(
            {
                "method": METHOD_NAMES.get(method, method),
                "clips": int(float(row["clips"])),
                "activity_accuracy": f3(row["activity_accuracy"]),
                "activity_macro_f1": f3(row["activity_macro_f1"]),
                "active_f1": f3(row["active_f1"]),
                "boundary_f1_pm3": f3(row["boundary_f1_pm3"]),
                "boundary_f1_pm5": f3(row["boundary_f1_pm5"]),
                "boundary_f1_pm10": f3(row["boundary_f1_pm10"]),
                "model_states": f3(row["model_states"]),
                "model_active_coverage": f3(row["model_active_coverage"]),
            }
        )
    return out


def markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def write_table(out_dir: Path, stem: str, rows: list[dict[str, object]]) -> None:
    write_csv(out_dir / f"{stem}.csv", rows)
    (out_dir / f"{stem}.md").write_text(markdown_table(rows), encoding="utf-8")


def bootstrap_key_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    keep_pairs = {
        ("optimistic", "conservative"),
        ("optimistic", "random_matched_conservative"),
        ("optimistic", "motion"),
        ("optimistic", "selfsim"),
        ("optimistic", "pose_stgcn"),
        ("optimistic", "rgb_resnet"),
        ("optimistic", "video_r3d"),
        ("optimistic", "video_videomae"),
        ("optimistic", "changepoint"),
        ("optimistic", "engineered_l2_instability"),
        ("optimistic", "raw_phi_l2_conservative"),
        ("optimistic", "no_critical_optimistic"),
    }
    keep_metrics = {"activity_accuracy", "active_f1", "boundary_f1_pm5", "boundary_f1_pm10"}
    out = []
    for row in rows:
        pair = (row["candidate"], row["baseline"])
        if row["reference"] != "consensus" or row["cohort"] != "valid":
            continue
        if pair not in keep_pairs or row["metric"] not in keep_metrics:
            continue
        out.append(
            {
                "candidate": METHOD_NAMES.get(row["candidate"], row["candidate"]),
                "baseline": METHOD_NAMES.get(row["baseline"], row["baseline"]),
                "metric": row["metric"],
                "n": int(float(row["n"])),
                "mean_diff": f3(row["mean_diff"]),
                "ci_low": f3(row["ci_low"]),
                "ci_high": f3(row["ci_high"]),
                "p_gt_zero": f3(row["p_gt_zero"]),
            }
        )
    return out


def qualitative_candidates(boundary_root: Path, reference: str = "consensus", method: str = "optimistic") -> list[dict[str, object]]:
    rows = read_csv(boundary_root / "results" / reference / method / "by_clip.csv")
    usable = [row for row in rows if str(row.get("valid", "")).lower() == "true"]
    for row in usable:
        row["_score"] = (
            float(row["activity_accuracy"])
            + float(row["active_f1"])
            + float(row["boundary_f1_pm5"])
            + float(row["boundary_f1_pm10"])
        )
    strong = sorted(usable, key=lambda row: float(row["_score"]), reverse=True)[:10]
    activity_fail = sorted(usable, key=lambda row: float(row["active_f1"]))[:10]
    boundary_disagree = sorted(
        [row for row in usable if float(row["activity_accuracy"]) >= 0.75],
        key=lambda row: float(row["boundary_f1_pm5"]),
    )[:10]
    tagged: list[tuple[str, dict[str, str]]] = []
    tagged.extend(("strong_success", row) for row in strong)
    tagged.extend(("activity_failure", row) for row in activity_fail)
    tagged.extend(("boundary_disagreement_candidate", row) for row in boundary_disagree)
    out = []
    seen = set()
    for category, row in tagged:
        key = (category, row["clip_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "category": category,
                "clip_id": row["clip_id"],
                "clean": row["clean"],
                "flags": row["flags"],
                "human_states": row["human_states"],
                "model_states": row["model_states"],
                "activity_accuracy": f3(row["activity_accuracy"]),
                "active_f1": f3(row["active_f1"]),
                "boundary_f1_pm5": f3(row["boundary_f1_pm5"]),
                "boundary_f1_pm10": f3(row["boundary_f1_pm10"]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-root", default=str(BOUNDARY_ROOT))
    parser.add_argument("--out-dir", default="paper_experiments/paper_ready/foil")
    args = parser.parse_args()

    boundary_root = Path(args.boundary_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boundary_rows = read_csv(boundary_root / "boundary_suite_results.csv")
    bootstrap_rows = read_csv(boundary_root / "bootstrap_pairwise_differences.csv")

    main_methods = [
        "random_matched_conservative",
        "motion",
        "selfsim",
        "changepoint",
        "kernel_changepoint",
        "rgb_resnet",
        "video_r3d",
        "video_videomae",
        "pose_stgcn",
        "engineered_l2_instability",
        "raw_phi_l2_conservative",
        "optimistic",
    ]
    ablation_methods = [
        "engineered_l2_instability",
        "raw_phi_l2_conservative",
        "no_critical_conservative",
        "conservative",
        "no_critical_optimistic",
        "optimistic",
    ]
    write_table(out_dir, "boundary_consensus_valid", select_boundary_rows(boundary_rows, "consensus", "valid", main_methods))
    write_table(out_dir, "boundary_consensus_clean", select_boundary_rows(boundary_rows, "consensus", "clean", main_methods))
    write_table(out_dir, "boundary_collaborator_valid", select_boundary_rows(boundary_rows, "collaborator", "valid", main_methods))
    write_table(out_dir, "boundary_single_annotator_valid", select_boundary_rows(boundary_rows, "collaborator", "valid", main_methods))
    write_table(out_dir, "boundary_ablations_consensus_valid", select_boundary_rows(boundary_rows, "consensus", "valid", ablation_methods))
    write_table(out_dir, "boundary_ablations_collaborator_valid", select_boundary_rows(boundary_rows, "collaborator", "valid", ablation_methods))
    write_table(out_dir, "boundary_ablations_single_annotator_valid", select_boundary_rows(boundary_rows, "collaborator", "valid", ablation_methods))
    write_table(out_dir, "bootstrap_key_differences", bootstrap_key_rows(bootstrap_rows))
    write_table(out_dir, "qualitative_candidates_optimistic_consensus", qualitative_candidates(boundary_root))
    slow_ablation_path = Path("paper_experiments/experiments/slow_variable_ablation/slow_variable_dev.csv")
    if slow_ablation_path.exists():
        write_table(out_dir, "slow_variable_dev", read_csv(slow_ablation_path))

    stats = {
        "manifests": [
            manifest_stats("development_tuning", MANIFEST_DEV),
            manifest_stats("final_candidates", MANIFEST_FINAL),
        ],
        "annotations": [
            annotation_stats("owner_raw", OWNER_DIR),
            annotation_stats("collaborator_raw", COLLAB_DIR),
            annotation_stats("adjudicated_consensus", CONSENSUS_DIR),
        ],
        "boundary_suite_root": str(boundary_root),
    }
    write_json(out_dir / "foil_dataset_and_annotation_stats.json", stats)
    write_csv(out_dir / "foil_manifest_stats.csv", stats["manifests"])
    write_csv(out_dir / "foil_annotation_stats.csv", stats["annotations"])

    consensus_stats = annotation_stats("adjudicated_consensus", CONSENSUS_DIR)
    consensus_row = select_boundary_rows(boundary_rows, "consensus", "valid", ["optimistic"])[0]
    conservative_row = select_boundary_rows(boundary_rows, "consensus", "valid", ["conservative"])[0]

    summary = f"""# Foil Pre-Writing Results

Generated from `{boundary_root}`.

## Primary Interpretation

- The adjudicated consensus set has {consensus_stats["valid_exchanges"]} valid clips, with {consensus_stats["clean_exchanges"]} clean clips after removing quality-flagged clips.
- The primary configuration is the strongest framewise activity model on consensus: activity accuracy {consensus_row["activity_accuracy"]} and active F1 {consensus_row["active_f1"]} on all valid consensus clips.
- The conservative configuration remains competitive for coarser boundary localization: boundary F1 +/-10 is {conservative_row["boundary_f1_pm10"]}, versus {consensus_row["boundary_f1_pm10"]} for the primary configuration.
- Both tuned models beat random matched, motion-energy, and generic L2/kernel change-point baselines on the main activity metrics and most boundary metrics.
- The no-critical inference ablation remains strong and sometimes improves boundary F1; describe this carefully as a sensitivity result, not as proof that critical-distance cues are necessary.

## Files

- `boundary_consensus_valid.csv`: main adjudicated result table.
- `boundary_consensus_clean.csv`: same table after excluding pose-issue clips.
- `boundary_collaborator_valid.csv`: larger collaborator-only robustness check.
- `boundary_ablations_consensus_valid.csv`: required ablation table.
- `bootstrap_key_differences.csv`: paired bootstrap intervals for the most important claims.
- `endpoint_shift_sensitivity.csv`: endpoint +/-3/+/-5 frame sensitivity after rebuilding shifted L2 clips and rerunning an alternative ReLIS extractor and raw-instability predictions.
- `foil_dataset_and_annotation_stats.json`: manifest and annotation statistics.
- `qualitative_candidates_optimistic_consensus.csv`: success/failure shortlist for figures and discussion.
"""
    (out_dir / "README.md").write_text(summary, encoding="utf-8")
    print(f"Wrote foil paper assets to {out_dir}")


if __name__ == "__main__":
    main()
