#!/usr/bin/env python3
"""Build a self-contained package for human-only consensus adjudication."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "adjudication_tool" / "run_adjudicator.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def disagreement_priority(row: dict[str, str]) -> float:
    """Return the human-only disagreement score used to order adjudication."""
    stored = row.get("adjudication_priority", "").strip()
    if stored:
        return float(stored)
    return (1.0 - float(row["activity_frame_agreement"])) + (
        1.0 - float(row["boundary_f1_pm5"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default="paper_experiments/experiments/final_boundary_evaluation/adjudication_queue.csv")
    parser.add_argument("--manifest", default="paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv")
    parser.add_argument("--raw-root", default="paper_experiments/annotations/raw")
    parser.add_argument("--out-dir", default="paper_experiments/adjudication_packages/foil_owner_collaborator_consensus")
    parser.add_argument("--zip-out", default="paper_experiments/adjudication_packages/foil_owner_collaborator_consensus.zip")
    args = parser.parse_args()

    queue = read_csv(Path(args.queue))
    for row in queue:
        row["adjudication_priority"] = str(disagreement_priority(row))
    queue.sort(
        key=lambda row: (-float(row["adjudication_priority"]), row["clip_id"])
    )
    manifest = {row["clip_id"]: row for row in read_csv(Path(args.manifest))}
    raw_root = Path(args.raw_root)
    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    for directory in (out / "videos", out / "references" / "owner", out / "references" / "collaborator_a", out / "consensus"):
        directory.mkdir(parents=True, exist_ok=True)

    clips = []
    for rank, metric in enumerate(queue, 1):
        clip_id = metric["clip_id"]
        row = manifest[clip_id]
        owner = raw_root / "owner" / f"{clip_id}.json"
        collaborator = raw_root / "collaborator_a" / f"{clip_id}.json"
        if not owner.exists() or not collaborator.exists():
            raise FileNotFoundError(f"missing reference annotation for {clip_id}")
        video = Path(row["video_path"])
        if not video.exists():
            raise FileNotFoundError(video)
        shutil.copy2(video, out / "videos" / video.name)
        shutil.copy2(owner, out / "references" / "owner" / owner.name)
        shutil.copy2(collaborator, out / "references" / "collaborator_a" / collaborator.name)
        clips.append(
            {
                "clip_id": clip_id,
                "weapon": row.get("weapon", "foil"),
                "outcome_label": row.get("outcome_label"),
                "video_file": video.name,
                "fps": float(row.get("fps", 25)),
                "frame_start": int(float(row["frame_start"])),
                "frame_end": int(float(row["frame_end"])),
                "hit_frame": int(float(row["hit_frame"])),
                "evaluation_end_source": row.get("evaluation_end_source", "automatic_scoring_light"),
                "priority_rank": rank,
                "adjudication_priority": float(metric["adjudication_priority"]),
                "activity_frame_agreement": float(metric["activity_frame_agreement"]),
                "boundary_f1_pm5": float(metric["boundary_f1_pm5"]),
                "collaborator_states": int(float(metric["states_a"])),
                "owner_states": int(float(metric["states_b"])),
                "collaborator_flags": metric.get("collaborator_flags", ""),
                "owner_flags": metric.get("owner_flags", ""),
            }
        )

    shutil.copy2(TOOL, out / "run_adjudicator.py")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "human_consensus_adjudication",
                "reference_annotators": ["collaborator_a", "owner"],
                "model_predictions_included": False,
                "order": "descending human-only disagreement priority",
                "clips": clips,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    clip_count = len(clips)
    (out / "README_START_HERE.md").write_text(
        f"""# Foil Consensus Adjudication Package

This package contains {clip_count} clips completed independently by both the owner and collaborator. It contains no model predictions.

## Run

```bash
python3 run_adjudicator.py
```

Open `http://127.0.0.1:8766`.

## Procedure

Both adjudicators should review each clip together, ideally by screen sharing or in one meeting.

1. Watch the video and compare the immutable collaborator and owner timelines.
2. Create a consensus timeline from the neutral one-state starting point.
   Click either human segment to seek to it. Drag useful segments onto the
   consensus row, or select one and click **Apply selected reference segment**.
   Applying a reference segment replaces only that exact frame interval.
3. Optionally select decision categories, enter names or initials, and add a
   short note when that metadata would be useful later.
4. Mark the clip complete.

Do not open model predictions until the consensus export is finalized. Original reference JSON files are read-only and are never overwritten by the tool.

Review is ordered by human disagreement, highest first. Complete all {clip_count} clips so the consensus evaluation uses a fixed, fully adjudicated cohort.

Click **Download consensus.zip** when finished and return that file. The export contains consensus JSON files and `adjudication_audit.csv`.
""",
        encoding="utf-8",
    )

    zip_out = Path(args.zip_out)
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    if zip_out.exists():
        zip_out.unlink()
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(out.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(out.parent))
    print(f"Wrote {zip_out} with {clip_count} clips")


if __name__ == "__main__":
    main()
