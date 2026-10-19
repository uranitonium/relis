#!/usr/bin/env python3
"""Create a self-contained annotation package zip."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from common import copy_or_link, read_csv, write_json


HERE = Path(__file__).resolve().parents[1]
ANNOTATOR_SCRIPT = HERE / "annotation_tool" / "run_annotator.py"


def build_package(rows: List[Dict[str, str]], out_dir: Path, annotator: str, split: str, copy_mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    manifest_clips: List[Dict[str, Any]] = []
    for row in rows:
        src = Path(row["video_path"])
        if not src.exists():
            print(f"warning: missing video for {row.get('clip_id')}: {src}")
            continue
        dst = videos_dir / src.name
        copy_or_link(src, dst, mode=copy_mode)
        manifest_clips.append(
            {
                "clip_id": row["clip_id"],
                "weapon": row.get("weapon", "foil"),
                "outcome_label": row.get("outcome_label", ""),
                "video_file": src.name,
                "fps": float(row.get("fps", 25.0)),
                "frame_start": int(float(row.get("frame_start", 0))),
                "frame_end": int(float(row.get("frame_end", 0))),
                "num_frames": int(float(row.get("num_frames", 0))),
                "hit_frame": int(float(row["hit_frame"])) if row.get("hit_frame", "") != "" else None,
                "evaluation_end_source": row.get("evaluation_end_source", ""),
            }
        )
    shutil.copy2(ANNOTATOR_SCRIPT, out_dir / "run_annotator.py")
    write_json(
        out_dir / "manifest.json",
        {
            "split": split,
            "annotator_id": annotator,
            "annotation_task": "interaction_state_partition",
            "schema_version": 2,
            "coordinate_system": "source_video_frame_inclusive",
            "clips": manifest_clips,
        },
    )
    (out_dir / "README_START_HERE.md").write_text(
        f"""# Interaction State Annotation Package

Annotator: `{annotator}`
Split: `{split}`
Clips: `{len(manifest_clips)}`

## Start

```bash
python3 run_annotator.py
```

If your system uses `python` instead of `python3`, this is also fine:

```bash
python run_annotator.py
```

Open the printed local URL, usually:

```text
http://127.0.0.1:8765
```

## Task

Partition each analyzed exchange into consecutive interaction states.

Complete this package independently. Do not inspect the other annotator's
labels, model predictions, or evaluation results, and do not discuss individual
clips with the other annotator until both annotation exports are frozen.

1. Watch the full analyzed interval once.
2. Add a cut at the first frame of each new interaction state.
3. Label every resulting state as `stable` or `active` relative to the exchange.
4. Mark the clip complete after reviewing the state partition through the automatic hit frame.

A new state begins when the bilateral interaction changes pattern: preparation changes into committed development, initiative changes, retreat becomes a stop or reversal, distance control changes meaningfully, or one active pattern changes into another. Two `active` states may be directly adjacent. Do not force a stable interval between them.

Do not add cuts for every step, ordinary motion within the same interaction pattern, pose jitter, or the score light. The model output is hidden during primary annotation.

### Exchange End

The scoring-light frame was detected automatically from the source clip. The
gray timeline tail after that frame is excluded from annotation and evaluation.
Do not label disengagement, celebration, referee response, or walkback.

Use flags for low-quality clips:

- `valid`: keep enabled unless the clip should be excluded.
- `pose_issue`: tracking/pose overlay makes judgment unreliable.
- `ambiguous`: the state partition is genuinely hard to define.
- `not_exchange`: clip is not a usable fencing exchange.
- `needs_review`: ask for adjudication.

Keyboard shortcuts: `A`/`D` frame step, `S` split, `1` stable, `2` active, and `Space` play/pause.

When finished, click **Download annotations.zip** and send that zip back. Old transition-zone annotations are not compatible with this package.
""",
        encoding="utf-8",
    )


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path == zip_path:
                continue
            zf.write(path, path.relative_to(src_dir.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare self-contained annotation package.")
    parser.add_argument("--manifest", default="paper_experiments/manifests/foil_calibration.csv")
    parser.add_argument("--split", default="foil_calibration")
    parser.add_argument("--annotator", default="collaborator_a")
    parser.add_argument("--out-root", default="paper_experiments/annotation_packages")
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    out_dir = Path(args.out_root) / f"{args.split}_{args.annotator}"
    build_package(rows, out_dir, args.annotator, args.split, args.copy_mode)
    if not args.no_zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"Wrote package: {zip_path}")
    print(f"Package folder: {out_dir}")


if __name__ == "__main__":
    main()
