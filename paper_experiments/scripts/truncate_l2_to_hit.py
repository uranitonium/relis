#!/usr/bin/env python3
"""Create L2 sequences and a manifest ending at each detected hit frame."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_csv, read_json, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="paper_experiments/manifests/foil_main.csv")
    parser.add_argument("--out-dir", default="paper_experiments/experiments/l2_hit_truncated")
    parser.add_argument("--out-manifest", default="paper_experiments/manifests/foil_main_hit_truncated.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in read_csv(args.manifest):
        if row.get("hit_frame", "") == "":
            raise ValueError(f"{row['clip_id']}: manifest has no hit_frame")
        hit_frame = int(float(row["hit_frame"]))
        data = read_json(row["l2_path"])
        frames = [int(frame) for frame in data.get("frames", [])]
        keep = [index for index, frame in enumerate(frames) if frame <= hit_frame]
        if not keep:
            raise ValueError(f"{row['clip_id']}: no L2 frames at or before hit frame {hit_frame}")
        last = keep[-1] + 1
        truncated = dict(data)
        for key in ("l2", "l2_mask", "frames"):
            truncated[key] = data[key][:last]
        truncated.setdefault("meta", {})
        truncated["meta"] = dict(truncated["meta"])
        truncated["meta"].update(
            {
                "evaluation_end_frame": hit_frame,
                "evaluation_end_source": "automatic_scoring_light",
                "original_sequence_length": len(data.get("l2", [])),
            }
        )
        output_path = out_dir / f"{row['clip_id']}_l2.json"
        write_json(output_path, truncated)
        output_row = dict(row)
        output_row["l2_path"] = str(output_path)
        output_row["frame_end"] = truncated["frames"][-1]
        output_row["num_frames"] = len(truncated["frames"])
        output_rows.append(output_row)

    write_csv(args.out_manifest, output_rows)
    print(f"Wrote {len(output_rows)} hit-truncated L2 sequences to {out_dir}")
    print(f"Wrote manifest: {args.out_manifest}")


if __name__ == "__main__":
    main()
