#!/usr/bin/env python3
"""Build manifests that point to full-length post-hit L2 sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_csv, read_sequence, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="paper_experiments/experiments/truncated_retraining/manifests/paper_final_candidates.csv")
    parser.add_argument("--l2-root", default="old/old2/l2_out_raw_firstcenter_boundary_full")
    parser.add_argument("--out", default="paper_experiments/experiments/untrimmed_recovery/final_untrimmed.csv")
    args = parser.parse_args()

    root = Path(args.l2_root)
    rows = []
    for row in read_csv(args.manifest):
        path = root / f"{row['clip_id']}_l2.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data = read_sequence(path)
        frames = [int(frame) for frame in data["frames"]]
        out = dict(row)
        out["l2_path"] = str(path)
        out["frame_start"] = frames[0]
        out["frame_end"] = frames[-1]
        out["num_frames"] = len(frames)
        out["evaluation_end_source"] = "untrimmed_video_end"
        rows.append(out)

    write_csv(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
