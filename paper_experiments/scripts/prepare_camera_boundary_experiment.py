#!/usr/bin/env python3
"""Prepare linked inputs, manifests, and paired L2 views for camera experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


REPO = Path(__file__).resolve().parents[2]
V2_DATA_ROOT = Path(os.getenv("FENCING_V2_DATA_ROOT", str(REPO.parent / "V2"))).resolve()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def ensure_link(target: Path, source: Path) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source:
        return
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"Refusing to replace unexpected path: {target}")
    target.symlink_to(source)


def prepare_links(args: argparse.Namespace) -> None:
    rows = read_csv(Path(args.manifest))
    workspace = Path(args.workspace).resolve()
    processed = Path(args.processed_dir).resolve()
    selected = []
    for row in rows:
        clip_id = row["clip_id"]
        video = resolve_repo_path(row["video_path"])
        if not video.exists() and args.video_fallback_dir:
            video = Path(args.video_fallback_dir).resolve() / f"{clip_id}_pose.mp4"
        ensure_link(workspace / "clips" / f"{clip_id}.mp4", video)
        for side in ("left", "right"):
            pose = processed / side / f"{clip_id}_pose_processed.json"
            ensure_link(workspace / "processed_raw" / side / pose.name, pose)
        selected.append({"clip_id": clip_id})
    workspace.mkdir(parents=True, exist_ok=True)
    selection = workspace / "selection.json"
    selection.write_text(json.dumps({"selected": selected}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"clips": len(rows), "workspace": str(workspace)}, indent=2))


def rewrite_manifest(args: argparse.Namespace) -> None:
    rows = read_csv(Path(args.manifest))
    l2_dir = Path(args.l2_dir).resolve()
    output = []
    for row in rows:
        updated = dict(row)
        updated["l2_path"] = str(l2_dir / f"{row['clip_id']}_l2.json")
        output.append(updated)
    write_csv(Path(args.out_manifest), output)
    print(json.dumps({"clips": len(output), "manifest": str(Path(args.out_manifest).resolve())}, indent=2))


def fuse_l2(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir).resolve()
    stabilized_dir = Path(args.stabilized_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(raw_dir.glob("*_l2.json"))
    if not raw_files:
        raise RuntimeError(f"No L2 files found in {raw_dir}")
    for raw_path in raw_files:
        stabilized_path = stabilized_dir / raw_path.name
        if not stabilized_path.exists():
            raise FileNotFoundError(stabilized_path)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        stabilized = json.loads(stabilized_path.read_text(encoding="utf-8"))
        raw_l2 = raw["l2"]
        stabilized_l2 = stabilized["l2"]
        if len(raw_l2) != len(stabilized_l2):
            raise ValueError(f"Length mismatch for {raw_path.name}")
        raw_mask = raw["l2_mask"]
        stabilized_mask = stabilized["l2_mask"]
        if len(raw_mask) != len(stabilized_mask):
            raise ValueError(f"Mask length mismatch for {raw_path.name}")
        fused = dict(raw)
        fused["l2"] = [left + right for left, right in zip(raw_l2, stabilized_l2)]
        fused["l2_mask"] = [bool(left and right) for left, right in zip(raw_mask, stabilized_mask)]
        fused["feature_names"] = [f"raw/{name}" for name in raw["feature_names"]] + [
            f"stabilized/{name}" for name in stabilized["feature_names"]
        ]
        fused["meta"] = dict(raw.get("meta", {}))
        fused["meta"]["camera_views"] = {
            "raw": str(raw_path),
            "stabilized": str(stabilized_path),
            "fusion": "feature_concatenation",
        }
        (out_dir / raw_path.name).write_text(json.dumps(fused, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"clips": len(raw_files), "out_dir": str(out_dir)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    links = subparsers.add_parser("links", help="Link videos and processed poses for one manifest")
    links.add_argument("--manifest", required=True)
    links.add_argument("--workspace", required=True)
    links.add_argument("--processed-dir", default=str(V2_DATA_ROOT / "processed_joints_v3"))
    links.add_argument("--video-fallback-dir", default=str(V2_DATA_ROOT / "training_highfps"))
    links.set_defaults(func=prepare_links)

    manifest = subparsers.add_parser("manifest", help="Point a copy of a manifest at an L2 directory")
    manifest.add_argument("--manifest", required=True)
    manifest.add_argument("--l2-dir", required=True)
    manifest.add_argument("--out-manifest", required=True)
    manifest.set_defaults(func=rewrite_manifest)

    fuse = subparsers.add_parser("fuse", help="Concatenate raw and stabilized L2 feature views")
    fuse.add_argument("--raw-dir", required=True)
    fuse.add_argument("--stabilized-dir", required=True)
    fuse.add_argument("--out-dir", required=True)
    fuse.set_defaults(func=fuse_l2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
