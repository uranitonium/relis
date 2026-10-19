#!/usr/bin/env python3
"""Import returned annotation zips or folders into annotations/raw."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from annotation_schema import normalize_annotation
from common import read_json, write_json


def iter_annotation_files(path: Path):
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="ann_import_"))
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmp)
        yield from sorted(tmp.rglob("*.json"))
        shutil.rmtree(tmp, ignore_errors=True)
    elif path.is_dir():
        yield from sorted(path.rglob("*.json"))
    elif path.is_file():
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Import annotation JSON files.")
    parser.add_argument("--input", required=True, help="annotations.zip, folder, or JSON")
    parser.add_argument("--out-dir", default="paper_experiments/annotations/raw")
    parser.add_argument("--annotator", default="", help="Override annotator_id")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    count = 0
    for f in iter_annotation_files(Path(args.input)):
        if f.name == "manifest.json":
            continue
        try:
            ann = normalize_annotation(read_json(f), strict=False)
        except Exception as exc:
            print(f"skip {f}: {exc}")
            continue
        if args.annotator:
            ann["annotator_id"] = args.annotator
        annotator = ann.get("annotator_id") or "unknown"
        out = out_dir / str(annotator) / f"{ann['clip_id']}.json"
        write_json(out, ann)
        count += 1
    print(f"Imported {count} annotation files into {out_dir}")


if __name__ == "__main__":
    main()

