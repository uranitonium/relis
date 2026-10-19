#!/usr/bin/env python3
"""Validate returned consensus adjudications and optionally import them."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from annotation_schema import normalize_annotation


ALLOWED_CATEGORIES = {
    "boundary_offset",
    "missed_or_extra_state",
    "active_stable_interpretation",
    "pose_or_tracking_issue",
    "invalid_clip",
    "no_substantive_disagreement",
}


def files_from(path: Path) -> tuple[list[Path], Path | None]:
    if path.suffix.lower() != ".zip":
        return sorted(path.rglob("*.json")), None
    temp = Path(tempfile.mkdtemp(prefix="adjudication_validate_"))
    with zipfile.ZipFile(path) as archive:
        archive.extractall(temp)
    return sorted((temp / "consensus").glob("*.json")), temp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected", type=int, default=39)
    parser.add_argument("--import-dir", default="")
    parser.add_argument("--allow-missing-adjudicators", action="store_true")
    args = parser.parse_args()
    files, temp = files_from(Path(args.input))
    errors = []
    complete = 0
    for file in files:
        try:
            raw = json.loads(file.read_text())
            ann = normalize_annotation(raw, strict=False)
            adjudication = raw.get("adjudication", {})
            categories = set(adjudication.get("reason_categories", []))
            if ann["annotator_id"] != "adjudicated_consensus":
                errors.append(f"{ann['clip_id']}: wrong annotator_id")
            if categories and not categories <= ALLOWED_CATEGORIES:
                errors.append(f"{ann['clip_id']}: invalid decision category")
            if adjudication.get("initialization_source", "neutral") not in {"neutral", "owner", "collaborator_a", "mixed"}:
                errors.append(f"{ann['clip_id']}: invalid initialization_source")
            if adjudication.get("model_predictions_viewed") is not False:
                errors.append(f"{ann['clip_id']}: model_predictions_viewed must be false")
            if ann["annotation_complete"] and adjudication.get("review_complete"):
                complete += 1
            if args.import_dir and ann["annotation_complete"]:
                out = Path(args.import_dir) / file.name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, out)
        except Exception as exc:
            errors.append(f"{file.name}: {exc}")
    if temp:
        shutil.rmtree(temp)
    print(f"consensus files={len(files)} complete={complete} expected={args.expected}")
    for error in errors:
        print("ERROR", error)
    if errors or complete != args.expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
