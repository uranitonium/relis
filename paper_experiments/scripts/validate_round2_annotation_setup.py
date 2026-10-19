#!/usr/bin/env python3
"""Validate the frozen Round 2 annotation manifests and package zips."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path("paper_experiments/round2/annotation_extension")
PACKAGES = ROOT / "packages"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_zip_manifest(path: Path) -> tuple[dict, set[str]]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        manifests = [name for name in names if name.endswith("/manifest.json")]
        if len(manifests) != 1:
            raise AssertionError(f"{path}: expected one manifest, found {manifests}")
        manifest = json.loads(archive.read(manifests[0]))
        prefix = manifests[0][: -len("manifest.json")]
        for clip in manifest.get("clips", []):
            expected_video = prefix + "videos/" + clip["video_file"]
            if expected_video not in names:
                raise AssertionError(f"{path}: missing {expected_video}")
        return manifest, names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    coverage = json.loads((ROOT / "reports" / "coverage_report.json").read_text())
    master_rows = read_csv(ROOT / "manifests" / "foil_round2_exchange_disjoint_all_124.csv")
    master_ids = {row["clip_id"] for row in master_rows}
    check(len(master_rows) == len(master_ids) == 124, "master manifest must contain 124 unique clips")
    check(coverage["inventory"]["available_source_ids"] == 89, "coverage report must record 89 available source IDs")
    check(coverage["combined_annotation_coverage_after_extension"]["source_ids"] == 85, "combined coverage must be 85 source IDs")
    for key in ("extension_vs_train", "extension_vs_development", "extension_vs_existing_annotations"):
        check(coverage["overlap_checks"][key] == 0, f"coverage overlap check failed: {key}")

    part_union: set[str] = set()
    for part in range(1, 5):
        part_path = ROOT / "manifests" / f"foil_round2_exchange_disjoint_part_{part:02d}.csv"
        part_rows = read_csv(part_path)
        part_ids = {row["clip_id"] for row in part_rows}
        check(len(part_rows) == len(part_ids) == 31, f"part {part:02d} must contain 31 unique clips")
        check(not (part_union & part_ids), f"part {part:02d} overlaps an earlier part")
        part_union |= part_ids

        packaged: dict[str, set[str]] = {}
        for annotator in ("owner", "collaborator_a"):
            package = PACKAGES / f"foil_round2_exchange_disjoint_part_{part:02d}_{annotator}.zip"
            check(package.exists(), f"missing package {package}")
            if not package.exists():
                continue
            manifest, names = read_zip_manifest(package)
            ids = {clip["clip_id"] for clip in manifest["clips"]}
            packaged[annotator] = ids
            check(manifest.get("annotator_id") == annotator, f"{package}: wrong annotator ID")
            check(ids == part_ids, f"{package}: clips differ from part manifest")
            check(len(ids) == 31, f"{package}: expected 31 clips")
            forbidden = [name for name in names if "prediction" in name.lower() or "score_cache" in name.lower()]
            check(not forbidden, f"{package}: contains model/evaluation artifacts {forbidden}")
            annotation_json = [name for name in names if "/annotations/" in name and name.endswith(".json")]
            check(not annotation_json, f"{package}: contains prefilled annotations")
        check(packaged.get("owner") == packaged.get("collaborator_a"), f"part {part:02d}: annotator clip sets differ")

    check(part_union == master_ids, "four part manifests do not exactly partition the master manifest")

    catchup_path = PACKAGES / "foil_round2_collaborator_catchup_1_collaborator_a.zip"
    catchup, catchup_names = read_zip_manifest(catchup_path)
    check(catchup.get("annotator_id") == "collaborator_a", "catch-up package has wrong annotator")
    check([clip["clip_id"] for clip in catchup["clips"]] == ["N10-0-2"], "catch-up package must contain only N10-0-2")
    check(not [name for name in catchup_names if "prediction" in name.lower()], "catch-up package contains predictions")

    adjudication_path = PACKAGES / "foil_round2_existing_42_consensus.zip"
    adjudication, adjudication_names = read_zip_manifest(adjudication_path)
    adjudication_ids = {clip["clip_id"] for clip in adjudication["clips"]}
    queue_ids = {row["clip_id"] for row in read_csv(ROOT / "adjudication" / "existing_completed_pairs_42_queue.csv")}
    check(len(adjudication_ids) == len(queue_ids) == 42, "adjudication package/queue must contain 42 clips")
    check(adjudication_ids == queue_ids, "adjudication package differs from queue")
    for clip_id in adjudication_ids:
        for annotator in ("owner", "collaborator_a"):
            suffix = f"/references/{annotator}/{clip_id}.json"
            check(any(name.endswith(suffix) for name in adjudication_names), f"adjudication package missing {annotator}/{clip_id}")
    forbidden_adjudication = [name for name in adjudication_names if "prediction" in name.lower() or "score_cache" in name.lower()]
    check(not forbidden_adjudication, "adjudication package contains model/evaluation artifacts")

    expected_hashes: dict[str, str] = {}
    for line in (ROOT / "PACKAGE_SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            expected_hashes[name] = digest
    actual_packages = {path.name for path in PACKAGES.glob("*.zip")}
    check(actual_packages == set(expected_hashes), "package set differs from checksum inventory")
    for name, expected in expected_hashes.items():
        package = PACKAGES / name
        if package.exists():
            check(sha256(package) == expected, f"checksum mismatch: {name}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Round 2 annotation setup validated")
    print("extension clips=124 parts=4x31 annotators=2")
    print("existing adjudication clips=42 catch-up clips=1")
    print("available source IDs=89 combined annotation coverage after extension=85")


if __name__ == "__main__":
    main()
