#!/usr/bin/env python3
"""Add evaluation ranges and reopen annotations for exchange-end review."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--preserve-confirmed", action="store_true")
    args = parser.parse_args()

    count = 0
    reopened = 0
    for path in sorted(Path(args.annotations).glob("*.json")):
        annotation = read_json(path)
        analyzed = annotation.get("analyzed_range") or {}
        if not args.preserve_confirmed or not annotation.get("exchange_end_confirmed"):
            annotation["evaluation_range"] = {
                "start_frame": int(analyzed["start_frame"]),
                "end_frame": int(analyzed["end_frame"]),
            }
            annotation["exchange_end_confirmed"] = False
            flags = annotation.get("quality_flags") or {}
            if not flags.get("not_exchange") and flags.get("valid", True):
                annotation["annotation_complete"] = False
                reopened += 1
        write_json(path, annotation)
        count += 1
    print(f"Prepared {count} annotations for endpoint review; reopened={reopened}")


if __name__ == "__main__":
    main()
