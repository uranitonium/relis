#!/usr/bin/env python3
"""Regenerate camera-stabilized L1 while preserving raw weapon-arm identity."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-processed-dir", required=True)
    parser.add_argument("--stabilized-processed-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--v2-pipeline-dir", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--subdirs", default="left,right")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    pipeline_dir = Path(args.v2_pipeline_dir).resolve()
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    from l1_stage import (  # type: ignore
        FEATURE_NAMES,
        L1Config,
        _build_batch_center_hint_map,
        build_joint_map,
        compute_l1_for_fencer,
        load_processed_json,
        save_l1_json,
    )

    raw_root = Path(args.raw_processed_dir).resolve()
    stable_root = Path(args.stabilized_processed_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    auto_config = L1Config(
        fps=args.fps,
        weapon_side="auto",
        weapon_auto_mode="first_center",
        weapon_ref_frames=15,
        use_normalized=False,
    )
    raw_hints = _build_batch_center_hint_map(
        str(raw_root), auto_config, args.max_frames
    )
    stable_hints = _build_batch_center_hint_map(
        str(stable_root), auto_config, args.max_frames
    )
    report = {
        "method": "camera_stabilized_l1_with_raw_weapon_arm_v1",
        "raw_processed_dir": str(raw_root),
        "stabilized_processed_dir": str(stable_root),
        "clips": {},
    }

    for subdir in [value.strip() for value in args.subdirs.split(",") if value.strip()]:
        prefer_inward_sign = -1 if subdir == "right" else 1
        for raw_path in sorted((raw_root / subdir).glob("*_pose_processed.json")):
            stable_path = stable_root / subdir / raw_path.name
            if not stable_path.exists():
                raise FileNotFoundError(stable_path)
            raw_xy, raw_conf, names, raw_frames, raw_anchor = load_processed_json(
                str(raw_path), use_normalized=False
            )
            (
                stable_xy,
                stable_conf,
                stable_names,
                stable_frames,
                stable_anchor,
            ) = load_processed_json(str(stable_path), use_normalized=False)
            if names != stable_names or not np.array_equal(raw_frames, stable_frames):
                raise ValueError(f"Raw/stabilized alignment mismatch: {raw_path.name}")
            if args.max_frames:
                limit = min(args.max_frames, len(raw_frames))
                raw_xy = raw_xy[:limit]
                raw_conf = raw_conf[:limit]
                raw_frames = raw_frames[:limit]
                stable_xy, stable_conf = stable_xy[:limit], stable_conf[:limit]
                raw_anchor = raw_anchor[:limit] if raw_anchor is not None else None
                stable_anchor = stable_anchor[:limit] if stable_anchor is not None else None

            joint_map = build_joint_map(names)
            _, _, raw_debug = compute_l1_for_fencer(
                raw_xy,
                raw_conf,
                args.fps,
                None,
                joint_map,
                auto_config,
                raw_xy=raw_anchor,
                center_x_hint=raw_hints.get(raw_path.name),
                prefer_inward_sign=prefer_inward_sign,
                return_debug=True,
            )
            _, _, stable_auto_debug = compute_l1_for_fencer(
                stable_xy,
                stable_conf,
                args.fps,
                None,
                joint_map,
                auto_config,
                raw_xy=stable_anchor,
                center_x_hint=stable_hints.get(raw_path.name),
                prefer_inward_sign=prefer_inward_sign,
                return_debug=True,
            )
            if raw_debug is None or stable_auto_debug is None:
                raise RuntimeError("Weapon-arm debug output is unavailable")
            raw_side = str(raw_debug["weapon_side"][0])
            stable_auto_side = str(stable_auto_debug["weapon_side"][0])
            fixed_config = dataclasses.replace(auto_config, weapon_side=raw_side)
            features, mask, _ = compute_l1_for_fencer(
                stable_xy,
                stable_conf,
                args.fps,
                None,
                joint_map,
                fixed_config,
                raw_xy=stable_anchor,
                center_x_hint=raw_hints.get(raw_path.name),
                prefer_inward_sign=prefer_inward_sign,
                return_debug=False,
            )
            output = out_root / subdir / raw_path.name.replace(
                "_pose_processed.json", "_l1.json"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            save_l1_json(
                str(output),
                features[:, None, :],
                mask[:, None],
                raw_frames,
                FEATURE_NAMES,
                meta={
                    "fps": args.fps,
                    "source": str(stable_path),
                    "weapon_side": raw_side,
                    "weapon_side_source": str(raw_path),
                },
            )
            processed_suffix = "_pose_processed.json"
            clip_id = raw_path.name[: -len(processed_suffix)]
            report["clips"].setdefault(clip_id, {})[subdir] = {
                "raw_weapon_side": raw_side,
                "stabilized_auto_weapon_side": stable_auto_side,
                "auto_selection_changed": raw_side != stable_auto_side,
            }

    changed = sum(
        any(side["auto_selection_changed"] for side in clip.values())
        for clip in report["clips"].values()
    )
    report["summary"] = {
        "clips": len(report["clips"]),
        "clips_with_changed_auto_selection": changed,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
