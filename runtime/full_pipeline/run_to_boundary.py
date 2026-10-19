#!/usr/bin/env python3
"""Run the website-current pipeline through boundary segmentation only.

Stages:
  1. mmdetection/mmpose fencer detection, selection, tracking, pose JSON/video
  2. processed joint JSON
  3. L1 per-fencer features
  4. L2 pairwise boundary features
  5. L3 boundary model inference

This intentionally stops at boundary segmentation. It does not run hard tactical
state scoring, LLM context export, OpenAI analysis, L4 inspector, or portal code.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE.parents[1]
BOUNDARY_PACKAGE_ROOT = MODULE_ROOT / "boundary_segmentation"
BOUNDARY_DEPLOYMENT_ROOT = MODULE_ROOT / "deployment"


def resolve_executable(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError("ffmpeg executable not found and imageio_ffmpeg fallback unavailable") from exc
    return name


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    if cmd:
        cmd = [resolve_executable(cmd[0]), *cmd[1:]]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def resolve_ckpt(path_or_dir: Path) -> Path:
    if path_or_dir.is_file():
        return path_or_dir
    preferred = [
        path_or_dir / "ckpt_best_ema.pt",
        path_or_dir / "ckpt_early_stop.pt",
        path_or_dir / "ckpt_last.pt",
    ]
    for path in preferred:
        if path.exists():
            return path
    steps = sorted(path_or_dir.glob("ckpt_step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if steps:
        return steps[0]
    raise FileNotFoundError(f"No boundary checkpoint found under {path_or_dir}")


def discover_videos(work: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    skip = {"pose_overlay_v3", "processed_joints_v3", "legacy_auto", "boundary_preds"}
    found: List[Path] = []
    for root, dirs, files in os.walk(work):
        root_p = Path(root)
        rel = root_p.relative_to(work)
        if rel.parts and rel.parts[0] in skip:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for name in files:
            p = root_p / name
            if p.suffix.lower() in exts:
                found.append(p.resolve())
    return sorted(found)


def transcode_pose_overlays_h264(pose_overlay_dir: Path) -> None:
    for src in sorted(pose_overlay_dir.rglob("*_pose.mp4")):
        dst = src.with_name(src.stem + "_h264.mp4")
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dst),
            ]
        )


def write_manifest(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full_pipeline through L3 boundary segmentation only.")
    parser.add_argument("--input-video", default="", help="Single video path.")
    parser.add_argument("--input-root", default="", help="Root folder scanned recursively for videos.")
    parser.add_argument("--working-dir", default=str(HERE / "working"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=0, help="0 disables truncation.")
    parser.add_argument("--weapon-auto-mode", default="legacy", choices=["legacy", "stable_center", "hybrid", "first_center"])
    parser.add_argument("--l2-toward-mode", default="horizontal", choices=["horizontal", "pair_axis"])
    parser.add_argument("--use-raw", dest="use_raw", action="store_true", default=True)
    parser.add_argument("--no-use-raw", dest="use_raw", action="store_false")
    parser.add_argument("--transcode-pose-h264", dest="transcode_pose_h264", action="store_true", default=True)
    parser.add_argument("--no-transcode-pose-h264", dest="transcode_pose_h264", action="store_false")

    parser.add_argument("--selection-preview-only", action="store_true", help="Write fencer selection preview and stop.")
    parser.add_argument("--selection-frame", type=int, default=5)
    parser.add_argument("--selection-max-frame", type=int, default=15)
    parser.add_argument("--selection-preview-dir", default="", help="Defaults to working/selection_preview.")
    parser.add_argument("--selection-override-json", default="", help="Confirmed left/right bbox override JSON.")
    parser.add_argument("--selection-assist-json", default="", help="Approximate left/right center assist JSON.")
    parser.add_argument("--track-cache-json", default="", help="Optional full tracking cache output path.")
    parser.add_argument("--pose-from-track-cache-json", default="", help="Materialize from prior track cache.")
    parser.add_argument("--pose-left-track-id", type=int, default=None)
    parser.add_argument("--pose-right-track-id", type=int, default=None)

    parser.add_argument("--boundary-config", default=str(BOUNDARY_DEPLOYMENT_ROOT / "checkpoints" / "legacy_auto" / "config.yaml"))
    parser.add_argument("--boundary-ckpt", default=str(BOUNDARY_DEPLOYMENT_ROOT / "checkpoints" / "legacy_auto"))
    parser.add_argument("--detector-cfg", default=str(HERE / "mmdetection" / "configs" / "rtmdet" / "rtmdet_tiny_8xb32-300e_coco.py"))
    parser.add_argument("--detector-wts", default=str(HERE / "weights" / "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"))
    parser.add_argument(
        "--pose-cfg",
        default=str(HERE / "mmpose" / "configs" / "body_2d_keypoint" / "rtmpose" / "coco" / "rtmpose-m_8xb256-420e_coco-256x192.py"),
    )
    parser.add_argument("--pose-wts", default=str(HERE / "weights" / "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"))
    args = parser.parse_args()

    work = Path(args.working_dir).resolve()
    pose_overlay_dir = work / "pose_overlay_v3"
    processed_dir = work / "processed_joints_v3"
    variant_root = work / "legacy_auto"
    l1_out = variant_root / "l1_out"
    l2_out = variant_root / "l2_out"
    boundary_preds = variant_root / "boundary_preds"
    for path in (work, pose_overlay_dir, processed_dir, l1_out, l2_out, boundary_preds):
        path.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    input_video = Path(args.input_video).resolve() if args.input_video else None
    input_root = Path(args.input_root).resolve() if args.input_root else None

    if not input_video and not input_root:
        discovered = discover_videos(work)
        if not discovered:
            raise SystemExit(f"No input videos found under {work}. Pass --input-video or --input-root.")
        input_video = discovered[0]
        print(f"Auto-discovered input video: {input_video}", flush=True)

    if args.selection_preview_only:
        preview_dir = Path(args.selection_preview_dir).resolve() if args.selection_preview_dir else work / "selection_preview"
        pose_cmd = [
            py,
            str(HERE / "pose_json_video_v3.py"),
            "--output-root",
            str(pose_overlay_dir),
            "--selection-preview-only",
            "--selection-frame",
            str(args.selection_frame),
            "--selection-max-frame",
            str(args.selection_max_frame),
            "--selection-preview-dir",
            str(preview_dir),
            "--detector-cfg",
            str(args.detector_cfg),
            "--detector-wts",
            str(args.detector_wts),
            "--pose-cfg",
            str(args.pose_cfg),
            "--pose-wts",
            str(args.pose_wts),
            "--device",
            args.device,
        ]
        if input_video:
            pose_cmd += ["--input-video", str(input_video)]
        else:
            pose_cmd += ["--input-root", str(input_root)]
        if args.selection_assist_json:
            pose_cmd += ["--selection-assist-json", str(Path(args.selection_assist_json).resolve())]
        run_cmd(pose_cmd, cwd=HERE)
        print(f"Selection preview written under: {preview_dir}", flush=True)
        return

    pose_cmd = [
        py,
        str(HERE / "pose_json_video_v3.py"),
        "--output-root",
        str(pose_overlay_dir),
        "--detector-cfg",
        str(args.detector_cfg),
        "--detector-wts",
        str(args.detector_wts),
        "--pose-cfg",
        str(args.pose_cfg),
        "--pose-wts",
        str(args.pose_wts),
    ]
    if input_video:
        pose_cmd += ["--input-video", str(input_video)]
    else:
        pose_cmd += ["--input-root", str(input_root)]
    if args.pose_from_track_cache_json:
        if args.pose_left_track_id is None or args.pose_right_track_id is None:
            raise SystemExit("--pose-from-track-cache-json requires --pose-left-track-id and --pose-right-track-id.")
        pose_cmd += [
            "--materialize-from-track-cache-json",
            str(Path(args.pose_from_track_cache_json).resolve()),
            "--materialize-left-track-id",
            str(args.pose_left_track_id),
            "--materialize-right-track-id",
            str(args.pose_right_track_id),
        ]
    else:
        pose_cmd += ["--device", args.device]
    if args.selection_override_json:
        pose_cmd += ["--selection-override-json", str(Path(args.selection_override_json).resolve())]
    if args.selection_assist_json:
        pose_cmd += ["--selection-assist-json", str(Path(args.selection_assist_json).resolve())]
    if args.track_cache_json:
        pose_cmd += ["--track-cache-json", str(Path(args.track_cache_json).resolve())]
    run_cmd(pose_cmd, cwd=HERE)

    if args.transcode_pose_h264:
        transcode_pose_overlays_h264(pose_overlay_dir)

    run_cmd(
        [
            py,
            str(HERE / "process_pose_joints_v3.py"),
            "--pose-overlay-dir",
            str(pose_overlay_dir),
            "--output-root",
            str(processed_dir),
        ],
        cwd=HERE,
    )

    l1_cmd = [
        py,
        str(HERE / "l1_stage.py"),
        "--processed-dir",
        str(processed_dir),
        "--out-dir",
        str(l1_out),
        "--max-frames",
        str(args.max_frames),
        "--weapon-side",
        "auto",
        "--weapon-auto-mode",
        args.weapon_auto_mode,
    ]
    if args.use_raw:
        l1_cmd.append("--use-raw")
    run_cmd(l1_cmd, cwd=HERE)

    run_cmd(
        [
            py,
            str(HERE / "l2_stage.py"),
            "--left-dir",
            str(l1_out / "left"),
            "--right-dir",
            str(l1_out / "right"),
            "--out-dir",
            str(l2_out),
            "--max-frames",
            str(args.max_frames),
            "--toward-mode",
            args.l2_toward_mode,
            "--feature-profile",
            "boundary",
        ],
        cwd=HERE,
    )

    boundary_cfg = Path(args.boundary_config).resolve()
    boundary_ckpt = resolve_ckpt(Path(args.boundary_ckpt).resolve())
    l2_files = sorted(l2_out.glob("*_l2.json"))
    if not l2_files:
        raise RuntimeError(f"No L2 files found under {l2_out}")
    for l2_file in l2_files:
        out_file = boundary_preds / l2_file.name.replace("_l2.json", "_boundary.json")
        run_cmd(
            [
                py,
                str(BOUNDARY_PACKAGE_ROOT / "infer.py"),
                "--config",
                str(boundary_cfg),
                "--ckpt",
                str(boundary_ckpt),
                "--input",
                str(l2_file),
                "--output",
                str(out_file),
                "--device",
                args.device,
            ],
            cwd=BOUNDARY_PACKAGE_ROOT,
        )

    write_manifest(
        work / "boundary_pipeline_manifest.json",
        {
            "input_video": str(input_video) if input_video else "",
            "input_root": str(input_root) if input_root else "",
            "pose_overlay_dir": str(pose_overlay_dir),
            "processed_dir": str(processed_dir),
            "l1_out": str(l1_out),
            "l2_out": str(l2_out),
            "boundary_preds": str(boundary_preds),
            "boundary_config": str(boundary_cfg),
            "boundary_ckpt": str(boundary_ckpt),
            "boundary_files": [str(p) for p in sorted(boundary_preds.glob("*_boundary.json"))],
        },
    )
    print(f"Boundary segmentation complete: {boundary_preds}", flush=True)


if __name__ == "__main__":
    main()
