# Boundary-Only Full Pipeline

This folder contains the current website pipeline through boundary segmentation only.

Included stages:

1. `mmdetection` person detection
2. `mmpose` top-down pose extraction
3. fencer selection preview / override / assist
4. pose post-processing
5. L1 movement features
6. L2 pairwise boundary features
7. L3 boundary inference

Intentionally excluded:

- hard tactical state scoring (`l3_stage_hard.py`)
- LLM context export
- OpenAI analysis
- L4/viewer/portal code

## Environment

Use a Python environment that can import `torch`, `mmdet`, `mmpose`, `cv2`, `norfair`, `numpy`, and `yaml`.
Example (any environment with the packages above):

```bash
python
```

## One-Command Boundary Run

```bash
cd runtime/full_pipeline
python run_to_boundary.py \
  --input-video /path/to/video.mp4 \
  --working-dir working/example \
  --device auto
```

Outputs are written under `working/example`:

- `pose_overlay_v3/`
- `processed_joints_v3/`
- `legacy_auto/l1_out/`
- `legacy_auto/l2_out/`
- `legacy_auto/boundary_preds/`
- `boundary_pipeline_manifest.json`

## Fencer Selection Preview

To match the website confirmation flow, generate a selection preview first:

```bash
python run_to_boundary.py \
  --input-video /path/to/video.mp4 \
  --working-dir working/example \
  --selection-preview-only \
  --selection-frame 5 \
  --selection-max-frame 15
```

After confirming left/right fencers, run the full boundary pipeline with:

```bash
python run_to_boundary.py \
  --input-video /path/to/video.mp4 \
  --working-dir working/example \
  --selection-override-json /path/to/selection_override.json
```

If no override is supplied, `pose_json_video_v3.py` uses its current automatic fencer selection behavior.

