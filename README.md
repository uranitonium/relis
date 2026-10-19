# ReLIS: Stable-Regime Segmentation of Competitive Dyadic Fencing Interactions

Code for the WACV 2027 (Applications Track) paper

> **Stable-Regime Segmentation of Competitive Dyadic Fencing Interactions**
> Ziwen Chen (University of Illinois Urbana-Champaign) and Evelyn Su (University of Toronto Schools)
> IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2027.

ReLIS (Relational Latent-Instability Segmenter) learns an opponent-relative
trajectory from two-fencer pose features and converts local violations of
trajectory stability into continuous activity evidence `b_t` and an ordered
partition into lower-activity (stable) and higher-activity intervals. Human
labels never enter neural training; a small labeled development set only
selects the checkpoint and extraction rules.

## Repository layout

| Path | Contents |
| --- | --- |
| `boundary_segmentation/` | ReLIS model (Transformer/TCN/BiLSTM encoders, boundary head), losses, dataset, training (`train.py`), evaluation (`eval.py`) and single-sequence inference (`infer.py`) |
| `boundary_segmentation/postprocess/` | Activity-evidence extraction (EMA smoothing, robust hysteresis, coverage clamp, gap merging, onset backtracking, valley splitting) |
| `configs/relis_primary.yaml` | Exact configuration of the primary paper model and its development-selected extractor |
| `deployment/checkpoints/legacy_auto/config.yaml` | Base configuration used by several experiment scripts |
| `runtime/full_pipeline/` | Pose-to-partition front end: detection, pose, fencer tracking, L1 movement and L2 pairwise relational features, ReLIS inference |
| `paper_experiments/scripts/` | Scripts for the paper's experiments: baselines (RGB, R3D, VideoMAE, Video Swin, ST-GCN, kernel change point, TS-CP2-style), ablations, extractor searches, bootstrap analyses, downstream state models, table/figure export |
| `paper_experiments/tests/` | Unit tests |

The pose front end (`runtime/full_pipeline/`) expects local installations of
MMDetection/MMPose and the RTMDet-tiny / RTMPose-m weights in
`runtime/full_pipeline/weights/`; these are not redistributed here.

## Installation

```bash
conda create -n relis python=3.8 -y
conda activate relis
pip install -r requirements.txt
# for the pose front end, additionally install mmengine/mmcv/mmdet/mmpose
# following the OpenMMLab instructions (see runtime/full_pipeline/README.md)
```

## Usage

### 1. Feature extraction (video -> pose -> L1/L2 relational features)

```bash
python runtime/full_pipeline/run_to_boundary.py --input-video path/to/exchange.mp4 \
  --boundary-config configs/relis_primary.yaml \
  --boundary-ckpt checkpoints/relis_primary_seed20270618_release.pt
python runtime/full_pipeline/run_to_boundary.py --help   # all options
```

This produces per-exchange `*_l2.json` files containing `l2` (`[T, 26]`
per-frame opponent-relative features) and `l2_mask`.

### 2. Training (unsupervised; no state labels)

```bash
# point --data-dir at a folder of *_l2.json files
PYTHONPATH=. python boundary_segmentation/train.py --config configs/relis_primary.yaml \
  --data-dir data/l2/train --out-dir outputs/relis
```

### 3. Inference

```bash
PYTHONPATH=. python boundary_segmentation/infer.py \
  --config configs/relis_primary.yaml \
  --ckpt checkpoints/relis_primary_seed20270618_release.pt \
  --input path/to/exchange_l2.json --output out.json
```

The output contains the latent trajectory `z`, activity evidence `b`, its
smoothed version `b_s`, `transition_zones` (higher-activity intervals) and
`stable_segments`.

### 4. Evaluation

```bash
PYTHONPATH=. python boundary_segmentation/eval.py --config configs/relis_primary.yaml --ckpt <checkpoint>
python paper_experiments/scripts/evaluate_boundaries.py --help
```

Paper-level experiment scripts take `--help`. Some scripts read optional
environment variables: `RELIS_PYTHON` (interpreter for the pose/OpenMMLab
environment; defaults to the current interpreter), `RELIS_STATE_ROOT`
(downstream state-model data) and `RELIS_VIDEO_ROOT` (videos for the RGB/video
baselines).

## Checkpoints

| File | Description | Size |
| --- | --- | --- |
| `relis_primary_seed20270618_release.pt` | Primary paper model (seed 20270618, 1,000 updates) | 2.2 MB |
| `relis_smooth02_seed2027061{4,5,6,7}_release.pt` | Matched seeds used for five-seed results | 2.2 MB each |

Checkpoints are distributed as GitHub release assets (coming soon). Release
files contain model weights and the training configuration only (optimizer
state removed). The primary model corresponds to the archived training
checkpoint with SHA-256
`8c5a59bf3a64db5b03a1b9548a8681cef0a81b6876ae0420e1bada29e33ce71d`.

## Data

Raw broadcast video, pose outputs derived from it, and the human annotations
of third-party footage are not included in this repository. Exact numeric
results for the paper's tables are provided in the paper's supplementary
archive.

## Citation

```bibtex
@inproceedings{chen2027relis,
  author    = {Chen, Ziwen and Su, Evelyn},
  title     = {Stable-Regime Segmentation of Competitive Dyadic Fencing Interactions},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License

MIT; see `LICENSE`. Third-party components used by the pose front end
(MMDetection, MMPose, RTMDet/RTMPose weights) keep their own licenses.
