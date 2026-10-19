# L3 Boundary Modeling (Unsupervised)

L3 boundary modeling learns a **regime‑failure detector** on top of L1/L2 feature streams. It does **not** use boundary labels or pseudo‑labels. Boundaries emerge because the model is penalized for instability everywhere except where instability is unavoidable.

One‑sentence mental model:
> The boundary head is a learned regime‑failure detector that allocates instability to the few places where interaction structure genuinely changes.

## Input / Output

**Input (per sequence)**
- `x`: float tensor `[T, m]` (per‑frame interaction features, e.g., L2 output)
- `mask`: bool/float `[T]` (1 valid, 0 padded)

**Output (per sequence)**
- `z`: `[T, d]` latent interaction state
- `b`: `[T]` boundary evidence in `[0,1]`
- `transition_zones`: list of `(start, end)` intervals
- `stable_segments`: list of `(start, end)` intervals

`b_t` is not a calibrated probability; it is a learned instability score.

## Repo Layout

See `configs/`, `models/`, `features/`, `losses/`, `postprocess/`, `utils/`, and the entrypoints `train.py`, `eval.py`, `infer.py`.

## Training

```bash
python train.py --config configs/default.yaml
```

Default config expects L2 JSON files under `../l2_out/right` (relative to `l3_boundary/`). Each file should have `l2` and `l2_mask` fields. Set `data.synthetic: true` in the config to run without external data.

## Evaluation

```bash
python eval.py --config configs/default.yaml --ckpt outputs/ckpt_step_1000.pt
```

Eval reports unsupervised sanity stats and saves qualitative plots to `eval_plots/`.

## Inference (single sequence)

```bash
python infer.py --config configs/default.yaml --ckpt outputs/ckpt_step_1000.pt --input path/to/file_l2.json --output out.json
```

`infer.py` returns `z`, `b`, `b_s`, `transition_zones`, and `stable_segments`.

## Losses (why boundaries emerge)

- **Sparsity**: encourages low boundary mass overall.
- **Smoothness**: discourages jittery boundary outputs.
- **Regime consistency**: forces `z` to stay compact where `b` is low, pushing instability into a small set of boundary frames.
- **Optional contrastive stability**: adjacent frames in stable regions are made similar.

All losses are mask‑aware; padded frames never contribute.
