#!/usr/bin/env python3
"""Inference for L3 boundary model."""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch
import yaml

from models.l3_model import L3BoundaryModel
from postprocess.segments import stable_segments
from postprocess.smooth import ema_smooth
from postprocess.zones import extract_transition_zones
from utils.io import load_json, save_json


def load_cfg(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_input(path: str) -> Dict[str, np.ndarray]:
    if path.endswith(".json"):
        data = load_json(path)
        x = np.asarray(data["l2"], dtype=np.float32)
        mask = np.asarray(data["l2_mask"], dtype=np.bool_)
        return {"x": x, "mask": mask}
    if path.endswith(".npz"):
        data = np.load(path)
        return {"x": np.asarray(data["x"], dtype=np.float32), "mask": np.asarray(data["mask"], dtype=np.bool_)}
    raise ValueError(f"Unsupported input type: {path}")


def infer_sequence(x: np.ndarray, mask: np.ndarray, model: L3BoundaryModel, cfg: Dict) -> Dict:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask.astype(np.bool_)).unsqueeze(0).to(device)
        h, z, slow, phi, b = model(x_t, mask_t)
    z = z.squeeze(0).cpu().numpy()
    b = b.squeeze(0).cpu().numpy()
    pp = cfg["postprocess"]
    b_s = ema_smooth(b, alpha=pp["ema_alpha"])
    zones = extract_transition_zones(
        b,
        mask,
        alpha=pp["ema_alpha"],
        thr_on=pp["thr_on"],
        min_len=pp["min_len"],
        merge_gap=pp["merge_gap"],
        mode=pp.get("mode", "robust_hysteresis"),
        thr_off=pp.get("thr_off"),
        abs_on=pp.get("abs_on", 0.0),
        abs_off=pp.get("abs_off"),
        robust_z_on=pp.get("robust_z_on", 1.0),
        robust_z_off=pp.get("robust_z_off", 0.4),
        min_prominence=pp.get("min_prominence", 0.0),
        min_peak_height=pp.get("min_peak_height", 0.0),
        boundary_frac_min=pp.get("boundary_frac_min", 0.0),
        boundary_frac_max=pp.get("boundary_frac_max", 1.0),
        budget_step=pp.get("budget_step", 0.15),
        max_budget_iters=pp.get("max_budget_iters", 3),
        onset_backtrack=pp.get("onset_backtrack", 0),
        split_min_zone_len=pp.get("split_min_zone_len", 0),
        split_min_rel_drop=pp.get("split_min_rel_drop", 0.0),
        split_max_splits=pp.get("split_max_splits", 0),
        micro_merge_zone_len=pp.get("micro_merge_zone_len", 0),
        micro_merge_gap=pp.get("micro_merge_gap", 0),
        micro_merge_max_span=pp.get("micro_merge_max_span", 0),
    )
    segs = stable_segments(mask, zones, min_seg_len=pp["min_seg_len"])
    return {
        "z": z,
        "b": b,
        "b_s": b_s,
        "transition_zones": zones,
        "stable_segments": segs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_input(args.input)
    input_dim = data["x"].shape[-1]
    model = L3BoundaryModel(input_dim=input_dim, cfg=cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    result = infer_sequence(data["x"], data["mask"], model, cfg)

    if args.output:
        if args.output.endswith(".npz"):
            np.savez(
                args.output,
                z=result["z"],
                b=result["b"],
                b_s=result["b_s"],
                transition_zones=result["transition_zones"],
                stable_segments=result["stable_segments"],
            )
        else:
            payload = {
                "z": result["z"].tolist(),
                "b": result["b"].tolist(),
                "b_s": result["b_s"].tolist(),
                "transition_zones": result["transition_zones"],
                "stable_segments": result["stable_segments"],
            }
            save_json(args.output, payload)
    else:
        print("transition_zones:", result["transition_zones"])
        print("stable_segments:", result["stable_segments"])


if __name__ == "__main__":
    main()
