#!/usr/bin/env python3
"""Evaluate L3 boundary model (unsupervised stats)."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.collate import collate_pad
from data.dataset import L2SequenceDataset, SyntheticSequenceDataset
from models.l3_model import L3BoundaryModel
from postprocess.segments import stable_segments
from postprocess.smooth import ema_smooth
from postprocess.zones import extract_transition_zones
from utils.metrics import summarize_zones
from utils.plotting import plot_boundary


def load_cfg(path: str) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def build_dataset(cfg: Dict):
    dcfg = cfg["data"]
    data_dir = dcfg["data_dir"]
    if not os.path.isabs(data_dir):
        cfg_dir = os.path.dirname(cfg.get("_config_path", ""))
        data_dir = os.path.normpath(os.path.join(cfg_dir, data_dir))
    if dcfg.get("synthetic", False):
        return SyntheticSequenceDataset(
            num_seq=dcfg["synthetic_num"],
            t_max=dcfg["synthetic_T"],
            feat_dim=dcfg["synthetic_m"],
            seed=cfg["seed"],
        )
    return L2SequenceDataset(
        data_dir=data_dir,
        pattern=dcfg["file_glob"],
        max_files=dcfg.get("max_files", 0),
        max_len=dcfg.get("max_len", 0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = build_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pad,
    )

    sample = next(iter(loader))
    input_dim = sample["x"].shape[-1]
    model = L3BoundaryModel(input_dim=input_dim, cfg=cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    total_b = 0.0
    total_mask = 0.0
    transition_all: List[List[Tuple[int, int]]] = []
    stable_all: List[List[Tuple[int, int]]] = []

    pp = cfg["postprocess"]
    plot_dir = cfg["eval"]["plot_dir"]
    os.makedirs(plot_dir, exist_ok=True)
    num_plots = cfg["eval"]["num_plots"]
    plot_count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            h, z, slow, phi, b = model(x, mask)

            total_b += (b * mask.float()).sum().item()
            total_mask += mask.float().sum().item()

            b_np = b.cpu().numpy()
            mask_np = mask.cpu().numpy()
            phi_np = phi.cpu().numpy()

            for i in range(b_np.shape[0]):
                b_i = b_np[i]
                m_i = mask_np[i]
                b_s = ema_smooth(b_i, alpha=pp["ema_alpha"])
                zones = extract_transition_zones(
                    b_i,
                    m_i,
                    alpha=pp["ema_alpha"],
                    thr_on=pp["thr_on"],
                    min_len=pp["min_len"],
                    merge_gap=pp["merge_gap"],
                    mode=pp.get("mode", "legacy"),
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
                    split_min_zone_len=pp.get("split_min_zone_len", 0),
                    split_min_rel_drop=pp.get("split_min_rel_drop", 0.0),
                    split_max_splits=pp.get("split_max_splits", 0),
                    micro_merge_zone_len=pp.get("micro_merge_zone_len", 0),
                    micro_merge_gap=pp.get("micro_merge_gap", 0),
                    micro_merge_max_span=pp.get("micro_merge_max_span", 0),
                )
                segs = stable_segments(m_i, zones, min_seg_len=pp["min_seg_len"])
                transition_all.append(zones)
                stable_all.append(segs)

                if plot_count < num_plots:
                    delta_z = phi_np[i, :, 0]
                    var_t = phi_np[i, :, 1]
                    plot_path = os.path.join(plot_dir, f"sample_{plot_count:03d}.png")
                    plot_boundary(plot_path, b_i, b_s, pp["thr_on"], delta_z=delta_z, var_t=var_t)
                    plot_count += 1

    b_mean = float(total_b / max(1.0, total_mask))
    summary = summarize_zones(b_mean, transition_all, stable_all)
    print("Eval summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
