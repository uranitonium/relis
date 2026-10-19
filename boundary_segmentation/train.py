#!/usr/bin/env python3
"""Train L3 boundary model (unsupervised)."""

from __future__ import annotations

import argparse
import os
from typing import Dict

import math
import torch
import yaml
from torch.utils.data import DataLoader

from data.collate import collate_pad, collate_pad_pairs
from data.dataset import L2SequenceDataset, PairedL2SequenceDataset, SyntheticSequenceDataset
from losses.align import boundary_phi_align_loss
from losses.b_variance import b_variance_floor_loss
from losses.consistency import consistency_loss
from losses.contrastive import contrastive_loss
from losses.prior import boundary_mass_prior_loss
from losses.regularizers import z_var_floor_loss
from losses.smoothness import smoothness_loss
from losses.sparsity import sparsity_loss
from models.l3_model import L3BoundaryModel
from utils.seed import set_seed


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
    paired_view_dir = dcfg.get("paired_view_dir", "")
    if paired_view_dir:
        if not os.path.isabs(paired_view_dir):
            cfg_dir = os.path.dirname(cfg.get("_config_path", ""))
            paired_view_dir = os.path.normpath(os.path.join(cfg_dir, paired_view_dir))
        return PairedL2SequenceDataset(
            data_dir=data_dir,
            paired_view_dir=paired_view_dir,
            pattern=dcfg["file_glob"],
            max_files=dcfg.get("max_files", 0),
            max_len=dcfg.get("max_len", 0),
        )
    return L2SequenceDataset(
        data_dir=data_dir,
        pattern=dcfg["file_glob"],
        max_files=dcfg.get("max_files", 0),
        max_len=dcfg.get("max_len", 0),
    )


def build_scheduler(optim, warmup_steps: int, max_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-ckpt", default="", help="Optional model checkpoint used for initialization")
    parser.add_argument(
        "--init-allow-missing",
        action="store_true",
        help="Allow newly introduced model parameters to be absent from the initialization checkpoint",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--out-dir", default="", help="Override checkpoint output directory")
    parser.add_argument("--data-dir", default="", help="Override the L2 training data directory")
    parser.add_argument("--paired-view-dir", default="", help="Aligned calibrated L2 view for invariance training")
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum training steps")
    parser.add_argument("--encoder-type", default="", help="Override temporal encoder type")
    parser.add_argument("--camera-view-dim", type=int, default=None)
    parser.add_argument("--camera-gate-bias", type=float, default=None)
    parser.add_argument("--camera-residual-view-dim", type=int, default=None)
    parser.add_argument("--camera-initial-mix", type=float, default=None)
    parser.add_argument("--freeze-base-for-camera-fusion", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--boundary-k", type=int, default=None, help="Override temporal phi window")
    parser.add_argument("--lambda-smooth", type=float, default=None)
    parser.add_argument("--lambda-cons", type=float, default=None)
    parser.add_argument("--lambda-contrast", type=float, default=None)
    parser.add_argument("--lambda-b-prior", type=float, default=None)
    parser.add_argument("--lambda-b-var", type=float, default=None)
    parser.add_argument("--lambda-z-var", type=float, default=None)
    parser.add_argument("--lambda-b-align", type=float, default=None)
    parser.add_argument("--lambda-camera-boundary", type=float, default=None)
    parser.add_argument("--lambda-camera-latent", type=float, default=None)
    parser.add_argument("--camera-view-augmentation-prob", type=float, default=None)
    parser.add_argument("--b-mass-target", type=float, default=None)
    parser.add_argument("--detach-b-align-target", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.out_dir:
        cfg["train"]["out_dir"] = os.path.abspath(args.out_dir)
    if args.data_dir:
        cfg["data"]["data_dir"] = os.path.abspath(args.data_dir)
    if args.paired_view_dir:
        cfg["data"]["paired_view_dir"] = os.path.abspath(args.paired_view_dir)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.encoder_type:
        cfg["model"]["encoder_type"] = args.encoder_type
    if args.camera_view_dim is not None:
        cfg["model"].setdefault("camera_dual", {})["view_dim"] = args.camera_view_dim
    if args.camera_gate_bias is not None:
        cfg["model"].setdefault("camera_dual", {})["gate_bias"] = args.camera_gate_bias
    if args.camera_residual_view_dim is not None:
        camera_cfg = cfg["model"].setdefault("camera_residual_fusion", {})
        camera_cfg["enabled"] = True
        camera_cfg["view_dim"] = args.camera_residual_view_dim
    if args.camera_initial_mix is not None:
        camera_cfg = cfg["model"].setdefault("camera_residual_fusion", {})
        camera_cfg["enabled"] = True
        camera_cfg["initial_mix"] = args.camera_initial_mix
    if args.freeze_base_for_camera_fusion:
        cfg["model"].setdefault("camera_residual_fusion", {})["freeze_base"] = True
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.warmup_steps is not None:
        cfg["train"]["warmup_steps"] = args.warmup_steps
    if args.boundary_k is not None:
        cfg["model"]["boundary"]["k"] = args.boundary_k
    if args.lambda_smooth is not None:
        cfg["loss"]["lambda_smooth"] = args.lambda_smooth
    if args.lambda_cons is not None:
        cfg["loss"]["lambda_cons"] = args.lambda_cons
    if args.lambda_contrast is not None:
        cfg["loss"]["lambda_contrast"] = args.lambda_contrast
        if args.lambda_contrast == 0:
            cfg["loss"]["use_contrast"] = False
    if args.lambda_b_prior is not None:
        cfg["loss"]["lambda_b_prior"] = args.lambda_b_prior
    if args.lambda_b_var is not None:
        cfg["loss"]["lambda_b_var"] = args.lambda_b_var
    if args.lambda_z_var is not None:
        cfg["loss"]["lambda_z_var"] = args.lambda_z_var
    if args.lambda_b_align is not None:
        cfg["loss"]["lambda_b_align"] = args.lambda_b_align
    if args.lambda_camera_boundary is not None:
        cfg["loss"]["lambda_camera_boundary"] = args.lambda_camera_boundary
    if args.lambda_camera_latent is not None:
        cfg["loss"]["lambda_camera_latent"] = args.lambda_camera_latent
    if args.camera_view_augmentation_prob is not None:
        cfg["data"]["paired_view_augmentation_prob"] = args.camera_view_augmentation_prob
    if args.b_mass_target is not None:
        cfg["loss"]["b_mass_target"] = args.b_mass_target
    if args.detach_b_align_target:
        cfg["loss"]["detach_b_align_target"] = True
    set_seed(cfg["seed"])

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = build_dataset(cfg)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check data_dir/file_glob or synthetic settings.")

    paired_views = isinstance(dataset, PairedL2SequenceDataset)
    camera_view_augmentation_prob = float(
        cfg["data"].get("paired_view_augmentation_prob", 0.0)
    )
    if not 0.0 <= camera_view_augmentation_prob <= 1.0:
        raise ValueError("paired_view_augmentation_prob must be between 0 and 1")
    if camera_view_augmentation_prob and not paired_views:
        raise ValueError("paired_view_augmentation_prob requires paired_view_dir")
    if camera_view_augmentation_prob and (
        cfg["loss"].get("lambda_camera_boundary", 0.0)
        or cfg["loss"].get("lambda_camera_latent", 0.0)
    ):
        raise ValueError("Camera view augmentation and consistency losses are separate experiments")
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["data"].get("shuffle", True),
        num_workers=0,
        collate_fn=collate_pad_pairs if paired_views else collate_pad,
        drop_last=False,
    )

    # infer input_dim
    sample = next(iter(loader))
    input_dim = sample["x"].shape[-1]
    model = L3BoundaryModel(input_dim=input_dim, cfg=cfg).to(device)
    if args.init_ckpt:
        init_state = torch.load(args.init_ckpt, map_location=device)
        incompatible = model.load_state_dict(
            init_state["model"], strict=not args.init_allow_missing
        )
        if args.init_allow_missing:
            invalid_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith(("camera_feature_logits", "camera_context."))
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Initialization mismatch: "
                    f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
                )
        print(f"[train] initialized model from: {os.path.abspath(args.init_ckpt)}")

    camera_fusion_cfg = cfg["model"].get("camera_residual_fusion", {})
    if camera_fusion_cfg.get("freeze_base", False):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name == "camera_feature_logits" or name.startswith("camera_context.")
            )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable model parameters")
    optim = torch.optim.AdamW(
        trainable_parameters,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    sched = build_scheduler(optim, cfg["train"]["warmup_steps"], cfg["train"]["max_steps"])

    out_dir = os.path.abspath(cfg["train"]["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] checkpoint dir: {out_dir}")
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    data_iter = iter(loader)
    es_cfg = cfg["train"].get("early_stop", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    es_min_steps = int(es_cfg.get("min_steps", 0))
    es_patience = int(es_cfg.get("patience_steps", 0))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_alpha = float(es_cfg.get("ema_alpha", 0.1))
    es_best = float("inf")
    es_bad = 0
    es_ema = None
    stopped_early = False

    def _save_ckpt(path: str, step: int, extra: Dict | None = None) -> None:
        state = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "step": step,
            "cfg": cfg,
        }
        if es_ema is not None:
            state["early_stop_ema"] = float(es_ema)
        if es_best != float("inf"):
            state["best_ema"] = float(es_best)
        if extra:
            state.update(extra)
        torch.save(state, path)

    for step in range(1, cfg["train"]["max_steps"] + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x = batch["x"].to(device)
        mask = batch["mask"].to(device)

        camera_boundary_weight = cfg["loss"].get("lambda_camera_boundary", 0.0)
        camera_latent_weight = cfg["loss"].get("lambda_camera_latent", 0.0)
        if camera_view_augmentation_prob:
            x_pair = batch["x_pair"].to(device)
            # A deterministic schedule avoids introducing a seed/RNG confound.
            positions = torch.arange(x.shape[0], device=device)
            percentile = (positions * 37 + step * 17) % 100
            use_pair = percentile < int(round(camera_view_augmentation_prob * 100))
            x = torch.where(use_pair[:, None, None], x_pair, x)
        camera_consistency_enabled = paired_views and bool(
            camera_boundary_weight or camera_latent_weight
        )
        if camera_consistency_enabled:
            cpu_rng_before = torch.get_rng_state()
            cuda_rng_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        h, z, slow, phi, b = model(x, mask)
        if camera_consistency_enabled:
            cpu_rng_after = torch.get_rng_state()
            cuda_rng_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state_all(cuda_rng_before)
            x_pair = batch["x_pair"].to(device)
            mask_pair = batch["mask_pair"].to(device)
            _, z_pair, _, _, b_pair = model(x_pair, mask_pair)
            torch.set_rng_state(cpu_rng_after)
            if cuda_rng_after is not None:
                torch.cuda.set_rng_state_all(cuda_rng_after)

        l_sparse = sparsity_loss(b, mask)
        l_smooth = smoothness_loss(b, mask)
        l_cons = consistency_loss(z, b, mask, cfg["model"]["boundary"]["k"])
        l_contrast = torch.tensor(0.0, device=device)
        if cfg["loss"]["use_contrast"]:
            l_contrast = contrastive_loss(z, b, mask, cfg["loss"]["contrast_temp"])
        l_zvar = z_var_floor_loss(z, mask, cfg["loss"].get("z_var_target", 0.0))
        l_bprior = boundary_mass_prior_loss(b, mask, cfg["loss"].get("b_mass_target", 0.0))
        l_bvar = b_variance_floor_loss(b, mask, cfg["loss"].get("b_var_target", 0.0))
        l_balign = boundary_phi_align_loss(
            b,
            phi,
            mask,
            temperature=cfg["loss"].get("b_align_temp", 1.0),
            detach_target=cfg["loss"].get("detach_b_align_target", False),
        )

        total = (
            cfg["loss"]["lambda_sparse"] * l_sparse
            + cfg["loss"]["lambda_smooth"] * l_smooth
            + cfg["loss"]["lambda_cons"] * l_cons
            + cfg["loss"]["lambda_contrast"] * l_contrast
            + cfg["loss"].get("lambda_z_var", 0.0) * l_zvar
            + cfg["loss"].get("lambda_b_prior", 0.0) * l_bprior
            + cfg["loss"].get("lambda_b_var", 0.0) * l_bvar
            + cfg["loss"].get("lambda_b_align", 0.0) * l_balign
        )

        l_camera_boundary = torch.tensor(0.0, device=device)
        l_camera_latent = torch.tensor(0.0, device=device)
        if camera_consistency_enabled:
            pair_mask = mask & mask_pair
            pair_weight = pair_mask.float()
            l_camera_boundary = (
                (b - b_pair).square() * pair_weight
            ).sum() / pair_weight.sum().clamp_min(1.0)
            latent_distance = 1.0 - torch.nn.functional.cosine_similarity(z, z_pair, dim=-1)
            l_camera_latent = (
                latent_distance * pair_weight
            ).sum() / pair_weight.sum().clamp_min(1.0)
            total = (
                total
                + camera_boundary_weight * l_camera_boundary
                + camera_latent_weight * l_camera_latent
            )

        optim.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        optim.step()
        sched.step()

        if es_enabled:
            metric = total.item()
            es_ema = metric if es_ema is None else (es_alpha * metric + (1.0 - es_alpha) * es_ema)
            if step >= es_min_steps:
                if es_ema < es_best - es_min_delta:
                    es_best = es_ema
                    es_bad = 0
                    _save_ckpt(
                        os.path.join(out_dir, "ckpt_best_ema.pt"),
                        step=step,
                        extra={"best_ema_step": step},
                    )
                else:
                    es_bad += 1

        if step % cfg["train"]["log_every"] == 0:
            with torch.no_grad():
                mean_b = (b * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
                b_masked = b[mask]
                max_b = b_masked.max().item() if b_masked.numel() else 0.0
                p95_b = torch.quantile(b_masked, 0.95).item() if b_masked.numel() else 0.0
                z_masked = z[mask]
                phi_masked = phi[mask]
                slow_masked = slow[mask]
                z_std = z_masked.std(dim=0).mean().item() if z_masked.numel() else 0.0
                phi_std = phi_masked.std(dim=0).mean().item() if phi_masked.numel() else 0.0
                slow_std = slow_masked.std(dim=0).mean().item() if slow_masked.numel() else 0.0
                thr = cfg["postprocess"]["thr_on"]
                boundary_frames = ((b > thr) & mask).float().sum().item()
            print(
                f"step {step} | total {total.item():.4f} | sparse {l_sparse.item():.4f} | "
                f"smooth {l_smooth.item():.4f} | cons {l_cons.item():.4f} | "
                f"contrast {l_contrast.item():.4f} | zvar {l_zvar.item():.4f} | "
                f"bprior {l_bprior.item():.4f} | bvar {l_bvar.item():.4f} | "
                f"balign {l_balign.item():.4f} | camera_b {l_camera_boundary.item():.4f} | "
                f"camera_z {l_camera_latent.item():.4f} | mean(b) {mean_b.item():.4f} | "
                f"max(b) {max_b:.4f} | p95(b) {p95_b:.4f} | "
                f"z_std {z_std:.4f} | phi_std {phi_std:.4f} | slow_std {slow_std:.4f} | "
                f"b>thr {int(boundary_frames)}"
            )

        if step % cfg["train"]["ckpt_every"] == 0:
            ckpt_path = os.path.join(out_dir, f"ckpt_step_{step}.pt")
            _save_ckpt(ckpt_path, step=step)
            _save_ckpt(os.path.join(out_dir, "ckpt_last.pt"), step=step)
        if es_enabled and es_bad >= es_patience and step >= es_min_steps:
            ckpt_path = os.path.join(out_dir, "ckpt_early_stop.pt")
            _save_ckpt(ckpt_path, step=step)
            print(f"Early stopping at step {step} (ema={es_ema:.6f}).")
            stopped_early = True
            break

    if not stopped_early:
        _save_ckpt(os.path.join(out_dir, "ckpt_final.pt"), step=cfg["train"]["max_steps"])


if __name__ == "__main__":
    main()
