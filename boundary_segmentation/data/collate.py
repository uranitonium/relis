"""Batch collation with padding."""

from __future__ import annotations

from typing import Dict, List

import torch


def collate_pad(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    lengths = [b["x"].shape[0] for b in batch]
    max_t = max(lengths)
    feat_dim = batch[0]["x"].shape[1]
    bsz = len(batch)

    x = torch.zeros((bsz, max_t, feat_dim), dtype=torch.float32)
    mask = torch.zeros((bsz, max_t), dtype=torch.bool)
    ids = []

    for i, item in enumerate(batch):
        t = item["x"].shape[0]
        x[i, :t] = torch.from_numpy(item["x"]).float()
        mask[i, :t] = torch.from_numpy(item["mask"]).bool()
        ids.append(item.get("id", f"seq_{i}"))

    return {"x": x, "mask": mask, "id": ids}


def collate_pad_pairs(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    primary = collate_pad(batch)
    paired = collate_pad(
        [
            {
                "x": item["x_pair"],
                "mask": item["mask_pair"],
                "id": item.get("id"),
            }
            for item in batch
        ]
    )
    primary["x_pair"] = paired["x"]
    primary["mask_pair"] = paired["mask"]
    return primary
