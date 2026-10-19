"""Datasets for L3 boundary modeling."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
from torch.utils.data import Dataset

from utils.io import list_files, load_json


class L2SequenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        pattern: str = "*_l2.json",
        max_files: int = 0,
        max_len: int = 0,
    ) -> None:
        self.files = list_files(data_dir, pattern)
        if max_files and max_files > 0:
            self.files = self.files[:max_files]
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.files)

    def _load_item(self, path: str) -> Dict[str, np.ndarray]:
        if path.endswith(".json"):
            data = load_json(path)
            x = np.asarray(data["l2"], dtype=np.float32)
            mask = np.asarray(data["l2_mask"], dtype=np.bool_)
            frames = np.asarray(data.get("frames", np.arange(len(x))), dtype=np.int64)
        elif path.endswith(".npz"):
            data = np.load(path)
            x = np.asarray(data["x"], dtype=np.float32)
            mask = np.asarray(data["mask"], dtype=np.bool_)
            frames = np.asarray(
                data["frames"] if "frames" in data else np.arange(len(x)),
                dtype=np.int64,
            )
        else:
            raise ValueError(f"Unsupported file type: {path}")
        if self.max_len and x.shape[0] > self.max_len:
            x = x[: self.max_len]
            mask = mask[: self.max_len]
            frames = frames[: self.max_len]
        return {"x": x, "mask": mask, "frames": frames}

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        path = self.files[idx]
        item = self._load_item(path)
        item["id"] = os.path.basename(path)
        return item


class PairedL2SequenceDataset(L2SequenceDataset):
    """Load aligned raw/calibrated views of the same L2 sequence."""

    def __init__(
        self,
        data_dir: str,
        paired_view_dir: str,
        pattern: str = "*_l2.json",
        max_files: int = 0,
        max_len: int = 0,
    ) -> None:
        super().__init__(data_dir, pattern=pattern, max_files=max_files, max_len=max_len)
        self.paired_view_dir = paired_view_dir
        missing = [
            os.path.basename(path)
            for path in self.files
            if not os.path.exists(os.path.join(self.paired_view_dir, os.path.basename(path)))
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} paired L2 sequences in {paired_view_dir}: {preview}"
            )

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        path = self.files[idx]
        paired_path = os.path.join(self.paired_view_dir, os.path.basename(path))
        item = self._load_item(path)
        paired = self._load_item(paired_path)
        if item["x"].shape != paired["x"].shape:
            raise ValueError(
                f"Paired L2 shape mismatch for {os.path.basename(path)}: "
                f"{item['x'].shape} != {paired['x'].shape}"
            )
        if not np.array_equal(item["frames"], paired["frames"]):
            raise ValueError(
                f"Paired L2 frames differ for {os.path.basename(path)}"
            )
        item["mask"] = np.logical_and(item["mask"], paired["mask"])
        item["x_pair"] = paired["x"]
        item["mask_pair"] = item["mask"].copy()
        item["id"] = os.path.basename(path)
        return item


class SyntheticSequenceDataset(Dataset):
    def __init__(
        self,
        num_seq: int,
        t_max: int,
        feat_dim: int,
        seed: int = 0,
    ) -> None:
        self.num_seq = num_seq
        self.t_max = t_max
        self.feat_dim = feat_dim
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.num_seq

    def _make_seq(self) -> Dict[str, np.ndarray]:
        t = self.rng.randint(max(20, self.t_max // 2), self.t_max + 1)
        x = np.zeros((t, self.feat_dim), dtype=np.float32)
        state = self.rng.randn(self.feat_dim).astype(np.float32)
        drift = self.rng.randn(self.feat_dim).astype(np.float32) * 0.01
        boundaries = sorted(self.rng.choice(np.arange(10, t - 10), size=max(1, t // 80), replace=False))
        bset = set(boundaries)
        for i in range(t):
            if i in bset:
                state += self.rng.randn(self.feat_dim).astype(np.float32) * 0.5
                drift = self.rng.randn(self.feat_dim).astype(np.float32) * 0.01
            state = state + drift + self.rng.randn(self.feat_dim).astype(np.float32) * 0.01
            x[i] = state
        mask = np.ones((t,), dtype=np.bool_)
        return {"x": x, "mask": mask}

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        item = self._make_seq()
        item["id"] = f"synthetic_{idx:05d}"
        return item
