"""Plotting helpers."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_boundary(
    path: str,
    b: np.ndarray,
    b_s: np.ndarray,
    thr_on: float,
    delta_z: Optional[np.ndarray] = None,
    var_t: Optional[np.ndarray] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(len(b))
    fig, ax = plt.subplots(3 if (delta_z is not None or var_t is not None) else 1, 1, figsize=(10, 6), sharex=True)
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
    ax[0].plot(t, b, label="b")
    ax[0].plot(t, b_s, label="b_s")
    ax[0].axhline(thr_on, color="red", linestyle="--", linewidth=1)
    ax[0].set_ylabel("boundary")
    ax[0].legend(loc="upper right")

    row = 1
    if delta_z is not None:
        ax[row].plot(t, delta_z, color="purple")
        ax[row].set_ylabel("delta_z")
        row += 1
    if var_t is not None:
        ax[row].plot(t, var_t, color="green")
        ax[row].set_ylabel("var")

    ax[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
