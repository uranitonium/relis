#!/usr/bin/env python3
"""Make the protocol-frozen aligned auxiliary-trajectory visualization."""

from __future__ import annotations

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

from analyze_auxiliary_state_semantics import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_CANDIDATES,
    DEFAULT_L2,
    DEFAULT_OUT,
    SEED_SPECS,
    collect_records,
    dyadic_tactic,
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def standardize(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    return (values - float(np.mean(values))) / (scale if scale > 1e-9 else 1.0)


def main() -> None:
    seed, config, checkpoint = SEED_SPECS[-1]
    records, _ = collect_records(
        DEFAULT_ANNOTATIONS,
        DEFAULT_CANDIDATES,
        DEFAULT_L2,
        config,
        checkpoint,
        "cuda",
    )
    grouped = defaultdict(list)
    for record in records:
        grouped[record.clip_id].append(record)
    chosen = sorted(grouped, key=lambda clip: (-len(grouped[clip]), clip))[:2]

    colors = {
        "any_attack": "#f0a35b",
        "mutual_pressure": "#6fb38a",
        "pressure_defend": "#6e94c5",
        "other": "#b8b8b8",
    }
    fig, axes = plt.subplots(4, 2, figsize=(10.3, 5.4), sharex="col")
    legend_labels: set[str] = set()
    for column, clip_id in enumerate(chosen):
        segments = sorted(grouped[clip_id], key=lambda record: record.start_frame)
        frames_parts = []
        u_parts = []
        b_parts = []
        for record in segments:
            frames_parts.append(np.linspace(record.start_frame, record.end_frame, len(record.u)))
            u_parts.append(record.u)
            b_parts.append(record.b.reshape(-1))
        frames = np.concatenate(frames_parts)
        u = np.concatenate(u_parts, axis=0)
        b = np.concatenate(b_parts)

        for row in range(4):
            ax = axes[row, column]
            for record in segments:
                label = dyadic_tactic(record.left_tactic, record.right_tactic)
                ax.axvspan(
                    record.start_frame,
                    record.end_frame,
                    color=colors[label],
                    alpha=0.18,
                    linewidth=0,
                )
                legend_labels.add(label)
            if row < 3:
                ax.plot(frames, standardize(u[:, row]), color="#303030", linewidth=1.05)
                ax.set_ylabel(f"$u_{row}$ (z)")
                ax.axhline(0, color="#777777", linewidth=0.45, alpha=0.55)
            else:
                ax.plot(frames, b, color="#9a3b32", linewidth=1.15)
                ax.set_ylabel("score $b$")
                ax.set_ylim(-0.03, 1.03)
            ax.grid(axis="y", alpha=0.18, linewidth=0.45)
            if row == 0:
                ax.set_title(f"{clip_id}: {len(segments)} labeled segments", fontsize=9)
        axes[-1, column].set_xlabel("source frame")

    ordered_labels = sorted(legend_labels)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[label], alpha=0.35)
        for label in ordered_labels
    ]
    fig.legend(
        handles,
        [label.replace("_", " ") for label in ordered_labels],
        ncol=len(ordered_labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        frameon=False,
    )
    fig.suptitle(
        f"ReLIS auxiliary trajectories aligned with human interaction-state labels (seed {seed})",
        y=0.995,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output = DEFAULT_OUT / "aligned_auxiliary_trajectories.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    counts = Counter(record.clip_id for record in records)
    print(f"Wrote {output}")
    print("Selected:", ", ".join(f"{clip} ({counts[clip]})" for clip in chosen))


if __name__ == "__main__":
    main()
