#!/usr/bin/env python3
"""Export compact paper-ready human agreement summaries."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from common import read_csv, write_csv


METRICS = [
    ("activity_frame_agreement", "Activity frame agreement"),
    ("boundary_f1_pm3", "Boundary F1 +/-3"),
    ("boundary_f1_pm5", "Boundary F1 +/-5"),
    ("boundary_f1_pm10", "Boundary F1 +/-10"),
]


def f3(value: float) -> str:
    return f"{value:.3f}"


def markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def floats(rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for row in rows:
        try:
            out.append(float(row[key]))
        except Exception:
            pass
    return out


def quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": "0.000", "median": "0.000", "p25": "0.000", "p75": "0.000"}
    return {
        "n": len(values),
        "mean": f3(statistics.mean(values)),
        "median": f3(statistics.median(values)),
        "p25": f3(quantile(values, 0.25)),
        "p75": f3(quantile(values, 0.75)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agreement", default="paper_experiments/annotations/agreement_report.csv")
    parser.add_argument("--out-dir", default="paper_experiments/paper_ready/foil")
    args = parser.parse_args()

    rows = read_csv(args.agreement)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for key, label in METRICS:
        row = {"metric": label, **summarize(floats(rows, key))}
        metric_rows.append(row)
    write_csv(out_dir / "human_agreement_summary.csv", metric_rows)
    (out_dir / "human_agreement_summary.md").write_text(markdown_table(metric_rows), encoding="utf-8")

    coverage_diff = [abs(value) for value in floats(rows, "active_coverage_difference_b_minus_a")]
    state_count_diff = [abs(value) for value in floats(rows, "state_count_difference_b_minus_a")]
    end_distance = floats(rows, "exchange_end_distance")
    extra_rows = [
        {"metric": "Absolute active coverage difference", **summarize(coverage_diff)},
        {"metric": "Absolute state-count difference", **summarize(state_count_diff)},
        {"metric": "Exchange-end frame distance", **summarize(end_distance)},
    ]
    write_csv(out_dir / "human_agreement_aux_summary.csv", extra_rows)
    (out_dir / "human_agreement_aux_summary.md").write_text(markdown_table(extra_rows), encoding="utf-8")
    print(f"Wrote human agreement summaries to {out_dir}")


if __name__ == "__main__":
    main()
