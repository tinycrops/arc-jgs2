"""Render the step-spectrum gate probe results from runs/corpus-field/step_gate.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN = Path("runs/corpus-field/step_gate.jsonl")
OUT = Path("runs/corpus-field/step_gate.png")

SURFACE = "#ffffff"
INK = "#1a1a19"
INK_2 = "#5f5e56"
BLUE = "#2a78d6"


def main() -> None:
    rows = [json.loads(line) for line in IN.read_text().splitlines()]
    nores = [r for r in rows if "resize" not in r["tags"]]

    groups = [
        ("solved: color map", [r["d_step"] for r in rows if "color_map" in r["plan"]]),
        ("solved: geometric", [r["d_step"] for r in rows if "whole_grid" in r["plan"]]),
        ("solved: crop", [r["d_step"] for r in rows if r["plan"].startswith("crop")]),
        ("abstained (all)", [r["d_step"] for r in rows if not r["plan"]]),
        ("count-shift, no resize", [r["d_step"] for r in nores if "object_count_shift" in r["tags"]]),
        ("no count-shift, no resize", [r["d_step"] for r in nores if "object_count_shift" not in r["tags"]]),
    ]

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    for i, (name, vals) in enumerate(groups):
        y = len(groups) - 1 - i
        vals_arr = np.asarray(vals)
        jitter = rng.uniform(-0.16, 0.16, size=len(vals_arr))
        alpha = 0.9 if len(vals_arr) < 20 else 0.25
        ax.scatter(vals_arr, y + jitter, s=14, color=INK_2, alpha=alpha, linewidths=0)
        med = float(np.median(vals_arr))
        ax.scatter([med], [y], s=90, marker="D", color=BLUE, zorder=3)
        ax.annotate(f"median {med:.2f}", (med, y + 0.30), color=BLUE, fontsize=9, ha="center")

    ax.set_yticks(range(len(groups)), [g for g, _ in reversed(groups)], color=INK)
    ax.set_xlabel("d_step: distance between input-stack and output-stack step spectra", color=INK_2)
    ax.set_title(
        "Step-spectrum conservation gates families, not solvability:\n"
        "recolor conserves exactly, geometric nearly, crop breaks it; count-shift tasks sit higher",
        color=INK, fontsize=11,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d8d7cd")
    ax.tick_params(colors=INK_2)
    ax.grid(axis="x", color="#eceae2", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
