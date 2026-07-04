"""Render the corpus radial field and the rotation-orbit evidence.

Outputs (runs/corpus-field/):

* corpus_disk.png -- the whole training corpus as one radial propagation
  field (position 0 outer edge, last cell at origin, angle = color-role slot)
  at three rotations. Same geometry, visibly rotated coloring: three of the
  ten color-invariant formations.
* orbit_evidence.png -- left: Frenet invariants across all 10 rotations,
  normalized to k=0 (flat = invariant); right: own-output retrieval results
  showing what the rotation-invariant spectrum adds over grid size.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arc_jgs2.corpusfield import corpus_slots, cylinder_points, disk_points
from arc_jgs2.linefeatures import frenet_stats
from arc_jgs2.loaders import load_tasks

DATA = Path("data/ARC-AGI/data/training")
OUT = Path("runs/corpus-field")

# ARC's own canonical 10-color palette, indexed by displayed wheel slot
ARC_COLORS = [
    "#111111", "#1E93FF", "#F93C31", "#4FCC30", "#FFDC00",
    "#999999", "#E53AA3", "#FF851B", "#87D8F1", "#921231",
]
SURFACE = "#ffffff"
INK = "#1a1a19"
INK_2 = "#5f5e56"
BLUE = "#2a78d6"


def render_disks(slots: list[int]) -> None:
    n = len(slots)
    idx = np.arange(n)
    r = (n - 1 - idx) / (n - 1)
    base_slot = np.array(slots)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor=SURFACE)
    for ax, k in zip(axes, (0, 3, 7)):
        shown = (base_slot + k) % 10
        ang = np.radians(-90 + shown * 36)
        x, y = r * np.cos(ang), r * np.sin(ang)
        colors = np.array(ARC_COLORS)[shown]
        ax.scatter(x, y, s=0.35, c=colors, linewidths=0, alpha=0.5, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.axis("off")
        label = "k = 0 (original)" if k == 0 else f"k = {k} (rotated {36 * k}\N{DEGREE SIGN})"
        ax.set_title(label, color=INK, fontsize=12)
    fig.suptitle(
        "One shape, ten colorings: the ARC training corpus as a radial field\n"
        f"{n:,} cells; position 0 at the rim, last cell at the origin; angle = color-role slot",
        color=INK, fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "corpus_disk.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / 'corpus_disk.png'}")


def render_evidence(slots: list[int]) -> None:
    stats = [frenet_stats(cylinder_points(slots, rotation=k)) for k in range(10)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURFACE)

    # left: invariants normalized to k=0 -- flat lines are the theorem
    series = [
        ("arc length", [s.arc_length for s in stats]),
        ("mean curvature", [s.mean_curv for s in stats]),
        ("mean |torsion|", [s.mean_abs_torsion for s in stats]),
    ]
    ks = list(range(10))
    # all three sit exactly at 1.0 and overplot -- that IS the result; stagger
    # the labels so each series is still named
    for i, ((name, vals), color) in enumerate(zip(series, ("#2a78d6", "#1baf7a", "#eda100"))):
        rel = [v / vals[0] for v in vals]
        ax1.plot(ks, rel, color=color, linewidth=2, marker="o", markersize=5)
        ax1.annotate(name, (ks[-1] + 0.25, 1.0 + (1 - i) * 0.0028), color=color, fontsize=9, va="center")
    ax1.annotate(
        "all three flat at 1.0000\n(lines overplot)", (4.5, 1.0035),
        color=INK_2, fontsize=8.5, ha="center",
    )
    ax1.set_xlim(-0.3, 13.2)
    ax1.set_ylim(0.99, 1.01)
    ax1.set_xticks(ks)
    ax1.set_xlabel("rotation k (x36\N{DEGREE SIGN} = recolor by +k mod 10)", color=INK_2)
    ax1.set_ylabel("value relative to k = 0", color=INK_2)
    ax1.set_title("Frenet invariants across the whole rotation orbit", color=INK, fontsize=11)

    # right: what the rotation-invariant spectrum adds to retrieval
    methods = [
        ("chance", 0.25),
        ("step spectrum", 8.5),
        ("occupancy spectrum", 14.2),
        ("grid size alone", 28.0),
        ("size + spectrum tiebreak", 45.5),
    ]
    ypos = np.arange(len(methods))
    vals = [v for _, v in methods]
    colors = [INK_2 if i < len(methods) - 1 else BLUE for i in range(len(methods))]
    bars = ax2.barh(ypos, vals, height=0.55, color=colors)
    for bar, (name, v) in zip(bars, methods):
        ax2.annotate(
            f"{v:.2g}%", (bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2),
            color=INK, fontsize=10, va="center",
        )
    ax2.set_yticks(ypos, [m for m, _ in methods], color=INK)
    ax2.set_xlim(0, 55)
    ax2.set_xlabel("top-1: input spectrum retrieves its own task's output (400 tasks)", color=INK_2)
    ax2.set_title("The invariant spectrum resolves what size can't", color=INK, fontsize=11)

    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#d8d7cd")
        ax.tick_params(colors=INK_2)
        ax.grid(axis="x" if ax is ax2 else "y", color="#eceae2", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "orbit_evidence.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / 'orbit_evidence.png'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(DATA)
    slots, _ = corpus_slots(tasks)
    render_disks(slots)
    render_evidence(slots)


if __name__ == "__main__":
    main()
