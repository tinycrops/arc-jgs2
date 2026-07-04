"""Render the corpus wave room from runs/corpus-field/wave_room.npz."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

IN = Path("runs/corpus-field/wave_room.npz")
OUT = Path("runs/corpus-field/wave_room.png")

SURFACE = "#ffffff"
INK = "#1a1a19"
# one-hue sequential ramp (blue), light -> dark
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d2f5c"]
)


def main() -> None:
    data = np.load(IN)
    mag0, mag3 = data["mag0"], data["mag3"]
    lo, hi = np.quantile(mag0, 0.02), np.quantile(mag0, 0.998)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), facecolor=SURFACE)
    for ax, mag, title in (
        (axes[0], mag0, "original coloring"),
        (axes[1], mag3, "recolored by +3 (pattern rotates 108\N{DEGREE SIGN}, nothing else changes)"),
    ):
        img = np.clip((np.log1p(mag.T) - np.log1p(lo)) / (np.log1p(hi) - np.log1p(lo)), 0, 1)
        ax.imshow(img, origin="lower", cmap=BLUES, extent=(-1.4, 1.4, -1.4, 1.4))
        ax.set_title(title, color=INK, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        "The corpus as a wave room: 316,468 coherent point sources on the radial disk\n"
        "(dark = antinodes / constructive interference, light = nodes; wavelength 0.1 room units)",
        color=INK, fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT, dpi=170, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
