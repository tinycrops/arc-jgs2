"""Propagate the serialized corpus into a wave field and analyze its nodes.

Setup: every cell of the radial disk embedding (corpus position -> radius,
color-role slot -> angle) is a coherent monochromatic point source; the room
field is u(x) = sum_j exp(ik|x-p_j|)/sqrt(|x-p_j|). Nodes = destructive
interference (|u| ~ 0), antinodes = constructive maxima. This is the scalar
Huygens idealization of "particles emitting waves"; a time-domain simulation
time-averages to the same standing structure.

What the analysis means, exactly (Jacobi-Anger): the m-th angular mode of the
far field at wavenumber k is

    A_m(k) = | sum_s e^{2*pi*i*m*s/10} * sum_{j in slot s} J_m(k * r_j) |

-- the m-th DFT of a BESSEL-WEIGHTED occupancy profile. At small k this
degenerates to the flat occupancy spectrum we already use; sweeping k probes
where along the corpus (radius) each color's mass sits. So the room's
node/antinode structure is a position x color joint transform, and its
magnitudes inherit the recolor invariance (rotation only rotates the pattern).

Tests that can lose:
1. Invariance: nodal statistics (|u| quantiles, nodal-area fraction) of the
   room must be unchanged when the corpus is recolored (rotated by k slots),
   up to grid discretization.
2. Information: per-task multi-k wave features vs the flat spectra on the
   own-output retrieval protocol. If retrieval does not improve, the wave
   layer is a re-derivation, not a new instrument.

Outputs: printed stats + runs/corpus-field/wave_room.png (the room, original
vs recolored) + runs/corpus-field/wave_features.jsonl.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np

from arc_jgs2.corpusfield import grid_slots
from arc_jgs2.loaders import Task, load_tasks

DATA = Path("data/ARC-AGI/data/training")
OUT = Path("runs/corpus-field")

ROOM_N = 1024
ROOM_HALF = 1.4  # room extends a bit beyond the unit disk
WAVELEN = 0.1  # room units
K_ROOM = 2 * math.pi / WAVELEN
K_SWEEP = [2 * math.pi * f for f in (1, 2, 4, 8)]  # per-task far-field dial
N_ANGLES = 360
MODES = list(range(11))  # angular modes m = 0..10


def side_stream(task: Task, side: str) -> list[int]:
    grids = [p.input for p in task.train] if side == "in" else [p.output for p in task.train if p.output is not None]
    return [s for g in grids for s in grid_slots(g)]


def disk_xy(slots: list[int], rotation: int = 0) -> tuple[np.ndarray, np.ndarray]:
    n = len(slots)
    idx = np.arange(n)
    r = (n - 1 - idx) / (n - 1) if n > 1 else np.zeros(1)
    ang = np.radians(-90 + ((np.asarray(slots) + rotation) % 10) * 36)
    return r * np.cos(ang), r * np.sin(ang)


def room_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|u| on the room grid via FFT convolution of the binned sources with the
    outgoing-wave kernel exp(ik*rho)/sqrt(rho)."""
    edges = np.linspace(-ROOM_HALF, ROOM_HALF, ROOM_N + 1)
    src, _, _ = np.histogram2d(x, y, bins=[edges, edges])

    m = 2 * ROOM_N
    dx = 2 * ROOM_HALF / ROOM_N
    freq_idx = np.fft.fftfreq(m, d=1.0) * m  # wrapped integer offsets
    ox, oy = np.meshgrid(freq_idx * dx, freq_idx * dx, indexing="ij")
    rho = np.hypot(ox, oy)
    kernel = np.exp(1j * K_ROOM * rho) / np.sqrt(rho + 0.5 * dx)

    pad = np.zeros((m, m), dtype=complex)
    pad[:ROOM_N, :ROOM_N] = src
    u = np.fft.ifft2(np.fft.fft2(pad) * np.fft.fft2(kernel))[:ROOM_N, :ROOM_N]
    return np.abs(u)


def nodal_stats(mag: np.ndarray) -> dict[str, float]:
    med = float(np.median(mag))
    return {
        "median": med,
        "q10": float(np.quantile(mag, 0.10)),
        "q90": float(np.quantile(mag, 0.90)),
        "max": float(mag.max()),
        "nodal_frac": float((mag < 0.1 * med).mean()),  # near-silent area
        "antinode_frac": float((mag > 3.0 * med).mean()),
    }


def wave_features(slots: list[int]) -> list[float]:
    """Rotation-invariant far-field magnitudes |A_m(k)| / N over the k sweep."""
    if not slots:
        return [0.0] * (len(K_SWEEP) * len(MODES))
    x, y = disk_xy(slots)
    phi = np.linspace(0, 2 * math.pi, N_ANGLES, endpoint=False)
    proj = np.outer(x, np.cos(phi)) + np.outer(y, np.sin(phi))  # N x angles
    feats: list[float] = []
    for k in K_SWEEP:
        far = np.exp(-1j * k * proj).sum(axis=0) / len(slots)
        spec = np.abs(np.fft.fft(far)) / N_ANGLES
        feats.extend(float(spec[m]) for m in MODES)
    return feats


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def _retrieval(in_vecs, out_vecs, label: str) -> None:
    n = len(in_vecs)
    ranks = []
    for i, q in enumerate(in_vecs):
        d_own = _dist(q, out_vecs[i])
        closer = tied = 0
        for j in range(n):
            if j == i:
                continue
            d = _dist(q, out_vecs[j])
            if d < d_own - 1e-12:
                closer += 1
            elif d < d_own + 1e-12:
                tied += 1
        ranks.append(1 + closer + 0.5 * tied)
    top1 = sum(1 for r in ranks if r <= 1)
    top5 = sum(1 for r in ranks if r <= 5)
    print(
        f"  {label:<30} top1={top1:>3}/{n} ({top1 / n:.1%})  "
        f"top5={top5:>3} ({top5 / n:.1%})  median rank={statistics.median(ranks):.0f}"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(DATA)
    # full corpus stream in task order (inputs and outputs interleaved per task)
    corpus: list[int] = []
    for t in tasks:
        for p in t.train:
            corpus.extend(grid_slots(p.input))
            if p.output is not None:
                corpus.extend(grid_slots(p.output))

    # --- 1. the room, and nodal invariance under recoloring ----------------
    print(f"room: {ROOM_N}x{ROOM_N}, wavelength {WAVELEN}, {len(corpus)} sources")
    mag0 = room_field(*disk_xy(corpus, rotation=0))
    mag3 = room_field(*disk_xy(corpus, rotation=3))
    s0, s3 = nodal_stats(mag0), nodal_stats(mag3)
    print("nodal statistics (original vs recolored by +3 -- claim: identical up to grid):")
    for key in s0:
        rel = abs(s0[key] - s3[key]) / (abs(s0[key]) or 1.0)
        print(f"  {key:<14} {s0[key]:>12.4f} {s3[key]:>12.4f}   rel diff {rel:.2e}")
    np.savez_compressed(OUT / "wave_room.npz", mag0=mag0.astype(np.float32), mag3=mag3.astype(np.float32))

    # --- 2. do multi-k wave features beat the flat spectra? ----------------
    print("\nown-output retrieval with wave features (flat-spectra reference: 14.2% / 8.5%):")
    wf_in = [wave_features(side_stream(t, "in")) for t in tasks]
    wf_out = [wave_features(side_stream(t, "out")) for t in tasks]
    _retrieval(wf_in, wf_out, "wave features (4k x 11m)")

    # size + wave tiebreak, same protocol as the spectrum tiebreak (46%)
    sizes_in = [math.log(1 + sum(len(p.input) * len(p.input[0]) for p in t.train)) for t in tasks]
    sizes_out = [
        math.log(1 + sum(len(p.output) * len(p.output[0]) for p in t.train if p.output)) for t in tasks
    ]
    n = len(tasks)
    ranks = []
    for i in range(n):
        key_own = (abs(sizes_in[i] - sizes_out[i]), _dist(wf_in[i], wf_out[i]))
        closer = tied = 0
        for j in range(n):
            if j == i:
                continue
            key = (abs(sizes_in[i] - sizes_out[j]), _dist(wf_in[i], wf_out[j]))
            if key < key_own:
                closer += 1
            elif key == key_own:
                tied += 1
        ranks.append(1 + closer + 0.5 * tied)
    top1 = sum(1 for r in ranks if r <= 1)
    top5 = sum(1 for r in ranks if r <= 5)
    print(
        f"  {'size, wave tiebreak':<30} top1={top1:>3}/{n} ({top1 / n:.1%})  "
        f"top5={top5:>3} ({top5 / n:.1%})  median rank={statistics.median(ranks):.0f}"
    )

    with (OUT / "wave_features.jsonl").open("w") as f:
        for t, wi, wo in zip(tasks, wf_in, wf_out):
            f.write(json.dumps({"task_id": t.task_id, "wave_in": wi, "wave_out": wo}) + "\n")
    print(f"\nwrote {OUT / 'wave_features.jsonl'} and {OUT / 'wave_room.npz'}")


if __name__ == "__main__":
    main()
