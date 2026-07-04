"""Per-grid wave fingerprints: does the k-dial earn its keep at grid scale?

At task scale the wave features beat the flat spectra standalone (17.5% vs
14.2% top-1) but tied as a size-tiebreak. Hypothesis: grid scale is where the
radial axis means something concrete (scan position ~ row), so the k-sweep
probes vertical color layout and the margin over flat spectra should GROW.

Two protocols, both tie-aware, both with baselines that can win instead:

A. Pair retrieval: each train INPUT grid's fingerprint retrieves its own
   OUTPUT grid among all ~1300 outputs. Chance ~0.08%.
B. Task cohesion: each train input grid's nearest other input grid -- is it
   from the same task? Chance ~0.2%.

Feature sets: log-cells (size), flat spectra (normalized occupancy spectrum +
step spectrum, 11 dims), wave fingerprints (4 wavenumbers x 11 modes, 44
dims), and size-tiebreak combos.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

import numpy as np

from arc_jgs2.corpusfield import angular_occupancy, grid_slots, power_spectrum, step_profile
from arc_jgs2.loaders import load_tasks
from arc_jgs2.wavefield import wave_fingerprint

DATA = Path("data/ARC-AGI/data/training")


def flat_spectra(slots: list[int]) -> list[float]:
    occ = power_spectrum(angular_occupancy(slots))
    total = occ[0] or 1.0
    steps = step_profile([slots])
    n_steps = sum(steps) or 1.0
    return [v / total for v in occ[1:6]] + power_spectrum([c / n_steps for c in steps])


def dist_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    return np.sqrt(np.maximum(d2, 0.0))


def ranks_from(d: np.ndarray, own: np.ndarray, exclude_self: bool = True) -> np.ndarray:
    eps = 1e-12
    n = d.shape[0]
    ranks = np.empty(n)
    for i in range(n):
        row, d0 = d[i], own[i]
        closer = int((row < d0 - eps).sum())
        tied = int((np.abs(row - d0) <= eps).sum())
        if exclude_self:
            tied -= 1  # own entry ties with itself
        ranks[i] = 1 + closer + 0.5 * max(tied, 0)
    return ranks


def lex_ranks(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    eps = 1e-12
    n = p.shape[0]
    ranks = np.empty(n)
    for i in range(n):
        p0, s0 = p[i, i], s[i, i]
        prow, srow = p[i], s[i]
        closer = int(((prow < p0 - eps) | ((np.abs(prow - p0) <= eps) & (srow < s0 - eps))).sum())
        tied = int(((np.abs(prow - p0) <= eps) & (np.abs(srow - s0) <= eps)).sum()) - 1
        ranks[i] = 1 + closer + 0.5 * max(tied, 0)
    return ranks


def report(label: str, ranks: np.ndarray) -> None:
    n = len(ranks)
    top1 = int((ranks <= 1).sum())
    top5 = int((ranks <= 5).sum())
    print(
        f"  {label:<28} top1={top1:>4}/{n} ({top1 / n:.1%})  "
        f"top5={top5:>4} ({top5 / n:.1%})  median rank={statistics.median(ranks):.0f}"
    )


def main() -> None:
    tasks = load_tasks(DATA)
    in_slots, out_slots, task_of = [], [], []
    for ti, t in enumerate(tasks):
        for p in t.train:
            if p.output is None:
                continue
            in_slots.append(grid_slots(p.input))
            out_slots.append(grid_slots(p.output))
            task_of.append(ti)
    n = len(in_slots)
    task_of_arr = np.asarray(task_of)
    print(f"train pairs: {n} across {len(tasks)} tasks")

    size_in = np.asarray([[math.log(1 + len(s))] for s in in_slots])
    size_out = np.asarray([[math.log(1 + len(s))] for s in out_slots])
    flat_in = np.asarray([flat_spectra(s) for s in in_slots])
    flat_out = np.asarray([flat_spectra(s) for s in out_slots])
    wave_in = np.asarray([wave_fingerprint(s) for s in in_slots])
    wave_out = np.asarray([wave_fingerprint(s) for s in out_slots])

    # --- protocol A: input grid -> own output grid -------------------------
    print(f"\nA. pair retrieval, input->own output (chance top1 {1 / n:.2%}):")
    d_size = dist_matrix(size_in, size_out)
    d_flat = dist_matrix(flat_in, flat_out)
    d_wave = dist_matrix(wave_in, wave_out)
    report("grid size alone", ranks_from(d_size, np.diag(d_size)))
    report("flat spectra", ranks_from(d_flat, np.diag(d_flat)))
    report("wave fingerprint", ranks_from(d_wave, np.diag(d_wave)))
    report("size, flat tiebreak", lex_ranks(d_size, d_flat))
    report("size, wave tiebreak", lex_ranks(d_size, d_wave))

    # --- protocol B: nearest other input grid is same-task? ----------------
    same_task_frac = float(np.mean([(task_of_arr == t).sum() - 1 for t in task_of_arr]) / (n - 1))
    print(f"\nB. task cohesion, nearest other input grid same-task (chance {same_task_frac:.2%}):")
    for label, vecs in (("grid size alone", size_in), ("flat spectra", flat_in), ("wave fingerprint", wave_in)):
        d = dist_matrix(vecs, vecs)
        np.fill_diagonal(d, np.inf)
        hits = 0.0
        for i in range(n):
            row = d[i]
            m = row.min()
            nearest = np.flatnonzero(np.abs(row - m) <= 1e-12)  # split ties evenly
            hits += float((task_of_arr[nearest] == task_of_arr[i]).mean())
        print(f"  {label:<28} same-task@1 = {hits / n:.1%}")


if __name__ == "__main__":
    main()
