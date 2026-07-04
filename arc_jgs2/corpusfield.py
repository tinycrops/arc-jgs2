"""Corpus cylinder rotation orbit and the radial propagation field.

`linefeatures.py` lifts one grid onto the color-wheel cylinder. Here the whole
dataset is stacked into ONE long cylinder polyline: every train grid of every
task, flattened and co-rotated (role order: background -> slot 0, rest by
frequency), concatenated with height = global cell index.

Two constructions on top of that stack:

* Rotation orbit. Rotating the polyline about the cylinder axis by k*36
  degrees is exactly the same object as adding +k (mod 10) to every color-role
  slot -- a cyclic recoloring of the entire corpus. Rotations are rigid
  motions, so every Frenet invariant is exactly preserved: ten formations
  (nine non-identity), one shape. The orbit makes the cyclic part of color
  relabeling a *geometric symmetry* instead of a nuisance. The honest
  boundary: the cylinder geometrizes Z_10, the cyclic shifts; the rest of the
  full permutation group S_10 is handled (approximately) by role order.
* Quotient readout. The invariant content of the orbit is harvested the
  standard harmonic-analysis way: any per-slot profile (e.g. occupancy) has
  DFT magnitudes that are invariant under the cyclic shift. Six numbers
  (k = 0..5 for a real length-10 signal) summarize a profile up to rotation.
* Radial field. Re-embed the stacked sequence in a disk: corpus position 0 on
  the outer edge, position n-1 at the origin, angle = color slot. Sequence
  position becomes wavefront radius (light from a point source, run backward),
  each task occupies an annulus, and the slot-shift symmetry becomes a literal
  rotation of the image -- the log-polar / Fourier-Mellin move from image
  registration, where nuisance transforms become coordinate translations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .grids import Grid, flatten
from .linefeatures import _slot_angle, role_order
from .loaders import Task


@dataclass(frozen=True)
class CorpusSpan:
    """Half-open [start, stop) range one grid occupies in the corpus stack."""

    task_id: str
    kind: str  # "train_in" or "train_out"
    start: int
    stop: int


def grid_slots(grid: Grid) -> list[int]:
    """Co-rotated color-role slots of one grid, row-major."""
    slot = {v: i for i, v in enumerate(role_order(grid))}
    return [slot[v] for v in flatten(grid)]


def corpus_slots(tasks: list[Task]) -> tuple[list[int], list[CorpusSpan]]:
    """Stack every train grid of every task into one slot sequence."""
    slots: list[int] = []
    spans: list[CorpusSpan] = []
    for task in tasks:
        for pair in task.train:
            for kind, grid in (("train_in", pair.input), ("train_out", pair.output)):
                if grid is None:
                    continue
                start = len(slots)
                slots.extend(grid_slots(grid))
                spans.append(CorpusSpan(task.task_id, kind, start, len(slots)))
    return slots, spans


def cylinder_points(slots: list[int], rotation: int = 0) -> list[tuple[float, float, float]]:
    """Lift to the cylinder polyline; `rotation` shifts every slot by +k mod 10,
    which equals rotating the whole polyline about the cylinder axis by k*36 deg."""
    return [
        (float(i), math.cos(_slot_angle((s + rotation) % 10)), math.sin(_slot_angle((s + rotation) % 10)))
        for i, s in enumerate(slots)
    ]


def disk_points(slots: list[int], rotation: int = 0) -> list[tuple[float, float]]:
    """Radial propagation field: position 0 at the outer edge (r=1), position
    n-1 at the origin, angle = color slot. Slot shifts rotate the image."""
    n = len(slots)
    out: list[tuple[float, float]] = []
    for i, s in enumerate(slots):
        r = (n - 1 - i) / (n - 1) if n > 1 else 0.0
        a = _slot_angle((s + rotation) % 10)
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


def angular_occupancy(slots: list[int]) -> list[float]:
    """How much of the sequence sits in each of the 10 angular slots."""
    counts = [0.0] * 10
    for s in slots:
        counts[s] += 1.0
    return counts


def step_profile(slots_per_grid: list[list[int]]) -> list[float]:
    """Histogram of slot-to-slot steps (delta mod 10) along each grid's
    polyline. Steps never cross grid boundaries. Already cyclic-shift
    invariant; its spectrum is kept as a compact fixed-length descriptor."""
    counts = [0.0] * 10
    for slots in slots_per_grid:
        for a, b in zip(slots, slots[1:]):
            counts[(b - a) % 10] += 1.0
    return counts


def task_spectrum_features(task: Task) -> dict[str, float]:
    """Rotation-invariant, size-normalized spectral profile of one task.

    Occupancy spectra are normalized by their k=0 term (total cells), step
    spectra by the step count, so the features describe color-role *dynamics*
    rather than grid size.
    """
    out: dict[str, float] = {}
    for side, grids in (
        ("in", [p.input for p in task.train]),
        ("out", [p.output for p in task.train if p.output is not None]),
    ):
        slots_per_grid = [grid_slots(g) for g in grids]
        occ = angular_occupancy([s for slots in slots_per_grid for s in slots])
        occ_spec = power_spectrum(occ)
        total = occ_spec[0] or 1.0
        for k in range(1, 6):
            out[f"occ_{side}_k{k}"] = occ_spec[k] / total

        steps = step_profile(slots_per_grid)
        n_steps = sum(steps) or 1.0
        step_spec = power_spectrum([c / n_steps for c in steps])
        for k in range(6):
            out[f"step_{side}_k{k}"] = step_spec[k]
    return out


def power_spectrum(profile: list[float]) -> list[float]:
    """DFT magnitudes |X_k|, k = 0..5, of a real length-10 per-slot profile.

    Exactly invariant under cyclic shifts of the profile -- the complete set of
    rotation-invariant linear features of the orbit.
    """
    n = len(profile)
    out: list[float] = []
    for k in range(n // 2 + 1):
        re = sum(x * math.cos(2 * math.pi * k * i / n) for i, x in enumerate(profile))
        im = -sum(x * math.sin(2 * math.pi * k * i / n) for i, x in enumerate(profile))
        out.append(math.hypot(re, im))
    return out
