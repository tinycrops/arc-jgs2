"""Does a smarter grouping/serialization of the corpus cylinder gain anything?

The stack currently concatenates tasks in dataset order (hash-sorted filenames,
i.e. effectively random). Everything we score downstream (per-task spectra,
retrieval, solvability) is order-INVARIANT by construction, so reordering
cannot move those numbers. The one objective where ordering can matter is
description length: a grouping is "more efficient" exactly when the serialized
stream costs fewer bits, because similar material sits adjacent and a
sequential model exploits it. That is what a "model that invents grouping
procedures" would be optimizing, so we measure the headroom directly.

Instruments, each order-sensitive in a different way:
* zlib -9 (32KB window: local grouping), lzma (long-range grouping)
* an online adaptive bigram coder over the slot stream (bits/cell = the
  sequential-prediction cost a simple learner actually pays; add-1 smoothing,
  counts updated after each symbol)
* junction arc length: the order-dependent part of the corpus polyline

Two independent serialization axes:
* task ordering: dataset order, random controls, size-sorted, greedy
  nearest-neighbor chain in task-spectrum space (a cheap stand-in for the
  best grouping a model could invent)
* within-grid scan: row-major (current), boustrophedon, column-major

Prediction that can lose either way: if task reordering moves bits/cell by
<~1% while scan order moves it more, the gains live inside grids, not in the
stacking order, and a grouping-inventor model is not worth building yet.
"""

from __future__ import annotations

import lzma
import math
import random
import statistics
import zlib
from pathlib import Path

from arc_jgs2.corpusfield import grid_slots, task_spectrum_features
from arc_jgs2.grids import Grid, dims
from arc_jgs2.linefeatures import _slot_angle
from arc_jgs2.loaders import Task, load_tasks

DATA = Path("data/ARC-AGI/data/training")

SPEC_KEYS = [f"occ_{s}_k{k}" for s in ("in", "out") for k in range(1, 6)] + [
    f"step_{s}_k{k}" for s in ("in", "out") for k in range(6)
]


def scan(grid: Grid, mode: str) -> list[int]:
    h, w = dims(grid)
    if mode == "row":
        cells = [grid[r][c] for r in range(h) for c in range(w)]
    elif mode == "boustrophedon":
        cells = [grid[r][c] for r in range(h) for c in (range(w) if r % 2 == 0 else range(w - 1, -1, -1))]
    elif mode == "column":
        cells = [grid[r][c] for c in range(w) for r in range(h)]
    else:
        raise ValueError(mode)
    # co-rotation is per grid, so scan order does not change the role map
    from arc_jgs2.linefeatures import role_order

    slot = {v: i for i, v in enumerate(role_order(grid))}
    return [slot[v] for v in cells]


def task_stream(task: Task, mode: str) -> list[int]:
    out: list[int] = []
    for pair in task.train:
        out.extend(scan(pair.input, mode))
        if pair.output is not None:
            out.extend(scan(pair.output, mode))
    return out


def stream_for(tasks: list[Task], order: list[int], mode: str) -> list[int]:
    return [s for i in order for s in task_stream(tasks[i], mode)]


def bigram_bits(slots: list[int]) -> float:
    """Cumulative cost (bits/cell) of an online add-1 bigram coder."""
    counts = [[1] * 10 for _ in range(10)]
    totals = [10] * 10
    prev = 0
    bits = 0.0
    for s in slots:
        bits -= math.log2(counts[prev][s] / totals[prev])
        counts[prev][s] += 1
        totals[prev] += 1
        prev = s
    return bits / len(slots)


def junction_arc(tasks: list[Task], order: list[int], mode: str) -> float:
    """Arc length of only the segments joining consecutive grids in the stack."""
    total = 0.0
    prev_last: int | None = None
    for i in order:
        for pair in tasks[i].train:
            for grid in (pair.input, pair.output):
                if grid is None:
                    continue
                s = scan(grid, mode)
                if prev_last is not None:
                    a, b = _slot_angle(prev_last), _slot_angle(s[0])
                    total += math.sqrt(1.0 + (math.cos(a) - math.cos(b)) ** 2 + (math.sin(a) - math.sin(b)) ** 2)
                prev_last = s[-1]
    return total


def measure(slots: list[int]) -> tuple[float, float, float]:
    raw = bytes(slots)
    return (
        len(zlib.compress(raw, 9)) * 8 / len(slots),
        len(lzma.compress(raw, preset=6)) * 8 / len(slots),
        bigram_bits(slots),
    )


def greedy_chain(vecs: list[list[float]]) -> list[int]:
    """Nearest-neighbor chain through spectrum space, started at task 0."""
    n = len(vecs)
    unused = set(range(1, n))
    order = [0]
    while unused:
        cur = vecs[order[-1]]
        nxt = min(unused, key=lambda j: sum((a - b) ** 2 for a, b in zip(cur, vecs[j])))
        unused.discard(nxt)
        order.append(nxt)
    return order


def main() -> None:
    tasks = load_tasks(DATA)
    n = len(tasks)
    identity = list(range(n))

    feats = [task_spectrum_features(t) for t in tasks]
    vecs = [[f[k] for k in SPEC_KEYS] for f in feats]
    sizes = [sum(len(p.input) * len(p.input[0]) for p in t.train) for t in tasks]

    orderings: list[tuple[str, list[int]]] = [
        ("dataset order (hash ~ random)", identity),
        ("size-sorted", sorted(identity, key=lambda i: sizes[i])),
        ("spectrum greedy chain", greedy_chain(vecs)),
    ]

    print("task-ordering axis (row-major scan held fixed):")
    print(f"  {'ordering':<32}{'zlib':>9}{'lzma':>9}{'bigram':>9}{'junction arc':>14}")
    for name, order in orderings:
        z, x, b = measure(stream_for(tasks, order, "row"))
        j = junction_arc(tasks, order, "row")
        print(f"  {name:<32}{z:>9.4f}{x:>9.4f}{b:>9.4f}{j:>14.1f}")

    rng = random.Random(0)
    zs, xs, bs = [], [], []
    for _ in range(5):
        order = identity[:]
        rng.shuffle(order)
        z, x, b = measure(stream_for(tasks, order, "row"))
        zs.append(z)
        xs.append(x)
        bs.append(b)
    print(
        f"  {'random shuffle (mean of 5)':<32}{statistics.mean(zs):>9.4f}{statistics.mean(xs):>9.4f}"
        f"{statistics.mean(bs):>9.4f}{'':>14}"
    )

    print("\nwithin-grid scan axis (dataset order held fixed):")
    print(f"  {'scan':<32}{'zlib':>9}{'lzma':>9}{'bigram':>9}")
    for mode in ("row", "boustrophedon", "column"):
        z, x, b = measure(stream_for(tasks, identity, mode))
        print(f"  {mode:<32}{z:>9.4f}{x:>9.4f}{b:>9.4f}")

    print("\nunits: bits/cell (raw stream = 8.0; log2(10) = 3.32 is the zero-model bound)")


if __name__ == "__main__":
    main()
