"""Corpus-cylinder rotation orbit: verify the invariance claim at full-corpus
scale, then ask whether the rotation-invariant spectrum is USEFUL.

Three questions, each designed so it can lose:

1. Claim check. Stack all 400 training tasks into one cylinder polyline and
   compute Frenet invariants under all 10 rotations. The claim (rotation =
   cyclic recoloring = rigid motion) predicts identical numbers.

2. Identity retrieval. For each task, does the input-stack spectrum retrieve
   its OWN task's output-stack among all 400 outputs? If yes, the invariant
   profile is (partly) conserved across the task's transformation -- task
   identity survives in rotation-invariant color-dynamics space. Baselines:
   chance (median rank ~200), a size-only feature, and a shuffled-pairing
   control.

3. Solvability. Do spectrum features separate solved vs abstained tasks
   beyond the known grid-size confound (log_cells AUC ~0.84)?

Honest confound, stated up front: role order already sorts colors by
frequency, so the occupancy profile is nearly a sorted histogram and the
occupancy spectrum adds little beyond that sorting. The genuinely new content
is the STEP spectrum (slot-to-slot transition dynamics along the polyline).
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

from arc_jgs2.corpusfield import corpus_slots, cylinder_points, task_spectrum_features
from arc_jgs2.linefeatures import frenet_stats
from arc_jgs2.loaders import load_tasks

DATA = Path("data/ARC-AGI/data/training")
SOLUTIONS = Path("runs/unionmap-training-solutions/solutions.jsonl")
OUT = Path("runs/corpus-field")

OCC_KEYS = [f"occ_{{}}_k{k}" for k in range(1, 6)]
STEP_KEYS = [f"step_{{}}_k{k}" for k in range(6)]


def _vec(feats: dict[str, float], side: str, keys: list[str]) -> list[float]:
    return [feats[k.format(side)] for k in keys]


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _retrieval(in_vecs: list[list[float]], out_vecs: list[list[float]], label: str) -> None:
    """Mid-rank for ties: exact-size duplicates must share credit, not all claim rank 1."""
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
        f"  {label:<28} top1={top1:>3}/{n} ({top1 / n:.1%})  "
        f"top5={top5:>3} ({top5 / n:.1%})  median rank={statistics.median(ranks):.0f}"
    )


def _auc(pos: list[float], neg: list[float]) -> float:
    c = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return c / (len(pos) * len(neg)) if pos and neg else float("nan")


def main() -> None:
    tasks = load_tasks(DATA)
    print(f"tasks: {len(tasks)}")

    # --- 1. claim check at corpus scale -----------------------------------
    slots, spans = corpus_slots(tasks)
    print(f"\ncorpus cylinder: {len(slots)} cells across {len(spans)} grids")
    print("Frenet invariants under all 10 rotations (claim: identical):")
    rows = []
    for k in range(10):
        g = frenet_stats(cylinder_points(slots, rotation=k))
        rows.append((k, g.arc_length, g.mean_curv, g.mean_torsion))
    spread = {
        "arc_length": max(r[1] for r in rows) - min(r[1] for r in rows),
        "mean_curv": max(r[2] for r in rows) - min(r[2] for r in rows),
        "mean_torsion": max(r[3] for r in rows) - min(r[3] for r in rows),
    }
    g0 = rows[0]
    print(f"  k=0: arc={g0[1]:.6f}  curv={g0[2]:.6f}  torsion={g0[3]:.6f}")
    print(f"  max spread over k=0..9: {', '.join(f'{k}={v:.2e}' for k, v in spread.items())}")

    # --- 2. identity retrieval --------------------------------------------
    feats = [task_spectrum_features(t) for t in tasks]
    sizes_in = [
        math.log(1 + sum(len(p.input) * len(p.input[0]) for p in t.train)) for t in tasks
    ]
    sizes_out = [
        math.log(1 + sum(len(p.output) * len(p.output[0]) for p in t.train if p.output)) for t in tasks
    ]

    print("\ninput-spectrum -> own-output retrieval (chance: top1 0.25%, median rank ~200):")
    occ_in = [_vec(f, "in", OCC_KEYS) for f in feats]
    occ_out = [_vec(f, "out", OCC_KEYS) for f in feats]
    step_in = [_vec(f, "in", STEP_KEYS) for f in feats]
    step_out = [_vec(f, "out", STEP_KEYS) for f in feats]
    both_in = [a + b for a, b in zip(occ_in, step_in)]
    both_out = [a + b for a, b in zip(occ_out, step_out)]

    _retrieval(occ_in, occ_out, "occupancy spectrum")
    _retrieval(step_in, step_out, "step spectrum")
    _retrieval(both_in, both_out, "occupancy + step")
    _retrieval([[s] for s in sizes_in], [[s] for s in sizes_out], "log-cells baseline")
    # does the spectrum break the size baseline's ties? lexicographic: size
    # distance first, spectrum distance only to order exact-size matches
    n = len(tasks)
    ranks = []
    for i in range(n):
        key_own = (abs(sizes_in[i] - sizes_out[i]), _dist(both_in[i], both_out[i]))
        closer = tied = 0
        for j in range(n):
            if j == i:
                continue
            key = (abs(sizes_in[i] - sizes_out[j]), _dist(both_in[i], both_out[j]))
            if key < key_own:
                closer += 1
            elif key == key_own:
                tied += 1
        ranks.append(1 + closer + 0.5 * tied)
    top1 = sum(1 for r in ranks if r <= 1)
    top5 = sum(1 for r in ranks if r <= 5)
    print(
        f"  {'size, spectrum tiebreak':<28} top1={top1:>3}/{n} ({top1 / n:.1%})  "
        f"top5={top5:>3} ({top5 / n:.1%})  median rank={statistics.median(ranks):.0f}"
    )

    rng = random.Random(0)
    perm = list(range(len(tasks)))
    rng.shuffle(perm)
    _retrieval(both_in, [both_out[perm[i]] for i in range(len(tasks))], "shuffled-pairing control")

    # --- 3. solvability beyond the size confound --------------------------
    solved_ids = set()
    for line in SOLUTIONS.read_text().splitlines():
        r = json.loads(line)
        if r["plan"] != "abstain":
            solved_ids.add(r["task_id"])
    solved_mask = [t.task_id in solved_ids for t in tasks]
    print(f"\nsolved={sum(solved_mask)}  abstained={len(tasks) - sum(solved_mask)}")
    print("per-feature AUC solved-vs-abstained (log_cells headline ~0.84 for reference):")

    feature_columns: dict[str, list[float]] = {"neg_log_cells_in": [-s for s in sizes_in]}
    for side in ("in", "out"):
        for tmpl in OCC_KEYS + STEP_KEYS:
            key = tmpl.format(side)
            feature_columns[key] = [f[key] for f in feats]

    scored = []
    for name, col in feature_columns.items():
        pos = [v for v, s in zip(col, solved_mask) if s]
        neg = [v for v, s in zip(col, solved_mask) if not s]
        a = _auc(pos, neg)
        scored.append((name, max(a, 1 - a)))
    scored.sort(key=lambda t: -t[1])
    for name, a in scored[:8]:
        print(f"  {name:<20} AUC={a:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "task_spectra.jsonl").open("w") as f:
        for t, ft in zip(tasks, feats):
            rec = {"task_id": t.task_id, "solved": t.task_id in solved_ids, **ft}
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"\nwrote {OUT / 'task_spectra.jsonl'}")


if __name__ == "__main__":
    main()
