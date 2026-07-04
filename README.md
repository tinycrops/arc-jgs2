# ARC-JGS2

Corpus-level ARC orientation workbench inspired by JGS2.

The goal is to treat ARC-1 tasks as many small logical worlds that can be oriented
end to end at once:

```text
raw grids
-> object parse
-> canonical object frames
-> relation/witness signatures
-> shared primitive hypotheses
-> per-task orientation table
```

This is not yet a magic ARC solver. It is the scaffold where a solver can learn
which local edits overshoot: transformations that look good for one object or
one train pair but break cross-example consistency, symmetry, counting, or role
bindings.

## Quick Start

Run the included demo tasks:

```bash
cd /home/ath/arc-jgs2
python3 -m arc_jgs2 orient --demo --out runs/demo
```

Run on an ARC-AGI / ARC-1 dataset directory:

```bash
python3 -m arc_jgs2 orient --data /path/to/ARC-AGI/data/training --out runs/training
python3 -m arc_jgs2 orient --data /path/to/ARC-AGI/data/evaluation --out runs/evaluation
```

Outputs:

- `orientation.jsonl`: one full orientation record per task.
- `orientation.csv`: compact corpus table.
- `summary.json`: primitive-family and quality counts.
- `index.html`: visual dashboard.

Run the conservative primitive solver:

```bash
python3 -m arc_jgs2 solve --data data/ARC-AGI/data/training --out runs/training-solutions
```

The solver only predicts when one primitive explains every train pair. Everything
else abstains, which keeps local overshoot visible instead of hiding it behind a
bad guess.

## Geometry Workbench & Gap Ledger

`arc_jgs2/linefeatures.py` treats each grid as a polyline on a "corpus cylinder":
cells flattened, colors mapped to angular slots by a per-grid role order
(background -> slot 0, rest by frequency -- the co-rotation that removes arbitrary
color labels). It extracts co-rotation-aware geometry: arc length / turning,
discrete Frenet **curvature and torsion** of the 3D polyline, row-vs-column
**directional anisotropy**, and 4-neighbor **edge/boundary** stats.

`analyze_line_features.py` asks whether that geometry predicts which tasks the
conservative solver can crack, and reports a composite solvability score plus a
**gap ledger** -- the highest-scoring tasks the solver still abstains on.

Honest finding: at the task level, **grid size dominates** (CV-AUC ~0.84);
curvature is the strongest single geometric term but the geometry is largely a
proxy for size and does not robustly beat it out-of-sample. Its real payoff is
the gap ledger: it flagged `d511f180` as "should be solvable," which exposed an
overly strict color-map primitive -> see below.

## Current Primitive Families

- exact-copy
- global color-map (merges compatible per-pair maps into one union map; an
  injectivity guard abstains when remapped colors collapse together, which is
  usually a relational recolor masquerading as a global map -- an overshoot)
- crop / bounding-box extraction
- geometric transform over whole grid
- output canvas resize
- object-count and component-shape signatures
- witness signatures for symmetry, palette change, size change, and object deltas

## JGS2 Translation

| JGS2 simulation | ARC-JGS2 |
| --- | --- |
| local sub-problem | object, region, color role, relation |
| local solve update | proposed edit or transformation |
| global energy | consistency over train pairs and test executable plan |
| overshoot | local rule fits a piece but breaks task-wide structure |
| co-rotated subspace | canonical object frame under translation/orientation/color role |
| cubature samples | small diagnostic witnesses for global consistency |
| damping Hessian | injectivity guard on the union color-map: rejects a locally consistent edit that is globally the wrong abstraction |

The gap ledger closes a loop: geometry triage surfaces a task the solver *should*
reach, inspection reveals the missing or over-strict primitive, and the fix is
added under the same conservative rule (commit only when one plan explains every
train pair; otherwise abstain so overshoot stays visible). The union color-map +
its overshoot guard is the first primitive found this way (training 14 -> 17
solved, still zero wrong predictions).

The next step is to add an iterative hypothesis engine that accepts, dampens, or
rejects primitive perturbations using the witness fields emitted here.
