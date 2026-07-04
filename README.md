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

## Corpus Rotation Orbit & Radial Field (`arc_jgs2/corpusfield.py`)

Stack every train grid of the whole dataset into ONE cylinder polyline
(316,468 cells for ARC-1 training) and rotate it about the cylinder axis by
k*36 degrees. Theorem, verified at full-corpus scale (`analyze_corpus_field.py`,
`tests/test_corpusfield.py`): the rotation is *exactly* the same object as
adding +k (mod 10) to every color-role slot, and because it is a rigid motion,
arc length and curvature are bit-identical across all ten rotations and
|torsion| matches to 5e-9. Ten formations -- nine non-identity -- one shape:
the cyclic part of color relabeling turned from a nuisance into a geometric
symmetry. (Estimator caveat: *signed* torsion jitters ~0.2% across rotations,
not because the theorem fails but because the polyline lives on a quantized
angle lattice where consecutive binormals are often exactly antiparallel, so
the sign is float noise. Honest boundary: the cylinder geometrizes Z_10, the
cyclic shifts; the rest of S_10 is handled approximately by role order.)

The useful quotient of the orbit is the same move crystallography, image
registration (Fourier-Mellin), and invariant theory all use: once a nuisance
acts by rotation, the DFT magnitudes of any per-slot profile are invariant
under it. Six numbers per profile (occupancy and step-transition spectra,
size-normalized) summarize a task up to recoloring.

The radial field makes the symmetry literal: re-embed the corpus sequence in a
disk -- position 0 at the outer rim, the last cell at the origin, angle = color
slot (light from a point source, run backward). The two polar coordinates
factor the two nuisances: recoloring is a rotation of the image, corpus
position is wavefront radius. Renders in `runs/corpus-field/` via
`render_corpus_field.py`.

Measured payoff (`analyze_corpus_field.py`, tie-aware ranks): a task's
input-stack spectrum retrieves its OWN output-stack among all 400 at 14% top-1
(chance 0.25%) -- the invariant profile partly survives each task's
transformation. Grid size alone gets 28%, but it is blind within size-ties;
using the spectrum only to break size ties lifts top-1 to 46% and median rank
6 -> 2. Confounds stated: role order already sorts colors by frequency, so the
occupancy spectrum adds little over a sorted histogram (the step spectrum is
the genuinely new adjacency signal), and for solved-vs-abstain the spectrum
does not beat the size confound (best AUC 0.833 vs log-cells 0.827, n=17
solved).

### Serialization Headroom (`analyze_serialization.py`)

Would a model that invents smarter grouping/stacking orders gain anything?
Everything scored downstream is order-invariant by construction, so the only
objective where ordering can matter is description length (MDL). Measured on
the 316k-cell stream: the best constructed grouping (greedy nearest-neighbor
chain in task-spectrum space) beats dataset order by just **2.0%** on zlib and
lzma alike; an online adaptive bigram coder gains **0.000 bits/cell** (a
low-order model saturates regardless of order); junction arc moves 0.06%.
Dataset order measures identical to random shuffle, confirming hash-order is
already a random control. Meanwhile the within-grid scan axis moves the
numbers as much or more in *both* directions (boustrophedon: -1.0% bigram but
+5.7% lzma), and the bigram-vs-lzma gap (1.20 vs 0.81 bits/cell) says most
structure is longer-range than any ordering fix. Verdict: a grouping-inventor
model has ~2% headroom -- not worth building; the bits live inside grids and
in better sequential models, not in the stacking order.

### Step-Spectrum Gate (`analyze_step_gate.py`)

`d_step` = distance between a task's input-stack and output-stack step
spectra. The gate hypothesis ("low d_step = rearrangement family") came with
three predictions; scoreboard:

- "Solver-family tasks sit in the low tail" **lost** as stated (AUC 0.555 ~
  chance) but decomposed exactly along representational lines: global
  color-map tasks conserve the spectrum *exactly* (median 0.000 -- role
  co-rotation absorbs recoloring, as designed), geometric transforms nearly
  (0.20), crop *breaks* it (0.90 -- the output is a different boundary
  regime). The gate sorts by transformation *type*, not solver reach.
- "object_count_shift tasks have higher d_step" won: AUC 0.679, rising to
  0.714 with the resize confound stratified out.
- "palette_shift should NOT raise d_step" mostly won: AUC 0.580 stratified,
  far below count-shift -- co-rotation absorbs most of recoloring.

Payoff: 13 abstained tasks have d_step exactly 0. Component-level checking
splits them into two coherent missing-primitive families: **pure object
motion** (`5521c0d9`, `dc433765`, `05f2a901`: every train pair conserves the
exact color+shape multiset, objects only move) and **relational recolor**
(`85c4e7cd`, `bda2d7a6`, `f76d97a5`: spectra conserved, shapes swap colors;
independently tagged `color_role_candidate` by the orienter). Those two
primitives are the next vocabulary additions. Chart:
`runs/corpus-field/step_gate.png`.

The distilled primitive, stated once: **find coordinates in which a nuisance
becomes a group action with a geometric realization, then harvest invariants
by harmonic analysis on the group.** The +9 rotations are the orbit; the
spectrum is the quotient; the disk is the coordinate system that makes both
visible. The MNIST-1D Takens lens below is the same primitive applied to time
instead of color.

## Testbed: MNIST-1D (`arc_jgs2/mnist1d_lens.py`)

`../mnist1d` (Greydanus & Kobak) is a cheap, continuous-signal testbed for the
same question: does lifting a sequence to 3D to expose hidden shape carry
predictive signal? ARC grids are discrete symbol sequences with no natural 3D
shape, so they're lifted onto a color-wheel cylinder. MNIST-1D signals are
already continuous curves, so the matching lift is a **Takens delay embedding**
`(x[t], x[t+tau], x[t+2*tau])` -- the standard nonlinear-dynamics trick for
recovering a scalar time series' shape -- read off with the exact same
`frenet_stats` Frenet invariants (factored out of `linefeatures.py` for reuse).

Question: does delay-embedded curvature/torsion distance between two digit
*templates* predict how often a real classifier confuses them (the MNIST-1D
analog of "geometry predicts ARC solvability")? Honest finding, stable across
tau in {1, 2, 3}: **no** -- geometric distance barely correlates with confusion
(Spearman ~0), while trivial raw-amplitude distance between templates
correlates strongly (~0.9). Confusion here is dominated by literal amplitude
overlap under the dataset's heavy translation/noise augmentation, not by
qualitative curve shape.

The invariants did earn their correctness, though: they exactly detected that
templates 0/6, 1/7, and 3/8 are amplitude-mirror pairs (`x -> 10-x` in the raw
templates, i.e. `x -> -x` after whitening) -- curvature (even under reflection)
matches exactly, and torsion (a chirality-sensitive pseudoscalar) flips sign for
the one pair whose curve isn't planar. That's the Frenet math behaving exactly
as differential geometry says it should, on a real, previously-unremarked
structural fact about the template set -- just not the fact this experiment
went looking for.

The next step is to add an iterative hypothesis engine that accepts, dampens, or
rejects primitive perturbations using the witness fields emitted here.
