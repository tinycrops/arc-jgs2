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

### Wave Room: Node/Antinode Analysis (`analyze_wave_field.py`)

Treat every cell of the radial disk as a coherent monochromatic point source
(scalar Huygens) and propagate: u(x) = sum exp(ik|x-p_j|)/sqrt(|x-p_j|).
What the node/antinode analysis IS, exactly (Jacobi-Anger): the m-th angular
mode of the field at wavenumber k is the m-th DFT of a **Bessel-weighted
occupancy profile** sum_s e^{2pi i m s/10} sum_{j in s} J_m(k r_j). At long
wavelengths this collapses to the flat occupancy spectrum already in use;
sweeping k dials in *where along the corpus* each color's mass sits. The wave
picture is therefore a position x color joint transform whose magnitudes
inherit the recolor invariance -- recoloring rotates the whole interference
pattern rigidly and changes nothing else (and since recoloring produces a
point set exactly congruent to the rotated original, any measured difference
in nodal statistics is pure grid discretization: ~1% on coarse stats, ~9% on
the near-silent nodal area, the most alias-sensitive set).

Measured: multi-k far-field magnitudes |A_m(k)| (k = 2pi x {1,2,4,8}, m =
0..10) beat the flat spectra on own-output retrieval standalone -- 17.5% vs
14.2% top-1 -- so the radial (position-color) coupling carries real identity
signal; but as a size-tiebreak they match the flat spectra (45.2% vs 45.5%),
i.e. within size-ties the extra coupling adds nothing yet. Room render (1024^2,
FFT convolution with the outgoing-wave kernel): `runs/corpus-field/wave_room.png`.

### Per-Grid Wave Fingerprints (`arc_jgs2/wavefield.py`, `analyze_grid_waves.py`)

The same far-field transform applied per grid (scan position ~ row becomes
the radius, so the k-sweep probes vertical color layout). Two protocols over
all 1302 train pairs:

- **Pair retrieval** (input grid -> own output among 1302, chance 0.08%):
  wave 11.1% beats flat spectra 8.6% standalone, and -- unlike at task scale
  -- the wave features now also win *after* the size tiebreak (22.1% vs
  20.7%, median rank 24 vs 26). The layout coupling earns its keep where
  layout lives.
- **Task cohesion** (nearest other input grid same-task, chance 0.2%): the
  reversal -- flat spectra 21.2% beat wave 17.7%. Position-specific layout
  varies across a task's examples, so the wave fingerprint is a *grid
  identity* instrument while the flat spectrum is a *task style* instrument.
  Different quotients for different questions; keep both.

### Animations (`render_animations.py`)

Four GIFs in `runs/corpus-field/`, each visualizing a claim already measured
statically above -- no new experiments, just motion:

- `corpus_disk_spin.gif` -- the rotation orbit made continuous: the 10
  discrete color-invariant formations are frames of one continuous rigid spin
  of the corpus disk, sampled every 4 degrees.
- `wave_room_spin.gif` -- the interference pattern rotating. u(x) depends
  only on source positions, so rotating every source by theta rotates the
  whole field by theta exactly -- true for ANY continuous theta, not just the
  10 color-cyclic angles (those are the subset that also double as valid
  recolorings). Rendered by rotating the precomputed magnitude image, not
  resimulating 316k sources per frame.
- `wave_room_buildup.gif` -- sources switching on in radius order (corpus
  position order), the literal "light from a point source, run backward"
  image from the original request. Cumulative FFT convolution, cheap because
  histogram accumulation is cheap and only the convolution reruns per frame.
- `wave_k_sweep_007bbfb7.gif` -- one real ARC grid's far field as the
  wavenumber sweeps low to high and back: fringe spacing visibly shrinks as
  k grows, the "dial" the wave-room section describes.

All four quantize to a single shared palette with no dithering (smooth
gradients dither-bloat badly under independent per-frame palettes -- this cut
file sizes 4-6x with no visible quality loss).

### Wave-Consistency Gate in the Solver (`arc_jgs2/wavecheck.py`)

The fingerprint check is wired into `solve_task` as a damper: every train
pair's fingerprint distance gives the task's transformation **band** [lo, hi];
a test prediction must move its pair a comparable distance. `solve` annotates
every prediction (`wave_ok`, `wave_flags`) and `--wave-veto` turns out-of-band
predictions into abstentions -- the gate can only remove predictions, so the
zero-wrong rule is preserved by construction.

Calibration (`analyze_wave_veto.py`) against materialized overshoot
candidates (61 wrong / 57 correct gradable predictions) found the failure
mode INVERTED from the naive guess: wrong predictions are mostly too *timid*
(they under-move the test pair; one-sided envelope AUC 0.288), so the check
is two-sided. At margin 1.25 / slack 0.10 it catches 11/61 wrong predictions
with 0/57 correct vetoed (slack covers legitimate fingerprint motion when a
recolor reorders role frequencies, observed on d511f180).

Live verification: training 48 attempted / 49-of-50 correct, evaluation 17
attempted / 19-of-19 -- zero flags, zero vetoes, nothing lost. Known blind
spot, stated plainly: the solver's one wrong training prediction (b230c067, a
relational recolor) sits mid-band at d=0.030 -- co-rotation absorbs
recoloring, so recoloring *errors* are invisible to the fingerprint. The
quotient that grants the invariance is exactly the direction the check cannot
police; a recolor-sensitive witness (e.g. exact color-map verification) has
to cover that axis.

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

## Domain-Generalization Tests

Does "find a nuisance realized as a group action, harvest invariants by
harmonic analysis on the group" generalize past ARC, or is it an ARC-shaped
curiosity? Three probes, aimed at falsifying it, prompted by asking whether
it would help on [Terminal-Bench](https://www.tbench.ai/).

**1. Terminal-Bench task survey.** Cloned `laude-institute/terminal-bench`
(241 tasks) and read task instructions directly rather than guessing. Result:
the primitive's precondition -- a small finite alphabet with an *arbitrary*
labeling to quotient out -- is almost entirely absent. 230/241 tasks are
bespoke engineering (COBOL reimplementation, kernel builds, log analysis,
circuit synthesis) with no relabeling nuisance at all. Of the 11 tasks tagged
`games`/puzzle, most are still bespoke (`ancient-puzzle` is a hardcoded
string-equality check against a Flask endpoint -- zero algebraic structure).
A genuine minority DO have it: `solve-sudoku` (digit relabeling is a real
symmetry of Sudoku validity, exactly analogous to ARC color relabeling),
`huarong-dao-solver` / `puzzle-solver` (sliding-block state spaces have a
real dihedral board symmetry usable for BFS canonicalization, the same
"quotient for search efficiency" idea as the gap ledger), `mahjong-winninghand`
(suit permutation + number-line reversal is a richer symmetry group than
ARC's plain cyclic one), and `feal-linear-cryptanalysis` (linear
cryptanalysis's bias/correlation calculus IS harmonic analysis over GF(2)^n
-- the professional version of exactly this primitive). `sha-puzzle` is a
clean negative control: hash functions are explicitly engineered to destroy
exploitable algebraic structure, so the primitive predicts zero traction
there, correctly. Verdict: no, not as a general terminal-agent booster; yes,
as a targeted tool for the minority of tasks that are secretly algebra.

**2. ConceptARC** (`victorvikram/ConceptARC`, 160 tasks, different authors,
designed to test whether solvers generalize a *concept* rather than pattern-
match training pairs). Ran the actual `arc_jgs2` solver and wave gate
unmodified, `python3 -m arc_jgs2 solve --data conceptarc/corpus --wave-veto`.
Attempt rate drops hard (48/400 on ARC-1 training -> 8/160 here, ~12% ->
~5%), expected: the primitive vocabulary is ARC-1-shaped and ConceptARC's
tasks probe unfamiliar concepts. More important: the "one primitive explains
every train pair -> zero wrong" guarantee -- rock solid on ARC-1 (49/50,
19/19) -- **breaks here**: only 2 of 8 attempted tasks get all 3 held-out
test variations right (11/24 known test cells correct overall). This is
ConceptARC doing exactly its job -- it deliberately varies surface features
across a concept's test cases to catch primitives that fit training pairs by
coincidence rather than generalizing the concept. The wave gate did not
catch any of it (0/8 flagged); inspecting the wrong predictions
(`InsideOutside9`, plan `recolor_by_size_rank`, d_test=0.06 comfortably
inside a band of [0, 0.17]) shows the exact blind spot already documented
above reproducing on a completely different task corpus: co-rotation
absorbs recoloring, so recolor-family errors are structurally invisible to
the fingerprint regardless of which grids they come from.

**3. Cross-domain cryptanalysis** (`analyze_cipher_domain.py`) -- zero code
or data shared with ARC, a different alphabet size (Z_26, not Z_10), testing
the primitive on its own oldest real precedent: Caesar-cipher frequency
analysis already IS "quotient by a cyclic nuisance, harvest via harmonic
analysis," a century before this repo existed. Self-consistency (recover an
injected shift from one text's own DFT phase, no external table): 26/26
exact -- the DFT-shift theorem holds. Realistic cross-text attack
(reference profile calibrated on this repo's README, ciphertexts made from
a disjoint sample -- the package docstrings -- at random shifts): the naive
port using only the k=1 harmonic's phase FAILS (12-14/100), regardless of
whether the reference came from the narrow README text or the canonical
English frequency table -- one harmonic discards too much of the profile.
Restoring the actual corpusfield.py design choice -- keep the *whole*
spectrum, not one harmonic, done here as circular cross-correlation via all
26 DFT bins (Wiener-Khinchin) -- recovers the exact shift 100/100 times,
matching the classical chi-squared frequency search exactly, at roughly 34x
lower cost per attack, because it is the correct closed-form realization of
the same idea instead of an explicit 26-way search. Reference-corpus choice
stopped mattering once the full spectrum was used.

Honest summary: the primitive is real and portable -- it reproduced a
century-old cryptanalytic technique from first principles and exposed the
same failure mode across two unrelated corpora -- but it is not a generic
intelligence upgrade. It pays off exactly where a domain secretly has an
arbitrary-relabeling nuisance, is invisible where a domain has none (most of
Terminal-Bench), and even inside its home domain (ARC) its guarantees are
scoped to the training distribution they were built against (ConceptARC).

## Testbed: Sudoku Digit Relabeling (`arc_jgs2/sudoku.py`)

The Terminal-Bench survey above named `solve-sudoku` as one of the rare
tasks with a REAL relabeling nuisance -- unlike ARC color (an assumption we
impose), permuting Sudoku's digits 1-9 by any bijection turns a valid puzzle
into another valid one, exactly, because every constraint is "all different"
and never a specific digit. This module wires that symmetry in as working
code, not just an analysis: `canonicalize`/`infer_relabeling` mirror
`linefeatures.role_order` and `solvers._infer_color_map_grids`; `solve` is a
real constraint-propagation + backtracking solver (naked singles, hidden
singles, MRV branching); `SudokuCache` is the payoff -- solve a puzzle's
canonical form once, reuse the solution for every relabeled duplicate,
mapped back through the inverse of that instance's own relabeling. 13 tests
in `tests/test_sudoku.py`, including an end-to-end invariance test that the
symmetry survives all the way through backtracking, not just canonicalization.

`analyze_sudoku_cache.py` measures the payoff at the scope where it actually
applies: the scenario is a casual app that reuses a small library of vetted
hole-pattern templates and stamps a random digit relabeling on each serving
for cheap "freshness." On 6 templates x 15 relabeled instances (90 puzzles),
the cache collapses 90 solves to 6 (12x wall-clock speedup, results verified
self-consistent and valid -- not bitwise-equal to an independent solve,
since 36-given puzzles aren't guaranteed unique, so two correct solvers can
legitimately land on two different valid completions). A control corpus
built from the same base grids but with an independently random hole
pattern per instance (no shared template) gets no cache hits at all (90/90
solved) -- the same code, correctly getting zero benefit outside its actual
scope, run explicitly rather than just asserted.

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
