"""Step-spectrum conservation gate: does the distance between a task's
input-stack and output-stack step spectra sort tasks into primitive families?

d_step(task) = L2 distance between the rotation-invariant step-transition
spectra of the train inputs vs train outputs (size-normalized, so grid size is
mostly out). The gate hypothesis, with predictions that can lose:

1. Tasks the conservative solver cracked (recolor / geometric transform /
   crop / scale -- all pure rearrangements or relabelings) sit in the
   LOW-d_step tail.
2. Tasks witnessed as `object_count_shift` (things appear or vanish) have
   HIGHER d_step than tasks without that witness.
3. Representation-specific prediction: `palette_shift` should NOT raise
   d_step, because role co-rotation absorbs recoloring before the spectrum is
   computed. If palette_shift raises d_step as much as object_count_shift,
   the co-rotation isn't doing its job and the gate idea loses.

Confound named up front: `resize` (138 tasks) shrinks/grows the output stack,
and smaller stacks have noisier spectra; resize also co-occurs with
object_count_shift. Groups are therefore also reported resize-stratified.

Gate payoff if it works: the lowest-d_step ABSTAINED tasks are candidates
whose transformation conserves adjacency dynamics -- i.e. likely rearrangement
primitives missing from the vocabulary (a second gap ledger, orthogonal to the
solvability-score one).
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from arc_jgs2.corpusfield import task_spectrum_features
from arc_jgs2.loaders import load_tasks

DATA = Path("data/ARC-AGI/data/training")
SOLUTIONS = Path("runs/unionmap-training-solutions/solutions.jsonl")
ORIENTATION = Path("runs/full-20260703-training-orientation/orientation.jsonl")
OUT = Path("runs/corpus-field")

STEP_KEYS = [f"step_{{}}_k{k}" for k in range(6)]
OCC_KEYS = [f"occ_{{}}_k{k}" for k in range(1, 6)]


def _dist(feats: dict[str, float], keys: list[str]) -> float:
    return math.sqrt(sum((feats[k.format("in")] - feats[k.format("out")]) ** 2 for k in keys))


def _auc(pos: list[float], neg: list[float]) -> float:
    c = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return c / (len(pos) * len(neg)) if pos and neg else float("nan")


def _describe(label: str, vals: list[float]) -> None:
    if not vals:
        print(f"  {label:<44} (empty)")
        return
    print(
        f"  {label:<44} n={len(vals):>3}  median={statistics.median(vals):.4f}  "
        f"mean={statistics.mean(vals):.4f}"
    )


def main() -> None:
    tasks = load_tasks(DATA)
    plans: dict[str, str] = {}
    for line in SOLUTIONS.read_text().splitlines():
        r = json.loads(line)
        if r["plan"] != "abstain":
            plans[r["task_id"]] = r["plan"]
    tags: dict[str, set[str]] = {}
    for line in ORIENTATION.read_text().splitlines():
        r = json.loads(line)
        tags[r["task_id"]] = set(r["global_tags"])

    rows = []
    for t in tasks:
        feats = task_spectrum_features(t)
        rows.append(
            {
                "task_id": t.task_id,
                "d_step": _dist(feats, STEP_KEYS),
                "d_occ": _dist(feats, OCC_KEYS),
                "plan": plans.get(t.task_id, ""),
                "tags": tags.get(t.task_id, set()),
            }
        )

    d_all = [r["d_step"] for r in rows]
    print(f"tasks={len(rows)}  d_step: median={statistics.median(d_all):.4f}  ")

    # --- prediction 1: solver families sit in the low tail ---------------
    print("\nd_step by solver outcome / plan family:")
    solved = [r for r in rows if r["plan"]]
    abstained = [r for r in rows if not r["plan"]]
    _describe("abstained", [r["d_step"] for r in abstained])
    _describe("solved (all plans)", [r["d_step"] for r in solved])
    fams = {
        "color map": lambda p: "color_map" in p,
        "geometric (flip/rot/transpose)": lambda p: "whole_grid" in p,
        "crop": lambda p: p.startswith("crop"),
        "scale": lambda p: p == "scale" or p.endswith("-> scale"),
    }
    for name, match in fams.items():
        _describe(f"  {name}", [r["d_step"] for r in solved if match(r["plan"])])
    print(f"  AUC(low d_step -> solved) = {_auc([-r['d_step'] for r in solved], [-r['d_step'] for r in abstained]):.3f}")
    ranked = sorted(rows, key=lambda r: r["d_step"])
    for k in (25, 50, 100):
        hit = sum(1 for r in ranked[:k] if r["plan"])
        print(f"  solved tasks in lowest-{k} d_step: {hit}/{len(solved)}")

    # --- predictions 2 & 3: witness tags ---------------------------------
    print("\nd_step by witness tag (prediction: count-shift high, palette-shift NOT high):")
    for tag in ("object_count_shift", "palette_shift", "resize"):
        with_tag = [r["d_step"] for r in rows if tag in r["tags"]]
        without = [r["d_step"] for r in rows if tag not in r["tags"]]
        _describe(f"{tag}: present", with_tag)
        _describe(f"{tag}: absent", without)
        print(f"    AUC(d_step -> {tag}) = {_auc(with_tag, without):.3f}")

    print("\nresize-stratified (confound check, no-resize tasks only):")
    nores = [r for r in rows if "resize" not in r["tags"]]
    for tag in ("object_count_shift", "palette_shift"):
        with_tag = [r["d_step"] for r in nores if tag in r["tags"]]
        without = [r["d_step"] for r in nores if tag not in r["tags"]]
        print(
            f"  {tag}: present median={statistics.median(with_tag):.4f} (n={len(with_tag)})  "
            f"absent median={statistics.median(without):.4f} (n={len(without)})  "
            f"AUC={_auc(with_tag, without):.3f}"
        )

    # --- the gate ledger ---------------------------------------------------
    print("\ngate ledger -- lowest-d_step tasks the solver still abstains on")
    print("(adjacency dynamics conserved in->out: likely rearrangement primitives we lack):")
    gaps = [r for r in ranked if not r["plan"]][:15]
    for r in gaps:
        print(f"  {r['task_id']:>10}  d_step={r['d_step']:.4f}  d_occ={r['d_occ']:.4f}  tags={','.join(sorted(r['tags'])) or '-'}")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "step_gate.jsonl").open("w") as f:
        for r in rows:
            rec = {**r, "tags": sorted(r["tags"])}
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"\nwrote {OUT / 'step_gate.jsonl'}")


if __name__ == "__main__":
    main()
