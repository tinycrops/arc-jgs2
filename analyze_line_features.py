"""Do the cylinder line's geometric features predict which tasks the
conservative primitive solver (arc_jgs2.solvers) can solve?

Reads the solve results already computed in runs/verify-training-solutions/,
computes per-task line features from each task's train grids, and reports
whether solved vs. abstained tasks separate on any feature.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from arc_jgs2.linefeatures import task_line_features
from arc_jgs2.loaders import load_tasks

SOLUTIONS = Path("runs/unionmap-training-solutions/solutions.jsonl")
DATA = Path("data/ARC-AGI/data/training")

# Composite solvability score. Honest caveat: grid size dominates (CV-AUC ~0.84);
# the geometry terms are largely a proxy for it and do not robustly beat size
# alone out-of-sample. The score is useful as a cheap in-distribution *triage*
# and, more valuably, as a gap-ledger: high-score tasks the solver still
# abstains on are the primitive-vocabulary gaps worth filling next.
SCORE_WEIGHTS = {"mean_curv": +0.99, "edge_change_frac": +0.89, "mean_torsion": -0.74, "log_cells": -0.72}


def _score_rows(rows: list[dict]) -> None:
    for r in rows:
        r["log_cells"] = math.log(r["total_cells"] + 1)
    stats = {
        k: (statistics.mean(r[k] for r in rows), statistics.pstdev(r[k] for r in rows) or 1e-9)
        for k in SCORE_WEIGHTS
    }
    for r in rows:
        r["score"] = sum(w * (r[k] - stats[k][0]) / stats[k][1] for k, w in SCORE_WEIGHTS.items())


def main() -> None:
    solved: dict[str, str] = {}
    for line in SOLUTIONS.read_text().splitlines():
        record = json.loads(line)
        if record["plan"] != "abstain":
            solved[record["task_id"]] = record["plan"]

    tasks = load_tasks(DATA)
    rows = []
    for task in tasks:
        grids = [pair.input for pair in task.train] + [pair.output for pair in task.train if pair.output is not None]
        feats = task_line_features(grids)
        feats["task_id"] = task.task_id
        feats["solved"] = task.task_id in solved
        feats["plan"] = solved.get(task.task_id, "")
        rows.append(feats)

    keys = [
        "n_colors", "color_entropy", "arc_length", "mean_step", "mean_turn", "var_turn",
        "mean_curv", "mean_torsion", "mean_abs_torsion", "zero_run_fraction",
        "col_mean_turn", "anisotropy_turn", "anisotropy_arc",
        "edge_change_frac", "edge_hv_anisotropy", "edge_mean_chord", "total_cells",
    ]

    solved_rows = [r for r in rows if r["solved"]]
    abstained_rows = [r for r in rows if not r["solved"]]
    print(f"solved={len(solved_rows)}  abstained={len(abstained_rows)}  total={len(rows)}\n")

    stats = []
    for key in keys:
        s = [r[key] for r in solved_rows]
        a = [r[key] for r in abstained_rows]
        s_mean, a_mean = statistics.mean(s), statistics.mean(a)
        pooled_sd = statistics.pstdev([r[key] for r in rows]) or 1e-9
        stats.append((key, s_mean, statistics.median(s), a_mean, statistics.median(a), (s_mean - a_mean) / pooled_sd))

    stats.sort(key=lambda t: -abs(t[5]))
    print(f"{'feature':<20}{'solved mean':>14}{'solved med':>13}{'abstain mean':>15}{'abstain med':>14}{'|sep| (SD)':>13}")
    for key, sm, smed, am, amed, sep in stats:
        print(f"{key:<20}{sm:>14.3f}{smed:>13.3f}{am:>15.3f}{amed:>14.3f}{sep:>13.3f}")

    # --- solvability score + gap ledger ---
    _score_rows(rows)
    ranked = sorted(rows, key=lambda r: -r["score"])

    def auc(pos: list[dict], neg: list[dict]) -> float:
        c = sum((p["score"] > n["score"]) + 0.5 * (p["score"] == n["score"]) for p in pos for n in neg)
        return c / (len(pos) * len(neg)) if pos and neg else float("nan")

    print(f"\ncomposite solvability score  AUC={auc(solved_rows, abstained_rows):.3f}  (0.5=random)")
    for k in (25, 50, 100):
        hit = sum(1 for r in ranked[:k] if r["solved"])
        print(f"  recall@{k}: {hit}/{len(solved_rows)} solved in top {k}")

    print("\ngap ledger -- highest-scoring tasks the solver still abstains on")
    print("(geometry says in-distribution; primitive vocabulary can't reach them yet):")
    gaps = [r for r in ranked if not r["solved"]][:12]
    for r in gaps:
        print(
            f"  {r['task_id']:>10}  score={r['score']:+.2f}  "
            f"curv={r['mean_curv']:.2f} edge={r['edge_change_frac']:.2f} cells={r['total_cells']:.0f}"
        )

    out_path = Path("runs/line-features")
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / "task_features.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"\nwrote {out_path / 'task_features.jsonl'}")


if __name__ == "__main__":
    main()
