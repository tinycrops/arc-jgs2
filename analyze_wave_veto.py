"""Calibrate the wave-fingerprint veto against real overshoot predictions.

The conservative solver never emits a wrong prediction (it abstains unless
one plan explains every train pair), so the veto has nothing to catch there.
Its intended job is damping the solver's characteristic FAILURE MODE if the
acceptance rule is ever relaxed: overshoot -- an op that exactly explains
some train pairs but not all. This script materializes that population:

  for every training task, every candidate single op with support
  0 < s < total is applied to the test inputs; ARC-1 training has the test
  outputs, so every such prediction is gradable.

Then it asks whether the wave-consistency score separates wrong from correct
predictions, and what (margin, slack) buys which catch/false-alarm tradeoff.
The should-pass set (full-support ops + the solver's real accepted plans)
must survive with zero vetoes for a threshold to be acceptable.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from arc_jgs2.loaders import load_tasks
from arc_jgs2.solvers import PlanState, _candidate_ops, _exact_support
from arc_jgs2.wavecheck import pair_distance
from arc_jgs2.loaders import Pair

DATA = Path("data/ARC-AGI/data/training")


def main() -> None:
    tasks = load_tasks(DATA)

    records = []  # (kind, correct, d_test, envelope)
    for task in tasks:
        train = [p for p in task.train if p.output is not None]
        total = len(train)
        if not total or not task.test:
            continue
        gradable = [(p.input, p.output) for p in task.test if p.output is not None]
        if not gradable:
            continue

        start = PlanState(
            plan=(),
            train_grids=tuple(p.input for p in train),
            test_grids=tuple(inp for inp, _ in gradable),
        )
        train_ds = [pair_distance(p.input, p.output) for p in train]
        hi, lo = max(train_ds), min(train_ds)

        for op in _candidate_ops(start, train):
            try:
                applied = tuple(op.apply(g) for g in start.train_grids)
            except Exception:
                continue
            support = _exact_support(applied, train)
            if support == 0:
                continue
            kind = "overshoot" if support < total else "full-support"
            for test_in, test_out in gradable:
                try:
                    pred = op.apply(test_in)
                except Exception:
                    continue
                records.append((kind, pred == test_out, pair_distance(test_in, pred), hi, lo))

    for kind in ("overshoot", "full-support"):
        sub = [r for r in records if r[0] == kind]
        n_right = sum(1 for r in sub if r[1])
        print(f"{kind}: {len(sub)} predictions, {n_right} correct, {len(sub) - n_right} wrong")

    # AUC of the raw consistency statistic d_test/(envelope+slack), wrong vs correct
    slack = 0.05
    wrong = [d / (e + slack) for _, ok, d, e, _lo in records if not ok]
    right = [d / (e + slack) for _, ok, d, e, _lo in records if ok]
    c = sum((w > r) + 0.5 * (w == r) for w in wrong for r in right)
    print(f"\nAUC(score flags wrong) = {c / (len(wrong) * len(right)):.3f}   "
          f"(wrong median {statistics.median(wrong):.2f}, correct median {statistics.median(right):.2f})")

    print(f"\none-sided sweep (veto when d_test > margin*hi + {slack}):")
    print(f"  {'margin':>7}{'catch wrong':>15}{'veto correct':>15}")
    for margin in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
        caught = sum(1 for _, ok, d, hi, lo in records if not ok and d > margin * hi + slack)
        false = sum(1 for _, ok, d, hi, lo in records if ok and d > margin * hi + slack)
        print(
            f"  {margin:>7.2f}{caught:>8}/{len(wrong)} ({caught / len(wrong):.0%})"
            f"{false:>8}/{len(right)} ({false / len(right):.1%})"
        )

    # the inverted AUC says wrong predictions are typically too TIMID, not too
    # wild: the true transformation moves train pairs a characteristic
    # distance, an overshooting partial op under-moves the test pair. So test
    # the two-sided BAND: veto when d_test leaves [lo/margin - slack, hi*margin + slack]
    print(f"\ntwo-sided band sweep (veto when d_test outside [lo/m - {slack}, hi*m + {slack}]):")
    print(f"  {'margin':>7}{'catch wrong':>15}{'veto correct':>15}{'catch: timid/wild':>19}")
    for margin in (1.0, 1.25, 1.5, 2.0, 3.0):
        timid = sum(1 for _, ok, d, hi, lo in records if not ok and d < lo / margin - slack)
        wild = sum(1 for _, ok, d, hi, lo in records if not ok and d > margin * hi + slack)
        false = sum(
            1 for _, ok, d, hi, lo in records if ok and (d > margin * hi + slack or d < lo / margin - slack)
        )
        caught = timid + wild
        print(
            f"  {margin:>7.2f}{caught:>8}/{len(wrong)} ({caught / len(wrong):.0%})"
            f"{false:>8}/{len(right)} ({false / len(right):.1%})"
            f"{timid:>12}/{wild}"
        )


if __name__ == "__main__":
    main()
