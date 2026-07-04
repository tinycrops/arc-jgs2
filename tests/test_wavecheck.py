"""Tests for the wave-fingerprint consistency check (arc_jgs2.wavecheck).

The check is plan-independent: every train pair yields a fingerprint distance
d(in_i, out_i), giving a band [lo, hi]. A test prediction passes when its own
(test input, prediction) distance stays inside the margin-scaled, slack-padded
band. Calibration on ARC-1 training overshoot candidates showed the failure
mode is two-sided: wrong predictions are usually too TIMID (they under-move
the test pair relative to how far train pairs travel), occasionally too wild.
The check is a damper: it can only turn predictions into abstentions.
"""

from __future__ import annotations

from arc_jgs2.loaders import Pair
from arc_jgs2.wavecheck import wave_consistency


def _identity_pairs() -> list[Pair]:
    g1 = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    g2 = [[2, 2, 0], [0, 2, 0], [0, 0, 2]]
    return [Pair(input=g1, output=[row[:] for row in g1]), Pair(input=g2, output=[row[:] for row in g2])]


def _big_change_pairs() -> list[Pair]:
    # checkerboard -> solid background: a large, consistent fingerprint move
    checker_a = [[(r + c) % 2 for c in range(6)] for r in range(6)]
    checker_b = [[(r + c + 1) % 2 * 2 for c in range(6)] for r in range(6)]
    solid = [[0] * 6 for _ in range(6)]
    return [Pair(input=checker_a, output=solid), Pair(input=checker_b, output=solid)]


def test_consistent_prediction_passes() -> None:
    test_in = [[3, 0, 3], [0, 3, 0], [3, 0, 3]]
    check = wave_consistency(_identity_pairs(), test_in, [row[:] for row in test_in])
    assert check.ok
    assert check.reason == ""
    assert check.d_test < 1e-12


def test_wild_prediction_fails() -> None:
    # train relation is identity (zero band); prediction rewrites the grid
    test_in = [[3, 0, 3], [0, 3, 0], [3, 0, 3]]
    scrambled = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    check = wave_consistency(_identity_pairs(), test_in, scrambled)
    assert not check.ok
    assert check.reason == "wild"
    assert check.d_test > check.hi


def test_timid_prediction_fails() -> None:
    # train pairs move far through fingerprint space; an identity "prediction"
    # that leaves the test input untouched is the classic under-move overshoot
    test_in = [[(r + c) % 2 * 3 for c in range(6)] for r in range(6)]
    check = wave_consistency(_big_change_pairs(), test_in, [row[:] for row in test_in])
    assert not check.ok
    assert check.reason == "timid"
    assert check.d_test < check.lo


def test_band_admits_train_sized_changes() -> None:
    solid = [[0] * 6 for _ in range(6)]
    test_in = [[(r + c) % 2 * 3 for c in range(6)] for r in range(6)]
    check = wave_consistency(_big_change_pairs(), test_in, solid)
    assert check.ok


def test_score_grows_with_violation() -> None:
    pairs = _identity_pairs()
    test_in = [[3, 0], [0, 3]]
    near = wave_consistency(pairs, test_in, [[3, 0], [0, 3]])
    far = wave_consistency(pairs, test_in, [[1, 2], [3, 4]])
    assert near.score < far.score


# --- solver wiring ----------------------------------------------------------

from arc_jgs2.loaders import Task
from arc_jgs2.solvers import SolveResult, apply_wave_gate, solve_task


def _tile_task() -> Task:
    return Task(
        task_id="tile_demo",
        train=(
            Pair(input=[[1, 2], [3, 4]], output=[[1, 2, 1, 2], [3, 4, 3, 4]]),
            Pair(input=[[5, 0], [0, 6]], output=[[5, 0, 5, 0], [0, 6, 0, 6]]),
        ),
        test=(Pair(input=[[7, 8], [9, 1]], output=[[7, 8, 7, 8], [9, 1, 9, 1]]),),
        source="unit",
    )


def test_solve_task_annotates_wave_fields() -> None:
    result = solve_task(_tile_task())
    assert result.plan != "abstain"
    assert result.wave_ok
    assert len(result.wave_flags) == len(result.predictions)
    assert all(flag == "" for flag in result.wave_flags)
    assert result.wave_vetoed_plan == ""


def test_wave_gate_vetoes_inconsistent_prediction() -> None:
    task = Task(
        task_id="veto_demo",
        train=tuple(_big_change_pairs()),
        test=(Pair(input=[[(r + c) % 2 * 3 for c in range(6)] for r in range(6)], output=None),),
        source="unit",
    )
    # a fabricated result whose prediction leaves the test input untouched:
    # timid relative to how far the train pairs travel
    timid_pred = [[(r + c) % 2 * 3 for c in range(6)] for r in range(6)]
    fake = SolveResult(
        task_id="veto_demo", plan="identity_overshoot", train_support=1, train_total=2,
        predictions=[timid_pred], test_known_total=0, test_known_correct=0, source="unit",
    )

    annotated = apply_wave_gate(fake, task, veto=False)
    assert not annotated.wave_ok
    assert annotated.wave_flags == ("timid",)
    assert annotated.plan == "identity_overshoot"  # annotation only

    vetoed = apply_wave_gate(fake, task, veto=True)
    assert vetoed.plan == "abstain"
    assert vetoed.predictions == []
    assert vetoed.wave_vetoed_plan == "identity_overshoot"
