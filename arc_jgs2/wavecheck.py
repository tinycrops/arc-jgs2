"""Wave-fingerprint consistency check: a damper on solver predictions.

Every train pair (in_i, out_i) yields a distance between wave fingerprints
(arc_jgs2.wavefield) of its serialized grids, giving the task's transformation
BAND [lo, hi] in fingerprint space. A test prediction is consistent when its
own (test input, prediction) distance stays inside the margin-scaled,
slack-padded band:

    lo / margin - slack  <=  d_test  <=  hi * margin + slack

The check is plan-independent -- it never looks at which primitive produced
the prediction, only at whether the predicted pair moves through fingerprint
space the way the train pairs do. In JGS2 terms it plays the damping role: a
local edit that fits some pairs but moves the test pair unlike the task's
global motion gets rejected. It can only turn predictions into abstentions,
never create them, so the conservative zero-wrong rule is preserved.

Calibration (analyze_wave_veto.py, ARC-1 training, 61 wrong / 57 correct
predictions from partial-support overshoot candidates plus full-support ops):
the failure mode is two-sided and mostly TIMID -- wrong predictions under-move
the test pair (median band-relative distance 0.18 vs 0.76 for correct), so a
one-sided "within envelope" veto is inverted (AUC 0.288). With the two-sided
band at margin=1.25, slack=0.10: catches 11/61 wrong, vetoes 0/57 correct.
The slack floor covers legitimate fingerprint motion when a recolor reorders
role frequencies on the test grid (observed on d511f180).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .corpusfield import grid_slots
from .grids import Grid
from .loaders import Pair
from .wavefield import wave_fingerprint

DEFAULT_MARGIN = 1.25
DEFAULT_SLACK = 0.10


@dataclass(frozen=True)
class WaveCheck:
    d_test: float
    lo: float  # lower band edge after margin/slack
    hi: float  # upper band edge after margin/slack
    score: float  # band-violation ratio: <= 1 inside, grows with violation
    ok: bool
    reason: str  # "" | "timid" | "wild"


def pair_distance(inp: Grid, out: Grid) -> float:
    a = wave_fingerprint(grid_slots(inp))
    b = wave_fingerprint(grid_slots(out))
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def wave_consistency(
    train: list[Pair],
    test_input: Grid,
    prediction: Grid,
    margin: float = DEFAULT_MARGIN,
    slack: float = DEFAULT_SLACK,
) -> WaveCheck:
    ds = [pair_distance(p.input, p.output) for p in train if p.output is not None]
    d_test = pair_distance(test_input, prediction)
    lo = max((min(ds, default=0.0) / margin) - slack, 0.0)
    hi = max(ds, default=0.0) * margin + slack

    if d_test > hi:
        reason, score = "wild", d_test / hi if hi else float("inf")
    elif d_test < lo:
        reason, score = "timid", lo / d_test if d_test else float("inf")
    else:
        reason = ""
        score = d_test / hi if hi else 0.0
    return WaveCheck(d_test=d_test, lo=lo, hi=hi, score=score, ok=reason == "", reason=reason)
