"""Tests for per-grid wave fingerprints (arc_jgs2.wavefield).

The fingerprint is the far-field magnitude |A_m(k)| of the grid's disk
embedding treated as coherent point sources, for a sweep of wavenumbers k and
angular modes m. Contract: recoloring (cyclic slot shift) must leave every
magnitude unchanged, because it only rotates the source pattern.
"""

from __future__ import annotations

from arc_jgs2.wavefield import DEFAULT_K_SWEEP, DEFAULT_MODES, wave_fingerprint


def test_fingerprint_shape_and_finiteness() -> None:
    slots = [0, 1, 2, 0, 3, 0, 0, 1]
    fp = wave_fingerprint(slots)
    assert len(fp) == len(DEFAULT_K_SWEEP) * len(DEFAULT_MODES)
    assert all(v == v and abs(v) < 1e6 for v in fp)  # finite, sane


def test_fingerprint_invariant_under_recoloring() -> None:
    slots = [0, 1, 2, 0, 3, 0, 0, 1, 4, 4, 2, 9]
    base = wave_fingerprint(slots)
    for k in range(1, 10):
        shifted = wave_fingerprint([(s + k) % 10 for s in slots])
        assert max(abs(a - b) for a, b in zip(base, shifted)) < 1e-9


def test_fingerprint_distinguishes_different_layouts() -> None:
    # same occupancy (5 zeros, 3 ones), different positions along the scan:
    # the k-sweep must see the difference the flat histogram cannot
    front = wave_fingerprint([1, 1, 1, 0, 0, 0, 0, 0])
    back = wave_fingerprint([0, 0, 0, 0, 0, 1, 1, 1])
    assert max(abs(a - b) for a, b in zip(front, back)) > 1e-3


def test_fingerprint_degenerate_inputs() -> None:
    assert wave_fingerprint([]) == [0.0] * (len(DEFAULT_K_SWEEP) * len(DEFAULT_MODES))
    fp = wave_fingerprint([4])
    assert len(fp) == len(DEFAULT_K_SWEEP) * len(DEFAULT_MODES)
    assert all(v == v for v in fp)
