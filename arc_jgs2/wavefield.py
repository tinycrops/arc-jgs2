"""Per-grid wave fingerprints.

A grid's serialized stream is lifted onto the radial disk (scan position ->
radius, color-role slot -> angle) and treated as coherent monochromatic point
sources. The fingerprint is the far-field magnitude spectrum

    A_m(k) = | (1/N) * sum_j exp(-i k p_j . phi_hat) |_m-th angular mode

over a sweep of wavenumbers k and angular modes m. By Jacobi-Anger each entry
is the m-th DFT of a Bessel-weighted occupancy profile: k dials in *where
along the scan* each color's mass sits, generalizing the flat occupancy
spectrum (its long-wavelength limit). Recoloring rotates the source pattern
rigidly, so every magnitude is invariant under cyclic slot shifts.

Unlike the rest of arc_jgs2 this module uses numpy: fingerprints are computed
for thousands of grids and the inner product is a dense complex sum.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_K_SWEEP = tuple(2 * math.pi * f for f in (1, 2, 4, 8))
DEFAULT_MODES = tuple(range(11))
_N_ANGLES = 360


def wave_fingerprint(
    slots: list[int],
    k_sweep: tuple[float, ...] = DEFAULT_K_SWEEP,
    modes: tuple[int, ...] = DEFAULT_MODES,
) -> list[float]:
    """Rotation-invariant far-field magnitudes of one serialized stream."""
    if not slots:
        return [0.0] * (len(k_sweep) * len(modes))

    n = len(slots)
    idx = np.arange(n)
    r = (n - 1 - idx) / (n - 1) if n > 1 else np.zeros(1)
    ang = np.radians(-90 + (np.asarray(slots) % 10) * 36)
    x, y = r * np.cos(ang), r * np.sin(ang)

    phi = np.linspace(0, 2 * math.pi, _N_ANGLES, endpoint=False)
    proj = np.outer(x, np.cos(phi)) + np.outer(y, np.sin(phi))

    feats: list[float] = []
    for k in k_sweep:
        far = np.exp(-1j * k * proj).sum(axis=0) / n
        spec = np.abs(np.fft.fft(far)) / _N_ANGLES
        feats.extend(float(spec[m]) for m in modes)
    return feats
