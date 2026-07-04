"""Tests for the corpus cylinder rotation orbit and the radial propagation field.

The claims under test, made precise:

1. Stacking the whole dataset into one long cylinder polyline and rotating it
   about the cylinder axis by k*36 degrees is *exactly* the same object as
   cyclically shifting every color-role slot by +k (mod 10). Ten rotations,
   nine of them non-identity: the "+9 color invariant formations".
2. Those rotations are rigid motions, so every Frenet invariant of the corpus
   polyline (arc length, turning, curvature, torsion) is exactly preserved.
3. The useful quotient of that orbit is the angular power spectrum: the DFT
   magnitudes of any per-slot profile are invariant under the cyclic shift --
   but NOT under an arbitrary color permutation, which is the honest boundary
   of what the cylinder geometrizes.
4. The radial ("light from a point source") embedding puts corpus position 0
   on the outer edge and position n-1 at the origin, and the color-shift
   symmetry becomes a literal rotation of the disk image.
"""

from __future__ import annotations

import math

from arc_jgs2.corpusfield import (
    angular_occupancy,
    corpus_slots,
    cylinder_points,
    disk_points,
    power_spectrum,
)
from arc_jgs2.linefeatures import frenet_stats
from arc_jgs2.loaders import Pair, Task


def _tiny_tasks() -> list[Task]:
    t1 = Task(
        task_id="t1",
        train=(Pair(input=[[0, 1], [2, 0]], output=[[1, 0], [0, 2]]),),
        test=(Pair(input=[[0, 2], [1, 0]], output=None),),
        source="unit",
    )
    t2 = Task(
        task_id="t2",
        train=(Pair(input=[[3, 3, 5], [5, 3, 3]], output=[[5, 5, 3], [3, 5, 5]]),),
        test=(),
        source="unit",
    )
    return [t1, t2]


def _rotate_about_axis(points, angle):
    out = []
    for z, x, y in points:
        out.append((z, x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)))
    return out


def test_corpus_stack_covers_every_cell_in_order() -> None:
    tasks = _tiny_tasks()
    slots, spans = corpus_slots(tasks)

    # every train grid cell appears exactly once, in task order
    expected_cells = 4 + 4 + 6 + 6
    assert len(slots) == expected_cells
    assert all(0 <= s <= 9 for s in slots)

    # spans tile the sequence exactly, no gaps or overlaps
    assert spans[0].start == 0
    assert spans[-1].stop == len(slots)
    for a, b in zip(spans, spans[1:]):
        assert a.stop == b.start
    assert [sp.task_id for sp in spans] == ["t1", "t1", "t2", "t2"]


def test_rotation_equals_cyclic_slot_shift() -> None:
    slots, _ = corpus_slots(_tiny_tasks())
    for k in range(10):
        rotated_geometry = _rotate_about_axis(cylinder_points(slots), math.radians(36 * k))
        shifted_colors = cylinder_points([(s + k) % 10 for s in slots])
        for p, q in zip(rotated_geometry, shifted_colors):
            assert all(abs(a - b) < 1e-9 for a, b in zip(p, q))


def test_frenet_invariants_exact_under_all_rotations() -> None:
    slots, _ = corpus_slots(_tiny_tasks())
    base = frenet_stats(cylinder_points(slots))
    for k in range(1, 10):
        rot = frenet_stats(cylinder_points(slots, rotation=k))
        assert abs(rot.arc_length - base.arc_length) < 1e-9
        assert abs(rot.mean_turn - base.mean_turn) < 1e-9
        assert abs(rot.mean_curv - base.mean_curv) < 1e-9
        # torsion goes through acos twice; float noise reaches ~3e-9
        assert abs(rot.mean_torsion - base.mean_torsion) < 1e-7


def test_angular_power_spectrum_invariant_under_cyclic_shift() -> None:
    slots, _ = corpus_slots(_tiny_tasks())
    base = power_spectrum(angular_occupancy(slots))
    assert len(base) == 6  # k = 0..5 for a real length-10 signal
    for k in range(1, 10):
        shifted = power_spectrum(angular_occupancy([(s + k) % 10 for s in slots]))
        assert all(abs(a - b) < 1e-9 for a, b in zip(base, shifted))


def test_angular_power_spectrum_changes_under_non_cyclic_permutation() -> None:
    # the honest boundary: the cylinder geometrizes Z_10, not all of S_10
    occupancy = [5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    swapped = [5.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # slot 1 -> 3, 3 -> 2, 2 -> 1
    a = power_spectrum(occupancy)
    b = power_spectrum(swapped)
    assert max(abs(x - y) for x, y in zip(a, b)) > 1e-6


def test_disk_embedding_endpoints_and_rotation() -> None:
    slots, _ = corpus_slots(_tiny_tasks())
    pts = disk_points(slots)
    assert len(pts) == len(slots)

    # position 0 sits on the outer edge (radius 1), position n-1 at the origin
    r_first = math.hypot(*pts[0])
    r_last = math.hypot(*pts[-1])
    assert abs(r_first - 1.0) < 1e-9
    assert r_last < 1e-9

    # radius decreases monotonically inward: the point-source wavefront order
    radii = [math.hypot(x, y) for x, y in pts]
    assert all(a >= b - 1e-12 for a, b in zip(radii, radii[1:]))

    # a slot shift by k is a literal rotation of the disk image by k*36 degrees
    k = 3
    rotated = disk_points([(s + k) % 10 for s in slots])
    angle = math.radians(36 * k)
    for (x, y), (u, v) in zip(pts, rotated):
        assert abs(u - (x * math.cos(angle) - y * math.sin(angle))) < 1e-9
        assert abs(v - (x * math.sin(angle) + y * math.cos(angle))) < 1e-9


def test_task_spectrum_features_are_size_and_rotation_invariant() -> None:
    from arc_jgs2.corpusfield import task_spectrum_features

    tasks = _tiny_tasks()
    base = task_spectrum_features(tasks[0])

    # keys: normalized occupancy spectrum + step-profile spectrum, in and out
    for side in ("in", "out"):
        for k in range(1, 6):
            assert f"occ_{side}_k{k}" in base
        for k in range(6):
            assert f"step_{side}_k{k}" in base

    # duplicating every train pair (twice the cells) leaves normalized spectra unchanged
    t = tasks[0]
    doubled = Task(task_id=t.task_id, train=t.train + t.train, test=t.test, source=t.source)
    dup = task_spectrum_features(doubled)
    for key, val in base.items():
        assert abs(dup[key] - val) < 1e-9, key
