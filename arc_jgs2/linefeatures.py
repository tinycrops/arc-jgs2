"""Geometric features of the "corpus cylinder" line.

Each grid is flattened; each cell's color is mapped to one of 10 angular slots
via a per-grid role order (background -> slot 0, then the rest of that grid's
palette ranked by frequency -- the same co-rotation used in the cylinder
visualization). Connecting consecutive cells' wheel positions with a line
segment turns the flattened color sequence into a polyline on a unit cylinder;
this module extracts geometric summary statistics from that polyline.

Beyond the raw arc-length / turning statistics of the row-major polyline, we
add three more co-rotation-aware signatures:

* Frenet invariants -- discrete curvature and *torsion* of the 3D polyline.
  Torsion measures how the path twists out of its local plane; it is invariant
  to the arbitrary color labeling (because slots come from role order) and to
  rigid re-embedding, so it is a clean "shape of the reasoning" descriptor.
* Directional anisotropy -- the same polyline traversed column-major instead of
  row-major. A task whose structure is axis-aligned (transpose/rotate/flip
  family) should leave a row/column asymmetry here.
* Edge / boundary statistics -- a true 2D measure over 4-neighbor grid edges
  that does not depend on the raster flattening order at all, including the
  horizontal-vs-vertical boundary asymmetry.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .grids import Grid, background_color, color_counts, dims, flatten


def role_order(grid: Grid) -> list[int]:
    """Background color -> slot 0, remaining palette ranked by frequency."""
    bg = background_color(grid)
    counts = color_counts(grid)
    rest = sorted((c for c in range(10) if c != bg), key=lambda c: -counts.get(c, 0))
    return [bg, *rest]


def _slot_angle(slot: int) -> float:
    return math.radians(-90 + slot * 36)


def _chord(slot_a: int, slot_b: int) -> float:
    delta = abs(slot_a - slot_b) % 10
    delta = min(delta, 10 - delta)
    return 2.0 * math.sin(math.radians(delta * 18))


def _points(slots: list[int]) -> list[tuple[float, float, float]]:
    """Lift a slot sequence to a polyline on the unit cylinder (height = index)."""
    return [(float(i), math.cos(_slot_angle(s)), math.sin(_slot_angle(s))) for i, s in enumerate(slots)]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@dataclass(frozen=True)
class PolyGeom:
    arc_length: float
    mean_step: float
    mean_turn: float
    var_turn: float
    mean_curv: float
    mean_torsion: float
    mean_abs_torsion: float


def _polyline_geometry(slots: list[int]) -> PolyGeom:
    """Arc length, turning, and discrete Frenet curvature/torsion of the polyline."""
    n = len(slots)
    if n < 2:
        return PolyGeom(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    steps = [math.sqrt(1.0 + _chord(slots[i], slots[i + 1]) ** 2) for i in range(n - 1)]
    arc_length = sum(steps)
    mean_step = arc_length / len(steps)

    points = _points(slots)
    segs = [
        (points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1], points[i + 1][2] - points[i][2])
        for i in range(n - 1)
    ]

    turns: list[float] = []
    curvs: list[float] = []
    for i in range(len(segs) - 1):
        a, b = segs[i], segs[i + 1]
        na, nb = _norm(a), _norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        cos_t = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)))
        theta = math.acos(cos_t)
        turns.append(theta)
        curvs.append(theta / (0.5 * (na + nb)))  # turn per unit length ~ curvature

    # discrete torsion: signed angle between consecutive binormals (needs 3 segments)
    torsions: list[float] = []
    for i in range(len(segs) - 2):
        b1 = _cross(segs[i], segs[i + 1])
        b2 = _cross(segs[i + 1], segs[i + 2])
        nb1, nb2 = _norm(b1), _norm(b2)
        if nb1 < 1e-9 or nb2 < 1e-9:
            continue  # collinear stretch: torsion undefined
        cos_t = max(-1.0, min(1.0, (b1[0] * b2[0] + b1[1] * b2[1] + b1[2] * b2[2]) / (nb1 * nb2)))
        angle = math.acos(cos_t)
        # sign from orientation of the twist relative to the middle segment
        sign = 1.0 if (_cross(b1, b2)[0] * segs[i + 1][0] + _cross(b1, b2)[1] * segs[i + 1][1] + _cross(b1, b2)[2] * segs[i + 1][2]) >= 0 else -1.0
        torsions.append(sign * angle)

    mean_turn = sum(turns) / len(turns) if turns else 0.0
    var_turn = sum((t - mean_turn) ** 2 for t in turns) / len(turns) if turns else 0.0
    mean_curv = sum(curvs) / len(curvs) if curvs else 0.0
    mean_torsion = sum(torsions) / len(torsions) if torsions else 0.0
    mean_abs_torsion = sum(abs(t) for t in torsions) / len(torsions) if torsions else 0.0

    return PolyGeom(arc_length, mean_step, mean_turn, var_turn, mean_curv, mean_torsion, mean_abs_torsion)


def _edge_features(grid: Grid, order: list[int]) -> tuple[float, float, float]:
    """4-neighbor boundary stats, independent of raster flattening order.

    Returns (edge_change_frac, edge_hv_anisotropy, edge_mean_chord):
    the fraction of adjacent-cell edges crossing a role change, the
    horizontal-vs-vertical asymmetry of those changes, and the mean color-wheel
    chord over changed edges.
    """
    h, w = dims(grid)
    slot = {v: i for i, v in enumerate(order)}
    h_edges = h_changes = v_edges = v_changes = 0
    chords: list[float] = []
    for r in range(h):
        for c in range(w):
            s = slot[grid[r][c]]
            if c + 1 < w:
                h_edges += 1
                t = slot[grid[r][c + 1]]
                if s != t:
                    h_changes += 1
                    chords.append(_chord(s, t))
            if r + 1 < h:
                v_edges += 1
                t = slot[grid[r + 1][c]]
                if s != t:
                    v_changes += 1
                    chords.append(_chord(s, t))

    total_edges = h_edges + v_edges
    total_changes = h_changes + v_changes
    change_frac = total_changes / total_edges if total_edges else 0.0
    h_rate = h_changes / h_edges if h_edges else 0.0
    v_rate = v_changes / v_edges if v_edges else 0.0
    denom = h_rate + v_rate
    hv_anisotropy = abs(h_rate - v_rate) / denom if denom else 0.0
    mean_chord = sum(chords) / len(chords) if chords else 0.0
    return change_frac, hv_anisotropy, mean_chord


@dataclass(frozen=True)
class LineFeatures:
    cells: int
    n_colors: int
    color_entropy: float
    # row-major polyline
    arc_length: float
    mean_step: float
    mean_turn: float
    var_turn: float
    mean_curv: float
    mean_torsion: float
    mean_abs_torsion: float
    max_zero_run: int
    zero_run_fraction: float
    # directional anisotropy (row-major vs column-major traversal)
    col_arc_length: float
    col_mean_turn: float
    anisotropy_turn: float
    anisotropy_arc: float
    # 4-neighbor boundary structure
    edge_change_frac: float
    edge_hv_anisotropy: float
    edge_mean_chord: float


def _column_major(grid: Grid) -> list[int]:
    h, w = dims(grid)
    return [grid[r][c] for c in range(w) for r in range(h)]


def compute_line_features(grid: Grid) -> LineFeatures:
    order = role_order(grid)
    slot = {v: i for i, v in enumerate(order)}

    row_slots = [slot[v] for v in flatten(grid)]
    col_slots = [slot[v] for v in _column_major(grid)]
    n = len(row_slots)

    counts = color_counts(grid)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0) if total else 0.0

    row = _polyline_geometry(row_slots)
    col = _polyline_geometry(col_slots)

    # still-runs along row-major order
    max_zero_run = zero_run = 0
    for i in range(n - 1):
        if row_slots[i] == row_slots[i + 1]:
            zero_run += 1
            max_zero_run = max(max_zero_run, zero_run)
        else:
            zero_run = 0

    edge_change_frac, edge_hv_anisotropy, edge_mean_chord = _edge_features(grid, order)

    arc_sum = row.arc_length + col.arc_length
    return LineFeatures(
        cells=n,
        n_colors=len(counts),
        color_entropy=entropy,
        arc_length=row.arc_length,
        mean_step=row.mean_step,
        mean_turn=row.mean_turn,
        var_turn=row.var_turn,
        mean_curv=row.mean_curv,
        mean_torsion=row.mean_torsion,
        mean_abs_torsion=row.mean_abs_torsion,
        max_zero_run=max_zero_run,
        zero_run_fraction=max_zero_run / n if n else 0.0,
        col_arc_length=col.arc_length,
        col_mean_turn=col.mean_turn,
        anisotropy_turn=row.mean_turn - col.mean_turn,
        anisotropy_arc=(row.arc_length - col.arc_length) / arc_sum if arc_sum else 0.0,
        edge_change_frac=edge_change_frac,
        edge_hv_anisotropy=edge_hv_anisotropy,
        edge_mean_chord=edge_mean_chord,
    )


# feature keys that are meaningful to average across a task's grids
_AVG_KEYS = (
    "n_colors",
    "color_entropy",
    "arc_length",
    "mean_step",
    "mean_turn",
    "var_turn",
    "mean_curv",
    "mean_torsion",
    "mean_abs_torsion",
    "zero_run_fraction",
    "col_mean_turn",
    "anisotropy_turn",
    "anisotropy_arc",
    "edge_change_frac",
    "edge_hv_anisotropy",
    "edge_mean_chord",
)


def task_line_features(train_grids: list[Grid]) -> dict[str, float]:
    """Cell-count-weighted average of per-grid features across a task's grids."""
    feats = [compute_line_features(grid) for grid in train_grids]
    weights = [f.cells for f in feats]
    total_w = sum(weights) or 1
    dicts = [asdict(f) for f in feats]
    out = {k: sum(d[k] * w for d, w in zip(dicts, weights)) / total_w for k in _AVG_KEYS}
    out["total_cells"] = float(sum(weights))
    out["n_grids"] = float(len(feats))
    return out
