"""Sudoku digit relabeling: a REAL instance of the corpus-wide primitive.

ARC's color relabeling is an assumption this repo imposes (colors are
arbitrary labels, so canonicalizing them is a modeling choice). Sudoku's
digit relabeling needs no such assumption: permuting 1-9 by ANY bijection
turns a valid puzzle into another valid puzzle with the correspondingly
relabeled solution, because rows/columns/boxes only ever require "all
different," never a specific digit. This module wires that symmetry in as a
real payoff: canonicalize a puzzle, solve the canonical form once, and reuse
that solution for every relabeled duplicate instead of re-solving.

canonicalize/infer_relabeling mirror linefeatures.role_order and
solvers._infer_color_map_grids; solve is a standard constraint-propagation +
backtracking solver (naked singles, hidden singles, MRV branching).
"""

from __future__ import annotations

Sudoku = list[list[int]]  # 9x9, 0 = blank


def _build_units() -> list[list[tuple[int, int]]]:
    units = [[(r, c) for c in range(9)] for r in range(9)]
    units += [[(r, c) for r in range(9)] for c in range(9)]
    units += [
        [(br * 3 + i, bc * 3 + j) for i in range(3) for j in range(3)]
        for br in range(3)
        for bc in range(3)
    ]
    return units


_UNITS = _build_units()
_PEERS: dict[tuple[int, int], set[tuple[int, int]]] = {
    (r, c): set() for r in range(9) for c in range(9)
}
for _unit in _UNITS:
    for _cell in _unit:
        _PEERS[_cell] |= set(_unit) - {_cell}


def validate_givens(grid: Sudoku) -> bool:
    """No repeated non-zero digit in any row, column, or box."""
    for unit in _UNITS:
        vals = [grid[r][c] for r, c in unit if grid[r][c] != 0]
        if len(vals) != len(set(vals)):
            return False
    return True


def role_order(grid: Sudoku) -> list[int]:
    """Digits 1-9 ranked by first appearance in a row-major scan of the
    givens. First-appearance order (not frequency, unlike linefeatures'
    color role_order) because sparse givens make frequency ties common;
    position order is total and deterministic even on a near-empty puzzle.
    Digits absent from the givens keep their natural order at the tail."""
    seen: list[int] = []
    for row in grid:
        for v in row:
            if v != 0 and v not in seen:
                seen.append(v)
    seen += [d for d in range(1, 10) if d not in seen]
    return seen


def _forward_map(order: list[int]) -> dict[int, int]:
    return {digit: slot + 1 for slot, digit in enumerate(order)}


def canonicalize(grid: Sudoku) -> Sudoku:
    forward = _forward_map(role_order(grid))
    return [[forward[v] if v else 0 for v in row] for row in grid]


def infer_relabeling(a: Sudoku, b: Sudoku) -> dict[int, int] | None:
    """The digit bijection f with f(a) == b, or None if none exists (which
    includes the common case of a differing blank pattern -- a relabeling
    can never change which cells are given)."""
    mapping: dict[int, int] = {}
    used: set[int] = set()
    for ra, rb in zip(a, b):
        for da, db in zip(ra, rb):
            if (da == 0) != (db == 0):
                return None
            if da == 0:
                continue
            if da in mapping:
                if mapping[da] != db:
                    return None
            elif db in used:
                return None
            else:
                mapping[da] = db
                used.add(db)
    return mapping


def is_equivalent(a: Sudoku, b: Sudoku) -> bool:
    return infer_relabeling(a, b) is not None


def _propagate(candidates: dict[tuple[int, int], set[int]]) -> dict[tuple[int, int], set[int]] | None:
    """Naked singles + hidden singles until fixed point or contradiction."""
    candidates = {cell: set(vals) for cell, vals in candidates.items()}
    changed = True
    while changed:
        changed = False
        for cell, vals in list(candidates.items()):
            if len(vals) == 1:
                (val,) = vals
                for peer in _PEERS[cell]:
                    if val in candidates[peer]:
                        if len(candidates[peer]) == 1:
                            return None  # two peers fixed to the same value
                        candidates[peer].discard(val)
                        changed = True
        if any(not vals for vals in candidates.values()):
            return None
        for unit in _UNITS:
            for val in range(1, 10):
                cells_with_val = [cell for cell in unit if val in candidates[cell]]
                if len(cells_with_val) == 1 and candidates[cells_with_val[0]] != {val}:
                    candidates[cells_with_val[0]] = {val}
                    changed = True
    return candidates


def _search(candidates: dict[tuple[int, int], set[int]]) -> Sudoku | None:
    propagated = _propagate(candidates)
    if propagated is None:
        return None
    if all(len(vals) == 1 for vals in propagated.values()):
        grid = [[0] * 9 for _ in range(9)]
        for (r, c), vals in propagated.items():
            grid[r][c] = next(iter(vals))
        return grid

    cell = min((c for c, v in propagated.items() if len(v) > 1), key=lambda c: len(propagated[c]))
    for val in sorted(propagated[cell]):
        attempt = {k: (set(v) if k != cell else {val}) for k, v in propagated.items()}
        result = _search(attempt)
        if result is not None:
            return result
    return None


def solve(grid: Sudoku) -> Sudoku | None:
    if not validate_givens(grid):
        return None
    candidates: dict[tuple[int, int], set[int]] = {}
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                candidates[(r, c)] = {grid[r][c]}
            else:
                used = {grid[pr][pc] for pr, pc in _PEERS[(r, c)] if grid[pr][pc] != 0}
                candidates[(r, c)] = set(range(1, 10)) - used
    return _search(candidates)


class SudokuCache:
    """Solve each puzzle's CANONICAL form once; every relabeled duplicate is
    a cache hit, mapped back through the inverse of its own relabeling."""

    def __init__(self) -> None:
        self._solutions: dict[tuple[tuple[int, ...], ...], Sudoku | None] = {}
        self.solve_calls = 0

    def solve(self, grid: Sudoku) -> Sudoku | None:
        order = role_order(grid)
        forward = _forward_map(order)
        canon = [[forward[v] if v else 0 for v in row] for row in grid]
        key = tuple(tuple(row) for row in canon)

        if key not in self._solutions:
            self.solve_calls += 1
            self._solutions[key] = solve(canon)

        solved_canon = self._solutions[key]
        if solved_canon is None:
            return None
        inverse = {slot + 1: digit for slot, digit in enumerate(order)}
        return [[inverse[v] for v in row] for row in solved_canon]
