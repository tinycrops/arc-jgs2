"""Tests for the Sudoku digit-relabeling primitive (arc_jgs2.sudoku).

Unlike ARC's color relabeling (an assumption we impose to make color an
arbitrary label), Sudoku's digit relabeling is a REAL symmetry: permuting
1-9 by any bijection turns a valid puzzle/solution into another valid one.
canonicalize/infer_relabeling mirror linefeatures.role_order and
solvers._infer_color_map_grids; solve is a real constraint-propagation +
backtracking solver; SudokuCache is the payoff -- solving a puzzle's
canonical form once and re-using it for every relabeled duplicate.
"""

from __future__ import annotations

import random

from arc_jgs2.sudoku import (
    SudokuCache,
    canonicalize,
    infer_relabeling,
    is_equivalent,
    role_order,
    solve,
    validate_givens,
)

# base = (col + 3*(row%3) + row//3) % 9 + 1: a standard construction that is
# always a complete, valid Sudoku grid (checked once, in
# test_base_grid_is_valid_solution, rather than trusted from memory).
def _base_solved() -> list[list[int]]:
    return [[(c + 3 * (r % 3) + r // 3) % 9 + 1 for c in range(9)] for r in range(9)]


def _punch_holes(grid: list[list[int]], cells: list[tuple[int, int]]) -> list[list[int]]:
    out = [row[:] for row in grid]
    for r, c in cells:
        out[r][c] = 0
    return out


def _relabel(grid: list[list[int]], mapping: dict[int, int]) -> list[list[int]]:
    return [[mapping[v] if v else 0 for v in row] for row in grid]


def _is_complete_valid_grid(grid: list[list[int]]) -> bool:
    if any(0 in row for row in grid):
        return False
    return validate_givens(grid)


def test_base_grid_is_valid_solution() -> None:
    assert _is_complete_valid_grid(_base_solved())


def test_role_order_is_first_appearance() -> None:
    grid = _punch_holes(_base_solved(), [])
    # row 0 of the base grid is 1..9 in order already
    assert role_order(grid) == list(range(1, 10))


def test_canonicalize_is_identity_on_first_appearance_order() -> None:
    grid = _base_solved()
    assert canonicalize(grid) == grid


def test_canonicalize_idempotent() -> None:
    grid = _punch_holes(_base_solved(), [(0, 0), (4, 4), (8, 8)])
    once = canonicalize(grid)
    twice = canonicalize(once)
    assert once == twice


def test_relabeled_puzzle_canonicalizes_identically() -> None:
    base = _punch_holes(_base_solved(), [(0, 1), (2, 3), (5, 5), (7, 0)])
    mapping = {i: ((i + 4) % 9) + 1 for i in range(1, 10)}
    relabeled = _relabel(base, mapping)
    assert canonicalize(base) == canonicalize(relabeled)


def test_infer_relabeling_recovers_exact_bijection() -> None:
    base = _punch_holes(_base_solved(), [(0, 1), (3, 3)])
    mapping = {1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 1, 7: 2, 8: 3, 9: 4}
    relabeled = _relabel(base, mapping)
    recovered = infer_relabeling(base, relabeled)
    assert recovered == mapping
    assert is_equivalent(base, relabeled)


def test_infer_relabeling_none_for_different_blank_pattern() -> None:
    a = _punch_holes(_base_solved(), [(0, 0)])
    b = _punch_holes(_base_solved(), [(0, 1)])
    assert infer_relabeling(a, b) is None
    assert not is_equivalent(a, b)


def test_infer_relabeling_none_for_non_relabeled_puzzle() -> None:
    a = _punch_holes(_base_solved(), [(0, 0)])
    b = [row[:] for row in a]
    b[1][1], b[1][2] = b[1][2], b[1][1]  # break the bijection, not a pure relabeling
    assert infer_relabeling(a, b) is None


def test_solve_returns_valid_completion_consistent_with_givens() -> None:
    solved = _base_solved()
    holes = [(r, c) for r in range(9) for c in range(9) if (r * 9 + c) % 2 == 0]
    puzzle = _punch_holes(solved, holes)

    result = solve(puzzle)

    assert result is not None
    assert _is_complete_valid_grid(result)
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != 0:
                assert result[r][c] == puzzle[r][c]


def test_solve_rejects_invalid_givens() -> None:
    grid = _punch_holes(_base_solved(), [])
    grid[0][1] = grid[0][0]  # duplicate in row 0
    assert solve(grid) is None


def test_solve_is_relabeling_invariant() -> None:
    """The end-to-end invariance claim: solving a relabeled puzzle and
    mapping the original solution forward must agree, all the way through
    constraint propagation and backtracking."""
    holes = [(0, 2), (1, 4), (3, 3), (5, 6), (6, 1), (8, 8)]
    base = _punch_holes(_base_solved(), holes)
    mapping = {1: 3, 2: 1, 3: 2, 4: 7, 5: 8, 6: 9, 7: 4, 8: 5, 9: 6}
    relabeled = _relabel(base, mapping)

    solved_base = solve(base)
    solved_relabeled = solve(relabeled)

    assert solved_base is not None and solved_relabeled is not None
    assert _relabel(solved_base, mapping) == solved_relabeled


def test_cache_reuses_solution_across_relabeled_duplicates() -> None:
    holes = [(0, 3), (2, 2), (4, 4), (6, 6), (8, 1)]
    base = _punch_holes(_base_solved(), holes)
    rng = random.Random(0)
    perm = list(range(1, 10))
    rng.shuffle(perm)
    mapping = dict(zip(range(1, 10), perm))
    duplicate = _relabel(base, mapping)

    cache = SudokuCache()
    first = cache.solve(base)
    second = cache.solve(duplicate)

    assert first is not None and second is not None
    assert cache.solve_calls == 1  # the real payoff: only solved once
    assert _relabel(first, mapping) == second


def test_cache_solves_distinct_puzzles_independently() -> None:
    base = _base_solved()
    a = _punch_holes(base, [(0, 0), (1, 1)])
    b = _punch_holes(base, [(0, 0), (2, 2)])  # different blank pattern

    cache = SudokuCache()
    cache.solve(a)
    cache.solve(b)

    assert cache.solve_calls == 2
