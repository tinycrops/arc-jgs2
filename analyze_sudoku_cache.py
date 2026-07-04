"""Does the Sudoku digit-relabeling cache pay off at realistic scale?

The scenario this models: casual Sudoku apps commonly generate "fresh"
puzzles cheaply by reusing a small library of vetted hole-pattern templates
(fixed given positions, fixed difficulty) and stamping a random digit
relabeling on the underlying solved grid each time -- freshness without
re-solving. SudokuCache.solve is exactly the matching move: canonicalize by
digit role_order, solve the canonical form once per template, reuse it for
every relabeled instance.

Honest scope, stated by testing it directly: the cache pays off ONLY when
the SAME givens-position pattern recurs under a different digit labeling --
infer_relabeling requires identical blank positions by construction (a
relabeling can't move which cells are given). A "control" condition
generates puzzles from the same base grids with an INDEPENDENTLY random hole
pattern per instance (no shared template) to show the cache does NOT
collapse those -- the same code, run on a corpus outside its scope,
correctly gets no benefit. Both conditions solve exactly the same puzzles;
the difference is only whether the corpus has the relabeling structure the
cache targets.
"""

from __future__ import annotations

import random
import time

from arc_jgs2.sudoku import SudokuCache, is_equivalent, solve, validate_givens


def _arithmetic_base() -> list[list[int]]:
    return [[(c + 3 * (r % 3) + r // 3) % 9 + 1 for c in range(9)] for r in range(9)]


def _shuffle_bands(grid: list[list[int]], rng: random.Random) -> list[list[int]]:
    """A real Sudoku symmetry (band/row/column/transpose), deliberately NOT
    a digit relabeling, used only to build genuinely distinct templates."""
    bands = [0, 1, 2]
    rng.shuffle(bands)
    rows_by_band = []
    for b in bands:
        rows = [b * 3, b * 3 + 1, b * 3 + 2]
        rng.shuffle(rows)
        rows_by_band.extend(rows)
    out = [grid[r][:] for r in rows_by_band]

    stacks = [0, 1, 2]
    rng.shuffle(stacks)
    cols_by_stack = []
    for s in stacks:
        cols = [s * 3, s * 3 + 1, s * 3 + 2]
        rng.shuffle(cols)
        cols_by_stack.extend(cols)
    out = [[row[c] for c in cols_by_stack] for row in out]

    if rng.random() < 0.5:
        out = [list(row) for row in zip(*out)]
    return out


def _random_relabeling(rng: random.Random) -> dict[int, int]:
    perm = list(range(1, 10))
    rng.shuffle(perm)
    return dict(zip(range(1, 10), perm))


def _relabel(grid: list[list[int]], mapping: dict[int, int]) -> list[list[int]]:
    return [[mapping[v] if v else 0 for v in row] for row in grid]


def _punch(grid: list[list[int]], cells: set[tuple[int, int]]) -> list[list[int]]:
    out = [row[:] for row in grid]
    for r, c in cells:
        out[r][c] = 0
    return out


def _random_holes(rng: random.Random, n: int) -> set[tuple[int, int]]:
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    return set(cells[:n])


def main() -> None:
    rng = random.Random(0)
    base = _arithmetic_base()

    n_templates = 6
    instances_per_template = 15
    n_holes = 45

    templates = []
    for _ in range(n_templates):
        grid = _shuffle_bands(base, rng)
        holes = _random_holes(rng, n_holes)
        templates.append((grid, holes))

    # sanity check: templates are genuinely distinct under digit relabeling,
    # not an accidental collision (would silently inflate the cache-hit story)
    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            assert not is_equivalent(templates[i][0], templates[j][0]), "template collision"

    # --- condition A: app-template scenario (shared hole pattern per template) ---
    corpus_a = []
    for grid, holes in templates:
        for _ in range(instances_per_template):
            mapping = _random_relabeling(rng)
            corpus_a.append(_punch(_relabel(grid, mapping), holes))
    rng.shuffle(corpus_a)

    # --- condition B: control, independently random holes per instance -----
    corpus_b = []
    for grid, _ in templates:
        for _ in range(instances_per_template):
            mapping = _random_relabeling(rng)
            holes = _random_holes(rng, n_holes)
            corpus_b.append(_punch(_relabel(grid, mapping), holes))
    rng.shuffle(corpus_b)

    def _consistent(puzzle: list[list[int]], result: list[list[int]] | None) -> bool:
        # NOT a bitwise-equality check against an independent solve(): with 36
        # givens (above the 17-clue uniqueness floor but not guaranteed
        # unique), two correct solvers can land on two different, both-valid
        # completions. Correctness means self-consistent, not identical.
        if result is None or any(0 in row for row in result):
            return False
        if not validate_givens(result):
            return False
        return all(puzzle[r][c] == 0 or puzzle[r][c] == result[r][c] for r in range(9) for c in range(9))

    for label, corpus in (("A: shared template + relabeling", corpus_a), ("B: independent holes (control)", corpus_b)):
        cache = SudokuCache()
        t0 = time.perf_counter()
        results = [cache.solve(p) for p in corpus]
        t_cached = time.perf_counter() - t0

        t0 = time.perf_counter()
        baseline = [solve(p) for p in corpus]
        t_uncached = time.perf_counter() - t0

        ok = all(_consistent(p, r) for p, r in zip(corpus, results)) and all(
            _consistent(p, r) for p, r in zip(corpus, baseline)
        )
        print(f"{label}:")
        print(
            f"  {len(corpus)} puzzles, {cache.solve_calls} underlying solves "
            f"({cache.solve_calls / len(corpus):.0%} of corpus)"
        )
        print(f"  wall time  cached={t_cached:.3f}s  uncached={t_uncached:.3f}s  ({t_uncached / t_cached:.1f}x)")
        print(f"  correctness: {'OK, every result self-consistent + valid' if ok else 'MISMATCH'}\n")


if __name__ == "__main__":
    main()
