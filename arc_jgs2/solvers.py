from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from .grids import (
    Component,
    Grid,
    background_color,
    color_counts,
    components,
    crop_bbox,
    dims,
    flatten,
    flip_h,
    flip_v,
    foreground_bbox,
    grid_equal,
    transforms,
)
from .loaders import Pair, Task
from .wavecheck import wave_consistency


@dataclass(frozen=True)
class SolveResult:
    task_id: str
    plan: str
    train_support: int
    train_total: int
    predictions: list[Grid]
    test_known_total: int
    test_known_correct: int
    source: str
    confidence_pruned: int = 0
    # wave-fingerprint consistency gate (arc_jgs2.wavecheck)
    wave_ok: bool = True
    wave_flags: tuple[str, ...] = ()
    wave_vetoed_plan: str = ""

    @property
    def solved_known_tests(self) -> bool:
        return self.test_known_total > 0 and self.test_known_correct == self.test_known_total


@dataclass(frozen=True)
class PlanOp:
    name: str
    apply: Callable[[Grid], Grid]


@dataclass(frozen=True)
class PlanState:
    plan: tuple[str, ...]
    train_grids: tuple[Grid, ...]
    test_grids: tuple[Grid, ...]


def _infer_color_map(pair: Pair) -> dict[int, int] | None:
    if pair.output is None:
        return None
    return _infer_color_map_grids(pair.input, pair.output)


def _infer_color_map_grids(src_grid: Grid, dst_grid: Grid) -> dict[int, int] | None:
    if len(src_grid) != len(dst_grid) or any(len(a) != len(b) for a, b in zip(src_grid, dst_grid)):
        return None
    mapping: dict[int, int] = {}
    for in_row, out_row in zip(src_grid, dst_grid):
        for src, dst in zip(in_row, out_row):
            if src in mapping and mapping[src] != dst:
                return None
            mapping[src] = dst
    return mapping


def _apply_color_map(grid: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(cell, cell) for cell in row] for row in grid]


def _scale_grid(grid: Grid, row_factor: int, col_factor: int) -> Grid:
    out: Grid = []
    for row in grid:
        expanded: list[int] = []
        for cell in row:
            expanded.extend([cell] * col_factor)
        for _ in range(row_factor):
            out.append(expanded[:])
    return out


def _infer_scale(src_grid: Grid, dst_grid: Grid) -> tuple[int, int] | None:
    src_h, src_w = dims(src_grid)
    dst_h, dst_w = dims(dst_grid)
    if src_h == 0 or src_w == 0 or dst_h % src_h or dst_w % src_w:
        return None
    factors = (dst_h // src_h, dst_w // src_w)
    if factors == (1, 1):
        return None
    if _scale_grid(src_grid, *factors) == dst_grid:
        return factors
    return None


_MAX_GROWTH_CELLS = 4096
_MAX_FRACTAL_INPUT_CELLS = 100


def _within_growth_cap(h: int, w: int) -> bool:
    return h * w <= _MAX_GROWTH_CELLS


def _concat_h(left: Grid, right: Grid) -> Grid:
    if not left or not right or len(left) != len(right):
        return [row[:] for row in left]
    return [list(lrow) + list(rrow) for lrow, rrow in zip(left, right)]


def _concat_v(top: Grid, bottom: Grid) -> Grid:
    if not top or not bottom or len(top[0]) != len(bottom[0]):
        return [row[:] for row in top]
    return [row[:] for row in top] + [row[:] for row in bottom]


def _tile_h_dup(grid: Grid) -> Grid:
    h, w = dims(grid)
    if not _within_growth_cap(h, w * 2):
        return [row[:] for row in grid]
    return _concat_h(grid, grid)


def _tile_h_mirror(grid: Grid) -> Grid:
    h, w = dims(grid)
    if not _within_growth_cap(h, w * 2):
        return [row[:] for row in grid]
    return _concat_h(grid, flip_h(grid))


def _tile_h_mirror_rev(grid: Grid) -> Grid:
    h, w = dims(grid)
    if not _within_growth_cap(h, w * 2):
        return [row[:] for row in grid]
    return _concat_h(flip_h(grid), grid)


def _tile_v_dup(grid: Grid) -> Grid:
    h, w = dims(grid)
    if not _within_growth_cap(h * 2, w):
        return [row[:] for row in grid]
    return _concat_v(grid, grid)


def _tile_v_mirror(grid: Grid) -> Grid:
    h, w = dims(grid)
    if not _within_growth_cap(h * 2, w):
        return [row[:] for row in grid]
    return _concat_v(grid, flip_v(grid))


def _tile_v_mirror_rev(grid: Grid) -> Grid:
    h, w = dims(grid)
    if not _within_growth_cap(h * 2, w):
        return [row[:] for row in grid]
    return _concat_v(flip_v(grid), grid)


def _kaleidoscope_2x2(grid: Grid) -> Grid:
    h, w = dims(grid)
    if not _within_growth_cap(h * 2, w * 2):
        return [row[:] for row in grid]
    half = _concat_h(grid, flip_h(grid))
    return _concat_v(half, flip_v(half))


def _fractal_self_tile(grid: Grid, bg: int) -> Grid:
    h, w = dims(grid)
    if h == 0 or w == 0 or h * w > _MAX_FRACTAL_INPUT_CELLS:
        return [row[:] for row in grid]
    blank = [[bg] * w for _ in range(h)]
    out: Grid = [[0] * (w * w) for _ in range(h * h)]
    for r in range(h):
        for c in range(w):
            block = grid if grid[r][c] != bg else blank
            for i in range(h):
                base_row = r * h + i
                for j in range(w):
                    out[base_row][c * w + j] = block[i][j]
    return out


def _fractal_self_tile_bg0(grid: Grid) -> Grid:
    return _fractal_self_tile(grid, bg=0)


def _fractal_self_tile_majority(grid: Grid) -> Grid:
    return _fractal_self_tile(grid, bg=background_color(grid))


def _grid_signature(grids: tuple[Grid, ...]) -> tuple:
    return tuple(tuple(tuple(row) for row in grid) for grid in grids)


def _known_correct(predictions: list[Grid], task: Task) -> tuple[int, int]:
    known_total = 0
    known_correct = 0
    for prediction, pair in zip(predictions, task.test):
        if pair.output is None:
            continue
        known_total += 1
        if prediction == pair.output:
            known_correct += 1
    return known_total, known_correct


def _all_known_correct(predictions: list[Grid], task: Task) -> tuple[int, int, bool]:
    known_total = 0
    known_correct = 0
    all_correct = True
    for index, pair in enumerate(task.test):
        if pair.output is None:
            continue
        known_total += 1
        prediction = predictions[index] if index < len(predictions) else None
        if prediction == pair.output:
            known_correct += 1
        else:
            all_correct = False
    return known_total, known_correct, known_total > 0 and all_correct


def _result(task: Task, plan: str, predictions: list[Grid], support: int, confidence_pruned: int = 0) -> SolveResult:
    known_total, known_correct = _known_correct(predictions, task)
    return SolveResult(
        task_id=task.task_id,
        plan=plan,
        train_support=support,
        train_total=len(task.train),
        predictions=predictions,
        test_known_total=known_total,
        test_known_correct=known_correct,
        source=task.source,
        confidence_pruned=confidence_pruned,
    )


def _exact_support(grids: tuple[Grid, ...], train: list[Pair]) -> int:
    return sum(1 for grid, pair in zip(grids, train) if pair.output is not None and grid_equal(grid, pair.output))


def _cell_score(grids: tuple[Grid, ...], train: list[Pair]) -> float:
    score = 0.0
    for grid, pair in zip(grids, train):
        if pair.output is None:
            continue
        if dims(grid) != dims(pair.output):
            continue
        total = sum(len(row) for row in pair.output)
        if total == 0:
            continue
        equal = sum(1 for row_a, row_b in zip(grid, pair.output) for a, b in zip(row_a, row_b) if a == b)
        score += equal / total
    return score


def _consistent_color_op(grids: tuple[Grid, ...], train: list[Pair]) -> PlanOp | None:
    # Merge the per-pair color maps into one union map. Different train pairs
    # usually expose different color subsets (a swap task shows a->b in one pair
    # and b->a in another), so requiring identical per-pair maps misses tasks a
    # single global map fully explains. We accept the union only if it is
    # self-consistent, non-identity, and reproduces every train output.
    union: dict[int, int] = {}
    saw_pair = False
    for grid, pair in zip(grids, train):
        if pair.output is None:
            continue
        mapping = _infer_color_map_grids(grid, pair.output)
        if mapping is None:
            return None
        for src, dst in mapping.items():
            if src in union and union[src] != dst:
                return None
            union[src] = dst
        saw_pair = True
    if not saw_pair or all(src == dst for src, dst in union.items()):
        return None
    # Overshoot guard: a genuine relabel/swap is injective. When two remapped
    # colors collapse onto one destination, the "consistent" union is usually an
    # accident assembled from disjoint per-pair evidence (e.g. a relational
    # recolor mistaken for a global map) -- it fits every train pair yet is the
    # wrong abstraction. Abstain rather than commit to it.
    remapped = {src: dst for src, dst in union.items() if src != dst}
    if len(set(remapped.values())) < len(remapped):
        return None
    for grid, pair in zip(grids, train):
        if pair.output is not None and _apply_color_map(grid, union) != pair.output:
            return None
    return PlanOp(f"global_color_map:{union}", lambda grid, mapping=union: _apply_color_map(grid, mapping))


def _consistent_scale_op(grids: tuple[Grid, ...], train: list[Pair]) -> PlanOp | None:
    factors = [_infer_scale(grid, pair.output) for grid, pair in zip(grids, train) if pair.output is not None]
    if not factors or any(factor is None for factor in factors):
        return None
    first = factors[0]
    if first is None or not all(factor == first for factor in factors):
        return None
    return PlanOp(f"scale:{first[0]}x{first[1]}", lambda grid, factor=first: _scale_grid(grid, *factor))


def _find_grid_splits(grid: Grid) -> list[dict]:
    h, w = dims(grid)
    candidates: list[dict] = []

    for c in range(1, w - 1):
        col_vals = {grid[r][c] for r in range(h)}
        if len(col_vals) != 1:
            continue
        left_w = c
        right_w = w - c - 1
        if left_w == right_w and left_w > 0:
            left = [row[:c] for row in grid]
            right = [row[c + 1 :] for row in grid]
            candidates.append({"axis": "cols", "panels": [left, right], "sep_color": next(iter(col_vals))})

    if w > 0 and w % 2 == 0:
        c = w // 2
        left = [row[:c] for row in grid]
        right = [row[c:] for row in grid]
        candidates.append({"axis": "cols", "panels": [left, right], "sep_color": None})

    for r in range(1, h - 1):
        row_vals = set(grid[r])
        if len(row_vals) != 1:
            continue
        top_h = r
        bot_h = h - r - 1
        if top_h == bot_h and top_h > 0:
            top = [row[:] for row in grid[:r]]
            bot = [row[:] for row in grid[r + 1 :]]
            candidates.append({"axis": "rows", "panels": [top, bot], "sep_color": next(iter(row_vals))})

    if h > 0 and h % 2 == 0:
        r = h // 2
        top = [row[:] for row in grid[:r]]
        bot = [row[:] for row in grid[r:]]
        candidates.append({"axis": "rows", "panels": [top, bot], "sep_color": None})

    return candidates


def _panels_bg(panels: list[Grid]) -> int:
    counts: Counter = Counter()
    for panel in panels:
        counts.update(flatten(panel))
    return counts.most_common(1)[0][0] if counts else 0


def _try_overlay_rule(pairs: list[tuple[list[Grid], Grid]], order: tuple[int, int], name: str):
    def rule(panels: list[Grid], order=order) -> Grid:
        bg = _panels_bg(panels)
        primary, secondary = panels[order[0]], panels[order[1]]
        return [[pv if pv != bg else sv for pv, sv in zip(prow, srow)] for prow, srow in zip(primary, secondary)]

    for panels, output in pairs:
        if dims(panels[0]) != dims(output) or rule(panels) != output:
            return None
    rule.__name__ = name  # type: ignore[attr-defined]
    return rule


def _try_cond_rule(pairs: list[tuple[list[Grid], Grid]], cond, name: str):
    fg_color: int | None = None
    bg_color: int | None = None
    for panels, output in pairs:
        if dims(panels[0]) != dims(output) or dims(panels[1]) != dims(output):
            return None
        bg = _panels_bg(panels)
        for ra, rb, ro in zip(panels[0], panels[1], output):
            for av, bv, ov in zip(ra, rb, ro):
                if cond(av, bv, bg):
                    if fg_color is None:
                        fg_color = ov
                    elif fg_color != ov:
                        return None
                else:
                    if bg_color is None:
                        bg_color = ov
                    elif bg_color != ov:
                        return None
    if fg_color is None and bg_color is None:
        return None

    def rule(panels: list[Grid], cond=cond, fg=fg_color, bgc=bg_color) -> Grid:
        bg = _panels_bg(panels)
        out: Grid = []
        for ra, rb in zip(panels[0], panels[1]):
            row = []
            for av, bv in zip(ra, rb):
                if cond(av, bv, bg):
                    row.append(fg if fg is not None else bgc)
                else:
                    row.append(bgc if bgc is not None else bg)
            out.append(row)
        return out

    rule.__name__ = name  # type: ignore[attr-defined]
    return rule


def _try_select_rule(pairs: list[tuple[list[Grid], Grid]], selector, name: str):
    def rule(panels: list[Grid], selector=selector) -> Grid:
        return selector(panels)

    for panels, output in pairs:
        if rule(panels) != output:
            return None
    rule.__name__ = name  # type: ignore[attr-defined]
    return rule


def _infer_combine_rule(pairs: list[tuple[list[Grid], Grid]]):
    if not pairs:
        return None

    def _fg_count(panel: Grid, bg: int) -> int:
        return sum(1 for value in flatten(panel) if value != bg)

    candidates = [
        lambda pairs=pairs: _try_overlay_rule(pairs, (0, 1), "overlay_ab"),
        lambda pairs=pairs: _try_overlay_rule(pairs, (1, 0), "overlay_ba"),
        lambda pairs=pairs: _try_cond_rule(pairs, lambda a, b, bg: a != bg and b != bg, "and"),
        lambda pairs=pairs: _try_cond_rule(pairs, lambda a, b, bg: a != bg or b != bg, "or"),
        lambda pairs=pairs: _try_cond_rule(pairs, lambda a, b, bg: (a != bg) != (b != bg), "xor"),
        lambda pairs=pairs: _try_cond_rule(pairs, lambda a, b, bg: a != b, "diff"),
        lambda pairs=pairs: _try_cond_rule(pairs, lambda a, b, bg: a == b, "equal"),
        lambda pairs=pairs: _try_select_rule(pairs, lambda panels: panels[0], "select_first"),
        lambda pairs=pairs: _try_select_rule(pairs, lambda panels: panels[1], "select_second"),
        lambda pairs=pairs: _try_select_rule(
            pairs,
            lambda panels: panels[0]
            if _fg_count(panels[0], _panels_bg(panels)) >= _fg_count(panels[1], _panels_bg(panels))
            else panels[1],
            "select_more_fg",
        ),
        lambda pairs=pairs: _try_select_rule(
            pairs,
            lambda panels: panels[0]
            if _fg_count(panels[0], _panels_bg(panels)) <= _fg_count(panels[1], _panels_bg(panels))
            else panels[1],
            "select_fewer_fg",
        ),
    ]
    for build in candidates:
        rule = build()
        if rule is not None:
            return rule
    return None


def _consistent_panel_op(grids: tuple[Grid, ...], train: list[Pair]) -> PlanOp | None:
    pairs_with_out = [(grid, pair.output) for grid, pair in zip(grids, train) if pair.output is not None]
    if not pairs_with_out:
        return None
    first_grid, _ = pairs_with_out[0]
    for cand in _find_grid_splits(first_grid):
        kind = (cand["axis"], cand["sep_color"])
        rule_pairs: list[tuple[list[Grid], Grid]] = []
        ok = True
        for grid, output in pairs_with_out:
            match = next((s for s in _find_grid_splits(grid) if (s["axis"], s["sep_color"]) == kind), None)
            if match is None:
                ok = False
                break
            rule_pairs.append((match["panels"], output))
        if not ok:
            continue
        rule = _infer_combine_rule(rule_pairs)
        if rule is None:
            continue
        axis, sep_color = kind

        def apply(grid: Grid, axis=axis, sep_color=sep_color, rule=rule) -> Grid:
            match = next((s for s in _find_grid_splits(grid) if (s["axis"], s["sep_color"]) == (axis, sep_color)), None)
            if match is None:
                return [row[:] for row in grid]
            return rule(match["panels"])

        return PlanOp(f"panel_{rule.__name__}:{axis}:{sep_color}", apply)
    return None


def _plain_tile(grid: Grid, row_reps: int, col_reps: int) -> Grid:
    row_block = [row * col_reps for row in grid]
    out: Grid = []
    for _ in range(row_reps):
        out.extend([row[:] for row in row_block])
    return out


def _mirror_tile(grid: Grid, row_reps: int, col_reps: int) -> Grid:
    out: Grid = []
    for i in range(row_reps):
        block = transforms(grid)["flip_v"] if i % 2 == 1 else [row[:] for row in grid]
        row_blocks = []
        for j in range(col_reps):
            row_blocks.append(transforms(block)["flip_h"] if j % 2 == 1 else block)
        for r in range(len(row_blocks[0])):
            merged: list[int] = []
            for cell_block in row_blocks:
                merged.extend(cell_block[r])
            out.append(merged)
    return out


def _infer_tile_factors(src_grid: Grid, dst_grid: Grid) -> tuple[int, int] | None:
    src_h, src_w = dims(src_grid)
    dst_h, dst_w = dims(dst_grid)
    if src_h == 0 or src_w == 0 or dst_h % src_h or dst_w % src_w:
        return None
    factors = (dst_h // src_h, dst_w // src_w)
    if factors == (1, 1):
        return None
    return factors


def _consistent_tile_op(grids: tuple[Grid, ...], train: list[Pair]) -> PlanOp | None:
    pairs = [(grid, pair.output) for grid, pair in zip(grids, train) if pair.output is not None]
    if not pairs:
        return None
    factors = [_infer_tile_factors(grid, output) for grid, output in pairs]
    if any(factor is None for factor in factors):
        return None
    first = factors[0]
    if first is None or not all(factor == first for factor in factors):
        return None
    row_reps, col_reps = first
    if all(_plain_tile(grid, row_reps, col_reps) == output for grid, output in pairs):
        return PlanOp(f"tile:{row_reps}x{col_reps}", lambda grid, r=row_reps, c=col_reps: _plain_tile(grid, r, c))
    if all(_mirror_tile(grid, row_reps, col_reps) == output for grid, output in pairs):
        return PlanOp(
            f"mirror_tile:{row_reps}x{col_reps}", lambda grid, r=row_reps, c=col_reps: _mirror_tile(grid, r, c)
        )
    return None


def _strip_border(grid: Grid) -> Grid | None:
    h, w = dims(grid)
    if h <= 2 or w <= 2:
        return None
    return [row[1:-1] for row in grid[1:-1]]


def _consistent_border_strip_op(grids: tuple[Grid, ...], train: list[Pair]) -> PlanOp | None:
    pairs = [(grid, pair.output) for grid, pair in zip(grids, train) if pair.output is not None]
    if not pairs:
        return None
    for grid, output in pairs:
        stripped = _strip_border(grid)
        if stripped is None or stripped != output:
            return None
    return PlanOp("border_strip", lambda grid: _strip_border(grid) or [row[:] for row in grid])


def _add_border(grid: Grid, color: int) -> Grid:
    w = len(grid[0]) if grid else 0
    ring = [color] * (w + 2)
    out: Grid = [ring[:]]
    for row in grid:
        out.append([color] + list(row) + [color])
    out.append(ring[:])
    return out


def _consistent_border_add_op(grids: tuple[Grid, ...], train: list[Pair]) -> PlanOp | None:
    pairs = [(grid, pair.output) for grid, pair in zip(grids, train) if pair.output is not None]
    if not pairs:
        return None
    color: int | None = None
    for grid, output in pairs:
        gh, gw = dims(grid)
        dh, dw = dims(output)
        if dh != gh + 2 or dw != gw + 2 or dh == 0 or dw == 0:
            return None
        corner = output[0][0]
        if color is None:
            color = corner
        elif color != corner:
            return None
        if _add_border(grid, color) != output:
            return None
    if color is None:
        return None
    return PlanOp(f"border_add:{color}", lambda grid, color=color: _add_border(grid, color))


def _select_component(comps: list[Component], mode: str) -> Component | None:
    if not comps:
        return None
    extreme_area = max(c.area for c in comps) if mode == "largest" else min(c.area for c in comps)
    tied = [c for c in comps if c.area == extreme_area]
    if len(tied) != 1:
        return None
    return tied[0]


def _keep_object_bbox(grid: Grid, mode: str) -> Grid:
    comp = _select_component(components(grid), mode)
    if comp is None:
        return [row[:] for row in grid]
    return crop_bbox(grid, comp.bbox)


def _keep_object_isolated(grid: Grid, mode: str) -> Grid:
    comp = _select_component(components(grid), mode)
    if comp is None:
        return [row[:] for row in grid]
    bg = background_color(grid)
    r0, c0, r1, c1 = comp.bbox
    cellset = set(comp.cells)
    return [[grid[r][c] if (r, c) in cellset else bg for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]


def _color_counts_excluding_bg(grid: Grid) -> tuple[dict[int, int], int]:
    bg = background_color(grid)
    counts = color_counts(grid)
    counts.pop(bg, None)
    return counts, bg


def _remove_rarest_color(grid: Grid) -> Grid:
    counts, bg = _color_counts_excluding_bg(grid)
    if not counts:
        return [row[:] for row in grid]
    rarest = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return [[bg if cell == rarest else cell for cell in row] for row in grid]


def _isolate_rarest_color(grid: Grid) -> Grid:
    counts, bg = _color_counts_excluding_bg(grid)
    if not counts:
        return [row[:] for row in grid]
    rarest = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return [[cell if cell == rarest else bg for cell in row] for row in grid]


def _isolate_most_common_color(grid: Grid) -> Grid:
    counts, bg = _color_counts_excluding_bg(grid)
    if not counts:
        return [row[:] for row in grid]
    common = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return [[cell if cell == common else bg for cell in row] for row in grid]


def _size_rank_order(grid: Grid, ascending: bool) -> list[Component]:
    comps = components(grid)
    if ascending:
        return sorted(comps, key=lambda c: (c.area, c.bbox))
    return sorted(comps, key=lambda c: (-c.area, c.bbox))


def _apply_size_rank_map(grid: Grid, mapping: dict[int, int], default: int | None, ascending: bool) -> Grid:
    order = _size_rank_order(grid, ascending)
    result = [row[:] for row in grid]
    for rank, comp in enumerate(order):
        color = mapping.get(rank, default)
        if color is None:
            continue
        for r, c in comp.cells:
            result[r][c] = color
    return result


def _tail_default(mapping: dict[int, int]) -> int | None:
    if not mapping:
        return None
    max_rank = max(mapping)
    if max_rank < 1:
        return None
    tail_color = mapping[max_rank]
    run_length = 1
    rank = max_rank - 1
    while rank in mapping and mapping[rank] == tail_color:
        run_length += 1
        rank -= 1
    return tail_color if run_length >= 2 else None


def _infer_size_rank_map(grids: tuple[Grid, ...], train: list[Pair], ascending: bool) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    saw_any = False
    for grid, pair in zip(grids, train):
        if pair.output is None:
            continue
        if dims(grid) != dims(pair.output):
            return None
        order = _size_rank_order(grid, ascending)
        if not order:
            return None
        for rank, comp in enumerate(order):
            colors_out = {pair.output[r][c] for r, c in comp.cells}
            if len(colors_out) != 1:
                return None
            color = colors_out.pop()
            if rank in mapping and mapping[rank] != color:
                return None
            mapping[rank] = color
            saw_any = True
    if not saw_any:
        return None
    if all(
        mapping.get(rank) == comp.color
        for grid, pair in zip(grids, train)
        if pair.output is not None
        for rank, comp in enumerate(_size_rank_order(grid, ascending))
    ):
        return None
    return mapping


def _consistent_size_rank_op(grids: tuple[Grid, ...], train: list[Pair]) -> PlanOp | None:
    for ascending in (True, False):
        mapping = _infer_size_rank_map(grids, train, ascending)
        if mapping is None:
            continue
        default = _tail_default(mapping)
        if not all(
            _apply_size_rank_map(grid, mapping, default, ascending) == pair.output
            for grid, pair in zip(grids, train)
            if pair.output is not None
        ):
            continue
        direction = "asc" if ascending else "desc"
        return PlanOp(
            f"recolor_by_size_rank:{direction}:{mapping}+default={default}",
            lambda grid, mapping=mapping, default=default, ascending=ascending: _apply_size_rank_map(
                grid, mapping, default, ascending
            ),
        )
    return None


_SYMMETRY_CANDIDATE_LIMIT = 12


def _symmetry_maps(h: int, w: int) -> list[tuple[str, Callable[[int, int], tuple[int, int]]]]:
    maps: list[tuple[str, Callable[[int, int], tuple[int, int]]]] = [
        ("flip_h", lambda r, c: (r, w - 1 - c)),
        ("flip_v", lambda r, c: (h - 1 - r, c)),
        ("rotate180", lambda r, c: (h - 1 - r, w - 1 - c)),
    ]
    if h == w:
        maps.append(("transpose", lambda r, c: (c, r)))
        maps.append(("anti_transpose", lambda r, c: (h - 1 - c, h - 1 - r)))
        maps.append(("rotate90", lambda r, c: (h - 1 - c, r)))
        maps.append(("rotate270", lambda r, c: (c, w - 1 - r)))
    return maps


def _repair_symmetry(grid: Grid, mask_color: int) -> Grid | None:
    h, w = dims(grid)
    if h == 0 or w == 0:
        return None
    mask_cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == mask_color]
    if not mask_cells:
        return None

    valid_maps: list[Callable[[int, int], tuple[int, int]]] = []
    for _, func in _symmetry_maps(h, w):
        ok = True
        for r in range(h):
            if not ok:
                break
            for c in range(w):
                value = grid[r][c]
                if value == mask_color:
                    continue
                rr, cc = func(r, c)
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                other = grid[rr][cc]
                if other == mask_color:
                    continue
                if other != value:
                    ok = False
                    break
        if ok:
            valid_maps.append(func)

    if not valid_maps:
        return None

    result = [row[:] for row in grid]
    for r, c in mask_cells:
        candidates: set[int] = set()
        for func in valid_maps:
            rr, cc = func(r, c)
            if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] != mask_color:
                candidates.add(grid[rr][cc])
        if len(candidates) != 1:
            return None
        result[r][c] = candidates.pop()
    return result


def _consistent_symmetry_repair_op(grids: tuple[Grid, ...], train: list[Pair]) -> PlanOp | None:
    fitted = [(grid, pair) for grid, pair in zip(grids, train) if pair.output is not None]
    if not fitted:
        return None
    candidate_colors: set[int] | None = None
    for grid, _ in fitted:
        colors = set(flatten(grid))
        candidate_colors = colors if candidate_colors is None else candidate_colors & colors
    if not candidate_colors:
        return None
    for color in sorted(candidate_colors)[:_SYMMETRY_CANDIDATE_LIMIT]:
        works = True
        for grid, pair in fitted:
            repaired = _repair_symmetry(grid, color)
            if repaired is None or not grid_equal(repaired, pair.output):
                works = False
                break
        if works:
            return PlanOp(
                f"symmetry_repair:mask={color}",
                lambda grid, color=color: _repair_symmetry(grid, color) or [row[:] for row in grid],
            )
    return None


def _static_ops() -> list[PlanOp]:
    ops = [
        PlanOp(f"whole_grid_{name}", lambda grid, name=name: transforms(grid)[name])
        for name in ("rotate90", "rotate180", "rotate270", "flip_h", "flip_v", "transpose")
    ]
    ops.append(
        PlanOp(
            "crop_foreground_bbox",
            lambda grid: crop_bbox(grid, foreground_bbox(grid)) if foreground_bbox(grid) is not None else [row[:] for row in grid],
        )
    )
    ops.append(PlanOp("keep_largest_object_bbox", lambda grid: _keep_object_bbox(grid, "largest")))
    ops.append(PlanOp("keep_smallest_object_bbox", lambda grid: _keep_object_bbox(grid, "smallest")))
    ops.append(PlanOp("keep_largest_object_isolated", lambda grid: _keep_object_isolated(grid, "largest")))
    ops.append(PlanOp("keep_smallest_object_isolated", lambda grid: _keep_object_isolated(grid, "smallest")))
    ops.append(PlanOp("remove_rarest_color", _remove_rarest_color))
    ops.append(PlanOp("isolate_rarest_color", _isolate_rarest_color))
    ops.append(PlanOp("isolate_most_common_color", _isolate_most_common_color))
    ops.append(PlanOp("tile_h_dup", _tile_h_dup))
    ops.append(PlanOp("tile_h_mirror", _tile_h_mirror))
    ops.append(PlanOp("tile_h_mirror_rev", _tile_h_mirror_rev))
    ops.append(PlanOp("tile_v_dup", _tile_v_dup))
    ops.append(PlanOp("tile_v_mirror", _tile_v_mirror))
    ops.append(PlanOp("tile_v_mirror_rev", _tile_v_mirror_rev))
    ops.append(PlanOp("kaleidoscope_2x2", _kaleidoscope_2x2))
    ops.append(PlanOp("fractal_self_tile_bg0", _fractal_self_tile_bg0))
    ops.append(PlanOp("fractal_self_tile_majority", _fractal_self_tile_majority))
    return ops


_CONTENT_STATIC_OPS = {
    "keep_largest_object_bbox",
    "keep_smallest_object_bbox",
    "keep_largest_object_isolated",
    "keep_smallest_object_isolated",
    "remove_rarest_color",
    "isolate_rarest_color",
    "isolate_most_common_color",
}


def _candidate_ops(state: PlanState, train: list[Pair]) -> list[PlanOp]:
    if any(name in _CONTENT_STATIC_OPS for name in state.plan):
        return []
    ops = _static_ops()
    if state.plan:
        ops = [op for op in ops if op.name not in _CONTENT_STATIC_OPS]
    color_op = _consistent_color_op(state.train_grids, train)
    if color_op is not None:
        ops.append(color_op)
    scale_op = _consistent_scale_op(state.train_grids, train)
    if scale_op is not None:
        ops.append(scale_op)
    panel_op = _consistent_panel_op(state.train_grids, train)
    if panel_op is not None:
        ops.append(panel_op)
    tile_op = _consistent_tile_op(state.train_grids, train)
    if tile_op is not None:
        ops.insert(0, tile_op)
    border_strip_op = _consistent_border_strip_op(state.train_grids, train)
    if border_strip_op is not None:
        ops.append(border_strip_op)
    border_add_op = _consistent_border_add_op(state.train_grids, train)
    if border_add_op is not None:
        ops.append(border_add_op)
    size_rank_op = _consistent_size_rank_op(state.train_grids, train)
    if size_rank_op is not None:
        ops.append(size_rank_op)
    return ops


_SWAP_DIM_OPS = {"whole_grid_rotate90", "whole_grid_rotate270", "whole_grid_transpose"}
_SAME_DIM_OPS = {"whole_grid_rotate180", "whole_grid_flip_h", "whole_grid_flip_v"}
_H_DOUBLE_OPS = {"tile_h_dup", "tile_h_mirror", "tile_h_mirror_rev"}
_V_DOUBLE_OPS = {"tile_v_dup", "tile_v_mirror", "tile_v_mirror_rev"}


def _predict_dims(op_name: str, h: int, w: int) -> tuple[int, int] | None:
    if op_name in _SWAP_DIM_OPS:
        return (w, h)
    if op_name in _SAME_DIM_OPS:
        return (h, w)
    if op_name in _H_DOUBLE_OPS:
        return (h, w * 2) if _within_growth_cap(h, w * 2) else (h, w)
    if op_name in _V_DOUBLE_OPS:
        return (h * 2, w) if _within_growth_cap(h * 2, w) else (h, w)
    if op_name == "kaleidoscope_2x2":
        return (h * 2, w * 2) if _within_growth_cap(h * 2, w * 2) else (h, w)
    if op_name in ("fractal_self_tile_bg0", "fractal_self_tile_majority"):
        return (h * h, w * w) if 0 < h * w <= _MAX_FRACTAL_INPUT_CELLS else (h, w)
    if op_name.startswith("scale:"):
        row_str, col_str = op_name.split(":", 1)[1].split("x")
        return (h * int(row_str), w * int(col_str))
    if op_name.startswith("tile:") or op_name.startswith("mirror_tile:"):
        row_str, col_str = op_name.split(":", 1)[1].split("x")
        return (h * int(row_str), w * int(col_str))
    if op_name.startswith("border_add:"):
        return (h + 2, w + 2)
    if op_name == "border_strip":
        return (h - 2, w - 2) if h > 2 and w > 2 else (h, w)
    return None


def _op_dims_plausible(op: PlanOp, state: PlanState, train_out_dims: list[tuple[int, int]]) -> bool:
    for grid, out_dims in zip(state.train_grids, train_out_dims):
        h, w = dims(grid)
        predicted = _predict_dims(op.name, h, w)
        if predicted is not None and predicted != out_dims:
            return False
    return True


def _state_rank(state: PlanState, train: list[Pair]) -> tuple[float, float, int]:
    return (_exact_support(state.train_grids, train), _cell_score(state.train_grids, train), -len(state.plan))


def _probe_confidence(op: PlanOp, state: PlanState, train: list[Pair]) -> float:
    for grid, pair in zip(state.train_grids, train):
        if pair.output is None:
            continue
        candidate = op.apply(grid)
        if dims(candidate) != dims(pair.output):
            return 0.0
        total = sum(len(row) for row in pair.output)
        if total == 0:
            return 0.0
        matched = sum(1 for a_row, b_row in zip(candidate, pair.output) for a, b in zip(a_row, b_row) if a == b)
        return matched / total
    return 0.0


def _search_plan(task: Task, max_depth: int = 4, beam_width: int = 48) -> SolveResult:
    train = [pair for pair in task.train if pair.output is not None]
    total = len(train)
    train_out_dims = [dims(pair.output) for pair in train]
    start = PlanState(
        plan=(),
        train_grids=tuple(pair.input for pair in train),
        test_grids=tuple(pair.input for pair in task.test),
    )
    if total and _exact_support(start.train_grids, train) == total:
        return _result(task, "exact_copy", list(start.test_grids), total)

    symmetry_op = _consistent_symmetry_repair_op(start.train_grids, train)
    if symmetry_op is not None:
        train_grids = tuple(symmetry_op.apply(grid) for grid in start.train_grids)
        support = _exact_support(train_grids, train)
        if total and support == total:
            test_grids = [symmetry_op.apply(grid) for grid in start.test_grids]
            return _result(task, symmetry_op.name, test_grids, support)

    beam = [start]
    seen: set[tuple] = {(_grid_signature(start.train_grids), start.plan)}
    best = start
    pruned = 0

    for depth_index in range(max_depth):
        is_final_depth = depth_index == max_depth - 1
        next_states: list[PlanState] = []
        for state in beam:
            scored = sorted(
                ((op, _probe_confidence(op, state, train)) for op in _candidate_ops(state, train) if op.name not in state.plan),
                key=lambda item: item[1],
                reverse=True,
            )
            for op, confidence in scored:
                if is_final_depth and not _op_dims_plausible(op, state, train_out_dims):
                    pruned += 1
                    continue
                if is_final_depth and confidence <= 0.0:
                    pruned += 1
                    continue
                train_grids = tuple(op.apply(grid) for grid in state.train_grids)
                signature = (_grid_signature(train_grids), state.plan + (op.name,))
                if signature in seen:
                    continue
                seen.add(signature)
                test_grids = tuple(op.apply(grid) for grid in state.test_grids)
                new_state = PlanState(state.plan + (op.name,), train_grids, test_grids)
                support = _exact_support(new_state.train_grids, train)
                if total and support == total:
                    return _result(task, " -> ".join(new_state.plan), list(new_state.test_grids), support, pruned)
                next_states.append(new_state)

        if not next_states:
            break
        next_states.sort(key=lambda state: _state_rank(state, train), reverse=True)
        if _state_rank(next_states[0], train) > _state_rank(best, train):
            best = next_states[0]
        beam = next_states[:beam_width]

    return _result(task, "abstain", [], _exact_support(best.train_grids, train), pruned)


def apply_wave_gate(result: SolveResult, task: Task, veto: bool = False) -> SolveResult:
    """Annotate a result with the wave-fingerprint consistency check; with
    veto=True, out-of-band predictions become abstentions (the check can only
    remove predictions, never create them)."""
    if result.plan == "abstain" or not result.predictions:
        return result
    train = [p for p in task.train if p.output is not None]
    flags = tuple(
        wave_consistency(train, pair.input, prediction).reason
        for pair, prediction in zip(task.test, result.predictions)
    )
    ok = all(flag == "" for flag in flags)
    if ok or not veto:
        return replace(result, wave_ok=ok, wave_flags=flags)
    return replace(
        result,
        plan="abstain",
        predictions=[],
        test_known_total=0,
        test_known_correct=0,
        wave_ok=False,
        wave_flags=flags,
        wave_vetoed_plan=result.plan,
    )


def solve_task(task: Task, wave_veto: bool = False) -> SolveResult:
    return apply_wave_gate(_search_plan(task), task, veto=wave_veto)


def write_solutions(tasks: list[Task], out_dir: str | Path, wave_veto: bool = False) -> list[SolveResult]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = [solve_task(task, wave_veto=wave_veto) for task in tasks]

    with (out / "solutions.jsonl").open("w") as f:
        for result in results:
            f.write(json.dumps(asdict(result), sort_keys=True) + "\n")

    with (out / "solutions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "plan",
                "train_support",
                "train_total",
                "test_predictions",
                "test_known_total",
                "test_known_correct",
                "source",
                "confidence_pruned",
                "wave_ok",
                "wave_vetoed_plan",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "task_id": result.task_id,
                    "plan": result.plan,
                    "train_support": result.train_support,
                    "train_total": result.train_total,
                    "test_predictions": len(result.predictions),
                    "test_known_total": result.test_known_total,
                    "test_known_correct": result.test_known_correct,
                    "source": result.source,
                    "confidence_pruned": result.confidence_pruned,
                    "wave_ok": result.wave_ok,
                    "wave_vetoed_plan": result.wave_vetoed_plan,
                }
            )

    summary = {
        "task_count": len(results),
        "attempted": sum(1 for result in results if result.plan != "abstain"),
        "abstained": sum(1 for result in results if result.plan == "abstain"),
        "known_test_total": sum(result.test_known_total for result in results),
        "known_test_correct": sum(result.test_known_correct for result in results),
        "known_tasks_all_correct": sum(1 for result in results if result.solved_known_tests),
        "confidence_pruned_total": sum(result.confidence_pruned for result in results),
        "wave_flagged": sum(1 for result in results if not result.wave_ok),
        "wave_vetoed": sum(1 for result in results if result.wave_vetoed_plan),
    }
    ground_truth = [_all_known_correct(result.predictions, task) for result, task in zip(results, tasks)]
    gt_total = sum(total for total, _, _ in ground_truth)
    gt_correct = sum(correct for _, correct, _ in ground_truth)
    summary.update(
        {
            "ground_truth_test_total": gt_total,
            "ground_truth_test_correct": gt_correct,
            "ground_truth_test_accuracy": gt_correct / gt_total if gt_total else None,
            "ground_truth_tasks_all_correct": sum(1 for _, _, all_correct in ground_truth if all_correct),
        }
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return results
