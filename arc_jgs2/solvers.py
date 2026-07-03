from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .grids import Grid, crop_bbox, dims, foreground_bbox, grid_equal, transforms
from .loaders import Pair, Task


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


def _result(task: Task, plan: str, predictions: list[Grid], support: int) -> SolveResult:
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
    mappings = [_infer_color_map_grids(grid, pair.output) for grid, pair in zip(grids, train) if pair.output is not None]
    if not mappings or any(mapping is None for mapping in mappings):
        return None
    first = mappings[0] or {}
    if not all(mapping == first for mapping in mappings):
        return None
    if all(all(src == dst for src, dst in (mapping or {}).items()) for mapping in mappings):
        return None
    return PlanOp(f"global_color_map:{first}", lambda grid, mapping=first: _apply_color_map(grid, mapping))


def _consistent_scale_op(grids: tuple[Grid, ...], train: list[Pair]) -> PlanOp | None:
    factors = [_infer_scale(grid, pair.output) for grid, pair in zip(grids, train) if pair.output is not None]
    if not factors or any(factor is None for factor in factors):
        return None
    first = factors[0]
    if first is None or not all(factor == first for factor in factors):
        return None
    return PlanOp(f"scale:{first[0]}x{first[1]}", lambda grid, factor=first: _scale_grid(grid, *factor))


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
    return ops


def _candidate_ops(state: PlanState, train: list[Pair]) -> list[PlanOp]:
    ops = _static_ops()
    color_op = _consistent_color_op(state.train_grids, train)
    if color_op is not None:
        ops.append(color_op)
    scale_op = _consistent_scale_op(state.train_grids, train)
    if scale_op is not None:
        ops.append(scale_op)
    return ops


def _state_rank(state: PlanState, train: list[Pair]) -> tuple[float, float, int]:
    return (_exact_support(state.train_grids, train), _cell_score(state.train_grids, train), -len(state.plan))


def _search_plan(task: Task, max_depth: int = 4, beam_width: int = 48) -> SolveResult:
    train = [pair for pair in task.train if pair.output is not None]
    total = len(train)
    start = PlanState(
        plan=(),
        train_grids=tuple(pair.input for pair in train),
        test_grids=tuple(pair.input for pair in task.test),
    )
    if total and _exact_support(start.train_grids, train) == total:
        return _result(task, "exact_copy", list(start.test_grids), total)

    beam = [start]
    seen: set[tuple] = {(_grid_signature(start.train_grids), start.plan)}
    best = start

    for _ in range(max_depth):
        next_states: list[PlanState] = []
        for state in beam:
            for op in _candidate_ops(state, train):
                if op.name in state.plan:
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
                    return _result(task, " -> ".join(new_state.plan), list(new_state.test_grids), support)
                next_states.append(new_state)

        if not next_states:
            break
        next_states.sort(key=lambda state: _state_rank(state, train), reverse=True)
        if _state_rank(next_states[0], train) > _state_rank(best, train):
            best = next_states[0]
        beam = next_states[:beam_width]

    return _result(task, "abstain", [], _exact_support(best.train_grids, train))


def solve_task(task: Task) -> SolveResult:
    return _search_plan(task)


def write_solutions(tasks: list[Task], out_dir: str | Path) -> list[SolveResult]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = [solve_task(task) for task in tasks]

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
                }
            )

    summary = {
        "task_count": len(results),
        "attempted": sum(1 for result in results if result.plan != "abstain"),
        "abstained": sum(1 for result in results if result.plan == "abstain"),
        "known_test_total": sum(result.test_known_total for result in results),
        "known_test_correct": sum(result.test_known_correct for result in results),
        "known_tasks_all_correct": sum(1 for result in results if result.solved_known_tests),
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
