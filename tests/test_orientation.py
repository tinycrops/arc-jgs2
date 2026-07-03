from arc_jgs2.loaders import Pair, Task, demo_tasks
from arc_jgs2.orient import orient_task
from arc_jgs2.solvers import solve_task


def test_demo_orients_expected_top_families() -> None:
    records = {task.task_id: orient_task(task) for task in demo_tasks()}
    assert records["demo_color_map"].top_family == "global_color_map"
    assert records["demo_crop_foreground"].top_family == "crop_foreground_bbox"
    assert records["demo_rotate90"].top_family == "whole_grid_rotate90"


def test_demo_solver_attempts_supported_primitives() -> None:
    results = {task.task_id: solve_task(task) for task in demo_tasks()}
    assert results["demo_color_map"].plan.startswith("global_color_map")
    assert results["demo_crop_foreground"].plan == "crop_foreground_bbox"
    assert results["demo_rotate90"].plan == "whole_grid_rotate90"


def test_solver_composes_crop_and_color_map() -> None:
    task = Task(
        task_id="crop_then_recolor",
        train=(
            Pair(input=[[0, 0, 0], [0, 1, 1], [0, 1, 0]], output=[[2, 2], [2, 0]]),
            Pair(input=[[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]], output=[[2, 0], [2, 2]]),
        ),
        test=(Pair(input=[[0, 0, 0], [0, 1, 0], [0, 1, 1]], output=[[2, 0], [2, 2]]),),
        source="unit",
    )

    result = solve_task(task)

    assert result.plan == "crop_foreground_bbox -> global_color_map:{1: 2, 0: 0}"
    assert result.test_known_correct == 1


def test_solver_composes_rotation_and_scale() -> None:
    task = Task(
        task_id="rotate_then_scale",
        train=(
            Pair(input=[[1, 2], [3, 4]], output=[[3, 3, 1, 1], [3, 3, 1, 1], [4, 4, 2, 2], [4, 4, 2, 2]]),
            Pair(input=[[5, 0], [6, 7]], output=[[6, 6, 5, 5], [6, 6, 5, 5], [7, 7, 0, 0], [7, 7, 0, 0]]),
        ),
        test=(Pair(input=[[8, 1], [2, 3]], output=[[2, 2, 8, 8], [2, 2, 8, 8], [3, 3, 1, 1], [3, 3, 1, 1]]),),
        source="unit",
    )

    result = solve_task(task)

    assert result.plan == "whole_grid_rotate90 -> scale:2x2"
    assert result.test_known_correct == 1
