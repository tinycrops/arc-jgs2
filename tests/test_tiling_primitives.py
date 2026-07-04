from arc_jgs2.loaders import Pair, Task
from arc_jgs2.solvers import solve_task


def test_solver_detects_panel_or_with_separator() -> None:
    task = Task(
        task_id="panel_or_demo",
        train=(
            Pair(input=[[1, 0, 4, 0, 1], [0, 1, 4, 1, 0]], output=[[1, 1], [1, 1]]),
            Pair(input=[[0, 0, 4, 1, 0], [1, 0, 4, 0, 0]], output=[[1, 0], [1, 0]]),
        ),
        test=(Pair(input=[[1, 1, 4, 0, 0], [0, 0, 4, 0, 1]], output=[[1, 1], [0, 1]]),),
        source="unit",
    )

    result = solve_task(task)

    assert result.plan == "panel_or:cols:4"
    assert result.test_known_correct == 1


def test_solver_detects_whole_grid_tile() -> None:
    task = Task(
        task_id="tile_demo",
        train=(
            Pair(input=[[1, 2], [3, 4]], output=[[1, 2, 1, 2], [3, 4, 3, 4]]),
            Pair(input=[[5, 0], [0, 6]], output=[[5, 0, 5, 0], [0, 6, 0, 6]]),
        ),
        test=(Pair(input=[[7, 8], [9, 1]], output=[[7, 8, 7, 8], [9, 1, 9, 1]]),),
        source="unit",
    )

    result = solve_task(task)

    assert result.plan == "tile:1x2"
    assert result.test_known_correct == 1


def test_solver_detects_border_add() -> None:
    task = Task(
        task_id="border_add_demo",
        train=(
            Pair(input=[[1, 2]], output=[[9, 9, 9, 9], [9, 1, 2, 9], [9, 9, 9, 9]]),
            Pair(input=[[3, 4]], output=[[9, 9, 9, 9], [9, 3, 4, 9], [9, 9, 9, 9]]),
        ),
        test=(Pair(input=[[5, 6]], output=[[9, 9, 9, 9], [9, 5, 6, 9], [9, 9, 9, 9]]),),
        source="unit",
    )

    result = solve_task(task)

    assert result.plan == "border_add:9"
    assert result.test_known_correct == 1
