from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .grids import Grid


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid | None = None


@dataclass(frozen=True)
class Task:
    task_id: str
    train: tuple[Pair, ...]
    test: tuple[Pair, ...]
    source: str


def _is_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(row, list) for row in value)
        and all(isinstance(cell, int) for row in value for cell in row)
    )


def _pair(raw: dict[str, Any]) -> Pair:
    if "input" not in raw or not _is_grid(raw["input"]):
        raise ValueError("ARC pair is missing an integer input grid")
    output = raw.get("output")
    if output is not None and not _is_grid(output):
        raise ValueError("ARC pair output is not an integer grid")
    return Pair(input=raw["input"], output=output)


def _task_from_raw(task_id: str, raw: dict[str, Any], source: Path) -> Task:
    train = tuple(_pair(pair) for pair in raw.get("train", []))
    test = tuple(_pair(pair) for pair in raw.get("test", []))
    if not train:
        raise ValueError(f"{task_id} has no train pairs")
    return Task(task_id=task_id, train=train, test=test, source=str(source))


def load_tasks(path: str | Path) -> list[Task]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)

    files = [root] if root.is_file() else sorted(root.rglob("*.json"))
    tasks: list[Task] = []
    for file_path in files:
        raw = json.loads(file_path.read_text())
        if isinstance(raw, dict) and "train" in raw:
            tasks.append(_task_from_raw(file_path.stem, raw, file_path))
            continue
        if isinstance(raw, dict):
            for task_id, task_raw in sorted(raw.items()):
                if isinstance(task_raw, dict) and "train" in task_raw:
                    tasks.append(_task_from_raw(str(task_id), task_raw, file_path))
    return tasks


def demo_tasks() -> list[Task]:
    return [
        Task(
            task_id="demo_color_map",
            train=(
                Pair(input=[[1, 0], [0, 2]], output=[[3, 0], [0, 4]]),
                Pair(input=[[2, 1], [0, 0]], output=[[4, 3], [0, 0]]),
            ),
            test=(Pair(input=[[1, 2], [2, 0]]),),
            source="builtin-demo",
        ),
        Task(
            task_id="demo_crop_foreground",
            train=(
                Pair(input=[[0, 0, 0], [0, 7, 7], [0, 7, 0]], output=[[7, 7], [7, 0]]),
                Pair(input=[[0, 0, 0, 0], [0, 5, 0, 0], [0, 5, 5, 0]], output=[[5, 0], [5, 5]]),
            ),
            test=(Pair(input=[[0, 0, 0], [0, 9, 0], [0, 9, 9]]),),
            source="builtin-demo",
        ),
        Task(
            task_id="demo_rotate90",
            train=(
                Pair(input=[[1, 2], [3, 4]], output=[[3, 1], [4, 2]]),
                Pair(input=[[5, 0], [6, 7]], output=[[6, 5], [7, 0]]),
            ),
            test=(Pair(input=[[8, 1], [2, 3]]),),
            source="builtin-demo",
        ),
    ]

