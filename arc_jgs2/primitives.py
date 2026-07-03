from __future__ import annotations

from dataclasses import dataclass

from .grids import Grid, crop_bbox, foreground_bbox, grid_equal, transforms
from .loaders import Pair


@dataclass(frozen=True)
class PrimitiveEvidence:
    name: str
    score: float
    support: int
    total: int
    detail: str


def _infer_color_map(pair: Pair) -> dict[int, int] | None:
    if pair.output is None:
        return None
    if len(pair.input) != len(pair.output) or any(len(a) != len(b) for a, b in zip(pair.input, pair.output)):
        return None
    mapping: dict[int, int] = {}
    for in_row, out_row in zip(pair.input, pair.output):
        for src, dst in zip(in_row, out_row):
            if src in mapping and mapping[src] != dst:
                return None
            mapping[src] = dst
    return mapping


def _apply_color_map(grid: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(cell, cell) for cell in row] for row in grid]


def primitive_evidence(train: tuple[Pair, ...]) -> list[PrimitiveEvidence]:
    scored: list[PrimitiveEvidence] = []
    pairs = [pair for pair in train if pair.output is not None]
    total = len(pairs)
    if not pairs:
        return []

    exact = sum(1 for pair in pairs if grid_equal(pair.input, pair.output or []))
    scored.append(PrimitiveEvidence("exact_copy", exact / total, exact, total, "input equals output"))

    for name in ("rotate90", "rotate180", "rotate270", "flip_h", "flip_v", "transpose"):
        support = sum(1 for pair in pairs if grid_equal(transforms(pair.input)[name], pair.output or []))
        scored.append(PrimitiveEvidence(f"whole_grid_{name}", support / total, support, total, "global geometric transform"))

    crop_support = 0
    for pair in pairs:
        bbox = foreground_bbox(pair.input)
        if bbox is not None and grid_equal(crop_bbox(pair.input, bbox), pair.output or []):
            crop_support += 1
    scored.append(PrimitiveEvidence("crop_foreground_bbox", crop_support / total, crop_support, total, "output is tight foreground crop"))

    maps = [_infer_color_map(pair) for pair in pairs]
    if all(mapping is not None for mapping in maps):
        first = maps[0] or {}
        same_map = all(mapping == first for mapping in maps)
        support = sum(1 for pair in pairs if grid_equal(_apply_color_map(pair.input, first), pair.output or []))
        detail = "consistent color role map " + repr(first) if same_map else "per-example color maps conflict"
        scored.append(PrimitiveEvidence("global_color_map", support / total, support, total, detail))
    else:
        scored.append(PrimitiveEvidence("global_color_map", 0.0, 0, total, "shape mismatch or non-functional cell map"))

    resize = sum(1 for pair in pairs if pair.output is not None and (len(pair.input), len(pair.input[0])) != (len(pair.output), len(pair.output[0])))
    scored.append(PrimitiveEvidence("canvas_resize", resize / total, resize, total, "input/output dimensions differ"))

    priority = {
        "global_color_map": 0,
        "crop_foreground_bbox": 1,
        "whole_grid_rotate90": 2,
        "whole_grid_rotate180": 2,
        "whole_grid_rotate270": 2,
        "whole_grid_flip_h": 2,
        "whole_grid_flip_v": 2,
        "whole_grid_transpose": 2,
        "exact_copy": 3,
        "canvas_resize": 9,
    }
    scored.sort(key=lambda item: (-item.score, priority.get(item.name, 5), item.name))
    return scored
