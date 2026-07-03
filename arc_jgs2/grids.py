from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable

Grid = list[list[int]]
Point = tuple[int, int]


@dataclass(frozen=True)
class Component:
    color: int
    cells: tuple[Point, ...]
    bbox: tuple[int, int, int, int]
    shape: tuple[Point, ...]

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def height(self) -> int:
        r0, _, r1, _ = self.bbox
        return r1 - r0 + 1

    @property
    def width(self) -> int:
        _, c0, _, c1 = self.bbox
        return c1 - c0 + 1


def dims(grid: Grid) -> tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


def flatten(grid: Grid) -> list[int]:
    return [cell for row in grid for cell in row]


def palette(grid: Grid) -> tuple[int, ...]:
    return tuple(sorted(set(flatten(grid))))


def color_counts(grid: Grid) -> dict[int, int]:
    return dict(sorted(Counter(flatten(grid)).items()))


def background_color(grid: Grid) -> int:
    counts = Counter(flatten(grid))
    if not counts:
        return 0
    return counts.most_common(1)[0][0]


def grid_equal(a: Grid, b: Grid) -> bool:
    return a == b


def transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)] if grid else []


def rotate90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])] if grid else []


def rotate180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def rotate270(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)][::-1] if grid else []


def flip_h(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def flip_v(grid: Grid) -> Grid:
    return grid[::-1]


def transforms(grid: Grid) -> dict[str, Grid]:
    return {
        "identity": [row[:] for row in grid],
        "rotate90": rotate90(grid),
        "rotate180": rotate180(grid),
        "rotate270": rotate270(grid),
        "flip_h": flip_h(grid),
        "flip_v": flip_v(grid),
        "transpose": transpose(grid),
    }


def crop_bbox(grid: Grid, bbox: tuple[int, int, int, int]) -> Grid:
    r0, c0, r1, c1 = bbox
    return [row[c0 : c1 + 1] for row in grid[r0 : r1 + 1]]


def foreground_bbox(grid: Grid, background: int | None = None) -> tuple[int, int, int, int] | None:
    bg = background_color(grid) if background is None else background
    cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value != bg]
    if not cells:
        return None
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def normalize_cells(cells: Iterable[Point]) -> tuple[Point, ...]:
    cells = tuple(sorted(cells))
    if not cells:
        return ()
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def components(grid: Grid, background: int | None = None, include_background: bool = False) -> list[Component]:
    h, w = dims(grid)
    bg = background_color(grid) if background is None else background
    seen: set[Point] = set()
    out: list[Component] = []

    for start_r in range(h):
        for start_c in range(w):
            if (start_r, start_c) in seen:
                continue
            color = grid[start_r][start_c]
            if color == bg and not include_background:
                seen.add((start_r, start_c))
                continue

            q: deque[Point] = deque([(start_r, start_c)])
            seen.add((start_r, start_c))
            cells: list[Point] = []
            while q:
                r, c = q.popleft()
                if grid[r][c] != color:
                    continue
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and grid[nr][nc] == color:
                        seen.add((nr, nc))
                        q.append((nr, nc))

            if cells and (color != bg or include_background):
                rows = [r for r, _ in cells]
                cols = [c for _, c in cells]
                bbox = (min(rows), min(cols), max(rows), max(cols))
                out.append(Component(color=color, cells=tuple(sorted(cells)), bbox=bbox, shape=normalize_cells(cells)))

    out.sort(key=lambda comp: (comp.color, comp.bbox, comp.area))
    return out

