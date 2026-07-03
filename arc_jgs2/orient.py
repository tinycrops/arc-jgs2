from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean

from .grids import background_color, color_counts, components, dims, palette
from .loaders import Pair, Task
from .primitives import PrimitiveEvidence, primitive_evidence


@dataclass(frozen=True)
class GridSignature:
    dims: tuple[int, int]
    palette: tuple[int, ...]
    background: int
    color_counts: dict[int, int]
    object_count: int
    object_colors: dict[int, int]
    object_shapes: list[dict]


@dataclass(frozen=True)
class PairWitness:
    input_dims: tuple[int, int]
    output_dims: tuple[int, int] | None
    palette_added: tuple[int, ...]
    palette_removed: tuple[int, ...]
    object_delta: int | None
    size_changed: bool | None
    same_palette: bool | None


@dataclass(frozen=True)
class OrientationRecord:
    task_id: str
    source: str
    train_count: int
    test_count: int
    parse_quality: str
    dominant_backgrounds: dict[int, int]
    train_input_dims: list[tuple[int, int]]
    train_output_dims: list[tuple[int, int]]
    test_input_dims: list[tuple[int, int]]
    primitive_evidence: list[PrimitiveEvidence]
    top_family: str
    top_score: float
    witnesses: list[PairWitness]
    global_tags: list[str]
    rejected_overshoots: list[str]
    notes: list[str]


def signature(grid: list[list[int]]) -> GridSignature:
    bg = background_color(grid)
    comps = components(grid, background=bg)
    object_colors = dict(sorted(Counter(comp.color for comp in comps).items()))
    object_shapes = [
        {
            "color": comp.color,
            "area": comp.area,
            "bbox": comp.bbox,
            "frame": (comp.height, comp.width),
            "shape_hash": hash(comp.shape),
        }
        for comp in comps[:20]
    ]
    return GridSignature(
        dims=dims(grid),
        palette=palette(grid),
        background=bg,
        color_counts=color_counts(grid),
        object_count=len(comps),
        object_colors=object_colors,
        object_shapes=object_shapes,
    )


def pair_witness(pair: Pair) -> PairWitness:
    in_sig = signature(pair.input)
    if pair.output is None:
        return PairWitness(
            input_dims=in_sig.dims,
            output_dims=None,
            palette_added=(),
            palette_removed=(),
            object_delta=None,
            size_changed=None,
            same_palette=None,
        )
    out_sig = signature(pair.output)
    in_palette = set(in_sig.palette)
    out_palette = set(out_sig.palette)
    return PairWitness(
        input_dims=in_sig.dims,
        output_dims=out_sig.dims,
        palette_added=tuple(sorted(out_palette - in_palette)),
        palette_removed=tuple(sorted(in_palette - out_palette)),
        object_delta=out_sig.object_count - in_sig.object_count,
        size_changed=in_sig.dims != out_sig.dims,
        same_palette=in_sig.palette == out_sig.palette,
    )


def _choose_top(evidence: list[PrimitiveEvidence], witnesses: list[PairWitness]) -> PrimitiveEvidence:
    explanatory = [item for item in evidence if item.name != "canvas_resize" and item.score > 0]
    if explanatory:
        return explanatory[0]
    if any(w.size_changed for w in witnesses):
        total = witnesses and len(witnesses) or 0
        return PrimitiveEvidence("unexplained_resize", 0.0, 0, total, "size changes but no current primitive explains content")
    total = witnesses and len(witnesses) or 0
    return PrimitiveEvidence("unexplained_same_canvas", 0.0, 0, total, "same canvas but no current primitive explains content")


def _tags(evidence: list[PrimitiveEvidence], witnesses: list[PairWitness], top: PrimitiveEvidence) -> list[str]:
    tags: set[str] = set()
    if top.score == 1.0 and not top.name.startswith("unexplained"):
        tags.add(f"solved_by:{top.name}")
    if any(w.size_changed for w in witnesses):
        tags.add("resize")
    if any(w.palette_added or w.palette_removed for w in witnesses):
        tags.add("palette_shift")
    if any(w.object_delta not in (None, 0) for w in witnesses):
        tags.add("object_count_shift")
    if any(ev.name.startswith("whole_grid_") and ev.score > 0 for ev in evidence):
        tags.add("geometry_candidate")
    if any(ev.name == "global_color_map" and ev.score > 0 for ev in evidence):
        tags.add("color_role_candidate")
    return sorted(tags)


def _rejected_overshoots(evidence: list[PrimitiveEvidence], top: PrimitiveEvidence) -> list[str]:
    rejected: list[str] = []
    for item in evidence:
        if item.name == top.name:
            continue
        if item.name == "canvas_resize" and item.score == 1.0 and top.name.startswith("unexplained"):
            rejected.append("canvas_resize_is_symptom_not_solution")
            continue
        if 0.0 < item.score < 1.0:
            rejected.append(f"partial_{item.name}:{item.support}/{item.total}")
        if item.name == "global_color_map" and item.score == 0.0 and "conflict" in item.detail:
            rejected.append("conflicting_color_maps")
    return rejected


def orient_task(task: Task) -> OrientationRecord:
    evidence = primitive_evidence(task.train)
    witnesses = [pair_witness(pair) for pair in task.train]
    train_inputs = [signature(pair.input) for pair in task.train]
    train_outputs = [signature(pair.output) for pair in task.train if pair.output is not None]
    test_inputs = [signature(pair.input) for pair in task.test]

    top = _choose_top(evidence, witnesses)
    backgrounds = Counter(sig.background for sig in train_inputs)
    object_counts = [sig.object_count for sig in train_inputs]
    notes: list[str] = []
    if object_counts:
        notes.append(f"mean_train_objects={mean(object_counts):.2f}")
    if top.score < 1.0:
        notes.append("needs_iterative_or_compositional_solver")
    if any(w.size_changed for w in witnesses) and top.name not in {"crop_foreground_bbox"}:
        notes.append("resize_not_explained_by_current_primitives")

    return OrientationRecord(
        task_id=task.task_id,
        source=task.source,
        train_count=len(task.train),
        test_count=len(task.test),
        parse_quality="ok",
        dominant_backgrounds=dict(sorted(backgrounds.items())),
        train_input_dims=[sig.dims for sig in train_inputs],
        train_output_dims=[sig.dims for sig in train_outputs],
        test_input_dims=[sig.dims for sig in test_inputs],
        primitive_evidence=evidence,
        top_family=top.name,
        top_score=top.score,
        witnesses=witnesses,
        global_tags=_tags(evidence, witnesses, top),
        rejected_overshoots=_rejected_overshoots(evidence, top),
        notes=notes,
    )


def to_jsonable(record: OrientationRecord) -> dict:
    raw = asdict(record)
    raw["dominant_backgrounds"] = {str(k): v for k, v in record.dominant_backgrounds.items()}
    return raw
