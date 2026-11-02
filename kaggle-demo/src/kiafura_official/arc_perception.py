"""Deterministic, test-output-blind perception for ARC grids."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("ARC grid must be non-empty and rectangular")
    return grid


@dataclass(frozen=True)
class ArcObject:
    object_id: int
    cells: tuple[Point, ...]
    colors: tuple[int, ...]
    bbox: tuple[int, int, int, int]
    shape: tuple[Point, ...]
    symmetries: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class ObjectCluster:
    signature: tuple[Point, ...]
    members: tuple[int, ...]
    colors: tuple[int, ...]


@dataclass(frozen=True)
class Relation:
    source: int
    target: int
    predicates: tuple[str, ...]


@dataclass(frozen=True)
class Scene:
    height: int
    width: int
    background: int
    objects: tuple[ArcObject, ...]
    multicolor_objects: tuple[ArcObject, ...]
    clusters: tuple[ObjectCluster, ...]
    relations: tuple[Relation, ...]
    symmetries: tuple[str, ...]
    periodicity: tuple[int | None, int | None]


@dataclass(frozen=True)
class DemoView:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class TaskView:
    """Leak-proof view: test outputs cannot be represented."""

    demos: tuple[DemoView, ...]
    test_inputs: tuple[Grid, ...]


@dataclass(frozen=True)
class TaskProfile:
    shape_relations: tuple[str, ...]
    color_relations: tuple[str, ...]
    object_bucket: int
    cluster_bucket: int


def task_view(task: Mapping[str, object]) -> TaskView:
    demos = tuple(
        DemoView(as_grid(pair["input"]), as_grid(pair["output"]))
        for pair in task["train"]  # type: ignore[index,union-attr]
    )
    # Intentionally access only ``input``. A poison ``output`` is never touched.
    test_inputs = tuple(
        as_grid(pair["input"]) for pair in task["test"]  # type: ignore[index,union-attr]
    )
    return TaskView(demos, test_inputs)


def _background(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def _symmetries(matrix: Grid) -> tuple[str, ...]:
    result: list[str] = []
    if matrix == tuple(reversed(matrix)):
        result.append("horizontal")
    if matrix == tuple(tuple(reversed(row)) for row in matrix):
        result.append("vertical")
    if len(matrix) == len(matrix[0]):
        transpose = tuple(tuple(matrix[row][column] for row in range(len(matrix))) for column in range(len(matrix[0])))
        if matrix == transpose:
            result.append("diagonal")
        if matrix == tuple(tuple(reversed(row)) for row in reversed(transpose)):
            result.append("anti_diagonal")
    return tuple(result)


def _period(values: tuple[object, ...]) -> int | None:
    for period in range(1, len(values)):
        if all(values[index] == values[index % period] for index in range(len(values))):
            return period
    return None


def _components(grid: Grid, *, multicolor: bool) -> tuple[ArcObject, ...]:
    background = _background(grid)
    height, width = len(grid), len(grid[0])
    remaining = {(row, column) for row in range(height) for column in range(width) if grid[row][column] != background}
    objects: list[ArcObject] = []
    while remaining:
        seed = min(remaining)
        allowed = None if multicolor else grid[seed[0]][seed[1]]
        frontier = [seed]
        cells: set[Point] = set()
        while frontier:
            point = frontier.pop()
            if point not in remaining:
                continue
            if allowed is not None and grid[point[0]][point[1]] != allowed:
                continue
            remaining.remove(point)
            cells.add(point)
            row, column = point
            frontier.extend(
                candidate for candidate in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1))
                if candidate in remaining
            )
        ordered = tuple(sorted(cells))
        top = min(row for row, _ in ordered)
        left = min(column for _, column in ordered)
        bottom = max(row for row, _ in ordered)
        right = max(column for _, column in ordered)
        shape = tuple((row - top, column - left) for row, column in ordered)
        canvas = tuple(
            tuple(1 if (row + top, column + left) in cells else 0 for column in range(right - left + 1))
            for row in range(bottom - top + 1)
        )
        objects.append(
            ArcObject(
                len(objects), ordered,
                tuple(sorted({grid[row][column] for row, column in ordered})),
                (top, left, bottom, right), shape, _symmetries(canvas),
            )
        )
    return tuple(objects)


def _clusters(objects: tuple[ArcObject, ...]) -> tuple[ObjectCluster, ...]:
    grouped: dict[tuple[Point, ...], list[ArcObject]] = defaultdict(list)
    for item in objects:
        grouped[item.shape].append(item)
    return tuple(
        ObjectCluster(signature, tuple(item.object_id for item in members), tuple(sorted({color for item in members for color in item.colors})))
        for signature, members in sorted(grouped.items())
    )


def relations_for(objects: tuple[ArcObject, ...]) -> tuple[Relation, ...]:
    result: list[Relation] = []
    for source in objects:
        for target in objects:
            if source.object_id == target.object_id:
                continue
            st, sl, sb, sr = source.bbox
            tt, tl, tb, tr = target.bbox
            predicates: list[str] = []
            if sr < tl:
                predicates.append("left_of")
            if sl > tr:
                predicates.append("right_of")
            if sb < tt:
                predicates.append("above")
            if st > tb:
                predicates.append("below")
            if source.shape == target.shape:
                predicates.append("same_shape")
            if source.size == target.size:
                predicates.append("same_size")
            if source.colors == target.colors:
                predicates.append("same_color")
            if not (sb < tt or st > tb):
                predicates.append("aligned_row")
            if not (sr < tl or sl > tr):
                predicates.append("aligned_column")
            if st <= tt and sl <= tl and sb >= tb and sr >= tr:
                predicates.append("contains")
            if st >= tt and sl >= tl and sb <= tb and sr <= tr:
                predicates.append("inside")
            if any(abs(a - c) + abs(b - d) == 1 for a, b in source.cells for c, d in target.cells):
                predicates.append("touching")
            if predicates:
                result.append(Relation(source.object_id, target.object_id, tuple(predicates)))
    return tuple(result)


@lru_cache(maxsize=8192)
def _perceive_cached(value: Grid) -> Scene:
    objects = _components(value, multicolor=False)
    multicolor = _components(value, multicolor=True)
    rows = tuple(value)
    columns = tuple(tuple(value[row][column] for row in range(len(value))) for column in range(len(value[0])))
    return Scene(
        len(value), len(value[0]), _background(value), objects, multicolor,
        _clusters(objects), relations_for(objects), _symmetries(value),
        (_period(rows), _period(columns)),
    )


def perceive(grid: Grid | Sequence[Sequence[int]]) -> Scene:
    value = grid if isinstance(grid, tuple) else as_grid(grid)
    return _perceive_cached(value)


def task_level(view: TaskView) -> int:
    scenes = [perceive(pair.input) for pair in view.demos] + [perceive(grid) for grid in view.test_inputs]
    objects = max((len(scene.objects) for scene in scenes), default=0)
    clusters = max((len(scene.clusters) for scene in scenes), default=0)
    relations = max((len(scene.relations) for scene in scenes), default=0)
    score = 1 + (objects > 1) + (clusters > 2) + (relations > 4) + (objects > 8)
    return min(5, score)


def task_profile(view: TaskView) -> TaskProfile:
    scenes = [perceive(pair.input) for pair in view.demos] + [perceive(grid) for grid in view.test_inputs]
    shape_relations: list[str] = []
    color_relations: list[str] = []
    for demo in view.demos:
        input_shape = (len(demo.input), len(demo.input[0]))
        output_shape = (len(demo.output), len(demo.output[0]))
        if input_shape == output_shape:
            shape_relations.append("same")
        elif output_shape[0] <= input_shape[0] and output_shape[1] <= input_shape[1]:
            shape_relations.append("contract")
        elif output_shape[0] >= input_shape[0] and output_shape[1] >= input_shape[1]:
            shape_relations.append("expand")
        else:
            shape_relations.append("mixed")
        input_colors = {cell for row in demo.input for cell in row}
        output_colors = {cell for row in demo.output for cell in row}
        color_relations.append(
            "same" if input_colors == output_colors
            else "adds" if output_colors - input_colors
            else "removes"
        )
    objects = max((len(scene.objects) for scene in scenes), default=0)
    clusters = max((len(scene.clusters) for scene in scenes), default=0)
    return TaskProfile(
        tuple(sorted(set(shape_relations))), tuple(sorted(set(color_relations))),
        min(4, objects // 3), min(4, clusters // 2),
    )


def task_perception(view: TaskView) -> tuple[Scene, ...]:
    return tuple(perceive(pair.input) for pair in view.demos) + tuple(perceive(grid) for grid in view.test_inputs)
