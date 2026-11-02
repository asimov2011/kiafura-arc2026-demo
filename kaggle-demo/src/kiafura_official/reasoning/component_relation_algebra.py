"""Typed component/relation operators derived from A-only experiences."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence


Grid = tuple[tuple[int, ...], ...]
Component = frozenset[tuple[int, int]]
ObjectSet = tuple[Component, ...]


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    return tuple(tuple(int(cell) for cell in row) for row in value)


def background_role(grid: Grid, *, interior: bool = False) -> int:
    rows = grid[1:-1] if interior and len(grid) > 2 else grid
    counts = Counter(value for row in rows for value in row)
    return min(counts, key=lambda value: (-counts[value], value))


def extract_components(
    grid: Grid,
    background: int,
    connectivity: int,
    *,
    color: int | None = None,
    interior: bool = False,
) -> ObjectSet:
    row_range = range(1, len(grid) - 1) if interior else range(len(grid))
    pending = {
        (row, column)
        for row in row_range
        for column, value in enumerate(grid[row])
        if value != background and (color is None or value == color)
    }
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        offsets = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )
    output = []
    while pending:
        seed = min(pending)
        pending.remove(seed)
        component = {seed}
        queue = deque([seed])
        while queue:
            row, column = queue.popleft()
            for dr, dc in offsets:
                neighbor = row + dr, column + dc
                if neighbor in pending:
                    pending.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        output.append(frozenset(component))
    return tuple(sorted(output, key=lambda item: (min(item), len(item))))


def erase_marker_rich(
    grid: Grid,
    marker: int,
    maximum_markers: int,
    connectivity: int,
) -> Grid:
    background = background_role(grid)
    output = [list(row) for row in grid]
    for component in extract_components(grid, background, connectivity):
        marker_count = sum(grid[row][column] == marker for row, column in component)
        if marker_count > maximum_markers:
            for row, column in component:
                output[row][column] = background
    return as_grid(output)


def move_large_to_edge(
    grid: Grid,
    minimum_size: int,
    connectivity: int,
    edge: str,
) -> Grid:
    background = background_role(grid)
    output = [list(row) for row in grid]
    for component in extract_components(grid, background, connectivity):
        if len(component) < minimum_size:
            continue
        top = min(row for row, _ in component)
        bottom = max(row for row, _ in component)
        left = min(column for _, column in component)
        right = max(column for _, column in component)
        row_shift = -top if edge == "top" else len(grid) - 1 - bottom if edge == "bottom" else 0
        column_shift = -left if edge == "left" else len(grid[0]) - 1 - right if edge == "right" else 0
        for row, column in component:
            output[row][column] = background
        for row, column in component:
            target = row + row_shift, column + column_shift
            if not (0 <= target[0] < len(grid) and 0 <= target[1] < len(grid[0])):
                raise ValueError("component leaves canvas")
            output[target[0]][target[1]] = grid[row][column]
    return as_grid(output)


def move_to_horizontal_color_rails(grid: Grid) -> Grid:
    if len(grid) < 3 or len(set(grid[0])) != 1 or len(set(grid[-1])) != 1:
        raise ValueError("two uniform horizontal rails required")
    top_color, bottom_color = grid[0][0], grid[-1][0]
    if top_color == bottom_color:
        raise ValueError("rail roles must be distinct")
    background = background_role(grid, interior=True)
    output = [list(row) for row in grid]
    for color, edge in ((top_color, "top"), (bottom_color, "bottom")):
        objects = extract_components(grid, background, 8, color=color, interior=True)
        if not objects:
            raise ValueError("each rail must govern at least one object")
        for component in objects:
            columns = {column for _, column in component}
            anchored_trace = edge == "bottom" and len(columns) == 1 and next(iter(columns)) in (0, len(grid[0]) - 1)
            if not anchored_trace:
                for row, column in component:
                    output[row][column] = background
        for component in objects:
            shift = 1 - min(row for row, _ in component) if edge == "top" else len(grid) - 2 - max(row for row, _ in component)
            for row, column in component:
                output[row + shift][column] = color
    return as_grid(output)


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]

    def execute(self, grid: Grid) -> Grid:
        values = dict(self.bindings)
        if self.graph == "erase_marker_rich":
            return erase_marker_rich(grid, values["marker"], values["maximum_markers"], values["connectivity"])
        if self.graph == "move_large_to_edge":
            return move_large_to_edge(grid, values["minimum_size"], values["connectivity"], values["edge"])
        if self.graph == "move_to_horizontal_color_rails":
            return move_to_horizontal_color_rails(grid)
        raise ValueError("unknown component-relation graph")


def _palette(grids: Sequence[Grid]) -> tuple[int, ...]:
    return tuple(sorted({value for grid in grids for row in grid for value in row}))


def _same_canvas(demos: Sequence[tuple[Grid, Grid]]) -> bool:
    return all(len(source) == len(expected) and len(source[0]) == len(expected[0]) for source, expected in demos)


def _effect_contract(graph: str, demos: Sequence[tuple[Grid, Grid]]) -> bool:
    if not _same_canvas(demos) or any(source == expected for source, expected in demos):
        return False
    if graph == "erase_marker_rich":
        return all(
            all(expected[row][column] in (source[row][column], background_role(source)) for row in range(len(source)) for column in range(len(source[0])))
            for source, expected in demos
        )
    if graph == "move_large_to_edge":
        return all(Counter(value for row in source for value in row) == Counter(value for row in expected for value in row) for source, expected in demos)
    if graph == "move_to_horizontal_color_rails":
        return all(source[0] == expected[0] and source[-1] == expected[-1] for source, expected in demos)
    return False


def candidates(demos: Sequence[tuple[Grid, Grid]]) -> tuple[Candidate, ...]:
    colors = _palette(tuple(grid for pair in demos for grid in pair))
    result: set[Candidate] = set()
    if _effect_contract("erase_marker_rich", demos):
        for marker, maximum_markers, connectivity in product(colors, range(0, 4), (4, 8)):
            result.add(Candidate("erase_marker_rich", tuple(sorted({"connectivity": connectivity, "marker": marker, "maximum_markers": maximum_markers}.items()))))
    if _effect_contract("move_large_to_edge", demos):
        maximum_component = max(len(source) * len(source[0]) for source, _ in demos)
        for minimum_size, connectivity, edge in product(range(2, min(12, maximum_component) + 1), (4, 8), ("top", "bottom", "left", "right")):
            result.add(Candidate("move_large_to_edge", tuple(sorted({"connectivity": connectivity, "edge": edge, "minimum_size": minimum_size}.items()))))
    if _effect_contract("move_to_horizontal_color_rails", demos):
        result.add(Candidate("move_to_horizontal_color_rails", ()))
    return tuple(sorted(result, key=lambda item: (item.graph, repr(item.bindings))))


def solve(
    demos: Sequence[tuple[Grid, Grid]],
    queries: Sequence[Grid],
    *,
    budget: int = 2_000_000,
    allowed_graphs: Sequence[str] | None = None,
) -> dict[str, Any]:
    allowed = set(allowed_graphs) if allowed_graphs is not None else None
    pool = tuple(candidate for candidate in candidates(demos) if allowed is None or candidate.graph in allowed)
    observed_cells = sum(len(source) * len(source[0]) + len(expected) * len(expected[0]) for source, expected in demos)
    work = observed_cells * max(1, len(pool))
    if work > budget:
        return {"status": "abstain_budget", "outputs": None, "work": work}
    exact = []
    for candidate in pool:
        try:
            if all(candidate.execute(source) == expected for source, expected in demos):
                exact.append(candidate)
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    regions: dict[tuple[Grid, ...], list[Candidate]] = {}
    for candidate in exact:
        try:
            outputs = tuple(candidate.execute(query) for query in queries)
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        regions.setdefault(outputs, []).append(candidate)
    if len(regions) != 1:
        return {"status": "abstain_no_consensus", "outputs": None, "work": work, "regions": len(regions)}
    outputs, members = next(iter(regions.items()))
    return {"status": "verified", "outputs": outputs, "work": work, "regions": 1, "graphs": sorted({member.graph for member in members})}
