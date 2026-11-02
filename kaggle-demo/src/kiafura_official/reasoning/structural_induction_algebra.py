"""A-only structural induction for the final two CT-2* regions.

The module exposes two open graphs and one global selector: retain candidates
that are exact on every demonstration, quotient them by query behaviour, and
emit only the uniquely minimum-description region.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence


Grid = tuple[tuple[int, ...], ...]


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    return tuple(tuple(int(cell) for cell in row) for row in value)


def background_of(grid: Grid) -> int:
    return Counter(value for row in grid for value in row).most_common(1)[0][0]


def solid_squares(grid: Grid, background: int) -> dict[int, list[tuple[int, int, int]]]:
    """Return 4-connected, solid monochromatic square components by side."""
    result: dict[int, list[tuple[int, int, int]]] = {}
    for color in sorted({value for row in grid for value in row if value != background}):
        pending = {
            (row, column)
            for row, values in enumerate(grid)
            for column, value in enumerate(values)
            if value == color
        }
        while pending:
            seed = pending.pop()
            component = {seed}
            stack = [seed]
            while stack:
                row, column = stack.pop()
                for neighbor in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    if neighbor in pending:
                        pending.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            rows = [row for row, _ in component]
            columns = [column for _, column in component]
            side = max(rows) - min(rows) + 1
            if (
                side >= 2
                and max(columns) - min(columns) + 1 == side
                and len(component) == side * side
            ):
                result.setdefault(side, []).append((min(rows), min(columns), color))
    return result


def recolor_hidden_rectangles(
    grid: Grid,
    minimum_height: int,
    minimum_width: int,
    output_role: int,
) -> Grid:
    """Recolor solid rectangular runs meeting task-visible size bounds."""
    height, width = len(grid), len(grid[0])
    selected: set[tuple[int, int]] = set()
    for left in range(width):
        for right in range(left + minimum_width - 1, width):
            active_color: int | None = None
            active_top = 0
            for row in range(height + 1):
                color = (
                    grid[row][left]
                    if row < height
                    and all(
                        grid[row][column] == grid[row][left]
                        for column in range(left, right + 1)
                    )
                    else None
                )
                if color == active_color and color is not None:
                    continue
                if active_color is not None and row - active_top >= minimum_height:
                    selected.update(
                        (source_row, column)
                        for source_row in range(active_top, row)
                        for column in range(left, right + 1)
                    )
                active_color = color
                active_top = row
    output = [list(row) for row in grid]
    for row, column in selected:
        output[row][column] = output_role
    return as_grid(output)


def compose_square_histogram(
    grid: Grid,
    sides: tuple[int, ...],
    cap_multiplier: int,
    descending_layers: bool,
    block_gap: int,
) -> Grid:
    """Count solid squares and render their nested histogram composition."""
    background = background_of(grid)
    layers = solid_squares(grid, background)
    if any(not layers.get(side) for side in sides):
        raise ValueError("required square side absent")
    colors = {
        side: Counter(color for _, _, color in layers[side]).most_common(1)[0][0]
        for side in sides
    }
    counts = {
        side: sum(color == colors[side] for _, _, color in layers[side])
        for side in sides
    }
    smallest, largest = min(sides), max(sides)
    counts[smallest] = min(counts[smallest], cap_multiplier * counts[largest])
    output_height = largest + 1
    blocks: list[list[list[int]]] = []
    for index in range(max(counts.values())):
        present = [side for side in sides if index < counts[side]]
        if not present:
            continue
        block_width = max(present) + block_gap
        block = [[background] * block_width for _ in range(output_height)]
        for side in sorted(present, reverse=descending_layers):
            color = colors[side]
            for row in range(side):
                for column in range(side):
                    block[row][column] = color
        blocks.append(block)
    return tuple(
        tuple(value for block in blocks for value in block[row])
        for row in range(output_height)
    )


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]
    description_cost: int

    def execute(self, grid: Grid) -> Grid:
        values = dict(self.bindings)
        if self.graph == "recolor_hidden_rectangles":
            return recolor_hidden_rectangles(
                grid,
                values["minimum_height"],
                values["minimum_width"],
                values["output_role"],
            )
        if self.graph == "compose_square_histogram":
            return compose_square_histogram(
                grid,
                values["sides"],
                values["cap_multiplier"],
                values["descending_layers"],
                values["block_gap"],
            )
        raise ValueError("unknown graph")


def rectangle_candidates(demos: Sequence[tuple[Grid, Grid]]) -> tuple[Candidate, ...]:
    if not demos or not all(
        len(source) == len(expected) and len(source[0]) == len(expected[0])
        for source, expected in demos
    ):
        return ()
    changed_to = {
        expected[row][column]
        for source, expected in demos
        for row in range(len(source))
        for column in range(len(source[0]))
        if source[row][column] != expected[row][column]
    }
    if len(changed_to) != 1:
        return ()
    output_role = next(iter(changed_to))
    maximum_height = min(len(source) for source, _ in demos)
    maximum_width = min(len(source[0]) for source, _ in demos)
    result = []
    for minimum_height in range(2, min(maximum_height, 10) + 1):
        for minimum_width in range(2, min(maximum_width, 10) + 1):
            candidate = Candidate(
                "recolor_hidden_rectangles",
                (
                    ("minimum_height", minimum_height),
                    ("minimum_width", minimum_width),
                    ("output_role", output_role),
                ),
                minimum_height + minimum_width,
            )
            if all(candidate.execute(source) == expected for source, expected in demos):
                result.append(candidate)
    return tuple(result)


def histogram_candidates(demos: Sequence[tuple[Grid, Grid]]) -> tuple[Candidate, ...]:
    if not demos or all(
        len(source) == len(expected) and len(source[0]) == len(expected[0])
        for source, expected in demos
    ):
        return ()
    side_sets = []
    for source, _ in demos:
        sides = {
            side
            for side, items in solid_squares(source, background_of(source)).items()
            if items
        }
        side_sets.append(sides)
    common_sides = tuple(sorted(set.intersection(*side_sets))) if side_sets else ()
    if len(common_sides) < 2:
        return ()
    result = []
    for cap_multiplier in range(1, 7):
        for descending_layers in (False, True):
            for block_gap in range(0, 4):
                candidate = Candidate(
                    "compose_square_histogram",
                    (
                        ("block_gap", block_gap),
                        ("cap_multiplier", cap_multiplier),
                        ("descending_layers", descending_layers),
                        ("sides", common_sides),
                    ),
                    len(common_sides) + cap_multiplier + block_gap + int(descending_layers),
                )
                try:
                    exact = all(
                        candidate.execute(source) == expected
                        for source, expected in demos
                    )
                except (IndexError, KeyError, RuntimeError, StopIteration, TypeError, ValueError):
                    exact = False
                if exact:
                    result.append(candidate)
    return tuple(result)


def candidates(demos: Sequence[tuple[Grid, Grid]]) -> tuple[Candidate, ...]:
    return tuple(
        sorted(
            rectangle_candidates(demos) + histogram_candidates(demos),
            key=lambda item: (item.description_cost, item.graph, repr(item.bindings)),
        )
    )


def solve(
    demos: Sequence[tuple[Grid, Grid]],
    queries: Sequence[Grid],
    *,
    budget: int = 2_000_000,
    allowed_graphs: Sequence[str] | None = None,
) -> dict[str, Any]:
    allowed = set(allowed_graphs) if allowed_graphs is not None else None
    pool = tuple(
        candidate
        for candidate in candidates(demos)
        if allowed is None or candidate.graph in allowed
    )
    work = sum(
        len(source) * len(source[0]) + len(expected) * len(expected[0])
        for source, expected in demos
    ) * max(1, len(pool))
    if work > budget:
        return {"status": "abstain_budget", "outputs": None, "work": work}
    regions: dict[tuple[Grid, ...], list[Candidate]] = {}
    for candidate in pool:
        try:
            outputs = tuple(candidate.execute(query) for query in queries)
        except (IndexError, KeyError, RuntimeError, StopIteration, TypeError, ValueError):
            continue
        regions.setdefault(outputs, []).append(candidate)
    if not regions:
        return {"status": "abstain", "outputs": None, "work": work, "regions": 0}
    ranked = sorted(
        (
            min(member.description_cost for member in members),
            outputs,
            members,
        )
        for outputs, members in regions.items()
    )
    minimum_cost = ranked[0][0]
    best = [item for item in ranked if item[0] == minimum_cost]
    if len(best) != 1:
        return {
            "status": "abstain_no_unique_mdl_region",
            "outputs": None,
            "work": work,
            "regions": len(regions),
        }
    _, outputs, members = best[0]
    return {
        "status": "verified",
        "outputs": outputs,
        "work": work,
        "regions": len(regions),
        "graphs": sorted({member.graph for member in members}),
        "selector": "unique_minimum_description_region",
    }
