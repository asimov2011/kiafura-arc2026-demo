"""Typed A-only operators for rays and periodic spatial propagation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    return tuple(tuple(int(cell) for cell in row) for row in value)


def background_of(grid: Grid) -> int:
    return Counter(value for row in grid for value in row).most_common(1)[0][0]


def extend_cross_rays(grid: Grid, marker: int) -> Grid:
    height, width = len(grid), len(grid[0])
    centers = [
        (row, column, grid[row][column])
        for row in range(1, height - 1)
        for column in range(1, width - 1)
        if grid[row][column] != marker
        and all(
            grid[row + dr][column + dc] == marker
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
    ]
    if not centers:
        raise ValueError("cross center absent")
    output = [list(row) for row in grid]
    for directions in (((0, -1), (0, 1)), ((-1, 0), (1, 0))):
        for row, column, color in centers:
            for dr, dc in directions:
                r, c = row + dr, column + dc
                while 0 <= r < height and 0 <= c < width:
                    if grid[r][c] == marker and (r, c) != (row + dr, column + dc):
                        break
                    if grid[r][c] != marker:
                        output[r][c] = color
                    r, c = r + dr, c + dc
    for row in range(height):
        for column in range(width):
            if grid[row][column] != background_of(grid):
                output[row][column] = grid[row][column]
    return as_grid(output)


def extend_bar_diagonals(grid: Grid, collision: int) -> Grid:
    height, width = len(grid), len(grid[0])
    background = background_of(grid)
    colors = sorted({value for row in grid for value in row if value != background})
    if len(colors) < 2:
        raise ValueError("bar pair absent")
    output = [[background] * width for _ in range(height)]
    for color in colors:
        cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == color]
        columns = sorted({column for _, column in cells})
        top, bottom = min(row for row, _ in cells), max(row for row, _ in cells)
        if len(columns) != 2:
            raise ValueError("non-bar input")
        thickness = bottom - top + 1
        for row in range(height):
            distance = top - row if row < top else row - bottom if row > bottom else 0
            displacement = (distance + thickness - 1) // thickness
            for column, direction in zip(columns, (-1, 1), strict=True):
                target = column + direction * displacement
                if 0 <= target < width:
                    current = output[row][target]
                    output[row][target] = collision if current not in (background, color) else color
    return as_grid(output)


def propagate_block_rays(grid: Grid, block: int, barrier: int, diagonal: int, staircase: int) -> Grid:
    height, width = len(grid), len(grid[0])
    background = background_of(grid)
    cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == block]
    if not cells or not any(barrier in row for row in grid):
        raise ValueError("block or barrier absent")
    top, bottom = min(r for r, _ in cells), max(r for r, _ in cells)
    left, right = min(c for _, c in cells), max(c for _, c in cells)
    if set(cells) != {(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)}:
        raise ValueError("block not rectangular")
    center_row, center_column = (top + bottom) // 2, (left + right) // 2
    output = [list(row) for row in grid]
    for row, column, dr, dc in ((top, left, -1, -1), (top, right, -1, 1), (bottom, left, 1, -1), (bottom, right, 1, 1)):
        row, column = row + dr, column + dc
        while 0 <= row < height and 0 <= column < width:
            horizontal_barrier = 0 < column < width - 1 and grid[row][column - 1] == barrier and grid[row][column + 1] == barrier
            if grid[row][column] == barrier or horizontal_barrier:
                break
            if output[row][column] == background:
                output[row][column] = diagonal
            row, column = row + dr, column + dc
    rays = (
        (top, center_column, ((-1, 0), (0, 1))),
        (center_row, right, ((0, 1), (1, 0))),
        (bottom, center_column, ((1, 0), (0, -1))),
        (center_row, left, ((0, -1), (-1, 0))),
    )
    for row, column, directions in rays:
        stopped = False
        while not stopped:
            for dr, dc in directions:
                for _ in range(2):
                    row, column = row + dr, column + dc
                    if not (0 <= row < height and 0 <= column < width) or grid[row][column] == barrier:
                        stopped = True
                        break
                    if output[row][column] == background:
                        output[row][column] = staircase
                if stopped:
                    break
    return as_grid(output)


def extend_diagonal_periodic(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    background = background_of(grid)
    cells = [(r, c) for r in range(height) for c in range(width) if grid[r][c] != background]
    if not cells:
        raise ValueError("periodic seed absent")
    mean_row = sum(r for r, _ in cells) / len(cells)
    mean_column = sum(c for _, c in cells) / len(cells)
    covariance = sum((r - mean_row) * (c - mean_column) for r, c in cells)
    direction = 1 if covariance >= 0 else -1
    period = None
    for candidate in range(1, min(height, width)):
        matched = 0
        valid = True
        for row in range(candidate, height):
            for column in range(width):
                previous = column - direction * candidate
                if 0 <= previous < width:
                    valid &= grid[row][column] == grid[row - candidate][previous]
                    matched += grid[row][column] != background
            if not valid:
                break
        if valid and matched:
            period = candidate
            break
    if period is None:
        raise ValueError("period absent")
    output = [[background] * (2 * width) for _ in range(2 * height)]
    row_offset = 0 if direction > 0 else height
    limit = 2 * max(height, width)
    for row, column in cells:
        for step in range(-limit, limit + 1):
            target_row = row + row_offset + step * period
            target_column = column + step * direction * period
            if 0 <= target_row < 2 * height and 0 <= target_column < 2 * width:
                output[target_row][target_column] = grid[row][column]
    return as_grid(output)


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]

    def execute(self, grid: Grid) -> Grid:
        values = dict(self.bindings)
        if self.graph == "extend_cross_rays":
            return extend_cross_rays(grid, values["marker"])
        if self.graph == "extend_bar_diagonals":
            return extend_bar_diagonals(grid, values["collision"])
        if self.graph == "propagate_block_rays":
            return propagate_block_rays(grid, values["block"], values["barrier"], values["diagonal"], values["staircase"])
        if self.graph == "extend_diagonal_periodic":
            return extend_diagonal_periodic(grid)
        raise ValueError("unknown graph")


def candidates(demos: Sequence[tuple[Grid, Grid]]) -> tuple[Candidate, ...]:
    same = all(len(source) == len(expected) and len(source[0]) == len(expected[0]) for source, expected in demos)
    if not same:
        if all(len(expected) == 2 * len(source) and len(expected[0]) == 2 * len(source[0]) for source, expected in demos):
            return (Candidate("extend_diagonal_periodic", ()),)
        return ()
    common_input = None
    output_palette = set()
    for source, expected in demos:
        background = background_of(source)
        source_colors = {value for row in source for value in row if value != background}
        output_palette.update(value for row in expected for value in row if value != background)
        common_input = source_colors if common_input is None else common_input & source_colors
    inputs = tuple(sorted(common_input or ()))
    outputs = tuple(sorted(output_palette))
    result = {Candidate("extend_cross_rays", (("marker", marker),)) for marker in inputs}
    result.update(Candidate("extend_bar_diagonals", (("collision", collision),)) for collision in outputs)
    for block, barrier in permutations(inputs, 2):
        for diagonal, staircase in permutations(outputs, 2):
            result.add(Candidate("propagate_block_rays", (("barrier", barrier), ("block", block), ("diagonal", diagonal), ("staircase", staircase))))
    return tuple(sorted(result, key=lambda item: (item.graph, repr(item.bindings))))


def solve(demos, queries, *, budget=2_000_000, allowed_graphs=None):
    allowed = set(allowed_graphs) if allowed_graphs is not None else None
    pool = tuple(candidate for candidate in candidates(demos) if allowed is None or candidate.graph in allowed)
    work = sum(len(a) * len(a[0]) + len(b) * len(b[0]) for a, b in demos) * max(1, len(pool))
    if work > budget:
        return {"status": "abstain_budget", "outputs": None, "work": work}
    exact = []
    for candidate in pool:
        try:
            if all(candidate.execute(source) == expected for source, expected in demos):
                exact.append(candidate)
        except (IndexError, KeyError, RuntimeError, StopIteration, TypeError, ValueError):
            continue
    regions = {}
    for candidate in exact:
        try:
            outputs = tuple(candidate.execute(query) for query in queries)
        except (IndexError, KeyError, RuntimeError, StopIteration, TypeError, ValueError):
            continue
        regions.setdefault(outputs, []).append(candidate)
    if len(regions) != 1:
        return {"status": "abstain_no_consensus", "outputs": None, "work": work, "regions": len(regions)}
    outputs, members = next(iter(regions.items()))
    return {"status": "verified", "outputs": outputs, "work": work, "regions": 1, "graphs": sorted({member.graph for member in members})}
