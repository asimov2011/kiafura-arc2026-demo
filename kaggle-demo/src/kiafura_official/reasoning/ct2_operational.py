"""CT-2 typed, task-id-blind operational tools.

The module contains only parameterised transformations.  It deliberately
contains neither source programs nor task identifiers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from typing import Callable, Iterable, Mapping, Sequence

from kiafura_official.arc_perception import Grid, as_grid


@dataclass(frozen=True)
class ToolSchema:
    tool_id: str
    input_type: str
    output_type: str
    parameter_names: tuple[str, ...]
    preconditions: tuple[str, ...]
    expected_effect: str
    invariants: tuple[str, ...]
    cost: int
    source_support: int = 0

    def manifest(self) -> dict[str, object]:
        return {
            "cost": self.cost,
            "expected_effect": self.expected_effect,
            "input_type": self.input_type,
            "invariants": list(self.invariants),
            "output_type": self.output_type,
            "parameter_names": list(self.parameter_names),
            "preconditions": list(self.preconditions),
            "source_support": self.source_support,
            "tool_id": self.tool_id,
        }


@dataclass(frozen=True)
class Candidate:
    tool_id: str
    binding: tuple[tuple[str, object], ...]
    demo_outputs: tuple[Grid, ...]
    query_outputs: tuple[Grid, ...]
    work: int

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(
            {
                "binding": self.binding,
                "tool_id": self.tool_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class Resolution:
    status: str
    outputs: tuple[Grid, ...] | None
    compatible_candidates: int
    compatible_tools: tuple[str, ...]
    work: int


def _grid(value: Sequence[Sequence[int]]) -> Grid:
    return as_grid(value)


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _background(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    if 0 in counts:
        return 0
    return min(counts, key=lambda color: (-counts[color], color))


def _palette(grid: Grid) -> tuple[int, ...]:
    return tuple(sorted({cell for row in grid for cell in row}))


def _d4(grid: Grid, mode: str) -> Grid:
    value = grid
    if mode.startswith("reflect_"):
        value = tuple(tuple(reversed(row)) for row in value)
        mode = mode.removeprefix("reflect_")
    turns = int(mode.removeprefix("rot"))
    for _ in range(turns):
        value = tuple(
            tuple(value[len(value) - 1 - row][column] for row in range(len(value)))
            for column in range(len(value[0]))
        )
    return value


def _crop(grid: Grid, selector: str) -> Grid:
    background = _background(grid)
    counts = Counter(cell for row in grid for cell in row if cell != background)
    if not counts:
        return grid
    if selector == "all_foreground":
        selected = set(counts)
    elif selector == "rarest":
        selected = {min(counts, key=lambda color: (counts[color], color))}
    elif selector == "most_common":
        selected = {min(counts, key=lambda color: (-counts[color], color))}
    else:
        raise ValueError("unknown crop selector")
    points = [
        (row, column)
        for row, values in enumerate(grid)
        for column, cell in enumerate(values)
        if cell in selected
    ]
    if not points:
        return grid
    top = min(row for row, _ in points)
    bottom = max(row for row, _ in points)
    left = min(column for _, column in points)
    right = max(column for _, column in points)
    return tuple(
        tuple(grid[row][left : right + 1])
        for row in range(top, bottom + 1)
    )


def _scale(grid: Grid, row_factor: int, column_factor: int) -> Grid:
    return tuple(
        tuple(
            cell
            for cell in row
            for _ in range(column_factor)
        )
        for row in grid
        for _ in range(row_factor)
    )


def _tile(
    grid: Grid,
    row_factor: int,
    column_factor: int,
    alternating_rows: bool,
    alternating_columns: bool,
) -> Grid:
    height, width = _shape(grid)
    rows: list[tuple[int, ...]] = []
    for block_row in range(row_factor):
        source = (
            tuple(reversed(grid))
            if alternating_rows and block_row % 2
            else grid
        )
        for row in source:
            values: list[int] = []
            for block_column in range(column_factor):
                part = (
                    tuple(reversed(row))
                    if alternating_columns and block_column % 2
                    else row
                )
                values.extend(part)
            rows.append(tuple(values))
    assert len(rows) == height * row_factor
    assert all(len(row) == width * column_factor for row in rows)
    return tuple(rows)


def _components(
    grid: Grid,
    connectivity: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    background = _background(grid)
    remaining = {
        (row, column)
        for row, values in enumerate(grid)
        for column, cell in enumerate(values)
        if cell != background
    }
    result: list[tuple[tuple[int, int], ...]] = []
    steps = (
        ((-1, 0), (1, 0), (0, -1), (0, 1))
        if connectivity == 4
        else tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )
    )
    while remaining:
        seed = min(remaining)
        color = grid[seed[0]][seed[1]]
        stack = [seed]
        component = set()
        while stack:
            point = stack.pop()
            if point not in remaining or grid[point[0]][point[1]] != color:
                continue
            remaining.remove(point)
            component.add(point)
            row, column = point
            stack.extend(
                (row + dr, column + dc)
                for dr, dc in steps
                if (row + dr, column + dc) in remaining
            )
        result.append(tuple(sorted(component)))
    return tuple(sorted(result, key=lambda item: (len(item), item)))


def _component_filter(
    grid: Grid,
    connectivity: int,
    rank: str,
    crop: bool,
) -> Grid:
    items = _components(grid, connectivity)
    if not items:
        return grid
    component = items[-1] if rank == "largest" else items[0]
    background = _background(grid)
    points = set(component)
    canvas = tuple(
        tuple(
            grid[row][column] if (row, column) in points else background
            for column in range(len(grid[0]))
        )
        for row in range(len(grid))
    )
    if not crop:
        return canvas
    top = min(row for row, _ in points)
    bottom = max(row for row, _ in points)
    left = min(column for _, column in points)
    right = max(column for _, column in points)
    return tuple(
        tuple(canvas[row][left : right + 1])
        for row in range(top, bottom + 1)
    )


def _bbox(grid: Grid, mode: str, color: int) -> Grid:
    background = _background(grid)
    points = [
        (row, column)
        for row, values in enumerate(grid)
        for column, cell in enumerate(values)
        if cell != background
    ]
    if not points:
        return grid
    top = min(row for row, _ in points)
    bottom = max(row for row, _ in points)
    left = min(column for _, column in points)
    right = max(column for _, column in points)
    output = [list(row) for row in grid]
    for row in range(top, bottom + 1):
        for column in range(left, right + 1):
            if mode == "fill" or row in (top, bottom) or column in (left, right):
                output[row][column] = color
    return _grid(output)


def _fill_holes(grid: Grid, color: int) -> Grid:
    background = _background(grid)
    height, width = _shape(grid)
    outside = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] == background
        and (row in (0, height - 1) or column in (0, width - 1))
    }
    stack = list(outside)
    while stack:
        row, column = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            point = row + dr, column + dc
            if (
                0 <= point[0] < height
                and 0 <= point[1] < width
                and grid[point[0]][point[1]] == background
                and point not in outside
            ):
                outside.add(point)
                stack.append(point)
    return tuple(
        tuple(
            color
            if grid[row][column] == background
            and (row, column) not in outside
            else grid[row][column]
            for column in range(width)
        )
        for row in range(height)
    )


def _connect(grid: Grid, line_color: int, diagonal: bool) -> Grid:
    height, width = _shape(grid)
    output = [list(row) for row in grid]
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    background = _background(grid)
    for row, values in enumerate(grid):
        for column, cell in enumerate(values):
            if cell != background:
                by_color[cell].append((row, column))
    for points in by_color.values():
        for (r1, c1), (r2, c2) in combinations(points, 2):
            dr = (r2 > r1) - (r2 < r1)
            dc = (c2 > c1) - (c2 < c1)
            aligned = r1 == r2 or c1 == c2 or (
                diagonal and abs(r2 - r1) == abs(c2 - c1)
            )
            if not aligned:
                continue
            row, column = r1 + dr, c1 + dc
            while (row, column) != (r2, c2):
                if 0 <= row < height and 0 <= column < width:
                    output[row][column] = line_color
                row += dr
                column += dc
    return _grid(output)


def _separator_panels(grid: Grid) -> tuple[str, int, tuple[Grid, ...]] | None:
    height, width = _shape(grid)
    row_separators = [
        row for row in range(height)
        if len(set(grid[row])) == 1
    ]
    column_separators = [
        column for column in range(width)
        if len({grid[row][column] for row in range(height)}) == 1
    ]
    for axis, separators in (("row", row_separators), ("column", column_separators)):
        if not separators:
            continue
        separator = separators[0]
        if axis == "row":
            parts = (grid[:separator], grid[separator + 1 :])
        else:
            parts = (
                tuple(tuple(row[:separator]) for row in grid),
                tuple(tuple(row[separator + 1 :]) for row in grid),
            )
        if (
            all(parts)
            and _shape(parts[0]) == _shape(parts[1])
        ):
            return axis, separator, parts
    return None


def _panel_combine(grid: Grid, mode: str, mark: int) -> Grid:
    split = _separator_panels(grid)
    if split is None:
        return grid
    _, _, (left, right) = split
    background = _background(grid)
    output = []
    for first, second in zip(left, right):
        row = []
        for a, b in zip(first, second):
            active_a = a != background
            active_b = b != background
            active = {
                "or": active_a or active_b,
                "and": active_a and active_b,
                "xor": active_a != active_b,
                "nor": not active_a and not active_b,
            }[mode]
            row.append(mark if active else background)
        output.append(tuple(row))
    return tuple(output)


def _block_reduce(grid: Grid, row_factor: int, column_factor: int, mode: str) -> Grid:
    height, width = _shape(grid)
    if height % row_factor or width % column_factor:
        return grid
    output = []
    for top in range(0, height, row_factor):
        row = []
        for left in range(0, width, column_factor):
            cells = [
                grid[r][c]
                for r in range(top, top + row_factor)
                for c in range(left, left + column_factor)
            ]
            counts = Counter(cells)
            if mode == "mode":
                value = min(counts, key=lambda color: (-counts[color], color))
            else:
                background = _background(grid)
                foreground = [value for value in cells if value != background]
                value = (
                    min(Counter(foreground), key=lambda color: (-Counter(foreground)[color], color))
                    if foreground else background
                )
            row.append(value)
        output.append(tuple(row))
    return tuple(output)


def _mirror_mosaic(grid: Grid, mode: str) -> Grid:
    horizontal = tuple(tuple(reversed(row)) for row in grid)
    vertical = tuple(reversed(grid))
    both = tuple(tuple(reversed(row)) for row in reversed(grid))
    if mode == "right":
        return tuple(a + b for a, b in zip(grid, horizontal))
    if mode == "left":
        return tuple(a + b for a, b in zip(horizontal, grid))
    if mode == "below":
        return grid + vertical
    if mode == "above":
        return vertical + grid
    if mode == "four_outward":
        top = tuple(a + b for a, b in zip(grid, horizontal))
        bottom = tuple(a + b for a, b in zip(vertical, both))
        return top + bottom
    if mode == "four_inward":
        top = tuple(a + b for a, b in zip(both, vertical))
        bottom = tuple(a + b for a, b in zip(horizontal, grid))
        return top + bottom
    raise ValueError("unknown mirror mosaic")


def _frame_interior(grid: Grid) -> Grid:
    height, width = _shape(grid)
    candidates = []
    for color in _palette(grid):
        if color == _background(grid):
            continue
        for top in range(height - 2):
            for bottom in range(top + 2, height):
                for left in range(width - 2):
                    if grid[top][left] != color or grid[bottom][left] != color:
                        continue
                    for right in range(left + 2, width):
                        border = {
                            (row, column)
                            for row in range(top, bottom + 1)
                            for column in range(left, right + 1)
                            if row in (top, bottom) or column in (left, right)
                        }
                        if all(grid[row][column] == color for row, column in border):
                            candidates.append(((bottom - top - 1) * (right - left - 1), top, bottom, left, right))
    if not candidates:
        return grid
    _, top, bottom, left, right = max(candidates)
    if bottom - top < 2 or right - left < 2:
        return grid
    return tuple(
        tuple(grid[row][left + 1 : right])
        for row in range(top + 1, bottom)
    )


def _largest_solid_rectangle(grid: Grid) -> Grid:
    background = _background(grid)
    rectangles = []
    height, width = _shape(grid)
    for color in _palette(grid):
        if color == background:
            continue
        for top in range(height):
            for left in range(width):
                if grid[top][left] != color:
                    continue
                maximum_width = width - left
                for bottom in range(top, height):
                    run = 0
                    while run < maximum_width and grid[bottom][left + run] == color:
                        run += 1
                    maximum_width = min(maximum_width, run)
                    if maximum_width == 0:
                        break
                    if bottom > top and maximum_width > 1:
                        rectangles.append((
                            (bottom - top + 1) * maximum_width,
                            top,
                            left,
                            bottom,
                            left + maximum_width - 1,
                            color,
                        ))
    if not rectangles:
        return grid
    _, top, left, bottom, right, color = max(rectangles)
    return tuple(
        tuple(
            color
            if top <= row <= bottom and left <= column <= right
            else background
            for column in range(len(grid[0]))
        )
        for row in range(len(grid))
    )


def _quadrant_unique(grid: Grid) -> Grid:
    height, width = _shape(grid)
    separator_rows = [
        row for row in range(height)
        if len(set(grid[row])) == 1
    ]
    separator_columns = [
        column for column in range(width)
        if len({grid[row][column] for row in range(height)}) == 1
    ]
    if len(separator_rows) != 1 or len(separator_columns) != 1:
        return grid
    separator_row = separator_rows[0]
    separator_column = separator_columns[0]
    quadrants = (
        tuple(tuple(row[:separator_column]) for row in grid[:separator_row]),
        tuple(tuple(row[separator_column + 1 :]) for row in grid[:separator_row]),
        tuple(tuple(row[:separator_column]) for row in grid[separator_row + 1 :]),
        tuple(tuple(row[separator_column + 1 :]) for row in grid[separator_row + 1 :]),
    )
    counts = Counter(quadrants)
    unique = [item for item in quadrants if counts[item] == 1]
    return unique[0] if len(unique) == 1 else grid


def _panel_select(grid: Grid, rank: str) -> Grid:
    split = _separator_panels(grid)
    if split is None:
        return grid
    _, _, panels = split
    background = _background(grid)
    key = lambda panel: sum(cell != background for row in panel for cell in row)
    return (
        min(panels, key=lambda panel: (key(panel), panel))
        if rank == "least_ink"
        else max(panels, key=lambda panel: (key(panel), panel))
    )


def _partition_dominant(grid: Grid) -> Grid:
    height, width = _shape(grid)
    separator_rows = [
        row for row in range(height)
        if len(set(grid[row])) == 1
    ]
    separator_columns = [
        column for column in range(width)
        if len({grid[row][column] for row in range(height)}) == 1
    ]
    row_bounds = (-1, *separator_rows, height)
    column_bounds = (-1, *separator_columns, width)
    candidates = []
    background = _background(grid)
    for row_index in range(len(row_bounds) - 1):
        for column_index in range(len(column_bounds) - 1):
            tile = tuple(
                tuple(row[column_bounds[column_index] + 1 : column_bounds[column_index + 1]])
                for row in grid[row_bounds[row_index] + 1 : row_bounds[row_index + 1]]
            )
            if not tile or not tile[0]:
                continue
            counts = Counter(cell for row in tile for cell in row if cell != background)
            for color, count in counts.items():
                candidates.append((count, -color, color, tile))
    if not candidates:
        return grid
    _, _, color, tile = max(candidates)
    return tuple(
        tuple(color if cell == color else background for cell in row)
        for row in tile
    )


def _self_kronecker(grid: Grid, invert: bool) -> Grid:
    background = _background(grid)
    foregrounds = [cell for row in grid for cell in row if cell != background]
    if not foregrounds:
        return grid
    foreground = Counter(foregrounds).most_common(1)[0][0]
    output = []
    for outer_row in grid:
        for inner_row in grid:
            row = []
            for outer in outer_row:
                row.extend(
                    foreground
                    if ((outer != background) and (inner != background)) != invert
                    else background
                    for inner in inner_row
                )
            output.append(tuple(row))
    return tuple(output)


def _checkerboard_rows(grid: Grid, reverse_phase: bool) -> Grid:
    if len(grid) != 2:
        return grid
    width = len(grid[0])
    first = Counter(grid[0]).most_common(1)[0][0]
    second = Counter(grid[1]).most_common(1)[0][0]
    return tuple(
        tuple(
            (first, second)[(row + column + int(reverse_phase)) % 2]
            for column in range(width)
        )
        for row in range(2)
    )


def _fill_corner_rectangles(grid: Grid, marker: int, fill: int) -> Grid:
    points = [
        (row, column)
        for row, values in enumerate(grid)
        for column, cell in enumerate(values)
        if cell == marker
    ]
    output = [list(row) for row in grid]
    for top, bottom in combinations(sorted({row for row, _ in points}), 2):
        for left, right in combinations(sorted({column for _, column in points}), 2):
            if all(
                (row, column) in points
                for row, column in (
                    (top, left), (top, right), (bottom, left), (bottom, right)
                )
            ):
                for row in range(top + 1, bottom):
                    for column in range(left + 1, right):
                        output[row][column] = fill
    return _grid(output)


def _recolor_inside_marker_bbox(
    grid: Grid,
    marker: int,
    target: int,
    replacement: int,
) -> Grid:
    points = [
        (row, column)
        for row, values in enumerate(grid)
        for column, cell in enumerate(values)
        if cell == marker
    ]
    if not points:
        return grid
    top = min(row for row, _ in points)
    bottom = max(row for row, _ in points)
    left = min(column for _, column in points)
    right = max(column for _, column in points)
    return tuple(
        tuple(
            replacement
            if top <= row <= bottom
            and left <= column <= right
            and cell == target
            else cell
            for column, cell in enumerate(values)
        )
        for row, values in enumerate(grid)
    )


def _header_intersections(grid: Grid, marker: int, fill: int) -> Grid:
    height, width = _shape(grid)
    top_columns = [column for column in range(width) if grid[0][column] == marker]
    right_rows = [row for row in range(height) if grid[row][width - 1] == marker]
    output = [list(row) for row in grid]
    for row in right_rows:
        for column in top_columns:
            if output[row][column] == _background(grid):
                output[row][column] = fill
    return _grid(output)


def _l_connect(
    grid: Grid,
    first_color: int,
    second_color: int,
    line_color: int,
    corner_mode: str,
) -> Grid:
    first = [
        (row, column)
        for row, values in enumerate(grid)
        for column, cell in enumerate(values)
        if cell == first_color
    ]
    second = [
        (row, column)
        for row, values in enumerate(grid)
        for column, cell in enumerate(values)
        if cell == second_color
    ]
    if len(first) != 1 or len(second) != 1:
        return grid
    (first_row, first_column), (second_row, second_column) = first[0], second[0]
    corner = (
        (first_row, second_column)
        if corner_mode == "first_row_second_column"
        else (second_row, first_column)
    )
    output = [list(row) for row in grid]
    for endpoint in ((first_row, first_column), (second_row, second_column)):
        row, column = endpoint
        row_step = (corner[0] > row) - (corner[0] < row)
        column_step = (corner[1] > column) - (corner[1] < column)
        while (row, column) != corner:
            if output[row][column] == _background(grid):
                output[row][column] = line_color
            if row != corner[0]:
                row += row_step
            else:
                column += column_step
        if output[row][column] == _background(grid):
            output[row][column] = line_color
    return _grid(output)


def _parity_half_crop(grid: Grid, odd_side: str) -> Grid:
    height, width = _shape(grid)
    half_height, half_width = height // 2, width // 2
    use_right = (height % 2 == 0) != (odd_side == "right")
    left = half_width if use_right else 0
    return tuple(
        tuple(row[left : left + half_width])
        for row in grid[:half_height]
    )


Transform = Callable[[Grid], Grid]


def _candidate(
    tool_id: str,
    binding: Mapping[str, object],
    transform: Transform,
    demos: Sequence[tuple[Grid, Grid]],
    queries: Sequence[Grid],
    work: int,
) -> Candidate | None:
    try:
        demo_outputs = tuple(transform(source) for source, _ in demos)
        if demo_outputs != tuple(expected for _, expected in demos):
            return None
        query_outputs = tuple(transform(source) for source in queries)
    except (AssertionError, IndexError, KeyError, TypeError, ValueError):
        return None
    return Candidate(
        tool_id,
        tuple(sorted(binding.items())),
        demo_outputs,
        query_outputs,
        work,
    )


def enumerate_candidates(
    demos: Sequence[tuple[Grid, Grid]],
    queries: Sequence[Grid],
    enabled_tools: Iterable[str] | None = None,
) -> tuple[Candidate, ...]:
    """Enumerate bindings from visible task data only."""
    enabled = set(enabled_tools) if enabled_tools is not None else None
    inputs = tuple(source for source, _ in demos)
    outputs = tuple(expected for _, expected in demos)
    all_grids = (*inputs, *queries)
    candidates: list[Candidate] = []

    def add(
        tool_id: str,
        binding: Mapping[str, object],
        transform: Transform,
    ) -> None:
        if enabled is not None and tool_id not in enabled:
            return
        item = _candidate(
            tool_id, binding, transform, demos, queries,
            sum(len(grid) * len(grid[0]) for grid in all_grids),
        )
        if item is not None:
            candidates.append(item)

    add("identity", {}, lambda grid: grid)
    for reflected in (False, True):
        for turns in range(4):
            mode = f"{'reflect_' if reflected else ''}rot{turns}"
            add("dihedral", {"mode": mode}, lambda grid, m=mode: _d4(grid, m))
    for selector in ("all_foreground", "rarest", "most_common"):
        add("crop_bbox", {"selector": selector}, lambda grid, s=selector: _crop(grid, s))

    ratios = set()
    for source, expected in demos:
        sh, sw = _shape(source)
        eh, ew = _shape(expected)
        if eh % sh == 0 and ew % sw == 0:
            ratios.add((eh // sh, ew // sw))
    for row_factor, column_factor in sorted(ratios):
        if 1 <= row_factor <= 6 and 1 <= column_factor <= 6:
            add(
                "cell_scale",
                {"column_factor": column_factor, "row_factor": row_factor},
                lambda grid, rf=row_factor, cf=column_factor: _scale(grid, rf, cf),
            )
            for alternating_rows in (False, True):
                for alternating_columns in (False, True):
                    add(
                        "grid_tile",
                        {
                            "alternating_columns": alternating_columns,
                            "alternating_rows": alternating_rows,
                            "column_factor": column_factor,
                            "row_factor": row_factor,
                        },
                        lambda grid, rf=row_factor, cf=column_factor, ar=alternating_rows, ac=alternating_columns: _tile(grid, rf, cf, ar, ac),
                    )

    if all(_shape(source) == _shape(expected) for source, expected in demos):
        mapping: dict[int, int] = {}
        valid = True
        for source, expected in demos:
            for source_row, expected_row in zip(source, expected):
                for before, after in zip(source_row, expected_row):
                    if before in mapping and mapping[before] != after:
                        valid = False
                    mapping[before] = after
        if valid:
            add(
                "color_map",
                {"mapping": tuple(sorted(mapping.items()))},
                lambda grid, m=dict(mapping): tuple(tuple(m.get(cell, cell) for cell in row) for row in grid),
            )

    for connectivity in (4, 8):
        for rank in ("smallest", "largest"):
            for crop in (False, True):
                add(
                    "component_select",
                    {"connectivity": connectivity, "crop": crop, "rank": rank},
                    lambda grid, c=connectivity, r=rank, x=crop: _component_filter(grid, c, r, x),
                )

    colors = sorted(
        {
            cell
            for grid in (*inputs, *outputs)
            for row in grid
            for cell in row
        }
    )
    for color in colors:
        for mode in ("fill", "frame"):
            add(
                "foreground_bbox",
                {"color": color, "mode": mode},
                lambda grid, m=mode, c=color: _bbox(grid, m, c),
            )
        add(
            "fill_holes",
            {"color": color},
            lambda grid, c=color: _fill_holes(grid, c),
        )
        for diagonal in (False, True):
            add(
                "connect_pairs",
                {"diagonal": diagonal, "line_color": color},
                lambda grid, c=color, d=diagonal: _connect(grid, c, d),
            )
        for mode in ("or", "and", "xor", "nor"):
            add(
                "panel_boolean",
                {"mark": color, "mode": mode},
                lambda grid, m=mode, c=color: _panel_combine(grid, m, c),
            )

    reductions = set()
    for source, expected in demos:
        sh, sw = _shape(source)
        eh, ew = _shape(expected)
        if sh % eh == 0 and sw % ew == 0:
            reductions.add((sh // eh, sw // ew))
    for row_factor, column_factor in sorted(reductions):
        if 1 <= row_factor <= 10 and 1 <= column_factor <= 10:
            for mode in ("mode", "foreground"):
                add(
                    "block_reduce",
                    {
                        "column_factor": column_factor,
                        "mode": mode,
                        "row_factor": row_factor,
                    },
                    lambda grid, rf=row_factor, cf=column_factor, m=mode: _block_reduce(grid, rf, cf, m),
                )

    same_shape = all(
        _shape(source) == _shape(expected)
        for source, expected in demos
    )
    contracted = all(
        len(expected) <= len(source)
        and len(expected[0]) <= len(source[0])
        for source, expected in demos
    )
    expanded = all(
        len(expected) >= len(source)
        and len(expected[0]) >= len(source[0])
        for source, expected in demos
    )
    if expanded:
        for mode in ("right", "left", "below", "above", "four_outward", "four_inward"):
            add(
                "mirror_mosaic",
                {"mode": mode},
                lambda grid, m=mode: _mirror_mosaic(grid, m),
            )
    if (
        contracted
        and all(
            len(_palette(source)) >= 5
            and len(expected) < len(source)
            and len(expected[0]) < len(source[0])
            for source, expected in demos
        )
    ):
        add("frame_interior", {}, _frame_interior)
    if contracted:
        add("quadrant_unique", {}, _quadrant_unique)
        for rank in ("least_ink", "most_ink"):
            add(
                "panel_select",
                {"rank": rank},
                lambda grid, r=rank: _panel_select(grid, r),
            )
        add("partition_dominant", {}, _partition_dominant)
    if (
        same_shape
        and all(
            len(_palette(source)) >= 5
            and len(_palette(expected)) == 2
            for source, expected in demos
        )
    ):
        add("largest_solid_rectangle", {}, _largest_solid_rectangle)
    if all(
        _shape(expected) == (
            len(source) * len(source),
            len(source[0]) * len(source[0]),
        )
        for source, expected in demos
    ):
        for invert in (False, True):
            add(
                "self_kronecker",
                {"invert": invert},
                lambda grid, i=invert: _self_kronecker(grid, i),
            )
    if same_shape and all(len(source) == 2 for source in inputs):
        for reverse_phase in (False, True):
            add(
                "checkerboard_rows",
                {"reverse_phase": reverse_phase},
                lambda grid, r=reverse_phase: _checkerboard_rows(grid, r),
            )
    input_colors = sorted(
        {
            cell
            for grid in inputs
            for row in grid
            for cell in row
        }
    )
    added_colors = sorted(
        {
            cell
            for grid in outputs
            for row in grid
            for cell in row
        } - set(input_colors)
    )
    fill_colors = added_colors or colors
    if same_shape and len(added_colors) == 1 and len(input_colors) <= 3:
        for marker in input_colors:
            for fill in fill_colors:
                add(
                    "fill_corner_rectangles",
                    {"fill": fill, "marker": marker},
                    lambda grid, m=marker, f=fill: _fill_corner_rectangles(grid, m, f),
                )
                add(
                    "header_intersections",
                    {"fill": fill, "marker": marker},
                    lambda grid, m=marker, f=fill: _header_intersections(grid, m, f),
                )
    if all(
        len(expected) == len(source) // 2
        and len(expected[0]) == len(source[0]) // 2
        for source, expected in demos
    ):
        for odd_side in ("left", "right"):
            add(
                "parity_half_crop",
                {"odd_side": odd_side},
                lambda grid, s=odd_side: _parity_half_crop(grid, s),
            )

    unique: dict[tuple[str, tuple[tuple[str, object], ...]], Candidate] = {}
    for item in candidates:
        unique[(item.tool_id, item.binding)] = item
    return tuple(unique[key] for key in sorted(unique))


def resolve(
    demos: Sequence[tuple[Grid, Grid]],
    queries: Sequence[Grid],
    enabled_tools: Iterable[str],
    work_budget: int,
) -> Resolution:
    expected = tuple(output for _, output in demos)
    exact: list[Candidate] = []
    work = 0
    for candidate in enumerate_candidates(demos, queries, enabled_tools):
        work += candidate.work
        if work > work_budget:
            return Resolution("budget", None, len(exact), tuple(sorted({item.tool_id for item in exact})), work)
        if candidate.demo_outputs == expected:
            exact.append(candidate)
    if not exact:
        return Resolution("abstain_no_candidate", None, 0, (), work)
    regions: dict[tuple[Grid, ...], list[Candidate]] = defaultdict(list)
    for candidate in exact:
        regions[candidate.query_outputs].append(candidate)
    if len(regions) != 1:
        return Resolution(
            "abstain_ambiguous",
            None,
            len(exact),
            tuple(sorted({item.tool_id for item in exact})),
            work,
        )
    outputs = next(iter(regions))
    return Resolution(
        "emit",
        outputs,
        len(exact),
        tuple(sorted({item.tool_id for item in exact})),
        work,
    )


BASE_TOOL_SCHEMAS = {
    item.tool_id: item
    for item in (
        ToolSchema("identity", "grid", "grid", (), ("rectangular grid",), "preserve state", ("shape", "palette", "cells"), 1),
        ToolSchema("dihedral", "grid", "grid", ("mode",), ("rectangular grid",), "apply one D4 isometry", ("cell multiset",), 1),
        ToolSchema("crop_bbox", "grid", "grid", ("selector",), ("selected foreground exists",), "project selected material to its bounding box", ("selected cells",), 2),
        ToolSchema("cell_scale", "grid", "grid", ("row_factor", "column_factor"), ("positive integer factors",), "replace cells by rectangular blocks", ("cell adjacency modulo scale",), 2),
        ToolSchema("grid_tile", "grid", "grid", ("row_factor", "column_factor", "alternating_rows", "alternating_columns"), ("positive integer factors",), "tile a grid with optional alternating reflections", ("source cells",), 3),
        ToolSchema("color_map", "grid", "grid", ("mapping",), ("functional color correspondence on demonstrations",), "apply a pointwise color function", ("shape", "positions"), 2),
        ToolSchema("component_select", "grid", "grid", ("connectivity", "rank", "crop"), ("foreground component exists",), "select a component by relational size", ("selected component shape and colors",), 4),
        ToolSchema("foreground_bbox", "grid", "grid", ("mode", "color"), ("foreground exists",), "render the foreground bounding rectangle", ("outer grid shape",), 3),
        ToolSchema("fill_holes", "grid", "grid", ("color",), ("background topology is observable",), "color background regions not connected to the border", ("non-background cells",), 4),
        ToolSchema("connect_pairs", "grid", "grid", ("line_color", "diagonal"), ("two equal-color endpoints are aligned",), "connect aligned equal-color endpoint pairs", ("endpoints", "outer grid shape"), 5),
        ToolSchema("panel_boolean", "grid", "grid", ("mode", "mark"), ("two equal panels separated by a full line",), "combine panel occupancy masks", ("panel geometry",), 4),
        ToolSchema("block_reduce", "grid", "grid", ("row_factor", "column_factor", "mode"), ("input dimensions divide output dimensions",), "reduce regular blocks by a deterministic statistic", ("block partition",), 3),
        ToolSchema("mirror_mosaic", "grid", "grid", ("mode",), ("rectangular grid",), "compose reflected copies around a shared boundary", ("source cells",), 3),
        ToolSchema("frame_interior", "grid", "grid", (), ("complete monochrome rectangular frame exists",), "extract the interior state of the largest frame", ("interior cells",), 4),
        ToolSchema("largest_solid_rectangle", "grid", "grid", (), ("solid monochrome rectangle exists",), "retain the largest solid rectangle", ("selected rectangle", "outer shape"), 5),
        ToolSchema("quadrant_unique", "grid", "grid", (), ("four equal quadrants and exactly one unique quadrant",), "project the unique quadrant", ("selected quadrant",), 3),
        ToolSchema("panel_select", "grid", "grid", ("rank",), ("two equal panels separated by a full line",), "select a panel by relative ink", ("selected panel",), 3),
        ToolSchema("partition_dominant", "grid", "grid", (), ("regular panels separated by full lines",), "select the locally dominant signal and suppress distractors", ("selected panel geometry",), 5),
        ToolSchema("self_kronecker", "grid", "grid", ("invert",), ("binary occupancy is observable",), "render the occupancy Kronecker product", ("palette roles",), 5),
        ToolSchema("checkerboard_rows", "grid", "grid", ("reverse_phase",), ("two source rows exist",), "interleave two row colors by parity", ("shape", "row color roles"), 2),
        ToolSchema("fill_corner_rectangles", "grid", "grid", ("marker", "fill"), ("four equal markers define rectangle corners",), "fill the strict interiors of marker rectangles", ("markers", "outer shape"), 5),
        ToolSchema("header_intersections", "grid", "grid", ("marker", "fill"), ("top and right headers contain markers",), "render intersections of row and column headers", ("headers", "outer shape"), 4),
        ToolSchema("marker_bbox_recolor", "grid", "grid", ("marker", "target", "replacement"), ("marker bounding box exists",), "recolor target cells inside the marker frame", ("markers", "outer shape"), 4),
        ToolSchema("l_connect", "grid", "grid", ("first_color", "second_color", "line_color", "corner_mode"), ("one endpoint of each role exists",), "connect role-bound endpoints by an orthogonal path", ("endpoints", "outer shape"), 5),
        ToolSchema("parity_half_crop", "grid", "grid", ("odd_side",), ("grid can be halved",), "select a top half-panel by size parity", ("selected cells",), 2),
    )
}
