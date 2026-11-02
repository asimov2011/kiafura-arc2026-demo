"""Open count/measure/render algebra derived from A-only experiences."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence


Grid = tuple[tuple[int, ...], ...]
LabeledCounts = tuple[tuple[int, int], ...]


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    return tuple(tuple(int(cell) for cell in row) for row in value)


def components(grid: Grid, color: int, connectivity: int) -> tuple[frozenset[tuple[int, int]], ...]:
    pending = {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == color}
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        offsets = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
    output = []
    while pending:
        seed = min(pending)
        pending.remove(seed)
        item = {seed}
        queue = deque([seed])
        while queue:
            r, c = queue.popleft()
            for dr, dc in offsets:
                point = (r + dr, c + dc)
                if point in pending:
                    pending.remove(point)
                    item.add(point)
                    queue.append(point)
        output.append(frozenset(item))
    return tuple(output)


def solid_square_count(grid: Grid, color: int, side: int, connectivity: int) -> LabeledCounts:
    count = 0
    for item in components(grid, color, connectivity):
        rows = {r for r, _ in item}
        columns = {c for _, c in item}
        if len(item) == side * side and len(rows) == len(columns) == side:
            count += 1
    return ((color, count),)


def component_counts(grid: Grid, colors: Sequence[int], connectivity: int) -> LabeledCounts:
    return tuple((color, len(components(grid, color, connectivity))) for color in colors)


def bbox_marker_counts(grid: Grid, marker: int, background: int) -> LabeledCounts:
    colors = sorted({value for row in grid for value in row} - {marker, background})
    output = []
    for color in colors:
        points = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == color]
        if not points:
            continue
        top, bottom = min(r for r, _ in points), max(r for r, _ in points)
        left, right = min(c for _, c in points), max(c for _, c in points)
        count = sum(grid[r][c] == marker for r in range(top, bottom + 1) for c in range(left, right + 1))
        output.append((color, count))
    return tuple(output)


def render_fixed_row(counts: LabeledCounts, label: int, background: int, width: int) -> Grid:
    count = counts[0][1]
    if count > width:
        raise ValueError("count exceeds visible output width")
    return (tuple([label] * count + [background] * (width - count)),)


def render_histogram(counts: LabeledCounts, background: int) -> Grid:
    height = max((count for _, count in counts), default=0)
    return tuple(
        tuple(color if row >= height - count else background for color, count in counts)
        for row in range(height)
    )


def render_sorted_bars(counts: LabeledCounts, background: int) -> Grid:
    ordered = sorted((count, color) for color, count in counts)
    width = max((count for count, _ in ordered), default=0)
    return tuple(tuple([color] * count + [background] * (width - count)) for count, color in ordered)


def extend_band_by_side_counts(grid: Grid, band: int, background: int) -> Grid:
    height, width = len(grid), len(grid[0])
    points = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == band]
    if not points:
        raise ValueError("band absent")
    top, bottom = min(r for r, _ in points), max(r for r, _ in points)
    left, right = min(c for _, c in points), max(c for _, c in points)
    output = [[band if value == band else background for value in row] for row in grid]
    if top == 0 and bottom == height - 1:
        for r in range(height):
            before = sum(grid[r][c] not in (background, band) for c in range(left))
            after = sum(grid[r][c] not in (background, band) for c in range(right + 1, width))
            for c in range(max(0, left - before), left): output[r][c] = band
            for c in range(right + 1, min(width, right + 1 + after)): output[r][c] = band
    elif left == 0 and right == width - 1:
        for c in range(width):
            before = sum(grid[r][c] not in (background, band) for r in range(top))
            after = sum(grid[r][c] not in (background, band) for r in range(bottom + 1, height))
            for r in range(max(0, top - before), top): output[r][c] = band
            for r in range(bottom + 1, min(height, bottom + 1 + after)): output[r][c] = band
    else:
        raise ValueError("band does not span one visible axis")
    return as_grid(output)


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]

    def execute(self, grid: Grid) -> Grid:
        values = dict(self.bindings)
        if self.graph == "solid_square_to_row":
            counts = solid_square_count(grid, values["source"], values["side"], values["connectivity"])
            return render_fixed_row(counts, values["label"], values["background"], values["width"])
        if self.graph == "components_to_histogram":
            return render_histogram(component_counts(grid, values["order"], values["connectivity"]), values["background"])
        if self.graph == "bbox_markers_to_bars":
            return render_sorted_bars(bbox_marker_counts(grid, values["marker"], values["input_background"]), values["output_background"])
        if self.graph == "band_side_counts":
            return extend_band_by_side_counts(grid, values["band"], values["background"])
        raise ValueError("unknown count-render graph")


def _palette(grids: Sequence[Grid]) -> tuple[int, ...]:
    return tuple(sorted({value for grid in grids for row in grid for value in row}))


def _background(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    return min(counts, key=lambda value: (-counts[value], value))


def candidates(demos: Sequence[tuple[Grid, Grid]]) -> tuple[Candidate, ...]:
    inputs = tuple(source for source, _ in demos)
    outputs = tuple(expected for _, expected in demos)
    colors = _palette((*inputs, *outputs))
    result: set[Candidate] = set()
    output_backgrounds = tuple(sorted({_background(grid) for grid in outputs}))

    if len({len(grid) for grid in outputs}) == 1 and next(iter({len(grid) for grid in outputs})) == 1:
        widths = {len(grid[0]) for grid in outputs}
        if len(widths) == 1:
            width = next(iter(widths))
            for source, side, connectivity, label, background in product(colors, range(1, 5), (4, 8), colors, output_backgrounds):
                result.add(Candidate("solid_square_to_row", tuple(sorted({"background": background, "connectivity": connectivity, "label": label, "side": side, "source": source, "width": width}.items()))))

    orders = []
    for expected in outputs:
        background = _background(expected)
        order = []
        for c in range(len(expected[0])):
            values = [expected[r][c] for r in range(len(expected)) if expected[r][c] != background]
            if not values or len(set(values)) != 1: break
            order.append(values[0])
        if len(order) == len(expected[0]): orders.append((tuple(order), background))
    if orders and len(set(orders)) == 1:
        order, background = orders[0]
        for connectivity in (4, 8):
            result.add(Candidate("components_to_histogram", tuple(sorted({"background": background, "connectivity": connectivity, "order": order}.items()))))

    for marker, input_background, output_background in product(colors, colors, output_backgrounds):
        result.add(Candidate("bbox_markers_to_bars", tuple(sorted({"input_background": input_background, "marker": marker, "output_background": output_background}.items()))))
    for band, background in product(colors, colors):
        if band != background:
            result.add(Candidate("band_side_counts", tuple(sorted({"background": background, "band": band}.items()))))
    return tuple(sorted(result, key=lambda item: (item.graph, repr(item.bindings))))


def solve(demos: Sequence[tuple[Grid, Grid]], queries: Sequence[Grid], *, budget: int = 2_000_000, allowed_graphs: Sequence[str] | None = None) -> dict[str, Any]:
    allowed = set(allowed_graphs) if allowed_graphs is not None else None
    pool = tuple(candidate for candidate in candidates(demos) if allowed is None or candidate.graph in allowed)
    precharge = sum((len(source) * len(source[0]) + len(expected) * len(expected[0])) for source, expected in demos) * max(1, len(pool))
    if precharge > budget:
        return {"status": "abstain_budget", "outputs": None, "work": precharge}
    exact = []
    for candidate in pool:
        try:
            if all(candidate.execute(source) == expected for source, expected in demos): exact.append(candidate)
        except (ValueError, IndexError, KeyError):
            continue
    regions: dict[tuple[Grid, ...], list[Candidate]] = {}
    for candidate in exact:
        outputs = tuple(candidate.execute(query) for query in queries)
        regions.setdefault(outputs, []).append(candidate)
    if len(regions) != 1:
        return {"status": "abstain_no_consensus", "outputs": None, "work": precharge, "regions": len(regions)}
    outputs, members = next(iter(regions.items()))
    return {"status": "verified", "outputs": outputs, "work": precharge, "regions": 1, "graphs": sorted({item.graph for item in members})}
