"""A2 frames/canvas algebra: candidati derivati dai soli demo.

Grafi:
  - frontier_count_canvas: canvas (righe complete +1) x (colonne complete
    +1) riempita col colore piu' frequente;
  - measured_property_canvas: su canvas, disegno fisso scelto da una
    proprieta' misurata dell'input (numcolors o colore non-background);
  - marker_frame: perimetro del bbox dei marker (inset 1) disegnato col
    colore raro interno;
  - backdrop_fill_mostcommon: bbox del colore-oggetto piu' frequente
    riempito col colore stesso;
  - marker_center_overlay: ogni componente 8-connessa contiene un solo
    marker; le celle sono traslate su una canvas col marker al centro;
  - corner_fold_canvas: bande vuote di righe/colonne separano quattro
    blocchi angolari, ripiegati sulla canvas ancorandoli ai quattro
    angoli (con eventuale sovrapposizione).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]
EXEC_ERRORS = (ValueError, IndexError, KeyError, StopIteration)


def _background(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    return min(counts, key=lambda value: (-counts[value], value))


def _components4(grid: Grid, background: int):
    height, width = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    items = []
    for r in range(height):
        for c in range(width):
            if grid[r][c] == background or (r, c) in seen:
                continue
            color = grid[r][c]
            stack = [(r, c)]
            cells = {(r, c)}
            seen.add((r, c))
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and (nr, nc) not in seen
                        and grid[nr][nc] == color
                    ):
                        cells.add((nr, nc))
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            items.append((color, frozenset(cells)))
    items.sort(key=lambda item: min(item[1]))
    return items


def _components8(grid: Grid, background: int):
    height, width = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    items = []
    for r in range(height):
        for c in range(width):
            if grid[r][c] == background or (r, c) in seen:
                continue
            stack = [(r, c)]
            cells = {(r, c)}
            seen.add((r, c))
            while stack:
                cr, cc = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < height
                            and 0 <= nc < width
                            and (nr, nc) not in seen
                            and grid[nr][nc] != background
                        ):
                            cells.add((nr, nc))
                            seen.add((nr, nc))
                            stack.append((nr, nc))
            items.append(frozenset(cells))
    items.sort(key=min)
    return items


def _property_value(grid: Grid, kind: str) -> int | None:
    if kind == "numcolors":
        return len({cell for row in grid for cell in row})
    background = _background(grid)
    others = sorted({
        cell for row in grid for cell in row if cell != background
    })
    if len(others) != 1:
        return None
    return others[0]


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]
    _cost: int = 1

    def cost(self, grid: Grid) -> int:
        return self._cost + len(grid) * len(grid[0])

    def execute(self, grid: Grid) -> Grid:
        values = dict(self.bindings)
        background = _background(grid)
        height, width = len(grid), len(grid[0])
        if self.graph == "frontier_count_canvas":
            counts = Counter(cell for row in grid for cell in row)
            most = min(counts, key=lambda v: (-counts[v], v))
            hlines = sum(
                1 for row in grid
                if len(set(row)) == 1 and row[0] != most
            )
            vlines = sum(
                1 for c in range(width)
                if len({grid[r][c] for r in range(height)}) == 1
                and grid[0][c] != most
            )
            return tuple(
                tuple(most for _ in range(vlines + 1))
                for _ in range(hlines + 1)
            )
        if self.graph == "measured_property_canvas":
            kind = values["property"]
            mark = values["mark"]
            blank = values["blank"]
            mapping = dict(values["mapping"])
            key = _property_value(grid, kind)
            if key is None or key not in mapping:
                raise ValueError("unknown property value")
            cells = mapping[key]
            shape = values["shape"]
            out = [[blank] * shape[1] for _ in range(shape[0])]
            for r, c in cells:
                if not (0 <= r < shape[0] and 0 <= c < shape[1]):
                    raise ValueError("pattern outside canvas")
                out[r][c] = mark
            return tuple(tuple(row) for row in out)
        if self.graph == "marker_frame":
            marker = values["marker"]
            points = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if grid[r][c] == marker
            ]
            if not points:
                raise ValueError("marker absent")
            top = min(r for r, _ in points) + 1
            bottom = max(r for r, _ in points) - 1
            left = min(c for _, c in points) + 1
            right = max(c for _, c in points) - 1
            if top > bottom or left > right:
                raise ValueError("no interior")
            inside = Counter(
                grid[r][c]
                for r in range(top, bottom + 1)
                for c in range(left, right + 1)
                if grid[r][c] not in (background, marker)
            )
            if not inside:
                raise ValueError("no interior color")
            frame_color = min(inside, key=lambda v: (inside[v], v))
            out = [list(row) for row in grid]
            for c in range(left, right + 1):
                out[top][c] = frame_color
                out[bottom][c] = frame_color
            for r in range(top, bottom + 1):
                out[r][left] = frame_color
                out[r][right] = frame_color
            return tuple(tuple(row) for row in out)
        if self.graph == "backdrop_fill_mostcommon":
            components = _components4(grid, background)
            if not components:
                raise ValueError("no components")
            color_counts = Counter()
            for color, cells in components:
                color_counts[color] += 1
            top_count = max(color_counts.values())
            chosen = min(
                color for color, n in color_counts.items() if n == top_count
            )
            cells = {
                (r, c)
                for r in range(height)
                for c in range(width)
                if grid[r][c] == chosen
            }
            rows = {r for r, _ in cells}
            cols = {c for _, c in cells}
            out = [list(row) for row in grid]
            for r in range(min(rows), max(rows) + 1):
                for c in range(min(cols), max(cols) + 1):
                    out[r][c] = chosen
            return tuple(tuple(row) for row in out)
        if self.graph == "column_snake_canvas":
            shape = values["shape"]
            items = sorted(
                (c, r, grid[r][c])
                for r in range(height)
                for c in range(width)
                if grid[r][c] != background
            )
            if len(items) > shape[0] * shape[1]:
                raise ValueError("too many points")
            canvas = [[background] * shape[1] for _ in range(shape[0])]
            index = 0
            for i in range(shape[0]):
                chunk = items[index:index + shape[1]]
                index += shape[1]
                if i % 2 == 0:
                    for j, (_, _, color) in enumerate(chunk):
                        canvas[i][j] = color
                else:
                    for j, (_, _, color) in enumerate(chunk):
                        canvas[i][shape[1] - 1 - j] = color
            return tuple(tuple(row) for row in canvas)
        if self.graph == "marker_center_overlay":
            marker = values["marker"]
            shape = values["shape"]
            center = ((shape[0] - 1) // 2, (shape[1] - 1) // 2)
            components = _components8(grid, background)
            if not components:
                raise ValueError("no components")
            canvas = [[background] * shape[1] for _ in range(shape[0])]
            for cells in components:
                marks = [(r, c) for r, c in cells if grid[r][c] == marker]
                if len(marks) != 1:
                    raise ValueError("component marker count != 1")
                mr, mc = marks[0]
                for r, c in cells:
                    tr = center[0] + r - mr
                    tc = center[1] + c - mc
                    if not (0 <= tr < shape[0] and 0 <= tc < shape[1]):
                        raise ValueError("translated cell outside canvas")
                    if canvas[tr][tc] not in (background, grid[r][c]):
                        raise ValueError("conflicting overlay")
                    canvas[tr][tc] = grid[r][c]
            return tuple(tuple(row) for row in canvas)
        if self.graph == "corner_fold_canvas":
            shape = values["shape"]
            blank_rows = [
                r for r in range(height)
                if all(cell == background for cell in grid[r])
            ]
            blank_cols = [
                c for c in range(width)
                if all(grid[r][c] == background for r in range(height))
            ]
            if not blank_rows or not blank_cols:
                raise ValueError("no blank bands")
            if (
                blank_rows != list(range(blank_rows[0], blank_rows[-1] + 1))
                or blank_cols != list(range(blank_cols[0], blank_cols[-1] + 1))
            ):
                raise ValueError("blank bands not contiguous")
            rb0, rb1 = blank_rows[0], blank_rows[-1] + 1
            cb0, cb1 = blank_cols[0], blank_cols[-1] + 1
            if rb0 == 0 or rb1 == height or cb0 == 0 or cb1 == width:
                raise ValueError("missing corner block")
            blocks = (
                (0, 0, range(0, rb0), range(0, cb0)),
                (0, shape[1] - (width - cb1), range(0, rb0), range(cb1, width)),
                (shape[0] - (height - rb1), 0, range(rb1, height), range(0, cb0)),
                (
                    shape[0] - (height - rb1),
                    shape[1] - (width - cb1),
                    range(rb1, height),
                    range(cb1, width),
                ),
            )
            canvas = [[background] * shape[1] for _ in range(shape[0])]
            for ar, ac, rows, cols in blocks:
                for r in rows:
                    for c in cols:
                        cell = grid[r][c]
                        if cell == background:
                            continue
                        tr, tc = ar + r - rows.start, ac + c - cols.start
                        if not (0 <= tr < shape[0] and 0 <= tc < shape[1]):
                            raise ValueError("block outside canvas")
                        if canvas[tr][tc] not in (background, cell):
                            raise ValueError("conflicting fold")
                        canvas[tr][tc] = cell
            return tuple(tuple(row) for row in canvas)
        raise ValueError("unknown a2 frames graph")


def candidates(demos: Demos) -> tuple[Candidate, ...]:
    result: dict[tuple[str, tuple], Candidate] = {}

    def add(candidate: Candidate) -> None:
        key = (candidate.graph, candidate.bindings)
        if key in result:
            return
        for source, expected in demos:
            try:
                if candidate.execute(source) != expected:
                    return
            except EXEC_ERRORS:
                return
        result[key] = candidate

    add(Candidate("frontier_count_canvas", ()))

    shapes = {(len(expected), len(expected[0])) for _, expected in demos}
    if len(shapes) == 1:
        shape = next(iter(shapes))
        for kind in ("numcolors", "other_color"):
            marks = set()
            blanks = set()
            mapping: dict[int, frozenset] = {}
            feasible = True
            for source, expected in demos:
                key = _property_value(source, kind)
                if key is None:
                    feasible = False
                    break
                nonblank_colors = {
                    cell for row in expected for cell in row
                }
                if len(nonblank_colors) != 2:
                    feasible = False
                    break
                counts = Counter(
                    cell for row in expected for cell in row)
                blank = min(counts, key=lambda v: (-counts[v], v))
                mark = next(
                    v for v in nonblank_colors if v != blank)
                cells = frozenset(
                    (r, c)
                    for r in range(len(expected))
                    for c in range(len(expected[0]))
                    if expected[r][c] == mark
                )
                if key in mapping and mapping[key] != cells:
                    feasible = False
                    break
                mapping[key] = cells
                marks.add(mark)
                blanks.add(blank)
            if feasible and mapping and len(marks) == 1 and len(blanks) == 1:
                add(Candidate(
                    "measured_property_canvas",
                    tuple(sorted({
                        "blank": next(iter(blanks)),
                        "mapping": tuple(sorted(
                            (k, tuple(sorted(v)))
                            for k, v in mapping.items())),
                        "mark": next(iter(marks)),
                        "property": kind,
                        "shape": shape,
                    }.items())),
                ))

    colors = sorted({
        value for source, _ in demos for row in source for value in row
    })
    for marker in colors:
        add(Candidate("marker_frame", (("marker", marker),)))
    add(Candidate("backdrop_fill_mostcommon", ()))

    shapes = {
        (len(expected), len(expected[0])) for _, expected in demos
    }
    for shape in sorted(shapes):
        add(Candidate("column_snake_canvas", (("shape", shape),)))
        add(Candidate("corner_fold_canvas", (("shape", shape),)))
        for marker in colors:
            add(Candidate(
                "marker_center_overlay",
                (("marker", marker), ("shape", shape)),
            ))

    return tuple(sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings))))
