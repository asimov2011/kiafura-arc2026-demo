"""A2 extract/subgrid algebra: candidati derivati dai soli demo.

Grafi:
  - adjacent_bbox_crop: crop del bbox della componente (8-connessa)
    adiacente alla componente marker;
  - single_component_upscale: crop del bbox dell'unica componente non
    background, scalata per un fattore intero derivato dai demo;
  - largest_recolor_subgrid: crop del bbox della componente massima fra
    quelle di colore diverso dal meno frequente, con quel colore
    ricolorato al meno frequente;
  - marker_vpad_crop: crop del bbox delle celle del colore marker,
    esteso di una riga sopra e sotto;
  - thin_ladder_bbox_crop: crop del bbox dell'unica componente senza
    2x2 pieni, senza righe da esattamente 2 celle e con almeno due
    righe piene o due colonne piene.
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


def _components8(grid: Grid, background: int):
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
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
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
            items.append(frozenset(cells))
    items.sort(key=lambda cells: min(cells))
    return items


def _objects_multicolor(grid: Grid, background: int):
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
    items.sort(key=lambda cells: min(cells))
    return items


def _bbox(cells) -> tuple[int, int, int, int]:
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: Grid, box: tuple[int, int, int, int]) -> Grid:
    top, left, bottom, right = box
    return tuple(
        tuple(grid[r][c] for c in range(left, right + 1))
        for r in range(top, bottom + 1)
    )


def _touching(a, b) -> bool:
    return any(
        abs(r - rr) <= 1 and abs(c - cc) <= 1 and (r, c) != (rr, cc)
        for r, c in a
        for rr, cc in b
    )


def _upscale(grid: Grid, factor: int) -> Grid:
    return tuple(
        tuple(cell for cell in row for _ in range(factor))
        for row in grid
        for _ in range(factor)
    )


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
        components = _components8(grid, background)
        if self.graph == "adjacent_bbox_crop":
            marker = values["marker"]
            markers = [
                cells for cells in components
                if grid[next(iter(cells))[0]][next(iter(cells))[1]] == marker
            ]
            if not markers:
                raise ValueError("marker absent")
            hits = [
                cells for cells in components
                if grid[next(iter(cells))[0]][next(iter(cells))[1]] != marker
                and any(_touching(cells, m) for m in markers)
            ]
            if len(hits) != 1:
                raise ValueError("ambiguous adjacent object")
            return _crop(grid, _bbox(hits[0]))
        if self.graph == "single_component_upscale":
            factor = values["factor"]
            if len(components) != 1:
                raise ValueError("not a single component")
            return _upscale(_crop(grid, _bbox(components[0])), factor)
        if self.graph == "largest_recolor_subgrid":
            counts = Counter(value for row in grid for value in row)
            rare = min(
                (value for value in counts if value != background),
                key=lambda value: (counts[value], value),
                default=None,
            )
            if rare is None:
                raise ValueError("no rare color")
            others = [
                cells for cells in components
                if grid[next(iter(cells))[0]][next(iter(cells))[1]] != rare
            ]
            if not others:
                raise ValueError("no candidate object")
            biggest = max(others, key=lambda cells: (len(cells), min(cells)))
            sub = _crop(grid, _bbox(biggest))
            big_color = grid[next(iter(biggest))[0]][next(iter(biggest))[1]]
            return tuple(
                tuple(rare if cell == big_color else cell for cell in row)
                for row in sub
            )
        if self.graph == "two_color_subgrid_replace":
            multi = []
            for cells in _objects_multicolor(grid, background):
                colors = {grid[r][c] for r, c in cells}
                if len(colors) == 2:
                    multi.append(cells)
            if len(multi) != 1:
                raise ValueError("no unique two-color object")
            sub = _crop(grid, _bbox(multi[0]))
            counts = Counter(
                cell for row in sub for cell in row if cell != background)
            if not counts:
                raise ValueError("empty object")
            majority = max(counts, key=lambda v: (counts[v], -v))
            return tuple(
                tuple(
                    majority if cell not in (majority, background) else cell
                    for cell in row
                )
                for row in sub
            )
        if self.graph == "bbox_fill_bg":
            fill = values["fill"]
            if len(components) != 1:
                raise ValueError("not a single component")
            top, left, bottom, right = _bbox(components[0])
            out = [list(row) for row in grid]
            for r in range(top, bottom + 1):
                for c in range(left, right + 1):
                    if grid[r][c] == background:
                        out[r][c] = fill
            return tuple(tuple(row) for row in out)
        if self.graph == "center_column_keep":
            middle = len(grid[0]) // 2
            return tuple(
                tuple(v if c == middle else background for c, v in enumerate(row))
                for row in grid
            )
        if self.graph == "dedup_rows_cols":
            rows = [grid[0]]
            for row in grid[1:]:
                if row != rows[-1]:
                    rows.append(row)
            height, width = len(rows), len(rows[0])
            keep = [0]
            for c in range(1, width):
                if any(rows[r][c] != rows[r][keep[-1]] for r in range(height)):
                    keep.append(c)
            return tuple(
                tuple(row[c] for c in keep) for row in rows)
        if self.graph == "band_colors_strip":
            stats: dict[int, list[int]] = {}
            for r, row in enumerate(grid):
                for c, value in enumerate(row):
                    entry = stats.get(value)
                    if entry is None:
                        stats[value] = [r, r, c, c, r, c, 1]
                    else:
                        entry[0] = min(entry[0], r)
                        entry[1] = max(entry[1], r)
                        entry[2] = min(entry[2], c)
                        entry[3] = max(entry[3], c)
                        entry[4] += r
                        entry[5] += c
                        entry[6] += 1
            spans_width = all(
                entry[2] == 0 and entry[3] == len(grid[0]) - 1
                for entry in stats.values()
            )
            any_spans_width = any(
                entry[2] == 0 and entry[3] == len(grid[0]) - 1
                for entry in stats.values()
            )
            spans_height = all(
                entry[0] == 0 and entry[1] == len(grid) - 1
                for entry in stats.values()
            )
            if spans_height and not spans_width:
                order = sorted(
                    stats, key=lambda value: (
                        stats[value][5] / stats[value][6], value))
                return (tuple(order),)
            if any_spans_width and not spans_height:
                order = sorted(
                    stats, key=lambda value: (
                        stats[value][4] / stats[value][6], value))
                return tuple((color,) for color in order)
            raise ValueError("band orientation ambiguous")
        if self.graph == "block_downscale_sample":
            points = [
                (r, c)
                for r, row in enumerate(grid)
                for c, value in enumerate(row)
                if value != background
            ]
            if not points:
                raise ValueError("empty block")
            top = min(r for r, _ in points)
            bottom = max(r for r, _ in points)
            left = min(c for _, c in points)
            right = max(c for _, c in points)
            block_h = bottom - top + 1
            block_w = right - left + 1
            slices = [grid[r][left:right + 1] for r in range(top, bottom + 1)]
            bands: list[tuple[int, int]] = []
            for index in range(block_h):
                if bands and slices[index] == slices[bands[-1][0]]:
                    bands[-1] = (bands[-1][0], index + 1)
                else:
                    bands.append((index, index + 1))
            out_h = len(bands)
            out_w = 0
            for start, _ in bands:
                runs = 1
                row = slices[start]
                for c in range(1, block_w):
                    if row[c] != row[c - 1]:
                        runs += 1
                out_w = max(out_w, runs)
            if out_h < 2 and out_w < 2:
                raise ValueError("nothing to downscale")
            return tuple(
                tuple(
                    grid[top + (2 * i + 1) * block_h // (2 * out_h)]
                    [left + (2 * j + 1) * block_w // (2 * out_w)]
                    for j in range(out_w)
                )
                for i in range(out_h)
            )
        if self.graph == "panel_hole_class_strip":
            mapping = dict(values["mapping"])
            counts = Counter(value for row in grid for value in row)
            sep = min(counts, key=lambda v: (counts[v], v))
            height, width = len(grid), len(grid[0])
            sep_cols = {
                c for c in range(width)
                if all(grid[r][c] == sep for r in range(height))
            }
            panels: list[tuple[int, int]] = []
            start = None
            for c in range(width):
                if c in sep_cols:
                    if start is not None:
                        panels.append((start, c))
                        start = None
                elif start is None:
                    start = c
            if start is not None:
                panels.append((start, width))
            if len(panels) < 2:
                raise ValueError("no panels")
            row: list[int] = []
            for c0, c1 in panels:
                holes = [
                    (r, c - c0)
                    for r in range(height)
                    for c in range(c0, c1)
                    if grid[r][c] == sep
                ]
                if not holes:
                    cls = "empty"
                else:
                    cols = {c for _, c in holes}
                    if 0 in cols:
                        cls = "left"
                    elif min(r for r, _ in holes) <= 1:
                        cls = "top"
                    else:
                        cls = "bottom"
                if cls not in mapping:
                    raise ValueError("unknown panel class")
                row.append(mapping[cls])
            return tuple((color,) * len(row) for color in row)
        if self.graph == "count_desc_histogram":
            blank = values["blank"]
            counts = Counter(value for row in grid for value in row)
            order = sorted(counts, key=lambda v: (-counts[v], v))
            height_out = counts[order[0]]
            return tuple(
                tuple(
                    color if r < counts[color] else blank
                    for color in order
                )
                for r in range(height_out)
            )
        if self.graph == "row_pattern_infill":
            hole = values["hole"]
            height, width = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            painted = 0
            for r in range(height):
                missing = [c for c in range(width) if grid[r][c] == hole]
                if not missing:
                    continue
                visible = [
                    c for c in range(width) if grid[r][c] != hole
                ]
                donor = None
                for rr in range(height):
                    if rr == r or any(
                        grid[rr][c] == hole for c in range(width)
                    ):
                        continue
                    if all(grid[rr][c] == grid[r][c] for c in visible):
                        donor = rr
                        break
                if donor is None:
                    raise ValueError("no matching donor row")
                for c in missing:
                    out[r][c] = grid[donor][c]
                    painted += 1
            if not painted:
                raise ValueError("nothing to infill")
            rows = {r for r in range(height) if hole in grid[r]}
            cols = {
                c for r in rows for c in range(width) if grid[r][c] == hole
            }
            top, bottom = min(rows), max(rows)
            left, right = min(cols), max(cols)
            return tuple(
                tuple(out[r][c] for c in range(left, right + 1))
                for r in range(top, bottom + 1)
            )
        if self.graph == "marker_vpad_crop":
            marker = values["marker"]
            height, width = len(grid), len(grid[0])
            cells = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if grid[r][c] == marker
            ]
            if not cells:
                raise ValueError("marker absent")
            top = min(r for r, _ in cells) - 1
            bottom = max(r for r, _ in cells) + 1
            left = min(c for _, c in cells)
            right = max(c for _, c in cells)
            if top < 0 or bottom >= height:
                raise ValueError("padded crop outside grid")
            return tuple(
                tuple(grid[r][c] for c in range(left, right + 1))
                for r in range(top, bottom + 1)
            )
        if self.graph == "corner_quad_decode":
            height, width = len(grid), len(grid[0])
            colors = sorted({value for row in grid for value in row})
            solutions = []
            for grid_color in colors:
                line_rows = [
                    r for r in range(height)
                    if sum(v == grid_color for v in grid[r]) * 2 > width
                ]
                line_cols = [
                    c for c in range(width)
                    if sum(grid[r][c] == grid_color for r in range(height)) * 2
                    > height
                ]
                if len(line_rows) < 2 or len(line_cols) < 2:
                    continue
                interior = Counter(
                    grid[r][c]
                    for r in range(height) if r not in line_rows
                    for c in range(width) if c not in line_cols
                )
                if not interior:
                    continue
                background = interior.most_common(1)[0][0]
                if background == grid_color:
                    continue
                marks = {
                    (r, c): grid[r][c]
                    for r in range(height)
                    for c in range(width)
                    if grid[r][c] not in (background, grid_color)
                }
                if not marks:
                    continue
                row_index = {r: i for i, r in enumerate(line_rows)}
                col_index = {c: j for j, c in enumerate(line_cols)}
                if any(
                    r not in row_index or c not in col_index
                    for r, c in marks
                ):
                    continue
                points = {
                    (row_index[r], col_index[c]): value
                    for (r, c), value in marks.items()
                }
                top = min(a for a, _ in points)
                bottom = max(a for a, _ in points)
                left = min(b for _, b in points)
                right = max(b for _, b in points)
                if bottom == top or right == left:
                    continue
                out = []
                painted = False
                for i in range(top, bottom):
                    row = []
                    for j in range(left, right):
                        corners = [
                            points.get((i, j)), points.get((i, j + 1)),
                            points.get((i + 1, j)),
                            points.get((i + 1, j + 1)),
                        ]
                        if all(x is not None for x in corners) \
                                and len(set(corners)) == 1:
                            row.append(corners[0])
                            painted = True
                        else:
                            row.append(background)
                    out.append(tuple(row))
                if painted:
                    solutions.append(tuple(out))
            if len(solutions) != 1:
                raise ValueError("corner decode ambiguous")
            return solutions[0]
        if self.graph == "thin_ladder_bbox_crop":
            height, width = len(grid), len(grid[0])
            seen: set[tuple[int, int]] = set()
            comps = []
            for r in range(height):
                for c in range(width):
                    if grid[r][c] == background or (r, c) in seen:
                        continue
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
                                and grid[nr][nc] != background
                            ):
                                cells.add((nr, nc))
                                seen.add((nr, nc))
                                stack.append((nr, nc))
                    comps.append(frozenset(cells))
            selected = []
            for cells in comps:
                r0, c0, r1, c1 = _bbox(cells)
                span_h, span_w = r1 - r0 + 1, c1 - c0 + 1
                if any(
                    (r, c) in cells and (r + 1, c) in cells
                    and (r, c + 1) in cells and (r + 1, c + 1) in cells
                    for r, c in cells
                ):
                    continue
                rows = [
                    sum(1 for cc in range(c0, c1 + 1) if (rr, cc) in cells)
                    for rr in range(r0, r1 + 1)
                ]
                cols = [
                    sum(1 for rr in range(r0, r1 + 1) if (rr, cc) in cells)
                    for cc in range(c0, c1 + 1)
                ]
                if any(count == 2 for count in rows):
                    continue
                full_rows = sum(1 for count in rows if count == span_w)
                full_cols = sum(1 for count in cols if count == span_h)
                if full_rows >= 2 or full_cols >= 2:
                    selected.append((r0, c0, r1, c1))
            if len(selected) != 1:
                raise ValueError("selection ambiguous")
            return _crop(grid, selected[0])
        raise ValueError("unknown a2 extract graph")


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

    colors = sorted({
        value for source, _ in demos for row in source for value in row
    })
    for marker in colors:
        add(Candidate("adjacent_bbox_crop", (("marker", marker),)))

    for factor in (1, 2, 3):
        add(Candidate("single_component_upscale", (("factor", factor),)))
    add(Candidate("largest_recolor_subgrid", ()))
    add(Candidate("two_color_subgrid_replace", ()))
    out_colors = sorted({
        value for _, expected in demos for row in expected for value in row
    })
    for fill in out_colors:
        add(Candidate("bbox_fill_bg", (("fill", fill),)))

    add(Candidate("center_column_keep", ()))
    for hole in colors:
        add(Candidate("row_pattern_infill", (("hole", hole),)))
    for marker in colors:
        add(Candidate("marker_vpad_crop", (("marker", marker),)))

    add(Candidate("dedup_rows_cols", ()))
    add(Candidate("band_colors_strip", ()))
    add(Candidate("block_downscale_sample", ()))
    add(Candidate("corner_quad_decode", ()))
    add(Candidate("thin_ladder_bbox_crop", ()))

    for blank in out_colors:
        add(Candidate("count_desc_histogram", (("blank", blank),)))

    strip_mapping, strip_feasible = {}, True
    for source, expected in demos:
        counts = Counter(value for row in source for value in row)
        sep = min(counts, key=lambda v: (counts[v], v))
        height, width = len(source), len(source[0])
        sep_cols = {
            c for c in range(width)
            if all(source[r][c] == sep for r in range(height))
        }
        panels: list[tuple[int, int]] = []
        start = None
        for c in range(width):
            if c in sep_cols:
                if start is not None:
                    panels.append((start, c))
                    start = None
            elif start is None:
                start = c
        if start is not None:
            panels.append((start, width))
        if (
            len(panels) < 2
            or len(expected) != len(panels)
            or len(expected[0]) != len(panels)
        ):
            strip_feasible = False
            break
        for index, (c0, c1) in enumerate(panels):
            row_values = set(expected[index])
            if len(row_values) != 1:
                strip_feasible = False
                break
            color = next(iter(row_values))
            holes = [
                (r, c - c0)
                for r in range(height)
                for c in range(c0, c1)
                if source[r][c] == sep
            ]
            if not holes:
                cls = "empty"
            else:
                cols = {c for _, c in holes}
                if 0 in cols:
                    cls = "left"
                elif min(r for r, _ in holes) <= 1:
                    cls = "top"
                else:
                    cls = "bottom"
            if strip_mapping.get(cls, color) != color:
                strip_feasible = False
                break
            strip_mapping[cls] = color
        if not strip_feasible:
            break
    if strip_feasible and strip_mapping:
        add(Candidate(
            "panel_hole_class_strip",
            (("mapping", tuple(sorted(strip_mapping.items()))),),
        ))

    return tuple(sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings))))
