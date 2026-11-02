"""A1 gruppo 2: algebre parametriche con candidati derivati dai soli demo.

Grafi (classe di trasformazione; binding derivati dai demo, filtro
demo-esatto, fail-closed):
  - row_markers_fill: due celle marcatore sulla stessa riga; le celle
    tra i marker sono riempite: metà sinistra col colore del marker
    sinistro, metà destra col colore del marker destro, cella centrale
    col colore nuovo derivato dai demo;
  - bbox_crop_horiz_tile: ritaglio del bbox delle celle non sfondo,
    replicato orizzontalmente k volte (k derivato dai demo).
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


def _row_markers_fill(grid: Grid, middle: int) -> Grid:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    out = [list(row) for row in grid]
    filled = 0
    for r in range(height):
        markers = [
            (c, grid[r][c])
            for c in range(width)
            if grid[r][c] != background
        ]
        if not markers:
            continue
        if len(markers) != 2:
            raise ValueError("row needs exactly two markers")
        (ca, va), (cb, vb) = sorted(markers)
        if (ca + cb) % 2:
            raise ValueError("no integer center")
        center = (ca + cb) // 2
        for c in range(ca + 1, center):
            out[r][c] = va
        out[r][center] = middle
        for c in range(center + 1, cb):
            out[r][c] = vb
        filled += 1
    if filled < 1:
        raise ValueError("no marker rows")
    return tuple(tuple(row) for row in out)


def _bbox_crop_horiz_tile(grid: Grid, k: int) -> Grid:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] != background
    ]
    if not cells:
        raise ValueError("no cells")
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    if k < 1:
        raise ValueError("bad tile count")
    crop = tuple(
        tuple(grid[r][c] for c in range(c0, c1 + 1))
        for r in range(r0, r1 + 1)
    )
    return tuple(tuple(row * k) for row in crop)


def _template_replace_components(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    counts = Counter(v for row in grid for v in row if v != background)
    if not counts:
        raise ValueError("no object cells")
    obj = max(counts, key=lambda v: (counts[v], -v))
    template_cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] not in (background, obj)
    ]
    if not template_cells:
        raise ValueError("no template cells")
    t0 = min(r for r, _ in template_cells)
    t1 = max(r for r, _ in template_cells)
    template_rows = []
    for r in range(t0, t1 + 1):
        row_cells = [
            grid[r][c]
            for c in range(width)
            if grid[r][c] not in (background, obj)
        ]
        if not row_cells:
            raise ValueError("empty template row")
        template_rows.append(tuple(row_cells))
    if not template_rows:
        raise ValueError("empty template")
    seen = [[False] * width for _ in range(height)]
    out = [[background] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            if grid[r][c] != obj or seen[r][c]:
                continue
            seen[r][c] = True
            stack = [(r, c)]
            comp = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width \
                            and not seen[nr][nc] and grid[nr][nc] == obj:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
                        comp.append((nr, nc))
            rows = {}
            for cr, cc in comp:
                rows.setdefault(cr, []).append(cc)
            for index, row in enumerate(sorted(rows)):
                if index >= len(template_rows):
                    raise ValueError("component taller than template")
                cells = sorted(rows[row])
                if len(cells) != len(template_rows[index]):
                    raise ValueError("component row width mismatch")
                for cc, value in zip(cells, template_rows[index]):
                    out[row][cc] = value
    return tuple(tuple(row) for row in out)


def _no_h_neighbor_recolor(grid: Grid, obj: int, new_value: int) -> Grid:
    height, width = len(grid), len(grid[0])
    out = [list(row) for row in grid]
    for r in range(height):
        for c in range(width):
            if grid[r][c] != obj:
                continue
            if any(
                0 <= cc < width and grid[r][cc] == obj
                for cc in (c - 1, c + 1)
            ):
                continue
            out[r][c] = new_value
    return tuple(tuple(row) for row in out)


def _vertical_endpoint_marks(grid: Grid, obj: int, mark: int) -> Grid:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    out = [list(row) for row in grid]
    seen = [[False] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            if grid[r][c] != obj or seen[r][c]:
                continue
            if r - 1 >= 0 and grid[r - 1][c] == obj:
                continue
            seen[r][c] = True
            run = [(r, c)]
            rr = r + 1
            while rr < height and grid[rr][c] == obj:
                seen[rr][c] = True
                run.append((rr, c))
                rr += 1
            top_r, top_c = run[0]
            bottom_r, bottom_c = run[-1]

            def mark_at(row, col):
                if 0 <= col < width and out[row][col] == background:
                    out[row][col] = mark
                    return True
                return False

            def side_mark(row, col, side):
                if side < 0 and mark_at(row, col - 1):
                    return
                if side < 0 and mark_at(row, col + 1):
                    return
                if side > 0 and mark_at(row, col + 1):
                    return
                if side > 0 and mark_at(row, col - 1):
                    return
                raise ValueError("no room for mark")

            def h_side(row, col):
                if col - 1 >= 0 and grid[row][col - 1] == obj:
                    return -1
                if col + 1 < width and grid[row][col + 1] == obj:
                    return 1
                return 0

            in_horizontal = lambda row, col: h_side(row, col) != 0
            if len(run) == 1:
                if not in_horizontal(top_r, top_c):
                    side_mark(top_r, top_c, -1)
                continue
            if not in_horizontal(top_r, top_c):
                side_mark(top_r, top_c, h_side(bottom_r, bottom_c) or -1)
            if not in_horizontal(bottom_r, bottom_c):
                side_mark(bottom_r, bottom_c, h_side(top_r, top_c) or -1)
    return tuple(tuple(row) for row in out)


def _marker_zigzag_columns(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    markers = [
        (c, grid[0][c])
        for c in range(width)
        if grid[0][c] != 0
    ]
    if not markers or any(
        grid[r][c] != 0
        for r in range(1, height)
        for c in range(width)
    ):
        raise ValueError("markers must sit alone on the first row")
    out = [[0] * width for _ in range(height)]
    for r in range(height):
        for c, value in markers:
            if r % 2 == 0:
                out[r][c] = value
            else:
                for cc in (c - 1, c + 1):
                    if 0 <= cc < width:
                        out[r][cc] = value
    return tuple(tuple(row) for row in out)


def _frame_corner_crop_recolor(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    marker = None
    marker_cells = []
    for value in sorted(
        {cell for row in grid for cell in row} - {background}
    ):
        cells = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if grid[r][c] == value
        ]
        if len(cells) == 4:
            marker = value
            marker_cells = cells
            break
    if marker is None:
        raise ValueError("no size-4 marker")
    others = [
        v for v in {cell for row in grid for cell in row}
        - {background, marker}
    ]
    if len(others) != 1:
        raise ValueError("object color ambiguous")
    obj = others[0]
    r0 = min(r for r, _ in marker_cells)
    r1 = max(r for r, _ in marker_cells)
    c0 = min(c for _, c in marker_cells)
    c1 = max(c for _, c in marker_cells)
    return tuple(
        tuple(
            marker if grid[r][c] == obj else background
            for c in range(c0 + 1, c1)
        )
        for r in range(r0 + 1, r1)
    )


def _top_right_crop(grid: Grid, out_h: int, out_w: int) -> Grid:
    height, width = len(grid), len(grid[0])
    if out_h > height or out_w > width:
        raise ValueError("crop larger than grid")
    return tuple(
        tuple(row[width - out_w:])
        for row in grid[:out_h]
    )


def _top_band_bottom_mirror(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    band = []
    for r in range(height):
        if any(cell != background for cell in grid[r]):
            band.append(grid[r])
        else:
            break
    if not band or len(band) >= height:
        raise ValueError("no top band")
    out = [list(row) for row in grid]
    for i, row in enumerate(reversed(band)):
        out[height - len(band) + i] = list(row)
    return tuple(tuple(row) for row in out)


def _object_outbox_paint(grid: Grid, obj: int, ring_value: int,
                         hole_value: int) -> Grid:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    seen = [[False] * width for _ in range(height)]
    out = [list(row) for row in grid]
    for r in range(height):
        for c in range(width):
            if grid[r][c] != obj or seen[r][c]:
                continue
            seen[r][c] = True
            stack = [(r, c)]
            comp = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width \
                            and not seen[nr][nc] and grid[nr][nc] == obj:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
                        comp.append((nr, nc))
            r0 = min(rr for rr, _ in comp)
            r1 = max(rr for rr, _ in comp)
            c0 = min(cc for _, cc in comp)
            c1 = max(cc for _, cc in comp)
            ring = set()
            for cc in range(c0 - 1, c1 + 2):
                for rr in (r0 - 1, r1 + 1):
                    if 0 <= rr < height and 0 <= cc < width:
                        ring.add((rr, cc))
            for rr in range(r0 - 1, r1 + 2):
                for cc in (c0 - 1, c1 + 1):
                    if 0 <= rr < height and 0 <= cc < width:
                        ring.add((rr, cc))
            for rr, cc in ring:
                if grid[rr][cc] != background and grid[rr][cc] != ring_value:
                    raise ValueError("ring conflict")
                out[rr][cc] = ring_value
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if grid[rr][cc] == background:
                        out[rr][cc] = hole_value
    return tuple(tuple(row) for row in out)


def _bottom_center_markers(grid: Grid, mark: int) -> Grid:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    out = [list(row) for row in grid]
    seen = [[False] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            if grid[r][c] == background or seen[r][c]:
                continue
            seen[r][c] = True
            stack = [(r, c)]
            comp = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width \
                            and not seen[nr][nc] \
                            and grid[nr][nc] == grid[r][c]:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
                        comp.append((nr, nc))
            cols = {cc for _, cc in comp}
            c0 = min(cols)
            c1 = max(cols)
            center_col = c0 + (c1 - c0 + 1) // 2
            if out[height - 1][center_col] != background \
                    and out[height - 1][center_col] != mark:
                raise ValueError("marker collision")
            out[height - 1][center_col] = mark
    return tuple(tuple(row) for row in out)


def _seed_neighbor_paint(grid: Grid, neighbor_color: int) -> Grid:
    height, width = len(grid), len(grid[0])
    counts = Counter(cell for row in grid for cell in row if cell != 0)
    if not counts:
        raise ValueError("no cells")
    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) < 2 or ordered[0][1] == ordered[1][1] \
            or ordered[0][1] != 1:
        raise ValueError("no unique least color")
    seed_color = ordered[0][0]
    seeds = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == seed_color
    ]
    if len(seeds) != 1:
        raise ValueError("need exactly one seed")
    r, c = seeds[0]
    out = [[0] * width for _ in range(height)]
    out[r][c] = seed_color
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                out[nr][nc] = neighbor_color
    return tuple(tuple(row) for row in out)


def _max_colorcount_component_crop(grid: Grid, target: int) -> Grid:
    height, width = len(grid), len(grid[0])
    background = 0
    seen = [[False] * width for _ in range(height)]
    best = None
    for r in range(height):
        for c in range(width):
            if grid[r][c] == background or seen[r][c]:
                continue
            seen[r][c] = True
            stack = [(r, c)]
            comp = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width \
                            and not seen[nr][nc] \
                            and grid[nr][nc] != background:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
                        comp.append((nr, nc))
            count = sum(1 for rr, cc in comp if grid[rr][cc] == target)
            if best is not None and count == best[0]:
                raise ValueError("tie on target count")
            if best is None or count > best[0]:
                best = (count, comp)
    if best is None or best[0] < 1:
        raise ValueError("no target cells")
    comp = best[1]
    r0 = min(rr for rr, _ in comp)
    r1 = max(rr for rr, _ in comp)
    c0 = min(cc for _, cc in comp)
    c1 = max(cc for _, cc in comp)
    return tuple(
        tuple(grid[rr][cc] for cc in range(c0, c1 + 1))
        for rr in range(r0, r1 + 1)
    )


def _quadrant_overlay_top_left(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    if height % 2 or width % 2:
        raise ValueError("need even dimensions")
    h, w = height // 2, width // 2
    tl = [[grid[r][c] for c in range(w)] for r in range(h)]
    tr = [[grid[r][w + c] for c in range(w)] for r in range(h)]
    bl = [[grid[h + r][c] for c in range(w)] for r in range(h)]
    br = [[grid[h + r][w + c] for c in range(w)] for r in range(h)]
    for quad in (br, bl, tr):
        for r in range(h):
            for c in range(w):
                if quad[r][c] != 0:
                    tl[r][c] = quad[r][c]
    return tuple(tuple(row) for row in tl)

def _corner_diagonal_rays(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    counts = Counter(cell for row in grid for cell in row)
    least = counts.most_common()[-1][0]
    cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == least
    ]
    if not cells:
        raise ValueError("no least-color cells")
    shifted = {(r - 1, c) for r, c in cells}
    ul = (min(r for r, _ in shifted), min(c for _, c in shifted))
    ur = (min(r for r, _ in shifted), max(c for _, c in shifted))
    rays = set()
    for (r, c), dc in ((ul, -1), (ur, 1)):
        k = 0
        while True:
            rr, cc = r - k, c + k * dc
            if 0 <= rr < height and 0 <= cc < width:
                rays.add((rr, cc))
            else:
                break
            k += 1
    out = [list(row) for row in grid]
    for r, c in rays:
        if out[r][c] == background:
            out[r][c] = least
    return tuple(tuple(row) for row in out)

def _partition_backdrop_fill(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    out = [list(row) for row in grid]
    claimed: dict[tuple[int, int], int] = {}
    for value in sorted(
        {cell for row in grid for cell in row} - {background}
    ):
        cells = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if grid[r][c] == value
        ]
        if not cells:
            continue
        r0 = min(r for r, _ in cells)
        r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        c1 = max(c for _, c in cells)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) in claimed and claimed[(r, c)] != value:
                    raise ValueError("backdrop overlap")
                claimed[(r, c)] = value
                out[r][c] = value
    return tuple(tuple(row) for row in out)


def _max_zero_rect_fill(grid: Grid, fill: int) -> Grid:
    height, width = len(grid), len(grid[0])
    background = 0
    best = None
    best_cells = None
    for r0 in range(height):
        for c0 in range(width):
            if grid[r0][c0] != background:
                continue
            for r1 in range(r0, height):
                for c1 in range(c0, width):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if best is not None and area < best[0]:
                        continue
                    if all(
                        grid[r][c] == background
                        for r in range(r0, r1 + 1)
                        for c in range(c0, c1 + 1)
                    ):
                        cells = tuple(
                            (r, c)
                            for r in range(r0, r1 + 1)
                            for c in range(c0, c1 + 1)
                        )
                        if best is not None and area == best[0]:
                            if cells < best_cells:
                                best_cells = cells
                                best = (area, r0, r1, c0, c1)
                            continue
                        best = (area, r0, r1, c0, c1)
                        best_cells = cells
    if best is None:
        raise ValueError("no zero rect")
    _, r0, r1, c0, c1 = best
    out = [list(row) for row in grid]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r][c] = fill
    return tuple(tuple(row) for row in out)


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]
    _cost: int = 1

    def cost(self, grid: Grid) -> int:
        return self._cost + len(grid) * len(grid[0])

    def execute(self, grid: Grid) -> Grid:
        if not grid or not grid[0] or any(
            len(row) != len(grid[0]) for row in grid
        ):
            raise ValueError("bad grid")
        values = dict(self.bindings)
        if self.graph == "row_markers_fill":
            return _row_markers_fill(grid, values["middle"])
        if self.graph == "bbox_crop_horiz_tile":
            return _bbox_crop_horiz_tile(grid, values["k"])
        if self.graph == "template_replace_components":
            return _template_replace_components(grid)
        if self.graph == "no_h_neighbor_recolor":
            return _no_h_neighbor_recolor(
                grid, values["obj"], values["new_value"],
            )
        if self.graph == "vertical_endpoint_marks":
            return _vertical_endpoint_marks(
                grid, values["obj"], values["mark"],
            )
        if self.graph == "marker_zigzag_columns":
            return _marker_zigzag_columns(grid)
        if self.graph == "frame_corner_crop_recolor":
            return _frame_corner_crop_recolor(grid)
        if self.graph == "top_right_crop":
            return _top_right_crop(
                grid, values["out_h"], values["out_w"],
            )
        if self.graph == "top_band_bottom_mirror":
            return _top_band_bottom_mirror(grid)
        if self.graph == "object_outbox_paint":
            return _object_outbox_paint(
                grid, values["obj"], values["ring"],
                values["hole"],
            )
        if self.graph == "bottom_center_markers":
            return _bottom_center_markers(grid, values["mark"])
        if self.graph == "seed_neighbor_paint":
            return _seed_neighbor_paint(grid, values["neighbor"])
        if self.graph == "max_colorcount_component_crop":
            return _max_colorcount_component_crop(grid, values["target"])
        if self.graph == "quadrant_overlay_top_left":
            return _quadrant_overlay_top_left(grid)
        if self.graph == "corner_diagonal_rays":
            return _corner_diagonal_rays(grid)
        if self.graph == "partition_backdrop_fill":
            return _partition_backdrop_fill(grid)
        if self.graph == "max_zero_rect_fill":
            return _max_zero_rect_fill(grid, values["fill"])
        raise ValueError("unknown a1 g2 graph")


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

    # row_markers_fill: colore centrale = colore nuovo nei demo.
    middles: set[int] = set()
    for source, expected in demos:
        out_colors = {cell for row in expected for cell in row}
        in_colors = {cell for row in source for cell in row}
        new_colors = out_colors - in_colors
        if len(new_colors) != 1:
            break
        middles.add(new_colors.pop())
    else:
        if len(middles) == 1:
            add(Candidate("row_markers_fill",
                          (("middle", middles.pop()),)))

    # bbox_crop_horiz_tile: k = outW // cropW dai demo.
    k_values: set[int] = set()
    for source, expected in demos:
        background = _background(source)
        cells = [
            (r, c)
            for r in range(len(source))
            for c in range(len(source[0]))
            if source[r][c] != background
        ]
        if not cells:
            k_values = None
            break
        c0 = min(c for _, c in cells)
        c1 = max(c for _, c in cells)
        crop_w = c1 - c0 + 1
        out_w = len(expected[0])
        if out_w % crop_w:
            k_values = None
            break
        k_values.add(out_w // crop_w)
    if k_values and len(k_values) == 1:
        add(Candidate("bbox_crop_horiz_tile",
                      (("k", k_values.pop()),)))

    add(Candidate("template_replace_components", ()))
    add(Candidate("marker_zigzag_columns", ()))
    add(Candidate("frame_corner_crop_recolor", ()))
    add(Candidate("top_band_bottom_mirror", ()))
    add(Candidate("quadrant_overlay_top_left", ()))
    add(Candidate("corner_diagonal_rays", ()))
    add(Candidate("partition_backdrop_fill", ()))

    # max_zero_rect_fill: colore di riempimento = colore nuovo negli output.
    fill_values: set[int] = set()
    for source, expected in demos:
        out_colors = {cell for row in expected for cell in row}
        in_colors = {cell for row in source for cell in row}
        diff = out_colors - in_colors
        if len(diff) != 1:
            fill_values = None
            break
        fill_values.add(diff.pop())
    if fill_values and len(fill_values) == 1:
        add(Candidate("max_zero_rect_fill",
                      (("fill", fill_values.pop()),)))

    # bottom_center_markers: mark = colore nuovo negli output.
    # seed_neighbor_paint: seed = colore meno frequente; neighbor =
    # l'altro colore dell'output.
    # max_colorcount_component_crop: target = colore comune ai demo.
    common_colors = set.intersection(*[
        {cell for row in source for cell in row} for source, _ in demos
    ]) if demos else set()
    new_colors: set[int] = set()
    for source, expected in demos:
        out_colors = {cell for row in expected for cell in row}
        in_colors = {cell for row in source for cell in row}
        new_colors |= out_colors - in_colors
    if len(new_colors) == 1:
        mark = new_colors.pop()
        add(Candidate("bottom_center_markers", (("mark", mark),)))
    neighbor_values: set[int] = set()
    for _, expected in demos:
        out_colors = {cell for row in expected for cell in row}
        neighbor_values |= out_colors - {0}
    common_neighbors = set.intersection(*[
        {cell for row in expected for cell in row} - {0}
        for _, expected in demos
    ]) if demos else set()
    if len(common_neighbors) == 1:
        add(Candidate("seed_neighbor_paint",
                      (("neighbor", common_neighbors.pop()),)))
    for target in sorted(common_colors - {0}):
        add(Candidate("max_colorcount_component_crop",
                      (("target", target),)))

    # top_right_crop: dimensioni dell'output dai demo.
    out_dims = {
        (len(expected), len(expected[0]))
        for _, expected in demos
    }
    if len(out_dims) == 1:
        out_h, out_w = out_dims.pop()
        add(Candidate("top_right_crop", (
            ("out_h", out_h), ("out_w", out_w),
        )))

    # object_outbox_paint: oggetto = unico colore non sfondo comune;
    # ring/hole = colori nuovi negli output, distinti per posizione.
    common = set.intersection(*[
        {cell for row in source for cell in row} for source, _ in demos
    ]) if demos else set()
    obj_candidates = sorted(common - {0})
    for obj in obj_candidates:
        ring_values: set[int] = set()
        hole_values: set[int] = set()
        ok = True
        for source, expected in demos:
            if len(expected) != len(source) \
                    or len(expected[0]) != len(source[0]):
                ok = False
                break
            background = _background(source)
            seen = [[False] * len(source[0]) for _ in range(len(source))]
            for r in range(len(source)):
                for c in range(len(source[0])):
                    if source[r][c] != obj or seen[r][c]:
                        continue
                    seen[r][c] = True
                    stack = [(r, c)]
                    comp = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < len(source) and 0 <= nc < len(source[0]) \
                                    and not seen[nr][nc] \
                                    and source[nr][nc] == obj:
                                seen[nr][nc] = True
                                stack.append((nr, nc))
                                comp.append((nr, nc))
                    r0 = min(rr for rr, _ in comp)
                    r1 = max(rr for rr, _ in comp)
                    c0 = min(cc for _, cc in comp)
                    c1 = max(cc for _, cc in comp)
                    for cc in range(c0 - 1, c1 + 2):
                        for rr in (r0 - 1, r1 + 1):
                            if 0 <= rr < len(source) and 0 <= cc < len(source[0]):
                                ring_values.add(expected[rr][cc])
                    for rr in range(r0 - 1, r1 + 2):
                        for cc in (c0 - 1, c1 + 1):
                            if 0 <= rr < len(source) and 0 <= cc < len(source[0]):
                                ring_values.add(expected[rr][cc])
                    for rr in range(r0, r1 + 1):
                        for cc in range(c0, c1 + 1):
                            if source[rr][cc] == background:
                                hole_values.add(expected[rr][cc])
        ring_values -= {obj, 0}
        hole_values -= {obj, 0}
        if ok and len(ring_values) == 1 and len(hole_values) == 1:
            add(Candidate("object_outbox_paint", (
                ("obj", obj),
                ("ring", ring_values.pop()),
                ("hole", hole_values.pop()),
            )))

    # no_h_neighbor_recolor: oggetto = unico colore non sfondo comune;
    # nuovo colore = colore presente solo negli output.
    common = set.intersection(*[
        {cell for row in source for cell in row} for source, _ in demos
    ]) if demos else set()
    new_values: set[int] = set()
    for source, expected in demos:
        out_colors = {cell for row in expected for cell in row}
        in_colors = {cell for row in source for cell in row}
        diff = out_colors - in_colors
        if len(diff) != 1:
            new_values = None
            break
        new_values.add(diff.pop())
    if new_values and len(new_values) == 1:
        for obj in sorted(common):
            if obj == 0:
                continue
            add(Candidate("no_h_neighbor_recolor", (
                ("obj", obj),
                ("new_value", next(iter(new_values))),
            )))
            add(Candidate("vertical_endpoint_marks", (
                ("obj", obj),
                ("mark", next(iter(new_values))),
            )))

    return tuple(sorted(
        result.values(), key=lambda item: (item.graph, repr(item.bindings))
    ))
