"""A1 G6 family algebra: candidati parametrici derivati dai soli demo.

Grafi (classe di trasformazione, nessun letterale di griglia):
  - shift_recolor_object: l'oggetto (colore legato) viene traslato di
    ``steps`` celle lungo ``direction``; le celle lasciate tornano
    sfondo; ogni cella spostata assume ``new_value``; le altre celle
    (marcatori) restano ferme. Fallisce se la traslazione esce dalla
    griglia o collide con celle non sfondo;
  - corner_lpath_connect: due celle endpoint di colori diversi vengono
    collegate da un percorso a L di ``path_value`` con gomito
    (riga del primo, colonna del secondo) secondo ``corner_mode``,
    senza toccare gli endpoint; passa solo su sfondo;
  - max_area_column_crop: le componenti a area massima vengono
    riassunte in una griglia di ``max_area`` righe, una colonna per
    colore selezionato (ordine = posizione orizzontale), riempita col
    colore stesso;
  - extreme_run_recolor: le corse rettilinee massimali del colore piu'
    frequente (non sfondo) lungo ``direction``: la piu' lunga diventa
    ``long_value``, la piu' corta ``short_value``, il resto sfondo;
  - diagonal_period_paint: la corsa diagonale (giu'-destra) piu' lunga
    del colore legato e' il seme di una famiglia periodica di periodo
    ``period`` lungo la direzione diagonale; l'output contiene solo le
    classi di congruenza del seme;
  - isolated_cell_recolor: le celle del colore oggetto senza vicini
    nella connettivita' ``connectivity`` vengono ricolorate con
    ``new_value``;
  - closed_component_recolor: le componenti 4-connesse del colore
    oggetto che racchiudono celle di sfondo (non raggiungibili dal
    bordo) vengono ricolorate con ``fill_value``;
  - rectangle_layer_recolor: ogni rettangolo massimale del colore
    oggetto (unione di rettangoli solidi) riceve strati: angoli
    ``corner_value``, bordo ``edge_value``, interno ``inner_value``;
  - pyramid_diagonal_rays: dalla riga superiore di una piramide a due
    livelli partono due raggi diagonali (su'-sinistra dall'estremo
    sinistro, su'-destra dal destro) color ``ray_value``, fin dentro i
    margini, solo su sfondo.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]
EXEC_ERRORS = (ValueError, IndexError, KeyError, StopIteration)

_DIRECTION_DELTAS = {
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
}
_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS_8 = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
)


def _background(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    return min(counts, key=lambda value: (-counts[value], value))


def _same_shape(first: Grid, second: Grid) -> bool:
    return len(first) == len(second) and len(first[0]) == len(second[0])


def _components(grid: Grid, value: int, neighbors: tuple) -> list[frozenset]:
    height, width = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    result: list[frozenset] = []
    for r in range(height):
        for c in range(width):
            if grid[r][c] != value or (r, c) in seen:
                continue
            seen.add((r, c))
            stack = [(r, c)]
            cells = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < height and 0 <= nc < width
                        and grid[nr][nc] == value and (nr, nc) not in seen
                    ):
                        seen.add((nr, nc))
                        stack.append((nr, nc))
                        cells.append((nr, nc))
            result.append(frozenset(cells))
    return result


def _runs(grid: Grid, value: int, direction: str) -> list[tuple[tuple[int, int], ...]]:
    """Maximal straight runs of ``value`` along ``direction``."""
    dr, dc = _DIRECTION_DELTAS[direction]
    height, width = len(grid), len(grid[0])
    runs = []
    for r in range(height):
        for c in range(width):
            if grid[r][c] != value:
                continue
            pr, pc = r - dr, c - dc
            if 0 <= pr < height and 0 <= pc < width and grid[pr][pc] == value:
                continue
            run = []
            rr, cc = r, c
            while 0 <= rr < height and 0 <= cc < width and grid[rr][cc] == value:
                run.append((rr, cc))
                rr += dr
                cc += dc
            runs.append(tuple(run))
    return runs


def _se_diagonal_runs(grid: Grid, value: int) -> list[tuple[tuple[int, int], ...]]:
    height, width = len(grid), len(grid[0])
    runs = []
    for r in range(height):
        for c in range(width):
            if grid[r][c] != value:
                continue
            if r > 0 and c > 0 and grid[r - 1][c - 1] == value:
                continue
            run = []
            rr, cc = r, c
            while rr < height and cc < width and grid[rr][cc] == value:
                run.append((rr, cc))
                rr += 1
                cc += 1
            runs.append(tuple(run))
    return runs


def _component_bbox(cells: frozenset) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(grid: Grid, cells: frozenset) -> bool:
    r0, c0, r1, c1 = _component_bbox(cells)
    if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
        return False
    return all(
        (r, c) in cells for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
    )


def _shift_object(grid: Grid, background: int, obj: int,
                  direction: str, steps: int, new_value: int) -> Grid:
    height, width = len(grid), len(grid[0])
    dr, dc = _DIRECTION_DELTAS[direction]
    cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == obj
    ]
    if not cells:
        raise ValueError("no object cells")
    out = [list(row) for row in grid]
    for r, c in cells:
        out[r][c] = background
    for r, c in cells:
        nr, nc = r + dr * steps, c + dc * steps
        if not (0 <= nr < height and 0 <= nc < width):
            raise ValueError("shift off grid")
        if out[nr][nc] != background:
            raise ValueError("shift collision")
        out[nr][nc] = new_value
    return tuple(tuple(row) for row in out)


def _lpath(grid: Grid, background: int, path_value: int,
           corner_mode: int) -> Grid:
    height, width = len(grid), len(grid[0])
    endpoints = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] != background
    ]
    if len(endpoints) != 2:
        raise ValueError("need exactly two endpoints")
    first, second = sorted(endpoints)
    anchor, pivot = (first, second) if corner_mode == 0 else (second, first)
    (ra, ca), (rd, cd) = anchor, pivot
    paint: set[tuple[int, int]] = set()
    lo, hi = sorted((ca, cd))
    for c in range(lo, hi + 1):
        if (ra, c) != anchor and (ra, c) != pivot:
            paint.add((ra, c))
    lo, hi = sorted((ra, rd))
    for r in range(lo, hi + 1):
        if (r, cd) != anchor and (r, cd) != pivot:
            paint.add((r, cd))
    if not paint:
        raise ValueError("no path cells")
    for r, c in paint:
        if grid[r][c] != background:
            raise ValueError("path blocked")
    out = [list(row) for row in grid]
    for r, c in paint:
        out[r][c] = path_value
    return tuple(tuple(row) for row in out)


def _lpath_colors(grid: Grid, background: int, path_value: int,
                  row_color: int, col_color: int) -> Grid:
    height, width = len(grid), len(grid[0])
    anchor = pivot = None
    for r in range(height):
        for c in range(width):
            if grid[r][c] == row_color:
                anchor = (r, c)
            if grid[r][c] == col_color:
                pivot = (r, c)
    if anchor is None or pivot is None:
        raise ValueError("missing endpoint")
    (ra, ca), (rd, cd) = anchor, pivot
    paint: set[tuple[int, int]] = set()
    lo, hi = sorted((ca, cd))
    for c in range(lo, hi + 1):
        if (ra, c) != anchor and (ra, c) != pivot:
            paint.add((ra, c))
    lo, hi = sorted((ra, rd))
    for r in range(lo, hi + 1):
        if (r, cd) != anchor and (r, cd) != pivot:
            paint.add((r, cd))
    if not paint:
        raise ValueError("no path cells")
    for r, c in paint:
        if grid[r][c] != background:
            raise ValueError("path blocked")
    out = [list(row) for row in grid]
    for r, c in paint:
        out[r][c] = path_value
    return tuple(tuple(row) for row in out)


def _area_column_crop(grid: Grid, background: int) -> Grid:
    height, width = len(grid), len(grid[0])
    colors = sorted({v for row in grid for v in row} - {background})
    objects = []
    for value in colors:
        for comp in _components(grid, value, _NEIGHBORS_8):
            objects.append((value, comp))
    if not objects:
        raise ValueError("no objects")
    max_area = max(len(cells) for _, cells in objects)
    selected = [
        (value, cells) for value, cells in objects if len(cells) == max_area
    ]
    selected.sort(key=lambda item: (
        min(c for _, c in item[1]),
        min(r for r, _ in item[1]),
        item[0],
    ))
    columns = tuple(value for value, _ in selected)
    return tuple(columns for _ in range(max_area))


def _extreme_run_recolor(grid: Grid, background: int, direction: str,
                         long_value: int, short_value: int) -> Grid:
    counts = Counter(v for row in grid for v in row if v != background)
    if not counts:
        raise ValueError("no foreground")
    line = max(counts, key=lambda v: (counts[v], -v))
    runs = _runs(grid, line, direction)
    if len(runs) < 2:
        raise ValueError("need at least two runs")
    longest = max(len(run) for run in runs)
    shortest = min(len(run) for run in runs)
    if longest == shortest:
        raise ValueError("ambiguous extremes")
    long_runs = [run for run in runs if len(run) == longest]
    short_runs = [run for run in runs if len(run) == shortest]
    if len(long_runs) != 1 or len(short_runs) != 1:
        raise ValueError("tied extremes")
    mapping = {}
    for cell in long_runs[0]:
        mapping[cell] = long_value
    for cell in short_runs[0]:
        mapping[cell] = short_value
    return tuple(
        tuple(mapping.get((r, c), background) for c in range(len(grid[0])))
        for r in range(len(grid))
    )


def _diagonal_period_paint(grid: Grid, background: int,
                           line_color: int, period: int) -> Grid:
    height, width = len(grid), len(grid[0])
    if not (1 <= period <= width):
        raise ValueError("bad period")
    runs = _se_diagonal_runs(grid, line_color)
    if not runs:
        raise ValueError("no line cells")
    longest = max(len(run) for run in runs)
    long_runs = [run for run in runs if len(run) == longest]
    if len(long_runs) != 1:
        raise ValueError("tied longest run")
    residues = {(r - c) % period for r, c in long_runs[0]}
    return tuple(
        tuple(
            line_color if (r - c) % period in residues else background
            for c in range(width)
        )
        for r in range(height)
    )


def _isolated_recolor(grid: Grid, background: int, obj: int,
                      connectivity: int, new_value: int) -> Grid:
    height, width = len(grid), len(grid[0])
    neighbors = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8
    if not any(grid[r][c] == obj for r in range(height) for c in range(width)):
        raise ValueError("no object cells")
    out = [list(row) for row in grid]
    for r in range(height):
        for c in range(width):
            if grid[r][c] != obj:
                continue
            alone = all(
                not (0 <= r + dr < height and 0 <= c + dc < width
                     and grid[r + dr][c + dc] == obj)
                for dr, dc in neighbors
            )
            if alone:
                out[r][c] = new_value
    return tuple(tuple(row) for row in out)


def _closed_recolor(grid: Grid, background: int, obj: int, fill_value: int) -> Grid:
    height, width = len(grid), len(grid[0])
    if not any(grid[r][c] == obj for r in range(height) for c in range(width)):
        raise ValueError("no object cells")
    open_cells: set[tuple[int, int]] = set()
    stack = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] != obj and (r == 0 or r == height - 1
                                  or c == 0 or c == width - 1)
    ]
    open_cells.update(stack)
    while stack:
        cr, cc = stack.pop()
        for dr, dc in _NEIGHBORS_4:
            nr, nc = cr + dr, cc + dc
            if (
                0 <= nr < height and 0 <= nc < width
                and grid[nr][nc] != obj and (nr, nc) not in open_cells
            ):
                open_cells.add((nr, nc))
                stack.append((nr, nc))
    out = [list(row) for row in grid]
    changed = False
    for comp in _components(grid, obj, _NEIGHBORS_4):
        encloses = any(
            (r + dr, c + dc) not in open_cells
            and 0 <= r + dr < height and 0 <= c + dc < width
            and grid[r + dr][c + dc] != obj
            for r, c in comp
            for dr, dc in _NEIGHBORS_4
        )
        if encloses:
            changed = True
            for r, c in comp:
                out[r][c] = fill_value
    if not changed:
        raise ValueError("no closed component")
    return tuple(tuple(row) for row in out)


def _rect_boxes(grid: Grid, obj: int) -> list[tuple[int, int, int, int]]:
    """Solid-rectangle decomposition (r0, c0, r1, c1) of the obj region."""
    height, width = len(grid), len(grid[0])
    cells = {
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == obj
    }
    if not cells:
        return []
    rectangles = []
    for r, c in sorted(cells):
        c1 = c
        while c1 + 1 < width and grid[r][c1 + 1] == obj:
            c1 += 1
        r1 = r
        while r1 + 1 < height and all(
            grid[r1 + 1][cc] == obj for cc in range(c, c1 + 1)
        ):
            r1 += 1
        rectangles.append((r, c, r1, c1, (r1 - r + 1) * (c1 - c + 1)))
    rectangles.sort(key=lambda item: (-item[4], item[0], item[1]))
    assigned: set[tuple[int, int]] = set()
    boxes = []
    for r, c, r1, c1, _ in rectangles:
        new_cells = [
            (rr, cc)
            for rr in range(r, r1 + 1)
            for cc in range(c, c1 + 1)
            if (rr, cc) in cells and (rr, cc) not in assigned
        ]
        if not new_cells:
            continue
        if len(new_cells) != (r1 - r + 1) * (c1 - c + 1):
            nr0 = min(rr for rr, _ in new_cells)
            nr1 = max(rr for rr, _ in new_cells)
            nc0 = min(cc for _, cc in new_cells)
            nc1 = max(cc for _, cc in new_cells)
            if len(new_cells) != (nr1 - nr0 + 1) * (nc1 - nc0 + 1):
                raise ValueError("ragged rectangle")
            r, c, r1, c1 = nr0, nc0, nr1, nc1
        assigned.update(new_cells)
        boxes.append((r, c, r1, c1))
    if assigned != cells:
        raise ValueError("decomposition incomplete")
    return boxes


def _rectangle_layers(grid: Grid, background: int, obj: int,
                      corner_value: int, edge_value: int,
                      inner_value: int) -> Grid:
    boxes = _rect_boxes(grid, obj)
    if not boxes:
        raise ValueError("no object cells")
    out = [list(row) for row in grid]
    for r0, c0, r1, c1 in boxes:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                corner = r in (r0, r1) and c in (c0, c1)
                edge = r in (r0, r1) or c in (c0, c1)
                out[r][c] = (
                    corner_value if corner
                    else edge_value if edge
                    else inner_value
                )
    return tuple(tuple(row) for row in out)


def _pyramid_rays(grid: Grid, background: int, ray_value: int) -> Grid:
    height, width = len(grid), len(grid[0])
    cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] != background
    ]
    if not cells:
        raise ValueError("empty grid")
    bottom = max(r for r, _ in cells)
    upper = bottom - 1
    if upper < 0:
        raise ValueError("no upper tier")
    upper_cols = sorted(c for r, c in cells if r == upper)
    if not upper_cols:
        raise ValueError("empty upper tier")
    bottom_cols = {c for r, c in cells if r == bottom}
    if not bottom_cols:
        raise ValueError("empty bottom tier")
    if any(r not in (bottom, upper) for r, _ in cells):
        raise ValueError("unexpected tier")
    out = [list(row) for row in grid]
    for origin_c, dc in ((upper_cols[0], -1), (upper_cols[-1], 1)):
        k = 1
        while True:
            nr, nc = upper - k, origin_c + dc * k
            if not (0 <= nr < height and 0 <= nc < width):
                break
            if grid[nr][nc] != background:
                raise ValueError("ray blocked")
            out[nr][nc] = ray_value
            k += 1
    return tuple(tuple(row) for row in out)


def _minority_bbox_crop(grid: Grid) -> Grid:
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
    return tuple(
        tuple(
            0 if grid[r][c] == background else grid[r][c]
            for c in range(c0, c1 + 1)
        )
        for r in range(r0, r1 + 1)
    )


def _straight_runs(grid: Grid, value: int):
    height, width = len(grid), len(grid[0])
    runs = []
    for r in range(height):
        c = 0
        while c < width:
            if grid[r][c] == value:
                c0 = c
                while c < width and grid[r][c] == value:
                    c += 1
                if c - c0 >= 2:
                    runs.append(tuple((r, cc) for cc in range(c0, c)))
            else:
                c += 1
    for c in range(width):
        r = 0
        while r < height:
            if grid[r][c] == value:
                r0 = r
                while r < height and grid[r][c] == value:
                    r += 1
                if r - r0 >= 2:
                    runs.append(tuple((rr, c) for rr in range(r0, r)))
            else:
                r += 1
    return runs


def _run_rank_recolor(grid: Grid, long_value: int, mid_value: int,
                      short_value: int) -> Grid:
    background = _background(grid)
    values = [v for row in grid for v in row if v != background]
    if not values:
        raise ValueError("no object cells")
    obj = max(set(values), key=lambda v: (values.count(v), -v))
    runs = _straight_runs(grid, obj)
    if len(runs) != 3:
        raise ValueError("expected exactly three runs")
    covered = set()
    for run in runs:
        for cell in run:
            if cell in covered:
                raise ValueError("overlapping runs")
            covered.add(cell)
    if covered != {(r, c) for r in range(len(grid))
                   for c in range(len(grid[0])) if grid[r][c] == obj}:
        raise ValueError("runs do not cover object")
    lengths = sorted({len(run) for run in runs}, reverse=True)
    if len(lengths) != 3:
        raise ValueError("runs need three distinct lengths")
    rank_value = {lengths[0]: long_value, lengths[1]: mid_value,
                  lengths[2]: short_value}
    out = [list(row) for row in grid]
    for run in runs:
        value = rank_value[len(run)]
        for r, c in run:
            out[r][c] = value
    return tuple(tuple(row) for row in out)


def _center_align_to_frame(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    marker = None
    for value in sorted(
        {cell for row in grid for cell in row} - {background}
    ):
        cells = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if grid[r][c] == value
        ]
        if len(cells) != 4:
            continue
        rows = sorted({r for r, _ in cells})
        cols = sorted({c for _, c in cells})
        if len(rows) == 2 and len(cols) == 2:
            marker = value
            break
    if marker is None:
        raise ValueError("no corner marker")
    top, bottom = min(r for r, _ in [(r, c) for r in range(height) for c in range(width) if grid[r][c] == marker]), max(r for r, _ in [(r, c) for r in range(height) for c in range(width) if grid[r][c] == marker])
    left, right = min(c for _, c in [(r, c) for r in range(height) for c in range(width) if grid[r][c] == marker]), max(c for _, c in [(r, c) for r in range(height) for c in range(width) if grid[r][c] == marker])
    others = [
        value for value in
        {cell for row in grid for cell in row} - {background, marker}
    ]
    if len(others) != 1:
        raise ValueError("shape color ambiguous")
    shape_value = others[0]
    comps = _components(grid, shape_value, ((1, 0), (-1, 0), (0, 1), (0, -1)))
    if len(comps) != 1:
        raise ValueError("shape component ambiguous")
    comp = comps[0]
    sr0 = min(r for r, _ in comp)
    sr1 = max(r for r, _ in comp)
    sc0 = min(c for _, c in comp)
    sc1 = max(c for _, c in comp)
    inner_top, inner_bottom = top + 1, bottom - 1
    inner_left, inner_right = left + 1, right - 1
    if inner_top > inner_bottom or inner_left > inner_right:
        raise ValueError("no inner region")
    if (inner_top + inner_bottom - sr0 - sr1) % 2:
        raise ValueError("vertical misalignment")
    dr = (inner_top + inner_bottom - sr0 - sr1) // 2
    dc = (inner_left + inner_right - sc0 - sc1) // 2
    out = [[background] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            if grid[r][c] == marker:
                out[r][c] = marker
    for r, c in comp:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < height and 0 <= nc < width):
            raise ValueError("shape out of bounds")
        if out[nr][nc] != background:
            raise ValueError("shape collision")
        out[nr][nc] = shape_value
    return tuple(tuple(row) for row in out)


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]
    _cost: int = 1

    def cost(self, grid: Grid) -> int:
        return self._cost + len(grid) * len(grid[0])

    def execute(self, grid: Grid) -> Grid:
        if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("bad grid")
        values = dict(self.bindings)
        if self.graph == "center_align_to_frame":
            return _center_align_to_frame(grid)
        if self.graph == "run_rank_recolor":
            return _run_rank_recolor(grid, values["long_value"],
                                     values["mid_value"],
                                     values["short_value"])
        if self.graph == "minority_bbox_crop":
            return _minority_bbox_crop(grid)
        if self.graph == "lpath_color_elbow":
            return _lpath_colors(
                grid, _background(grid), values["path_value"],
                values["row_color"], values["col_color"],
            )
        if self.graph == "shift_recolor_object":
            return _shift_object(
                grid, _background(grid), values["object_color"],
                values["direction"], values["steps"], values["new_value"],
            )
        if self.graph == "corner_lpath_connect":
            return _lpath(
                grid, _background(grid), values["path_value"],
                values["corner_mode"],
            )
        if self.graph == "max_area_column_crop":
            return _area_column_crop(grid, _background(grid))
        if self.graph == "extreme_run_recolor":
            return _extreme_run_recolor(
                grid, _background(grid), values["direction"],
                values["long_value"], values["short_value"],
            )
        if self.graph == "diagonal_period_paint":
            return _diagonal_period_paint(
                grid, _background(grid), values["line_color"],
                values["period"],
            )
        if self.graph == "isolated_cell_recolor":
            return _isolated_recolor(
                grid, values["background"], values["object_color"],
                values["connectivity"], values["new_value"],
            )
        if self.graph == "closed_component_recolor":
            return _closed_recolor(
                grid, _background(grid), values["object_color"],
                values["fill_value"],
            )
        if self.graph == "rectangle_layer_recolor":
            return _rectangle_layers(
                grid, _background(grid), values["object_color"],
                values["corner_value"], values["edge_value"],
                values["inner_value"],
            )
        if self.graph == "pyramid_diagonal_rays":
            return _pyramid_rays(grid, _background(grid), values["ray_value"])
        raise ValueError("unknown a1 g6 graph")


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

    _emit_shift(demos, add)
    _emit_lpath(demos, add)
    _emit_area_crop(demos, add)
    add(Candidate("center_align_to_frame", ()))
    add(Candidate("minority_bbox_crop", ()))
    _emit_extreme_runs(demos, add)
    _emit_run_rank(demos, add)
    _emit_diagonal_period(demos, add)
    _emit_isolated(demos, add)
    _emit_closed(demos, add)
    _emit_rectangles(demos, add)
    _emit_pyramid_rays(demos, add)

    return tuple(
        sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings)))
    )


def _emit_shift(demos: Demos, add) -> None:
    backgrounds = {_background(source) for source, _ in demos}
    if len(backgrounds) != 1:
        return
    background = backgrounds.pop()
    colors = sorted({
        v for source, _ in demos for row in source for v in row
    } - {background})
    for obj in colors:
        for direction in sorted(_DIRECTION_DELTAS):
            for steps in (1, 2):
                dr, dc = _DIRECTION_DELTAS[direction]
                new_values: set[int] = set()
                ok = True
                for source, expected in demos:
                    if not _same_shape(source, expected):
                        ok = False
                        break
                    height, width = len(source), len(source[0])
                    cells = [
                        (r, c)
                        for r in range(height)
                        for c in range(width)
                        if source[r][c] == obj
                    ]
                    if not cells:
                        ok = False
                        break
                    erased = set(cells)
                    for r, c in cells:
                        nr, nc = r + dr * steps, c + dc * steps
                        if not (0 <= nr < height and 0 <= nc < width):
                            ok = False
                            break
                        if (nr, nc) not in erased and source[nr][nc] != background:
                            ok = False
                            break
                        new_values.add(expected[nr][nc])
                    if not ok:
                        break
                if not ok or len(new_values) != 1:
                    continue
                add(Candidate("shift_recolor_object", (
                    ("direction", direction),
                    ("new_value", new_values.pop()),
                    ("object_color", obj),
                    ("steps", steps),
                )))


def _emit_lpath(demos: Demos, add) -> None:
    backgrounds = {_background(source) for source, _ in demos}
    if len(backgrounds) != 1:
        return
    background = backgrounds.pop()
    endpoint_positions = []
    for source, _ in demos:
        endpoints = [
            (r, c)
            for r in range(len(source))
            for c in range(len(source[0]))
            if source[r][c] != background
        ]
        if len(endpoints) != 2:
            return
        endpoint_positions.append(tuple(sorted(endpoints)))
    path_values = set()
    for source, expected in demos:
        if not _same_shape(source, expected):
            return
        changed = {
            expected[r][c]
            for r in range(len(source))
            for c in range(len(source[0]))
            if source[r][c] != expected[r][c]
        }
        if len(changed) != 1:
            return
        path_values.add(changed.pop())
    if len(path_values) != 1:
        return
    path_value = path_values.pop()
    for corner_mode in (0, 1):
        ok = True
        for source, expected in demos:
            try:
                if _lpath(source, background, path_value, corner_mode) != expected:
                    ok = False
                    break
            except EXEC_ERRORS:
                ok = False
                break
        if ok:
            add(Candidate("corner_lpath_connect", (
                ("corner_mode", corner_mode),
                ("path_value", path_value),
            )))
    endpoint_colors = [
        {source[r][c] for r, c in
         [(x, y) for x in range(len(source)) for y in range(len(source[0]))
          if source[x][y] != background]}
        for source, _ in demos
    ]
    if all(colors == endpoint_colors[0] for colors in endpoint_colors) \
            and len(endpoint_colors[0]) == 2:
        for row_color, col_color in (
            tuple(sorted(endpoint_colors[0])),
            tuple(reversed(sorted(endpoint_colors[0]))),
        ):
            add(Candidate("lpath_color_elbow", (
                ("path_value", path_value),
                ("row_color", row_color),
                ("col_color", col_color),
            )))


def _emit_area_crop(demos: Demos, add) -> None:
    for _, expected in demos:
        if not expected or not expected[0]:
            return
    for source, expected in demos:
        try:
            if _area_column_crop(source, _background(source)) != expected:
                return
        except EXEC_ERRORS:
            return
    add(Candidate("max_area_column_crop", ()))


def _emit_extreme_runs(demos: Demos, add) -> None:
    backgrounds = {_background(source) for source, _ in demos}
    if len(backgrounds) != 1:
        return
    background = backgrounds.pop()
    for direction in sorted(_DIRECTION_DELTAS):
        long_values: set[int] = set()
        short_values: set[int] = set()
        ok = True
        for source, expected in demos:
            if not _same_shape(source, expected):
                ok = False
                break
            counts = Counter(v for row in source for v in row if v != background)
            if not counts:
                ok = False
                break
            line = max(counts, key=lambda v: (counts[v], -v))
            runs = _runs(source, line, direction)
            if len(runs) < 2:
                ok = False
                break
            longest = max(len(run) for run in runs)
            shortest = min(len(run) for run in runs)
            long_runs = [run for run in runs if len(run) == longest]
            short_runs = [run for run in runs if len(run) == shortest]
            if longest == shortest or len(long_runs) != 1 or len(short_runs) != 1:
                ok = False
                break
            for r, c in long_runs[0]:
                long_values.add(expected[r][c])
            for r, c in short_runs[0]:
                short_values.add(expected[r][c])
        if not ok or len(long_values) != 1 or len(short_values) != 1:
            continue
        add(Candidate("extreme_run_recolor", (
            ("direction", direction),
            ("long_value", long_values.pop()),
            ("short_value", short_values.pop()),
        )))


def _emit_diagonal_period(demos: Demos, add) -> None:
    backgrounds = {_background(source) for source, _ in demos}
    if len(backgrounds) != 1:
        return
    background = backgrounds.pop()
    line_colors = set()
    for source, _ in demos:
        counts = Counter(v for row in source for v in row if v != background)
        if not counts:
            return
        line_colors.add(max(counts, key=lambda v: (counts[v], -v)))
    if len(line_colors) != 1:
        return
    line_color = line_colors.pop()
    periods: set[int] | None = None
    for source, expected in demos:
        if not _same_shape(source, expected):
            return
        runs = _se_diagonal_runs(source, line_color)
        if not runs:
            return
        longest = max(len(run) for run in runs)
        long_runs = [run for run in runs if len(run) == longest]
        if len(long_runs) != 1:
            return
        width = len(source[0])
        good = set()
        for period in range(longest, width + 1):
            residues = {(r - c) % period for r, c in long_runs[0]}
            got = tuple(
                tuple(
                    line_color if (r - c) % period in residues else background
                    for c in range(width)
                )
                for r in range(len(source))
            )
            if got == expected:
                good.add(period)
        if not good:
            return
        periods = good if periods is None else periods & good
    if not periods:
        return
    for period in sorted(periods):
        add(Candidate("diagonal_period_paint", (
            ("line_color", line_color),
            ("period", period),
        )))


def _emit_isolated(demos: Demos, add) -> None:
    common_colors = set.intersection(*[
        {v for row in source for v in row} for source, _ in demos
    ]) if demos else set()
    for background in sorted(common_colors):
        object_colors = set()
        for source, _ in demos:
            colors = {v for row in source for v in row} - {background}
            if len(colors) != 1:
                object_colors = None
                break
            object_colors.add(colors.pop())
        if not object_colors or len(object_colors) != 1:
            continue
        obj = object_colors.pop()
        new_values = set()
        for source, expected in demos:
            if not _same_shape(source, expected):
                new_values = None
                break
            changed = {
                expected[r][c]
                for r in range(len(source))
                for c in range(len(source[0]))
                if source[r][c] != expected[r][c]
            }
            if len(changed) != 1:
                new_values = None
                break
            new_values.add(changed.pop())
        if not new_values or len(new_values) != 1:
            continue
        new_value = new_values.pop()
        for connectivity in (4, 8):
            ok = True
            for source, expected in demos:
                try:
                    got = _isolated_recolor(source, background, obj,
                                            connectivity, new_value)
                except EXEC_ERRORS:
                    ok = False
                    break
                if got != expected:
                    ok = False
                    break
            if ok:
                add(Candidate("isolated_cell_recolor", (
                    ("connectivity", connectivity),
                    ("new_value", new_value),
                    ("object_color", obj),
                    ("background", background),
                )))


def _emit_closed(demos: Demos, add) -> None:
    backgrounds = {_background(source) for source, _ in demos}
    if len(backgrounds) != 1:
        return
    background = backgrounds.pop()
    object_colors = set()
    for source, _ in demos:
        colors = {v for row in source for v in row} - {background}
        if len(colors) != 1:
            return
        object_colors.add(colors.pop())
    if len(object_colors) != 1:
        return
    obj = object_colors.pop()
    fill_values = set()
    for source, expected in demos:
        if not _same_shape(source, expected):
            return
        changed = {
            expected[r][c]
            for r in range(len(source))
            for c in range(len(source[0]))
            if source[r][c] == obj and source[r][c] != expected[r][c]
        }
        if len(changed) != 1:
            return
        fill_values.add(changed.pop())
    if len(fill_values) != 1:
        return
    fill_value = fill_values.pop()
    ok = True
    for source, expected in demos:
        try:
            if _closed_recolor(source, background, obj, fill_value) != expected:
                ok = False
                break
        except EXEC_ERRORS:
            ok = False
            break
    if ok:
        add(Candidate("closed_component_recolor", (
            ("fill_value", fill_value),
            ("object_color", obj),
        )))


def _emit_run_rank(demos: Demos, add) -> None:
    obj_colors = set()
    for source, _ in demos:
        background = _background(source)
        values = [v for row in source for v in row if v != background]
        if not values:
            return
        obj_colors.add(max(set(values), key=lambda v: (values.count(v), -v)))
    if len(obj_colors) != 1:
        return
    obj = obj_colors.pop()
    long_values: set[int] = set()
    mid_values: set[int] = set()
    short_values: set[int] = set()
    for source, expected in demos:
        runs = _straight_runs(source, obj)
        if len(runs) != 3:
            return
        covered = set()
        for run in runs:
            for cell in run:
                if cell in covered:
                    return
                covered.add(cell)
        if covered != {(r, c) for r in range(len(source))
                       for c in range(len(source[0]))
                       if source[r][c] == obj}:
            return
        lengths = sorted({len(run) for run in runs}, reverse=True)
        if len(lengths) != 3:
            return
        rank_value = {lengths[0]: 0, lengths[1]: 1, lengths[2]: 2}
        for run in runs:
            values_for_run = {expected[r][c] for r, c in run}
            if len(values_for_run) != 1:
                return
            rank = rank_value[len(run)]
            if rank == 0:
                long_values.add(values_for_run.pop())
            elif rank == 1:
                mid_values.add(values_for_run.pop())
            else:
                short_values.add(values_for_run.pop())
    if len(long_values) != 1 or len(mid_values) != 1 \
            or len(short_values) != 1:
        return
    add(Candidate("run_rank_recolor", (
        ("long_value", long_values.pop()),
        ("mid_value", mid_values.pop()),
        ("short_value", short_values.pop()),
    )))


def _emit_rectangles(demos: Demos, add) -> None:
    common_colors = set.intersection(*[
        {v for row in source for v in row} for source, _ in demos
    ]) if demos else set()
    for background in sorted(common_colors):
        object_colors = set()
        for source, _ in demos:
            colors = {v for row in source for v in row} - {background}
            if len(colors) != 1:
                object_colors = None
                break
            object_colors.add(colors.pop())
        if not object_colors or len(object_colors) != 1:
            continue
        obj = object_colors.pop()
        corner_values: set[int] = set()
        edge_values: set[int] = set()
        inner_values: set[int] = set()
        for source, expected in demos:
            if not _same_shape(source, expected):
                corner_values = None
                break
            try:
                boxes = _rect_boxes(source, obj)
            except EXEC_ERRORS:
                corner_values = None
                break
            if not boxes:
                corner_values = None
                break
            for r0, c0, r1, c1 in boxes:
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        corner = r in (r0, r1) and c in (c0, c1)
                        edge = r in (r0, r1) or c in (c0, c1)
                        if corner:
                            corner_values.add(expected[r][c])
                        elif edge:
                            edge_values.add(expected[r][c])
                        else:
                            inner_values.add(expected[r][c])
        if not corner_values or len(corner_values) != 1 \
                or len(edge_values) != 1 or len(inner_values) != 1:
            continue
        ok = True
        for source, expected in demos:
            try:
                got = _rectangle_layers(
                    source, background, obj,
                    next(iter(corner_values)), next(iter(edge_values)),
                    next(iter(inner_values)),
                )
            except EXEC_ERRORS:
                ok = False
                break
            if got != expected:
                ok = False
                break
        if ok:
            add(Candidate("rectangle_layer_recolor", (
                ("corner_value", corner_values.pop()),
                ("edge_value", edge_values.pop()),
                ("inner_value", inner_values.pop()),
                ("object_color", obj),
                ("background", background),
            )))


def _emit_pyramid_rays(demos: Demos, add) -> None:
    backgrounds = {_background(source) for source, _ in demos}
    if len(backgrounds) != 1:
        return
    background = backgrounds.pop()
    ray_values = set()
    for source, expected in demos:
        if not _same_shape(source, expected):
            return
        changed = {
            expected[r][c]
            for r in range(len(source))
            for c in range(len(source[0]))
            if source[r][c] != expected[r][c]
        }
        if len(changed) != 1:
            return
        ray_values.add(changed.pop())
    if len(ray_values) != 1:
        return
    ray_value = ray_values.pop()
    ok = True
    for source, expected in demos:
        try:
            if _pyramid_rays(source, background, ray_value) != expected:
                ok = False
                break
        except EXEC_ERRORS:
            ok = False
            break
    if ok:
        add(Candidate("pyramid_diagonal_rays", (("ray_value", ray_value),)))
