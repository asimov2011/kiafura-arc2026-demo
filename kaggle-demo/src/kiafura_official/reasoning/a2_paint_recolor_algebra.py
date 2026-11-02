"""A2 paint/recolor algebra: candidati derivati dai soli demo.

Grafi:
  - rays_from_singletons: raggi cardinali dai singleton, nel loro colore;
  - uniform_rows_canvas: righe uniformi -> mark, altre -> background;
  - hole_size_recolor: buchi di background ricolorati per taglia, resto
    del background al colore default;
  - component_size_recolor: componenti ricolorate per taglia;
  - modulo_recolor: celle di un colore su indice congruo a un modulo;
  - fill_inbox_components: inbox del bbox di ogni componente riempito;
  - fill_outbox_all: outbox del bbox di tutto il foreground riempito;
  - funnel_drip_streams: i punti sotto il corpo unico, nelle sue
    colonne, scendono in flussi fino al bordo;
  - blob_defect_rays: blocco a due colori; le celle interne incuneate
    tra celle esterne definiscono raggi verticali, la riga interna piu'
    larga il raggio orizzontale;
  - triple_completion_pairs: le terne complete nell'input definiscono
    le coppie (centro, fianco); centri isolati ricevono i fianchi e i
    vuoti tra fianchi ricevono il centro;
  - band_span_closure: griglia a bande separata da righe/colonne
    uniformi; i blocchi dello stesso colore su una banda si estendono
    tra gli estremi, salvo nelle celle coperte da uno span ortogonale
    dello stesso colore;
  - mover_rise_up: le celle del colore mover salgono per colonna
    attraverso il background finche' bloccate;
  - column_bottom_half_mark: la meta' inferiore di ogni segmento
    verticale del colore e' ricolorata col mark;
  - slot_row_col_closure: celle regolari separate da bande vuote; i
    colori speciali si estendono alla riga/colonna di celle piu'
    occupata, nello slot di origine;
  - marker_cluster_bbox_fill: i marker si raggruppano per distanza
    Chebyshev <= 2; i buchi nel bbox di ogni cluster sono riempiti;
  - maximal_rect_interior_fill: gli interni dei rettangoli i cui lati
    sono segmenti massimali di muro (un lato puo' essere il bordo della
    griglia) sono riempiti col colore 4;
  - glyph_marker_dihedral_stamp: glifo con colore di riempimento
    maggioritario e due marker singleton; ogni coppia di blocchi
    quadrati uniformi dei colori marker riceve la forma di
    riempimento in scala k, sotto l'unica trasformazione diedrale che
    allinea i marker ai blocchi;
  - maximal_zero_rect_paint: i rettangoli massimali di background
    (lato minimo >= 2) sono riempiti col colore 2; nei gruppi di
    rettangoli sovrapposti sopravvive il piu' grande per area;
  - core_blade_reflection_stamp: ogni oggetto ha un nucleo 2x2 (anche
    con celle di background) e una lama monocromatica attaccata a un
    angolo del nucleo; la lama viene riflessa sugli altri angoli non
    nulli del nucleo e ricolorata col colore dell'angolo;
  - edge_beam_marker_step: gli emettitori 8 su un bordo lanciano raggi
    verso l'interno; a ogni linea definita dai marker 2 sul bordo
    parallelo il raggio si sposta di una cella in direzione opposta al
    bordo dei marker, e le celle fuori griglia cadono.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]
EXEC_ERRORS = (ValueError, IndexError, KeyError, StopIteration)

DIRECTIONS = {
    "down": (1, 0),
    "up": (-1, 0),
    "right": (0, 1),
    "left": (0, -1),
}

TRIPLE_OFFSETS = {
    "horizontal": ((0, -1), (0, 1)),
    "vertical": ((-1, 0), (1, 0)),
    "cross": ((0, -1), (0, 1), (-1, 0), (1, 0)),
    "diagonal": ((-1, -1), (-1, 1), (1, -1), (1, 1)),
}


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
            items.append((color, frozenset(cells)))
    items.sort(key=lambda item: min(item[1]))
    return items


def _separator_lines(grid: Grid, background: int):
    height, width = len(grid), len(grid[0])
    sep_rows = [
        r for r in range(height)
        if len(set(grid[r])) == 1 and grid[r][0] != background
    ]
    sep_cols = [
        c for c in range(width)
        if len({grid[r][c] for r in range(height)}) == 1
        and grid[0][c] != background
    ]
    return sep_rows, sep_cols


def _segments(length: int, separators) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for mark in sorted(set(separators)):
        if mark > start:
            spans.append((start, mark))
        start = mark + 1
    if start < length:
        spans.append((start, length))
    return spans


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
        if self.graph == "rays_from_singletons":
            dr, dc = DIRECTIONS[values["direction"]]
            out = [list(row) for row in grid]
            fired = 0
            for color, cells in _components4(grid, background):
                if len(cells) != 1:
                    continue
                r, c = next(iter(cells))
                nr, nc = r + dr, c + dc
                while 0 <= nr < height and 0 <= nc < width:
                    if grid[nr][nc] != background:
                        break
                    out[nr][nc] = color
                    fired += 1
                    nr += dr
                    nc += dc
            if not fired:
                raise ValueError("no ray drawn")
            return tuple(tuple(row) for row in out)
        if self.graph == "uniform_rows_canvas":
            mark = values["mark"]
            blank = values["background"]
            out = []
            for row in grid:
                if len(set(row)) == 1:
                    out.append(tuple(mark for _ in row))
                else:
                    out.append(tuple(blank for _ in row))
            return tuple(out)
        if self.graph == "hole_size_recolor":
            mapping = dict(values["mapping"])
            default = values["default"]
            out = [list(row) for row in grid]
            components = _components4(grid, background)
            if not components:
                raise ValueError("no components")
            sizes = [len(cells) for _, cells in components]
            largest_size = max(sizes)
            for color, cells in components:
                if len(cells) == largest_size and sizes.count(largest_size) == 1:
                    target = default
                else:
                    target = mapping.get(len(cells), default)
                for r, c in cells:
                    out[r][c] = target
            return tuple(tuple(row) for row in out)
        if self.graph == "component_size_recolor":
            mapping = dict(values["mapping"])
            default = values["default"]
            out = [list(row) for row in grid]
            changed = False
            for color, cells in _components4(grid, background):
                new_color = mapping.get(len(cells), default)
                for r, c in cells:
                    out[r][c] = new_color
                    changed = True
            background_new = values["background_color"]
            for r in range(height):
                for c in range(width):
                    if grid[r][c] == background:
                        out[r][c] = background_new
            if not changed:
                raise ValueError("no components")
            return tuple(tuple(row) for row in out)
        if self.graph == "modulo_recolor":
            axis = values["axis"]
            modulus = values["modulus"]
            residue = values["residue"]
            source = values["source"]
            mark = values["mark"]
            out = [list(row) for row in grid]
            changed = False
            for r in range(height):
                for c in range(width):
                    index = c if axis == "column" else r
                    if grid[r][c] == source and index % modulus == residue:
                        out[r][c] = mark
                        changed = True
            if not changed:
                raise ValueError("nothing to recolor")
            return tuple(tuple(row) for row in out)
        if self.graph == "fill_inbox_components":
            mark = values["mark"]
            out = [list(row) for row in grid]
            filled = 0
            for _, cells in _components4(grid, background):
                rows = {r for r, _ in cells}
                cols = {c for _, c in cells}
                for r in range(min(rows) + 1, max(rows)):
                    for c in range(min(cols) + 1, max(cols)):
                        out[r][c] = mark
                        filled += 1
            if not filled:
                raise ValueError("no inbox to fill")
            return tuple(tuple(row) for row in out)
        if self.graph == "fill_outbox_all":
            mark = values["mark"]
            components = _components4(grid, background)
            if not components:
                raise ValueError("no foreground")
            out = [list(row) for row in grid]
            for _, cells in components:
                rows = {r for r, _ in cells}
                cols = {c for _, c in cells}
                for r in range(max(0, min(rows) - 1),
                               min(height, max(rows) + 2)):
                    for c in range(max(0, min(cols) - 1),
                                   min(width, max(cols) + 2)):
                        out[r][c] = mark
            return tuple(tuple(row) for row in out)
        if self.graph == "panel_sparse_map_fill":
            mapping = dict(values["mapping"])
            sep_rows, sep_cols = _separator_lines(grid, background)
            row_bounds = _segments(height, sep_rows)
            col_bounds = _segments(width, sep_cols)
            if not row_bounds or not col_bounds:
                raise ValueError("no panels")
            out = [list(row) for row in grid]
            for r0, r1 in row_bounds:
                for c0, c1 in col_bounds:
                    found = None
                    for r in range(r0, r1):
                        for c in range(c0, c1):
                            value = grid[r][c]
                            if value == background:
                                continue
                            if found is not None:
                                raise ValueError("panel not sparse")
                            found = value
                    if found is None or found not in mapping:
                        raise ValueError("panel value unmapped")
                    fill = mapping[found]
                    for r in range(r0, r1):
                        for c in range(c0, c1):
                            out[r][c] = fill
            return tuple(tuple(row) for row in out)
        if self.graph == "open_line_seam_fill":
            seam = values["seam"]
            open_rows = [
                r for r in range(1, height - 1)
                if grid[r][0] != background
                and grid[r][width - 1] != background
                and all(grid[r][c] == background for c in range(1, width - 1))
            ]
            open_cols = [
                c for c in range(1, width - 1)
                if grid[0][c] != background
                and grid[height - 1][c] != background
                and all(grid[r][c] == background for r in range(1, height - 1))
            ]
            if not open_rows and not open_cols:
                raise ValueError("no open line")
            out = [list(row) for row in grid]
            for r in open_rows:
                for c in range(1, width - 1):
                    out[r][c] = seam
            for c in open_cols:
                for r in range(1, height - 1):
                    out[r][c] = seam
            return tuple(tuple(row) for row in out)
        if self.graph == "alternating_row_fill":
            alt = values["alt"]
            out = [list(row) for row in grid]
            painted = 0
            for r in range(height):
                cols = [c for c in range(width) if grid[r][c] != background]
                if len(cols) != 1:
                    continue
                c0 = cols[0]
                color = grid[r][c0]
                for c in range(c0, width):
                    value = color if (c - c0) % 2 == 0 else alt
                    if out[r][c] != value:
                        out[r][c] = value
                        painted += 1
            if not painted:
                raise ValueError("nothing to fill")
            return tuple(tuple(row) for row in out)
        if self.graph == "triple_completion_pairs":
            pairs = set()
            for r in range(height):
                for c in range(width):
                    v = grid[r][c]
                    if v == background:
                        continue
                    for sig, offs in TRIPLE_OFFSETS.items():
                        if not all(
                            0 <= r + dr < height and 0 <= c + dc < width
                            for dr, dc in offs
                        ):
                            continue
                        vals = [grid[r + dr][c + dc] for dr, dc in offs]
                        if (
                            len(set(vals)) == 1
                            and vals[0] != background
                            and vals[0] != v
                        ):
                            pairs.add((v, vals[0], sig))
            if not pairs:
                raise ValueError("no triples")
            out = [list(row) for row in grid]
            for r in range(height):
                for c in range(width):
                    v = grid[r][c]
                    for center, flank, sig in pairs:
                        offs = TRIPLE_OFFSETS[sig]
                        if not all(
                            0 <= r + dr < height and 0 <= c + dc < width
                            for dr, dc in offs
                        ):
                            continue
                        vals = [grid[r + dr][c + dc] for dr, dc in offs]
                        if all(x == flank for x in vals) and v in (
                            background, center
                        ):
                            out[r][c] = center
                        elif v == center and all(
                            x == background for x in vals
                        ):
                            for dr, dc in offs:
                                out[r + dr][c + dc] = flank
            if out == [list(row) for row in grid]:
                raise ValueError("nothing to complete")
            return tuple(tuple(row) for row in out)
        if self.graph == "dash_bar_middle":
            out = [list(row) for row in grid]
            painted = 0
            for color, cells in _components4(grid, background):
                rows = {r for r, _ in cells}
                cols = {c for _, c in cells}
                if len(cells) != len(rows) * len(cols):
                    continue
                if len(rows) < 3 or len(rows) % 2 == 0 or len(cols) < 3:
                    continue
                mid = (min(rows) + max(rows)) // 2
                left = min(cols)
                for c in range(left + 1, left + len(cols), 2):
                    out[mid][c] = background
                    painted += 1
            if not painted:
                raise ValueError("no bar to dash")
            return tuple(tuple(row) for row in out)
        if self.graph == "block_distinct_extend_down":
            fill = values["fill"]
            blocks = []
            seen: set[tuple[int, int]] = set()
            for r in range(height - 1):
                for c in range(width - 1):
                    cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
                    if any(grid[rr][cc] == background for rr, cc in cells):
                        continue
                    if any((rr, cc) in seen for rr, cc in cells):
                        continue
                    blocks.append(cells)
                    seen.update(cells)
            if not blocks:
                raise ValueError("no 2x2 block")
            out = [list(row) for row in grid]
            painted = 0
            for cells in blocks:
                colors = {grid[rr][cc] for rr, cc in cells}
                top = min(rr for rr, _ in cells)
                bottom = max(rr for rr, _ in cells)
                left = min(cc for _, cc in cells)
                right = max(cc for _, cc in cells)
                for r in range(bottom + 1, bottom + 1 + len(colors)):
                    if r >= height:
                        raise ValueError("extension out of bounds")
                    for c in range(left, right + 1):
                        out[r][c] = fill
                        painted += 1
            if not painted:
                raise ValueError("nothing painted")
            return tuple(tuple(row) for row in out)
        if self.graph == "empty_window_fill":
            hole = values["hole"]
            fill = values["fill"]
            windows = [
                (r, c)
                for r in range(height - 2)
                for c in range(width - 2)
                if all(
                    grid[r + dr][c + dc] == hole
                    for dr in range(3)
                    for dc in range(3)
                )
            ]
            kept: list[tuple[int, int]] = []
            for r, c in windows:
                box = {
                    (r + dr, c + dc) for dr in range(3) for dc in range(3)
                }
                if any(
                    cell in box
                    for kr, kc in kept
                    for cell in (
                        (kr + dr, kc + dc)
                        for dr in range(3)
                        for dc in range(3)
                    )
                ):
                    continue
                kept.append((r, c))
            if not kept:
                raise ValueError("no empty window")
            out = [list(row) for row in grid]
            for r, c in kept:
                for dr in range(3):
                    for dc in range(3):
                        out[r + dr][c + dc] = fill
            return tuple(tuple(row) for row in out)
        if self.graph == "funnel_drip_streams":
            components = _components8(grid, background)
            if len(components) < 2:
                raise ValueError("layout mismatch")
            sizes = sorted(len(cells) for _, cells in components)
            if sizes[-1] == sizes[-2]:
                raise ValueError("body not unique")
            body = max(components, key=lambda item: len(item[1]))[1]
            dot_cells = [
                (r, c)
                for color, cells in components
                if cells is not body
                for r, c in cells
            ]
            dot_colors = {grid[r][c] for r, c in dot_cells}
            if len(dot_colors) != 1:
                raise ValueError("dot colors not uniform")
            dot = next(iter(dot_colors))
            if any(grid[r][c] == dot for r, c in body):
                raise ValueError("body uses dot color")
            top = min(r for r, _ in body)
            bottom = max(r for r, _ in body)
            left = min(c for _, c in body)
            right = max(c for _, c in body)
            stream_cols = sorted({
                c for r, c in dot_cells
                if r > bottom and left <= c <= right
            })
            if not stream_cols:
                raise ValueError("no dripping dots")
            out = [list(row) for row in grid]
            for c in stream_cols:
                start = bottom
                while start > top and grid[start - 1][c] == background:
                    start -= 1
                for r in range(start, height):
                    out[r][c] = dot
            return tuple(tuple(row) for row in out)
        if self.graph == "blob_defect_rays":
            cells = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if grid[r][c] != background
            ]
            if not cells:
                raise ValueError("no block")
            colors = {grid[r][c] for r, c in cells}
            if len(colors) != 2:
                raise ValueError("not a two-color block")
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells)
            left = min(c for _, c in cells)
            right = max(c for _, c in cells)
            outer = grid[top][left]
            inner = next(x for x in colors if x != outer)
            defect_cols = sorted({
                c for r, c in cells
                if grid[r][c] == inner
                and 0 < c < width - 1
                and grid[r][c - 1] == outer
                and grid[r][c + 1] == outer
            })
            if not defect_cols:
                raise ValueError("no defects")
            row_counts = Counter(
                r for r, c in cells if grid[r][c] == inner
            )
            best = max(row_counts.values())
            widest = [r for r, n in row_counts.items() if n == best]
            if len(widest) != 1:
                raise ValueError("widest inner row not unique")
            wrow = widest[0]
            out = [list(row) for row in grid]
            for c in defect_cols:
                for r in range(0, top):
                    out[r][c] = inner
                for r in range(top, bottom + 1):
                    out[r][c] = outer
                for r in range(bottom + 1, height):
                    out[r][c] = inner
            for c in range(0, left):
                out[wrow][c] = inner
            for c in range(left, right + 1):
                out[wrow][c] = outer
            for c in range(right + 1, width):
                out[wrow][c] = inner
            return tuple(tuple(row) for row in out)
        if self.graph == "band_span_closure":
            uniform_rows = Counter(
                grid[r][0] for r in range(height)
                if len(set(grid[r])) == 1
            )
            uniform_cols = Counter(
                grid[0][c] for c in range(width)
                if len({grid[r][c] for r in range(height)}) == 1
            )
            line_colors = [
                v for v in uniform_rows
                if uniform_rows[v] >= 2 and uniform_cols.get(v, 0) >= 2
            ]
            if len(line_colors) != 1:
                raise ValueError("line color ambiguous")
            line = line_colors[0]
            line_rows = [
                r for r in range(height)
                if grid[r][0] == line and len(set(grid[r])) == 1
            ]
            line_cols = [
                c for c in range(width)
                if grid[0][c] == line
                and len({grid[r][c] for r in range(height)}) == 1
            ]
            rbounds = [-1] + line_rows + [height]
            cbounds = [-1] + line_cols + [width]
            bands_r = [
                (rbounds[i] + 1, rbounds[i + 1])
                for i in range(len(rbounds) - 1)
                if rbounds[i + 1] > rbounds[i] + 1
            ]
            bands_c = [
                (cbounds[i] + 1, cbounds[i + 1])
                for i in range(len(cbounds) - 1)
                if cbounds[i + 1] > cbounds[i] + 1
            ]
            if not bands_r or not bands_c:
                raise ValueError("no bands")

            def interior(i, j):
                r0, r1 = bands_r[i]
                c0, c1 = bands_c[j]
                ir0 = r0 + (1 if r1 - r0 >= 3 else 0)
                ir1 = r1 - (1 if r1 - r0 >= 3 else 0)
                ic0 = c0 + (1 if c1 - c0 >= 3 else 0)
                ic1 = c1 - (1 if c1 - c0 >= 3 else 0)
                return ir0, ir1, ic0, ic1

            inner = Counter(
                grid[r][c]
                for i in range(len(bands_r))
                for j in range(len(bands_c))
                for r in range(interior(i, j)[0], interior(i, j)[1])
                for c in range(interior(i, j)[2], interior(i, j)[3])
            )
            inner.pop(line, None)
            if not inner:
                raise ValueError("no inner cells")
            empty = min(inner, key=lambda v: (-inner[v], v))

            def cell_color(i, j):
                ir0, ir1, ic0, ic1 = interior(i, j)
                colors = {
                    grid[r][c]
                    for r in range(ir0, ir1)
                    for c in range(ic0, ic1)
                } - {line, empty}
                if len(colors) == 1:
                    return next(iter(colors))
                return None

            positions: dict[int, set] = {}
            for i in range(len(bands_r)):
                for j in range(len(bands_c)):
                    color = cell_color(i, j)
                    if color is not None:
                        positions.setdefault(color, set()).add((i, j))
            row_spans = []
            col_spans = []
            col_span_info: dict[int, set] = {}
            row_span_info: dict[int, set] = {}
            for color, cells in positions.items():
                rows_b: dict[int, set] = {}
                cols_b: dict[int, set] = {}
                for i, j in cells:
                    rows_b.setdefault(i, set()).add(j)
                    cols_b.setdefault(j, set()).add(i)
                for j, rows_here in cols_b.items():
                    if len(rows_here) >= 2:
                        col_spans.append((j, min(rows_here),
                                          max(rows_here), color))
                        col_span_info.setdefault(j, set()).add(
                            (color, min(rows_here), max(rows_here)))
                for i, cols_here in rows_b.items():
                    if len(cols_here) >= 2:
                        row_spans.append((i, min(cols_here),
                                          max(cols_here), color))
                        row_span_info.setdefault(i, set()).add(
                            (color, min(cols_here), max(cols_here)))
            if not row_spans and not col_spans:
                raise ValueError("no spans")
            out = [list(row) for row in grid]
            for j, a, b, color in col_spans:
                c0, c1 = bands_c[j]
                for i in range(a, b + 1):
                    if any(
                        c2 == color and a2 <= j <= b2
                        for c2, a2, b2 in row_span_info.get(i, ())
                    ):
                        continue
                    r0, r1 = bands_r[i]
                    for r in range(r0, r1):
                        for c in range(c0, c1):
                            if out[r][c] == empty:
                                out[r][c] = color
            for i, a, b, color in row_spans:
                r0, r1 = bands_r[i]
                for j in range(a, b + 1):
                    if any(
                        c2 == color and a2 <= i <= b2
                        for c2, a2, b2 in col_span_info.get(j, ())
                    ):
                        continue
                    c0, c1 = bands_c[j]
                    for r in range(r0, r1):
                        for c in range(c0, c1):
                            if out[r][c] == empty:
                                out[r][c] = color
            if out == [list(row) for row in grid]:
                raise ValueError("nothing to close")
            return tuple(tuple(row) for row in out)
        if self.graph == "mover_rise_up":
            mover = values["mover"]
            out = [list(row) for row in grid]
            moved = 0
            for c in range(width):
                cells = sorted(
                    r for r in range(height) if grid[r][c] == mover
                )
                for r in cells:
                    out[r][c] = background
                for r in cells:
                    target = r
                    while target - 1 >= 0 and out[target - 1][c] == background:
                        target -= 1
                    out[target][c] = mover
                    if target != r:
                        moved += 1
            if not moved:
                raise ValueError("nothing rose")
            return tuple(tuple(row) for row in out)
        if self.graph == "column_bottom_half_mark":
            color = values["color"]
            mark = values["mark"]
            out = [list(row) for row in grid]
            painted = 0
            for col in range(width):
                start = None
                runs = []
                for r in range(height):
                    if grid[r][col] == color:
                        if start is None:
                            start = r
                    elif start is not None:
                        runs.append((start, r - 1))
                        start = None
                if start is not None:
                    runs.append((start, height - 1))
                for a, b in runs:
                    n = (b - a + 1) // 2
                    for r in range(b - n + 1, b + 1):
                        out[r][col] = mark
                        painted += 1
            if not painted:
                raise ValueError("nothing to mark")
            return tuple(tuple(row) for row in out)
        if self.graph == "slot_row_col_closure":
            uniform_rows = Counter(
                grid[r][0] for r in range(height)
                if len(set(grid[r])) == 1
            )
            uniform_cols = Counter(
                grid[0][c] for c in range(width)
                if len({grid[r][c] for r in range(height)}) == 1
            )
            blank_colors = [
                v for v in uniform_rows if v in uniform_cols
            ]
            if len(blank_colors) != 1:
                raise ValueError("blank color ambiguous")
            blank = blank_colors[0]
            blank_rows = [
                r for r in range(height)
                if all(cell == blank for cell in grid[r])
            ]
            blank_cols = [
                c for c in range(width)
                if all(grid[r][c] == blank for r in range(height))
            ]
            if not blank_rows or not blank_cols:
                raise ValueError("no blank bands")
            rbounds = [-1] + blank_rows + [height]
            cbounds = [-1] + blank_cols + [width]
            bands_r = [
                (rbounds[i] + 1, rbounds[i + 1])
                for i in range(len(rbounds) - 1)
                if rbounds[i + 1] > rbounds[i] + 1
            ]
            bands_c = [
                (cbounds[i] + 1, cbounds[i + 1])
                for i in range(len(cbounds) - 1)
                if cbounds[i + 1] > cbounds[i] + 1
            ]
            sizes = {
                (r1 - r0, c1 - c0)
                for r0, r1 in bands_r
                for c0, c1 in bands_c
            }
            if len(sizes) != 1:
                raise ValueError("irregular cells")
            occurrences: dict[int, list] = {}
            for i, (r0, r1) in enumerate(bands_r):
                for j, (c0, c1) in enumerate(bands_c):
                    cell_counts = Counter(
                        grid[r][c]
                        for r in range(r0, r1)
                        for c in range(c0, c1)
                    )
                    base = min(
                        cell_counts,
                        key=lambda v: (-cell_counts[v], v),
                    )
                    for r in range(r0, r1):
                        for c in range(c0, c1):
                            v = grid[r][c]
                            if v != base and v != blank:
                                occurrences.setdefault(v, []).append(
                                    (i, j, r - r0, c - c0)
                                )
            if not occurrences:
                raise ValueError("no special colors")
            out = [list(row) for row in grid]
            for v, cells in occurrences.items():
                row_counts = Counter(i for i, j, dr, dc in cells)
                col_counts = Counter(j for i, j, dr, dc in cells)
                fill_rows = set()
                fill_cols = set()
                if max(row_counts.values()) >= 2:
                    best = max(row_counts.values())
                    fill_rows.add(
                        min(i for i, n in row_counts.items() if n == best)
                    )
                if max(col_counts.values()) >= 2:
                    best = max(col_counts.values())
                    fill_cols.add(
                        min(j for j, n in col_counts.items() if n == best)
                    )
                row_slot = {}
                for i in fill_rows:
                    first = sorted(
                        (j, dr, dc)
                        for ii, j, dr, dc in cells
                        if ii == i
                    )[0]
                    row_slot[i] = (first[1], first[2])
                col_slot = {}
                for j in fill_cols:
                    first = sorted(
                        (
                            (i, dr, dc)
                            for ii, j2, dr, dc in cells
                            if j2 == j
                        ),
                        reverse=True,
                    )[0]
                    col_slot[j] = (first[1], first[2])
                for i in fill_rows:
                    for j in range(len(bands_c)):
                        dr, dc = row_slot[i]
                        out[bands_r[i][0] + dr][bands_c[j][0] + dc] = v
                for j in fill_cols:
                    for i in range(len(bands_r)):
                        if i in fill_rows:
                            continue
                        dr, dc = col_slot[j]
                        out[bands_r[i][0] + dr][bands_c[j][0] + dc] = v
            if out == [list(row) for row in grid]:
                raise ValueError("nothing to close")
            return tuple(tuple(row) for row in out)
        if self.graph == "marker_cluster_bbox_fill":
            marker = values["marker"]
            fill = values["fill"]
            cells = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if grid[r][c] == marker
            ]
            if not cells:
                raise ValueError("no markers")
            remaining = set(cells)
            comps = []
            while remaining:
                seed = min(remaining)
                comp = {seed}
                remaining.discard(seed)
                stack = [seed]
                while stack:
                    r, c = stack.pop()
                    for item in list(remaining):
                        if max(abs(item[0] - r), abs(item[1] - c)) <= 2:
                            remaining.discard(item)
                            comp.add(item)
                            stack.append(item)
                comps.append(comp)
            out = [list(row) for row in grid]
            painted = 0
            for comp in comps:
                top = min(r for r, _ in comp)
                bottom = max(r for r, _ in comp)
                left = min(c for _, c in comp)
                right = max(c for _, c in comp)
                if top == bottom and left == right:
                    continue
                for r in range(top, bottom + 1):
                    for c in range(left, right + 1):
                        if grid[r][c] != marker and out[r][c] != fill:
                            out[r][c] = fill
                            painted += 1
            if not painted:
                raise ValueError("nothing to fill")
            return tuple(tuple(row) for row in out)
        if self.graph == "template_cap_stamp":
            background = _background(grid)
            seen: set[tuple[int, int]] = set()
            comps: list[frozenset] = []
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
                    comps.append(frozenset(cells))
            templates = [
                cells for cells in comps
                if len({grid[r][c] for r, c in cells}) == 2
            ]
            if len(templates) != 1:
                raise ValueError("template not unique")
            template = templates[0]
            top = min(r for r, _ in template)
            left = min(c for _, c in template)
            other_cells = [
                (r, c) for cells in comps for r, c in cells
                if cells is not template
            ]
            tcolors = {grid[r][c] for r, c in template}
            caps = [
                value for value in tcolors
                if any(grid[r][c] == value for r, c in other_cells)
            ]
            if len(caps) != 1:
                raise ValueError("cap color ambiguous")
            cap = caps[0]
            body_color = (tcolors - {cap}).pop()
            cap_offsets = sorted(
                (r - top, c - left) for r, c in template if grid[r][c] == cap
            )
            body_offsets = sorted(
                (r - top, c - left) for r, c in template if grid[r][c] != cap
            )
            anchors = set()
            for cells in comps:
                if cells is template:
                    continue
                if not all(grid[r][c] == cap for r, c in cells):
                    continue
                side = int(len(cells) ** 0.5)
                if side * side != len(cells):
                    continue
                key_top = min(r for r, _ in cells)
                key_left = min(c for _, c in cells)
                for off_r, off_c in cap_offsets:
                    anchors.add(
                        (key_top - side * off_r,
                         key_left - side * off_c, side)
                    )
            out = [list(row) for row in grid]
            stamped = False
            for anchor_top, anchor_left, side in sorted(anchors):
                valid = True
                for off_r, off_c in cap_offsets:
                    for dr in range(side):
                        for dc in range(side):
                            r = anchor_top + side * off_r + dr
                            c = anchor_left + side * off_c + dc
                            if not (
                                0 <= r < height and 0 <= c < width
                                and grid[r][c] == cap
                            ):
                                valid = False
                if not valid:
                    continue
                for body_r, body_c in body_offsets:
                    for dr in range(side):
                        for dc in range(side):
                            r = anchor_top + side * body_r + dr
                            c = anchor_left + side * body_c + dc
                            if 0 <= r < height and 0 <= c < width:
                                out[r][c] = body_color
                                stamped = True
            if not stamped:
                raise ValueError("nothing stamped")
            return tuple(tuple(row) for row in out)
        if self.graph == "hole_cluster_glyph_stamp":
            background = _background(grid)
            components = _components4(grid, background)
            if not components:
                raise ValueError("no components")
            rect_color, rect = max(components, key=lambda item: len(item[1]))
            top = min(r for r, _ in rect)
            bottom = max(r for r, _ in rect)
            left = min(c for _, c in rect)
            right = max(c for _, c in rect)
            holes = {
                (r, c)
                for r in range(top, bottom + 1)
                for c in range(left, right + 1)
                if grid[r][c] != rect_color
            }
            outside = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if grid[r][c] != background
                and not (top <= r <= bottom and left <= c <= right)
            ]
            glyph_grids = []
            remaining = set(outside)
            while remaining:
                seed = min(remaining)
                remaining.discard(seed)
                comp = {seed}
                stack = [seed]
                while stack:
                    r, c = stack.pop()
                    for item in list(remaining):
                        if max(abs(item[0] - r), abs(item[1] - c)) <= 1:
                            remaining.discard(item)
                            comp.add(item)
                            stack.append(item)
                g_top = min(r for r, _ in comp)
                g_left = min(c for _, c in comp)
                g_bottom = max(r for r, _ in comp)
                g_right = max(c for _, c in comp)
                glyph_grids.append(tuple(
                    tuple(
                        grid[r][c] if (r, c) in comp else background
                        for c in range(g_left, g_right + 1)
                    )
                    for r in range(g_top, g_bottom + 1)
                ))
            if not glyph_grids or not holes:
                raise ValueError("no glyphs or holes")
            clusters = []
            remaining = set(holes)
            while remaining:
                seed = min(remaining)
                remaining.discard(seed)
                comp = {seed}
                stack = [seed]
                while stack:
                    r, c = stack.pop()
                    for item in list(remaining):
                        if max(abs(item[0] - r), abs(item[1] - c)) <= 1:
                            remaining.discard(item)
                            comp.add(item)
                            stack.append(item)
                clusters.append(comp)
            out_h = bottom - top + 1
            out_w = right - left + 1
            out = [[rect_color] * out_w for _ in range(out_h)]
            stamped = False
            for cluster in sorted(clusters, key=min):
                cl_top = min(r for r, _ in cluster)
                cl_left = min(c for _, c in cluster)
                shape = {(r - cl_top, c - cl_left) for r, c in cluster}
                matches = []
                for index, glyph in enumerate(glyph_grids):
                    rotated = glyph
                    seen_backbones = set()
                    for _ in range(4):
                        backbone = [
                            (r, c)
                            for r, row in enumerate(rotated)
                            for c, value in enumerate(row)
                            if value == rect_color
                        ]
                        if backbone:
                            r_min = min(r for r, _ in backbone)
                            c_min = min(c for _, c in backbone)
                            normalized = frozenset(
                                (r - r_min, c - c_min) for r, c in backbone
                            )
                            if normalized not in seen_backbones:
                                seen_backbones.add(normalized)
                                if normalized <= shape:
                                    matches.append(
                                        (len(normalized), index, rotated,
                                         r_min, c_min)
                                    )
                        rotated = tuple(
                            tuple(
                                rotated[len(rotated) - 1 - c][r]
                                for c in range(len(rotated))
                            )
                            for r in range(len(rotated[0]))
                        )
                if not matches:
                    continue
                best_size = max(size for size, *_ in matches)
                best = [m for m in matches if m[0] == best_size]
                if len(best) != 1:
                    raise ValueError("ambiguous glyph match")
                _, _, glyph, r_min, c_min = best[0]
                for r, row in enumerate(glyph):
                    for c, value in enumerate(row):
                        if value == background:
                            continue
                        tr = cl_top - top + r - r_min
                        tc = cl_left - left + c - c_min
                        if not (0 <= tr < out_h and 0 <= tc < out_w):
                            raise ValueError("stamp off canvas")
                        out[tr][tc] = value
                        stamped = True
            if not stamped:
                raise ValueError("nothing stamped")
            return tuple(tuple(row) for row in out)
        if self.graph == "anchor_shape_flood":
            background = _background(grid)
            row_runs = Counter(
                grid[r][0] for r in range(height)
                if len(set(grid[r])) == 1
            )
            col_runs = Counter(
                grid[0][c] for c in range(width)
                if len({grid[r][c] for r in range(height)}) == 1
            )
            walls = [
                value for value in set(row_runs) & set(col_runs)
                if row_runs[value] >= 2 and col_runs[value] >= 2
                and value != background
            ]
            if len(walls) != 1:
                raise ValueError("wall ambiguous")
            wall = walls[0]
            sep_rows = [
                r for r in range(height)
                if all(value == wall for value in grid[r])
            ]
            sep_cols = [
                c for c in range(width)
                if all(grid[r][c] == wall for r in range(height))
            ]
            row_breaks = [-1] + sep_rows + [height]
            col_breaks = [-1] + sep_cols + [width]
            row_bands = [
                (row_breaks[i] + 1, row_breaks[i + 1])
                for i in range(len(row_breaks) - 1)
                if row_breaks[i + 1] - row_breaks[i] - 1 > 0
            ]
            col_bands = [
                (col_breaks[i] + 1, col_breaks[i + 1])
                for i in range(len(col_breaks) - 1)
                if col_breaks[i + 1] - col_breaks[i] - 1 > 0
            ]
            if len({b - a for a, b in row_bands}) != 1 \
                    or len({b - a for a, b in col_bands}) != 1:
                raise ValueError("uneven bands")
            markers: dict[tuple[int, int], frozenset] = {}
            for i, (r0, r1) in enumerate(row_bands):
                for j, (c0, c1) in enumerate(col_bands):
                    cells = frozenset(
                        (r - r0, c - c0)
                        for r in range(r0, r1)
                        for c in range(c0, c1)
                        if grid[r][c] not in (background, wall)
                    )
                    if cells:
                        colors = {grid[r0 + dr][c0 + dc] for dr, dc in cells}
                        if len(colors) > 1:
                            raise ValueError("two marker colors in cell")
                        markers[i, j] = cells
            if not markers:
                raise ValueError("no marker")
            anchor_i, anchor_j = min(markers)
            anchor = markers[anchor_i, anchor_j]
            out = [list(row) for row in grid]
            stamped = False
            for i, (r0, r1) in enumerate(row_bands):
                for j, (c0, c1) in enumerate(col_bands):
                    if (i, j) == (anchor_i, anchor_j):
                        continue
                    own = markers.get((i, j), frozenset())
                    for dr, dc in sorted(anchor - own):
                        r, c = r0 + dr, c0 + dc
                        if grid[r][c] == background:
                            out[r][c] = wall
                            stamped = True
            if not stamped:
                raise ValueError("nothing stamped")
            return tuple(tuple(row) for row in out)
        if self.graph == "maximal_rect_interior_fill":
            fill = 4
            if any(fill == cell for row in grid for cell in row):
                raise ValueError("fill color already present")
            hmax = {}
            for r in range(height):
                c = 0
                while c < width:
                    if grid[r][c] != background:
                        end = c
                        while end + 1 < width and grid[r][end + 1] != background:
                            end += 1
                        for cc in range(c, end + 1):
                            hmax[(r, cc)] = (c, end)
                        c = end + 1
                    else:
                        c += 1
            vmax = {}
            for c in range(width):
                r = 0
                while r < height:
                    if grid[r][c] != background:
                        end = r
                        while end + 1 < height and grid[end + 1][c] != background:
                            end += 1
                        for rr in range(r, end + 1):
                            vmax[(rr, c)] = (r, end)
                        r = end + 1
                    else:
                        r += 1
            painted = set()
            for r1 in range(-1, height):
                for r2 in range(r1 + 2, height + 1):
                    for c1 in range(-1, width):
                        for c2 in range(c1 + 2, width + 1):
                            edges = (r1 == -1, r2 == height, c1 == -1, c2 == width)
                            if sum(edges) >= 2:
                                continue
                            if not any(edges):
                                ok = (hmax.get((r1, c1)) == (c1, c2)
                                      and hmax.get((r2, c1)) == (c1, c2)
                                      and vmax.get((r1, c1)) == (r1, r2)
                                      and vmax.get((r1, c2)) == (r1, r2))
                            elif edges[0]:
                                ok = (hmax.get((r2, c1)) == (c1, c2)
                                      and vmax.get((r2, c1)) == (0, r2)
                                      and vmax.get((r2, c2)) == (0, r2))
                            elif edges[1]:
                                ok = (hmax.get((r1, c1)) == (c1, c2)
                                      and vmax.get((r1, c1)) == (r1, height - 1)
                                      and vmax.get((r1, c2)) == (r1, height - 1))
                            elif edges[2]:
                                ok = (vmax.get((r1, c2)) == (r1, r2)
                                      and hmax.get((r1, c2)) == (0, c2)
                                      and hmax.get((r2, c2)) == (0, c2))
                            else:
                                ok = (vmax.get((r1, c1)) == (r1, r2)
                                      and hmax.get((r1, c1)) == (c1, width - 1)
                                      and hmax.get((r2, c1)) == (c1, width - 1))
                            if not ok:
                                continue
                            for r in range(max(0, r1 + 1), min(height, r2)):
                                for c in range(max(0, c1 + 1), min(width, c2)):
                                    if grid[r][c] == background:
                                        painted.add((r, c))
            if not painted:
                raise ValueError("no maximal rectangle interior")
            out = [list(row) for row in grid]
            for r, c in sorted(painted):
                out[r][c] = fill
            return tuple(tuple(row) for row in out)
        if self.graph == "glyph_marker_dihedral_stamp":
            seen: set[tuple[int, int]] = set()
            regions = []
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
                    regions.append(frozenset(cells))
            glyphs = []
            for cells in regions:
                colors = Counter(grid[r][c] for r, c in cells)
                if len(colors) >= 3:
                    glyphs.append((cells, colors))
            if len(glyphs) != 1:
                raise ValueError("glyph ambiguous")
            glyph_cells, colors = glyphs[0]
            fill = max(sorted(colors), key=lambda v: colors[v])
            markers = sorted(
                color for color, count in colors.items()
                if count == 1 and color != fill
            )
            if len(markers) != 2:
                raise ValueError("marker count")
            grow = [r for r, _ in glyph_cells]
            gcol = [c for _, c in glyph_cells]
            r0, c0 = min(grow), min(gcol)
            if max(grow) - r0 != max(gcol) - c0:
                raise ValueError("glyph not square")
            n = max(grow) - r0 + 1
            marker_pos = {}
            for color in markers:
                pos = [(r - r0, c - c0) for r, c in glyph_cells
                       if grid[r][c] == color]
                if len(pos) != 1:
                    raise ValueError("marker cells")
                marker_pos[color] = pos[0]
            fill_cells = sorted((r - r0, c - c0) for r, c in glyph_cells
                                if grid[r][c] == fill)
            blocks: dict[int, list[tuple[int, int, int]]] = {m: [] for m in markers}
            for color, cells in _components4(grid, background):
                if color not in blocks or cells & glyph_cells:
                    continue
                rs = [r for r, _ in cells]
                cs = [c for _, c in cells]
                if max(rs) - min(rs) != max(cs) - min(cs):
                    continue
                k = max(rs) - min(rs) + 1
                if len(cells) != k * k:
                    continue
                blocks[color].append((min(rs), min(cs), k))
            def transform_list(size):
                return [
                    lambda r, c: (r, c),
                    lambda r, c: (c, size - 1 - r),
                    lambda r, c: (size - 1 - r, size - 1 - c),
                    lambda r, c: (size - 1 - c, r),
                    lambda r, c: (r, size - 1 - c),
                    lambda r, c: (size - 1 - r, c),
                    lambda r, c: (c, r),
                    lambda r, c: (size - 1 - c, size - 1 - r),
                ]
            stamps = []
            for ax, ay, ka in sorted(blocks[markers[0]]):
                for bx, by, kb in sorted(blocks[markers[1]]):
                    if ka != kb:
                        continue
                    pa = marker_pos[markers[0]]
                    pb = marker_pos[markers[1]]
                    pair_stamps = set()
                    for transform in transform_list(n):
                        ta = transform(*pa)
                        tb = transform(*pb)
                        ar, ac = ax - ka * ta[0], ay - ka * ta[1]
                        if (ar + ka * tb[0], ac + ka * tb[1]) != (bx, by):
                            continue
                        cells = set()
                        for fr, fc in fill_cells:
                            tr, tc = transform(fr, fc)
                            br, bc = ar + ka * tr, ac + ka * tc
                            for dr in range(ka):
                                for dc in range(ka):
                                    cells.add((br + dr, bc + dc))
                        pair_stamps.add(frozenset(cells))
                    if len(pair_stamps) > 1:
                        raise ValueError("ambiguous transform")
                    if pair_stamps:
                        stamps.append(next(iter(pair_stamps)))
            if not stamps:
                raise ValueError("no stamp pair")
            target: set[tuple[int, int]] = set()
            for cells in stamps:
                for r, c in cells:
                    if not (0 <= r < height and 0 <= c < width):
                        raise ValueError("stamp out of bounds")
                    if grid[r][c] != background or (r, c) in target:
                        raise ValueError("stamp collides")
                    target.add((r, c))
            out = [list(row) for row in grid]
            for r, c in sorted(target):
                out[r][c] = fill
            return tuple(tuple(row) for row in out)
        if self.graph == "maximal_zero_rect_paint":
            fill = 2
            if any(fill == cell for row in grid for cell in row):
                raise ValueError("fill color already present")

            def maximal_rects(color):
                heights = [0] * width
                found = set()
                for bottom in range(height):
                    for c in range(width):
                        heights[c] = (
                            heights[c] + 1
                            if grid[bottom][c] == color else 0
                        )
                    for left in range(width):
                        if heights[left] < 2:
                            continue
                        for right in range(left + 1, width):
                            if heights[right] < 2:
                                break
                            htop = min(heights[left:right + 1])
                            if htop < 2:
                                continue
                            top = bottom - htop + 1
                            if (
                                left > 0
                                and min(heights[left - 1:right + 1]) >= htop
                            ):
                                continue
                            if (
                                right + 1 < width
                                and min(heights[left:right + 2]) >= htop
                            ):
                                continue
                            if top > 0 and all(
                                grid[top - 1][c] == color
                                for c in range(left, right + 1)
                            ):
                                continue
                            if bottom + 1 < height and all(
                                grid[bottom + 1][c] == color
                                for c in range(left, right + 1)
                            ):
                                continue
                            found.add((top, left, bottom, right))
                return found

            rect_sets = {}
            counts = Counter(cell for row in grid for cell in row)
            for color in sorted(counts):
                if color == fill:
                    continue
                items = maximal_rects(color)
                if items:
                    rect_sets[color] = items
            if not rect_sets:
                raise ValueError("no rectangle color")
            rect_color = min(rect_sets, key=lambda c: (counts[c], c))
            rects = rect_sets[rect_color]
            ordered = sorted(rects)
            parent = list(range(len(ordered)))

            def find(index: int) -> int:
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    a, b = ordered[i], ordered[j]
                    if not (a[2] < b[0] or b[2] < a[0]
                            or a[3] < b[1] or b[3] < a[1]):
                        union(i, j)
            best: dict[int, tuple[int, tuple[int, int, int, int]]] = {}
            for i, item in enumerate(ordered):
                area = (item[2] - item[0] + 1) * (item[3] - item[1] + 1)
                key = find(i)
                if key not in best or (-area, item) < best[key]:
                    best[key] = (-area, item)
            if not best:
                raise ValueError("no zero rectangle")
            out = [list(row) for row in grid]
            for _, item in sorted(best.values(), key=lambda pair: pair[1]):
                for r in range(item[0], item[2] + 1):
                    for c in range(item[1], item[3] + 1):
                        out[r][c] = fill
            return tuple(tuple(row) for row in out)
        if self.graph == "core_blade_reflection_stamp":
            seen: set[tuple[int, int]] = set()
            components = []
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
                    components.append(frozenset(cells))
            out = [list(row) for row in grid]
            stamped = False
            for comp in sorted(components, key=min):
                rows = [r for r, _ in comp]
                cols = [c for _, c in comp]
                decompositions = []
                for r0 in range(min(rows) - 1, max(rows) + 1):
                    for c0 in range(min(cols) - 1, max(cols) + 1):
                        window = ((r0, c0), (r0, c0 + 1),
                                  (r0 + 1, c0), (r0 + 1, c0 + 1))
                        core_cells = {pos for pos in window if pos in comp}
                        if not core_cells:
                            continue
                        blade = comp - core_cells
                        if not blade:
                            continue
                        values = {grid[r][c] for r, c in blade}
                        if len(values) != 1:
                            continue
                        value = next(iter(values))
                        sources = []
                        for pos in window:
                            if not (0 <= pos[0] < height
                                    and 0 <= pos[1] < width):
                                continue
                            if grid[pos[0]][pos[1]] != value:
                                continue
                            if any(
                                abs(br - pos[0]) <= 1 and abs(bc - pos[1]) <= 1
                                for br, bc in blade
                            ):
                                sources.append(pos)
                        if len(sources) != 1:
                            continue
                        core = {
                            pos: grid[pos[0]][pos[1]]
                            for pos in window
                            if 0 <= pos[0] < height and 0 <= pos[1] < width
                        }
                        decompositions.append(
                            (r0, c0, sources[0], blade, core)
                        )
                if len(decompositions) != 1:
                    raise ValueError("decomposition ambiguous")
                r0, c0, source, blade, core = decompositions[0]
                sr, sc = source
                for (rt, ct), corner in sorted(core.items()):
                    if (rt, ct) == source or corner == background:
                        continue
                    for br, bc in sorted(blade):
                        nr = br if rt == sr else 2 * r0 + 1 - br
                        nc = bc if ct == sc else 2 * c0 + 1 - bc
                        if not (0 <= nr < height and 0 <= nc < width):
                            raise ValueError("stamp out of bounds")
                        if out[nr][nc] not in (background, corner):
                            raise ValueError("stamp collides")
                        if out[nr][nc] == background:
                            stamped = True
                        out[nr][nc] = corner
            if not stamped:
                raise ValueError("nothing stamped")
            return tuple(tuple(row) for row in out)
        if self.graph == "edge_beam_marker_step":
            emitter_color, marker_color = 8, 2
            present = {cell for row in grid for cell in row}
            if present != {background, emitter_color, marker_color}:
                raise ValueError("unexpected colors")
            emitters = {(r, c) for r in range(height) for c in range(width)
                        if grid[r][c] == emitter_color}
            markers = {(r, c) for r in range(height) for c in range(width)
                       if grid[r][c] == marker_color}
            if not emitters or not markers:
                raise ValueError("missing emitters or markers")
            top = {(0, c) for c in range(width)}
            bottom = {(height - 1, c) for c in range(width)}
            left = {(r, 0) for r in range(height)}
            right = {(r, width - 1) for r in range(height)}
            if emitters <= top:
                orientation, low = "v", True
            elif emitters <= bottom:
                orientation, low = "v", False
            elif emitters <= left:
                orientation, low = "h", True
            elif emitters <= right:
                orientation, low = "h", False
            else:
                raise ValueError("emitters not on one edge")
            if orientation == "v":
                candidates = (left, right)
                lines = sorted({r for r, _ in markers})
            else:
                candidates = (top, bottom)
                lines = sorted({c for _, c in markers})
            if not (markers <= candidates[0] or markers <= candidates[1]):
                raise ValueError("markers not on one parallel edge")
            if orientation == "v":
                shift = -1 if markers <= right else 1
            else:
                shift = -1 if markers <= bottom else 1
            bounds = height if orientation == "v" else width
            if low:
                segments = []
                start = 0
                for line in lines:
                    segments.append((start, line - 1))
                    start = line
                segments.append((start, bounds - 1))
            else:
                segments = []
                start = bounds - 1
                for line in reversed(lines):
                    segments.append((line + 1, start))
                    start = line
                segments.append((0, start))
            out = [list(row) for row in grid]
            added = False
            for index, (seg_start, seg_end) in enumerate(segments):
                offset = shift * index
                for er, ec in sorted(emitters):
                    lane = ec if orientation == "v" else er
                    lane2 = lane + offset
                    if not (0 <= lane2 < (width if orientation == "v"
                                          else height)):
                        continue
                    for along in range(seg_start, seg_end + 1):
                        r, c = ((along, lane2) if orientation == "v"
                                else (lane2, along))
                        if out[r][c] not in (background, emitter_color):
                            raise ValueError("beam collides")
                        if out[r][c] == background:
                            added = True
                        out[r][c] = emitter_color
            if not added:
                raise ValueError("no beam painted")
            return tuple(tuple(row) for row in out)
        raise ValueError("unknown a2 paint graph")


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
        value
        for source, expected in demos
        for grid in (source, expected)
        for row in grid
        for value in row
    })

    for direction in DIRECTIONS:
        add(Candidate("rays_from_singletons",
                      (("direction", direction),)))

    for mark in colors:
        for blank in colors:
            add(Candidate(
                "uniform_rows_canvas",
                tuple(sorted({"background": blank,
                              "mark": mark}.items())),
            ))



    # component_size_recolor
    size_recolor_emitted = False
    for background_color in colors:
        mapping, feasible = {}, True
        default = None
        for source, expected in demos:
            if _shape_diff(source, expected):
                feasible = False
                break
            background = _background(source)
            for color, cells in _components4(source, background):
                got = expected[next(iter(cells))[0]][next(iter(cells))[1]]
                size = len(cells)
                if size in mapping:
                    if mapping[size] != got:
                        feasible = False
                        break
                else:
                    mapping[size] = got
            if not feasible:
                break
        distinct = set(mapping.values())
        if feasible and mapping and len(distinct) == 2:
            counts = Counter(mapping.values())
            default = counts.most_common(1)[0][0]
            exceptions = [
                size for size, value in mapping.items() if value != default
            ]
            if len(exceptions) != 1:
                feasible = False
        if feasible and mapping and len(distinct) == 2:
            add(Candidate(
                "component_size_recolor",
                tuple(sorted({
                    "background_color": background_color,
                    "default": default,
                    "mapping": tuple(sorted(mapping.items())),
                }.items())),
            ))
            size_recolor_emitted = True

    # hole_size_recolor: mapping taglie dai demo; il default e' il colore
    # che il background dell'input prende nell'output (osservato).
    default_colors = set()
    mapping, feasible = {}, True
    for source, expected in demos:
        if _shape_diff(source, expected):
            feasible = False
            break
        background = _background(source)
        background_outputs = {
            expected[r][c]
            for r in range(len(source))
            for c in range(len(source[0]))
            if source[r][c] == background
        }
        if len(background_outputs) != 1:
            feasible = False
            break
        default_colors.add(next(iter(background_outputs)))
        components = _components4(source, background)
        if not components:
            feasible = False
            break
        for _, cells in components:
            got = expected[next(iter(cells))[0]][next(iter(cells))[1]]
            size = len(cells)
            if mapping.get(size, got) != got:
                feasible = False
                break
            mapping[size] = got
        if not feasible:
            break
    hole_feasible = bool(feasible and mapping and len(default_colors) == 1)
    if hole_feasible:
        default = next(iter(default_colors))
        full = {
            size: color for size, color in mapping.items()
            if color != default
        }
        if full and not size_recolor_emitted:
            add(Candidate(
                "hole_size_recolor",
                tuple(sorted({"default": default,
                              "mapping": tuple(sorted(full.items()))}.items())),
            ))
    # modulo_recolor
    for axis in ("column", "row"):
        for modulus in (2, 3, 4):
            for residue in range(modulus):
                for source in colors:
                    for mark in colors:
                        add(Candidate(
                            "modulo_recolor",
                            tuple(sorted({
                                "axis": axis, "mark": mark,
                                "modulus": modulus, "residue": residue,
                                "source": source,
                            }.items())),
                        ))

    for mark in colors:
        add(Candidate("fill_inbox_components", (("mark", mark),)))
        add(Candidate("fill_outbox_all", (("mark", mark),)))

    panel_mapping, panel_feasible = {}, True
    for source, expected in demos:
        if _shape_diff(source, expected):
            panel_feasible = False
            break
        background = _background(source)
        sep_rows, sep_cols = _separator_lines(source, background)
        if not sep_rows and not sep_cols:
            panel_feasible = False
            break
        for r0, r1 in _segments(len(source), sep_rows):
            for c0, c1 in _segments(len(source[0]), sep_cols):
                sparse = {
                    source[r][c]
                    for r in range(r0, r1)
                    for c in range(c0, c1)
                    if source[r][c] != background
                }
                fills = {
                    expected[r][c]
                    for r in range(r0, r1)
                    for c in range(c0, c1)
                }
                if len(sparse) != 1 or len(fills) != 1:
                    panel_feasible = False
                    break
                key, fill = next(iter(sparse)), next(iter(fills))
                if panel_mapping.get(key, fill) != fill:
                    panel_feasible = False
                    break
                panel_mapping[key] = fill
            if not panel_feasible:
                break
    if panel_feasible and panel_mapping:
        add(Candidate(
            "panel_sparse_map_fill",
            (("mapping", tuple(sorted(panel_mapping.items()))),),
        ))

    out_colors = sorted({
        value for _, expected in demos for row in expected for value in row
    })
    for seam in out_colors:
        add(Candidate("open_line_seam_fill", (("seam", seam),)))

    add(Candidate("dash_bar_middle", ()))

    for fill in out_colors:
        add(Candidate("block_distinct_extend_down", (("fill", fill),)))

    window_in_colors = sorted({
        value for source, _ in demos for row in source for value in row
    })
    for hole in window_in_colors:
        for fill in out_colors:
            if hole != fill:
                add(Candidate(
                    "empty_window_fill",
                    tuple(sorted({"fill": fill, "hole": hole}.items())),
                ))

    for alt in colors:
        add(Candidate("alternating_row_fill", (("alt", alt),)))

    add(Candidate("triple_completion_pairs", ()))
    add(Candidate("slot_row_col_closure", ()))

    add(Candidate("funnel_drip_streams", ()))
    add(Candidate("blob_defect_rays", ()))
    add(Candidate("band_span_closure", ()))
    add(Candidate("template_cap_stamp", ()))
    add(Candidate("hole_cluster_glyph_stamp", ()))
    add(Candidate("anchor_shape_flood", ()))
    add(Candidate("maximal_rect_interior_fill", ()))
    add(Candidate("glyph_marker_dihedral_stamp", ()))
    add(Candidate("maximal_zero_rect_paint", ()))
    add(Candidate("core_blade_reflection_stamp", ()))
    add(Candidate("edge_beam_marker_step", ()))
    for mover in colors:
        add(Candidate("mover_rise_up", (("mover", mover),)))

    for color in colors:
        for mark in out_colors:
            if color != mark:
                add(Candidate(
                    "column_bottom_half_mark",
                    tuple(sorted({"color": color, "mark": mark}.items())),
                ))

    for marker in colors:
        for fill in out_colors:
            if marker != fill:
                add(Candidate(
                    "marker_cluster_bbox_fill",
                    tuple(sorted({"fill": fill, "marker": marker}.items())),
                ))

    return tuple(sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings))))


def _shape_diff(source: Grid, expected: Grid) -> bool:
    return len(source) != len(expected) or len(source[0]) != len(expected[0])


def _bg_regions(grid: Grid, background: int) -> list[frozenset]:
    height, width = len(grid), len(grid[0])
    regions: list[frozenset] = []
    seen: set[tuple[int, int]] = set()
    for r in range(height):
        for c in range(width):
            if grid[r][c] != background or (r, c) in seen:
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
                        and grid[nr][nc] == background
                    ):
                        cells.add((nr, nc))
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            regions.append(frozenset(cells))
    return regions
