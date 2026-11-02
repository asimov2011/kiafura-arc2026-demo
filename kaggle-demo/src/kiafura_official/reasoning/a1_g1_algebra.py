"""A1 gruppo 1: algebre parametriche con candidati derivati dai soli demo.

Grafi (parametrici per classe di trasformazione, nessun letterale di
griglia o identita' di task; i binding — colori, fattori, palette — sono
derivati dalle coppie train):
  - period_extend_vertical_recolor: la sequenza di righe e' estesa col suo
    periodo minimo a un'altezza multipla (fattore razionale derivato dai
    demo) e ricolorata con la mappa derivata dalla sovrapposizione;
  - mode_block_tile: l'output e' una griglia di blocchi (uno per cella
    dell'input, ciascuno grande quanto l'input): le celle del colore
    modale dell'input diventano copie dell'input, le altre sono riempite
    col colore di riempimento derivato dai demo;
  - marker_shared_component_classify: classificazione 1x1: due blocchi
    2x2 del colore marcatore; se entrambi toccano la stessa componente
    (8-connessa) del colore compagno l'output e' il compagno, altrimenti
    lo sfondo;
  - height_rank_recolor: le barre verticali del colore sorgente (run che
    toccano il bordo inferiore) sono ricolorate col rango per altezza
    decrescente; palette rango->colore derivata dai demo;
  - periodic_line_fill: due o piu' marcatori equispaziati lungo l'asse
    lungo definiscono un ciclo di colori; le linee complete (righe o
    colonne) sono riempite periodicamente in avanti dal primo marcatore;
  - dihedral_closure: chiusura delle celle colorate sotto le riflessioni
    rispetto al centro del bounding box (simmetria verticale e
    orizzontale);
  - diagonal_pair_connect: ogni colore con esattamente due celle allineate
    in diagonale e' completato col segmento diagonale tra le due celle;
  - shape_into_frame_hole: ogni regione di sfondo racchiusa (buca) riceve
    per traslazione l'unica componente libera il cui bounding box
    coincide con quello della buca;
  - marker_to_line_attract: linee complete (righe o colonne) monocrome
    attirano le celle libere dello stesso colore, che si fermano
    adiacenti alla linea piu' vicina; le celle senza linea spariscono;
  - endpoint_pair_row_fill: ogni riga con estremi uguali non-sfondo e
    interno tutto sfondo e' riempita col colore degli estremi;
  - directional_object_stamp: l'oggetto piu' grande e' stampato a copie
    traslate nelle direzioni indicate dalle componenti marcatore (passo =
    dimensione bbox + 1), ricolorato col colore del marcatore; ogni
    marcatore deve cadere nella prima copia.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import gcd
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]
EXEC_ERRORS = (ValueError, IndexError, KeyError, StopIteration)
STAMP_CAP = 64


def _rectangular(grid: Grid) -> None:
    if not grid or not grid[0]:
        raise ValueError("empty grid")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("ragged grid")


def _background(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    top = max(counts.values())
    modes = sorted(value for value, n in counts.items() if n == top)
    if len(modes) != 1:
        raise ValueError("ambiguous background")
    return modes[0]


def _counts(grid: Grid) -> Counter:
    return Counter(value for row in grid for value in row)


def _neighbors4(height: int, width: int, r: int, c: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _components4(cells) -> list[frozenset]:
    """4-connected components of a set of cells, sorted deterministically."""
    remaining = set(cells)
    found = []
    for start in sorted(cells):
        if start not in remaining:
            continue
        stack = [start]
        component = set()
        while stack:
            cell = stack.pop()
            if cell in component:
                continue
            component.add(cell)
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining and (nr, nc) not in component:
                    stack.append((nr, nc))
        remaining -= component
        found.append(frozenset(component))
    return sorted(found, key=min)


def _value_components(grid: Grid, skip: int | None = None) -> list[tuple[int, frozenset]]:
    """Same-value 4-connected components, sorted; skip is excluded."""
    height, width = len(grid), len(grid[0])
    out = []
    for value in sorted({cell for row in grid for cell in row}):
        if value == skip:
            continue
        cells = {
            (r, c)
            for r in range(height)
            for c in range(width)
            if grid[r][c] == value
        }
        for component in _components4(cells):
            out.append((value, component))
    return sorted(out, key=lambda item: (min(item[1]), item[0]))


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _unique_mode(grid: Grid) -> int:
    counts = _counts(grid)
    top = max(counts.values())
    modes = [value for value, n in counts.items() if n == top]
    if len(modes) != 1:
        raise ValueError("ambiguous mode")
    return modes[0]


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]
    _cost: int = 1

    def cost(self, grid: Grid) -> int:
        return self._cost + len(grid) * len(grid[0])

    def execute(self, grid: Grid) -> Grid:
        _rectangular(grid)
        values = dict(self.bindings)
        if self.graph == "period_extend_vertical_recolor":
            return _period_extend_vertical_recolor(
                grid, values["extend_num"], values["extend_den"],
                dict(values["recolor"]),
            )
        if self.graph == "mode_block_tile":
            return _mode_block_tile(grid, values["fill"])
        if self.graph == "marker_shared_component_classify":
            return _marker_shared_component_classify(
                grid, values["marker"], values["companion"],
                values["background"],
            )
        if self.graph == "height_rank_recolor":
            return _height_rank_recolor(
                grid, values["source"], values["background"],
                values["palette"],
            )
        if self.graph == "periodic_line_fill":
            return _periodic_line_fill(grid, values["background"])
        if self.graph == "dihedral_closure":
            return _dihedral_closure(grid, values["background"])
        if self.graph == "dominant_centroid_dihedral":
            return _dominant_centroid_dihedral(grid, values["background"])
        if self.graph == "diagonal_pair_connect":
            return _diagonal_pair_connect(grid, values["background"])
        if self.graph == "shape_into_frame_hole":
            return _shape_into_frame_hole(grid, values["background"])
        if self.graph == "marker_to_line_attract":
            return _marker_to_line_attract(grid, values["background"])
        if self.graph == "endpoint_pair_row_fill":
            return _endpoint_pair_row_fill(grid, values["background"])
        if self.graph == "directional_object_stamp":
            return _directional_object_stamp(grid)
        raise ValueError("unknown a1 g1 graph")


def _period_extend_vertical_recolor(grid, num, den, recolor):
    height = len(grid)
    total = height * num
    if den < 1 or num <= den or total % den:
        raise ValueError("bad extend factor")
    out_height = total // den
    rows = tuple(tuple(row) for row in grid)
    period = None
    for p in range(1, height + 1):
        if all(rows[r] == rows[r - p] for r in range(p, height)):
            period = p
            break
    if period is None:
        raise ValueError("no row period")
    out = []
    for r in range(out_height):
        row = rows[r % period]
        mapped = []
        for cell in row:
            if cell not in recolor:
                raise ValueError("unmapped color")
            mapped.append(recolor[cell])
        out.append(tuple(mapped))
    return tuple(out)


def _mode_block_tile(grid, fill):
    height, width = len(grid), len(grid[0])
    mode = _unique_mode(grid)
    out = []
    for big_r in range(height * height):
        block_r = big_r // height
        inner_r = big_r % height
        row = []
        for big_c in range(width * width):
            block_c = big_c // width
            inner_c = big_c % width
            if grid[block_r][block_c] == mode:
                row.append(grid[inner_r][inner_c])
            else:
                row.append(fill)
        out.append(tuple(row))
    return tuple(out)


def _marker_shared_component_classify(grid, marker, companion, background):
    height, width = len(grid), len(grid[0])
    blocks = [
        (r, c)
        for r in range(height - 1)
        for c in range(width - 1)
        if grid[r][c] == marker
        and grid[r + 1][c] == marker
        and grid[r][c + 1] == marker
        and grid[r + 1][c + 1] == marker
    ]
    if len(blocks) != 2:
        raise ValueError("expected two marker blocks")
    comp_cells = {
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == companion
    }
    if not comp_cells:
        raise ValueError("no companion cells")
    label = {}
    current = 0
    for start in sorted(comp_cells):
        if start in label:
            continue
        current += 1
        stack = [start]
        label[start] = current
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    q = (r + dr, c + dc)
                    if q in comp_cells and q not in label:
                        label[q] = current
                        stack.append(q)

    def touched(block):
        r0, c0 = block
        ids = set()
        for rr in range(r0 - 1, r0 + 3):
            for cc in range(c0 - 1, c0 + 3):
                if (rr, cc) in label:
                    ids.add(label[(rr, cc)])
        return ids

    first, second = (touched(block) for block in blocks)
    result = companion if first & second else background
    return ((result,),)


def _height_rank_recolor(grid, source, background, palette):
    height, width = len(grid), len(grid[0])
    bars = []
    for c in range(width):
        run = 0
        r = height - 1
        while r >= 0 and grid[r][c] == source:
            run += 1
            r -= 1
        if run:
            bars.append((run, c))
    covered = sum(run for run, _ in bars)
    total = sum(1 for row in grid for cell in row if cell == source)
    if covered != total or not bars:
        raise ValueError("source cells outside bars")
    if len(bars) != len(palette):
        raise ValueError("bar count mismatch")
    if len({run for run, _ in bars}) != len(bars):
        raise ValueError("height tie")
    order = sorted(bars, key=lambda item: (-item[0], item[1]))
    out = [[background] * width for _ in range(height)]
    for color, (run, c) in zip(palette, order):
        for r in range(height - run, height):
            out[r][c] = color
    return tuple(tuple(row) for row in out)


def _periodic_line_fill(grid, background):
    height, width = len(grid), len(grid[0])
    if height == width:
        raise ValueError("square grid")
    axis_rows = height > width
    axis_size = height if axis_rows else width
    by_coord: dict[int, set[int]] = {}
    for r in range(height):
        for c in range(width):
            value = grid[r][c]
            if value == background:
                continue
            coord = r if axis_rows else c
            by_coord.setdefault(coord, set()).add(value)
    coords = sorted(by_coord)
    if len(coords) < 2:
        raise ValueError("not enough markers")
    for coord in coords:
        if len(by_coord[coord]) != 1:
            raise ValueError("conflicting marker colors")
    steps = {b - a for a, b in zip(coords, coords[1:])}
    if len(steps) != 1:
        raise ValueError("uneven marker spacing")
    step = steps.pop()
    colors = tuple(next(iter(by_coord[coord])) for coord in coords)
    out = [[background] * width for _ in range(height)]
    k = 0
    coord = coords[0]
    while coord < axis_size:
        color = colors[k % len(colors)]
        if axis_rows:
            for index in range(width):
                out[coord][index] = color
        else:
            for index in range(height):
                out[index][coord] = color
        k += 1
        coord = coords[0] + k * step
    return tuple(tuple(row) for row in out)


def _dihedral_closure(grid, background):
    height, width = len(grid), len(grid[0])
    cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] != background
    ]
    if not cells:
        raise ValueError("no cells")
    r0, r1, c0, c1 = _bbox(cells)
    if (r0 + r1) % 2 or (c0 + c1) % 2:
        raise ValueError("no integer center")
    out = [[background] * width for _ in range(height)]
    center_r, center_c = (r0 + r1) / 2, (c0 + c1) / 2
    for r, c in cells:
        value = grid[r][c]
        dr, dc = r - center_r, c - center_c
        orbit = set()
        for a, b in ((dr, dc), (dr, -dc), (-dr, dc), (-dr, -dc),
                     (dc, dr), (dc, -dc), (-dc, dr), (-dc, -dc)):
            orbit.add((center_r + a, center_c + b))
        for rr, cc in orbit:
            if out[int(rr)][int(cc)] not in (background, value):
                raise ValueError("orbit conflict")
            out[int(rr)][int(cc)] = value
    return tuple(tuple(row) for row in out)


def _dominant_centroid_dihedral(grid, background):
    height, width = len(grid), len(grid[0])
    counts = Counter(value for row in grid for value in row)
    dominant = max(
        (value for value in counts if value != background),
        key=lambda value: (counts[value], -value),
    )
    cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == dominant
    ]
    total_r = sum(r for r, _ in cells)
    total_c = sum(c for _, c in cells)
    if total_r % len(cells) or total_c % len(cells):
        raise ValueError("no integer centroid")
    center_r, center_c = total_r // len(cells), total_c // len(cells)
    out = [[background] * width for _ in range(height)]
    colored = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] != background
    ]
    for r, c in colored:
        value = grid[r][c]
        dr, dc = r - center_r, c - center_c
        for a, b in ((dr, dc), (dr, -dc), (-dr, dc), (-dr, -dc),
                     (dc, dr), (dc, -dr), (-dc, dr), (-dc, -dr)):
            rr, cc = center_r + a, center_c + b
            if out[rr][cc] not in (background, value):
                raise ValueError("orbit conflict")
            out[rr][cc] = value
    return tuple(tuple(row) for row in out)


def _diagonal_pair_connect(grid, background):
    height, width = len(grid), len(grid[0])
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(height):
        for c in range(width):
            value = grid[r][c]
            if value != background:
                by_color.setdefault(value, []).append((r, c))
    if not by_color:
        raise ValueError("no cells")
    out = [list(row) for row in grid]
    for value in sorted(by_color):
        points = sorted(by_color[value])
        if len(points) != 2:
            raise ValueError("color without a pair")
        (r1, c1), (r2, c2) = points
        dr, dc = r2 - r1, c2 - c1
        if dr == 0 or abs(dr) != abs(dc):
            raise ValueError("pair not diagonal")
        sr, sc = (1 if dr > 0 else -1), (1 if dc > 0 else -1)
        r, c = r1, c1
        for _ in range(abs(dr) + 1):
            if out[r][c] not in (background, value):
                raise ValueError("segment conflict")
            out[r][c] = value
            r += sr
            c += sc
    return tuple(tuple(row) for row in out)


def _shape_into_frame_hole(grid, background):
    height, width = len(grid), len(grid[0])
    seen = set()
    stack = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r in (0, height - 1) or c in (0, width - 1))
        and grid[r][c] == background
    ]
    seen.update(stack)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(height, width, r, c):
            if (nr, nc) not in seen and grid[nr][nc] == background:
                seen.add((nr, nc))
                stack.append((nr, nc))
    interior = {
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == background and (r, c) not in seen
    }
    holes = _components4(interior)
    if not holes:
        raise ValueError("no holes")
    components = _value_components(grid, skip=background)
    used: list[frozenset] = []
    moves: list[tuple[int, frozenset, int, int]] = []
    for hole in holes:
        hr0, hr1, hc0, hc1 = _bbox(hole)
        matches = []
        for value, component in components:
            cr0, cr1, cc0, cc1 = _bbox(component)
            if (cr1 - cr0, cc1 - cc0) != (hr1 - hr0, hc1 - hc0):
                continue
            if component & hole:
                continue
            shift_r, shift_c = hr0 - cr0, hc0 - cc0
            if any((r + shift_r, c + shift_c) not in hole
                   for r, c in component):
                continue
            matches.append((value, component, shift_r, shift_c))
        if len(matches) != 1:
            raise ValueError("hole pairing ambiguous")
        if any(component in used for _, component, _, _ in matches):
            raise ValueError("component reused")
        used.append(matches[0][1])
        moves.append(matches[0])
    out = [list(row) for row in grid]
    for value, component, shift_r, shift_c in moves:
        for r, c in component:
            out[r][c] = background
            out[r + shift_r][c + shift_c] = value
    return tuple(tuple(row) for row in out)


def _marker_to_line_attract(grid, background):
    height, width = len(grid), len(grid[0])
    row_bars: dict[int, int] = {}
    col_bars: dict[int, int] = {}
    for r in range(height):
        row = grid[r]
        if len(set(row)) == 1 and row[0] != background:
            row_bars[r] = row[0]
    for c in range(width):
        column = {grid[r][c] for r in range(height)}
        if len(column) == 1 and grid[0][c] != background:
            col_bars[c] = grid[0][c]
    if (row_bars and col_bars) or not (row_bars or col_bars):
        raise ValueError("bar orientation")
    out = [[background] * width for _ in range(height)]
    for r, color in row_bars.items():
        for c in range(width):
            out[r][c] = color
    for c, color in col_bars.items():
        for r in range(height):
            out[r][c] = color
    destinations: dict[tuple[int, int], int] = {}
    for r in range(height):
        for c in range(width):
            value = grid[r][c]
            if value == background:
                continue
            if r in row_bars or c in col_bars:
                continue
            if row_bars:
                lines = sorted(b for b, v in row_bars.items() if v == value)
                if not lines:
                    continue
                bar = min(lines, key=lambda b: (abs(r - b), b))
                target = bar + (1 if r > bar else -1)
                key = (target, c)
                if not (0 <= target < height):
                    raise ValueError("attract off grid")
                if target in row_bars:
                    raise ValueError("attract onto bar")
            else:
                lines = sorted(b for b, v in col_bars.items() if v == value)
                if not lines:
                    continue
                bar = min(lines, key=lambda b: (abs(c - b), b))
                target = bar + (1 if c > bar else -1)
                key = (r, target)
                if not (0 <= target < width):
                    raise ValueError("attract off grid")
                if target in col_bars:
                    raise ValueError("attract onto bar")
            if key in destinations and destinations[key] != value:
                raise ValueError("marker collision")
            destinations[key] = value
    for (r, c), value in sorted(destinations.items()):
        if out[r][c] not in (background, value):
            raise ValueError("marker onto bar")
        out[r][c] = value
    return tuple(tuple(row) for row in out)


def _endpoint_pair_row_fill(grid, background):
    out = []
    for row in grid:
        if (
            len(row) >= 3
            and row[0] == row[-1]
            and row[0] != background
            and all(cell == background for cell in row[1:-1])
        ):
            out.append(tuple([row[0]] * len(row)))
        else:
            out.append(tuple(row))
    return tuple(out)


def _directional_object_stamp(grid):
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    components = _value_components(grid, skip=background)
    if len(components) < 2:
        raise ValueError("no markers")
    sizes = sorted(len(component) for _, component in components)
    if sizes[-1] == sizes[-2]:
        raise ValueError("ambiguous object")
    obj_value, obj_cells = max(components, key=lambda item: len(item[1]))
    r0, r1, c0, c1 = _bbox(obj_cells)
    block_h, block_w = r1 - r0 + 1, c1 - c0 + 1
    directions: dict[tuple[int, int], int] = {}
    for value, component in components:
        if component is obj_cells:
            continue
        mr0, mr1, mc0, mc1 = _bbox(component)
        dr = -1 if mr1 < r0 else (1 if mr0 > r1 else 0)
        dc = -1 if mc1 < c0 else (1 if mc0 > c1 else 0)
        if (dr, dc) == (0, 0):
            raise ValueError("marker touches object")
        step_r, step_c = dr * (block_h + 1), dc * (block_w + 1)
        for r, c in component:
            if (r - step_r, c - step_c) not in obj_cells:
                raise ValueError("marker outside first copy")
        if (dr, dc) in directions and directions[(dr, dc)] != value:
            raise ValueError("direction color conflict")
        directions[(dr, dc)] = value
    if not directions:
        raise ValueError("no directions")
    out = [list(row) for row in grid]
    for (dr, dc) in sorted(directions):
        color = directions[(dr, dc)]
        step_r, step_c = dr * (block_h + 1), dc * (block_w + 1)
        k = 1
        while k <= STAMP_CAP:
            painted = False
            for r, c in sorted(obj_cells):
                nr, nc = r + k * step_r, c + k * step_c
                if 0 <= nr < height and 0 <= nc < width:
                    if out[nr][nc] not in (background, color):
                        raise ValueError("stamp collision")
                    out[nr][nc] = color
                    painted = True
            if not painted:
                break
            k += 1
        else:
            raise ValueError("stamp cap exceeded")
    return tuple(tuple(row) for row in out)


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

    # --- period_extend_vertical_recolor: fattore e mappa dai demo ---
    factor = None
    recolor: dict[int, int] | None = {}
    for source, expected in demos:
        if (
            len(source[0]) != len(expected[0])
            or len(expected) <= len(source)
        ):
            factor = None
            break
        common = gcd(len(expected), len(source))
        here = (len(expected) // common, len(source) // common)
        if factor is None:
            factor = here
        elif factor != here:
            factor = None
            break
        if recolor is not None:
            for r in range(len(source)):
                for c in range(len(source[0])):
                    a, b = source[r][c], expected[r][c]
                    if recolor.get(a, b) != b:
                        recolor = None
                        break
                    recolor[a] = b
                if recolor is None:
                    break
    if factor is not None and recolor is not None:
        add(Candidate(
            "period_extend_vertical_recolor",
            (
                ("extend_den", factor[1]),
                ("extend_num", factor[0]),
                ("recolor", tuple(sorted(recolor.items()))),
            ),
        ))

    # --- mode_block_tile: colore di riempimento dai demo ---
    fill = None
    ok = True
    for source, expected in demos:
        height, width = len(source), len(source[0])
        if len(expected) != height * height or len(expected[0]) != width * width:
            ok = False
            break
        try:
            mode = _unique_mode(source)
        except EXEC_ERRORS:
            ok = False
            break
        for i in range(height):
            for j in range(width):
                if source[i][j] == mode:
                    continue
                block = {
                    expected[i * height + di][j * width + dj]
                    for di in range(height)
                    for dj in range(width)
                }
                if len(block) != 1:
                    ok = False
                    break
                value = block.pop()
                if fill is None:
                    fill = value
                elif fill != value:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok and fill is not None:
        add(Candidate("mode_block_tile", (("fill", fill),)))

    # --- marker_shared_component_classify: colori dai demo ---
    try:
        backgrounds = {_background(source) for source, _ in demos}
    except EXEC_ERRORS:
        backgrounds = set()
    if len(backgrounds) == 1:
        background = backgrounds.pop()
        marker = None
        companion = None
        consistent = True
        for source, expected in demos:
            others = sorted(
                {cell for row in source for cell in row} - {background}
            )
            if len(others) != 2 or len(expected) != 1 or len(expected[0]) != 1:
                consistent = False
                break
            here_marker = None
            for value in others:
                cells = {
                    (r, c)
                    for r in range(len(source))
                    for c in range(len(source[0]))
                    if source[r][c] == value
                }
                blocks = sum(
                    1
                    for r, c in cells
                    if (r + 1, c) in cells
                    and (r, c + 1) in cells
                    and (r + 1, c + 1) in cells
                )
                if blocks >= 2:
                    here_marker = value
                    break
            if here_marker is None:
                consistent = False
                break
            here_companion = next(v for v in others if v != here_marker)
            if marker is None:
                marker, companion = here_marker, here_companion
            elif (marker, companion) != (here_marker, here_companion):
                consistent = False
                break
        if consistent and marker is not None:
            add(Candidate(
                "marker_shared_component_classify",
                (
                    ("background", background),
                    ("companion", companion),
                    ("marker", marker),
                ),
            ))

        # --- grafi a solo binding di sfondo ---
        for graph in (
            "periodic_line_fill",
            "dihedral_closure",
            "dominant_centroid_dihedral",
            "diagonal_pair_connect",
            "shape_into_frame_hole",
            "marker_to_line_attract",
            "endpoint_pair_row_fill",
        ):
            add(Candidate(graph, (("background", background),)))
        add(Candidate("directional_object_stamp", ()))

    # --- height_rank_recolor: palette rango->colore dai demo ---
    palette: tuple[int, ...] | None = ()
    rank_source = None
    rank_background = None
    for source, expected in demos:
        try:
            background = _background(source)
        except EXEC_ERRORS:
            palette = None
            break
        others = sorted(
            {cell for row in source for cell in row} - {background}
        )
        if len(others) != 1:
            palette = None
            break
        source_color = others[0]
        height = len(source)
        bars = []
        for c in range(len(source[0])):
            run = 0
            r = height - 1
            while r >= 0 and source[r][c] == source_color:
                run += 1
                r -= 1
            if run:
                bars.append((run, c))
        if sum(run for run, _ in bars) != sum(
            1 for row in source for cell in row if cell == source_color
        ) or len({run for run, _ in bars}) != len(bars) or not bars:
            palette = None
            break
        order = sorted(bars, key=lambda item: (-item[0], item[1]))
        here = tuple(expected[height - run][c] for run, c in order)
        if palette == ():
            palette = here
            rank_source, rank_background = source_color, background
        elif palette != here or (rank_source, rank_background) != (
            source_color, background
        ):
            palette = None
            break
    if palette is not None and palette != ():
        add(Candidate(
            "height_rank_recolor",
            (
                ("background", rank_background),
                ("palette", palette),
                ("source", rank_source),
            ),
        ))

    return tuple(
        sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings)))
    )
