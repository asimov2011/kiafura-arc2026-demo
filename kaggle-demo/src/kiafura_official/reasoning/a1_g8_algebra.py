"""A1 gruppo 8: algebre parametriche con candidati derivati dai soli demo.

Grafi (parametrici per classe di trasformazione, nessun letterale di
griglia o identita' di task; i colori "nuovi" sono binding derivati dai
demo):
  - cross_periodic_tile: le braccia di una croce definiscono sequenze
    periodiche; l'intera riga e colonna del centro sono piastrellate coi
    periodi minimi delle braccia;
  - odd_component_crop: ritaglio dell'unica componente rettangolare non
    uniforme tra componenti rettangolari uniformi;
  - dual_block_marker_transfer: due tele monocrome; le componenti marcatore
    di ciascuna tela sono stampate sull'altra (posizioni relative al
    riquadro della componente);
  - trail_stamp_diagonal: un marcatore singolo tocca una figura; la figura
    (cella marcatore + vicini della figura) e' stampata lungo la diagonale
    libera piu' lunga, con colore alternato marcatore/tela;
  - dot_pair_midpoint_cross: due punti allineati a distanza pari; un segno
    a croce del colore di output e' posto nel punto medio;
  - line_length_recolor: segmenti rettilinei ricolorati in base alla
    lunghezza (mappa lunghezza->colore derivata dai demo);
  - diagonal_hole_fill: le diagonali passanti per l'unico foro sono
    riempite col colore del foro;
  - quadrant_merge_majority: merge dei quattro quadranti per maggioranza;
    parita' risolta col quarto quadrante;
  - pair_rings_dashed: coppia di punti; anelli 3x3 a colori scambiati,
    connettore tratteggiato tra i punti e sue righe/colonne specchio
    tratteggiate oltre gli anelli (colore del tratteggio dai demo);
  - staircase_extend: i segmenti orizzontali di una scala (ordinati per
    riga) sono estesi a destra: indice pari fino al bordo, indice dispari
    di una cella;
  - downsample_2x2_presence: blocchi 2x2 allineati sono contratti in celle
    di presenza (output = griglia dimezzata per eccesso).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]
EXEC_ERRORS = (ValueError, IndexError, KeyError, StopIteration)


def _counts(grid: Grid) -> Counter:
    return Counter(value for row in grid for value in row)


def _most_common(counts: Counter) -> int:
    return min(counts, key=lambda value: (-counts[value], value))


def _rectangular(grid: Grid) -> None:
    if not grid or not grid[0]:
        raise ValueError("empty grid")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("ragged grid")


def _min_period(seq: Sequence[int]) -> tuple[int, ...]:
    """Minimal period prefix of seq (full sequence if aperiodic)."""
    size = len(seq)
    for p in range(1, size + 1):
        if all(seq[i] == seq[i - p] for i in range(p, size)):
            return tuple(seq[:p])
    return tuple(seq)


def _neighbors4(height: int, width: int, r: int, c: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _components4(cells, height, width):
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
            for nr, nc in _neighbors4(height, width, r, c):
                if (nr, nc) in remaining and (nr, nc) not in component:
                    stack.append((nr, nc))
        remaining -= component
        found.append(component)
    return sorted(found, key=min)


# ---------------------------------------------------------------- grafi


def _cross_periodic_tile(grid: Grid) -> Grid:
    _rectangular(grid)
    height, width = len(grid), len(grid[0])
    cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] != 0
    ]
    if not cells:
        raise ValueError("no cross")
    comps = _components4(cells, height, width)
    if len(comps) != 1:
        raise ValueError("multiple components")
    cellset = set(cells)
    centers = [
        cell
        for cell in cellset
        if all(
            (cell[0] + dr, cell[1] + dc) in cellset
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
    ]
    if len(centers) != 1:
        raise ValueError("no unique cross center")
    cr, cc = centers[0]

    def arm(dr: int, dc: int) -> list[int]:
        seq = []
        r, c = cr + dr, cc + dc
        while 0 <= r < height and 0 <= c < width and grid[r][c] != 0:
            seq.append(grid[r][c])
            r += dr
            c += dc
        return seq

    top, bottom = arm(-1, 0), arm(1, 0)
    left, right = arm(0, -1), arm(0, 1)
    if sum(map(len, (top, bottom, left, right))) + 1 != len(cells):
        raise ValueError("not a cross")
    center = grid[cr][cc]
    vseq = top[::-1] + [center] + bottom
    hseq = left[::-1] + [center] + right
    vper = _min_period(vseq)
    hper = _min_period(hseq)
    vlen = len(top)

    def v_at(r: int) -> int:
        return vper[(r - cr + vlen) % len(vper)]

    hlen = len(left)

    def h_at(c: int) -> int:
        return hper[(c - cc + hlen) % len(hper)]

    out = [list(row) for row in grid]
    for r in range(height):
        if out[r][cc] not in (0, v_at(r)):
            raise ValueError("column conflict")
        out[r][cc] = v_at(r)
    for c in range(width):
        if out[cr][c] not in (0, h_at(c)):
            raise ValueError("row conflict")
        out[cr][c] = h_at(c)
    return tuple(tuple(row) for row in out)


def _odd_component_crop(grid: Grid) -> Grid:
    _rectangular(grid)
    height, width = len(grid), len(grid[0])
    cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] != 0
    ]
    if not cells:
        raise ValueError("no components")
    mixed = []
    for component in _components4(cells, height, width):
        values = {grid[r][c] for r, c in component}
        if len(values) > 1:
            mixed.append(component)
    if len(mixed) != 1:
        raise ValueError("expected exactly one mixed component")
    component = mixed[0]
    top = min(r for r, _ in component)
    bottom = max(r for r, _ in component)
    left = min(c for _, c in component)
    right = max(c for _, c in component)
    return tuple(row[left:right + 1] for row in grid[top:bottom + 1])


def _dual_block_marker_transfer(grid: Grid) -> Grid:
    _rectangular(grid)
    height, width = len(grid), len(grid[0])

    def dominant(rows: range, cols: range) -> int:
        counts = Counter(grid[r][c] for r in rows for c in cols)
        return _most_common(counts)

    def uniform(rows: range, cols: range, value: int) -> bool:
        return all(grid[r][c] == value for r in rows for c in cols)

    splits = []
    for s in range(1, height - 1):
        bg_top = dominant(range(0, s), range(width))
        bg_bot = dominant(range(s, height), range(width))
        if bg_top == bg_bot:
            continue
        if uniform(range(0, s), range(width), bg_top) \
                and uniform(range(s, height), range(width), bg_bot):
            splits.append(("h", s, bg_top, bg_bot))
    for s in range(1, width - 1):
        bg_left = dominant(range(height), range(0, s))
        bg_right = dominant(range(height), range(s, width))
        if bg_left == bg_right:
            continue
        if uniform(range(height), range(0, s), bg_left) \
                and uniform(range(height), range(s, width), bg_right):
            splits.append(("v", s, bg_left, bg_right))
    if not splits:
        raise ValueError("no dual block split")

    orientation, split, bg_a, bg_b = sorted(splits)[0]
    if orientation == "h":
        ra0, ra1, ca0, ca1 = 0, split, 0, width
        rb0, rb1, cb0, cb1 = split, height, 0, width
    else:
        ra0, ra1, ca0, ca1 = 0, height, 0, split
        rb0, rb1, cb0, cb1 = 0, height, split, width

    def marker_components(r0, r1, c0, c1, bg):
        cells = [
            (r, c)
            for r in range(r0, r1)
            for c in range(c0, c1)
            if grid[r][c] != bg
        ]
        if not cells:
            raise ValueError("empty marker set")
        comps = []
        for component in _components4(cells, height, width):
            top = min(r for r, _ in component)
            bottom = max(r for r, _ in component)
            left = min(c for _, c in component)
            right = max(c for _, c in component)
            if top < r0 or bottom >= r1 or left < c0 or right >= c1:
                raise ValueError("marker outside its block")
            comps.append((component, top - r0, left - c0))
        return comps

    comps_a = marker_components(ra0, ra1, ca0, ca1, bg_a)
    comps_b = marker_components(rb0, rb1, cb0, cb1, bg_b)

    out = [list(row) for row in grid]

    def stamp(components, target_bg, tr0, tc0):
        for component, rel_top, rel_left in components:
            for r, c in component:
                nr = tr0 + (r - rel_top)
                nc = tc0 + (c - rel_left)
                if not (0 <= nr < height and 0 <= nc < width):
                    raise ValueError("stamp off grid")
                value = grid[r][c]
                if out[nr][nc] not in (target_bg, value):
                    raise ValueError("stamp conflict")
                out[nr][nc] = value

    stamp(comps_b, bg_a, ra0, ca0)
    stamp(comps_a, bg_b, rb0, cb0)
    return tuple(tuple(row) for row in out)


def _trail_stamp_diagonal(grid: Grid) -> Grid:
    _rectangular(grid)
    height, width = len(grid), len(grid[0])
    counts = _counts(grid)
    if len(counts) != 3:
        raise ValueError("expected canvas, glyph and marker")
    canvas = _most_common(counts)
    singletons = [value for value, n in counts.items()
                  if n == 1 and value != canvas]
    if len(singletons) != 1:
        raise ValueError("expected one singleton marker")
    marker = singletons[0]
    markers = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == marker
    ]
    mr, mc = markers[0]
    glyph = {
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] not in (canvas, marker)
    }
    if not glyph or not any(
        (nr, nc) in glyph for nr, nc in _neighbors4(height, width, mr, mc)
    ):
        raise ValueError("marker must touch the glyph")
    shape = {(mr, mc)} | {
        (nr, nc)
        for nr, nc in _neighbors4(height, width, mr, mc)
        if (nr, nc) in glyph
    }

    best = None
    for dr, dc in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
        max_steps = 0
        r, c = mr + dr, mc + dc
        while 0 <= r < height and 0 <= c < width:
            max_steps += 1
            r += dr
            c += dc
        k_max = 0
        for k in range(1, max_steps + 1):
            moved = {(r + k * dr, c + k * dc) for r, c in shape}
            if any(
                0 <= rr < height and 0 <= cc < width
                and grid[rr][cc] not in (canvas, marker)
                for rr, cc in moved
            ):
                break
            k_max = k
        if best is None or k_max > best[0]:
            best = (k_max, dr, dc)
    k_max, dr, dc = best
    if k_max < 1:
        raise ValueError("no trail direction")
    out = [list(row) for row in grid]
    for k in range(1, k_max + 1):
        value = marker if k % 2 == 1 else canvas
        for r, c in shape:
            rr, cc = r + k * dr, c + k * dc
            if not (0 <= rr < height and 0 <= cc < width):
                continue
            if out[rr][cc] not in (canvas, value):
                raise ValueError("trail conflict")
            out[rr][cc] = value
    return tuple(tuple(row) for row in out)


def _dot_pair_midpoint_cross(grid: Grid, dot: int, cross: int) -> Grid:
    _rectangular(grid)
    height, width = len(grid), len(grid[0])
    if dot == cross:
        raise ValueError("dot equals cross color")
    dots = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == dot
    ]
    if len(dots) != 2:
        raise ValueError("expected two dots")
    (r1, c1), (r2, c2) = sorted(dots)
    if r1 != r2 and c1 != c2:
        raise ValueError("dots not aligned")
    distance = abs(r2 - r1) + abs(c2 - c1)
    if distance == 0 or distance % 2:
        raise ValueError("bad dot distance")
    cr, cc = (r1 + r2) // 2, (c1 + c2) // 2
    out = [list(row) for row in grid]
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = cr + dr, cc + dc
        if not (0 <= nr < height and 0 <= nc < width):
            raise ValueError("cross off grid")
        if out[nr][nc] not in (0, cross):
            raise ValueError("cross conflict")
        out[nr][nc] = cross
    return tuple(tuple(row) for row in out)


def _line_segments(grid: Grid, line: int) -> list[tuple[frozenset, int]]:
    height, width = len(grid), len(grid[0])
    cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == line
    ]
    if not cells:
        raise ValueError("no line cells")
    segments = []
    for component in _components4(cells, height, width):
        rows = {r for r, _ in component}
        cols = {c for _, c in component}
        if len(rows) == 1:
            length = max(cols) - min(cols) + 1
        elif len(cols) == 1:
            length = max(rows) - min(rows) + 1
        else:
            raise ValueError("non straight segment")
        if length != len(component):
            raise ValueError("segment with gap")
        segments.append((frozenset(component), length))
    return segments


def _line_length_recolor(
    grid: Grid, color_map: dict[int, int], canvas: int, line: int
) -> Grid:
    _rectangular(grid)
    counts = _counts(grid)
    if canvas not in counts or line not in counts or canvas == line:
        raise ValueError("canvas/line missing")
    if any(value != canvas and value != line for value in counts):
        raise ValueError("extra colors")
    out = [list(row) for row in grid]
    for component, length in _line_segments(grid, line):
        if length not in color_map:
            raise KeyError("unmapped segment length")
        target = color_map[length]
        for r, c in component:
            out[r][c] = target
    return tuple(tuple(row) for row in out)


def _diagonal_hole_fill(grid: Grid, hole: int) -> Grid:
    _rectangular(grid)
    height, width = len(grid), len(grid[0])
    counts = _counts(grid)
    if hole not in counts:
        raise ValueError("hole color missing")
    others = [value for value in counts if value != hole]
    if len(others) != 1 or counts[hole] != 1:
        raise ValueError("expected one hole on uniform canvas")
    canvas = others[0]
    holes = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == hole
    ]
    hr, hc = holes[0]
    out = [list(row) for row in grid]
    for r in range(height):
        for c in range(width):
            if abs(r - hr) == abs(c - hc):
                if out[r][c] not in (canvas, hole):
                    raise ValueError("diagonal conflict")
                out[r][c] = hole
    return tuple(tuple(row) for row in out)


def _quadrant_merge_majority(grid: Grid) -> Grid:
    _rectangular(grid)
    height, width = len(grid), len(grid[0])
    if height % 2 or width % 2:
        raise ValueError("odd dimensions")
    half_h, half_w = height // 2, width // 2
    out = []
    for r in range(half_h):
        row = []
        for c in range(half_w):
            q1 = grid[r][c]
            q2 = grid[r][c + half_w]
            q3 = grid[r + half_h][c]
            q4 = grid[r + half_h][c + half_w]
            counts = Counter((q1, q2, q3, q4))
            best = max(counts.values())
            if best >= 3:
                value = next(v for v, n in counts.items() if n == best)
            elif best == 2:
                pair = sorted(v for v, n in counts.items() if n == 2)
                value = q4 if q4 in pair else pair[0]
            else:
                value = q4
            row.append(value)
        out.append(tuple(row))
    return tuple(out)


def _pair_rings_dashed(grid: Grid, background: int, dash: int) -> Grid:
    _rectangular(grid)
    height, width = len(grid), len(grid[0])
    dots = [
        (r, c, grid[r][c])
        for r in range(height)
        for c in range(width)
        if grid[r][c] != background
    ]
    if len(dots) != 2:
        raise ValueError("expected two dots")
    (ra, ca, va), (rb, cb, vb) = sorted(dots)
    if va == vb or va == background or vb == background \
            or va == dash or vb == dash:
        raise ValueError("bad dot colors")
    out = [[background] * width for _ in range(height)]

    def ring(r: int, c: int, color: int) -> None:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    raise ValueError("ring off grid")
                if dr == 0 and dc == 0:
                    continue
                out[nr][nc] = color

    ring(ra, ca, vb)
    out[ra][ca] = va
    ring(rb, cb, va)
    out[rb][cb] = vb
    if ra == rb:
        gap = cb - ca
        if gap < 3:
            raise ValueError("gap too small")
        for c in range(ca + 1, cb):
            if (c - ca) % 2 == 1:
                out[ra][c] = dash
        base = max(ra, rb) + 2
        count = gap - 3
        for r in range(base, base + count):
            if r >= height:
                raise ValueError("mirror off grid")
            out[r][ca] = dash
            out[r][cb] = dash
    elif ca == cb:
        gap = rb - ra
        if gap < 3:
            raise ValueError("gap too small")
        for r in range(ra + 1, rb):
            if (r - ra) % 2 == 1:
                out[r][ca] = dash
        base = max(ca, cb) + 2
        count = gap - 3
        for c in range(base, base + count):
            if c >= width:
                raise ValueError("mirror off grid")
            out[ra][c] = dash
            out[rb][c] = dash
    else:
        raise ValueError("dots not aligned")
    return tuple(tuple(row) for row in out)


def _staircase_extend(grid: Grid, figure: int) -> Grid:
    _rectangular(grid)
    height, width = len(grid), len(grid[0])
    counts = _counts(grid)
    if figure not in counts:
        raise ValueError("figure color missing")
    others = [value for value in counts if value != figure]
    if not others:
        raise ValueError("no background")
    background = _most_common(Counter({v: counts[v] for v in others}))
    out = [list(row) for row in grid]
    cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == figure
    ]
    segments = []
    for component in _components4(cells, height, width):
        rows = {r for r, _ in component}
        if len(rows) != 1:
            raise ValueError("segment spans rows")
        segments.append((min(rows), max(c for _, c in component)))
    segments.sort()
    if any(
        first[0] == second[0]
        for first, second in zip(segments, segments[1:])
    ):
        raise ValueError("two segments on one row")
    for index, (row_index, end_col) in enumerate(segments):
        targets = (
            range(end_col + 1, width)
            if index % 2 == 0
            else (end_col + 1,)
        )
        for c in targets:
            if c >= width:
                raise ValueError("extension off grid")
            if out[row_index][c] != background:
                raise ValueError("extension conflict")
            out[row_index][c] = figure
    return tuple(tuple(row) for row in out)


def _downsample_2x2_presence(grid: Grid, obj: int, mark: int) -> Grid:
    _rectangular(grid)
    height, width = len(grid), len(grid[0])
    counts = _counts(grid)
    if obj not in counts or obj == mark:
        raise ValueError("object color missing")
    background = _background_of(counts, obj)
    if any(value not in (obj, background) for value in counts):
        raise ValueError("extra colors")
    cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == obj
    ]
    if not cells:
        raise ValueError("no blocks")
    anchors = set()
    for r, c in cells:
        if r % 2 or c % 2:
            raise ValueError("unaligned block cell")
        anchors.add((r, c))
    for r, c in anchors:
        for dr in (0, 1):
            for dc in (0, 1):
                if r + dr >= height or c + dc >= width \
                        or grid[r + dr][c + dc] != obj:
                    raise ValueError("incomplete 2x2 block")
    out_h = (height + 1) // 2
    out_w = (width + 1) // 2
    return tuple(
        tuple(mark if (r, c) in anchors else background for c in range(out_w))
        for r in range(out_h)
    )


def _background_of(counts: Counter, obj: int) -> int:
    others = [value for value in counts if value != obj]
    if not others:
        raise ValueError("no background")
    return _most_common(Counter({v: counts[v] for v in others}))


# ---------------------------------------------------------------- candidato


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]

    def cost(self, grid: Grid) -> int:
        return 1 + len(grid) * len(grid[0])

    def execute(self, grid: Grid) -> Grid:
        values = dict(self.bindings)
        if self.graph == "cross_periodic_tile":
            return _cross_periodic_tile(grid)
        if self.graph == "odd_component_crop":
            return _odd_component_crop(grid)
        if self.graph == "dual_block_marker_transfer":
            return _dual_block_marker_transfer(grid)
        if self.graph == "trail_stamp_diagonal":
            return _trail_stamp_diagonal(grid)
        if self.graph == "dot_pair_midpoint_cross":
            return _dot_pair_midpoint_cross(
                grid, values["dot"], values["cross"]
            )
        if self.graph == "line_length_recolor":
            pairs = sorted(
                (int(k), v) for k, v in values["color_map"]
            )
            return _line_length_recolor(
                grid, dict(pairs), values["canvas"], values["line"]
            )
        if self.graph == "diagonal_hole_fill":
            return _diagonal_hole_fill(grid, values["hole"])
        if self.graph == "quadrant_merge_majority":
            return _quadrant_merge_majority(grid)
        if self.graph == "pair_rings_dashed":
            return _pair_rings_dashed(
                grid, values["background"], values["dash"]
            )
        if self.graph == "staircase_extend":
            return _staircase_extend(grid, values["figure"])
        if self.graph == "downsample_2x2_presence":
            return _downsample_2x2_presence(
                grid, values["object"], values["mark"]
            )
        raise ValueError("unknown a1 g8 graph")


# ---------------------------------------------------------------- derivazione


def _intersect(sets: list[set[int]]) -> set[int]:
    result = None
    for item in sets:
        result = item if result is None else result & item
    return result or set()


def candidates(demos: Demos) -> tuple[Candidate, ...]:
    """Generatore di candidati: binding derivati dai soli demo; ogni
    candidato emesso riproduce exact tutti i demo (filtro in add)."""
    if not demos:
        return ()
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

    add(Candidate("cross_periodic_tile", ()))
    add(Candidate("odd_component_crop", ()))
    add(Candidate("dual_block_marker_transfer", ()))
    add(Candidate("trail_stamp_diagonal", ()))
    add(Candidate("quadrant_merge_majority", ()))

    source_colors = [
        {value for row in source for value in row} for source, _ in demos
    ]
    new_colors = [
        {value for row in expected for value in row} - source_colors[i]
        for i, (source, expected) in enumerate(demos)
    ]

    for hole in sorted(_intersect(source_colors)):
        add(Candidate("diagonal_hole_fill", (("hole", hole),)))

    for figure in sorted(_intersect(source_colors)):
        add(Candidate("staircase_extend", (("figure", figure),)))

    for dot in sorted(_intersect([
        {v for v, n in _counts(source).items() if n == 2}
        for source, _ in demos
    ])):
        for cross in sorted(_intersect(new_colors)):
            add(Candidate(
                "dot_pair_midpoint_cross",
                (("cross", cross), ("dot", dot)),
            ))

    for background in sorted(_intersect(source_colors)):
        for dash in sorted(_intersect(new_colors)):
            add(Candidate(
                "pair_rings_dashed",
                (("background", background), ("dash", dash)),
            ))

    for obj in sorted(_intersect(source_colors)):
        for mark in sorted(_intersect(new_colors)):
            add(Candidate(
                "downsample_2x2_presence",
                (("mark", mark), ("object", obj)),
            ))

    _line_length_candidates(demos, add)

    return tuple(
        sorted(
            result.values(), key=lambda item: (item.graph, repr(item.bindings))
        )
    )


def _line_length_candidates(demos: Demos, add) -> None:
    """Mappa lunghezza->colore derivata dai demo (ricoloritura di
    segmenti rettilinei su tela)."""
    constraints: dict[int, int] = {}
    canvases: set[int] | None = None
    lines: set[int] | None = None
    for source, expected in demos:
        if len(source) != len(expected) or len(source[0]) != len(expected[0]):
            return
        counts = _counts(source)
        if len(counts) != 2:
            return
        canvas = _most_common(counts)
        line = next(v for v in counts if v != canvas)
        try:
            segments = _line_segments(source, line)
        except EXEC_ERRORS:
            return
        if not segments:
            return
        for component, length in segments:
            out_values = {expected[r][c] for r, c in component}
            if len(out_values) != 1:
                return
            value = out_values.pop()
            if constraints.get(length, value) != value:
                return
            constraints[length] = value
        canvases = {canvas} if canvases is None else canvases & {canvas}
        lines = {line} if lines is None else lines & {line}
    if not constraints or not canvases or not lines:
        return
    add(Candidate(
        "line_length_recolor",
        (
            ("canvas", next(iter(canvases))),
            ("color_map", tuple(sorted(
                (str(k), v) for k, v in constraints.items()
            ))),
            ("line", next(iter(lines))),
        ),
    ))
