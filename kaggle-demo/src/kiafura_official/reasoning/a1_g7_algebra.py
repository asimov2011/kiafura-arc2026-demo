"""A1 gruppo 7: algebre parametriche con candidati derivati dai soli demo.

Grafi (parametrici per classe di trasformazione, nessun letterale di
griglia o identita' di task; i colori/colonne/soglie sono binding
derivati dai demo o quantita' strutturali ricalcolate all'esecuzione):
  - crop_two_color_swap: ritaglio del riquadro del contenuto non
    background; i due colori del blocco sono scambiati;
  - row_marker_recolor: le celle del colore figura sono ricolorate col
    valore della colonna marcatore sulla stessa riga;
  - period_tile_crop: la griglia e' una tassellazione kxk di blocchi
    tutti congruenti (orbita diedrale) al blocco chiave; output = il
    blocco chiave minimo (k minimo che divide le dimensioni);
  - parity_marker_field: marcatori isolati; ogni cella appartiene al
    marcatore piu' vicino in distanza L1 (parita' -> background); la
    cella e' dipinta solo se cade nel pattern a bande pari del
    marcatore (bracci congruenti alla parita' della sua riga/colonna);
  - cross_stamp: ogni singleton del colore marcatore stampa una croce
    4-connessa coi valori per direzione derivati dai demo;
  - count_row: output = riga singola lunga quanto il numero di celle
    non background, riempita dell'unico colore non background;
  - base_stripe_deflect: riga base bicolore; le strisce del colore
    striatura salgono dalla base; una cella marcatore direttamente
    sopra una colonna di striscia devia la striscia di una colonna a
    destra per tutte le righe sopra il marcatore (riga del marcatore:
    striscia deviata accanto al marcatore; riga sotto: entrambe);
  - box_stamp: ogni cella del colore target e' cerchiata da un anello
    3x3 del colore cornice (derivato dai demo);
  - noisy_region_pick: le regioni monocrome separate sono contaminate
    da celle sparse del colore rumore; output = colore della regione
    con piu' celle rumore nel proprio riquadro;
  - bounce_fill: la griglia e' riempita dal colore riempimento con un
    marcatore che rimbalza tra i bordi di colonna (periodo 2*(W-1)),
    ancorato alla posizione del singleton di input.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]
EXEC_ERRORS = (ValueError, IndexError, KeyError, StopIteration)

CROSS_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    return min(counts, key=lambda value: (-counts[value], value))


def _rectangular(grid: Grid) -> None:
    if not grid or not grid[0]:
        raise ValueError("empty grid")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("ragged grid")


def _neighbors4(height: int, width: int, r: int, c: int):
    for dr, dc in CROSS_DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _intersect(sets: Sequence[set]) -> set:
    result = None
    for item in sets:
        result = item if result is None else result & item
    return set() if result is None else result


def _dihedral_orbit(block: Grid) -> set:
    """All rotations/reflections of a rectangular block."""

    def rot90(piece):
        height, width = len(piece), len(piece[0])
        return tuple(
            tuple(piece[height - 1 - c][r] for c in range(height))
            for r in range(width)
        )

    def flip(piece):
        return tuple(tuple(reversed(row)) for row in piece)

    orbit = set()
    current = block
    for _ in range(4):
        orbit.add(current)
        orbit.add(flip(current))
        current = rot90(current)
    return orbit


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
        if self.graph == "crop_two_color_swap":
            return _crop_two_color_swap(grid)
        if self.graph == "row_marker_recolor":
            return _row_marker_recolor(grid, values)
        if self.graph == "period_tile_crop":
            return _period_tile_crop(grid)
        if self.graph == "parity_marker_field":
            return _parity_marker_field(grid)
        if self.graph == "cross_stamp":
            return _cross_stamp(grid, values)
        if self.graph == "count_row":
            return _count_row(grid, values)
        if self.graph == "base_stripe_deflect":
            return _base_stripe_deflect(grid)
        if self.graph == "box_stamp":
            return _box_stamp(grid, values)
        if self.graph == "noisy_region_pick":
            return _noisy_region_pick(grid, values)
        if self.graph == "bounce_fill":
            return _bounce_fill(grid, values)
        raise ValueError("unknown a1 g7 graph")


def _crop_two_color_swap(grid: Grid) -> Grid:
    background = _background(grid)
    height, width = len(grid), len(grid[0])
    cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] != background
    ]
    if not cells:
        raise ValueError("no content")
    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    right = max(c for _, c in cells)
    if (top, left, bottom, right) == (0, 0, height - 1, width - 1):
        raise ValueError("content is the whole grid")
    colors = sorted({
        grid[r][c]
        for r in range(top, bottom + 1)
        for c in range(left, right + 1)
    })
    if len(colors) != 2:
        raise ValueError("block needs exactly two colors")
    low, high = colors
    return tuple(
        tuple(
            high if grid[r][c] == low else low
            for c in range(left, right + 1)
        )
        for r in range(top, bottom + 1)
    )


def _row_marker_recolor(grid: Grid, values) -> Grid:
    shape = values["shape_color"]
    marker_col = values["marker_col"]
    if not 0 <= marker_col < len(grid[0]):
        raise ValueError("marker column out of range")
    out = []
    for row_index, row in enumerate(grid):
        marker = row[marker_col]
        if marker == shape:
            raise ValueError("marker equals shape color")
        out.append(tuple(
            marker if cell == shape else cell for cell in row
        ))
    return tuple(out)


def _period_tile_crop(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    limit = min(height, width, 12)
    for k in range(2, limit + 1):
        if height % k or width % k:
            continue
        key = tuple(grid[r][:k] for r in range(k))
        orbit = _dihedral_orbit(key)
        if all(
            tuple(grid[br + i][bc:bc + k] for i in range(k)) in orbit
            for br in range(0, height, k)
            for bc in range(0, width, k)
        ):
            return key
    raise ValueError("no dihedral tiling")


def _parity_marker_field(grid: Grid) -> Grid:
    background = _background(grid)
    height, width = len(grid), len(grid[0])
    markers = [
        (r, c, grid[r][c])
        for r in range(height)
        for c in range(width)
        if grid[r][c] != background
    ]
    if len(markers) < 2:
        raise ValueError("need at least two markers")
    if len({value for _, _, value in markers}) != len(markers):
        raise ValueError("duplicate marker colors")
    out = []
    for r in range(height):
        row = []
        for c in range(width):
            best = sorted(
                (abs(r - mr) + abs(c - mc), index)
                for index, (mr, mc, _) in enumerate(markers)
            )
            if len(best) > 1 and best[0][0] == best[1][0]:
                row.append(background)
                continue
            mr, mc, value = markers[best[0][1]]
            dr, dc = abs(r - mr), abs(c - mc)
            painted = (
                (dc >= dr and (c - mc) % 2 == 0)
                or (dr >= dc and (r - mr) % 2 == 0)
            )
            row.append(value if painted else background)
        out.append(tuple(row))
    return tuple(out)


def _cross_stamp(grid: Grid, values) -> Grid:
    marker = values["marker"]
    stamp = dict(values["stamp"])
    background = _background(grid)
    height, width = len(grid), len(grid[0])
    centers = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == marker
    ]
    if not centers:
        raise ValueError("no marker cells")
    for r, c in centers:
        if any(grid[nr][nc] == marker
               for nr, nc in _neighbors4(height, width, r, c)):
            raise ValueError("marker cells not isolated")
    allowed = {background, marker} | set(stamp.values())
    if any(cell not in allowed for row in grid for cell in row):
        raise ValueError("foreign colors in grid")
    out = [[background] * width for _ in range(height)]
    for r, c in centers:
        out[r][c] = marker
    for r, c in centers:
        for dr, dc in CROSS_DIRS:
            key = (dr, dc)
            if key not in stamp:
                raise ValueError("missing direction binding")
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            current = out[nr][nc]
            if current not in (background, stamp[key]):
                raise ValueError("stamp collision")
            out[nr][nc] = stamp[key]
    return tuple(tuple(row) for row in out)


def _count_row(grid: Grid, values) -> Grid:
    background = values["background"]
    colors = {cell for row in grid for cell in row} - {background}
    if len(colors) != 1:
        raise ValueError("expected a single non-background color")
    color = colors.pop()
    total = sum(1 for row in grid for cell in row if cell != background)
    if not 1 <= total <= 10000:
        raise ValueError("bad count")
    return ((color,) * total,)


def _base_stripe_deflect(grid: Grid) -> Grid:
    background = _background(grid)
    height, width = len(grid), len(grid[0])
    bases = [
        r for r in range(height)
        if sum(1 for cell in grid[r] if cell != background) >= 2
    ]
    if len(bases) != 1:
        raise ValueError("need exactly one stripe base row")
    base = bases[0]
    base_cols = [c for c in range(width) if grid[base][c] != background]
    colors = {grid[base][c] for c in base_cols}
    if len(colors) == 2:
        off_base = [
            color for color in colors
            if any(
                grid[r][c] == color
                for r in range(height)
                for c in range(width)
                if r != base
            )
        ]
        if len(off_base) != 1:
            raise ValueError("marker color ambiguous")
        marker = off_base[0]
        stripe = (colors - {marker}).pop()
    elif len(colors) == 1:
        stripe = colors.pop()
        others = {
            grid[r][c]
            for r in range(height)
            for c in range(width)
            if r != base and grid[r][c] != background
        }
        if len(others) != 1:
            raise ValueError("marker color ambiguous")
        marker = others.pop()
    else:
        raise ValueError("base needs one or two colors")
    stripe_cols = [c for c in base_cols if grid[base][c] == stripe]
    for r in range(height):
        if r == base:
            continue
        for c in range(width):
            if grid[r][c] not in (background, marker):
                raise ValueError("foreign color off base")
    for r in range(base + 1, height):
        if any(cell != background for cell in grid[r]):
            raise ValueError("content below base")
    out = [[background] * width for _ in range(height)]
    out[base] = list(grid[base])
    for col in stripe_cols:
        blockers = [r for r in range(base) if grid[r][col] == marker]
        if len(blockers) > 1:
            raise ValueError("multiple blockers on stripe column")
        if not blockers:
            for r in range(base):
                if out[r][col] != background:
                    raise ValueError("stripe collision")
                out[r][col] = stripe
            continue
        q = blockers[0]
        shifted = col + 1
        if shifted >= width:
            raise ValueError("deflection off grid")
        for r in range(q + 1):
            if out[r][shifted] != background:
                raise ValueError("stripe collision")
            out[r][shifted] = stripe
        if out[q + 1][shifted] != background \
                or out[q + 1][col] != background:
            raise ValueError("stripe collision")
        out[q + 1][shifted] = stripe
        out[q + 1][col] = stripe
        for r in range(q + 2, base):
            if out[r][col] != background:
                raise ValueError("stripe collision")
            out[r][col] = stripe
    for r in range(height):
        for c in range(width):
            if grid[r][c] == marker:
                out[r][c] = marker
    return tuple(tuple(row) for row in out)


def _box_stamp(grid: Grid, values) -> Grid:
    target = values["target"]
    frame = values["frame"]
    background = _background(grid)
    height, width = len(grid), len(grid[0])
    centers = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == target
    ]
    if not centers:
        raise ValueError("no target cells")
    out = [list(row) for row in grid]
    for r, c in centers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                current = out[nr][nc]
                if current not in (background, frame):
                    raise ValueError("box collision")
                out[nr][nc] = frame
    return tuple(tuple(row) for row in out)


def _noisy_region_pick(grid: Grid, values) -> Grid:
    background = _background(grid)
    height, width = len(grid), len(grid[0])
    candidates_noise = []
    for value in sorted({cell for row in grid for cell in row}):
        if value == background:
            continue
        cells = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if grid[r][c] == value
        ]
        if all(
            all(grid[nr][nc] != value
                for nr, nc in _neighbors4(height, width, r, c))
            for r, c in cells
        ):
            candidates_noise.append(value)
    if len(candidates_noise) != 1:
        raise ValueError("noise color ambiguous")
    noise = candidates_noise[0]
    seen = [[False] * width for _ in range(height)]
    best = None
    for r in range(height):
        for c in range(width):
            if grid[r][c] == noise or seen[r][c]:
                continue
            color = grid[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            component = []
            while stack:
                cr, cc = stack.pop()
                component.append((cr, cc))
                for nr, nc in _neighbors4(height, width, cr, cc):
                    if not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            top = min(cr for cr, _ in component)
            bottom = max(cr for cr, _ in component)
            left = min(cc for _, cc in component)
            right = max(cc for _, cc in component)
            count = sum(
                1
                for nr in range(top, bottom + 1)
                for nc in range(left, right + 1)
                if grid[nr][nc] == noise
            )
            if best is not None and count == best[0]:
                raise ValueError("region tie")
            if best is None or count > best[0]:
                best = (count, color)
    if best is None:
        raise ValueError("no regions")
    return ((best[1],),)


def _bounce_fill(grid: Grid, values) -> Grid:
    fill = values["fill"]
    marker = values["marker"]
    background = _background(grid)
    height, width = len(grid), len(grid[0])
    singles = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] != background
    ]
    if len(singles) != 1:
        raise ValueError("need a single marker cell")
    anchor_r, anchor_c = singles[0]
    if grid[anchor_r][anchor_c] != marker:
        raise ValueError("marker mismatch")
    period = 2 * (width - 1)
    if period < 2:
        raise ValueError("grid too narrow to bounce")
    out = []
    for r in range(height):
        unfolded = anchor_c + (anchor_r - r)
        phase = unfolded % period
        position = phase if phase < width else period - phase
        row = [fill] * width
        row[position] = marker
        out.append(tuple(row))
    return tuple(out)


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

    add(Candidate("crop_two_color_swap", ()))
    add(Candidate("period_tile_crop", ()))
    add(Candidate("parity_marker_field", ()))
    add(Candidate("base_stripe_deflect", ()))

    source_colors = [
        {cell for row in source for cell in row} for source, _ in demos
    ]
    output_colors = [
        {cell for row in expected for cell in row} for _, expected in demos
    ]
    new_colors = sorted(_intersect([
        out_colors - src_colors
        for src_colors, out_colors in zip(source_colors, output_colors)
    ]))
    common_input = sorted(_intersect(source_colors))
    backgrounds = [
        _background(source) for source, _ in demos
    ]

    # row_marker_recolor: colore figura = colore che cambia in ogni
    # demo; colonna marcatore = colonna invariante che fornisce il
    # valore di uscita sulle celle figura.
    changed_sets = []
    for source, expected in demos:
        if len(source) != len(expected) or len(source[0]) != len(expected[0]):
            changed_sets.append(set())
            continue
        changed_sets.append({
            source[r][c]
            for r in range(len(source))
            for c in range(len(source[0]))
            if source[r][c] != expected[r][c]
        })
    for shape in sorted(_intersect(changed_sets)):
        good_cols = None
        for col in range(len(demos[0][0][0])):
            ok = True
            for source, expected in demos:
                if any(source[r][col] != expected[r][col]
                       for r in range(len(source))):
                    ok = False
                    break
                if ok and any(
                    source[r][col] != expected[r][c]
                    for r in range(len(source))
                    for c in range(len(source[0]))
                    if source[r][c] == shape
                ):
                    ok = False
                    break
            if ok:
                good_cols = col if good_cols is None else good_cols
                break
        if good_cols is not None:
            add(Candidate(
                "row_marker_recolor",
                (("marker_col", good_cols), ("shape_color", shape)),
            ))

    # cross_stamp: marcatore = colore singleton isolato; valori per
    # direzione derivati dalle celle adiacenti nei demo.
    for marker in common_input:
        stamp: dict[tuple[int, int], set] = {}
        consistent = True
        for source, expected in demos:
            height, width = len(source), len(source[0])
            centers = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if source[r][c] == marker
            ]
            if not centers:
                consistent = False
                break
            for r, c in centers:
                if any(source[nr][nc] == marker
                       for nr, nc in _neighbors4(height, width, r, c)):
                    consistent = False
                    break
                for dr, dc in CROSS_DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < len(expected) and 0 <= nc < len(expected[0]):
                        stamp.setdefault((dr, dc), set()).add(
                            expected[nr][nc])
            if not consistent:
                break
        if not consistent or len(stamp) < 2:
            continue
        if any(len(options) != 1 for options in stamp.values()):
            continue
        stamp_map = tuple(sorted(
            (direction, next(iter(options)))
            for direction, options in stamp.items()
        ))
        add(Candidate(
            "cross_stamp",
            (("marker", marker), ("stamp", stamp_map)),
        ))

    # count_row: unico colore non background contato in una riga
    # (background derivato dai demo; colore e conteggio a runtime).
    for bg in common_input:
        add(Candidate("count_row", (("background", bg),)))

    # box_stamp: cornice = colore nuovo nei demo; target = colore di
    # input conservato al centro degli anelli.
    for frame in new_colors:
        for target in common_input:
            add(Candidate(
                "box_stamp",
                (("frame", frame), ("target", target)),
            ))

    # noisy_region_pick: rumore = colore isolato derivato dai demo
    # (a runtime); anche candidato non vincolato per rumore che non e'
    # comune a tutti i demo.
    add(Candidate("noisy_region_pick", ()))
    for noise in common_input:
        if noise in backgrounds:
            continue
        add(Candidate("noisy_region_pick", (("noise", noise),)))

    # bounce_fill: riempimento = colore nuovo; marcatore = singleton.
    for fill in new_colors:
        for marker in common_input:
            if marker in backgrounds:
                continue
            add(Candidate(
                "bounce_fill",
                (("fill", fill), ("marker", marker)),
            ))

    return tuple(sorted(
        result.values(), key=lambda item: (item.graph, repr(item.bindings))
    ))
