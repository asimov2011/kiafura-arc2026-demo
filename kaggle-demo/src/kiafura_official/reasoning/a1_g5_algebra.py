"""A1 g5 algebra: famiglia di generatori parametrici dai soli demo.

Grafi (ognuno = classe di trasformazione; binding derivati dai demo):
  - nested_depth_reverse: rettangoli concentrici monocromi, i livelli di
    profondita' sono scambiati in ordine inverso (interno <-> esterno);
  - blob_intrusion_crosshatch: in una griglia rumorosa il rettangolo del
    colore dominante (componente massima) contiene intrusioni; output =
    rettangolo con righe/colonne delle intrusioni marcate dal colore di
    intrusione;
  - bbox_tile_anchored: il blocco di celle non-sfondo e' una tile che si
    ripete ancorata alla sua posizione a coprire tutta la griglia;
  - max_dot_blob_crop: ritaglio del rettangolo (blocco non-sfondo) con il
    massimo numero di celle "punto" (colore di minoranza del blocco);
  - fractal_block_enumeration: pattern P (due colori); output = griglia di
    blocchi S×S (S = numero di celle sfondo di P) con una copia di P per
    ogni cella piena di P, in ordine di lettura (row-major);
  - line_grid_corner_fills: griglia partizionata da righe/colonne piene
    del colore linea; tre regioni riempite: angolo alto-sinistro, regione
    centrale (banda che contiene il centro), angolo basso-destro; colori
    di riempimento come binding dai demo;
  - minority_recolor: tutte le celle non del colore modale sono ricolorate
    col colore di riempimento (binding dai demo);
  - frame_shape_nearest_side: quattro barre colorate definiscono una
    cornice attorno a una regione vuota; la forma sparsa (bbox uguale alla
    regione) e' disegnata dentro la cornice: ogni cella della forma prende
    il colore del lato piu' vicino (se unico), altrimenti il colore della
    forma; angoli della cornice = sfondo;
  - suffix_pattern_tile: prefisso di righe uniformi (sfondo) poi un
    blocco di righe non uniformi; il blocco si ripete dall'alto ancorato
    alla sua posizione;
  - centered_ring_completion: celle sparse su anelli quadrati centrati
    nella griglia; ogni cella d'angolo definisce il colore degli angoli
    dell'anello, ogni cella di lato il colore alternato (passo 2 dagli
    angoli) dei lati; output = tutti gli anelli completati.
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


def _panel_content_mirror(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    sep_rows = [
        r for r in range(height)
        if len(set(grid[r])) == 1 and grid[r][0] != 0
    ]
    sep_cols = [
        c for c in range(width)
        if len({grid[r][c] for r in range(height)}) == 1
        and grid[0][c] != 0
    ]
    if not sep_rows or sep_rows != sep_cols:
        raise ValueError("misaligned separators")
    sep = grid[sep_rows[0]][0]
    if any(grid[r][0] != sep for r in sep_rows) \
            or any(grid[0][c] != sep for c in sep_cols):
        raise ValueError("mixed separator colors")
    n = len(sep_rows)
    def gaps(indices, total):
        starts = [0] + [i + 1 for i in indices]
        ends = indices + [total]
        return [b - a for a, b in zip(starts, ends)]
    row_spans = gaps(sep_rows, height)
    col_spans = gaps(sep_cols, width)
    if any(span < 1 for span in row_spans + col_spans) \
            or any(span != row_spans[0] for span in row_spans) \
            or any(span != col_spans[0] for span in col_spans):
        raise ValueError("uneven panels")
    out = []
    row_starts = [0] + [i + 1 for i in sep_rows]
    col_starts = [0] + [i + 1 for i in sep_cols]
    for i in range(n + 1):
        row = []
        for j in range(n + 1):
            panel = {
                grid[r][c]
                for r in range(row_starts[i], row_starts[i] + row_spans[0])
                for c in range(col_starts[j], col_starts[j] + col_spans[0])
            }
            if len(panel) == 1:
                value = panel.pop()
                if value not in (0, sep):
                    row.append(value)
                    continue
            row.append(0)
        out.append(tuple(reversed(row)))
    return tuple(out)


def _largest_component(grid: Grid) -> tuple[int, list[tuple[int, int]]]:
    """Largest 4-connected monochromatic component (unique maximum)."""
    height, width = len(grid), len(grid[0])
    seen = [[False] * width for _ in range(height)]
    components: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(height):
        for c in range(width):
            if seen[r][c]:
                continue
            value = grid[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < height and 0 <= ny < width \
                            and not seen[nx][ny] and grid[nx][ny] == value:
                        seen[nx][ny] = True
                        stack.append((nx, ny))
            components.append((value, comp))
    if not components:
        raise ValueError("empty grid")
    max_size = max(len(comp) for _, comp in components)
    maxima = [(value, comp) for value, comp in components
              if len(comp) == max_size]
    if len(maxima) != 1:
        raise ValueError("ambiguous largest component")
    return maxima[0]


def _components_8(grid: Grid, colors: frozenset[int]):
    """8-connected components restricted to `colors`."""
    height, width = len(grid), len(grid[0])
    seen = [[False] * width for _ in range(height)]
    result = []
    for r in range(height):
        for c in range(width):
            if seen[r][c] or grid[r][c] not in colors:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width \
                                and not seen[nx][ny] \
                                and grid[nx][ny] in colors:
                            seen[nx][ny] = True
                            stack.append((nx, ny))
            result.append(comp)
    return result


def _blob_rectangle(grid: Grid) -> tuple[int, int, int, int, int]:
    """(color, r0, r1, c0, c1) of the noisy rectangle: largest component
    bbox trimmed of sparse noise rows/cols (fail-closed on ambiguity)."""
    value, comp = _largest_component(grid)
    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    changed = True
    while changed:
        changed = False
        for r in sorted(rows):
            count = sum(1 for c in cols if grid[r][c] == value)
            if count * 2 < len(cols):
                rows.discard(r)
                changed = True
        for c in sorted(cols):
            count = sum(1 for r in rows if grid[r][c] == value)
            if count * 2 < len(rows):
                cols.discard(c)
                changed = True
    if not rows or not cols:
        raise ValueError("rectangle vanished")
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if rows != set(range(r0, r1 + 1)) or cols != set(range(c0, c1 + 1)):
        raise ValueError("fragmented rectangle")
    return value, r0, r1, c0, c1


def _line_grid_parse(grid: Grid):
    """(line_color, row_bands, col_bands): bands are maximal contiguous
    non-line row/col intervals; fail-closed if structure is ambiguous."""
    height, width = len(grid), len(grid[0])
    line_colors = set()
    for value in {cell for row in grid for cell in row}:
        has_row = any(
            all(grid[r][c] == value for c in range(width))
            for r in range(height)
        )
        has_col = any(
            all(grid[r][c] == value for r in range(height))
            for c in range(width)
        )
        if has_row and has_col:
            line_colors.add(value)
    if len(line_colors) != 1:
        raise ValueError("no unique line color")
    line = line_colors.pop()
    line_rows = {
        r for r in range(height)
        if all(grid[r][c] == line for c in range(width))
    }
    line_cols = {
        c for c in range(width)
        if all(grid[r][c] == line for r in range(height))
    }

    def bands(lines: set[int], size: int):
        out, start = [], None
        for i in range(size):
            if i in lines:
                if start is not None:
                    out.append((start, i - 1))
                    start = None
            elif start is None:
                start = i
        if start is not None:
            out.append((start, size - 1))
        return out

    row_bands = bands(line_rows, height)
    col_bands = bands(line_cols, width)
    if not row_bands or not col_bands:
        raise ValueError("no regions")
    return line, row_bands, col_bands


def _frame_parse(grid: Grid):
    """(bg, top, bottom, left, right, interior dims, shape cells, shape
    color) of the bar-frame configuration; unique or fail-closed."""
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    horiz, vert = [], []
    for r in range(height):
        c = 0
        while c < width:
            if grid[r][c] != background:
                value = grid[r][c]
                c2 = c
                while c2 < width and grid[r][c2] == value:
                    c2 += 1
                if c2 - c >= 2:
                    horiz.append((r, c, c2 - 1, value))
                c = c2
            else:
                c += 1
    for c in range(width):
        r = 0
        while r < height:
            if grid[r][c] != background:
                value = grid[r][c]
                r2 = r
                while r2 < height and grid[r2][c] == value:
                    r2 += 1
                if r2 - r >= 2:
                    vert.append((c, r, r2 - 1, value))
                r = r2
            else:
                r += 1
    configs = []
    for top_r, c0, c1, top_v in horiz:
        for bot_r, c0b, c1b, bot_v in horiz:
            if bot_r <= top_r + 1 or c0b != c0 or c1b != c1:
                continue
            r0, r1 = top_r + 1, bot_r - 1
            left = [
                item for item in vert
                if item[0] == c0 - 1 and item[1] == r0 and item[2] == r1
            ]
            right = [
                item for item in vert
                if item[0] == c1 + 1 and item[1] == r0 and item[2] == r1
            ]
            if len(left) != 1 or len(right) != 1:
                continue
            left_v, right_v = left[0][3], right[0][3]
            if len({top_v, bot_v, left_v, right_v}) != 4:
                continue
            if any(
                grid[r][c] != background
                for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            ):
                continue
            rest = [
                (r, c)
                for r in range(height) for c in range(width)
                if grid[r][c] != background
                and not (r == top_r and c0 <= c <= c1)
                and not (r == bot_r and c0 <= c <= c1)
                and not (c == c0 - 1 and r0 <= r <= r1)
                and not (c == c1 + 1 and r0 <= r <= r1)
            ]
            if not rest:
                continue
            shape_colors = {grid[r][c] for r, c in rest}
            if len(shape_colors) != 1:
                continue
            rr0 = min(r for r, _ in rest)
            rr1 = max(r for r, _ in rest)
            cc0 = min(c for _, c in rest)
            cc1 = max(c for _, c in rest)
            if rr1 - rr0 + 1 != r1 - r0 + 1 or cc1 - cc0 + 1 != c1 - c0 + 1:
                continue
            shape = {(r - rr0, c - cc0) for r, c in rest}
            configs.append((
                background, top_v, bot_v, left_v, right_v,
                r1 - r0 + 1, c1 - c0 + 1, shape, shape_colors.pop(),
            ))
    if len(configs) != 1:
        raise ValueError("ambiguous frame configuration")
    return configs[0]


def _ring_parse(grid: Grid):
    """(bg, {depth: corner_color}, {depth: side_color}); cells must lie on
    their ring boundary; consistent colors; fail-closed otherwise."""
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    corners: dict[int, int] = {}
    sides: dict[int, int] = {}
    for r in range(height):
        for c in range(width):
            value = grid[r][c]
            if value == background:
                continue
            depth = min(r, c, height - 1 - r, width - 1 - c)
            if not (r in (depth, height - 1 - depth)
                    or c in (depth, width - 1 - depth)):
                raise ValueError("cell off ring")
            is_corner = (
                r in (depth, height - 1 - depth)
                and c in (depth, width - 1 - depth)
            )
            target = corners if is_corner else sides
            if depth in target and target[depth] != value:
                raise ValueError("conflicting ring color")
            target[depth] = value
    if any(depth not in corners for depth in sides):
        raise ValueError("side cell without corner")
    return background, corners, sides


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]
    _cost: int = 1

    def cost(self, grid: Grid) -> int:
        return self._cost + len(grid) * len(grid[0])

    def execute(self, grid: Grid) -> Grid:
        values = dict(self.bindings)
        if self.graph == "panel_content_mirror":
            return _panel_content_mirror(grid)
        if self.graph == "nested_depth_reverse":
            height, width = len(grid), len(grid[0])
            if min(height, width) < 2:
                raise ValueError("grid too small")
            colors: dict[int, set[int]] = {}
            for r in range(height):
                for c in range(width):
                    depth = min(r, c, height - 1 - r, width - 1 - c)
                    colors.setdefault(depth, set()).add(grid[r][c])
            if any(len(found) != 1 for found in colors.values()):
                raise ValueError("mixed depth ring")
            by_depth = {d: found.pop() for d, found in colors.items()}
            if len(set(by_depth.values())) < 2:
                raise ValueError("single color")
            deepest = max(by_depth)
            return tuple(
                tuple(
                    by_depth[deepest - min(r, c, height - 1 - r, width - 1 - c)]
                    for c in range(width)
                )
                for r in range(height)
            )
        if self.graph == "blob_intrusion_crosshatch":
            value, r0, r1, c0, c1 = _blob_rectangle(grid)
            intrusion_colors = set()
            mark_rows = set()
            mark_cols = set()
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if grid[r][c] != value:
                        intrusion_colors.add(grid[r][c])
                        mark_rows.add(r - r0)
                        mark_cols.add(c - c0)
            if len(intrusion_colors) != 1:
                raise ValueError("no unique intrusion color")
            intrusion = intrusion_colors.pop()
            height, width = r1 - r0 + 1, c1 - c0 + 1
            return tuple(
                tuple(
                    intrusion
                    if r in mark_rows or c in mark_cols
                    else value
                    for c in range(width)
                )
                for r in range(height)
            )
        if self.graph == "bbox_tile_anchored":
            height, width = len(grid), len(grid[0])
            background = _background(grid)
            cells = [
                (r, c)
                for r in range(height) for c in range(width)
                if grid[r][c] != background
            ]
            if not cells:
                raise ValueError("no tile")
            r0 = min(r for r, _ in cells)
            r1 = max(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            c1 = max(c for _, c in cells)
            tile_h, tile_w = r1 - r0 + 1, c1 - c0 + 1
            return tuple(
                tuple(
                    grid[r0 + (r - r0) % tile_h][c0 + (c - c0) % tile_w]
                    for c in range(width)
                )
                for r in range(height)
            )
        if self.graph == "max_dot_blob_crop":
            height, width = len(grid), len(grid[0])
            background = _largest_component(grid)[0]
            palette = {cell for row in grid for cell in row} - {background}
            if not palette:
                raise ValueError("no blobs")
            best = None
            for value in sorted(palette):
                for comp in _components_8(grid, frozenset({value})):
                    rr0 = min(r for r, _ in comp)
                    rr1 = max(r for r, _ in comp)
                    cc0 = min(c for _, c in comp)
                    cc1 = max(c for _, c in comp)
                    others = {
                        grid[r][c]
                        for r in range(rr0, rr1 + 1)
                        for c in range(cc0, cc1 + 1)
                        if grid[r][c] != value
                    }
                    if not others or background in others or len(others) != 1:
                        continue
                    dot = others.pop()
                    dot_count = sum(
                        1
                        for r in range(rr0, rr1 + 1)
                        for c in range(cc0, cc1 + 1)
                        if grid[r][c] == dot
                    )
                    value_count = sum(
                        1
                        for r in range(rr0, rr1 + 1)
                        for c in range(cc0, cc1 + 1)
                        if grid[r][c] == value
                    )
                    if dot_count < 1 or dot_count >= value_count:
                        continue
                    if best is not None and dot_count == best[0]:
                        raise ValueError("tie on dot count")
                    if best is None or dot_count > best[0]:
                        best = (dot_count, (rr0, rr1, cc0, cc1))
            if best is None or best[0] < 1:
                raise ValueError("no dotted blob")
            _, (rr0, rr1, cc0, cc1) = best
            return tuple(
                tuple(grid[r][c] for c in range(cc0, cc1 + 1))
                for r in range(rr0, rr1 + 1)
            )
        if self.graph == "fractal_block_enumeration":
            pattern_h, pattern_w = len(grid), len(grid[0])
            palette = {cell for row in grid for cell in row}
            if len(palette) != 2:
                raise ValueError("pattern palette arity")
            background = values["background"]
            slot = sum(
                1 for row in grid for cell in row if cell == background
            )
            if slot < 1:
                raise ValueError("no background slot")
            filled = [
                (r, c)
                for r in range(pattern_h)
                for c in range(pattern_w)
                if grid[r][c] != background
            ]
            if len(filled) > slot * slot:
                raise ValueError("too many blocks")
            out = [
                [background] * (pattern_w * slot)
                for _ in range(pattern_h * slot)
            ]
            for index, (r, c) in enumerate(filled):
                block_r, block_c = divmod(index, slot)
                for i in range(pattern_h):
                    for j in range(pattern_w):
                        out[block_r * pattern_h + i][block_c * pattern_w + j] \
                            = grid[i][j]
            return tuple(tuple(row) for row in out)
        if self.graph == "line_grid_corner_fills":
            _, row_bands, col_bands = _line_grid_parse(grid)
            color_tl = values["color_tl"]
            color_mid = values["color_mid"]
            color_br = values["color_br"]
            height, width = len(grid), len(grid[0])

            def band_at(bands, index):
                for k, (a, b) in enumerate(bands):
                    if a <= index <= b:
                        return k
                raise ValueError("index on line")

            mid_r = band_at(row_bands, height // 2)
            mid_c = band_at(col_bands, width // 2)
            targets = {(0, 0), (mid_r, mid_c),
                       (len(row_bands) - 1, len(col_bands) - 1)}
            if len(targets) != 3:
                raise ValueError("regions collapse")
            out = [list(row) for row in grid]
            for target, color in (
                ((0, 0), color_tl),
                ((mid_r, mid_c), color_mid),
                ((len(row_bands) - 1, len(col_bands) - 1), color_br),
            ):
                ra, rb = row_bands[target[0]]
                ca, cb = col_bands[target[1]]
                for r in range(ra, rb + 1):
                    for c in range(ca, cb + 1):
                        out[r][c] = color
            return tuple(tuple(row) for row in out)
        if self.graph == "minority_recolor":
            paint = values["paint"]
            background = _background(grid)
            return tuple(
                tuple(paint if cell != background else cell for cell in row)
                for row in grid
            )
        if self.graph == "frame_shape_nearest_side":
            (background, top_v, bot_v, left_v, right_v,
             inner_h, inner_w, shape, shape_color) = _frame_parse(grid)
            side_color = {"t": top_v, "b": bot_v, "l": left_v, "r": right_v}
            interior = [[background] * inner_w for _ in range(inner_h)]
            for r, c in shape:
                dists = ((r, "t"), (inner_h - 1 - r, "b"),
                         (c, "l"), (inner_w - 1 - c, "r"))
                nearest = min(dist for dist, _ in dists)
                sides = [s for dist, s in dists if dist == nearest]
                interior[r][c] = (
                    shape_color if len(sides) != 1
                    else side_color[sides[0]]
                )
            out = [[background] * (inner_w + 2) for _ in range(inner_h + 2)]
            for c in range(1, inner_w + 1):
                out[0][c] = top_v
                out[inner_h + 1][c] = bot_v
            for r in range(1, inner_h + 1):
                out[r][0] = left_v
                out[r][inner_w + 1] = right_v
            for r in range(inner_h):
                for c in range(inner_w):
                    out[r + 1][c + 1] = interior[r][c]
            return tuple(tuple(row) for row in out)
        if self.graph == "suffix_pattern_tile":
            height = len(grid)
            start = None
            for r in range(height):
                if len(set(grid[r])) > 1:
                    start = r
                    break
            if start is None or start < 1:
                raise ValueError("no pattern block")
            if any(len(set(grid[r])) == 1 for r in range(start, height)):
                raise ValueError("uniform row inside pattern")
            period = height - start
            return tuple(grid[start + (r - start) % period]
                         for r in range(height))
        if self.graph == "centered_ring_completion":
            height, width = len(grid), len(grid[0])
            background, corners, sides = _ring_parse(grid)
            out = [[background] * width for _ in range(height)]

            def paint(r, c, color):
                if not (0 <= r < height and 0 <= c < width):
                    raise ValueError("ring off grid")
                if out[r][c] != background and out[r][c] != color:
                    raise ValueError("ring collision")
                out[r][c] = color

            for depth, corner_color in corners.items():
                for r, c in (
                    (depth, depth), (depth, width - 1 - depth),
                    (height - 1 - depth, depth),
                    (height - 1 - depth, width - 1 - depth),
                ):
                    paint(r, c, corner_color)
                side_color = sides.get(depth)
                if side_color is None:
                    continue
                for offset in range(2, max(height, width), 2):
                    for r, c in (
                        (depth, depth + offset),
                        (depth, width - 1 - depth - offset),
                        (height - 1 - depth, depth + offset),
                        (height - 1 - depth, width - 1 - depth - offset),
                    ):
                        if depth < c < width - 1 - depth:
                            paint(r, c, side_color)
                    for r, c in (
                        (depth + offset, depth),
                        (depth + offset, width - 1 - depth),
                        (height - 1 - depth - offset, depth),
                        (height - 1 - depth - offset, width - 1 - depth),
                    ):
                        if depth < r < height - 1 - depth:
                            paint(r, c, side_color)
            return tuple(tuple(row) for row in out)
        raise ValueError("unknown a1 g5 graph")


def _line_grid_colors(demos: Demos):
    """Derive (tl, mid, br) fill colors from demos, or None."""
    agreed = None
    for source, expected in demos:
        try:
            _, row_bands, col_bands = _line_grid_parse(source)
        except EXEC_ERRORS:
            return None
        height, width = len(source), len(source[0])
        if len(expected) != height or len(expected[0]) != width:
            return None

        def band_at(bands, index):
            for k, (a, b) in enumerate(bands):
                if a <= index <= b:
                    return k
            return None

        mid_r = band_at(row_bands, height // 2)
        mid_c = band_at(col_bands, width // 2)
        if mid_r is None or mid_c is None:
            return None
        spots = ((0, 0), (mid_r, mid_c),
                 (len(row_bands) - 1, len(col_bands) - 1))
        if len(set(spots)) != 3:
            return None
        colors = []
        for bi, bj in spots:
            ra, rb = row_bands[bi]
            ca, cb = col_bands[bj]
            found = {
                expected[r][c]
                for r in range(ra, rb + 1) for c in range(ca, cb + 1)
            }
            if len(found) != 1:
                return None
            colors.append(found.pop())
        here = tuple(colors)
        if agreed is not None and here != agreed:
            return None
        agreed = here
    return agreed


def _minority_paint(demos: Demos):
    """Derive the recolor paint color from demos, or None."""
    agreed = None
    for source, expected in demos:
        if len(source) != len(expected) \
                or len(source[0]) != len(expected[0]):
            return None
        background = _background(source)
        found = set()
        for r in range(len(source)):
            for c in range(len(source[0])):
                if source[r][c] != background:
                    if expected[r][c] == source[r][c]:
                        return None
                    found.add(expected[r][c])
                elif expected[r][c] != background:
                    return None
        if len(found) != 1:
            return None
        here = found.pop()
        if agreed is not None and here != agreed:
            return None
        agreed = here
    return agreed


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

    add(Candidate("nested_depth_reverse", ()))
    add(Candidate("blob_intrusion_crosshatch", ()))
    add(Candidate("bbox_tile_anchored", ()))
    add(Candidate("max_dot_blob_crop", ()))
    add(Candidate("frame_shape_nearest_side", ()))
    add(Candidate("suffix_pattern_tile", ()))
    add(Candidate("centered_ring_completion", ()))
    add(Candidate("panel_content_mirror", ()))

    # fractal_block_enumeration: colore slot/background dai demo.
    common = set.intersection(*[
        {cell for row in source for cell in row}
        for source, _ in demos
    ]) if demos else set()
    for slot_color in sorted(common):
        add(Candidate(
            "fractal_block_enumeration",
            (("background", slot_color),),
        ))

    line_colors = _line_grid_colors(demos)
    if line_colors is not None:
        add(Candidate(
            "line_grid_corner_fills",
            (("color_br", line_colors[2]),
             ("color_mid", line_colors[1]),
             ("color_tl", line_colors[0])),
        ))

    paint = _minority_paint(demos)
    if paint is not None:
        add(Candidate("minority_recolor", (("paint", paint),)))

    return tuple(sorted(
        result.values(), key=lambda item: (item.graph, repr(item.bindings))
    ))
