"""A1-G4 algebra: candidati parametrici derivati dai soli demo.

Famiglia di generatori specializzati per classi di trasformazione
(struttura del metodo di A100/A2: binding derivati dai demo, filtro
demo-esatto nel costruttore, esecuzione fail-closed).

Grafi:
  - transpose_block_select: l'input e' una pila di N blocchi uguali;
    uscita = primo blocco non uguale alla propria trasposta;
  - frame_shape_inset: cornice rettangolare cava + figura; uscita =
    ritaglio della cornice con la figura scalata a riempire l'interno;
  - ring_to_plus: ogni anello 3x3 (bordo di 8 celle) e' coperto e
    sostituito da una croce di 5 celle centrata nel buco;
  - lcorner_ray: ogni oggetto a L (bbox meno un angolo) emette una
    diagonale dall'angolo mancante nella direzione mancante-gomito;
    l'angolo mancante resta sfondo;
  - recolor_by_size: ogni oggetto e' ricolorato in base alla taglia
    (mappa taglia->colore derivata dai demo);
  - macro_mask_small_grid: pattern macro-bloccoso + griglia piccola;
    uscita = griglia piccola mascherata dal pattern downsamplato;
  - border_fill: le celle sul bordo della griglia sono dipinte;
  - bbox_crop_flip: ritaglio del bbox del non-sfondo + simmetria;
  - mirror_band_shift: l'oggetto mobile e' specchiato in orizzontale e
    spostato nella banda laterale indicata dall'oggetto puntatore;
  - half_crop: meta' superiore (input verticale) o sinistra (orizz.);
  - rot_quad_tile: tassellazione 2x2 con le 4 rotazioni dell'input.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]
EXEC_ERRORS = (ValueError, IndexError, KeyError, StopIteration)

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = _ORTH + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _background(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    return min(counts, key=lambda value: (-counts[value], value))


def _least_color(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    return min(counts, key=lambda value: (counts[value], value))


def _objects(
    grid: Grid,
    *,
    without_bg: bool = True,
    diagonal: bool = False,
    univalued: bool = True,
) -> list[frozenset[tuple[int, int]]]:
    """Componenti connesse (4- o 8-connettivita'), senza sfondo."""
    height, width = len(grid), len(grid[0])
    bg = _background(grid) if without_bg else None
    steps = _DIAG if diagonal else _ORTH
    seen: set[tuple[int, int]] = set()
    result: list[frozenset[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if (r, c) in seen or grid[r][c] == bg:
                continue
            cells: set[tuple[int, int]] = set()
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                cr, cc = stack.pop()
                cells.add((cr, cc))
                current = grid[cr][cc]
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if (nr, nc) in seen:
                        continue
                    value = grid[nr][nc]
                    if value == bg:
                        continue
                    if univalued and value != current:
                        continue
                    seen.add((nr, nc))
                    stack.append((nr, nc))
            result.append(frozenset(cells))
    return result


def _bbox(cells):
    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    right = max(c for _, c in cells)
    return top, left, bottom, right


def _crop(grid: Grid, cells) -> Grid:
    top, left, bottom, right = _bbox(cells)
    return tuple(row[left:right + 1] for row in grid[top:bottom + 1])


def _transpose(grid: Grid) -> Grid:
    return tuple(tuple(column) for column in zip(*grid))


def _vmirror(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in grid)


def _hmirror(grid: Grid) -> Grid:
    return tuple(reversed(grid))


def _rot180(grid: Grid) -> Grid:
    return _vmirror(_hmirror(grid))


def _rot90(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    return tuple(
        tuple(grid[height - 1 - j][i] for j in range(height))
        for i in range(width)
    )


def _rot270(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    return tuple(
        tuple(grid[j][width - 1 - i] for j in range(height))
        for i in range(width)
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
        if self.graph == "transpose_block_select":
            blocks = values["blocks"]
            height = len(grid)
            if height % blocks:
                raise ValueError("height not divisible")
            block_h = height // blocks
            for i in range(blocks):
                chunk = grid[i * block_h:(i + 1) * block_h]
                if chunk != _transpose(chunk):
                    return chunk
            raise ValueError("no asymmetric block")
        if self.graph == "frame_shape_inset":
            background = _background(grid)
            objs = _objects(grid, diagonal=True)
            if len(objs) != 2:
                raise ValueError("expected two objects")
            frames = []
            for obj in objs:
                top, left, bottom, right = _bbox(obj)
                ring = {
                    (r, c)
                    for r in range(top, bottom + 1)
                    for c in range(left, right + 1)
                    if r == top or r == bottom or c == left or c == right
                }
                if obj == ring and bottom - top >= 2 and right - left >= 2:
                    frames.append(obj)
            if len(frames) != 1:
                raise ValueError("no unique frame")
            frame = frames[0]
            shape = next(obj for obj in objs if obj is not frame)
            top, left, bottom, right = _bbox(frame)
            out_h = bottom - top + 1
            out_w = right - left + 1
            in_h, in_w = out_h - 2, out_w - 2
            s_top, s_left, s_bottom, s_right = _bbox(shape)
            s_h = s_bottom - s_top + 1
            s_w = s_right - s_left + 1
            if in_h % s_h or in_w % s_w:
                raise ValueError("interior not divisible")
            rf = in_h // s_h
            cf = in_w // s_w
            if rf != cf or rf < 1:
                raise ValueError("bad scale factor")
            frame_color = grid[min(frame)[0]][min(frame)[1]]
            shape_color = grid[min(shape)[0]][min(shape)[1]]
            out = [
                [
                    frame_color
                    if r in (0, out_h - 1) or c in (0, out_w - 1)
                    else background
                    for c in range(out_w)
                ]
                for r in range(out_h)
            ]
            for r, c in shape:
                for di in range(rf):
                    for dj in range(cf):
                        out[1 + (r - s_top) * rf + di][
                            1 + (c - s_left) * cf + dj
                        ] = shape_color
            return tuple(tuple(row) for row in out)
        if self.graph == "ring_to_plus":
            color = values["color"]
            background = _background(grid)
            objs = _objects(grid)
            rings = []
            for obj in objs:
                top, left, bottom, right = _bbox(obj)
                if bottom - top != 2 or right - left != 2:
                    continue
                center = (top + 1, left + 1)
                bbox_cells = {
                    (r, c)
                    for r in range(top, bottom + 1)
                    for c in range(left, right + 1)
                }
                if obj == bbox_cells - {center}:
                    rings.append((obj, center))
            if not rings:
                raise ValueError("no rings")
            out = [list(row) for row in grid]
            for obj, _ in rings:
                for r, c in obj:
                    out[r][c] = background
            height, width = len(grid), len(grid[0])
            for _, (cr, cc) in rings:
                for dr, dc in ((0, 0),) + _ORTH:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        raise ValueError("plus off grid")
                    out[nr][nc] = color
            return tuple(tuple(row) for row in out)
        if self.graph == "lcorner_ray":
            background = _background(grid)
            color = _least_color(grid)
            if color == background:
                raise ValueError("least color is background")
            height, width = len(grid), len(grid[0])
            objs = _objects(grid)
            targets = [
                obj
                for obj in objs
                if grid[min(obj)[0]][min(obj)[1]] == color
            ]
            if not targets:
                raise ValueError("no objects of least color")
            rays: set[tuple[int, int]] = set()
            holes: set[tuple[int, int]] = set()
            for obj in targets:
                top, left, bottom, right = _bbox(obj)
                backdrop = {
                    (r, c)
                    for r in range(top, bottom + 1)
                    for c in range(left, right + 1)
                }
                missing = backdrop - obj
                if len(missing) != 1:
                    raise ValueError("not an L corner object")
                (mr, mc), = missing
                elbows = sorted(
                    cell
                    for cell in obj
                    if sum(
                        (cell[0] + dr, cell[1] + dc) in obj
                        for dr, dc in _ORTH
                    ) == 2
                )
                if not elbows:
                    raise ValueError("no elbow cell")
                er, ec = elbows[0]
                dr, dc = mr - er, mc - ec
                if abs(dr) != 1 or abs(dc) != 1:
                    raise ValueError("elbow not diagonal to hole")
                r, c = mr, mc
                while 0 <= r < height and 0 <= c < width:
                    rays.add((r, c))
                    r += dr
                    c += dc
                holes.add((mr, mc))
            out = [list(row) for row in grid]
            for r, c in rays:
                out[r][c] = color
            for r, c in holes:
                out[r][c] = background
            return tuple(tuple(row) for row in out)
        if self.graph == "recolor_by_size":
            mapping = dict(values["mapping"])
            background = _background(grid)
            objs = _objects(grid)
            out = [list(row) for row in grid]
            for obj in objs:
                if len(obj) not in mapping:
                    raise ValueError("unmapped object size")
                for r, c in obj:
                    out[r][c] = mapping[len(obj)]
            return tuple(tuple(row) for row in out)
        if self.graph == "macro_mask_small_grid":
            background = _background(grid)
            objs = _objects(grid, diagonal=True, univalued=False)
            if len(objs) != 2:
                raise ValueError("expected two objects")
            big, small = sorted(objs, key=len, reverse=True)
            if len(big) == len(small):
                raise ValueError("equal sized objects")
            small_crop = _crop(grid, small)
            big_crop = _crop(grid, big)
            k = len(small_crop)
            if k < 2 or any(len(row) != k for row in small_crop):
                raise ValueError("small grid not square")
            bh, bw = len(big_crop), len(big_crop[0])
            if bh != bw or bh % k:
                raise ValueError("big crop not k*b square")
            scale = bh // k
            if scale < 2:
                raise ValueError("scale too small")
            return tuple(
                tuple(
                    small_crop[i][j]
                    if big_crop[i * scale][j * scale] != background
                    else background
                    for j in range(k)
                )
                for i in range(k)
            )
        if self.graph == "border_fill":
            color = values["color"]
            height, width = len(grid), len(grid[0])
            return tuple(
                tuple(
                    color
                    if r in (0, height - 1) or c in (0, width - 1)
                    else cell
                    for c, cell in enumerate(row)
                )
                for r, row in enumerate(grid)
            )
        if self.graph == "bbox_crop_flip":
            mode = values["mode"]
            background = _background(grid)
            cells = {
                (r, c)
                for r, row in enumerate(grid)
                for c, cell in enumerate(row)
                if cell != background
            }
            if not cells:
                raise ValueError("empty foreground")
            cropped = _crop(grid, cells)
            if mode == "none":
                return cropped
            if mode == "vmirror":
                return _vmirror(cropped)
            if mode == "hmirror":
                return _hmirror(cropped)
            if mode == "rot180":
                return _rot180(cropped)
            raise ValueError("unknown flip mode")
        if self.graph == "mirror_band_shift":
            mover = values["mover"]
            pointer = values["pointer"]
            mover_cells = {
                (r, c)
                for r, row in enumerate(grid)
                for c, cell in enumerate(row)
                if cell == mover
            }
            pointer_cells = {
                (r, c)
                for r, row in enumerate(grid)
                for c, cell in enumerate(row)
                if cell == pointer
            }
            if not mover_cells or not pointer_cells:
                raise ValueError("missing mover or pointer")
            _, left, _, right = _bbox(mover_cells)
            axis = left + right
            band = right - left + 1
            p_top, p_left, _, _ = _bbox(pointer_cells)
            sign = -1 if grid[p_top][p_left] == pointer else 1
            height, width = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for r, c in mover_cells:
                nr, nc = r, axis - c + sign * band
                if not (0 <= nr < height and 0 <= nc < width):
                    raise ValueError("mirrored copy off grid")
                out[nr][nc] = mover
            return tuple(tuple(row) for row in out)
        if self.graph == "half_crop":
            height, width = len(grid), len(grid[0])
            if height > width:
                if height % 2:
                    raise ValueError("odd height")
                return grid[:height // 2]
            if width > height:
                if width % 2:
                    raise ValueError("odd width")
                return tuple(row[:width // 2] for row in grid)
            raise ValueError("square grid")
        if self.graph == "rot_quad_tile":
            height, width = len(grid), len(grid[0])
            if height != width:
                raise ValueError("not square")
            rot90 = _rot90(grid)
            rot180 = _rot180(grid)
            rot270 = _rot270(grid)
            return tuple(
                tuple(list(grid[i]) + list(rot90[i]))
                for i in range(height)
            ) + tuple(
                tuple(list(rot270[i]) + list(rot180[i]))
                for i in range(height)
            )
        raise ValueError("unknown a1 g4 graph")


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

    # --- transpose_block_select: N = rapporto altezze -------------------
    ratios = None
    for source, expected in demos:
        here = set()
        if (
            expected
            and len(source[0]) == len(expected[0])
            and len(source) % len(expected) == 0
        ):
            factor = len(source) // len(expected)
            if factor >= 2:
                here.add(factor)
        ratios = here if ratios is None else ratios & here
    for blocks in sorted(ratios or ()):
        add(Candidate("transpose_block_select", (("blocks", blocks),)))

    # --- grafi a struttura fissa (binding derivati in execute) ----------
    add(Candidate("frame_shape_inset", ()))
    add(Candidate("lcorner_ray", ()))
    add(Candidate("macro_mask_small_grid", ()))
    add(Candidate("half_crop", ()))
    add(Candidate("rot_quad_tile", ()))
    for mode in ("none", "vmirror", "hmirror", "rot180"):
        add(Candidate("bbox_crop_flip", (("mode", mode),)))

    # --- ring_to_plus / border_fill: colori nuovi nei demo --------------
    new_colors = None
    for source, expected in demos:
        palette_in = {cell for row in source for cell in row}
        palette_out = {cell for row in expected for cell in row}
        here = palette_out - palette_in
        new_colors = here if new_colors is None else new_colors & here
    for color in sorted(new_colors or ()):
        add(Candidate("ring_to_plus", (("color", color),)))
        add(Candidate("border_fill", (("color", color),)))

    # --- recolor_by_size: mappa taglia -> colore dai demo ---------------
    size_map = None
    for source, expected in demos:
        if len(source) != len(expected) or len(source[0]) != len(expected[0]):
            size_map = {}
            break
        here: dict[int, int] = {}
        ok = True
        for obj in _objects(source):
            colors = {expected[r][c] for r, c in obj}
            if len(colors) != 1:
                ok = False
                break
            color = colors.pop()
            if color == _background(source):
                ok = False
                break
            if here.setdefault(len(obj), color) != color:
                ok = False
                break
        if not ok:
            size_map = {}
            break
        if size_map is None:
            size_map = here
        else:
            size_map = {
                size: color
                for size, color in size_map.items()
                if here.get(size, color) == color
            }
    if size_map:
        add(Candidate(
            "recolor_by_size",
            (("mapping", tuple(sorted(size_map.items()))),),
        ))

    # --- mirror_band_shift: coppia mobili/puntatore dai demo ------------
    palette = sorted({
        cell for source, _ in demos for row in source for cell in row
    })
    for mover in palette:
        for pointer in palette:
            if mover == pointer:
                continue
            add(Candidate(
                "mirror_band_shift",
                (("mover", mover), ("pointer", pointer)),
            ))

    return tuple(sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings))))
