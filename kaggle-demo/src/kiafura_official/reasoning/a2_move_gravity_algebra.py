"""A2 move/gravity algebra: candidati derivati dai soli demo.

Nessuna griglia letterale, nessun task ID, nessun programma completo in
memoria: ogni grafo e' una regola parametrica i cui binding sono ricavati
dalle dimostrazioni e verificati fail-closed su di esse prima
dell'emissione.

Due regimi osservati sui demo:
  - move: le componenti sorgente sono rimosse e ridipinte traslate;
  - add : l'input resta intatto e copie traslate (o nuovi punti) sono
          aggiunte.

Grafi:
  - constant_color_move   (move): offset costante per colore;
  - align_edge            (move): allineamento all'estremo del gruppo di
                          componenti con proiezione sovrapposta;
  - gravitate_translation (move|add): offset costante del singleton mover;
  - clone_toward_bar      (add): singleton allineate a una barra clonate
                          nella cella adiacente alla barra;
  - marker_rigid_slide_dock (move): colonne solo-sfondo eliminate; pezzi
                          con marcatori laterali scivolano verticalmente
                          a passi finche' non agganciano ancore/pezzi gia'
                          piazzati; i marcatori residui sono ricolorati
                          dal colore del pezzo fermo adiacente.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]

MAX_COMPONENTS = 26
MAX_BINDINGS = 400_000
EXEC_ERRORS = (ValueError, IndexError, KeyError, StopIteration)


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _background(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    return min(counts, key=lambda value: (-counts[value], value))


def _flood(
    grid: Grid, seed: tuple[int, int], color: int, background: int | None
) -> frozenset[tuple[int, int]]:
    """Connected component of ``seed``; background cells connect 4-way only
    when ``background`` is that color (used to segment added cells)."""
    height, width = _shape(grid)
    cells = {seed}
    queue = deque([seed])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < height
                and 0 <= nc < width
                and (nr, nc) not in cells
                and grid[nr][nc] == color
            ):
                cells.add((nr, nc))
                queue.append((nr, nc))
    return frozenset(cells)


def _components(grid: Grid, background: int) -> tuple[dict[str, Any], ...]:
    seen: set[tuple[int, int]] = set()
    items: list[dict[str, Any]] = []
    height, width = _shape(grid)
    for r in range(height):
        for c in range(width):
            if grid[r][c] == background or (r, c) in seen:
                continue
            cells = _flood(grid, (r, c), grid[r][c], None)
            seen |= cells
            rows = {p for p, _ in cells}
            cols = {p for _, p in cells}
            items.append(
                {
                    "color": grid[r][c],
                    "cells": cells,
                    "top": min(rows),
                    "bottom": max(rows),
                    "left": min(cols),
                    "right": max(cols),
                }
            )
    return tuple(sorted(items, key=lambda item: (item["top"], item["left"])))



def _color_groups(grid: Grid, background: int) -> dict[int, frozenset]:
    groups: dict[int, set] = {}
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value != background:
                groups.setdefault(value, set()).add((r, c))
    return {color: frozenset(cells) for color, cells in groups.items()}


def _extent(cells, edge: str) -> int:
    if edge == "bottom":
        return max(r for r, _ in cells)
    if edge == "top":
        return min(r for r, _ in cells)
    if edge == "right":
        return max(c for _, c in cells)
    return min(c for _, c in cells)


def _apply_moves(grid: Grid, moves: dict) -> Grid:
    height, width = _shape(grid)
    background = _background(grid)
    out = [list(row) for row in grid]
    colors = {}
    for cells, delta in moves.items():
        color = grid[next(iter(cells))[0]][next(iter(cells))[1]]
        colors[cells] = color
        for r, c in cells:
            out[r][c] = background
    for cells, delta in moves.items():
        dr, dc = delta
        for r, c in cells:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                raise ValueError("translation leaves the grid")
            out[nr][nc] = colors[cells]
    return tuple(tuple(row) for row in out)


def _apply_adds(grid: Grid, adds: dict) -> Grid:
    height, width = _shape(grid)
    out = [list(row) for row in grid]
    for (r, c), color in adds.items():
        if not (0 <= r < height and 0 <= c < width):
            raise ValueError("painted cell leaves the grid")
        out[r][c] = color
    return tuple(tuple(row) for row in out)


def _added_removed(
    source: Grid, expected: Grid
) -> tuple[dict[int, set[tuple[int, int]]], dict[int, set[tuple[int, int]]]] | None:
    if _shape(source) != _shape(expected):
        return None
    added: dict[int, set[tuple[int, int]]] = {}
    removed: dict[int, set[tuple[int, int]]] = {}
    for r, row in enumerate(source):
        for c, value in enumerate(row):
            other = expected[r][c]
            if other != value:
                added.setdefault(other, set()).add((r, c))
                removed.setdefault(value, set()).add((r, c))
    return added, removed


def _normalize(cells: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return frozenset((r - top, c - left) for r, c in cells)


def _infer_move(
    source: Grid, expected: Grid, background: int
) -> dict[frozenset, tuple[int, int]] | None:
    """Component translation assignment (delta or stay), else None."""
    if _shape(source) != _shape(expected):
        return None
    components = _components(source, background)
    if not components or len(components) > MAX_COMPONENTS:
        return None
    height, width = _shape(source)
    options: list[list[tuple[int, int]]] = []
    for comp in components:
        color = comp["color"]
        shape_cells = comp["cells"]
        anchor = min(shape_cells)
        deltas = {(0, 0)}
        for er in range(height):
            for ec in range(width):
                if expected[er][ec] != color:
                    continue
                dr, dc = er - anchor[0], ec - anchor[1]
                if (dr, dc) == (0, 0):
                    continue
                moved = {(r + dr, c + dc) for r, c in shape_cells}
                if all(
                    0 <= r < height and 0 <= c < width for r, c in moved
                ) and all(expected[r][c] == color for r, c in moved):
                    deltas.add((dr, dc))
        options.append(sorted(deltas))
    total = 1
    for opts in options:
        total *= len(opts)
        if total > MAX_BINDINGS:
            return None
    base = [list(row) for row in source]
    for assignment in product(*options):
        grid = [row[:] for row in base]
        for comp, (dr, dc) in zip(components, assignment):
            if (dr, dc) == (0, 0):
                continue
            for r, c in comp["cells"]:
                grid[r][c] = background
        for comp, (dr, dc) in zip(components, assignment):
            if (dr, dc) == (0, 0):
                continue
            for r, c in comp["cells"]:
                grid[r + dr][c + dc] = comp["color"]
        if tuple(tuple(row) for row in grid) == expected:
            return {
                comp["cells"]: delta
                for comp, delta in zip(components, assignment)
            }
    return None


def _contact_translation(
    mover: frozenset[tuple[int, int]], target: frozenset[tuple[int, int]]
) -> tuple[int, int] | None:
    """Smallest translation making mover 4-adjacent to target, no overlap."""
    best = None
    for dr in range(-30, 31):
        for dc in range(-30, 31):
            if dr == 0 and dc == 0:
                continue
            moved = {(r + dr, c + dc) for r, c in mover}
            if moved & target:
                continue
            contact = any(
                (r + ddr, c + ddc) in target
                for r, c in moved
                for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            if not contact:
                continue
            key = (abs(dr) + abs(dc), dr, dc)
            if best is None or key < best:
                best = key
    if best is None:
        return None
    _, dr, dc = best
    return dr, dc


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
        if self.graph == "constant_color_move":
            moves_by_color = dict(values["moves"])
            moves = {}
            for comp in _components(grid, background):
                if comp["color"] in moves_by_color:
                    moves[comp["cells"]] = moves_by_color[comp["color"]]
            if not moves:
                raise ValueError("no movable component")
            return _apply_moves(grid, moves)
        if self.graph == "align_edge":
            edge = values["edge"]
            groups = _color_groups(grid, background)
            if not groups:
                raise ValueError("no components")
            target = (
                max(_extent(cells, edge) for cells in groups.values())
                if edge in ("bottom", "right")
                else min(_extent(cells, edge) for cells in groups.values())
            )
            moves = {}
            for cells in groups.values():
                delta = target - _extent(cells, edge)
                if delta:
                    moves[cells] = (delta, 0) if edge in ("bottom", "top") else (0, delta)
            if not moves:
                raise ValueError("nothing to align")
            return _apply_moves(grid, moves)
        if self.graph == "move_all":
            delta = values["delta"]
            moves = {
                cells: delta
                for cells in _color_groups(grid, background).values()
            }
            if not moves:
                raise ValueError("no components")
            return _apply_moves(grid, moves)
        if self.graph == "connect_to_bar":
            bar_color = values["bar"]
            bar_cells = {
                (r, c)
                for r, row in enumerate(grid)
                for c, value in enumerate(row)
                if value == bar_color
            }
            if not bar_cells:
                raise ValueError("bar absent")
            bar_rows = {r for r, _ in bar_cells}
            bar_cols = {c for _, c in bar_cells}
            adds: dict[tuple[int, int], int] = {}
            for comp in _components(grid, background):
                if comp["color"] == bar_color or len(comp["cells"]) != 1:
                    continue
                (r, c) = next(iter(comp["cells"]))
                row_hit = r in bar_rows
                col_hit = c in bar_cols
                if row_hit == col_hit:
                    continue
                if row_hit:
                    cols = sorted(cc for rr, cc in bar_cells if rr == r)
                    if c < min(cols):
                        gap = [(r, cc) for cc in range(c + 1, min(cols))]
                    elif c > max(cols):
                        gap = [(r, cc) for cc in range(max(cols) + 1, c)]
                    else:
                        continue
                else:
                    rows = sorted(rr for rr, cc in bar_cells if cc == c)
                    if r < min(rows):
                        gap = [(rr, c) for rr in range(r + 1, min(rows))]
                    elif r > max(rows):
                        gap = [(rr, c) for rr in range(max(rows) + 1, r)]
                    else:
                        continue
                for cell in gap:
                    if grid[cell[0]][cell[1]] != background:
                        raise ValueError("path obstructed")
                    adds[cell] = comp["color"]
            if not adds:
                raise ValueError("no aligned singleton")
            return _apply_adds(grid, adds)
        if self.graph == "clone_toward_bar":
            bar_color = values["bar"]
            bar_cells = {
                (r, c)
                for r, row in enumerate(grid)
                for c, value in enumerate(row)
                if value == bar_color
            }
            bar_rows = {r for r, _ in bar_cells}
            bar_cols = {c for _, c in bar_cells}
            adds: dict[tuple[int, int], int] = {}
            for comp in _components(grid, background):
                if comp["color"] == bar_color or len(comp["cells"]) != 1:
                    continue
                (r, c) = next(iter(comp["cells"]))
                adjacent = any(
                    (r + dr, c + dc) in bar_cells
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
                if adjacent:
                    continue
                landing = None
                if r in bar_rows:
                    same_row = sorted(cc for rr, cc in bar_cells if rr == r)
                    if same_row and all(cc > c for cc in same_row):
                        landing = (r, min(same_row))
                    elif same_row and all(cc < c for cc in same_row):
                        landing = (r, max(same_row))
                elif c in bar_cols:
                    same_col = sorted(rr for rr, cc in bar_cells if cc == c)
                    if same_col and all(rr > r for rr in same_col):
                        landing = (min(same_col), c)
                    elif same_col and all(rr < r for rr in same_col):
                        landing = (max(same_col), c)
                if landing is None:
                    continue
                adds[landing] = comp["color"]
            if not adds:
                raise ValueError("no aligned singleton")
            return _apply_adds(grid, adds)
        if self.graph == "gravitate_contact":
            mover = values["mover"]
            target = values["target"]
            keep = values["keep"]
            components = _components(grid, background)
            target_comps = [c for c in components if c["color"] == target]
            if not target_comps:
                raise ValueError("target absent")
            moves = {}
            adds = {}
            for comp in components:
                if comp["color"] != mover:
                    continue
                best = None
                for tcomp in target_comps:
                    delta = _contact_translation(comp["cells"], tcomp["cells"])
                    if delta is None:
                        continue
                    key = (abs(delta[0]) + abs(delta[1]), delta)
                    if best is None or key < best[0]:
                        best = (key, delta)
                if best is None:
                    raise ValueError("no reachable target")
                delta = best[1]
                if delta == (0, 0):
                    continue
                if keep:
                    for r, c in comp["cells"]:
                        nr, nc = r + delta[0], c + delta[1]
                        if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
                            raise ValueError("translation leaves the grid")
                        adds[(nr, nc)] = mover
                else:
                    moves[comp["cells"]] = delta
            if not moves and not adds:
                raise ValueError("nothing to gravitate")
            if keep:
                return _apply_adds(grid, adds)
            return _apply_moves(grid, moves)
        if self.graph == "gravitate_step":
            mover = values["mover"]
            target = values["target"]
            components = _components(grid, background)
            target_comps = [c for c in components if c["color"] == target]
            if not target_comps:
                raise ValueError("target absent")
            moves = {}
            for comp in components:
                if comp["color"] != mover:
                    continue
                (r, c) = min(comp["cells"])
                best = None
                for tcomp in target_comps:
                    (tr, tc) = min(tcomp["cells"])
                    key = (abs(tr - r) + abs(tc - c), tr, tc)
                    if best is None or key < best[0]:
                        best = (key, tr, tc)
                _, tr, tc = best
                delta = ((tr - r) and (tr - r) // abs(tr - r),
                         (tc - c) and (tc - c) // abs(tc - c))
                if delta == (0, 0):
                    continue
                moves[comp["cells"]] = delta
            if not moves:
                raise ValueError("nothing to gravitate")
            return _apply_moves(grid, moves)
        if self.graph == "slide_until":
            direction = values["direction"]
            gap = values["gap"]
            height, width = _shape(grid)
            moves = {}
            for cells in _color_groups(grid, background).values():
                if direction == "right":
                    delta = (width - 1 - gap) - _extent(cells, "right")
                    step = (0, delta)
                elif direction == "left":
                    delta = gap - _extent(cells, "left")
                    step = (0, delta)
                elif direction == "down":
                    delta = (height - 1 - gap) - _extent(cells, "bottom")
                    step = (delta, 0)
                else:
                    delta = gap - _extent(cells, "top")
                    step = (delta, 0)
                if delta:
                    moves[cells] = step
            if not moves:
                raise ValueError("nothing to slide")
            return _apply_moves(grid, moves)
        if self.graph == "gravitate_contact":
            mover = values["mover"]
            target = values["target"]
            keep = values["keep"]
            components = _components(grid, background)
            target_comps = [c for c in components if c["color"] == target]
            if not target_comps:
                raise ValueError("target absent")
            moves = {}
            adds = {}
            for comp in components:
                if comp["color"] != mover:
                    continue
                best = None
                for tcomp in target_comps:
                    delta = _contact_translation(comp["cells"], tcomp["cells"])
                    if delta is None:
                        continue
                    key = (abs(delta[0]) + abs(delta[1]), delta)
                    if best is None or key < best[0]:
                        best = (key, delta)
                if best is None:
                    raise ValueError("no reachable target")
                delta = best[1]
                if delta == (0, 0):
                    continue
                if keep:
                    for r, c in comp["cells"]:
                        nr, nc = r + delta[0], c + delta[1]
                        if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
                            raise ValueError("translation leaves the grid")
                        adds[(nr, nc)] = mover
                else:
                    moves[comp["cells"]] = delta
            if not moves and not adds:
                raise ValueError("nothing to gravitate")
            if keep:
                return _apply_adds(grid, adds)
            return _apply_moves(grid, moves)
        if self.graph == "gravitate_step":
            mover = values["mover"]
            target = values["target"]
            components = _components(grid, background)
            target_comps = [c for c in components if c["color"] == target]
            if not target_comps:
                raise ValueError("target absent")
            moves = {}
            for comp in components:
                if comp["color"] != mover:
                    continue
                (r, c) = min(comp["cells"])
                best = None
                for tcomp in target_comps:
                    (tr, tc) = min(tcomp["cells"])
                    key = (abs(tr - r) + abs(tc - c), tr, tc)
                    if best is None or key < best[0]:
                        best = (key, tr, tc)
                _, tr, tc = best
                delta = ((tr - r) and (tr - r) // abs(tr - r),
                         (tc - c) and (tc - c) // abs(tc - c))
                if delta == (0, 0):
                    continue
                moves[comp["cells"]] = delta
            if not moves:
                raise ValueError("nothing to gravitate")
            return _apply_moves(grid, moves)
        if self.graph == "slide_until":
            direction = values["direction"]
            gap = values["gap"]
            height, width = _shape(grid)
            moves = {}
            for comp in _components(grid, background):
                if direction == "right":
                    delta = (width - 1 - gap) - comp["right"]
                    step = (0, delta)
                elif direction == "left":
                    delta = gap - comp["left"]
                    step = (0, delta)
                elif direction == "down":
                    delta = (height - 1 - gap) - comp["bottom"]
                    step = (delta, 0)
                else:
                    delta = gap - comp["top"]
                    step = (delta, 0)
                if delta:
                    moves[comp["cells"]] = step
            if not moves:
                raise ValueError("nothing to slide")
            return _apply_moves(grid, moves)
        if self.graph == "align_to_anchor":
            edge = values["edge"]
            anchor = values["anchor"]
            groups = _color_groups(grid, background)
            if not groups:
                raise ValueError("no components")
            if anchor in groups:
                target = _extent(groups[anchor], edge)
            else:
                target = values["position"]
            moves = {}
            for cells in groups.values():
                delta = target - _extent(cells, edge)
                if delta:
                    moves[cells] = (delta, 0) if edge in ("bottom", "top") else (0, delta)
            if not moves:
                raise ValueError("nothing to align")
            return _apply_moves(grid, moves)
        if self.graph == "slide_until":
            direction = values["direction"]
            gap = values["gap"]
            height, width = _shape(grid)
            moves = {}
            for comp in _components(grid, background):
                if direction == "right":
                    delta = (width - 1 - gap) - comp["right"]
                    step = (0, delta)
                elif direction == "left":
                    delta = gap - comp["left"]
                    step = (0, delta)
                elif direction == "down":
                    delta = (height - 1 - gap) - comp["bottom"]
                    step = (delta, 0)
                else:
                    delta = gap - comp["top"]
                    step = (delta, 0)
                if delta:
                    moves[comp["cells"]] = step
            if not moves:
                raise ValueError("nothing to slide")
            return _apply_moves(grid, moves)
        if self.graph == "align_to_anchor":
            edge = values["edge"]
            anchor = values["anchor"]
            components = list(_components(grid, background))
            anchors = [comp for comp in components if comp["color"] == anchor]
            if anchors:
                if edge == "bottom":
                    target = max(comp["bottom"] for comp in anchors)
                elif edge == "top":
                    target = min(comp["top"] for comp in anchors)
                elif edge == "right":
                    target = max(comp["right"] for comp in anchors)
                else:
                    target = min(comp["left"] for comp in anchors)
            else:
                target = values["position"]
            if edge == "bottom":
                moves = {
                    comp["cells"]: (target - comp["bottom"], 0)
                    for comp in components
                    if target - comp["bottom"]
                }
            elif edge == "top":
                moves = {
                    comp["cells"]: (target - comp["top"], 0)
                    for comp in components
                    if target - comp["top"]
                }
            elif edge == "right":
                moves = {
                    comp["cells"]: (0, target - comp["right"])
                    for comp in components
                    if target - comp["right"]
                }
            else:
                moves = {
                    comp["cells"]: (0, target - comp["left"])
                    for comp in components
                    if target - comp["left"]
                }
            if not moves:
                raise ValueError("nothing to align")
            return _apply_moves(grid, moves)
        if self.graph == "marker_rigid_slide_dock":
            marker = 5
            height, width = _shape(grid)
            keep = [
                c for c in range(width)
                if any(grid[r][c] != background for r in range(height))
            ]
            grid = tuple(tuple(row[c] for c in keep) for row in grid)
            height, width = _shape(grid)
            fives = {
                (r, c) for r in range(height) for c in range(width)
                if grid[r][c] == marker
            }
            if not fives:
                raise ValueError("marker absent")
            seen: set[tuple[int, int]] = set()
            comps: list[tuple[int, frozenset]] = []
            for r in range(height):
                for c in range(width):
                    v = grid[r][c]
                    if v in (background, marker) or (r, c) in seen:
                        continue
                    stack = [(r, c)]
                    cells = {(r, c)}
                    seen.add((r, c))
                    while stack:
                        x, y = stack.pop()
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < height and 0 <= ny < width
                                    and (nx, ny) not in seen
                                    and grid[nx][ny] == v):
                                seen.add((nx, ny))
                                cells.add((nx, ny))
                                stack.append((nx, ny))
                    comps.append((v, frozenset(cells)))
            occ_rows = sorted({
                r for r in range(height)
                if any(grid[r][c] != background for c in range(width))
            })
            if not occ_rows:
                raise ValueError("empty grid")
            top_occ, bot_occ = occ_rows[0], occ_rows[-1]
            units: list[tuple[int, frozenset, int]] = []
            for v, cells in comps:
                rows = sorted({r for r, _ in cells})
                r_top, r_bot = rows[0], rows[-1]
                top_cells = [(r_top, c) for r, c in cells if r == r_top]
                bot_cells = [(r_bot, c) for r, c in cells if r == r_bot]
                c_lmin = min(c for _, c in top_cells)
                c_rmax = max(c for _, c in top_cells)
                left = (
                    (r_top, c_lmin - 1)
                    if (r_top, c_lmin - 1) in fives else None
                )
                right = (
                    (r_top, c_rmax + 1)
                    if (r_top, c_rmax + 1) in fives else None
                )
                above = [
                    (r_top - 1, c) for _, c in sorted(top_cells)
                    if (r_top - 1, c) in fives
                ]
                below = [
                    (r_bot + 1, c) for _, c in sorted(bot_cells)
                    if (r_bot + 1, c) in fives
                ]
                if len(above) > 1 or len(below) > 1:
                    raise ValueError("ambiguous side markers")
                above = above[0] if above else None
                below = below[0] if below else None
                sides = frozenset(
                    name for name, item in (("L", left), ("R", right),
                                            ("A", above), ("B", below))
                    if item is not None
                )
                move: int | None
                if "L" not in sides:
                    move = None
                elif sides == frozenset({"L", "R", "B"}):
                    move = -1
                elif sides == frozenset({"L", "B"}):
                    move = None
                elif sides == frozenset({"L", "R"}):
                    move = 1
                elif sides == frozenset({"L", "A"}):
                    move = 1
                elif sides == frozenset({"L"}):
                    if r_top == top_occ:
                        move = 1
                    elif r_bot == bot_occ:
                        move = -1
                    else:
                        move = None
                else:
                    raise ValueError("unseen side set")
                if move is None:
                    continue
                unit = set(cells)
                unit.add(left)
                if below is not None:
                    unit.add(below)
                if above is not None and "A" in sides:
                    unit.add(above)
                if right is not None and sides == frozenset({"L", "R"}):
                    unit.add(right)
                units.append((v, frozenset(unit), move))
            if not units:
                raise ValueError("no moving unit")
            moving_cells: set[tuple[int, int]] = set()
            for _, unit, _ in units:
                moving_cells |= set(unit)
            anchors = sorted(fives - moving_cells)
            stat_cells: set[tuple[int, int]] = set()
            for _, cells in comps:
                if not set(cells) & moving_cells:
                    stat_cells |= set(cells)
            anchor_color: dict[tuple[int, int], int] = {}
            for pos in anchors:
                cols = set()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = pos[0] + dr, pos[1] + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and (nr, nc) in stat_cells):
                        cols.add(grid[nr][nc])
                if len(cols) != 1:
                    raise ValueError("anchor color ambiguous")
                anchor_color[pos] = next(iter(cols))
            out = [list(row) for row in grid]
            for r, c in moving_cells:
                out[r][c] = background
            placed: set[tuple[int, int]] = set()
            for pos, col in anchor_color.items():
                out[pos[0]][pos[1]] = col
            todo = sorted(units, key=lambda item: min(item[1]))
            progress = True
            while todo and progress:
                progress = False
                for v, unit, move in list(todo):
                    cells = sorted(unit)
                    k = 1
                    final = None
                    while True:
                        moved = [(r + k * move, c) for r, c in cells]
                        if not all(
                            0 <= r < height and 0 <= c < width
                            for r, c in moved
                        ):
                            break
                        if any(out[r][c] != background or (r, c) in placed
                               for r, c in moved):
                            break
                        mset = set(moved)
                        hit = any(
                            (r + dr, c + dc) in placed
                            or (r + dr, c + dc) in anchor_color
                            for r, c in moved
                            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                            if 0 <= r + dr < height and 0 <= c + dc < width
                            and (r + dr, c + dc) not in mset
                        )
                        if hit:
                            final = moved
                            break
                        k += 1
                    if final is not None:
                        for r, c in final:
                            out[r][c] = v
                            placed.add((r, c))
                        todo.remove((v, unit, move))
                        progress = True
            if todo:
                raise ValueError("slide does not converge")
            if any(value == marker for row in out for value in row):
                raise ValueError("marker remains")
            return tuple(tuple(row) for row in out)
        if self.graph == "shear_right":
            height, width = _shape(grid)
            out = [list(row) for row in grid]
            targets: dict[tuple[int, int], int] = {}
            for color, cells in _color_groups(grid, background).items():
                bottom = _extent(cells, "bottom")
                right = _extent(cells, "right")
                for r, c in cells:
                    if r == bottom or c == right:
                        continue
                    if c + 1 >= width:
                        raise ValueError("translation leaves the grid")
                    out[r][c] = background
                    targets[(r, c + 1)] = color
            if not targets:
                raise ValueError("nothing to shear")
            for moved, color in targets.items():
                out[moved[0]][moved[1]] = color
            return tuple(tuple(row) for row in out)
        raise ValueError("unknown a2 move graph")


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

    backgrounds = [_background(source) for source, _ in demos]
    colors = sorted(
        {
            value
            for source, _ in demos
            for row in source
            for value in row
        }
    )
    nonbackground = sorted(
        set(colors) - set(backgrounds)
    )

    # constant_color_move: serve una mappa colore->delta identica in ogni
    # demo, ricavata dall'inferenza per componente.
    mappings = []
    infer_ok = True
    for (source, expected), background in zip(demos, backgrounds):
        mapping = _infer_move(source, expected, background)
        if mapping is None:
            infer_ok = False
            break
        mappings.append(mapping)
    if infer_ok:
        color_deltas: list[dict[int, tuple[int, int]]] = []
        feasible = True
        for i, mapping in enumerate(mappings):
            source = demos[i][0]
            comp_colors = {
                comp["cells"]: comp["color"]
                for comp in _components(source, backgrounds[i])
            }
            cmap: dict[int, tuple[int, int]] = {}
            for cells, delta in mapping.items():
                if delta == (0, 0):
                    continue
                color = comp_colors[cells]
                if cmap.get(color, delta) != delta:
                    feasible = False
                    break
                cmap[color] = delta
            if not feasible:
                break
            color_deltas.append(cmap)
        if feasible and color_deltas and all(color_deltas):
            first = color_deltas[0]
            if all(cmap == first for cmap in color_deltas[1:]):
                add(Candidate(
                    "constant_color_move",
                    (("moves", tuple(sorted(first.items()))),),
                ))
    # move_all: enumerazione diretta di traslazioni piccole.
    for dr in range(-8, 9):
        for dc in range(-8, 9):
            if dr == 0 and dc == 0:
                continue
            add(Candidate("move_all", (("delta", (dr, dc)),)))

    # align_edge
    stationary_colors = set()
    if infer_ok:
        stationary_colors = set(nonbackground)
        for i, mapping in enumerate(mappings):
            source = demos[i][0]
            comp_colors = {
                comp["cells"]: comp["color"]
                for comp in _components(source, backgrounds[i])
            }
            for cells, delta in mapping.items():
                if delta != (0, 0):
                    stationary_colors.discard(comp_colors[cells])
    for edge in ("bottom", "top", "left", "right"):
        add(Candidate("align_edge", (("edge", edge),)))
        for anchor in sorted(stationary_colors):
            source0 = demos[0][0]
            groups = _color_groups(source0, _background(source0))
            if anchor not in groups:
                continue
            position = _extent(groups[anchor], edge)
            add(Candidate(
                "align_to_anchor",
                tuple(sorted({"anchor": anchor, "edge": edge,
                              "position": position}.items())),
            ))

    # shear_right
    add(Candidate("shear_right", ()))

    # marker_rigid_slide_dock
    add(Candidate("marker_rigid_slide_dock", ()))

    # slide_until
    for direction in ("right", "left", "down", "up"):
        for gap in range(4):
            add(Candidate(
                "slide_until",
                tuple(sorted({"direction": direction, "gap": gap}.items())),
            ))

    # gravitate_contact e gravitate_step per ogni coppia di colori.
    for mover in nonbackground:
        for target in nonbackground:
            if mover == target:
                continue
            bindings = tuple(sorted({"keep": False, "mover": mover,
                                     "target": target}.items()))
            add(Candidate("gravitate_contact", bindings))
            add(Candidate("gravitate_step", bindings))
            bindings_keep = tuple(sorted({"keep": True, "mover": mover,
                                          "target": target}.items()))
            add(Candidate("gravitate_contact", bindings_keep))

    # clone_toward_bar
    for bar in nonbackground:
        add(Candidate("clone_toward_bar", (("bar", bar),)))
        add(Candidate("connect_to_bar", (("bar", bar),)))

    return tuple(sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings))))
