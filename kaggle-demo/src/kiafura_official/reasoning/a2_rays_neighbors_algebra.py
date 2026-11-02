"""A2 rays/neighbors algebra: candidati derivati dai soli demo.

Grafi:
  - diagonal_neighbor_paint: per ogni cella del colore marker, le quattro
    celle diagonali sono dipinte con colori fissi per direzione e la cella
    marker e' rimossa;
  - neighbor_ring_paint: intorno a ogni cella marker, vicini ortogonali e
    diagonali dipinti con due colori distinti, marker rimosso;
  - column_matching_recolor: singleton ricolorano la colonna della
    componente (barra) che interseca la loro colonna;
  - seed_comb_bars: da un seed singolo, barre verticali ogni 2 colonne
    con connettori alternati alto/basso dal colore accento.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]
EXEC_ERRORS = (ValueError, IndexError, KeyError, StopIteration)

DIAGONALS = {
    "up_left": (-1, -1),
    "up_right": (-1, 1),
    "down_left": (1, -1),
    "down_right": (1, 1),
}
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


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
            rows = {p for p, _ in cells}
            cols = {p for _, p in cells}
            items.append(
                {
                    "color": color,
                    "cells": frozenset(cells),
                    "top": min(rows),
                    "bottom": max(rows),
                    "left": min(cols),
                    "right": max(cols),
                }
            )
    items.sort(key=lambda item: (item["top"], item["left"]))
    return items


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
        marker = values.get("marker")
        points = []
        if marker is not None:
            points = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if grid[r][c] == marker
            ]
            if not points:
                raise ValueError("marker absent")
        if self.graph == "diagonal_neighbor_paint":
            colors = dict(values["colors"])
            out = [list(row) for row in grid]
            for r, c in points:
                out[r][c] = background
                for name, (dr, dc) in DIAGONALS.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        out[nr][nc] = colors[name]
            return tuple(tuple(row) for row in out)
        if self.graph == "neighbor_ring_paint":
            orth = values["orth_color"]
            diag = values["diag_color"]
            out = [list(row) for row in grid]
            for r, c in points:
                out[r][c] = background
                for dr, dc in ORTHOGONAL:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        out[nr][nc] = orth
                for dr, dc in DIAGONALS.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        out[nr][nc] = diag
            return tuple(tuple(row) for row in out)
        if self.graph == "column_matching_recolor":
            components = _components4(grid, background)
            singletons = [
                comp for comp in components if len(comp["cells"]) == 1
            ]
            bars = [
                comp for comp in components if len(comp["cells"]) > 1
            ]
            if not singletons or not bars:
                raise ValueError("missing singletons or bars")
            out = [list(row) for row in grid]
            painted = 0
            for sing in singletons:
                (r, c) = next(iter(sing["cells"]))
                for comp in bars:
                    if not (comp["left"] <= c <= comp["right"]):
                        continue
                    if comp["top"] <= r <= comp["bottom"]:
                        continue
                    for rr, cc in comp["cells"]:
                        out[rr][cc] = sing["color"]
                        painted += 1
            if not painted:
                raise ValueError("no matching column")
            return tuple(tuple(row) for row in out)
        if self.graph == "corner_diagonal_trail":
            marker = values["marker"]
            found = None
            for r in range(height - 1):
                for c in range(width - 1):
                    cells = [
                        (r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)
                    ]
                    colors = {grid[rr][cc] for rr, cc in cells}
                    if (
                        len(colors) != 2
                        or marker not in colors
                        or background in colors
                    ):
                        continue
                    if found is not None:
                        raise ValueError("multiple marker blocks")
                    found = cells
            if found is None:
                raise ValueError("marker block absent")
            paint = (
                {grid[rr][cc] for rr, cc in found} - {marker}
            ).pop()
            r0, c0 = found[0]
            directions = set()
            for rr, cc in found:
                if grid[rr][cc] == marker:
                    directions.add(
                        (2 * (rr - r0) - 1, 2 * (cc - c0) - 1))
            out = [list(row) for row in grid]
            for rr, cc in found:
                for dr, dc in directions:
                    k = 0
                    while True:
                        nr, nc = rr + k * dr, cc + k * dc
                        if not (0 <= nr < height and 0 <= nc < width):
                            break
                        out[nr][nc] = paint
                        k += 1
            return tuple(tuple(row) for row in out)
        if self.graph == "edge_bar_antidiagonal":
            diag = values["diag"]
            fill = values["fill"]
            bar = grid[0][0]
            if bar == background:
                raise ValueError("no bar")
            if all(grid[r][0] == bar for r in range(height)) and all(
                grid[r][c] == background
                for r in range(height)
                for c in range(1, width)
            ):
                out = [list(row) for row in grid]
                for r in range(min(height, width)):
                    out[r][width - 1 - r] = diag
                for c in range(1, width):
                    out[height - 1][c] = fill
                for r in range(height):
                    out[r][0] = bar
                return tuple(tuple(row) for row in out)
            if all(grid[0][c] == bar for c in range(width)) and all(
                grid[r][c] == background
                for r in range(1, height)
                for c in range(width)
            ):
                out = [list(row) for row in grid]
                for c in range(min(height, width)):
                    out[height - 1 - c][c] = diag
                for r in range(1, height):
                    out[r][width - 1] = fill
                for c in range(width):
                    out[0][c] = bar
                return tuple(tuple(row) for row in out)
            raise ValueError("no edge bar")
        if self.graph == "row_diagonal_sweep":
            if height != 1:
                raise ValueError("need a single row")
            row = grid[0]
            items = [
                (c, value) for c, value in enumerate(row)
                if value != background
            ]
            if not items:
                raise ValueError("no values")
            size = len(items) * width
            out = [[background] * size for _ in range(size)]
            for r in range(size):
                start = size - 1 - r
                for c, value in items:
                    col = start + c
                    if 0 <= col < size:
                        out[r][col] = value
            return tuple(tuple(item) for item in out)
        if self.graph == "stair_arms":
            anchor = values["anchor"]
            paint = values["paint"]
            anchors = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if grid[r][c] == anchor
            ]
            if len(anchors) != 1:
                raise ValueError("need one anchor")
            r0, c0 = anchors[0]
            arm_up = [(0, 0), (0, 1), (0, 2), (1, 0), (-1, 2)]
            arm_dn = [(0, 0), (0, -1), (0, -2), (1, -2), (-1, 0)]
            out = [list(row) for row in grid]
            painted = 0
            for k in range(max(height, width)):
                r, c = r0 - 2 - 2 * k, c0 + 2 * k
                if r < -2:
                    break
                for dr, dc in arm_up:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and (rr, cc) != (r0, c0) and out[rr][cc] != paint:
                        out[rr][cc] = paint
                        painted += 1
            for k in range(max(height, width)):
                r, c = r0 + 2 + 2 * k, c0 - 2 * k
                if r > height + 1:
                    break
                for dr, dc in arm_dn:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and (rr, cc) != (r0, c0) and out[rr][cc] != paint:
                        out[rr][cc] = paint
                        painted += 1
            if not painted:
                raise ValueError("nothing painted")
            return tuple(tuple(row) for row in out)
        if self.graph == "seed_comb_bars":
            accent = values["accent"]
            seeds = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if grid[r][c] != background
            ]
            if len(seeds) != 1:
                raise ValueError("seed not unique")
            r0, c0 = seeds[0]
            seed_color = grid[r0][c0]
            out = [[background] * width for _ in range(height)]
            k = 0
            while c0 + 2 * k < width:
                for r in range(r0 + 1):
                    out[r][c0 + 2 * k] = seed_color
                k += 1
            k = 0
            while c0 + 2 * k + 1 < width:
                row = 0 if k % 2 == 0 else r0
                out[row][c0 + 2 * k + 1] = accent
                k += 1
            return tuple(tuple(row) for row in out)
        raise ValueError("unknown a2 rays graph")


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

    markers = sorted({
        value
        for source, _ in demos
        for row in source
        for value in row
    })

    # diagonal_neighbor_paint: colori per direzione derivati dai demo.
    for marker in markers:
        colors: dict[str, int] = {}
        feasible = True
        for source, expected in demos:
            if len(source) != len(expected) or len(source[0]) != len(expected[0]):
                feasible = False
                break
            height, width = len(source), len(source[0])
            points = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if source[r][c] == marker
            ]
            if not points:
                feasible = False
                break
            for r, c in points:
                for name, (dr, dc) in DIAGONALS.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        got = expected[nr][nc]
                        if colors.get(name, got) != got:
                            feasible = False
                            break
                        colors[name] = got
                if not feasible:
                    break
            if not feasible:
                break
        if feasible and colors:
            missing = [name for name in DIAGONALS if name not in colors]
            palette = sorted({
                value for _, expected in demos
                for row in expected for value in row
            })
            unused = [
                color for color in palette
                if color not in colors.values()
            ]
            if not missing and len(colors) == 4:
                add(Candidate(
                    "diagonal_neighbor_paint",
                    tuple(sorted({"colors": tuple(sorted(colors.items())),
                                  "marker": marker}.items())),
                ))
            elif len(missing) == 1 and len(unused) == 1:
                colors = dict(colors)
                colors[missing[0]] = unused[0]
                add(Candidate(
                    "diagonal_neighbor_paint",
                    tuple(sorted({"colors": tuple(sorted(colors.items())),
                                  "marker": marker}.items())),
                ))

    # neighbor_ring_paint
    for marker in markers:
        orth, diag = None, None
        feasible = True
        for source, expected in demos:
            if len(source) != len(expected) or len(source[0]) != len(expected[0]):
                feasible = False
                break
            height, width = len(source), len(source[0])
            points = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if source[r][c] == marker
            ]
            if not points:
                feasible = False
                break
            for r, c in points:
                for dr, dc in ORTHOGONAL:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        got = expected[nr][nc]
                        if orth is None:
                            orth = got
                        elif orth != got:
                            feasible = False
                for dr, dc in DIAGONALS.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        got = expected[nr][nc]
                        if diag is None:
                            diag = got
                        elif diag != got:
                            feasible = False
                if not feasible:
                    break
            if not feasible:
                break
        if feasible and orth is not None and diag is not None:
            add(Candidate(
                "neighbor_ring_paint",
                tuple(sorted({"diag_color": diag, "marker": marker,
                              "orth_color": orth}.items())),
            ))

    # column_matching_recolor
    add(Candidate("column_matching_recolor", (("marker", 0),)))
    add(Candidate("row_diagonal_sweep", ()))

    stair_colors = sorted({
        value
        for source, expected in demos
        for grid in (source, expected)
        for row in grid
        for value in row
    })

    for anchor in stair_colors:
        for paint in stair_colors:
            if anchor != paint:
                add(Candidate(
                    "stair_arms",
                    tuple(sorted({"anchor": anchor, "paint": paint}.items())),
                ))

    for marker in markers:
        add(Candidate("corner_diagonal_trail", (("marker", marker),)))

    all_colors = sorted({
        value
        for source, expected in demos
        for grid in (source, expected)
        for row in grid
        for value in row
    })
    for diag in all_colors:
        for fill in all_colors:
            if diag != fill:
                add(Candidate(
                    "edge_bar_antidiagonal",
                    tuple(sorted({"diag": diag, "fill": fill}.items())),
                ))

    for accent in all_colors:
        add(Candidate("seed_comb_bars", (("accent", accent),)))

    return tuple(sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings))))
