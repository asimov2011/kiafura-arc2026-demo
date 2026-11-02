"""A2 connections algebra: candidati derivati dai soli demo.

Grafi:
  - marker_pair_connect: coppie di singleton marker allineate per riga o
    colonna connesse col colore di collegamento;
  - cross_center_ring: intorno all'incrocio di una riga e una colonna
    piene, l'intorno 3x1 e' dipinto col colore dell'anello;
  - enclosed_fill: le regioni di background chiuse sono riempite.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]
EXEC_ERRORS = (ValueError, IndexError, KeyError, StopIteration)


def _background(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    return min(counts, key=lambda value: (-counts[value], value))


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
        if self.graph == "cross_center_ring":
            ring = values["ring"]
            centers = []
            for r in range(height):
                for c in range(width):
                    x = grid[r][c]
                    if x == background:
                        continue
                    for axis in ("v", "h"):
                        if axis == "v":
                            full = all(grid[rr][c] == x for rr in range(height))
                            line = grid[r]
                        else:
                            full = all(grid[r][cc] == x for cc in range(width))
                            line = tuple(grid[rr][c] for rr in range(height))
                        if not full:
                            continue
                        counts = Counter(line)
                        if len(counts) != 2:
                            continue
                        (y, n_y), (_, n_x) = counts.most_common(2)
                        if n_x != 1 or y == background or y == x:
                            continue
                        centers.append((r, c))
            if len(centers) != 1:
                raise ValueError("no unique cross")
            cr, cc = centers[0]
            out = [list(row) for row in grid]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    r, c = cr + dr, cc + dc
                    if 0 <= r < height and 0 <= c < width:
                        out[r][c] = ring
            return tuple(tuple(row) for row in out)
        raise ValueError("unknown a2 connections graph")


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
    for ring in colors:
        add(Candidate("cross_center_ring", (("ring", ring),)))

    return tuple(sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings))))
