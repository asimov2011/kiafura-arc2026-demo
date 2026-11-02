"""A2 classifier/canvas algebra: candidati derivati dai soli demo.

Grafi (proprieta' globale dell'input -> canvas):
  - symmetry_canvas: colore vero/falso del test di simmetria speculare;
  - largest_area_canvas: canvas riempita col colore dell'oggetto di area
    (o taglia) massima;
  - smallest_neighbor_canvas: colore dell'oggetto adiacente (bbox) al
    piu' piccolo.
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


def _components(
    grid: Grid, background: int, connectivity: int = 4,
    include_background: bool = False,
):
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if connectivity == 8:
        offsets = tuple(
            (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )
    height, width = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    items = []
    for r in range(height):
        for c in range(width):
            if (r, c) in seen:
                continue
            if grid[r][c] == background and not include_background:
                continue
            color = grid[r][c]
            stack = [(r, c)]
            cells = {(r, c)}
            seen.add((r, c))
            while stack:
                cr, cc = stack.pop()
                for dr, dc in offsets:
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
                    "size": len(cells),
                    "top": min(rows),
                    "bottom": max(rows),
                    "left": min(cols),
                    "right": max(cols),
                }
            )
    items.sort(key=lambda item: (item["top"], item["left"]))
    return items


def _canvas(shape: tuple[int, int], color: int) -> Grid:
    return tuple(tuple(color for _ in range(shape[1]))
                 for _ in range(shape[0]))


def _mirror(grid: Grid, mode: str) -> Grid:
    if mode == "v":
        return tuple(tuple(reversed(row)) for row in grid)
    if mode == "h":
        return tuple(reversed(grid))
    raise ValueError("unknown mirror mode")


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]
    _cost: int = 1

    def cost(self, grid: Grid) -> int:
        return self._cost + len(grid) * len(grid[0])

    def execute(self, grid: Grid) -> Grid:
        values = dict(self.bindings)
        shape = values["shape"]
        background = _background(grid)
        if self.graph == "symmetry_canvas":
            mode = values["mode"]
            if _mirror(grid, mode) == grid:
                return _canvas(shape, values["true_color"])
            return _canvas(shape, values["false_color"])
        if self.graph == "largest_area_canvas":
            metric = values["metric"]
            components = _components(grid, background)
            if not components:
                raise ValueError("no components")
            if metric == "area":
                best = max(
                    components,
                    key=lambda comp: (
                        (comp["bottom"] - comp["top"] + 1)
                        * (comp["right"] - comp["left"] + 1),
                        comp["size"],
                    ),
                )
            else:
                best = max(components, key=lambda comp: comp["size"])
            return _canvas(shape, best["color"])
        if self.graph == "smallest_neighbor_canvas":
            components = _components(
                grid, background, values["connectivity"],
                include_background=True)
            if len(components) < 2:
                raise ValueError("not enough components")
            smallest = min(
                components,
                key=lambda comp: (comp["size"], comp["top"], comp["left"]))
            best = None
            for comp in components:
                if comp is smallest or comp["color"] == background:
                    continue
                keyed = _aligned_distance(smallest, comp)
                if keyed is None:
                    continue
                if best is None or keyed < best[0]:
                    best = (keyed, comp)
            if best is None:
                raise ValueError("no aligned object")
            return _canvas(shape, best[1]["color"])
        if self.graph == "shape_class_canvas":
            background = values["background"]
            components = _components(grid, background, values["connectivity"])
            if not components:
                raise ValueError("no components")
            cells = max(components, key=lambda comp: comp["size"])["cells"]
            top = min(r for r, _ in cells)
            left = min(c for _, c in cells)
            norm = {(r - top, c - left) for r, c in cells}
            height = max(r for r, _ in norm) + 1
            width = max(c for _, c in norm) + 1
            size = len(norm)
            has_square = any(
                all((r + dr, c + dc) in norm for dr in (0, 1) for dc in (0, 1))
                for r in range(height - 1)
                for c in range(width - 1)
            )
            if size >= 4 and has_square:
                return _canvas(shape, values["l_color"])
            if (
                height == 3
                and width == 3
                and size == 5
                and (1, 1) in norm
                and (0, 0) not in norm
            ):
                return _canvas(shape, values["plus_color"])
            if (
                height == 3
                and width == 3
                and size == 5
                and (1, 1) in norm
                and all((r, c) in norm for r in (0, 2) for c in (0, 2))
            ):
                return _canvas(shape, values["x_color"])
            return _canvas(shape, values["default_color"])
        raise ValueError("unknown a2 canvas graph")


def _aligned_distance(a: dict, b: dict):
    """(distance, tiebreak) when the two objects share a row or column
    projection, else None."""
    keys = []
    if a["top"] <= b["bottom"] and b["top"] <= a["bottom"]:
        if b["left"] > a["right"]:
            gap = b["left"] - a["right"] - 1
        elif a["left"] > b["right"]:
            gap = a["left"] - b["right"] - 1
        else:
            gap = 0
        keys.append((gap, min(b["top"], b["left"])))
    if a["left"] <= b["right"] and b["left"] <= a["right"]:
        if b["top"] > a["bottom"]:
            gap = b["top"] - a["bottom"] - 1
        elif a["top"] > b["bottom"]:
            gap = a["top"] - b["bottom"] - 1
        else:
            gap = 0
        keys.append((gap, min(b["top"], b["left"])))
    if not keys:
        return None
    return min(keys)


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

    shapes = {
        (len(expected), len(expected[0])) for _, expected in demos
    }
    if len(shapes) != 1:
        return ()
    shape = next(iter(shapes))

    colors = sorted({
        value
        for _, expected in demos
        for row in expected
        for value in row
    })
    if any(
        len({value for row in expected for value in row}) != 1
        for _, expected in demos
    ):
        return ()

    backgrounds = sorted({
        value for source, _ in demos for row in source for value in row
    })

    # shape_class_canvas
    for bg in backgrounds:
     for connectivity in (4, 8):
        classes: dict[str, set[int]] = {
            "l": set(), "plus": set(), "x": set(), "default": set()}
        feasible = True
        for source, expected in demos:
            components = _components(source, bg, connectivity)
            if not components:
                feasible = False
                break
            cells = max(components, key=lambda comp: comp["size"])["cells"]
            top = min(r for r, _ in cells)
            left = min(c for _, c in cells)
            norm = {(r - top, c - left) for r, c in cells}
            height = max(r for r, _ in norm) + 1
            width = max(c for _, c in norm) + 1
            size = len(norm)
            has_square = any(
                all((r + dr, c + dc) in norm for dr in (0, 1) for dc in (0, 1))
                for r in range(height - 1)
                for c in range(width - 1)
            )
            if size >= 4 and has_square:
                kind = "l"
            elif (
                height == 3 and width == 3 and size == 5
                and (1, 1) in norm and (0, 0) not in norm
            ):
                kind = "plus"
            elif (
                height == 3 and width == 3 and size == 5
                and (1, 1) in norm
                and all((r, c) in norm for r in (0, 2) for c in (0, 2))
            ):
                kind = "x"
            else:
                kind = "default"
            classes[kind].add(expected[0][0])
        if not feasible:
            continue
        if any(len(values) > 1 for values in classes.values()):
            continue
        if not classes["l"] or not classes["default"]:
            continue
        fill = lambda name: next(iter(classes[name])) if classes[name] else next(iter(classes["default"]))
        add(Candidate(
            "shape_class_canvas",
            tuple(sorted({
                "background": bg,
                "connectivity": connectivity,
                "default_color": fill("default"),
                "l_color": fill("l"),
                "plus_color": fill("plus"),
                "shape": shape,
                "x_color": fill("x"),
            }.items())),
        ))

    # symmetry_canvas
    for mode in ("v", "h"):
        true_colors = set()
        false_colors = set()
        feasible = True
        for source, expected in demos:
            symmetric = _mirror(source, mode) == source
            color = expected[0][0]
            (true_colors if symmetric else false_colors).add(color)
        if len(true_colors) > 1 or len(false_colors) > 1:
            feasible = False
        if feasible and true_colors and false_colors:
            add(Candidate(
                "symmetry_canvas",
                tuple(sorted({"false_color": next(iter(false_colors)),
                              "mode": mode, "shape": shape,
                              "true_color": next(iter(true_colors))}.items())),
            ))

    # largest_area_canvas
    for metric in ("area", "size"):
        feasible = True
        for source, expected in demos:
            background = _background(source)
            components = _components(source, background)
            if not components:
                feasible = False
                break
            if metric == "area":
                best = max(
                    components,
                    key=lambda comp: (
                        (comp["bottom"] - comp["top"] + 1)
                        * (comp["right"] - comp["left"] + 1),
                        comp["size"],
                    ),
                )
            else:
                best = max(components, key=lambda comp: comp["size"])
            if best["color"] != expected[0][0]:
                feasible = False
                break
        if feasible:
            add(Candidate(
                "largest_area_canvas",
                tuple(sorted({"metric": metric,
                              "shape": shape}.items())),
            ))

    # smallest_neighbor_canvas
    for connectivity in (4, 8):
        feasible = True
        for source, expected in demos:
            background = _background(source)
            components = _components(
                source, background, connectivity, include_background=True)
            if len(components) < 2:
                feasible = False
                break
            smallest = min(
                components,
                key=lambda comp: (comp["size"], comp["top"], comp["left"]))
            best = None
            for comp in components:
                if comp is smallest or comp["color"] == background:
                    continue
                keyed = _aligned_distance(smallest, comp)
                if keyed is None:
                    continue
                if best is None or keyed < best[0]:
                    best = (keyed, comp)
            if best is None or best[1]["color"] != expected[0][0]:
                feasible = False
                break
        if feasible:
            add(Candidate(
                "smallest_neighbor_canvas",
                tuple(sorted({"connectivity": connectivity,
                              "shape": shape}.items())),
            ))

    return tuple(sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings))))
