"""A2 half/panel algebra: candidati derivati dai soli demo.

Grafi:
  - half_overlay: una meta' (righe) dell'input viene dipinta sull'altra;
  - half_intersect: output = canvas delle posizioni in cui entrambe le
    meta' hanno il colore 'vuoto';
  - shift_intersect: celle di un colore traslate in due versi, le
    intersezioni dipinte con un nuovo colore;
  - key_mask_upscale: una riga e una colonna uniformi (separatore)
    dividono l'input in quadranti; il quadrante chiave (piccolo) e'
    scalato al quadrante maschera e la maschera ne prende i colori.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]
EXEC_ERRORS = (ValueError, IndexError, KeyError, StopIteration)


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _half(grid: Grid, part: str) -> Grid:
    height = len(grid)
    half = height // 2
    if part == "top":
        return grid[:half]
    return grid[height - half:]


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]
    _cost: int = 1

    def cost(self, grid: Grid) -> int:
        return self._cost + len(grid) * len(grid[0])

    def execute(self, grid: Grid) -> Grid:
        values = dict(self.bindings)
        height, width = _shape(grid)
        if self.graph == "half_overlay":
            source_part = values["source"]
            blank = values["blank"]
            source = _half(grid, source_part)
            out = [list(row) for row in grid]
            for r, row in enumerate(source):
                target_row = (
                    height - len(source) + r
                    if source_part == "top"
                    else r
                )
                for c, value in enumerate(row):
                    if value != blank:
                        out[target_row][c] = value
            return tuple(tuple(row) for row in out)
        if self.graph == "half_intersect":
            blank = values["blank"]
            mark = values["mark"]
            background = values["background"]
            top = _half(grid, "top")
            bottom = _half(grid, "bottom")
            if _shape(top) != _shape(bottom):
                raise ValueError("odd split")
            out = []
            for top_row, bottom_row in zip(top, bottom):
                out.append(tuple(
                    mark if t == blank and b == blank else background
                    for t, b in zip(top_row, bottom_row)
                ))
            return tuple(out)
        if self.graph == "shift_intersect":
            color = values["color"]
            mark = values["mark"]
            cells = {
                (r, c)
                for r, row in enumerate(grid)
                for c, value in enumerate(row)
                if value == color
            }
            if not cells:
                raise ValueError("color absent")
            dr, dc = values["delta"]
            forward = {(r + dr, c + dc) for r, c in cells}
            backward = {(r - dr, c - dc) for r, c in cells}
            hits = forward & backward
            if not hits:
                raise ValueError("no intersection")
            out = [list(row) for row in grid]
            for r, c in hits:
                if not (0 <= r < height and 0 <= c < width):
                    continue
                out[r][c] = mark
            return tuple(tuple(row) for row in out)
        if self.graph == "separator_overlay":
            sep = values["separator"]
            blank = values["blank"]
            source_part = values["source"]
            sep_rows = [
                r for r, row in enumerate(grid)
                if all(value == sep for value in row)
            ]
            if len(sep_rows) != 1:
                raise ValueError("separator row not unique")
            srow = sep_rows[0]
            top = grid[:srow]
            bottom = grid[srow + 1:]
            base = list(top) if source_part == "bottom" else list(bottom)
            overlay = bottom if source_part == "bottom" else top
            out = [list(row) for row in base]
            for r, row in enumerate(overlay):
                if r >= len(out):
                    break
                for c, value in enumerate(row):
                    if value != blank and value != sep:
                        out[r][c] = value
            return tuple(tuple(row) for row in out)
        if self.graph == "key_mask_upscale":
            sep = values["separator"]
            mask = values["mask"]
            background = values["background"]
            sep_rows = [
                r for r, row in enumerate(grid)
                if all(value == sep for value in row)
            ]
            sep_cols = [
                c for c in range(width)
                if all(grid[r][c] == sep for r in range(height))
            ]
            if len(sep_rows) != 1 or len(sep_cols) != 1:
                raise ValueError("separators not unique")
            srow, scol = sep_rows[0], sep_cols[0]
            row_slices = [slice(0, srow), slice(srow + 1, height)]
            col_slices = [slice(0, scol), slice(scol + 1, width)]
            quads = [
                (rs, cs, (rs.stop - rs.start) * (cs.stop - cs.start))
                for rs in row_slices
                for cs in col_slices
                if rs.stop - rs.start > 0 and cs.stop - cs.start > 0
            ]
            quads.sort(key=lambda item: item[2])
            key_rs, key_cs = quads[0][0], quads[0][1]
            mask_rs, mask_cs = quads[-1][0], quads[-1][1]
            if quads[0][2] != 4:
                raise ValueError("key quadrant not 2x2")
            key = tuple(
                tuple(
                    grid[r][c] for c in range(key_cs.start, key_cs.stop)
                )
                for r in range(key_rs.start, key_rs.stop)
            )
            mask_height = mask_rs.stop - mask_rs.start
            mask_width = mask_cs.stop - mask_cs.start
            if (
                mask_height % len(key) != 0
                or mask_width % len(key[0]) != 0
            ):
                raise ValueError("mask not a multiple of key")
            scale_r = mask_height // len(key)
            scale_c = mask_width // len(key[0])
            cells = [
                (r - mask_rs.start, c - mask_cs.start)
                for r in range(mask_rs.start, mask_rs.stop)
                for c in range(mask_cs.start, mask_cs.stop)
                if grid[r][c] == mask
            ]
            if not cells:
                raise ValueError("mask absent")
            out = [
                [background] * mask_width for _ in range(mask_height)
            ]
            for r, c in cells:
                out[r][c] = key[r // scale_r][c // scale_c]
            return tuple(tuple(row) for row in out)
        raise ValueError("unknown a2 half graph")


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

    for source_part in ("top", "bottom"):
        for blank in sorted({
            value for source, _ in demos for row in source for value in row
        }):
            add(Candidate(
                "half_overlay",
                tuple(sorted({"blank": blank,
                              "source": source_part}.items())),
            ))

    colors = sorted({
        value
        for source, expected in demos
        for grid in (source, expected)
        for row in grid
        for value in row
    })
    for blank in colors:
        for mark in colors:
            for background in colors:
                add(Candidate(
                    "half_intersect",
                    tuple(sorted({"background": background, "blank": blank,
                                  "mark": mark}.items())),
                ))

    for sep in colors:
        for blank in colors:
            for source_part in ("top", "bottom"):
                add(Candidate(
                    "separator_overlay",
                    tuple(sorted({"blank": blank, "separator": sep,
                                  "source": source_part}.items())),
                ))

    for color in colors:
        for mark in colors:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    if dr == 0 and dc == 0:
                        continue
                    add(Candidate(
                        "shift_intersect",
                        tuple(sorted({"color": color, "delta": (dr, dc),
                                      "mark": mark}.items())),
                    ))

    for sep in colors:
        for mask in colors:
            if mask == sep:
                continue
            for background in colors:
                add(Candidate(
                    "key_mask_upscale",
                    tuple(sorted({"background": background, "mask": mask,
                                  "separator": sep}.items())),
                ))

    return tuple(sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings))))
