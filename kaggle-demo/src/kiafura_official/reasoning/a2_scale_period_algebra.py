"""A2 scale/period algebra: candidati derivati dai soli demo.

Grafi:
  - upscale_by_palette: upscale per cella con fattore derivato dalla
    palette dell'input (numeri di colori - 1);
  - period_extend_horizontal: ogni riga non uniforme e' estesa col suo
    periodo minimo a larghezza multipla;
  - pattern_fill_region: le celle del colore target sono riempite col
    pattern periodico (vp, hp) osservato nel resto della griglia;
  - axis_factor_upscale: upscale con fattori per riga/colonna derivati
    dai demo;
  - period_hole_fill: il pattern 2D visibile e' periodico con periodo
    minimo (vp, hp) derivato dall'input; le celle "buco" sono riempite
    col valore della cella congruente del periodo.
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


def _grid_periods(grid, rows, cols, target):
    """Minimal (vp, hp) such that grid[r][c] == grid[r0+((r-r0)%vp)]
    [c0+((c-c0)%hp)] over the visible (non-target) cells."""
    r0, c0 = min(rows), min(cols)
    height = max(rows) - r0 + 1
    width = max(cols) - c0 + 1
    for vp in range(1, height + 1):
        ok_v = True
        for r in range(r0 + vp, r0 + height):
            for c in range(c0, c0 + width):
                if grid[r][c] == target:
                    continue
                if grid[r][c] != grid[r0 + (r - r0) % vp][c]:
                    ok_v = False
                    break
            if not ok_v:
                break
        if not ok_v:
            continue
        for hp in range(1, width + 1):
            ok = True
            for r in range(r0, r0 + height):
                for c in range(c0, c0 + width):
                    if grid[r][c] == target:
                        continue
                    if grid[r][c] != grid[r0 + (r - r0) % vp][c0 + (c - c0) % hp]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return vp, hp
    return None


def _min_period(row: Sequence[int]) -> int:
    width = len(row)
    for p in range(1, width + 1):
        if all(row[i] == row[i - p] for i in range(p, width)):
            return p
    return width


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]
    _cost: int = 1

    def cost(self, grid: Grid) -> int:
        return self._cost + len(grid) * len(grid[0])

    def execute(self, grid: Grid) -> Grid:
        values = dict(self.bindings)
        if self.graph == "upscale_by_palette":
            palette = {cell for row in grid for cell in row}
            factor = len(palette) - 1
            if factor < 1:
                raise ValueError("factor too small")
            return tuple(
                tuple(cell for cell in row for _ in range(factor))
                for row in grid
                for _ in range(factor)
            )
        if self.graph == "period_extend_horizontal":
            factor = values["factor"]
            background = _background(grid)
            out = []
            for row in grid:
                if all(cell == background for cell in row):
                    out.append(tuple(row) * factor)
                    continue
                period = _min_period(row)
                extended = [row[c % period] for c in range(len(row) * factor)]
                out.append(tuple(extended))
            return tuple(out)
        if self.graph == "axis_factor_upscale":
            row_factors = values["row_factors"]
            col_factors = values["col_factors"]
            if len(row_factors) != len(grid) or len(col_factors) != len(grid[0]):
                raise ValueError("factor arity mismatch")
            out = []
            for r, row in enumerate(grid):
                expanded = tuple(
                    cell
                    for c, cell in enumerate(row)
                    for _ in range(col_factors[c])
                )
                for _ in range(row_factors[r]):
                    out.append(expanded)
            return tuple(out)
        if self.graph == "period_hole_fill":
            hole = values["hole"]
            height, width = len(grid), len(grid[0])
            known = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if grid[r][c] != hole
            ]
            if not known:
                raise ValueError("no known cells")
            vp = None
            for p in range(1, height + 1):
                ok = True
                for r, c in known:
                    for r2, c2 in known:
                        if c == c2 and (r - r2) % p == 0 \
                                and grid[r][c] != grid[r2][c2]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    vp = p
                    break
            if vp is None:
                raise ValueError("no vertical period")
            hp = None
            for p in range(1, width + 1):
                ok = True
                for r, c in known:
                    for r2, c2 in known:
                        if r == r2 and (c - c2) % p == 0 \
                                and grid[r][c] != grid[r2][c2]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    hp = p
                    break
            if hp is None:
                raise ValueError("no horizontal period")
            out = [list(row) for row in grid]
            for r in range(height):
                for c in range(width):
                    if grid[r][c] != hole:
                        continue
                    values_here = {
                        grid[r2][c2]
                        for r2, c2 in known
                        if (r - r2) % vp == 0 and (c - c2) % hp == 0
                    }
                    if len(values_here) != 1:
                        raise ValueError("ambiguous hole fill")
                    out[r][c] = values_here.pop()
            return tuple(tuple(row) for row in out)
        if self.graph == "self_similar_block_stamp":
            background = _background(grid)
            height, width = len(grid), len(grid[0])
            cells = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if grid[r][c] != background
            ]
            if not cells:
                raise ValueError("empty grid")
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells)
            left = min(c for _, c in cells)
            right = max(c for _, c in cells)
            block_h = bottom - top + 1
            block_w = right - left + 1
            if height % block_h or width % block_w:
                raise ValueError("bbox does not divide grid")
            row_factor = height // block_h
            col_factor = width // block_w
            if row_factor != col_factor or row_factor < 2:
                raise ValueError("bad factor")
            if top % block_h or left % block_w:
                raise ValueError("unaligned bbox")
            pattern = {(r - top, c - left) for r, c in cells}
            block_row = top // block_h
            block_col = left // block_w
            out = [list(row) for row in grid]
            painted = 0
            for pat_r, pat_c in sorted(pattern):
                shift_r = (pat_r - block_row) * block_h
                shift_c = (pat_c - block_col) * block_w
                if shift_r == 0 and shift_c == 0:
                    continue
                for r, c in cells:
                    nr, nc = r + shift_r, c + shift_c
                    if not (0 <= nr < height and 0 <= nc < width):
                        raise ValueError("copy off grid")
                    if out[nr][nc] != background \
                            and out[nr][nc] != grid[r][c]:
                        raise ValueError("copy collision")
                    if out[nr][nc] == background:
                        out[nr][nc] = grid[r][c]
                        painted += 1
            if not painted:
                raise ValueError("nothing stamped")
            return tuple(tuple(row) for row in out)
        if self.graph == "period_frame_phase_shift":
            height, width = len(grid), len(grid[0])
            first_frame_row = None
            for r in range(height - 1, -1, -1):
                if len(set(grid[r])) == 1:
                    first_frame_row = r
                else:
                    break
            first_frame_col = None
            for c in range(width - 1, -1, -1):
                if len({grid[r][c] for r in range(height)}) == 1:
                    first_frame_col = c
                else:
                    break
            if first_frame_row is None or first_frame_col is None:
                raise ValueError("no frame")
            frame = grid[first_frame_row][0]
            if any(grid[r][0] != frame
                   for r in range(first_frame_row, height)):
                raise ValueError("row frame mixed")
            if any(grid[0][c] != frame
                   for c in range(first_frame_col, width)):
                raise ValueError("col frame mixed")
            pat_rows = first_frame_row
            pat_cols = first_frame_col
            if pat_rows < 1 or pat_cols < 1:
                raise ValueError("no pattern region")
            pattern = [grid[r][:pat_cols] for r in range(pat_rows)]
            if any(v == frame for row in pattern for v in row):
                raise ValueError("frame inside pattern")
            row_period = None
            for p in range(1, pat_rows + 1):
                if all(pattern[r] == pattern[r - p]
                       for r in range(p, pat_rows)):
                    row_period = p
                    break
            col_period = None
            for p in range(1, pat_cols + 1):
                if all(row[c] == row[c - p] for row in pattern
                       for c in range(p, pat_cols)):
                    col_period = p
                    break
            if row_period is None or col_period is None:
                raise ValueError("no period")
            out = tuple(
                tuple(
                    pattern[r % row_period][(c + 1) % col_period]
                    for c in range(width)
                )
                for r in range(height)
            )
            if out == grid:
                raise ValueError("no change")
            return out
        raise ValueError("unknown a2 scale graph")


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

    add(Candidate("upscale_by_palette", ()))
    add(Candidate("self_similar_block_stamp", ()))
    add(Candidate("period_frame_phase_shift", ()))

    def compositions(total: int, parts: int, cap: int = 4):
        if parts == 1:
            if 1 <= total <= cap:
                yield (total,)
            return
        for first in range(1, min(cap, total - parts + 1) + 1):
            for rest in compositions(total - first, parts - 1, cap):
                yield (first,) + rest

    def upscale_factors(source, rf, cf):
        return tuple(
            tuple(
                cell
                for c, cell in enumerate(row)
                for _ in range(cf[c])
            )
            for r, row in enumerate(source)
            for _ in range(rf[r])
        )

    axis_pairs = None
    for source, expected in demos:
        good = set()
        for rf in compositions(len(expected), len(source)):
            for cf in compositions(len(expected[0]), len(source[0])):
                if upscale_factors(source, rf, cf) == expected:
                    good.add((rf, cf))
        axis_pairs = good if axis_pairs is None else axis_pairs & good
    if axis_pairs:
        for rf, cf in sorted(axis_pairs):
            add(Candidate(
                "axis_factor_upscale",
                (("col_factors", cf), ("row_factors", rf)),
            ))

    factors = set()
    for source, expected in demos:
        if (
            len(expected) == len(source)
            and len(expected[0]) % len(source[0]) == 0
        ):
            factors.add(len(expected[0]) // len(source[0]))
    for factor in sorted(factors):
        if 2 <= factor <= 3:
            add(Candidate("period_extend_horizontal",
                            (("factor", factor),)))

    colors = sorted({
        value for source, _ in demos for row in source for value in row
    })

    changed_colors = None
    for source, expected in demos:
        if len(source) != len(expected) or len(source[0]) != len(expected[0]):
            changed_colors = set()
            break
        here = {
            source[r][c]
            for r in range(len(source))
            for c in range(len(source[0]))
            if source[r][c] != expected[r][c]
        }
        changed_colors = here if changed_colors is None else changed_colors & here
    for hole in sorted(changed_colors or ()):
        add(Candidate("period_hole_fill", (("hole", hole),)))

    return tuple(sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings))))
