"""A300 residual operational family, derived only from A300 source experience.

The graphs are de-bound transformation schemas.  Candidate bindings are
enumerated from finite domains or witnessed by the demonstrations, and every
candidate is retained only when it reconstructs every demonstration exactly.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]
Cells = frozenset[tuple[int, int]]
EXEC_ERRORS = (IndexError, KeyError, StopIteration, TypeError, ValueError)
N4 = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _background(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda value: (-counts[value], value))


def _components(
    grid: Grid,
    *,
    include_background: bool = False,
    diagonal: bool = False,
) -> list[tuple[int, Cells]]:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    unseen = {
        (r, c)
        for r in range(height)
        for c in range(width)
        if include_background or grid[r][c] != background
    }
    result: list[tuple[int, Cells]] = []
    while unseen:
        seed = min(unseen)
        color = grid[seed[0]][seed[1]]
        queue = deque([seed])
        cells = {seed}
        unseen.remove(seed)
        while queue:
            r, c = queue.popleft()
            neighbors = N4 if not diagonal else tuple(
                (dr, dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
            )
            for dr, dc in neighbors:
                nxt = (r + dr, c + dc)
                if nxt in unseen and grid[nxt[0]][nxt[1]] == color:
                    unseen.remove(nxt)
                    cells.add(nxt)
                    queue.append(nxt)
        result.append((color, frozenset(cells)))
    return sorted(result, key=lambda item: (min(item[1]), item[0]))


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    cells = tuple(cells)
    return (
        min(r for r, _ in cells),
        min(c for _, c in cells),
        max(r for r, _ in cells),
        max(c for _, c in cells),
    )


def _normalize(cells: Iterable[tuple[int, int]]) -> Cells:
    cells = tuple(cells)
    r0, c0, _, _ = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _translate_all(grid: Grid, delta: tuple[int, int]) -> Grid:
    background = _background(grid)
    height, width = len(grid), len(grid[0])
    out = [[background] * width for _ in range(height)]
    dr, dc = delta
    seen = False
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == background:
                continue
            seen = True
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                raise ValueError("translation clips object")
            out[nr][nc] = value
    if not seen:
        raise ValueError("no object")
    return tuple(tuple(row) for row in out)


def _palette_upscale(grid: Grid) -> Grid:
    factor = len({cell for row in grid for cell in row}) - 1
    if factor < 1:
        raise ValueError("invalid palette scale")
    return tuple(
        tuple(cell for cell in row for _ in range(factor))
        for row in grid
        for _ in range(factor)
    )


def _periodic_hole_fill(grid: Grid, hole: int) -> Grid:
    height, width = len(grid), len(grid[0])
    known = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] != hole
    ]
    if not known:
        raise ValueError("no visible period")

    def period(axis: int, limit: int) -> int:
        for candidate in range(1, limit + 1):
            valid = True
            for left in known:
                for right in known:
                    other = 1 - axis
                    if left[other] != right[other]:
                        continue
                    if (left[axis] - right[axis]) % candidate == 0 \
                            and grid[left[0]][left[1]] != grid[right[0]][right[1]]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return candidate
        raise ValueError("period absent")

    vertical, horizontal = period(0, height), period(1, width)
    out = [list(row) for row in grid]
    for r in range(height):
        for c in range(width):
            if grid[r][c] != hole:
                continue
            values = {
                grid[rr][cc]
                for rr, cc in known
                if (r - rr) % vertical == 0 and (c - cc) % horizontal == 0
            }
            if len(values) != 1:
                raise ValueError("ambiguous periodic hole")
            out[r][c] = values.pop()
    return tuple(tuple(row) for row in out)


def _axis_bridge_to_markers(grid: Grid) -> Grid:
    components = _components(grid)
    if len(components) < 2:
        raise ValueError("insufficient components")
    marker_color = min(
        {value for row in grid for value in row},
        key=lambda value: (sum(cell == value for row in grid for cell in row), value),
    )
    largest = max(components, key=lambda item: (len(item[1]), -min(item[1])[0], -min(item[1])[1]))[1]
    markers = {
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value == marker_color
    }
    if not markers or markers & largest:
        raise ValueError("marker/object ambiguity")
    background = _background(grid)
    out = [list(row) for row in grid]
    wrote = False
    for mr, mc in markers:
        for r, c in largest:
            if mr == r:
                for cc in range(min(mc, c), max(mc, c) + 1):
                    if out[r][cc] == background:
                        out[r][cc] = marker_color
                        wrote = True
            elif mc == c:
                for rr in range(min(mr, r), max(mr, r) + 1):
                    if out[rr][c] == background:
                        out[rr][c] = marker_color
                        wrote = True
    if not wrote:
        raise ValueError("no aligned marker")
    return tuple(tuple(row) for row in out)


def _orient_reflected_template(grid: Grid, fill: int) -> Grid:
    background = _background(grid)
    components = _components(grid)
    if len(components) < 2:
        raise ValueError("template or pointer absent")
    _, template = max(components, key=lambda item: len(item[1]))
    _, pointer = min(components, key=lambda item: len(item[1]))
    if len(template) == len(pointer):
        raise ValueError("template ambiguity")
    r0, c0, r1, c1 = _bbox(template)
    height, width = r1 - r0 + 1, c1 - c0 + 1
    pr = sum(r for r, _ in pointer) / len(pointer)
    pc = sum(c for _, c in pointer) / len(pointer)
    tr = (r0 + r1) / 2
    tc = (c0 + c1) / 2
    dr, dc = pr - tr, pc - tc
    vertical = abs(dr) > abs(dc)
    direction = (1 if dr > 0 else -1, 0) if vertical else (0, 1 if dc > 0 else -1)
    template_grid = [[fill] * width for _ in range(height)]
    for r, c in template:
        template_grid[r - r0][c - c0] = grid[r][c]
    if vertical:
        template_grid = list(reversed(template_grid))
        origin = (r1 + 1, c0) if direction[0] > 0 else (r0 - height, c0)
    else:
        template_grid = [list(reversed(row)) for row in template_grid]
        origin = (r0, c1 + 1) if direction[1] > 0 else (r0, c0 - width)
    out = [[fill if value == background else value for value in row] for row in grid]
    for rr, row in enumerate(template_grid):
        for cc, value in enumerate(row):
            if value == fill:
                continue
            r, c = origin[0] + rr, origin[1] + cc
            if 0 <= r < len(out) and 0 <= c < len(out[0]):
                out[r][c] = value
    return tuple(tuple(row) for row in out)


def _relocate_shape_matches(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    holes = [
        cells
        for color, cells in _components(grid, include_background=True)
        if color == 0 and not any(
            r in (0, height - 1) or c in (0, width - 1) for r, c in cells
        )
    ]
    if len(holes) != 2:
        raise ValueError("expected two enclosed holes")
    objects = [(color, cells) for color, cells in _components(grid) if color != 0]
    matches: list[tuple[Cells, int, Cells]] = []
    for hole in sorted(holes, key=min):
        shape = _normalize(hole)
        candidates = [(color, cells) for color, cells in objects if _normalize(cells) == shape]
        if not candidates:
            raise ValueError("shape match not unique")
        hr, hc, _, _ = _bbox(hole)
        ranked = sorted(
            candidates,
            key=lambda item: (
                abs(_bbox(item[1])[0] - hr) + abs(_bbox(item[1])[1] - hc),
                min(item[1]),
            ),
        )
        if len(ranked) > 1:
            first_distance = abs(_bbox(ranked[0][1])[0] - hr) + abs(_bbox(ranked[0][1])[1] - hc)
            next_distance = abs(_bbox(ranked[1][1])[0] - hr) + abs(_bbox(ranked[1][1])[1] - hc)
            if first_distance == next_distance:
                raise ValueError("shape match not unique")
        color, cells = ranked[0]
        matches.append((hole, color, cells))
    out = [list(row) for row in grid]
    for _, _, cells in matches:
        for r, c in cells:
            out[r][c] = 0
    for hole, color, _ in matches:
        for r, c in hole:
            out[r][c] = color
    return tuple(tuple(row) for row in out)


def _repeat_anchor_shape(grid: Grid, spacing: int) -> Grid:
    components = _components(grid, diagonal=True)
    if len(components) < 2:
        raise ValueError("anchor or satellite absent")
    _, anchor = max(components, key=lambda item: len(item[1]))
    ar0, ac0, ar1, ac1 = _bbox(anchor)
    out = [list(row) for row in grid]
    wrote = False
    for color, satellite in components:
        if satellite == anchor:
            continue
        sr0, sc0, sr1, sc1 = _bbox(satellite)
        dr = -1 if sr1 < ar0 else 1 if sr0 > ar1 else 0
        dc = -1 if sc1 < ac0 else 1 if sc0 > ac1 else 0
        for multiplier in range(1, max(len(grid), len(grid[0])) // spacing + 2):
            any_inside = False
            for r, c in anchor:
                rr, cc = r + multiplier * spacing * dr, c + multiplier * spacing * dc
                if 0 <= rr < len(grid) and 0 <= cc < len(grid[0]):
                    out[rr][cc] = color
                    wrote = any_inside = True
            if not any_inside:
                break
    if not wrote:
        raise ValueError("no repeated anchor")
    return tuple(tuple(row) for row in out)


def _largest_empty_square(grid: Grid, fill: int) -> Grid:
    background = _background(grid)
    height, width = len(grid), len(grid[0])
    choices: list[tuple[int, int, int]] = []
    for size in range(2, min(9, height, width) + 1):
        for r0 in range(height - size + 1):
            for c0 in range(width - size + 1):
                if all(
                    grid[r][c] == background
                    for r in range(r0, r0 + size)
                    for c in range(c0, c0 + size)
                ):
                    choices.append((size, r0, c0))
    if not choices:
        raise ValueError("no empty square")
    size, r0, c0 = max(choices, key=lambda item: (item[0], -item[1], -item[2]))
    out = [list(row) for row in grid]
    for r in range(r0, r0 + size):
        for c in range(c0, c0 + size):
            out[r][c] = fill
    return tuple(tuple(row) for row in out)


def _largest_empty_rectangle(grid: Grid, fill: int) -> Grid:
    background = 0
    height, width = len(grid), len(grid[0])
    choices: list[tuple[int, int, int, int]] = []
    for rect_h in range(2, min(9, height) + 1):
        for rect_w in range(2, min(9, width) + 1):
            for r0 in range(height - rect_h + 1):
                for c0 in range(width - rect_w + 1):
                    if all(
                        grid[r][c] == background
                        for r in range(r0, r0 + rect_h)
                        for c in range(c0, c0 + rect_w)
                    ):
                        choices.append((rect_h * rect_w, r0, c0, rect_h, rect_w))
    if not choices:
        raise ValueError("no empty rectangle")
    _, r0, c0, rect_h, rect_w = max(
        choices, key=lambda item: (item[0], item[4], -item[1], -item[2])
    )
    out = [list(row) for row in grid]
    for r in range(r0, r0 + rect_h):
        for c in range(c0, c0 + rect_w):
            out[r][c] = fill
    return tuple(tuple(row) for row in out)


def _corner_oriented_block_pair(grid: Grid, color: int) -> Grid:
    if len(grid) != len(grid[0]) or len(grid) < 2:
        raise ValueError("square key required")
    size = len(grid)
    block = size + 1
    out_size = size * 3
    base = [[0] * out_size for _ in range(out_size)]
    for offset in (0, block):
        for r in range(offset, offset + block):
            for c in range(offset, offset + block):
                base[r][c] = color
    occupied = {
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value != _background(grid)
    }
    corners = ((0, size - 1), (size - 1, size - 1), (size - 1, 0))
    turns = 1 if corners[0] in occupied else 2 if corners[1] in occupied \
        else 3 if corners[2] in occupied else 0
    result = tuple(tuple(row) for row in base)
    for _ in range(turns):
        result = tuple(tuple(result[out_size - 1 - c][r] for c in range(out_size)) for r in range(out_size))
    return result


def _repair_occluded_reflection(grid: Grid) -> Grid:
    counts = Counter(cell for row in grid for cell in row)
    background = _background(grid)
    nonbackground = sorted(
        (value for value in counts if value != background),
        key=lambda value: (counts[value], value),
    )
    if len(nonbackground) != 2:
        raise ValueError("expected pattern and occluder")
    occluder, pattern = nonbackground
    out = [
        [background if value == occluder else value for value in row]
        for row in grid
    ]
    cells = {
        (r, c)
        for r, row in enumerate(out)
        for c, value in enumerate(row)
        if value == pattern
    }
    if not cells:
        raise ValueError("pattern absent")
    _, c0, _, c1 = _bbox(cells)
    reflected = {(r, c0 + c1 - c) for r, c in cells}
    shifts = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            shifted = {(r + dr, c + dc) for r, c in reflected}
            overlap = len(shifted & cells)
            shifts.append((overlap, -abs(dr) - abs(dc), -dr, -dc, shifted))
    shifted = max(shifts, key=lambda item: item[:4])[4]
    for r, c in shifted:
        if 0 <= r < len(out) and 0 <= c < len(out[0]):
            out[r][c] = pattern
    return tuple(tuple(row) for row in out)


def _complete_box_and_sparse_frontiers(grid: Grid, obj: int, fill: int) -> Grid:
    cells = {
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value == obj
    }
    if not cells:
        raise ValueError("object absent")
    r0, c0, r1, c1 = _bbox(cells)
    box = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    interior = cells - box
    rows = {r for r, _ in interior}
    cols = {c for _, c in interior}
    horizontal = {
        (r, c) for r in rows for c in range(c0, c1 + 1)
    }
    vertical = {
        (r, c) for c in cols for r in range(r0, r1 + 1)
    }
    frontier = horizontal if len(horizontal) < len(vertical) else vertical
    background = _background(grid)
    out = [list(row) for row in grid]
    for r, c in box | frontier:
        if out[r][c] == background:
            out[r][c] = fill
    return tuple(tuple(row) for row in out)


def _expand_cells_toward_frame(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    out = [[0] * (width + 2) for _ in range(height + 2)]
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            rr, cc = r + 1, c + 1
            out[rr][cc] = value
            candidates = []
            if r < height - 1 - r:
                candidates.append((rr - 1, cc))
            elif r > height - 1 - r:
                candidates.append((rr + 1, cc))
            if c < width - 1 - c:
                candidates.append((rr, cc - 1))
            elif c > width - 1 - c:
                candidates.append((rr, cc + 1))
            for er, ec in candidates:
                out[er][ec] = value
    return tuple(tuple(row) for row in out)


def _gravitate_with_trailing_bar(
    grid: Grid,
    obj: int,
    bar: int,
    trail: int,
) -> Grid:
    object_cells = {
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value == obj
    }
    bar_cells = {
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value == bar
    }
    if not object_cells or not bar_cells:
        raise ValueError("object or bar absent")
    or0, oc0, or1, oc1 = _bbox(object_cells)
    br0, bc0, br1, bc1 = _bbox(bar_cells)
    if br0 > or1:
        delta = (br0 - or1 - 1, 0)
    elif br1 < or0:
        delta = (br1 - or0 + 1, 0)
    elif bc0 > oc1:
        delta = (0, bc0 - oc1 - 1)
    elif bc1 < oc0:
        delta = (0, bc1 - oc0 + 1)
    else:
        raise ValueError("bar does not define a direction")
    background = _background(grid)
    out = [list(row) for row in grid]
    for r, c in object_cells:
        out[r][c] = background
    dr, dc = delta
    moved = {(r + dr, c + dc) for r, c in object_cells}
    for r, c in moved:
        if not (0 <= r < len(out) and 0 <= c < len(out[0])):
            raise ValueError("move clips")
        out[r][c] = obj
    mr0, mc0, mr1, mc1 = _bbox(moved)
    if dr > 0:
        trail_cells = {(mr0 - 1, c) for c in range(len(out[0]))}
    elif dr < 0:
        trail_cells = {(mr1 + 1, c) for c in range(len(out[0]))}
    elif dc > 0:
        trail_cells = {(r, mc0 - 1) for r in range(len(out))}
    else:
        trail_cells = {(r, mc1 + 1) for r in range(len(out))}
    for r, c in trail_cells:
        if 0 <= r < len(out) and 0 <= c < len(out[0]) and out[r][c] == background:
            out[r][c] = trail
    return tuple(tuple(row) for row in out)


def _render_object_count(grid: Grid, color: int) -> Grid:
    count = len(_components(grid))
    if not 0 <= count <= 9:
        raise ValueError("count outside 3x3 canvas")
    return tuple(
        tuple(color if r * 3 + c in range(0, count * 2, 2) else 0 for c in range(3))
        for r in range(3)
    )


def _nested_outbox_completion(grid: Grid, replacement: int) -> Grid:
    foreground = min(
        {cell for row in grid for cell in row},
        key=lambda value: (sum(cell == value for row in grid for cell in row), value),
    )
    components = [cells for color, cells in _components(grid) if color == foreground]
    if not components:
        raise ValueError("foreground absent")
    seed = max(components, key=len)
    all_components = _components(grid, include_background=True)
    min_width_component = min(
        all_components,
        key=lambda item: (_bbox(item[1])[3] - _bbox(item[1])[1] + 1, min(item[1])),
    )[1]
    stride = 2 if len(min_width_component) == 1 else 3
    height, width = len(grid), len(grid[0])
    out = [list(row) for row in grid]

    def outbox(cells: Cells) -> Cells:
        r0, c0, r1, c1 = _bbox(cells)
        return frozenset(
            (r, c)
            for r in range(r0 - 1, r1 + 2)
            for c in range(c0 - 1, c1 + 2)
            if r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1)
        )

    current = seed
    for _ in range(3):
        for _ in range(stride):
            current = outbox(current)
        for r, c in current:
            if 0 <= r < height and 0 <= c < width:
                out[r][c] = foreground
    background = _background(grid)
    return tuple(
        tuple(replacement if value == background else value for value in row)
        for row in out
    )


def _repeat_mask_toward_marker(grid: Grid) -> Grid:
    background = _background(grid)
    colors = sorted(
        {cell for row in grid for cell in row} - {background},
        key=lambda value: (sum(cell == value for row in grid for cell in row), value),
    )
    if len(colors) < 2 or 0 not in {cell for row in grid for cell in row}:
        raise ValueError("mask or marker absent")
    marker = colors[0]
    mask = {
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value == 0
    }
    marker_cells = {
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value == marker
    }
    if not mask or not marker_cells:
        raise ValueError("mask or marker absent")
    r0, c0, r1, c1 = _bbox(mask)
    mr0, mc0, mr1, mc1 = _bbox(marker_cells)
    mask_center = ((r0 + r1) / 2, (c0 + c1) / 2)
    marker_center = ((mr0 + mr1) / 2, (mc0 + mc1) / 2)
    dr = (marker_center[0] > mask_center[0]) - (marker_center[0] < mask_center[0])
    dc = (marker_center[1] > mask_center[1]) - (marker_center[1] < mask_center[1])
    shape = (r1 - r0 + 1, c1 - c0 + 1)
    diagonal = {
        (r0 + offset, c0 + offset)
        for offset in range(min(shape))
    }
    full_diagonal = diagonal == (diagonal & mask)
    stride = shape if full_diagonal else (
        shape[0] - (1 if dr else 0),
        shape[1] - (1 if dc else 0),
    )
    out = [list(row) for row in grid]
    for multiplier in range(1, 5):
        for r, c in mask:
            rr = r + multiplier * stride[0] * dr
            cc = c + multiplier * stride[1] * dc
            if 0 <= rr < len(out) and 0 <= cc < len(out[0]):
                out[rr][cc] = marker
    return tuple(tuple(row) for row in out)


def _diagonal_scaffold_from_occluders(grid: Grid) -> Grid:
    palette = {cell for row in grid for cell in row}
    if 0 not in palette or 5 not in palette or len(palette - {0, 5}) != 1:
        raise ValueError("expected background, occluder and marker")
    marker = next(iter(palette - {0, 5}))
    starts: set[tuple[int, int]] = set()
    for _, cells in _components(grid):
        if not cells or grid[min(cells)[0]][min(cells)[1]] != 5:
            continue
        r0, c0, r1, c1 = _bbox(cells)
        upper_right = (r0, c1)
        if grid[upper_right[0]][upper_right[1]] == 5:
            starts.add((upper_right[0] - 1, upper_right[1] + 1))
        else:
            starts.add((r1 + 1, c0 - 1))
    if not starts:
        raise ValueError("no scaffold starts")
    out = [
        [0 if value == 5 else value for value in row]
        for row in grid
    ]
    for sr, sc in starts:
        for dr, dc in ((1, 1), (-1, -1)):
            r, c = sr, sc
            while 0 <= r < len(out) and 0 <= c < len(out[0]):
                out[r][c] = marker
                r += dr
                c += dc
    return tuple(tuple(row) for row in out)


def _repeat_cross_sequences(grid: Grid) -> Grid:
    background = _background(grid)
    height, width = len(grid), len(grid[0])
    row_counts = [sum(value != background for value in row) for row in grid]
    col_counts = [
        sum(grid[r][c] != background for r in range(height))
        for c in range(width)
    ]
    row = max(range(height), key=lambda r: (row_counts[r], -r))
    col = max(range(width), key=lambda c: (col_counts[c], -c))
    row_cells = [(c, grid[row][c]) for c in range(width) if grid[row][c] != background]
    col_cells = [(r, grid[r][col]) for r in range(height) if grid[r][col] != background]
    if len(row_cells) < 2 or len(col_cells) < 2:
        raise ValueError("cross sequences absent")
    c0, c1 = row_cells[0][0], row_cells[-1][0]
    r0, r1 = col_cells[0][0], col_cells[-1][0]
    hseq = tuple(grid[row][c] for c in range(c0, c1 + 1))
    vseq = tuple(grid[r][col] for r in range(r0, r1 + 1))
    if background in hseq or background in vseq:
        raise ValueError("sequence has gaps")
    out = [[background] * width for _ in range(height)]
    for c in range(width):
        out[row][c] = hseq[(c - c0) % len(hseq)]
    for r in range(height):
        out[r][col] = vseq[(r - r0) % len(vseq)]
    return tuple(tuple(values) for values in out)


def _crop_enclosed_panel_and_connect(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    holes = [
        cells
        for color, cells in _components(grid, include_background=True)
        if color == 0 and not any(
            r in (0, height - 1) or c in (0, width - 1) for r, c in cells
        )
    ]
    if len(holes) != 1:
        raise ValueError("enclosed panel not unique")
    r0, c0, r1, c1 = _bbox(holes[0])
    r0, c0, r1, c1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
    if r0 < 0 or c0 < 0 or r1 >= height or c1 >= width:
        raise ValueError("panel outbox clips")
    panel = tuple(tuple(grid[r][c0:c1 + 1]) for r in range(r0, r1 + 1))
    counts = Counter(cell for row in panel for cell in row if cell != 0)
    if not counts:
        raise ValueError("panel foreground absent")
    dominant = min(counts, key=lambda value: (-counts[value], value))
    cells = {(r, c) for r, row in enumerate(panel) for c, value in enumerate(row) if value == dominant}
    vertical: set[tuple[int, int]] = set()
    horizontal: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr, cc in cells:
            if c == cc:
                vertical.update((x, c) for x in range(min(r, rr), max(r, rr) + 1))
            if r == rr:
                horizontal.update((r, x) for x in range(min(c, cc), max(c, cc) + 1))
    chosen = vertical if len(vertical) > len(horizontal) else horizontal
    out = [list(row) for row in panel]
    for r, c in chosen:
        out[r][c] = dominant
    return tuple(tuple(row) for row in out)


def _multicolor_components(grid: Grid) -> list[Cells]:
    background = _background(grid)
    unseen = {
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value != background
    }
    result = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        queue = deque([seed])
        cells = {seed}
        while queue:
            r, c = queue.popleft()
            for dr, dc in N4:
                nxt = (r + dr, c + dc)
                if nxt in unseen:
                    unseen.remove(nxt)
                    cells.add(nxt)
                    queue.append(nxt)
        result.append(frozenset(cells))
    return sorted(result, key=min)


def _stamp_panel_templates(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    if height > width and height % 2 == 0:
        panels = (
            tuple(grid[: height // 2]),
            tuple(grid[height // 2 :]),
        )
    elif width % 2 == 0:
        panels = (
            tuple(tuple(row[: width // 2]) for row in grid),
            tuple(tuple(row[width // 2 :]) for row in grid),
        )
    else:
        raise ValueError("input cannot be split evenly")
    panels = tuple(sorted(panels, key=lambda panel: len({cell for row in panel for cell in row})))
    target, source = panels
    objects = _multicolor_components(source)
    if not objects:
        raise ValueError("template panel empty")
    merged_colors = Counter(source[r][c] for cells in objects for r, c in cells)
    structural = min(merged_colors, key=lambda value: (-merged_colors[value], value))
    out = [list(row) for row in target]
    stamped = False
    for cells in objects:
        marker_cells = {(r, c) for r, c in cells if source[r][c] != structural}
        if not marker_cells:
            continue
        occurrences = []
        mr0, mc0, _, _ = _bbox(marker_cells)
        normalized = [(r - mr0, c - mc0, source[r][c]) for r, c in marker_cells]
        for r0 in range(len(target)):
            for c0 in range(len(target[0])):
                if all(
                    r0 + dr < len(target)
                    and c0 + dc < len(target[0])
                    and target[r0 + dr][c0 + dc] == value
                    for dr, dc, value in normalized
                ):
                    occurrences.append((r0, c0))
        if not occurrences:
            continue
        target_origin = min(occurrences)
        dr, dc = target_origin[0] - mr0, target_origin[1] - mc0
        shifted = {(r + dr, c + dc, source[r][c]) for r, c in cells}
        columns = {c for _, c, _ in shifted}
        if len(columns) <= 2:
            continue
        for r, c, value in shifted:
            if 0 <= r < len(out) and 0 <= c < len(out[0]):
                out[r][c] = value
        stamped = True
    if not stamped:
        raise ValueError("no template occurrence")
    return tuple(tuple(row) for row in out)


def _decorate_marker_rectangle(grid: Grid, connector: int) -> Grid:
    background = _background(grid)
    markers = [
        (r, c, value)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value != background
    ]
    colors = sorted({value for _, _, value in markers})
    if len(markers) != 4 or len(colors) != 2:
        raise ValueError("four two-color markers required")
    out = [list(row) for row in grid]
    for r, c, value in markers:
        other = colors[1] if value == colors[0] else colors[0]
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < len(out) and 0 <= cc < len(out[0]) and (rr, cc) != (r, c):
                    out[rr][cc] = other
    marker_cells = {(r, c) for r, c, _ in markers}
    r0, c0, r1, c1 = _bbox(marker_cells)
    outline = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    } - marker_cells
    for r, c in outline:
        distance = min(abs(r - mr) + abs(c - mc) for mr, mc in marker_cells)
        if distance % 2 == 0 and out[r][c] == background:
            out[r][c] = connector
    return tuple(tuple(row) for row in out)


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]
    _cost: int = 1

    def cost(self, grid: Grid) -> int:
        return self._cost + len(grid) * len(grid[0])

    def execute(self, grid: Grid) -> Grid:
        values = dict(self.bindings)
        if self.graph == "translate_all_objects":
            return _translate_all(grid, values["delta"])
        if self.graph == "upscale_by_palette_cardinality":
            return _palette_upscale(grid)
        if self.graph == "periodic_hole_completion":
            return _periodic_hole_fill(grid, values["hole"])
        if self.graph == "axis_bridge_to_markers":
            return _axis_bridge_to_markers(grid)
        if self.graph == "orient_reflected_template":
            return _orient_reflected_template(grid, values["fill"])
        if self.graph == "relocate_shape_matches":
            return _relocate_shape_matches(grid)
        if self.graph == "repeat_anchor_shape":
            return _repeat_anchor_shape(grid, values["spacing"])
        if self.graph == "largest_empty_square":
            return _largest_empty_square(grid, values["fill"])
        if self.graph == "largest_empty_rectangle":
            return _largest_empty_rectangle(grid, values["fill"])
        if self.graph == "corner_oriented_block_pair":
            return _corner_oriented_block_pair(grid, values["color"])
        if self.graph == "repair_occluded_reflection":
            return _repair_occluded_reflection(grid)
        if self.graph == "complete_box_and_sparse_frontiers":
            return _complete_box_and_sparse_frontiers(
                grid, values["object"], values["fill"]
            )
        if self.graph == "expand_cells_toward_frame":
            return _expand_cells_toward_frame(grid)
        if self.graph == "gravitate_with_trailing_bar":
            return _gravitate_with_trailing_bar(
                grid, values["object"], values["bar"], values["trail"]
            )
        if self.graph == "render_object_count":
            return _render_object_count(grid, values["color"])
        if self.graph == "nested_outbox_completion":
            return _nested_outbox_completion(grid, values["replacement"])
        if self.graph == "repeat_mask_toward_marker":
            return _repeat_mask_toward_marker(grid)
        if self.graph == "diagonal_scaffold_from_occluders":
            return _diagonal_scaffold_from_occluders(grid)
        if self.graph == "repeat_cross_sequences":
            return _repeat_cross_sequences(grid)
        if self.graph == "crop_enclosed_panel_and_connect":
            return _crop_enclosed_panel_and_connect(grid)
        if self.graph == "stamp_panel_templates":
            return _stamp_panel_templates(grid)
        if self.graph == "decorate_marker_rectangle":
            return _decorate_marker_rectangle(grid, values["connector"])
        raise ValueError("unknown A300 residual graph")


def candidates(demos: Demos) -> tuple[Candidate, ...]:
    retained: dict[tuple[str, tuple], Candidate] = {}

    def add(candidate: Candidate) -> None:
        key = (candidate.graph, candidate.bindings)
        if key in retained:
            return
        for source, expected in demos:
            try:
                if candidate.execute(source) != expected:
                    return
            except EXEC_ERRORS:
                return
        retained[key] = candidate

    for dr in range(-8, 9):
        for dc in range(-8, 9):
            if (dr, dc) != (0, 0):
                add(Candidate("translate_all_objects", (("delta", (dr, dc)),)))
    add(Candidate("upscale_by_palette_cardinality", ()))
    palette = sorted({cell for source, _ in demos for row in source for cell in row})
    for hole in palette:
        add(Candidate("periodic_hole_completion", (("hole", hole),)))
    add(Candidate("axis_bridge_to_markers", ()))
    output_colors = sorted({cell for _, expected in demos for row in expected for cell in row})
    for fill in output_colors:
        add(Candidate("orient_reflected_template", (("fill", fill),)))
    add(Candidate("relocate_shape_matches", ()))
    for spacing in range(1, 9):
        add(Candidate("repeat_anchor_shape", (("spacing", spacing),)))
    for fill in output_colors:
        add(Candidate("largest_empty_square", (("fill", fill),)))
        add(Candidate("largest_empty_rectangle", (("fill", fill),)))
    common_nonbackground = set.intersection(*[
        {cell for row in source for cell in row} - {_background(source)}
        for source, _ in demos
    ]) if demos else set()
    for color in sorted(common_nonbackground):
        add(Candidate("corner_oriented_block_pair", (("color", color),)))
    add(Candidate("repair_occluded_reflection", ()))
    source_colors = sorted({cell for source, _ in demos for row in source for cell in row})
    for obj in source_colors:
        for fill in output_colors:
            if obj != fill:
                add(Candidate(
                    "complete_box_and_sparse_frontiers",
                    (("fill", fill), ("object", obj)),
                ))
    add(Candidate("expand_cells_toward_frame", ()))
    for obj in source_colors:
        for bar in source_colors:
            if obj == bar:
                continue
            for trail in output_colors:
                if trail not in (obj, bar):
                    add(Candidate(
                        "gravitate_with_trailing_bar",
                        (("bar", bar), ("object", obj), ("trail", trail)),
                    ))
    for color in output_colors:
        add(Candidate("render_object_count", (("color", color),)))
        add(Candidate("nested_outbox_completion", (("replacement", color),)))
    add(Candidate("repeat_mask_toward_marker", ()))
    add(Candidate("diagonal_scaffold_from_occluders", ()))
    add(Candidate("repeat_cross_sequences", ()))
    add(Candidate("crop_enclosed_panel_and_connect", ()))
    add(Candidate("stamp_panel_templates", ()))
    for connector in output_colors:
        add(Candidate("decorate_marker_rectangle", (("connector", connector),)))
    return tuple(sorted(retained.values(), key=lambda item: (item.graph, repr(item.bindings))))
