"""A2 color-permutation algebra: candidati derivati dai soli demo.

Grafi:
  - color_map: funzione puntuale colore->colore, derivata dalla
    corrispondenza cella-per-cella sui demo (forme identiche);
  - switch_pair: scambio di due colori.

Nessun task ID, griglia letterale o programma in memoria: i binding sono
relazioni fra colori osservate nei demo, verificate fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]
EXEC_ERRORS = (ValueError, IndexError, KeyError, StopIteration)


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


@dataclass(frozen=True)
class Candidate:
    graph: str
    bindings: tuple[tuple[str, Any], ...]
    _cost: int = 1

    def cost(self, grid: Grid) -> int:
        return self._cost + len(grid) * len(grid[0])

    def execute(self, grid: Grid) -> Grid:
        values = dict(self.bindings)
        if self.graph == "color_map":
            mapping = dict(values["mapping"])
            return tuple(
                tuple(mapping.get(cell, cell) for cell in row)
                for row in grid
            )
        if self.graph == "switch_pair":
            a, b = values["a"], values["b"]
            mapping = {a: b, b: a}
            return tuple(
                tuple(mapping.get(cell, cell) for cell in row)
                for row in grid
            )
        if self.graph == "palette_extremes_replace":
            pivot = values["pivot"]
            replacement = values["replacement"]
            base = tuple(
                tuple(replacement if cell == pivot else cell for cell in row)
                for row in grid
            )
            colors = sorted({cell for row in base for cell in row})
            if len(colors) < 2:
                raise ValueError("palette too small")
            lo, hi = colors[0], colors[-1]
            mapping = {lo: hi, hi: lo}
            return tuple(
                tuple(mapping.get(cell, cell) for cell in row)
                for row in base
            )
        if self.graph == "least_color_swap":
            counts: dict[int, int] = {}
            for row in grid:
                for cell in row:
                    counts[cell] = counts.get(cell, 0) + 1
            least = min(counts, key=lambda value: (counts[value], value))
            replaced = tuple(
                tuple(0 if cell == least else cell for cell in row)
                for row in grid
            )
            counts2: dict[int, int] = {}
            for row in replaced:
                for cell in row:
                    counts2[cell] = counts2.get(cell, 0) + 1
            least2 = min(counts2, key=lambda value: (counts2[value], value))
            return tuple(
                tuple(least if cell == least2 else cell for cell in row)
                for row in replaced
            )
        if self.graph == "ring_shift_two":
            height, width = len(grid), len(grid[0])
            depth = min(height, width) // 2
            if depth < 3:
                raise ValueError("not enough rings")
            colors = []
            for d in range(depth):
                ring = set()
                for c in range(d, width - d):
                    ring.add(grid[d][c])
                    ring.add(grid[height - 1 - d][c])
                for r in range(d, height - d):
                    ring.add(grid[r][d])
                    ring.add(grid[r][width - 1 - d])
                if len(ring) != 1:
                    raise ValueError("ring not uniform")
                colors.append(next(iter(ring)))
            shifted = [
                colors[i + 2] if i + 2 < depth else colors[i - 1]
                for i in range(depth)
            ]
            out = [list(row) for row in grid]
            for d in range(depth):
                color = shifted[d]
                for c in range(d, width - d):
                    out[d][c] = color
                    out[height - 1 - d][c] = color
                for r in range(d, height - d):
                    out[r][d] = color
                    out[r][width - 1 - d] = color
            return tuple(tuple(row) for row in out)
        raise ValueError("unknown a2 color graph")


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

    if not all(
        _shape(source) == _shape(expected) for source, expected in demos
    ):
        return ()

    # color_map: la corrispondenza deve essere funzionale e identica in
    # ogni demo; ogni colore di input deve comparire (nessun implicito).
    mapping: dict[int, int] = {}
    seen: set[int] = set()
    feasible = True
    for source, expected in demos:
        for source_row, expected_row in zip(source, expected):
            for before, after in zip(source_row, expected_row):
                seen.add(before)
                if before in mapping and mapping[before] != after:
                    feasible = False
                    break
                mapping[before] = after
            if not feasible:
                break
        if not feasible:
            break
    if feasible:
        values_list = [mapping[color] for color in mapping]
        injective = len(set(values_list)) == len(values_list)
        nontrivial = injective and any(
            mapping.get(color, color) != color for color in mapping)
        input_colors = set(seen)
        mapped_colors = set(mapping)
        unmapped = sorted(input_colors - mapped_colors)
        if nontrivial and not unmapped:
            add(Candidate(
                "color_map",
                (("mapping", tuple(sorted(mapping.items()))),),
            ))
        elif nontrivial and len(unmapped) == 1:
            for fill in sorted(set(mapping.values()) | mapped_colors):
                full = dict(mapping)
                full[unmapped[0]] = fill
                add(Candidate(
                    "color_map",
                    (("mapping", tuple(sorted(full.items()))),),
                ))

    # switch_pair: enumerazione delle coppie di colori visibili.
    colors = sorted({
        value
        for grid in (g for source, expected in demos
                     for g in (source, expected))
        for row in grid
        for value in row
    })
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            add(Candidate(
                "switch_pair",
                tuple(sorted({"a": a, "b": b}.items())),
            ))

    add(Candidate("least_color_swap", ()))
    add(Candidate("ring_shift_two", ()))

    # palette_extremes_replace: pivot->replacement + scambio estremi.
    input_colors = sorted({
        value for source, _ in demos for row in source for value in row
    })
    output_colors = sorted({
        value for _, expected in demos for row in expected for value in row
    })
    added = sorted(set(output_colors) - set(input_colors))
    for pivot in input_colors:
        for replacement in added or [0]:
            add(Candidate(
                "palette_extremes_replace",
                tuple(sorted({"pivot": pivot,
                              "replacement": replacement}.items())),
            ))

    return tuple(sorted(result.values(), key=lambda item: (item.graph, repr(item.bindings))))
