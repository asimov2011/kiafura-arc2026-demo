"""Sound inverse witnesses for the frozen panel-composition operators."""

from __future__ import annotations

from collections import Counter
from itertools import permutations, product
from typing import Sequence

from kiafura_official.reasoning.panel_composition_algebra import Candidate, Grid


def _palette(demos: Sequence[tuple[Grid, Grid]]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                value
                for pair in demos
                for grid in pair
                for row in grid
                for value in row
            }
        )
    )


def _backgrounds(demos: Sequence[tuple[Grid, Grid]]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                Counter(value for row in source for value in row).most_common(1)[0][0]
                for source, _ in demos
            }
        )
    )


def _has_full_separator(grid: Grid, role: int) -> bool:
    return any(all(value == role for value in row) for row in grid) or any(
        all(grid[row][column] == role for row in range(len(grid)))
        for column in range(len(grid[0]))
    )


def _separator_column(grid: Grid, role: int) -> int | None:
    return next(
        (
            column
            for column in range(len(grid[0]))
            if all(grid[row][column] == role for row in range(len(grid)))
        ),
        None,
    )


def _relocation_role_pairs(
    demos: Sequence[tuple[Grid, Grid]],
    separator: int,
) -> tuple[tuple[int, int], ...]:
    """Return every visible non-structural replacement in the moved band."""

    pairs = set()
    for source, expected in demos:
        column = _separator_column(source, separator)
        if column is None:
            return ()
        destination = 2 * column + 1
        if destination >= len(source[0]):
            return ()
        for row in range(len(source)):
            for index in range(column + 1, destination):
                before, after = source[row][index], expected[row][index]
                if before != after:
                    pairs.add((before, after))
    return tuple(sorted(pairs))


def _preserve_values(
    demos: Sequence[tuple[Grid, Grid]],
    separator: int,
    source_role: int,
    target_role: int,
) -> tuple[int, ...]:
    """Propagate the exact branch condition without choosing a hidden member."""

    allowed = set(range(4))
    for source, expected in demos:
        column = _separator_column(source, separator)
        if column is None:
            return ()
        destination = 2 * column + 1
        if destination >= len(source[0]):
            return ()
        count = sum(
            source[row][index] == target_role
            for row in range(len(source))
            for index in range(column)
        )
        role_cells = [
            (row, index)
            for row in range(len(source))
            for index in range(column + 1, destination)
            if source[row][index] == source_role
        ]
        changed = [
            coordinate
            for coordinate in role_cells
            if expected[coordinate[0]][coordinate[1]] == target_role
            and expected[coordinate[0]][coordinate[1]] != source_role
        ]
        unchanged = [
            coordinate
            for coordinate in role_cells
            if expected[coordinate[0]][coordinate[1]] == source_role
        ]
        if changed:
            allowed.discard(count)
        if unchanged:
            allowed &= {count}
    return tuple(sorted(allowed))


def candidates(demos: Sequence[tuple[Grid, Grid]]) -> tuple[Candidate, ...]:
    """Compile only bindings satisfying necessary operator preconditions.

    Every condition is inverse semantics of an existing operator.  Lack of an
    observation preserves alternatives; it never selects an arbitrary value.
    """

    colors = _palette(demos)
    backgrounds = _backgrounds(demos)
    same_canvas = all(
        len(source) == len(expected) and len(source[0]) == len(expected[0])
        for source, expected in demos
    )
    result: set[Candidate] = set()
    if not same_canvas:
        result.add(Candidate("compress_cross_panels", ()))
        return tuple(result)

    separators = tuple(
        role
        for role in colors
        if all(_has_full_separator(source, role) for source, _ in demos)
    )
    column_separators = tuple(
        role
        for role in colors
        if all(_separator_column(source, role) is not None for source, _ in demos)
    )
    markers = tuple(
        role
        for role in colors
        if all(
            sum(value == role for row in source for value in row) == 1
            for source, _ in demos
        )
    )
    for background, separator in product(backgrounds, separators):
        if background != separator:
            result.add(
                Candidate(
                    "replicate_nonempty_panel",
                    tuple(sorted({"background": background, "separator": separator}.items())),
                )
            )
    for background, marker in product(backgrounds, markers):
        if background != marker:
            result.add(
                Candidate(
                    "stamp_pattern_at_marker",
                    tuple(sorted({"background": background, "marker": marker}.items())),
                )
            )
    for background, separator in product(backgrounds, column_separators):
        visible_pairs = _relocation_role_pairs(demos, separator)
        role_pairs = visible_pairs or tuple(permutations(colors, 2))
        for source_role, target_role in role_pairs:
            if len({background, separator, source_role, target_role}) != 4:
                continue
            for preserve_count in _preserve_values(
                demos,
                separator,
                source_role,
                target_role,
            ):
                result.add(
                    Candidate(
                        "relocate_separator",
                        tuple(
                            sorted(
                                {
                                    "background": background,
                                    "preserve_count": preserve_count,
                                    "separator": separator,
                                    "source_role": source_role,
                                    "target_role": target_role,
                                }.items()
                            )
                        ),
                    )
                )
    return tuple(sorted(result, key=lambda item: (item.graph, repr(item.bindings))))
