"""Sound output-to-binding witnesses for high-cost A100 operator families.

The predicates below encode necessary effects of frozen operators.  They only
remove a candidate when its declared causal effect cannot explain a visible
demonstration delta; absence of evidence keeps the alternative alive.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Sequence

from kiafura_official.reasoning import advanced_spatial_algebra as advanced
from kiafura_official.reasoning import component_relation_algebra as component
from kiafura_official.reasoning import connection_topology_algebra as connection
from kiafura_official.reasoning import count_render_algebra as count_render
from kiafura_official.reasoning import geometric_pattern_algebra as geometry
from kiafura_official.reasoning import hierarchical_panel_algebra as hierarchical
from kiafura_official.reasoning import local_pattern_algebra as local
from kiafura_official.reasoning import object_structure_algebra as objects
from kiafura_official.reasoning import pattern_recolor_algebra as pattern
from kiafura_official.reasoning import ray_pattern_algebra as ray
from kiafura_official.reasoning import region_topology_algebra as region
from kiafura_official.reasoning import relational_repair_algebra as repair
from kiafura_official.reasoning import selection_extraction_algebra as selection
from kiafura_official.reasoning import spatial_transform_algebra as spatial


Grid = tuple[tuple[int, ...], ...]
Demos = Sequence[tuple[Grid, Grid]]


def _pairs(source: Grid, expected: Grid) -> set[tuple[int, int]]:
    if len(source) != len(expected) or len(source[0]) != len(expected[0]):
        return set()
    return {
        (source[row][column], expected[row][column])
        for row in range(len(source))
        for column in range(len(source[0]))
        if source[row][column] != expected[row][column]
    }


def _changed(demos: Demos) -> tuple[set[tuple[int, int]], ...]:
    return tuple(_pairs(source, expected) for source, expected in demos)


def _all_deltas_match(
    changes: Sequence[set[tuple[int, int]]],
    predicate: Callable[[int, int], bool],
) -> bool:
    return all(all(predicate(before, after) for before, after in pairs) for pairs in changes)


def _all_sources_contain(demos: Demos, role: int) -> bool:
    return all(any(role in row for row in source) for source, _ in demos)


def _palette(grid: Grid) -> set[int]:
    return {value for row in grid for value in row}


def _fold_precondition(
    grid: Grid, background: int, needle: int, support: int
) -> bool:
    """Exact non-rendering precondition of ``fold_tail``."""

    height, width = len(grid), len(grid[0])
    found = 0
    for empty_row in range(height):
        for empty_column in range(width):
            if grid[empty_row][empty_column] != background:
                continue
            direction = None
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                row, column = empty_row - dr, empty_column - dc
                while (
                    0 <= row < height
                    and 0 <= column < width
                    and grid[row][column] == needle
                ):
                    row, column = row - dr, column - dc
                if (
                    0 <= row < height
                    and 0 <= column < width
                    and grid[row][column] == support
                ):
                    direction = (dr, dc)
                    break
            if direction is None:
                continue
            dr, dc = direction
            tail_length = 0
            row, column = empty_row + dr, empty_column + dc
            while (
                0 <= row < height
                and 0 <= column < width
                and grid[row][column] == needle
            ):
                tail_length += 1
                row, column = row + dr, column + dc
            if not tail_length:
                continue
            found += 1
            new_dr, new_dc = (
                (0, 1 if empty_column < (width - 1) / 2 else -1)
                if dr
                else (1 if empty_row < (height - 1) / 2 else -1, 0)
            )
            for distance in range(1, tail_length + 1):
                row = empty_row + new_dr * distance
                column = empty_column + new_dc * distance
                if not (0 <= row < height and 0 <= column < width):
                    return False
    return found > 0


def _copy_anchor_precondition(grid: Grid, background: int) -> bool:
    items = advanced.components(grid, background, 8)
    singletons = {
        grid[next(iter(item))[0]][next(iter(item))[1]]
        for item in items
        if len(item) == 1
    }
    for item in items:
        colors = Counter(grid[row][column] for row, column in item)
        if len(colors) >= 2 and min(colors, key=colors.get) in singletons:
            return True
    return False


def _decode_local_summary(grid: Grid, background: int):
    nonbackground = [
        value for row in grid for value in row if value != background
    ]
    if not nonbackground:
        return None
    base = Counter(nonbackground).most_common(1)[0][0]
    active_rows = [
        row
        for row, values in enumerate(grid)
        if any(value != background for value in values)
    ]
    active_columns = [
        column
        for column in range(len(grid[0]))
        if any(grid[row][column] != background for row in range(len(grid)))
    ]
    top, bottom = min(active_rows), max(active_rows)
    left, right = min(active_columns), max(active_columns)
    separator_rows = [
        row
        for row in range(top, bottom + 1)
        if all(value == background for value in grid[row])
    ]
    separator_columns = [
        column
        for column in range(left, right + 1)
        if all(grid[row][column] == background for row in range(top, bottom + 1))
    ]
    tile_height = separator_rows[0] - top if separator_rows else bottom - top + 1
    tile_width = separator_columns[0] - left if separator_columns else right - left + 1
    if tile_height <= 0 or tile_width <= 0:
        return None
    bindings = {}
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            if value not in (background, base):
                bindings[
                    (
                        (row - top) % (tile_height + 1) // 2,
                        (column - left) % (tile_width + 1) // 2,
                    )
                ] = value
    if not bindings:
        return None
    tile_rows = max(row for row, _ in bindings) + 1
    tile_columns = max(column for _, column in bindings) + 1
    return (
        tile_rows * tile_height + tile_rows - 1,
        tile_columns * tile_width + tile_columns - 1,
        {base, *bindings.values()},
    )


def _partition_ranges(separators: Sequence[int], limit: int):
    result = []
    start = 0
    for stop in (*separators, limit):
        if start < stop:
            result.append((start, stop))
        start = stop + 1
    return result


def _canonical_macro_summary(grid: Grid, separator: int, excluded: int):
    height, width = len(grid), len(grid[0])
    separator_rows = [
        row
        for row in range(height)
        if sum(value == separator for value in grid[row]) >= width // 2
    ]
    separator_columns = [
        column
        for column in range(width)
        if sum(grid[row][column] == separator for row in range(height))
        >= height // 2
    ]
    row_ranges = _partition_ranges(separator_rows, height)
    column_ranges = _partition_ranges(separator_columns, width)
    tiles = []
    for top, bottom in row_ranges:
        for left, right in column_ranges:
            tile = tuple(
                tuple(grid[row][column] for column in range(left, right))
                for row in range(top, bottom)
            )
            if excluded not in _palette(tile):
                tiles.append(tile)
    if not tiles:
        return None
    shapes = {(len(tile), len(tile[0])) for tile in tiles if tile and tile[0]}
    if len(shapes) != 1:
        return None
    canonical = Counter(tiles).most_common(1)[0][0]
    base = Counter(value for row in canonical for value in row).most_common(1)[0][0]
    active = {
        (row, column)
        for row, values in enumerate(canonical)
        for column, value in enumerate(values)
        if value != base
    }
    if any(
        row >= len(row_ranges) or column >= len(column_ranges)
        for row, column in active
    ):
        return None
    return {separator, base, *(_palette(canonical))}


def _legend_summary(grid: Grid, background: int, frame_color: int):
    frame = [
        (row, column)
        for row, values in enumerate(grid)
        for column, value in enumerate(values)
        if value == frame_color
    ]
    if not frame:
        return None
    top, bottom = min(row for row, _ in frame), max(row for row, _ in frame)
    left, right = min(column for _, column in frame), max(column for _, column in frame)
    markers = {
        grid[row][column]
        for row in range(top + 1, bottom)
        for column in range(left + 1, right)
        if grid[row][column] not in (background, frame_color)
    }
    if not markers:
        return None
    for color in markers:
        if not any(
            value == color and not (top <= row <= bottom and left <= column <= right)
            for row, values in enumerate(grid)
            for column, value in enumerate(values)
        ):
            return None
    return {background, *markers}


def local_pattern_candidates(demos: Demos):
    changes = _changed(demos)
    output = []
    for candidate in local.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph == "recolor_center_with_rays":
            keep = all(
                pairs and pairs <= {(values["center"], values["replacement"])}
                for pairs in changes
            )
        elif candidate.graph == "complete_cross_coverage":
            keep = _all_deltas_match(
                changes,
                lambda before, after: before == values["target"]
                and after == values["replacement"],
            ) and any(changes)
        elif candidate.graph == "recolor_header_parity":
            keep = _all_deltas_match(
                changes,
                lambda before, after: before == values["source"]
                and after == values["replacement"],
            ) and any(changes)
        elif candidate.graph == "annotate_nearest_full_axis":
            keep = _all_deltas_match(
                changes,
                lambda before, after: before == values["background"],
            ) and all(
                not pairs or any(values["marker"] in row for row in source)
                for (source, _), pairs in zip(demos, changes, strict=True)
            )
        elif candidate.graph == "cross_marker_lines":
            roles = {values["marker_a"], values["marker_b"], values["intersection"]}
            keep = _all_deltas_match(changes, lambda before, after: after in roles)
            keep &= all(
                sum(value in {values["marker_a"], values["marker_b"]} for row in source for value in row)
                >= 2
                for source, _ in demos
            )
        if keep:
            output.append(candidate)
    return tuple(output)


def advanced_spatial_candidates(demos: Demos):
    changes = _changed(demos)
    output = []
    for candidate in advanced.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph == "fold_tail":
            keep = _all_deltas_match(
                changes,
                lambda before, after: after in {values["erased"], values["needle"]},
            )
            keep &= all(
                all(role in {value for row in source for value in row} for role in (values["background"], values["needle"], values["support"]))
                for source, _ in demos
            )
            keep &= all(
                _fold_precondition(
                    source,
                    values["background"],
                    values["needle"],
                    values["support"],
                )
                for source, _ in demos
            )
            if values["erased"] != values["needle"]:
                keep &= all(
                    (values["needle"], values["erased"]) in pairs
                    for pairs in changes
                )
        elif candidate.graph == "build_oriented_barrier":
            roles = {values["marker"], values["on"], values["off"], values["gap"]}
            keep = _all_deltas_match(changes, lambda before, after: after in roles)
            keep &= all(_palette(expected) <= roles for _, expected in demos)
            keep &= all(
                sum(value == values["marker"] for row in source for value in row) == 1
                for source, _ in demos
            )
        elif candidate.graph == "tile_complementary_pieces":
            keep = all(
                len(advanced.components(source, values["background"], 4)) == 2
                for source, _ in demos
            )
        elif candidate.graph == "reflect_pivot_axes":
            for source, pairs in zip((source for source, _ in demos), changes, strict=True):
                counts = Counter(
                    value for row in source for value in row if value != values["background"]
                )
                if len(counts) < 2:
                    keep = False
                    break
                ink = max(counts, key=counts.get)
                keep &= all(after == ink for _, after in pairs)
        elif candidate.graph == "copy_anchor_templates":
            keep = all(
                _copy_anchor_precondition(source, values["background"])
                for source, _ in demos
            )
        if keep:
            output.append(candidate)
    return tuple(output)


def hierarchical_panel_candidates(demos: Demos):
    """Compile necessary typed summaries before hierarchical expansion."""

    output = []
    for candidate in hierarchical.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph == "decode_local_bindings":
            for source, expected in demos:
                summary = _decode_local_summary(source, values["background"])
                if summary is None:
                    keep = False
                    break
                height, width, palette = summary
                keep &= (len(expected), len(expected[0])) == (height, width)
                keep &= _palette(expected) <= palette
        elif candidate.graph == "propagate_canonical_macro_tile":
            for source, expected in demos:
                palette = _canonical_macro_summary(
                    source, values["separator"], values["excluded"]
                )
                if palette is None:
                    keep = False
                    break
                keep &= (len(expected), len(expected[0])) == (
                    len(source),
                    len(source[0]),
                )
                keep &= _palette(expected) <= palette
        elif candidate.graph == "arrange_legend_glyphs":
            for source, expected in demos:
                palette = _legend_summary(
                    source, values["background"], values["frame_color"]
                )
                if palette is None:
                    keep = False
                    break
                keep &= _palette(expected) <= palette
        if keep:
            output.append(candidate)
    return tuple(output)


def pattern_recolor_candidates(demos: Demos):
    changes = _changed(demos)
    output = []
    for candidate in pattern.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph == "recolor_diagonal_orientations":
            keep = all(
                pairs
                and all(
                    before == values["source"]
                    and after in {values["dominant"], values["minority"]}
                    for before, after in pairs
                )
                for pairs in changes
            )
        elif candidate.graph == "continue_progression":
            keep = all(
                pairs and all(after == values["ink"] for _, after in pairs)
                for pairs in changes
            )
            keep &= all(
                sum(value == values["seed"] for row in source for value in row) >= 2
                for source, _ in demos
            )
        if keep:
            output.append(candidate)
    return tuple(output)


def connection_topology_candidates(demos: Demos):
    changes = _changed(demos)
    output = []
    for candidate in connection.candidates(demos):
        values = dict(candidate.bindings)
        if candidate.graph == "connect_border_marker_paths":
            keep = _all_deltas_match(
                changes, lambda before, after: after == values["marker"]
            )
        elif candidate.graph == "connect_terminals_to_spine":
            keep = _all_deltas_match(
                changes,
                lambda before, after: before == values["background"]
                and after == values["ink"],
            )
        elif candidate.graph == "connect_domino_via_obstacle":
            keep = _all_deltas_match(
                changes,
                lambda before, after: before
                in {values["background"], values["target"]}
                and after == values["source"],
            )
        else:
            keep = _all_deltas_match(
                changes,
                lambda before, after: before == values["background"]
                and after == values["ink"],
            )
        if keep and any(changes):
            output.append(candidate)
    return tuple(output)


def count_render_candidates(demos: Demos):
    output = []
    for candidate in count_render.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph == "solid_square_to_row":
            keep = all(
                {value for row in expected for value in row}
                <= {values["label"], values["background"]}
                for _, expected in demos
            )
        elif candidate.graph == "band_side_counts":
            keep = all(
                {value for row in expected for value in row}
                <= {values["band"], values["background"]}
                for _, expected in demos
            )
        elif candidate.graph == "bbox_markers_to_bars":
            keep = True
            for source, expected in demos:
                predicted_roles = {
                    value
                    for row in source
                    for value in row
                    if value
                    not in {values["marker"], values["input_background"]}
                }
                expected_roles = {
                    value
                    for row in expected
                    for value in row
                    if value != values["output_background"]
                }
                keep &= len(expected) == len(predicted_roles)
                keep &= expected_roles <= predicted_roles
        if keep:
            output.append(candidate)
    return tuple(output)


def ray_pattern_candidates(demos: Demos):
    changes = _changed(demos)
    output = []
    for candidate in ray.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph == "extend_cross_rays":
            marker = values["marker"]
            keep = all(
                any(
                    source[row][column] != marker
                    and all(
                        source[row + dr][column + dc] == marker
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    )
                    for row in range(1, len(source) - 1)
                    for column in range(1, len(source[0]) - 1)
                )
                for source, _ in demos
            )
        elif candidate.graph == "extend_bar_diagonals":
            keep = _all_deltas_match(
                changes,
                lambda before, after: after == values["collision"]
                or any(after in row for source, _ in demos for row in source),
            )
        elif candidate.graph == "propagate_block_rays":
            keep = _all_deltas_match(
                changes,
                lambda before, after: after
                in {values["diagonal"], values["staircase"]},
            )
            keep &= all(
                all(
                    any(role in row for row in source)
                    for role in (values["block"], values["barrier"])
                )
                for source, _ in demos
            )
        if keep:
            output.append(candidate)
    return tuple(output)


def region_topology_candidates(demos: Demos):
    changes = _changed(demos)
    output = []
    for candidate in region.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph == "project_markers_to_separator":
            keep = _all_sources_contain(demos, values["separator"])
            keep &= _all_deltas_match(
                changes,
                lambda before, after: before == values["background"]
                or after == values["background"],
            )
        elif candidate.graph == "fill_ranked_rectangles":
            keep = _all_deltas_match(
                changes,
                lambda before, after: after in set(values["rank_colors"]),
            )
        if keep:
            output.append(candidate)
    return tuple(output)


def object_structure_candidates(demos: Demos):
    changes = _changed(demos)
    output = []
    for candidate in objects.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph == "factor_polyomino":
            keep = all(
                any(before == values["source_color"] for before, _ in pairs)
                and all(
                    before != values["source_color"]
                    or after in {values["square_color"], values["line_color"]}
                    for before, after in pairs
                )
                for pairs in changes
            )
        if keep:
            output.append(candidate)
    return tuple(output)


def geometric_pattern_candidates(demos: Demos):
    changes = _changed(demos)
    output = []
    for candidate in geometry.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph == "complete_alternating_triangle":
            keep = _all_deltas_match(
                changes,
                lambda before, after: after
                in {values["primary"], values["secondary"]},
            )
            keep &= _all_sources_contain(demos, values["primary"])
        if keep:
            output.append(candidate)
    return tuple(output)


def spatial_transform_candidates(demos: Demos):
    changes = _changed(demos)
    output = []
    for candidate in spatial.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph == "reflect_shape_from_marker":
            keep = _all_deltas_match(
                changes,
                lambda before, after: after
                in {values["background"], values["shape_color"]},
            )
            keep &= all(
                all(
                    any(role in row for row in source)
                    for role in (values["shape_color"], values["marker_color"])
                )
                for source, _ in demos
            )
        if keep:
            output.append(candidate)
    return tuple(output)


def selection_extraction_candidates(demos: Demos):
    changes = _changed(demos)
    output = []
    deletion_graphs = {
        "retain_key_internal_color",
        "remove_bbox_contained_components",
    }
    for candidate in selection.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph in deletion_graphs:
            keep = _all_deltas_match(
                changes,
                lambda before, after: after == values["background"],
            ) and any(changes)
        elif candidate.graph == "retain_dominant_lattice":
            keep = True
            for (source, _), pairs in zip(demos, changes, strict=True):
                counts = Counter(
                    value
                    for row in source
                    for value in row
                    if value != values["background"]
                )
                if not counts:
                    keep = False
                    break
                main = min(counts, key=lambda value: (-counts[value], value))
                keep &= all(
                    after in {values["background"], main}
                    for _, after in pairs
                )
        if keep:
            output.append(candidate)
    return tuple(output)


def relational_repair_candidates(demos: Demos):
    changes = _changed(demos)
    output = []
    for candidate in repair.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph == "contour_turns":
            keep = _all_deltas_match(
                changes,
                lambda before, after: after
                in {values["outer_color"], values["inner_color"]},
            )
        elif candidate.graph == "repair_occluded_analogues":
            keep = all(
                any(values["occluder"] in row for row in source)
                for source, _ in demos
            )
        elif candidate.graph == "propagate_boundary_dominance":
            keep = _all_sources_contain(demos, values["marker"])
        if keep:
            output.append(candidate)
    return tuple(output)


def component_relation_candidates(demos: Demos):
    changes = _changed(demos)
    output = []
    for candidate in component.candidates(demos):
        values = dict(candidate.bindings)
        keep = True
        if candidate.graph == "erase_marker_rich":
            keep = all(
                all(after == component.background_role(source) for _, after in pairs)
                for (source, _), pairs in zip(demos, changes, strict=True)
            )
            keep &= _all_sources_contain(demos, values["marker"])
        if keep:
            output.append(candidate)
    return tuple(output)
