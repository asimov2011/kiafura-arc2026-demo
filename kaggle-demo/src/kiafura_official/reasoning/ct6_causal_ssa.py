"""Typed causal SSA backend for the frozen CT-6 O1--O10 contracts.

This module is self-contained by design: it does not import the historical
CT-2/CT-6 solvers and contains no task identifiers or target grids.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
from itertools import permutations, product
import json
from typing import Any, Iterable, Iterator, Mapping, Sequence


Grid = tuple[tuple[int, ...], ...]
Demo = tuple[Grid, Grid]
Point = tuple[int, int]


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("grid must be non-empty and rectangular")
    return grid


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    return value


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


@dataclass(frozen=True)
class WorkVector:
    cell_visits: int = 0
    relation_tests: int = 0
    proposed_writes: int = 0
    binding_candidates: int = 0

    def __add__(self, other: "WorkVector") -> "WorkVector":
        return WorkVector(
            self.cell_visits + other.cell_visits,
            self.relation_tests + other.relation_tests,
            self.proposed_writes + other.proposed_writes,
            self.binding_candidates + other.binding_candidates,
        )

    def within(self, limit: "WorkVector") -> bool:
        return all(a <= b for a, b in zip(self.as_tuple(), limit.as_tuple()))

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.cell_visits,
            self.relation_tests,
            self.proposed_writes,
            self.binding_candidates,
        )

    def as_dict(self) -> dict[str, int]:
        return dict(zip(
            ("cell_visits", "relation_tests", "proposed_writes", "binding_candidates"),
            self.as_tuple(),
            strict=True,
        ))


DEFAULT_LIMIT = WorkVector(250_000_000, 250_000_000, 250_000_000, 250_000_000)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class WorkAccount:
    limit: WorkVector
    used: WorkVector = WorkVector()

    def reserve(self, delta: WorkVector) -> None:
        proposed = self.used + delta
        if not proposed.within(self.limit):
            raise BudgetExceeded(f"work {proposed.as_tuple()} exceeds {self.limit.as_tuple()}")
        self.used = proposed


@dataclass(frozen=True)
class SSANode:
    node_id: str
    factor_id: str
    mode: str
    inputs: tuple[str, ...]
    output_type: str
    static: tuple[tuple[str, Any], ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "inputs": list(self.inputs),
            "mode": self.mode,
            "node_id": self.node_id,
            "output_type": self.output_type,
            "static": dict(self.static),
        }


@dataclass(frozen=True)
class CausalGraph:
    operator_id: str
    topology_id: str
    binding_schema: tuple[tuple[str, str], ...]
    nodes: tuple[SSANode, ...]
    output_node: str
    support: int

    def manifest(self) -> dict[str, Any]:
        factors = tuple(sorted({node.factor_id for node in self.nodes} | {"F7", "F10"}))
        return {
            "categories": {
                "control": {
                    "policy": "all-demo exact; one observational query region or abstain",
                    "topology": self.topology_id,
                },
                "executable_graph": [node.manifest() for node in self.nodes],
                "parameters_bindings": dict(self.binding_schema),
                "representation": {
                    "factors": list(factors),
                    "input": "demonstrations+query_grids",
                    "output": "verified_grid_region_or_abstention",
                },
                "residual": [
                    "execution", "shape", "dimension", "palette", "cell", "failed_demos"
                ],
                "resources": {
                    "accounting": "reserve Pareto WorkVector before candidate execution",
                    "limit": DEFAULT_LIMIT.as_dict(),
                },
                "verification_arrest": "F10 exact_consensus",
            },
            "operator_id": self.operator_id,
            "output_node": self.output_node,
            "support": self.support,
            "topology_id": self.topology_id,
        }


def _node(
    node_id: str,
    factor: str,
    mode: str,
    inputs: Sequence[str],
    output_type: str,
    **static: Any,
) -> SSANode:
    return SSANode(node_id, factor, mode, tuple(inputs), output_type, tuple(sorted(static.items())))


def frozen_graphs() -> tuple[CausalGraph, ...]:
    return (
        CausalGraph("O1", "T1", (("marker", "Color"), ("target", "Color"), ("replacement", "Color")), (
            _node("n1", "F1", "points_equal_binding", ("grid",), "PointSet", slot="marker"),
            _node("n2", "F3", "bbox", ("n1",), "BBox"),
            _node("n3", "F1", "points_equal_binding", ("grid",), "PointSet", slot="target"),
            _node("n4", "F5", "restrict_to_bbox", ("n3", "n2"), "PointSet"),
            _node("n5", "F9", "copy_paint_binding", ("grid", "n4"), "Grid", slot="replacement"),
        ), "n5", 2),
        CausalGraph("O2", "T2", (("background", "Color"),), (
            _node("n1", "F1", "points_by_nonbackground_color", ("grid",), "RolePointMap"),
            _node("n2", "F2", "components_by_role", ("n1",), "RoleComponentMap", connectivity=4),
            _node("n3", "F4", "minimum_diagonal_pairs", ("n2",), "RolePairMap"),
            _node("n4", "F5", "diagonal_segments", ("n3",), "RolePointMap"),
            _node("n5", "F9", "copy_paint_roles", ("grid", "n4"), "Grid"),
        ), "n5", 2),
        CausalGraph("O3", "T3", (("background", "Color"), ("source", "Color"), ("frame", "Color"), ("size", "PositiveInt")), (
            _node("n1", "F1", "points_equal_binding", ("grid",), "PointSet", slot="source"),
            _node("n2", "F2", "components", ("n1",), "ComponentSet", connectivity=4),
            _node("n3", "F3", "component_count_bbox", ("n2",), "MeasuredComponents"),
            _node("n4", "F4", "filter_component_size", ("n3",), "BBoxSet", slot="size"),
            _node("n5", "F5", "bbox_halo_one", ("grid", "n4"), "PointSet"),
            _node("n6", "F9", "copy_paint_background_binding", ("grid", "n5"), "Grid", background_slot="background", value_slot="frame"),
        ), "n6", 3),
        CausalGraph("O4", "T4", (("background", "Color"), ("source", "Color"), ("line", "Color")), (
            _node("n1", "F1", "points_equal_binding", ("grid",), "PointSet", slot="source"),
            _node("n2", "F2", "components", ("n1",), "ComponentSet", connectivity=4),
            _node("n3", "F3", "component_bboxes", ("n2",), "BBoxSet"),
            _node("n4", "F4", "visible_horizontal_box_pairs", ("n3",), "BBoxPairSet"),
            _node("n5", "F5", "horizontal_gap_rectangles", ("n4",), "PointSet"),
            _node("n6", "F9", "copy_paint_binding", ("grid", "n5"), "Grid", slot="line"),
        ), "n6", 2),
        CausalGraph("O5", "T5", (("background", "Color"), ("mapping", "ColorMap")), (
            _node("n1", "F1", "mapped_marker_points", ("grid",), "ValuedPointSet"),
            _node("n2", "F5", "chebyshev_neighbors_one", ("grid", "n1"), "ValuedPointSet"),
            _node("n3", "F9", "copy_paint_values_restore", ("grid", "n2", "n1"), "Grid"),
        ), "n3", 3),
        CausalGraph("O6", "T6", (("background", "Color"), ("separator_role", "SeparatorRole")), (
            _node("n1", "F6", "full_uniform_partitions", ("grid",), "PanelRegions"),
            _node("n2", "F3", "equal_panel_shapes", ("n1",), "PanelRegions"),
            _node("n3", "F8", "first_nonbackground_reading_order", ("n2",), "GridRegion"),
            _node("n4", "F9", "unique_grid_region", ("n3",), "Grid"),
        ), "n4", 3),
        CausalGraph("O7", "T7", (("background", "Color"), ("mark", "Color")), (
            _node("n1", "F6", "central_uniform_divider", ("grid",), "Divider"),
            _node("n2", "F8", "aligned_background_conjunction", ("grid", "n1"), "Mask"),
            _node("n3", "F9", "blank_mark_mask", ("grid", "n1", "n2"), "Grid"),
        ), "n3", 2),
        CausalGraph("O8", "T8", (("background", "Color"), ("first_color", "Color"), ("second_color", "Color"), ("line_color", "Color"), ("corner_mode", "CornerMode")), (
            _node("n1", "F1", "points_equal_binding", ("grid",), "PointSet", slot="first_color"),
            _node("n2", "F1", "points_equal_binding", ("grid",), "PointSet", slot="second_color"),
            _node("n3", "F4", "two_singleton_endpoints", ("n1", "n2"), "PointPair"),
            _node("n4", "F5", "orthogonal_l_path", ("n3",), "PointSet"),
            _node("n5", "F9", "copy_paint_background_binding", ("grid", "n4"), "Grid", background_slot="background", value_slot="line_color"),
        ), "n5", 2),
        CausalGraph("O9", "T9", (("crop_height", "PositiveInt"), ("crop_width", "PositiveInt"), ("odd_side", "Side")), (
            _node("n1", "F3", "height_parity", ("grid",), "Parity"),
            _node("n2", "F4", "parity_conditioned_side", ("n1",), "Side"),
            _node("n3", "F5", "top_corner_crop", ("grid", "n2"), "Grid"),
            _node("n4", "F9", "identity_grid", ("n3",), "Grid"),
        ), "n4", 3),
        CausalGraph("O10", "T10", (("background", "Color"), ("translations", "DisplacementMap")), (
            _node("n1", "F1", "nonbackground_valued_points", ("grid",), "ValuedPointSet"),
            _node("n2", "F5", "translate_points_by_value", ("grid", "n1"), "ValuedPointSet"),
            _node("n3", "F9", "blank_paint_values", ("grid", "n2"), "Grid"),
        ), "n3", 3),
    )


def memory_manifest(graphs: Sequence[CausalGraph] | None = None) -> dict[str, Any]:
    selected = frozen_graphs() if graphs is None else tuple(graphs)
    manifest = {
        "anti_copy": {
            "bytecode": 0,
            "complete_programs": 0,
            "literal_grids": 0,
            "source_imports": 0,
            "task_ids": 0,
        },
        "budget": DEFAULT_LIMIT.as_dict(),
        "factors": {
            "F1": "role_points",
            "F2": "components",
            "F3": "measure",
            "F4": "relate_select",
            "F5": "coordinate_field",
            "F6": "partition",
            "F7": "derive_binding",
            "F8": "aligned_reduce",
            "F9": "render",
            "F10": "exact_consensus",
        },
        "format": "kiafura-ct6-causal-ssa-memory-v2",
        "graphs": [graph.manifest() for graph in selected],
        "provenance": {
            "contract_sha256": "48e281c0f89eec3ed1d61f3a9bb9bd595b354ca9cdbd02929b8c6c3458a18cf2",
            "source_partition": "A-only",
        },
    }
    manifest["memory_sha256"] = canonical_sha256(manifest)
    return manifest


def graphs_from_manifest(memory: Mapping[str, Any]) -> tuple[CausalGraph, ...]:
    canonical = {graph.operator_id: graph for graph in frozen_graphs()}
    result = []
    for item in memory["graphs"]:
        operator_id = item["operator_id"]
        graph = canonical.get(operator_id)
        if graph is None:
            raise ValueError("memory graph differs from frozen specification")
        expected = graph.manifest()
        observed = dict(item)
        support = observed.pop("support", None)
        expected.pop("support")
        if expected != observed or not isinstance(support, int) or support < 1:
            raise ValueError("memory graph differs from frozen specification")
        result.append(replace(graph, support=support))
    return tuple(result)


def _palette(grids: Iterable[Grid]) -> tuple[int, ...]:
    return tuple(sorted({cell for grid in grids for row in grid for cell in row}))


def _context(demos: Sequence[Demo]) -> dict[str, Any]:
    inputs = tuple(source for source, _ in demos)
    outputs = tuple(expected for _, expected in demos)
    colors = _palette((*inputs, *outputs))
    input_colors = set(_palette(inputs))
    output_colors = set(_palette(outputs))
    added = tuple(sorted(output_colors - input_colors)) or colors
    dominant = tuple(
        min(Counter(cell for row in grid for cell in row), key=lambda value, grid=grid: (
            -Counter(cell for row in grid for cell in row)[value], value
        ))
        for grid in inputs
    )
    backgrounds = (dominant[0],) if dominant and len(set(dominant)) == 1 else colors
    return {
        "added": added,
        "backgrounds": backgrounds,
        "colors": colors,
        "input_colors": tuple(sorted(input_colors)),
        "inputs": inputs,
        "outputs": outputs,
    }


def _infer_surround_mapping(demos: Sequence[Demo], background: int) -> tuple[tuple[int, int], ...] | None:
    mapping: dict[int, int] = {}
    observed = 0
    for source, expected in demos:
        if len(source) != len(expected) or len(source[0]) != len(expected[0]):
            return None
        for row, values in enumerate(source):
            for column, value in enumerate(values):
                if value == background or expected[row][column] != value:
                    continue
                neighbors = {
                    expected[row + dr][column + dc]
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (dr, dc) != (0, 0)
                    and 0 <= row + dr < len(source)
                    and 0 <= column + dc < len(source[0])
                    and source[row + dr][column + dc] == background
                }
                if len(neighbors) != 1:
                    continue
                surround = next(iter(neighbors))
                if surround == background:
                    continue
                if value in mapping and mapping[value] != surround:
                    return None
                mapping[value] = surround
                observed += 1
    return tuple(sorted(mapping.items())) if observed and mapping else None


def _infer_translations(demos: Sequence[Demo], background: int) -> tuple[tuple[int, Point], ...] | None:
    mapping: dict[int, Point] = {}
    for source, expected in demos:
        colors = {cell for row in source for cell in row if cell != background}
        if {cell for row in expected for cell in row if cell != background} != colors:
            return None
        for color in colors:
            before = {(r, c) for r, row in enumerate(source) for c, value in enumerate(row) if value == color}
            after = {(r, c) for r, row in enumerate(expected) for c, value in enumerate(row) if value == color}
            if len(before) != len(after) or not before:
                return None
            candidates = sorted({(tr - r, tc - c) for r, c in before for tr, tc in after})
            delta = next((item for item in candidates if {(r + item[0], c + item[1]) for r, c in before} == after), None)
            if delta is None or (color in mapping and mapping[color] != delta):
                return None
            mapping[color] = delta
    return tuple(sorted(mapping.items())) if mapping else None


def binding_candidates(graph: CausalGraph, demos: Sequence[Demo]) -> Iterator[dict[str, Any]]:
    ctx = _context(demos)
    colors = ctx["colors"]
    added = ctx["added"]
    backgrounds = ctx["backgrounds"]
    input_colors = ctx["input_colors"]
    if graph.operator_id == "O1":
        for marker, target, replacement in product(colors, colors, added):
            if len({marker, target, replacement}) == 3:
                yield {"marker": marker, "target": target, "replacement": replacement}
    elif graph.operator_id == "O2":
        for background in backgrounds:
            yield {"background": background}
    elif graph.operator_id == "O3":
        for background, source, frame, size in product(
            backgrounds, input_colors, added, range(1, 6)
        ):
            if source != background and frame not in (background, source):
                yield {"background": background, "source": source, "frame": frame, "size": size}
    elif graph.operator_id == "O4":
        for background, source, line in product(backgrounds, input_colors, added):
            if source != background and line not in (background, source):
                yield {"background": background, "source": source, "line": line}
    elif graph.operator_id == "O5":
        for background in backgrounds:
            mapping = _infer_surround_mapping(demos, background)
            if mapping is not None:
                yield {"background": background, "mapping": mapping}
    elif graph.operator_id == "O6":
        for background in backgrounds:
            yield {"background": background, "separator_role": "full_uniform_lines"}
    elif graph.operator_id == "O7":
        for background, mark in product(backgrounds, added):
            if mark != background:
                yield {"background": background, "mark": mark}
    elif graph.operator_id == "O8":
        for first, second in permutations(colors, 2):
            for line, background, corner in product(
                added, backgrounds, ("first_row_second_column", "second_row_first_column")
            ):
                if len({first, second, line, background}) == 4:
                    yield {"background": background, "first_color": first, "second_color": second, "line_color": line, "corner_mode": corner}
    elif graph.operator_id == "O9":
        shapes = {(len(expected), len(expected[0])) for expected in ctx["outputs"]}
        if len(shapes) == 1:
            height, width = next(iter(shapes))
            if all(height <= len(source) and width <= len(source[0]) for source in ctx["inputs"]):
                for side in ("left", "right"):
                    yield {"crop_height": height, "crop_width": width, "odd_side": side}
    elif graph.operator_id == "O10":
        for background in backgrounds:
            mapping = _infer_translations(demos, background)
            if mapping is not None:
                yield {"background": background, "translations": mapping}


def _binding_derivation_bound(graph: CausalGraph, demos: Sequence[Demo]) -> WorkVector:
    """Reserve a conservative envelope before inspecting values or expanding bindings."""
    visible_cells = sum(
        len(grid) * len(grid[0])
        for demo in demos
        for grid in demo
    )
    relation_multiplier = 9 if graph.operator_id == "O5" else (
        max(1, visible_cells) if graph.operator_id == "O10" else 1
    )
    return WorkVector(
        cell_visits=visible_cells,
        relation_tests=visible_cells * relation_multiplier,
    )


def _components(points: Iterable[Point], connectivity: int) -> tuple[frozenset[Point], ...]:
    remaining = set(points)
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1)) if connectivity == 4 else tuple(
        (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
    )
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = {seed}
        stack = [seed]
        while stack:
            row, column = stack.pop()
            for dr, dc in steps:
                neighbor = row + dr, column + dc
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        result.append(frozenset(component))
    return tuple(sorted(result, key=lambda item: (len(item), tuple(sorted(item)))))


def _bbox(points: Iterable[Point]) -> tuple[int, int, int, int]:
    values = tuple(points)
    if not values:
        raise ValueError("bbox requires points")
    rows = [row for row, _ in values]
    columns = [column for _, column in values]
    return min(rows), max(rows), min(columns), max(columns)


def _panel_boxes(grid: Grid, separator: int) -> tuple[tuple[int, int, int, int], ...]:
    height, width = len(grid), len(grid[0])
    rows = [r for r, values in enumerate(grid) if all(value == separator for value in values)]
    columns = [c for c in range(width) if all(grid[r][c] == separator for r in range(height))]
    if not rows and not columns:
        raise ValueError("no full separator")
    rb = (-1, *rows, height)
    cb = (-1, *columns, width)
    boxes = tuple(
        (rb[i] + 1, rb[i + 1], cb[j] + 1, cb[j + 1])
        for i in range(len(rb) - 1)
        for j in range(len(cb) - 1)
        if rb[i] + 1 < rb[i + 1] and cb[j] + 1 < cb[j + 1]
    )
    if len(boxes) < 2:
        raise ValueError("separator does not define panels")
    return boxes


def _panel(grid: Grid, box: tuple[int, int, int, int]) -> Grid:
    top, bottom, left, right = box
    return tuple(tuple(row[left:right]) for row in grid[top:bottom])


def _estimate(graph: CausalGraph, grid: Grid, binding: Mapping[str, Any]) -> WorkVector:
    n = len(grid) * len(grid[0])
    colors = len(_palette((grid,)))
    if graph.operator_id == "O1":
        return WorkVector(2 * n + n, n, n, 0)
    if graph.operator_id == "O2":
        return WorkVector((colors + 1) * n, n * n, n, 0)
    if graph.operator_id in {"O3", "O5", "O10"}:
        return WorkVector(3 * n, 9 * n, n, 0)
    if graph.operator_id == "O4":
        return WorkVector(3 * n, n * n, n, 0)
    if graph.operator_id == "O6":
        return WorkVector(max(1, colors) * 3 * n, max(1, colors) * n, n, 0)
    if graph.operator_id in {"O7", "O8"}:
        return WorkVector(2 * n, n, n, 0)
    if graph.operator_id == "O9":
        area = int(binding["crop_height"]) * int(binding["crop_width"])
        return WorkVector(area, 1, area, 0)
    raise ValueError(graph.operator_id)


def _factor(node: SSANode, state: dict[str, Any], binding: Mapping[str, Any]) -> Any:
    static = dict(node.static)
    inputs = [state[name] for name in node.inputs]
    mode = node.mode
    grid = state.get("grid")
    if mode == "points_equal_binding":
        color = binding[static["slot"]]
        return frozenset((r, c) for r, row in enumerate(inputs[0]) for c, value in enumerate(row) if value == color)
    if mode == "points_by_nonbackground_color":
        background = binding["background"]
        return {color: frozenset((r, c) for r, row in enumerate(inputs[0]) for c, value in enumerate(row) if value == color) for color in _palette((inputs[0],)) if color != background}
    if mode == "mapped_marker_points":
        mapping = dict(binding["mapping"])
        return tuple((r, c, mapping[value]) for r, row in enumerate(inputs[0]) for c, value in enumerate(row) if value in mapping)
    if mode == "nonbackground_valued_points":
        background = binding["background"]
        return tuple((r, c, value) for r, row in enumerate(inputs[0]) for c, value in enumerate(row) if value != background)
    if mode == "bbox":
        return _bbox(inputs[0])
    if mode == "component_count_bbox":
        return tuple((len(component), _bbox(component)) for component in inputs[0])
    if mode == "component_bboxes":
        return tuple(_bbox(component) for component in inputs[0])
    if mode == "height_parity":
        return len(inputs[0]) % 2
    if mode == "equal_panel_shapes":
        return inputs[0]
    if mode == "components":
        connectivity = binding[static["connectivity_slot"]] if "connectivity_slot" in static else static["connectivity"]
        return _components(inputs[0], int(connectivity))
    if mode == "components_by_role":
        return {color: _components(points, int(static["connectivity"])) for color, points in inputs[0].items()}
    if mode == "minimum_diagonal_pairs":
        result = {}
        for color, components in inputs[0].items():
            if len(components) != 2:
                continue
            pairs = sorted(
                (max(abs(a[0] - b[0]), abs(a[1] - b[1])), a, b)
                for a in components[0] for b in components[1]
                if abs(a[0] - b[0]) == abs(a[1] - b[1])
            )
            if pairs:
                _, first, second = pairs[0]
                result[color] = first, second
        return result
    if mode == "filter_component_size":
        return tuple(box for size, box in inputs[0] if size == binding[static["slot"]])
    if mode == "visible_horizontal_box_pairs":
        boxes = inputs[0]
        result = []
        for li, left in enumerate(boxes):
            tl, bl, _, rl = left
            for ri, right in enumerate(boxes):
                if li == ri:
                    continue
                tr, br, lr, _ = right
                if lr <= rl or max(tl, tr) > min(bl, br):
                    continue
                blocked = any(
                    index not in (li, ri) and rl < cl and cr < lr and max(tl, tr, ct) <= min(bl, br, cb)
                    for index, (ct, cb, cl, cr) in enumerate(boxes)
                )
                if not blocked:
                    result.append((left, right))
        return tuple(result)
    if mode == "two_singleton_endpoints":
        if len(inputs[0]) != 1 or len(inputs[1]) != 1:
            raise ValueError("endpoints not singleton")
        return next(iter(inputs[0])), next(iter(inputs[1]))
    if mode == "parity_conditioned_side":
        odd_side = binding["odd_side"]
        return odd_side if inputs[0] else ("right" if odd_side == "left" else "left")
    if mode == "restrict_to_bbox":
        top, bottom, left, right = inputs[1]
        return frozenset((r, c) for r, c in inputs[0] if top <= r <= bottom and left <= c <= right)
    if mode == "diagonal_segments":
        result = {}
        for color, (first, second) in inputs[0].items():
            row, column = first
            tr, tc = second
            dr = 1 if tr > row else -1
            dc = 1 if tc > column else -1
            points = []
            row += dr
            column += dc
            while (row, column) != (tr, tc):
                points.append((row, column))
                row += dr
                column += dc
            result[color] = tuple(points)
        return result
    if mode == "bbox_halo_one":
        result = set()
        for top, bottom, left, right in inputs[1]:
            for row in range(max(0, top - 1), min(len(inputs[0]), bottom + 2)):
                for column in range(max(0, left - 1), min(len(inputs[0][0]), right + 2)):
                    result.add((row, column))
        return frozenset(result)
    if mode == "horizontal_gap_rectangles":
        result = set()
        for left, right in inputs[0]:
            tl, bl, _, rl = left
            tr, br, lr, _ = right
            for row in range(max(tl, tr), min(bl, br) + 1):
                for column in range(rl + 1, lr):
                    result.add((row, column))
        return frozenset(result)
    if mode == "chebyshev_neighbors_one":
        valued = []
        for row, column, value in inputs[1]:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    target = row + dr, column + dc
                    if (dr, dc) != (0, 0) and 0 <= target[0] < len(inputs[0]) and 0 <= target[1] < len(inputs[0][0]):
                        valued.append((target[0], target[1], value))
        return tuple(valued)
    if mode == "orthogonal_l_path":
        (fr, fc), (sr, sc) = inputs[0]
        corner = (fr, sc) if binding["corner_mode"] == "first_row_second_column" else (sr, fc)
        points = []
        for endpoint in ((fr, fc), (sr, sc)):
            row, column = endpoint
            dr = (corner[0] > row) - (corner[0] < row)
            dc = (corner[1] > column) - (corner[1] < column)
            while (row, column) != corner:
                points.append((row, column))
                if row != corner[0]:
                    row += dr
                else:
                    column += dc
            points.append((row, column))
        return tuple(points)
    if mode == "top_corner_crop":
        width = int(binding["crop_width"])
        left = 0 if inputs[1] == "left" else len(inputs[0][0]) - width
        return tuple(tuple(row[left:left + width]) for row in inputs[0][:int(binding["crop_height"])])
    if mode == "translate_points_by_value":
        mapping = dict(binding["translations"])
        result = []
        for row, column, value in inputs[1]:
            if value not in mapping:
                raise ValueError("unobserved translation")
            dr, dc = mapping[value]
            target = row + dr, column + dc
            if not (0 <= target[0] < len(inputs[0]) and 0 <= target[1] < len(inputs[0][0])):
                raise ValueError("translation outside canvas")
            result.append((target[0], target[1], value))
        return tuple(result)
    if mode == "full_uniform_partitions":
        regions = []
        for separator in _palette((inputs[0],)):
            try:
                boxes = _panel_boxes(inputs[0], separator)
                panels = tuple(_panel(inputs[0], box) for box in boxes)
                regions.append(panels)
            except ValueError:
                continue
        return tuple(regions)
    if mode == "central_uniform_divider":
        dividers = [column for column in range(len(inputs[0][0])) if len({inputs[0][row][column] for row in range(len(inputs[0]))}) == 1 and column * 2 + 1 == len(inputs[0][0])]
        if len(dividers) != 1:
            raise ValueError("central divider absent")
        return dividers[0]
    if mode == "first_nonbackground_reading_order":
        outputs = set()
        background = binding["background"]
        for panels in inputs[0]:
            shapes = {(len(panel), len(panel[0])) for panel in panels}
            if len(shapes) != 1:
                continue
            height, width = next(iter(shapes))
            outputs.add(tuple(tuple(next((panel[row][column] for panel in panels if panel[row][column] != background), background) for column in range(width)) for row in range(height)))
        return frozenset(outputs)
    if mode == "aligned_background_conjunction":
        divider = inputs[1]
        background = binding["background"]
        return frozenset((row, column) for row in range(len(inputs[0])) for column in range(divider) if inputs[0][row][column] == background and inputs[0][row][divider + 1 + column] == background)
    if mode == "copy_paint_binding":
        output = [list(row) for row in inputs[0]]
        value = binding[static["slot"]]
        for row, column in inputs[1]:
            output[row][column] = value
        return tuple(tuple(row) for row in output)
    if mode == "copy_paint_background_binding":
        output = [list(row) for row in inputs[0]]
        background = binding[static["background_slot"]]
        value = binding[static["value_slot"]]
        for row, column in inputs[1]:
            if output[row][column] == background:
                output[row][column] = value
        return tuple(tuple(row) for row in output)
    if mode == "copy_paint_roles":
        output = [list(row) for row in inputs[0]]
        for color, points in inputs[1].items():
            for row, column in points:
                output[row][column] = color
        return tuple(tuple(row) for row in output)
    if mode == "copy_paint_values_restore":
        output = [list(row) for row in inputs[0]]
        for row, column, value in inputs[1]:
            output[row][column] = value
        for row, column, _ in inputs[2]:
            output[row][column] = inputs[0][row][column]
        return tuple(tuple(row) for row in output)
    if mode == "unique_grid_region":
        if len(inputs[0]) != 1:
            raise ValueError("grid region ambiguous")
        return next(iter(inputs[0]))
    if mode == "blank_mark_mask":
        divider = inputs[1]
        background = binding["background"]
        mark = binding["mark"]
        mask = inputs[2]
        return tuple(tuple(mark if (row, column) in mask else background for column in range(divider)) for row in range(len(inputs[0])))
    if mode == "identity_grid":
        return inputs[0]
    if mode == "blank_paint_values":
        background = binding["background"]
        output = [[background] * len(inputs[0][0]) for _ in inputs[0]]
        for row, column, value in inputs[1]:
            output[row][column] = value
        return tuple(tuple(row) for row in output)
    raise ValueError(f"unsupported factor mode {node.factor_id}:{mode}")


def execute_graph(
    graph: CausalGraph,
    grid: Grid,
    binding: Mapping[str, Any],
    account: WorkAccount,
) -> tuple[Grid, tuple[dict[str, Any], ...]]:
    account.reserve(_estimate(graph, grid, binding))
    state: dict[str, Any] = {"grid": grid}
    trace = []
    for node in graph.nodes:
        value = _factor(node, state, binding)
        state[node.node_id] = value
        trace.append({
            "factor_id": node.factor_id,
            "node_id": node.node_id,
            "output_sha256": canonical_sha256(value),
            "output_type": node.output_type,
        })
    output = state[graph.output_node]
    if not isinstance(output, tuple) or not output or not isinstance(output[0], tuple):
        raise ValueError("graph output is not a Grid")
    return output, tuple(trace)


@dataclass(frozen=True)
class Resolution:
    status: str
    outputs: tuple[Grid, ...] | None
    activated: tuple[dict[str, Any], ...]
    work: WorkVector


def resolve(
    demos: Sequence[Demo],
    queries: Sequence[Grid],
    memory: Mapping[str, Any],
) -> Resolution:
    limit = WorkVector(**memory["budget"])
    account = WorkAccount(limit)
    regions: dict[str, dict[str, Any]] = {}
    try:
        for graph in graphs_from_manifest(memory):
            account.reserve(_binding_derivation_bound(graph, demos))
            for binding in binding_candidates(graph, demos):
                account.reserve(WorkVector(binding_candidates=1))
                traces = []
                exact = True
                for source, expected in demos:
                    try:
                        obtained, trace = execute_graph(graph, source, binding, account)
                    except (IndexError, KeyError, TypeError, ValueError):
                        exact = False
                        break
                    traces.append(trace)
                    if obtained != expected:
                        exact = False
                        break
                if not exact:
                    continue
                query_outputs = []
                query_traces = []
                try:
                    for query in queries:
                        obtained, trace = execute_graph(graph, query, binding, account)
                        query_outputs.append(obtained)
                        query_traces.append(trace)
                except (IndexError, KeyError, TypeError, ValueError):
                    continue
                outputs = tuple(query_outputs)
                key = canonical_sha256(outputs)
                regions.setdefault(key, {"outputs": outputs, "activated": []})
                regions[key]["activated"].append({
                    "binding": _jsonable(binding),
                    "operator_id": graph.operator_id,
                    "topology_id": graph.topology_id,
                    "trace_sha256": canonical_sha256((traces, query_traces)),
                })
    except BudgetExceeded:
        return Resolution("budget", None, (), account.used)
    if len(regions) != 1:
        return Resolution("abstain", None, (), account.used)
    region = next(iter(regions.values()))
    activated = tuple(sorted(region["activated"], key=lambda item: canonical_sha256(item)))
    return Resolution("exact_demo_consensus", region["outputs"], activated, account.used)
