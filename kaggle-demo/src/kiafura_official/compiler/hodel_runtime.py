"""Fail-closed runtime for Typed Graph IR.

M0A deliberately keeps the existing small executor set.  It does *not* pretend
that every registered Hodel primitive is executable: a missing executor, an
unresolved input, or an unresolved function reference raises
``GraphExecutionError`` and therefore yields abstention in ``execute_memory``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from schema.typed_graph_ir import Constant, MemoryManifest, ParametricGraph, ValueType

Grid = list[list[int]]
Executor = Callable[..., Any]


class GraphExecutionError(RuntimeError):
    pass


class ExecutorUnavailableError(GraphExecutionError):
    pass


def _g(value: Any) -> Grid:
    if isinstance(value, (list, tuple)) and (
        not value or isinstance(value[0], (list, tuple))
    ):
        return [list(row) for row in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    raise GraphExecutionError(f"Expected grid-like value, got {type(value).__name__}")


def _bg(grid: Grid) -> int:
    if not grid or not grid[0]:
        raise GraphExecutionError("Background is undefined for an empty grid")
    return Counter(value for row in grid for value in row).most_common(1)[0][0]


def _components(grid: Grid, bg: int, diag: bool = True) -> list[set[tuple[int, int]]]:
    height, width = len(grid), len(grid[0])
    pending = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] != bg
    }
    steps = (
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        if diag
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    result: list[set[tuple[int, int]]] = []
    while pending:
        seed = min(pending)
        pending.remove(seed)
        stack = [seed]
        component = {seed}
        while stack:
            row, column = stack.pop()
            for drow, dcolumn in steps:
                neighbor = (row + drow, column + dcolumn)
                if neighbor in pending:
                    pending.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


# Existing conservative executor subset.
def op_hconcat(a: Any, b: Any) -> Grid:
    ga, gb = _g(a), _g(b)
    if len(ga) != len(gb):
        raise GraphExecutionError("hconcat requires equal heights")
    return [list(ra) + list(rb) for ra, rb in zip(ga, gb, strict=True)]


def op_vconcat(a: Any, b: Any) -> Grid:
    ga, gb = _g(a), _g(b)
    if ga and gb and len(ga[0]) != len(gb[0]):
        raise GraphExecutionError("vconcat requires equal widths")
    return [list(row) for row in ga + gb]


def op_hupscale(a: Any, factor: int) -> Grid:
    if not isinstance(factor, int) or factor <= 0:
        raise GraphExecutionError(f"Invalid hupscale factor {factor!r}")
    return [[value for value in row for _ in range(factor)] for row in _g(a)]


def op_vupscale(a: Any, factor: int) -> Grid:
    if not isinstance(factor, int) or factor <= 0:
        raise GraphExecutionError(f"Invalid vupscale factor {factor!r}")
    return [list(row) for row in _g(a) for _ in range(factor)]


def op_hmirror(a: Any) -> Any:
    # dsl.py hmirror: grids flip row order; patches mirror around the
    # horizontal midline of their bounding box.
    if isinstance(a, (set, frozenset)):
        from src.kiafura_official.compiler.hodel_primitives import (
            _is_object,
            op_lrcorner,
            op_ulcorner2,
        )

        d = op_ulcorner2(a)[0] + op_lrcorner(a)[0]
        if _is_object(a):
            return type(a)((v, (d - i, j)) for v, (i, j) in a)
        return type(a)((d - i, j) for i, j in a)
    return [list(row) for row in _g(a)[::-1]]


def op_vmirror(a: Any) -> Any:
    if isinstance(a, (set, frozenset)):
        from src.kiafura_official.compiler.hodel_primitives import _vmirror_any

        return _vmirror_any(a)
    return [list(row[::-1]) for row in _g(a)]


def op_rot90(a: Any) -> Grid:
    return [list(row) for row in zip(*_g(a)[::-1], strict=True)]


def op_rot180(a: Any) -> Grid:
    return [list(reversed(row)) for row in reversed(_g(a))]


def op_rot270(a: Any) -> Grid:
    return [list(row) for row in zip(*_g(a), strict=True)][::-1]


def op_crop(a: Any, origin: tuple[int, int], size: tuple[int, int]) -> Grid:
    grid = _g(a)
    row0, column0 = origin
    rows, columns = size
    return [row[column0 : column0 + columns] for row in grid[row0 : row0 + rows]]


def op_ofcolor(a: Any, color: int) -> frozenset[tuple[int, int]]:
    return frozenset(
        (row, column)
        for row, values in enumerate(_g(a))
        for column, value in enumerate(values)
        if value == color
    )


def op_first(values: Any) -> Any:
    # dsl.py: first item of the container.  Tuples/lists keep positional
    # order; for sets (unordered in dsl.py) the deterministic analogue is the
    # least element under the total _sort_key order.  (The previous min()
    # implementation was not a total order: on sets of unorderable elements
    # such as frozensets it silently returned an arbitrary element, and
    # op_last returned the same one.)
    if not values:
        raise GraphExecutionError("first() on empty collection")
    if isinstance(values, (tuple, list)):
        return values[0]
    from src.kiafura_official.compiler.hodel_primitives import _ordered

    return _ordered(values)[0]


def op_last(values: Any) -> Any:
    # dsl.py: last item of the container.  See op_first for the set rule.
    if not values:
        raise GraphExecutionError("last() on empty collection")
    if isinstance(values, (tuple, list)):
        return values[-1]
    from src.kiafura_official.compiler.hodel_primitives import _ordered

    return _ordered(values)[-1]


def op_size(values: Any) -> int:
    return len(values)


def op_ulcorner(points: Any) -> tuple[int, int]:
    # dsl.py ulcorner works on any patch via toindices (the previous
    # implementation iterated raw cells and broke on colored objects).
    from src.kiafura_official.compiler.hodel_primitives import op_ulcorner2

    return op_ulcorner2(points)


def op_urcorner(points: Any) -> tuple[int, int]:
    from src.kiafura_official.compiler.hodel_primitives import op_urcorner2

    return op_urcorner2(points)


def op_height(value: Any) -> int:
    # dsl.py height: number of rows for grids, bounding-box height for patches.
    if isinstance(value, (set, frozenset)):
        from src.kiafura_official.compiler.hodel_primitives import _patch_dims

        return _patch_dims(value)[0]
    return len(_g(value))


def op_width(value: Any) -> int:
    if isinstance(value, (set, frozenset)):
        from src.kiafura_official.compiler.hodel_primitives import _patch_dims

        return _patch_dims(value)[1]
    grid = _g(value)
    return len(grid[0]) if grid else 0


def op_merge(collection: Any) -> Any:
    # dsl.py merge flattens one level preserving the outer container type
    # (set union for sets, concatenation for tuples); the previous
    # implementation was set-union only and broke on tuples of grids.
    from src.kiafura_official.compiler.hodel_primitives import _merge

    return _merge(collection)


def op_fill(a: Any, color: int, points: Any) -> Grid:
    # dsl.py fill applies toindices(patch): both index sets and colored
    # objects are valid patches.
    from src.kiafura_official.compiler.hodel_primitives import _toindices

    grid = _g(a)
    output = [list(row) for row in grid]
    for row, column in _toindices(points):
        if 0 <= row < len(output) and 0 <= column < len(output[0]):
            output[row][column] = color
    return output


def op_replace(a: Any, old: int, new: int) -> Grid:
    return [[new if value == old else value for value in row] for row in _g(a)]


def op_switch(a: Any, first: int, second: int) -> Grid:
    return [
        [second if value == first else first if value == second else value for value in row]
        for row in _g(a)
    ]


def op_tophalf(a: Any) -> Grid:
    grid = _g(a)
    return [list(row) for row in grid[: len(grid) // 2]]


def op_bottomhalf(a: Any) -> Grid:
    # dsl.py drops the middle row for odd heights: grid[h//2 + h%2:].
    grid = _g(a)
    return [list(row) for row in grid[len(grid) // 2 + len(grid) % 2 :]]


def op_lefthalf(a: Any) -> Grid:
    return [list(row[: len(row) // 2]) for row in _g(a)]


def op_righthalf(a: Any) -> Grid:
    # dsl.py righthalf = rot270(bottomhalf(rot90(g))): for odd widths the
    # middle column is dropped, i.e. columns [w//2 + w%2:].
    return [list(row[len(row) // 2 + len(row) % 2 :]) for row in _g(a)]


def op_canvas(color: int, size: tuple[int, int]) -> Grid:
    rows, columns = size
    return [[color] * columns for _ in range(rows)]


def op_initset(value: Any) -> frozenset[Any]:
    # dsl.py builds frozenset({value}); grid values are frozen to nested
    # tuples so they stay hashable.
    from src.kiafura_official.compiler.hodel_primitives import _hashable

    return frozenset({_hashable(value)})


def op_astuple(a: Any, b: Any) -> tuple[Any, Any]:
    return a, b


def op_totuple(values: Any) -> tuple[Any, ...]:
    # dsl.py totuple is tuple(container); for sets we use the deterministic
    # _sort_key order instead of arbitrary frozenset iteration.
    if isinstance(values, (tuple, list)):
        return tuple(values)
    from src.kiafura_official.compiler.hodel_primitives import _sort_key

    return tuple(sorted(values, key=_sort_key))


def op_shape(a: Any) -> tuple[int, int]:
    # dsl.py shape works on grids and patches alike.
    if isinstance(a, (set, frozenset)):
        from src.kiafura_official.compiler.hodel_primitives import _patch_dims

        return _patch_dims(a)
    grid = _g(a)
    return len(grid), len(grid[0]) if grid else 0


def op_palette(a: Any) -> set[int]:
    # dsl.py palette accepts grids and colored objects.
    if isinstance(a, (set, frozenset)):
        return {value for value, _ in a}
    return {value for row in _g(a) for value in row}


def op_mostcolor(a: Any) -> int:
    if isinstance(a, (set, frozenset)):
        return Counter(value for value, _ in a).most_common(1)[0][0]
    grid = _g(a)
    return Counter(value for row in grid for value in row).most_common(1)[0][0]


def op_leastcolor(a: Any) -> int:
    if isinstance(a, (set, frozenset)):
        return Counter(value for value, _ in a).most_common()[-1][0]
    grid = _g(a)
    return Counter(value for row in grid for value in row).most_common()[-1][0]


def op_increment(value: Any) -> Any:
    # dsl.py Numerical-polymorphic: ints and (i, j) pairs.
    if isinstance(value, tuple):
        return (value[0] + 1, value[1] + 1)
    return value + 1


def op_decrement(value: Any) -> Any:
    if isinstance(value, tuple):
        return (value[0] - 1, value[1] - 1)
    return value - 1


def op_subtract(a: Any, b: Any) -> Any:
    from src.kiafura_official.compiler.hodel_primitives import _num2

    return _num2(a, b, lambda x, y: x - y, "subtract")


def op_double(value: Any) -> Any:
    if isinstance(value, tuple):
        return (value[0] * 2, value[1] * 2)
    return value * 2


def op_identity(value: Any) -> Any:
    return value


EXECUTOR_REGISTRY: dict[str, Executor] = {
    "hconcat": op_hconcat,
    "vconcat": op_vconcat,
    "hupscale": op_hupscale,
    "vupscale": op_vupscale,
    "hmirror": op_hmirror,
    "vmirror": op_vmirror,
    "rot90": op_rot90,
    "rot180": op_rot180,
    "rot270": op_rot270,
    "crop": op_crop,
    "ofcolor": op_ofcolor,
    "first": op_first,
    "last": op_last,
    "size": op_size,
    "ulcorner": op_ulcorner,
    "urcorner": op_urcorner,
    "height": op_height,
    "width": op_width,
    "merge": op_merge,
    "fill": op_fill,
    "replace": op_replace,
    "switch": op_switch,
    "tophalf": op_tophalf,
    "bottomhalf": op_bottomhalf,
    "lefthalf": op_lefthalf,
    "righthalf": op_righthalf,
    "canvas": op_canvas,
    "initset": op_initset,
    "astuple": op_astuple,
    "totuple": op_totuple,
    "shape": op_shape,
    "palette": op_palette,
    "mostcolor": op_mostcolor,
    "leastcolor": op_leastcolor,
    "increment": op_increment,
    "decrement": op_decrement,
    "subtract": op_subtract,
    "double": op_double,
    "identity": op_identity,
}


# B900 completion: the full Hodel primitive set lives in hodel_primitives and
# is merged here so EXECUTOR_REGISTRY stays the single final map.  Existing
# executors above are not overridden.
from src.kiafura_official.compiler.hodel_primitives import PRIMITIVE_EXECUTORS

for _name, _executor in PRIMITIVE_EXECUTORS.items():
    if _name in EXECUTOR_REGISTRY:
        raise RuntimeError(f"Primitive {_name!r} would override an existing executor")
    EXECUTOR_REGISTRY[_name] = _executor


def executor_available(operator: str) -> bool:
    return operator == "invoke" or operator in EXECUTOR_REGISTRY


def _resolve_constant(constant: Constant) -> Any:
    if constant.type == ValueType.FUNCTION:
        executor = EXECUTOR_REGISTRY.get(str(constant.value))
        if executor is None:
            raise ExecutorUnavailableError(
                f"No executor for function literal {constant.value!r}"
            )
        return executor
    return constant.value


def execute_graph(
    graph: ParametricGraph,
    input_grid: Grid,
    bindings: dict[str, Any] | None = None,
) -> Grid | None:
    bindings = bindings or {}
    values: dict[str, Any] = {"INPUT": input_grid}

    for parameter in graph.parameters:
        if parameter.name in bindings:
            bound = bindings[parameter.name]
            values[parameter.name] = (
                _resolve_constant(bound) if isinstance(bound, Constant) else bound
            )
        elif parameter.default is not None:
            values[parameter.name] = _resolve_constant(parameter.default)
        else:
            raise GraphExecutionError(
                f"Missing required binding {parameter.name!r} in graph {graph.graph_id}"
            )

    for node in graph.nodes:
        try:
            missing = [reference for reference in node.inputs if reference not in values]
            if missing:
                raise GraphExecutionError(
                    f"Unresolved inputs {missing!r} for {node.output_name}"
                )
            arguments = [values[reference] for reference in node.inputs]

            if node.operator == "invoke":
                if not arguments or not callable(arguments[0]):
                    raise GraphExecutionError(
                        f"INVOKE target is not callable at {node.output_name}"
                    )
                result = arguments[0](*arguments[1:])
            else:
                executor = EXECUTOR_REGISTRY.get(node.operator)
                if executor is None:
                    raise ExecutorUnavailableError(
                        f"No executor registered for {node.operator!r}"
                    )
                result = executor(*arguments)

            values[node.output_name] = result
        except GraphExecutionError:
            raise
        except (IndexError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise GraphExecutionError(
                f"Error in {node.operator}{node.inputs} at {node.output_name}: {exc}"
            ) from exc

    if graph.output_node not in values:
        raise GraphExecutionError(
            f"Graph output {graph.output_node!r} was not produced"
        )
    output = values[graph.output_node]
    return output


def execute_memory(
    memory: MemoryManifest,
    input_grid: Grid,
    parameters: dict[str, Any] | None = None,
) -> list[tuple[str, Grid | None]]:
    """Execute every graph; any structural/semantic failure becomes abstention."""

    results: list[tuple[str, Grid | None]] = []
    for graph in (*memory.graphs, *memory.singletons):
        try:
            output = execute_graph(graph, input_grid, parameters)
        except GraphExecutionError:
            output = None
        results.append((graph.graph_id, output))
    return results
