"""Hodel DSL primitive executors for the Typed Graph IR runtime.

Semantics follow Michael Hodel's public ARC DSL (``dsl.py``), adapted to this
repo's representations:

- grids are ``list[list[int]]`` (not tuples of tuples);
- patches/objects are frozensets of ``(i, j)`` indices or
  ``(color, (i, j))`` cells (frozenset, like dsl.py, so patches stay hashable
  and composable inside containers);
- all operations are total or raise ``GraphExecutionError`` (fail-closed);
- every iteration whose order can leak into the result goes through
  ``_sort_key`` so execution is deterministic.

Where dsl.py itself is order-dependent on a frozenset (``first``/``argmax``/
``mostcommon`` ties, ``totuple`` of a set, ...), the deterministic analogue
used here orders by ``_sort_key``; this is documented per function.

Composed functions (``compose``/``chain``/``fork``/``lbind``/``rbind``/
``power``/``matcher``) get a structural ``__name__`` so that sets of functions
have a deterministic order too.
"""

from __future__ import annotations

import inspect
from collections import Counter
from typing import Any, Callable

from src.kiafura_official.compiler.hodel_runtime import GraphExecutionError

Grid = list[list[int]]
Index = tuple[int, int]
Cell = tuple[int, Index]


# ---------------------------------------------------------------------------
# Deterministic ordering helpers
# ---------------------------------------------------------------------------

def _fname(function: Callable) -> str:
    return getattr(function, "__name__", None) or type(function).__name__


def _sort_key(value: Any) -> tuple:
    """Total, deterministic sort key across the value kinds we handle."""

    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, str):
        return (1, value)
    if isinstance(value, tuple):
        return (2, tuple(_sort_key(v) for v in value))
    if isinstance(value, (set, frozenset)):
        return (3, tuple(sorted((_sort_key(v) for v in value))))
    if isinstance(value, list):
        return (4, tuple(_sort_key(v) for v in value))
    if callable(value):
        return (5, _fname(value))
    return (6, repr(value))


def _ordered(container: Any) -> list:
    """Deterministic iteration: tuples/lists keep their order, sets sort."""

    if isinstance(container, (list, tuple)):
        return list(container)
    return sorted(container, key=_sort_key)


def _require_container(value: Any, name: str) -> Any:
    if not isinstance(value, (set, frozenset, tuple, list)):
        raise GraphExecutionError(f"{name} expects a container, got {type(value).__name__}")
    return value


def _freeze(value: Any) -> Any:
    """Hashable, deterministic copy: grids become nested tuples, sets frozensets."""

    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    return value


def _hashable(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return _freeze(value)


def _merge(containers: Any) -> Any:
    """dsl.py merge: flatten one level, preserving the outer container type.

    Sets union into a frozenset; tuples/lists concatenate.  Set elements are
    flattened in ``_sort_key`` order to keep the result deterministic.
    """

    if isinstance(containers, (set, frozenset)):
        result: set = set()
        for part in containers:
            result.update(part)
        return frozenset(result)
    if isinstance(containers, (tuple, list)):
        out: list = []
        for part in containers:
            if isinstance(part, (set, frozenset)):
                out.extend(sorted(part, key=_sort_key))
            else:
                out.extend(part)
        return tuple(out)
    raise GraphExecutionError(f"merge expects a container, got {type(containers).__name__}")


# ---------------------------------------------------------------------------
# Grid / patch representation helpers
# ---------------------------------------------------------------------------

def _grid(value: Any) -> Grid:
    if isinstance(value, (list, tuple)) and (
        not value or isinstance(value[0], (list, tuple))
    ):
        return [list(row) for row in value]
    raise GraphExecutionError(f"Expected grid-like value, got {type(value).__name__}")


def _is_object(patch: Any) -> bool:
    """True when the patch cells carry colors: elements are (color, (i, j))."""

    if not patch:
        return False
    first = _ordered(patch)[0]
    return isinstance(first, tuple) and len(first) == 2 and isinstance(first[1], tuple)


def _toindices(patch: Any) -> frozenset[Index]:
    _require_container(patch, "toindices")
    if len(patch) == 0:
        return frozenset()
    if _is_object(patch):
        return frozenset(index for _, index in patch)
    return frozenset(patch)


def _mostcolor(grid: Grid) -> int:
    if not grid or not grid[0]:
        raise GraphExecutionError("mostcolor is undefined for an empty grid")
    return Counter(value for row in grid for value in row).most_common(1)[0][0]


def _uppermost(patch: Any) -> int:
    return min(i for i, _ in _toindices(patch))


def _lowermost(patch: Any) -> int:
    return max(i for i, _ in _toindices(patch))


def _leftmost(patch: Any) -> int:
    return min(j for _, j in _toindices(patch))


def _rightmost(patch: Any) -> int:
    return max(j for _, j in _toindices(patch))


def _patch_dims(piece: Any) -> tuple[int, int]:
    """dsl.py ``height``/``width`` for both grids and patches."""

    if isinstance(piece, (list, tuple)) and (
        not piece or isinstance(piece[0], (list, tuple))
    ):
        grid = _grid(piece)
        return len(grid), len(grid[0]) if grid else 0
    indices = _toindices(piece)
    if not indices:
        return 0, 0
    rows = [i for i, _ in indices]
    columns = [j for _, j in indices]
    return max(rows) - min(rows) + 1, max(columns) - min(columns) + 1


def _shift(patch: Any, directions: Index) -> Any:
    if not patch:
        return patch
    di, dj = directions
    if _is_object(patch):
        return frozenset((value, (i + di, j + dj)) for value, (i, j) in patch)
    return frozenset((i + di, j + dj) for i, j in patch)


def _backdrop(patch: Any) -> frozenset[Index]:
    indices = _toindices(patch)
    if not indices:
        return frozenset()
    si = min(i for i, _ in indices)
    ei = max(i for i, _ in indices)
    sj = min(j for _, j in indices)
    ej = max(j for _, j in indices)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))


def _box_lines(ai: int, aj: int, bi: int, bj: int) -> frozenset[Index]:
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


# ---------------------------------------------------------------------------
# Scalar arithmetic (dsl.py Numerical-polymorphic)
# ---------------------------------------------------------------------------

def _num2(a: Any, b: Any, fn: Callable[[int, int], int], name: str) -> Any:
    if isinstance(a, int) and isinstance(b, int):
        return fn(a, b)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return (fn(a[0], b[0]), fn(a[1], b[1]))
    if isinstance(a, int) and isinstance(b, tuple):
        return (fn(a, b[0]), fn(a, b[1]))
    if isinstance(a, tuple) and isinstance(b, int):
        return (fn(a[0], b), fn(a[1], b))
    raise GraphExecutionError(f"{name} expects int/pair operands, got {type(a).__name__}, {type(b).__name__}")


def _num1(x: Any, fn: Callable[[int], int], name: str) -> Any:
    if isinstance(x, int):
        return fn(x)
    if isinstance(x, tuple) and len(x) == 2:
        return (fn(x[0]), fn(x[1]))
    raise GraphExecutionError(f"{name} expects int or pair, got {type(x).__name__}")


def op_add(a: Any, b: Any) -> Any:
    return _num2(a, b, lambda x, y: x + y, "add")


def op_multiply(a: Any, b: Any) -> Any:
    return _num2(a, b, lambda x, y: x * y, "multiply")


def op_divide(a: Any, b: Any) -> Any:
    def _div(x: int, y: int) -> int:
        if y == 0:
            raise GraphExecutionError("divide by zero")
        return x // y

    return _num2(a, b, _div, "divide")


def op_halve(x: Any) -> Any:
    return _num1(x, lambda v: v // 2, "halve")


def op_invert(x: Any) -> Any:
    return _num1(x, lambda v: -v, "invert")


def op_sign(x: Any) -> Any:
    return _num1(x, lambda v: 0 if v == 0 else (1 if v > 0 else -1), "sign")


def op_crement(x: Any) -> Any:
    return _num1(x, lambda v: 0 if v == 0 else (v + 1 if v > 0 else v - 1), "crement")


def op_even(n: Any) -> bool:
    if not isinstance(n, int):
        raise GraphExecutionError(f"even expects int, got {type(n).__name__}")
    return n % 2 == 0


def op_positive(n: Any) -> bool:
    if not isinstance(n, int):
        raise GraphExecutionError(f"positive expects int, got {type(n).__name__}")
    return n > 0


def op_flip(b: Any) -> bool:
    return not b


def op_both(a: Any, b: Any) -> bool:
    return bool(a) and bool(b)


def op_either(a: Any, b: Any) -> bool:
    return bool(a) or bool(b)


def op_greater(a: int, b: int) -> bool:
    return a > b


def op_equality(a: Any, b: Any) -> bool:
    return a == b


def op_contained(value: Any, container: Any) -> bool:
    _require_container(container, "contained")
    return _hashable(value) in container


def op_toivec(i: int) -> Index:
    return (i, 0)


def op_tojvec(j: int) -> Index:
    return (0, j)


# ---------------------------------------------------------------------------
# Generic containers
# ---------------------------------------------------------------------------

def op_combine(a: Any, b: Any) -> Any:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(a) + tuple(b)
    if isinstance(a, (set, frozenset)) and isinstance(b, (set, frozenset)):
        return frozenset(set(a) | set(b))
    raise GraphExecutionError(
        f"combine expects two sets or two tuples, got {type(a).__name__}, {type(b).__name__}"
    )


def op_intersection(a: Any, b: Any) -> Any:
    return frozenset(set(a) & set(b)) if isinstance(a, (set, frozenset)) else type(a)(
        e for e in a if e in b
    )


def op_difference(a: Any, b: Any) -> Any:
    return type(a)(e for e in a if e not in b)


def op_insert(value: Any, container: Any) -> Any:
    if isinstance(container, (set, frozenset)):
        return frozenset(set(container) | {_hashable(value)})
    if isinstance(container, tuple):
        return tuple(container) + (value,)
    raise GraphExecutionError(f"insert expects set/tuple, got {type(container).__name__}")


def op_remove(value: Any, container: Any) -> Any:
    probe = _hashable(value)
    return type(container)(e for e in container if e != value and e != probe)


def op_other(container: Any, value: Any) -> Any:
    probe = _hashable(value)
    rest = [e for e in _ordered(container) if e != value and e != probe]
    if not rest:
        raise GraphExecutionError("other() on a container without another value")
    return rest[0]


def op_product(a: Any, b: Any) -> frozenset[tuple[Any, Any]]:
    return frozenset((i, j) for j in b for i in a)


def op_pair(a: Any, b: Any) -> tuple:
    return tuple(zip(a, b))


def op_repeat(item: Any, num: int) -> tuple:
    if not isinstance(num, int) or num < 0:
        raise GraphExecutionError(f"repeat expects a non-negative int, got {num!r}")
    return tuple(item for _ in range(num))


def op_dedupe(tup: Any) -> tuple:
    seen: list[Any] = []
    result = []
    for e in tup:
        if e not in seen:
            seen.append(e)
            result.append(e)
    return tuple(result)


def op_interval(start: int, stop: int, step: int) -> tuple[int, ...]:
    if step == 0:
        raise GraphExecutionError("interval with step 0")
    return tuple(range(start, stop, step))


def op_minimum(container: Any) -> int:
    items = _ordered(container)
    return min(items) if items else 0


def op_maximum(container: Any) -> int:
    items = _ordered(container)
    return max(items) if items else 0


def op_mostcommon(container: Any) -> Any:
    items = list(container)
    if not items:
        raise GraphExecutionError("mostcommon() on empty container")
    # Elements may be unhashable in this repo (grids are lists); count on
    # frozen copies.  Deterministic tie-break: smallest under _sort_key.
    keys = [_freeze(e) for e in items]
    counts = Counter(keys)
    best_count = max(counts.values())
    chosen = min((k for k in counts if counts[k] == best_count), key=_sort_key)
    return items[keys.index(chosen)]


def op_leastcommon(container: Any) -> Any:
    items = list(container)
    if not items:
        raise GraphExecutionError("leastcommon() on empty container")
    keys = [_freeze(e) for e in items]
    counts = Counter(keys)
    best_count = min(counts.values())
    chosen = min((k for k in counts if counts[k] == best_count), key=_sort_key)
    return items[keys.index(chosen)]


def op_order(container: Any, compfunc: Callable) -> tuple:
    return tuple(sorted(_ordered(container), key=compfunc))


def op_valmax(container: Any, compfunc: Callable) -> Any:
    items = _ordered(container)
    if not items:
        return compfunc(0)
    best = max(compfunc(e) for e in items)
    return compfunc(min((e for e in items if compfunc(e) == best), key=_sort_key))


def op_valmin(container: Any, compfunc: Callable) -> Any:
    items = _ordered(container)
    if not items:
        return compfunc(0)
    best = min(compfunc(e) for e in items)
    return compfunc(min((e for e in items if compfunc(e) == best), key=_sort_key))


def op_argmax(container: Any, compfunc: Callable) -> Any:
    items = _ordered(container)
    if not items:
        raise GraphExecutionError("argmax() on empty container")
    best = max(compfunc(e) for e in items)
    return min((e for e in items if compfunc(e) == best), key=_sort_key)


def op_argmin(container: Any, compfunc: Callable) -> Any:
    items = _ordered(container)
    if not items:
        raise GraphExecutionError("argmin() on empty container")
    best = min(compfunc(e) for e in items)
    return min((e for e in items if compfunc(e) == best), key=_sort_key)


def op_extract(container: Any, condition: Callable) -> Any:
    for e in _ordered(container):
        if condition(e):
            return e
    raise GraphExecutionError("extract() found no matching element")


def op_sfilter(container: Any, condition: Callable) -> Any:
    return type(container)(e for e in container if condition(e))


def op_mfilter(container: Any, function: Callable) -> Any:
    return _merge(op_sfilter(container, function))


def op_apply(function: Callable, container: Any) -> Any:
    return type(container)(_hashable(function(e)) for e in container)


def op_rapply(functions: Any, value: Any) -> Any:
    return type(functions)(function(value) for function in _ordered(functions))


def op_mapply(function: Callable, container: Any) -> Any:
    return _merge(op_apply(function, container))


def op_papply(function: Callable, a: Any, b: Any) -> tuple:
    return tuple(function(i, j) for i, j in zip(a, b))


def op_mpapply(function: Callable, a: Any, b: Any) -> Any:
    return _merge(op_papply(function, a, b))


def op_prapply(function: Callable, a: Any, b: Any) -> frozenset:
    return frozenset(_hashable(function(i, j)) for j in b for i in a)


# ---------------------------------------------------------------------------
# Higher-order constructors
# ---------------------------------------------------------------------------

def op_compose(outer: Callable, inner: Callable) -> Callable:
    composed = lambda x: outer(inner(x))  # noqa: E731
    composed.__name__ = f"compose({_fname(outer)},{_fname(inner)})"
    return composed


def op_chain(h: Callable, g: Callable, f: Callable) -> Callable:
    chained = lambda x: h(g(f(x)))  # noqa: E731
    chained.__name__ = f"chain({_fname(h)},{_fname(g)},{_fname(f)})"
    return chained


def op_fork(outer: Callable, a: Callable, b: Callable) -> Callable:
    forked = lambda x: outer(a(x), b(x))  # noqa: E731
    forked.__name__ = f"fork({_fname(outer)},{_fname(a)},{_fname(b)})"
    return forked


def _arity_of(function: Callable) -> int:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return 2
    return sum(
        1
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )


def _stable_repr(value: Any) -> str:
    if isinstance(value, (int, str, bool)):
        return repr(value)
    return repr(_sort_key(value))


def op_lbind(function: Callable, fixed: Any) -> Callable:
    n = _arity_of(function)
    if n == 2:
        bound = lambda y: function(fixed, y)  # noqa: E731
    elif n == 3:
        bound = lambda y, z: function(fixed, y, z)  # noqa: E731
    else:
        bound = lambda y, z, a: function(fixed, y, z, a)  # noqa: E731
    bound.__name__ = f"lbind({_fname(function)},{_stable_repr(fixed)})"
    return bound


def op_rbind(function: Callable, fixed: Any) -> Callable:
    n = _arity_of(function)
    if n == 2:
        bound = lambda x: function(x, fixed)  # noqa: E731
    elif n == 3:
        bound = lambda x, y: function(x, y, fixed)  # noqa: E731
    else:
        bound = lambda x, y, z: function(x, y, z, fixed)  # noqa: E731
    bound.__name__ = f"rbind({_fname(function)},{_stable_repr(fixed)})"
    return bound


def op_power(function: Callable, n: int) -> Callable:
    if not isinstance(n, int) or n < 1:
        raise GraphExecutionError(f"power expects int >= 1, got {n!r}")
    if n == 1:
        return function

    def powered(x: Any) -> Any:
        result = x
        for _ in range(n):
            result = function(result)
        return result

    powered.__name__ = f"power({_fname(function)},{n})"
    return powered


def op_matcher(function: Callable, target: Any) -> Callable:
    matched = lambda x: function(x) == target  # noqa: E731
    matched.__name__ = f"matcher({_fname(function)},{_stable_repr(target)})"
    return matched


def op_branch(condition: Any, a: Any, b: Any) -> Any:
    return a if condition else b


# ---------------------------------------------------------------------------
# Patch geometry
# ---------------------------------------------------------------------------

def op_toindices(patch: Any) -> frozenset[Index]:
    return _toindices(patch)


def op_asindices(grid: Any) -> frozenset[Index]:
    g = _grid(grid)
    return frozenset((i, j) for i in range(len(g)) for j in range(len(g[0])))


def op_ulcorner2(patch: Any) -> Index:
    # dsl.py: tuple(map(min, zip(*toindices(patch)))) — total on the empty
    # patch, where it returns the empty tuple (zip of zero iterables).
    indices = _toindices(patch)
    if not indices:
        return ()
    return min(i for i, _ in indices), min(j for _, j in indices)


def op_urcorner2(patch: Any) -> Index:
    indices = _toindices(patch)
    if not indices:
        return ()
    return min(i for i, _ in indices), max(j for _, j in indices)


def op_llcorner(patch: Any) -> Index:
    indices = _toindices(patch)
    if not indices:
        return ()
    return max(i for i, _ in indices), min(j for _, j in indices)


def op_lrcorner(patch: Any) -> Index:
    indices = _toindices(patch)
    if not indices:
        return ()
    return max(i for i, _ in indices), max(j for _, j in indices)


def op_uppermost(patch: Any) -> int:
    return _uppermost(patch)


def op_lowermost(patch: Any) -> int:
    return _lowermost(patch)


def op_leftmost(patch: Any) -> int:
    return _leftmost(patch)


def op_rightmost(patch: Any) -> int:
    return _rightmost(patch)


def op_center(patch: Any) -> Index:
    h, w = _patch_dims(patch)
    return _uppermost(patch) + h // 2, _leftmost(patch) + w // 2


def op_centerofmass(patch: Any) -> Index:
    indices = _toindices(patch)
    if not indices:
        raise GraphExecutionError("centerofmass() on empty patch")
    n = len(indices)
    return sum(i for i, _ in indices) // n, sum(j for _, j in indices) // n


def op_normalize(patch: Any) -> Any:
    if not patch:
        return patch
    return _shift(patch, (-_uppermost(patch), -_leftmost(patch)))


def op_backdrop(patch: Any) -> frozenset[Index]:
    return _backdrop(patch)


def op_delta(patch: Any) -> frozenset[Index]:
    if not patch:
        return frozenset()
    return _backdrop(patch) - _toindices(patch)


def op_inbox(patch: Any) -> frozenset[Index]:
    return _box_lines(
        _uppermost(patch) + 1,
        _leftmost(patch) + 1,
        _lowermost(patch) - 1,
        _rightmost(patch) - 1,
    )


def op_outbox(patch: Any) -> frozenset[Index]:
    return _box_lines(
        _uppermost(patch) - 1,
        _leftmost(patch) - 1,
        _lowermost(patch) + 1,
        _rightmost(patch) + 1,
    )


def op_box(patch: Any) -> frozenset[Index]:
    if not patch:
        return frozenset()
    ai, aj = op_ulcorner2(patch)
    bi, bj = op_lrcorner(patch)
    return _box_lines(ai, aj, bi, bj)


def op_corners(patch: Any) -> frozenset[Index]:
    return frozenset(
        {
            op_ulcorner2(patch),
            op_urcorner2(patch),
            op_llcorner(patch),
            op_lrcorner(patch),
        }
    )


def op_position(a: Any, b: Any) -> Index:
    ia, ja = op_center(_toindices(a))
    ib, jb = op_center(_toindices(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    if ja == jb:
        return (1 if ia < ib else -1, 0)
    if ia < ib:
        return (1, 1 if ja < jb else -1)
    return (-1, 1 if ja < jb else -1)


def op_vline(patch: Any) -> bool:
    h, w = _patch_dims(patch)
    return h == len(patch) and w == 1


def op_hline(patch: Any) -> bool:
    h, w = _patch_dims(patch)
    return w == len(patch) and h == 1


def op_vmatching(a: Any, b: Any) -> bool:
    cols_a = {j for _, j in _toindices(a)}
    cols_b = {j for _, j in _toindices(b)}
    return bool(cols_a & cols_b)


def op_hmatching(a: Any, b: Any) -> bool:
    rows_a = {i for i, _ in _toindices(a)}
    rows_b = {i for i, _ in _toindices(b)}
    return bool(rows_a & rows_b)


def op_manhattan(a: Any, b: Any) -> int:
    ia = _toindices(a)
    ib = _toindices(b)
    if not ia or not ib:
        raise GraphExecutionError("manhattan() on empty patch")
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in ia for bi, bj in ib)


def op_adjacent(a: Any, b: Any) -> bool:
    return op_manhattan(a, b) == 1


def op_bordering(patch: Any, grid: Any) -> bool:
    g = _grid(grid)
    return (
        _uppermost(patch) == 0
        or _leftmost(patch) == 0
        or _lowermost(patch) == len(g) - 1
        or _rightmost(patch) == len(g[0]) - 1
    )


def op_square(piece: Any) -> bool:
    if isinstance(piece, (list, tuple)) and (
        not piece or isinstance(piece[0], (list, tuple))
    ):
        grid = _grid(piece)
        return len(grid) == (len(grid[0]) if grid else 0)
    h, w = _patch_dims(piece)
    return h * w == len(piece) and h == w


def op_portrait(piece: Any) -> bool:
    h, w = _patch_dims(piece)
    return h > w


def op_shift(patch: Any, directions: Index) -> Any:
    _require_container(patch, "shift")
    return _shift(patch, directions)


def op_gravitate(source: Any, destination: Any) -> Index:
    si, sj = op_center(source)
    di, dj = op_center(destination)
    i, j = 0, 0
    if op_vmatching(source, destination):
        i = 1 if si < di else -1
    else:
        j = 1 if sj < dj else -1
    gi, gj = i, j
    moved = source
    c = 0
    while not op_adjacent(moved, destination) and c < 42:
        c += 1
        gi += i
        gj += j
        moved = _shift(moved, (i, j))
    return (gi - i, gj - j)


# ---------------------------------------------------------------------------
# Coordinates and lines
# ---------------------------------------------------------------------------

def op_dneighbors(loc: Index) -> frozenset[Index]:
    i, j = loc
    return frozenset({(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)})


def op_ineighbors(loc: Index) -> frozenset[Index]:
    i, j = loc
    return frozenset({(i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)})


def op_neighbors(loc: Index) -> frozenset[Index]:
    return op_dneighbors(loc) | op_ineighbors(loc)


def op_connect(a: Index, b: Index) -> frozenset[Index]:
    ai, aj = a
    bi, bj = b
    si, ei = min(ai, bi), max(ai, bi) + 1
    sj, ej = min(aj, bj), max(aj, bj) + 1
    if ai == bi:
        return frozenset((ai, j) for j in range(sj, ej))
    if aj == bj:
        return frozenset((i, aj) for i in range(si, ei))
    if bi - ai == bj - aj:
        return frozenset(zip(range(si, ei), range(sj, ej)))
    if bi - ai == aj - bj:
        return frozenset(zip(range(si, ei), range(ej - 1, sj - 1, -1)))
    return frozenset()


def op_shoot(start: Index, direction: Index) -> frozenset[Index]:
    return op_connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))


def op_hfrontier(location: Index) -> frozenset[Index]:
    return frozenset((location[0], j) for j in range(30))


def op_vfrontier(location: Index) -> frozenset[Index]:
    return frozenset((i, location[1]) for i in range(30))


def op_index(grid: Any, loc: Index) -> Any:
    g = _grid(grid)
    i, j = loc
    if not (0 <= i < len(g) and 0 <= j < len(g[0])):
        return None
    return g[i][j]


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------

def op_asobject(grid: Any) -> frozenset[Cell]:
    g = _grid(grid)
    return frozenset((v, (i, j)) for i, row in enumerate(g) for j, v in enumerate(row))


def op_toobject(patch: Any, grid: Any) -> frozenset[Cell]:
    g = _grid(grid)
    h, w = len(g), len(g[0])
    return frozenset(
        (g[i][j], (i, j)) for i, j in _toindices(patch) if 0 <= i < h and 0 <= j < w
    )


def op_objects(grid: Any, univalued: bool, diagonal: bool, without_bg: bool) -> frozenset:
    g = _grid(grid)
    bg = _mostcolor(g) if without_bg else None
    h, w = len(g), len(g[0])
    neighborhood = op_neighbors if diagonal else op_dneighbors
    occupied: set[Index] = set()
    objs: set[frozenset] = set()
    for loc in ((i, j) for i in range(h) for j in range(w)):
        if loc in occupied:
            continue
        val = g[loc[0]][loc[1]]
        if val == bg:
            continue
        obj: set[Cell] = {(val, loc)}
        cands = {loc}
        while cands:
            frontier: set[Index] = set()
            for cand in cands:
                v = g[cand[0]][cand[1]]
                if (val == v) if univalued else (v != bg):
                    obj.add((v, cand))
                    occupied.add(cand)
                    frontier |= {
                        (i, j)
                        for i, j in neighborhood(cand)
                        if 0 <= i < h and 0 <= j < w
                    }
            cands = frontier - occupied
        objs.add(frozenset(obj))
    return frozenset(objs)


def op_partition(grid: Any) -> frozenset:
    g = _grid(grid)
    palette = {v for row in g for v in row}
    return frozenset(
        frozenset((v, (i, j)) for i, row in enumerate(g) for j, v in enumerate(row) if v == color)
        for color in palette
    )


def op_fgpartition(grid: Any) -> frozenset:
    g = _grid(grid)
    bg = _mostcolor(g)
    palette = {v for row in g for v in row} - {bg}
    return frozenset(
        frozenset((v, (i, j)) for i, row in enumerate(g) for j, v in enumerate(row) if v == color)
        for color in palette
    )


def op_frontiers(grid: Any) -> frozenset:
    g = _grid(grid)
    h, w = len(g), len(g[0])
    row_indices = [i for i, row in enumerate(g) if len(set(row)) == 1]
    col_indices = [j for j in range(w) if len({g[i][j] for i in range(h)}) == 1]
    result: set[frozenset] = set()
    for i in row_indices:
        result.add(frozenset((g[i][j], (i, j)) for j in range(w)))
    for j in col_indices:
        result.add(frozenset((g[i][j], (i, j)) for i in range(h)))
    return frozenset(result)


def op_colorfilter(objs: Any, value: int) -> frozenset:
    return frozenset(obj for obj in objs if _ordered(obj)[0][0] == value)


def op_sizefilter(container: Any, n: int) -> frozenset:
    return frozenset(item for item in container if len(item) == n)


def op_color(obj: Any) -> int:
    if not obj:
        raise GraphExecutionError("color() on empty object")
    return _ordered(obj)[0][0]


def op_numcolors(element: Any) -> int:
    if isinstance(element, (list, tuple)) and (
        not element or isinstance(element[0], (list, tuple))
    ):
        return len({v for row in _grid(element) for v in row})
    return len({v for v, _ in element})


def op_colorcount(element: Any, value: int) -> int:
    if isinstance(element, (list, tuple)) and (
        not element or isinstance(element[0], (list, tuple))
    ):
        return sum(row.count(value) for row in _grid(element))
    return sum(1 for v, _ in element if v == value)


def op_recolor(value: int, patch: Any) -> frozenset[Cell]:
    return frozenset((value, index) for index in _toindices(patch))


def op_hperiod(obj: Any) -> int:
    normalized = op_normalize(obj)
    _, w = _patch_dims(normalized)
    for p in range(1, w):
        offsetted = _shift(normalized, (0, -p))
        pruned = {(c, (i, j)) for c, (i, j) in offsetted if j >= 0}
        if pruned <= set(normalized):
            return p
    return w


def op_vperiod(obj: Any) -> int:
    normalized = op_normalize(obj)
    h, _ = _patch_dims(normalized)
    for p in range(1, h):
        offsetted = _shift(normalized, (-p, 0))
        pruned = {(c, (i, j)) for c, (i, j) in offsetted if i >= 0}
        if pruned <= set(normalized):
            return p
    return h


# ---------------------------------------------------------------------------
# Grid transforms and drawing
# ---------------------------------------------------------------------------

def op_dmirror(piece: Any) -> Any:
    if isinstance(piece, (list, tuple)) and (
        not piece or isinstance(piece[0], (list, tuple))
    ):
        grid = _grid(piece)
        return [list(row) for row in zip(*grid)] if grid else []
    a, b = op_ulcorner2(piece)
    if _is_object(piece):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)


def _vmirror_any(piece: Any) -> Any:
    """dsl.py vmirror, grid and patch cases (used by cmirror too)."""

    if isinstance(piece, (list, tuple)) and (
        not piece or isinstance(piece[0], (list, tuple))
    ):
        return [list(reversed(row)) for row in _grid(piece)]
    d = op_ulcorner2(piece)[1] + op_lrcorner(piece)[1]
    if _is_object(piece):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)


def op_cmirror(piece: Any) -> Any:
    if isinstance(piece, (list, tuple)) and (
        not piece or isinstance(piece[0], (list, tuple))
    ):
        grid = _grid(piece)
        flipped = [list(reversed(row)) for row in reversed(grid)]
        return [list(row) for row in zip(*flipped)] if flipped else []
    return _vmirror_any(op_dmirror(_vmirror_any(piece)))


def op_trim(grid: Any) -> Grid:
    g = _grid(grid)
    return [list(row[1:-1]) for row in g[1:-1]]


def op_compress(grid: Any) -> Grid:
    g = _grid(grid)
    h, w = len(g), len(g[0])
    ri = {i for i, row in enumerate(g) if len(set(row)) == 1}
    ci = {j for j in range(w) if len({g[i][j] for i in range(h)}) == 1}
    return [
        [v for j, v in enumerate(row) if j not in ci]
        for i, row in enumerate(g)
        if i not in ri
    ]


def op_hsplit(grid: Any, n: int) -> tuple:
    g = _grid(grid)
    if not isinstance(n, int) or n <= 0:
        raise GraphExecutionError(f"hsplit expects int >= 1, got {n!r}")
    h, w = len(g), len(g[0]) // n
    offset = 1 if len(g[0]) % n != 0 else 0
    return tuple(
        [list(row[w * i + i * offset : w * i + i * offset + w]) for row in g]
        for i in range(n)
    )


def op_vsplit(grid: Any, n: int) -> tuple:
    g = _grid(grid)
    if not isinstance(n, int) or n <= 0:
        raise GraphExecutionError(f"vsplit expects int >= 1, got {n!r}")
    h, w = len(g) // n, len(g[0])
    offset = 1 if len(g) % n != 0 else 0
    return tuple(
        [list(row) for row in g[h * i + i * offset : h * i + i * offset + h]]
        for i in range(n)
    )


def op_cellwise(a: Any, b: Any, fallback: int) -> Grid:
    ga, gb = _grid(a), _grid(b)
    if len(ga) != len(gb) or (ga and gb and len(ga[0]) != len(gb[0])):
        raise GraphExecutionError("cellwise requires equal shapes")
    return [
        [va if va == vb else fallback for va, vb in zip(ra, rb)]
        for ra, rb in zip(ga, gb)
    ]


def op_paint(grid: Any, obj: Any) -> Grid:
    g = _grid(grid)
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            out[i][j] = value
    return out


def op_cover(grid: Any, patch: Any) -> Grid:
    g = _grid(grid)
    bg = _mostcolor(g)
    out = [list(row) for row in g]
    for i, j in _toindices(patch):
        if 0 <= i < len(out) and 0 <= j < len(out[0]):
            out[i][j] = bg
    return out


def op_underfill(grid: Any, value: int, patch: Any) -> Grid:
    g = _grid(grid)
    bg = _mostcolor(g)
    out = [list(row) for row in g]
    for i, j in _toindices(patch):
        if 0 <= i < len(out) and 0 <= j < len(out[0]) and out[i][j] == bg:
            out[i][j] = value
    return out


def op_underpaint(grid: Any, obj: Any) -> Grid:
    g = _grid(grid)
    bg = _mostcolor(g)
    out = [list(row) for row in g]
    for value, (i, j) in obj:
        if 0 <= i < len(out) and 0 <= j < len(out[0]) and out[i][j] == bg:
            out[i][j] = value
    return out


def op_move(grid: Any, obj: Any, offset: Index) -> Grid:
    return op_paint(op_cover(grid, obj), _shift(obj, offset))


def op_subgrid(patch: Any, grid: Any) -> Grid:
    g = _grid(grid)
    indices = _toindices(patch)
    if not indices:
        raise GraphExecutionError("subgrid() on empty patch")
    h, w = _patch_dims(patch)
    r0, c0 = op_ulcorner2(patch)
    return [list(row[c0 : c0 + w]) for row in g[r0 : r0 + h]]


def op_occurrences(grid: Any, obj: Any) -> frozenset[Index]:
    g = _grid(grid)
    h, w = len(g), len(g[0])
    oh, ow = _patch_dims(obj)
    normed = op_normalize(obj)
    occs: set[Index] = set()
    for i in range(h - oh + 1):
        for j in range(w - ow + 1):
            occurs = True
            for v, (a, b) in _shift(normed, (i, j)):
                if not (0 <= a < h and 0 <= b < w and g[a][b] == v):
                    occurs = False
                    break
            if occurs:
                occs.add((i, j))
    return frozenset(occs)


def op_upscale(element: Any, factor: int) -> Any:
    if not isinstance(factor, int) or factor <= 0:
        raise GraphExecutionError(f"upscale expects int >= 1, got {factor!r}")
    if isinstance(element, (list, tuple)) and (
        not element or isinstance(element[0], (list, tuple))
    ):
        grid = _grid(element)
        return [
            [v for v in row for _ in range(factor)]
            for row in grid
            for _ in range(factor)
        ]
    if not element:
        return frozenset()
    di_inv, dj_inv = op_ulcorner2(element)
    normed = _shift(element, (-di_inv, -dj_inv))
    out: set[Cell] = set()
    for value, (i, j) in normed:
        for io in range(factor):
            for jo in range(factor):
                out.add((value, (i * factor + io, j * factor + jo)))
    return _shift(frozenset(out), (di_inv, dj_inv))


def op_downscale(grid: Any, factor: int) -> Grid:
    if not isinstance(factor, int) or factor <= 0:
        raise GraphExecutionError(f"downscale expects int >= 1, got {factor!r}")
    g = _grid(grid)
    return [
        [g[i][j] for j in range(len(g[0])) if j % factor == 0]
        for i in range(len(g))
        if i % factor == 0
    ]


PRIMITIVE_EXECUTORS: dict[str, Callable[..., Any]] = {
    # scalar arithmetic and predicates
    "add": op_add,
    "multiply": op_multiply,
    "divide": op_divide,
    "halve": op_halve,
    "invert": op_invert,
    "sign": op_sign,
    "crement": op_crement,
    "even": op_even,
    "positive": op_positive,
    "flip": op_flip,
    "both": op_both,
    "either": op_either,
    "greater": op_greater,
    "equality": op_equality,
    "contained": op_contained,
    "toivec": op_toivec,
    "tojvec": op_tojvec,
    # generic containers
    "combine": op_combine,
    "intersection": op_intersection,
    "difference": op_difference,
    "insert": op_insert,
    "remove": op_remove,
    "other": op_other,
    "product": op_product,
    "pair": op_pair,
    "repeat": op_repeat,
    "dedupe": op_dedupe,
    "interval": op_interval,
    "minimum": op_minimum,
    "maximum": op_maximum,
    "mostcommon": op_mostcommon,
    "leastcommon": op_leastcommon,
    "order": op_order,
    "valmax": op_valmax,
    "valmin": op_valmin,
    "argmax": op_argmax,
    "argmin": op_argmin,
    "extract": op_extract,
    "sfilter": op_sfilter,
    "mfilter": op_mfilter,
    "apply": op_apply,
    "rapply": op_rapply,
    "mapply": op_mapply,
    "papply": op_papply,
    "mpapply": op_mpapply,
    "prapply": op_prapply,
    # higher-order constructors
    "compose": op_compose,
    "chain": op_chain,
    "fork": op_fork,
    "lbind": op_lbind,
    "rbind": op_rbind,
    "power": op_power,
    "matcher": op_matcher,
    "branch": op_branch,
    # patch geometry
    "toindices": op_toindices,
    "asindices": op_asindices,
    "llcorner": op_llcorner,
    "lrcorner": op_lrcorner,
    "uppermost": op_uppermost,
    "lowermost": op_lowermost,
    "leftmost": op_leftmost,
    "rightmost": op_rightmost,
    "center": op_center,
    "centerofmass": op_centerofmass,
    "normalize": op_normalize,
    "backdrop": op_backdrop,
    "delta": op_delta,
    "inbox": op_inbox,
    "outbox": op_outbox,
    "box": op_box,
    "corners": op_corners,
    "position": op_position,
    "vline": op_vline,
    "hline": op_hline,
    "vmatching": op_vmatching,
    "hmatching": op_hmatching,
    "manhattan": op_manhattan,
    "adjacent": op_adjacent,
    "bordering": op_bordering,
    "square": op_square,
    "portrait": op_portrait,
    "shift": op_shift,
    "gravitate": op_gravitate,
    # coordinates and lines
    "dneighbors": op_dneighbors,
    "ineighbors": op_ineighbors,
    "neighbors": op_neighbors,
    "connect": op_connect,
    "shoot": op_shoot,
    "hfrontier": op_hfrontier,
    "vfrontier": op_vfrontier,
    "index": op_index,
    # objects
    "asobject": op_asobject,
    "toobject": op_toobject,
    "objects": op_objects,
    "partition": op_partition,
    "fgpartition": op_fgpartition,
    "frontiers": op_frontiers,
    "colorfilter": op_colorfilter,
    "sizefilter": op_sizefilter,
    "color": op_color,
    "numcolors": op_numcolors,
    "colorcount": op_colorcount,
    "recolor": op_recolor,
    "hperiod": op_hperiod,
    "vperiod": op_vperiod,
    # grid transforms and drawing
    "dmirror": op_dmirror,
    "cmirror": op_cmirror,
    "trim": op_trim,
    "compress": op_compress,
    "hsplit": op_hsplit,
    "vsplit": op_vsplit,
    "cellwise": op_cellwise,
    "paint": op_paint,
    "cover": op_cover,
    "underfill": op_underfill,
    "underpaint": op_underpaint,
    "move": op_move,
    "subgrid": op_subgrid,
    "occurrences": op_occurrences,
    "upscale": op_upscale,
    "downscale": op_downscale,
}
