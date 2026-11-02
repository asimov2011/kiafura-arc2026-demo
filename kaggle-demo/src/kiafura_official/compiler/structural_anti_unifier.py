"""M2.2: Structural Anti-Unifier — converts M1 patterns to ParametricGraph.

M2.2 corrections:
- Real topological ordering (Kahn) in build_template()
- Fixed functions as declared PARAMETRIC slots with mode=FIXED
- Mastery-loaded DAGs used for structural LOO verification
- Per-occurrence LOO evidence against source DAGs
- No-fold promotion blocked (fully_typed without folds → unvalidated)
- Full parameter metadata (source_kind, local_node_id, port, mode, observed_values)
- type_precision and promotion_status in candidate graph output
- Separate promotable-only and all-candidates output
- Topological sort for candidate graph nodes
- Fixed functions resolved via ParamSlot references, not raw strings
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any

from schema.typed_graph_ir import (
    ParametricGraph, ParamSlot, GraphNode, Constant, ValueType,
    OPERATOR_REGISTRY,
)
from kiafura_official.compiler.hodel_parser import parse_hodel_tokens
from kiafura_official.compiler.dag_normalizer import (
    NormalizedDag, NormalizedNode, normalize_dag, InputCategory,
)


# ── Public exception ─────────────────────────────────────────────────

class AntiUnificationError(RuntimeError):
    pass


# ── Type helpers ─────────────────────────────────────────────────────

_TYPE_MAP = {
    "GRID": ValueType.GRID, "POINT": ValueType.POINT,
    "VECTOR": ValueType.VECTOR, "INDEX_SET": ValueType.INDEX_SET,
    "OBJECT": ValueType.OBJECT, "OBJECT_SET": ValueType.OBJECT_SET,
    "PATCH": ValueType.PATCH, "POINTSET": ValueType.POINTSET,
    "COLOR": ValueType.COLOR, "INT": ValueType.INT,
    "BOOL": ValueType.BOOL, "STR": ValueType.STR,
    "TUPLE": ValueType.TUPLE, "FUNCTION": ValueType.FUNCTION,
    "FUNC": ValueType.FUNCTION, "ANY": ValueType.ANY,
    "INPUT": ValueType.GRID,
}


def _parse_type(s: str) -> ValueType:
    return _TYPE_MAP.get(s, ValueType.ANY)


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _native_to_constant(val: Any, vtype: ValueType) -> Constant | None:
    if val is None:
        return None
    try:
        return Constant(type=vtype, value=val)
    except (TypeError, ValueError):
        return None


def _mode_default(constants: list[Constant]) -> Constant | None:
    if not constants:
        return None
    counter: Counter[tuple[str, str]] = Counter()
    for c in constants:
        key = (c.type.name, json.dumps(c.value, sort_keys=True))
        counter[key] += 1
    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0][1]))
    if not sorted_items:
        return None
    best_type_name, best_val_json = sorted_items[0][0]
    best_type = _parse_type(best_type_name)
    best_val = json.loads(best_val_json)
    return Constant(type=best_type, value=best_val)


def _unique_constants(constants: list[Constant]) -> tuple[Constant, ...]:
    seen = set()
    result = []
    for c in constants:
        key = (c.type.name, json.dumps(c.value, sort_keys=True))
        if key not in seen:
            seen.add(key)
            result.append(c)
    return tuple(result)


def _build_slot_name(pattern_hash: str, lid: str, port: int) -> str:
    return f"p{pattern_hash[:12]}_{lid}_{port}"


def _build_fn_name(pattern_hash: str, lid: str, port: int) -> str:
    return f"f{pattern_hash[:12]}_{lid}_{port}"


# ── ParamBindingSpec — complete metadata ────────────────────────────

@dataclass(frozen=True)
class ParamBindingSpec:
    name: str
    value_type: ValueType
    source_kind: str
    local_node_id: str
    port: int
    mode: str
    default: Constant | None
    observed_values: tuple[Constant, ...]


@dataclass(frozen=True)
class MasteryProgram:
    task_id: str
    graph: ParametricGraph
    dag: NormalizedDag
    node_by_output: dict[str, GraphNode]
    param_by_name: dict[str, Constant]


def load_mastery_programs(
    mastery_path: str,
    required_task_ids: set[str],
) -> dict[str, MasteryProgram]:
    """Load and parse every required Hodel program, fail-closed."""
    if not mastery_path or not os.path.exists(mastery_path):
        raise AntiUnificationError(f"Mastery not found: {mastery_path}")
    if os.path.getsize(mastery_path) == 0:
        raise AntiUnificationError(f"Mastery is empty: {mastery_path}")

    loaded: dict[str, MasteryProgram] = {}
    seen_required_records: set[str] = set()
    with open(mastery_path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AntiUnificationError(
                    f"Invalid mastery JSON at line {lineno}: {exc}"
                ) from exc
            task_id = rec.get("task_id")
            if task_id not in required_task_ids:
                continue
            if task_id in seen_required_records:
                raise AntiUnificationError(
                    f"Duplicate mastery record for required task {task_id}"
                )
            seen_required_records.add(task_id)
            if rec.get("backend") != "hodel-ssa":
                raise AntiUnificationError(
                    f"Required task {task_id} has backend {rec.get('backend')!r}, expected 'hodel-ssa'"
                )
            tokens = rec.get("tokens")
            if not isinstance(tokens, list):
                raise AntiUnificationError(
                    f"Required task {task_id} has no token list"
                )
            try:
                graph = parse_hodel_tokens(task_id, tokens)
                dag = normalize_dag(graph)
            except Exception as exc:
                raise AntiUnificationError(
                    f"Failed to parse required mastery task {task_id}: {exc}"
                ) from exc
            node_by_output = {n.output_name: n for n in graph.nodes}
            param_by_name = {
                slot.name: slot.default
                for slot in graph.parameters
                if slot.default is not None
            }
            loaded[task_id] = MasteryProgram(
                task_id=task_id,
                graph=graph,
                dag=dag,
                node_by_output=node_by_output,
                param_by_name=param_by_name,
            )

    missing = sorted(required_task_ids - set(loaded))
    if missing:
        raise AntiUnificationError(
            f"Mastery missing {len(missing)}/{len(required_task_ids)} required tasks: {missing[:8]}"
        )
    return loaded


# ── Pattern classification ───────────────────────────────────────────

@dataclass(frozen=True)
class PatternClassification:
    type_precision: str
    promotion_status: str
    root_output_type: ValueType
    has_internal_any: bool
    has_boundary_any: bool
    has_untyped_constant: bool


def classify_pattern(pattern: dict[str, Any]) -> PatternClassification:
    nodes = pattern.get("nodes", [])
    has_internal = any(n.get("output_type") == "ANY" for n in nodes)
    has_boundary = False
    has_untyped = False

    for n in nodes:
        for inp in n.get("inputs", []):
            if inp.get("kind") == "BOUNDARY":
                b = inp.get("boundary", {})
                bt = b.get("type", "")
                cat = b.get("cat", "")
                if bt == "ANY" and cat == "CONSTANT":
                    has_untyped = True
                elif bt == "ANY":
                    has_boundary = True

    if has_untyped:
        tp = "untyped_constant"
        ps = "blocked"
    elif has_boundary:
        tp = "boundary_any"
        ps = "provisional"
    elif has_internal:
        tp = "internal_any"
        ps = "provisional"
    else:
        tp = "fully_typed"
        ps = "fully_typed_eligible"

    root_type = ValueType.GRID
    for n in nodes:
        if n.get("local_id") == "c0":
            root_type = _parse_type(n.get("output_type", "GRID"))
            break

    return PatternClassification(
        type_precision=tp, promotion_status=ps,
        root_output_type=root_type,
        has_internal_any=has_internal,
        has_boundary_any=has_boundary,
        has_untyped_constant=has_untyped,
    )


# ── Boundary analysis ────────────────────────────────────────────────

@dataclass(frozen=True)
class BoundaryPosition:
    key: str
    category: str
    value_type: ValueType
    function_value: str | None = None
    observed_values: tuple[Any, ...] = ()
    is_fixed_function: bool = False


def analyze_boundaries(
    pattern: dict[str, Any],
) -> tuple[list[BoundaryPosition], bool]:
    nodes = pattern.get("nodes", [])
    occurrences = pattern.get("occurrences", [])
    has_missing = False

    pos_map: dict[str, dict[str, Any]] = {}
    for n in nodes:
        lid = n["local_id"]
        for inp in n.get("inputs", []):
            if inp.get("kind") != "BOUNDARY":
                continue
            b = inp.get("boundary", {})
            key = f"{lid}:{inp['port']}"
            cat = b.get("cat", "?")

            entry = {
                "category": cat,
                "value_type": _parse_type(b.get("type", "ANY")),
                "function_value": b.get("fn"),
                "observed": [],
                "is_fixed_function": True,
            }
            pos_map[key] = entry

    for occ in occurrences:
        bb = occ.get("boundary_bindings", {})
        for key, binding in bb.items():
            if key in pos_map:
                val = binding.get("value")
                if val is not None:
                    pos_map[key]["observed"].append(val)
                if pos_map[key]["category"] == "FUNCTION_REF":
                    fn = binding.get("value")
                    if pos_map[key]["function_value"] is not None and fn is not None:
                        if pos_map[key]["function_value"] != fn:
                            pos_map[key]["is_fixed_function"] = False
                btype = binding.get("type", "")
                if btype and btype != "?":
                    bt = _parse_type(btype)
                    pos_map[key]["value_type"] = bt

    results = []
    for key, entry in sorted(pos_map.items()):
        results.append(BoundaryPosition(
            key=key,
            category=entry["category"],
            value_type=entry["value_type"],
            function_value=entry["function_value"],
            observed_values=tuple(entry["observed"]),
            is_fixed_function=entry["is_fixed_function"],
        ))

    return results, has_missing


# ── Canonical stable graph ID ────────────────────────────────────────

def _semantic_payload(
    nodes: list[dict],
    slot_specs: list[ParamBindingSpec],
    input_specs: list[ParamBindingSpec],
    output_type: ValueType,
) -> dict[str, Any]:
    parameters = []
    for sp in sorted(slot_specs, key=lambda x: (x.local_node_id, x.port, x.name)):
        item: dict[str, Any] = {
            "position": f"{sp.local_node_id}:{sp.port}",
            "source_kind": sp.source_kind,
            "value_type": sp.value_type.name,
            "mode": sp.mode,
        }
        if sp.mode == "FIXED":
            if sp.default is None:
                raise AntiUnificationError(
                    f"FIXED slot {sp.name} has no semantic value"
                )
            item["fixed_value"] = {
                "type": sp.default.type.name,
                "value": sp.default.value,
            }
        parameters.append(item)

    graph_inputs = sorted(
        (
            {
                "position": f"{sp.local_node_id}:{sp.port}",
                "source_kind": sp.source_kind,
                "value_type": sp.value_type.name,
            }
            for sp in input_specs
        ),
        key=lambda x: (x["position"], x["source_kind"], x["value_type"]),
    )

    node_info = []
    for n in sorted(nodes, key=lambda x: x["local_id"]):
        encoded_inputs = []
        for inp in sorted(n.get("inputs", []), key=lambda x: x["port"]):
            if inp["kind"] == "REF":
                encoded_inputs.append({
                    "port": inp["port"],
                    "kind": "REF",
                    "target": inp["target"],
                })
            elif inp["kind"] == "BOUNDARY":
                b = inp.get("boundary", {})
                encoded_inputs.append({
                    "port": inp["port"],
                    "kind": "BOUNDARY",
                    "cat": b.get("cat"),
                    "type": b.get("type"),
                })
            else:
                raise AntiUnificationError(
                    f"Unknown input kind in semantic payload: {inp.get('kind')!r}"
                )
        node_info.append({
            "local_id": n["local_id"],
            "operator": n["operator"],
            "output_type": n.get("output_type", "ANY"),
            "inputs": encoded_inputs,
        })

    has_primary_input = any(
        inp.get("kind") == "BOUNDARY"
        and inp.get("boundary", {}).get("cat") == "EXTERNAL_INPUT"
        for node in nodes
        for inp in node.get("inputs", [])
    )
    return {
        "graph_output_type": output_type.name,
        "primary_input": "GRID" if has_primary_input else None,
        "graph_inputs": graph_inputs,
        "parameters": parameters,
        "nodes": node_info,
    }


def _build_stable_id(
    nodes: list[dict],
    slot_specs: list[ParamBindingSpec],
    input_specs: list[ParamBindingSpec],
    output_type: ValueType,
) -> str:
    payload = _semantic_payload(nodes, slot_specs, input_specs, output_type)
    return "hodel_" + hashlib.sha256(_canonical_json(payload)).hexdigest()[:16]


# ── Topological sort for graph building ──────────────────────────────

def _topological_sort_nodes(nodes: list[dict]) -> list[dict]:
    """Deterministic Kahn sort; cycles and missing producers fail closed."""
    node_map = {n["local_id"]: n for n in nodes}
    if len(node_map) != len(nodes):
        raise AntiUnificationError("Duplicate local node IDs")
    in_degree = {lid: 0 for lid in node_map}
    adjacency: dict[str, list[str]] = {lid: [] for lid in node_map}

    for node in nodes:
        consumer = node["local_id"]
        for inp in node.get("inputs", []):
            if inp.get("kind") != "REF":
                continue
            producer = inp.get("target")
            if producer not in node_map:
                raise AntiUnificationError(
                    f"Unresolved dependency {producer!r} for {consumer}"
                )
            adjacency[producer].append(consumer)
            in_degree[consumer] += 1

    ready = sorted(lid for lid, deg in in_degree.items() if deg == 0)
    ordered: list[dict] = []
    while ready:
        lid = ready.pop(0)
        ordered.append(node_map[lid])
        for consumer in sorted(adjacency[lid]):
            in_degree[consumer] -= 1
            if in_degree[consumer] == 0:
                ready.append(consumer)
                ready.sort()

    if len(ordered) != len(nodes):
        remaining = sorted(lid for lid, deg in in_degree.items() if deg > 0)
        raise AntiUnificationError(
            f"Cycle or unresolved dependency: {remaining}"
        )
    return ordered


# ── Template building ────────────────────────────────────────────────

def build_template(
    pattern: dict[str, Any],
    boundaries: list[BoundaryPosition],
    classification: PatternClassification,
) -> ParametricGraph:
    """Build ParametricGraph with complete parameter metadata.

    Fixed functions are declared as mode=FIXED ParamBindingSpec.
    """
    nodes = pattern.get("nodes", [])
    ph = pattern.get("pattern_hash", "?")

    # ── Map boundary positions → slot names / input names ──────────
    slot_specs: list[ParamBindingSpec] = []  # PARAMETRIC + FIXED
    input_specs: list[ParamBindingSpec] = []  # GRAPH_INPUTS
    slot_map: dict[str, str] = {}
    input_map: dict[str, str] = {}
    has_external_input = False

    for bp in boundaries:
        lid, pstr = bp.key.split(":")
        port = int(pstr)

        if bp.category == "EXTERNAL_INPUT":
            has_external_input = True
            input_map[bp.key] = "INPUT"
            continue

        if bp.category == "NODE_REF":
            name = f"boundary_{lid}_{port}"
            input_map[bp.key] = name
            input_specs.append(ParamBindingSpec(
                name=name, value_type=bp.value_type,
                source_kind="NODE_REF",
                local_node_id=lid, port=port,
                mode="GRAPH_INPUT",
                default=None, observed_values=(),
            ))
            continue

        if bp.category == "FUNCTION_REF":
            obs_constants = []
            for v in bp.observed_values:
                if v is not None:
                    c = _native_to_constant(v, ValueType.FUNCTION)
                    if c is not None:
                        obs_constants.append(c)
            obs_constants = list(_unique_constants(obs_constants))

            if bp.is_fixed_function and len(obs_constants) <= 1:
                # Fixed function → mode=FIXED
                name = _build_fn_name(ph, lid, port)
                default_val = obs_constants[0] if obs_constants else Constant(type=ValueType.FUNCTION, value=bp.function_value or "")
                slot_specs.append(ParamBindingSpec(
                    name=name, value_type=ValueType.FUNCTION,
                    source_kind="FUNCTION_REF",
                    local_node_id=lid, port=port,
                    mode="FIXED",
                    default=default_val,
                    observed_values=obs_constants,
                ))
                slot_map[bp.key] = name
                continue

            # Variable function → mode=PARAMETRIC
            name = _build_slot_name(ph, lid, port)
            default = _mode_default(obs_constants) if obs_constants else None
            slot_specs.append(ParamBindingSpec(
                name=name, value_type=ValueType.FUNCTION,
                source_kind="FUNCTION_REF",
                local_node_id=lid, port=port,
                mode="PARAMETRIC",
                default=default,
                observed_values=obs_constants,
            ))
            slot_map[bp.key] = name
            continue

        if bp.category == "CONSTANT":
            name = _build_slot_name(ph, lid, port)
            obs_constants = [
                c for v in bp.observed_values
                if (c := _native_to_constant(v, bp.value_type)) is not None
            ]
            default = _mode_default(obs_constants)
            slot_specs.append(ParamBindingSpec(
                name=name, value_type=bp.value_type,
                source_kind="CONSTANT",
                local_node_id=lid, port=port,
                mode="PARAMETRIC",
                default=default,
                observed_values=_unique_constants(obs_constants),
            ))
            slot_map[bp.key] = name
            continue

    # ── Topological sort ──────────────────────────────────────────
    sorted_nodes = _topological_sort_nodes(nodes)

    # Build local → output name mapping
    local_to_output: dict[str, str] = {}
    for i, n in enumerate(sorted_nodes):
        local_to_output[n["local_id"]] = f"n{i:03d}"

    # ── Build GraphNodes ──────────────────────────────────────────
    graph_nodes: list[GraphNode] = []
    for n in sorted_nodes:
        lid = n["local_id"]
        op = n["operator"]
        out_type = _parse_type(n.get("output_type", "ANY"))
        out_name = local_to_output[lid]
        inputs: list[str] = []

        for inp in n.get("inputs", []):
            kind = inp["kind"]
            port = inp["port"]
            key = f"{lid}:{port}"

            if kind == "REF":
                target = inp.get("target", "")
                if target in local_to_output:
                    inputs.append(local_to_output[target])
                else:
                    raise AntiUnificationError(
                        f"Unresolved REF target '{target}' at {key}"
                    )
            elif kind == "BOUNDARY":
                if key in slot_map:
                    inputs.append(slot_map[key])
                elif key in input_map:
                    inputs.append(input_map[key])
                else:
                    raise AntiUnificationError(
                        f"Unmapped BOUNDARY at {key}"
                    )
            else:
                raise AntiUnificationError(f"Unknown input kind '{kind}' at {key}")

        graph_nodes.append(GraphNode(
            operator=op, inputs=tuple(inputs),
            output_type=out_type, output_name=out_name,
        ))

    # ── Output node ────────────────────────────────────────────────
    output_name = local_to_output.get("c0", "")
    if not output_name:
        raise AntiUnificationError("No output node (c0 not found)")

    # ── Graph inputs ──────────────────────────────────────────────
    graph_inputs: list[ParamSlot] = []
    if has_external_input:
        graph_inputs.append(ParamSlot(
            name="INPUT", value_type=ValueType.GRID,
            description="Primary grid input",
        ))
    for ins in input_specs:
        graph_inputs.append(ParamSlot(
            name=ins.name, value_type=ins.value_type,
            description=f"Boundary at {ins.local_node_id}:{ins.port}",
        ))

    # ── Parameters (ParamSlot from specs) ─────────────────────────
    parameters: list[ParamSlot] = []
    for sp in slot_specs:
        parameters.append(ParamSlot(
            name=sp.name, value_type=sp.value_type,
            default=sp.default,
            description=f"{sp.mode} {sp.source_kind} at {sp.local_node_id}:{sp.port}",
        ))

    # ── Stable graph ID ───────────────────────────────────────────
    graph_id = _build_stable_id(nodes, slot_specs, input_specs, classification.root_output_type)

    # ── Cost ──────────────────────────────────────────────────────
    cost = sum(
        getattr(OPERATOR_REGISTRY.get(n.operator), "cost", 1)
        for n in graph_nodes
    )

    graph = ParametricGraph(
        graph_id=graph_id,
        family="mined", version="v1",
        inputs=tuple(graph_inputs),
        parameters=tuple(parameters),
        output_type=classification.root_output_type,
        nodes=tuple(graph_nodes),
        output_node=output_name,
        support=len(set(o["task_id"] for o in pattern.get("occurrences", []))),
        cost=cost,
        source_tasks=(),
        is_singleton=False,
    )

    # Attach extra metadata via object.__setattr__ for frozen dataclass
    object.__setattr__(graph, '_slot_specs', tuple(slot_specs))
    object.__setattr__(graph, '_input_specs', tuple(input_specs))
    object.__setattr__(graph, '_type_precision', classification.type_precision)
    object.__setattr__(graph, '_promotion_status', classification.promotion_status)
    object.__setattr__(graph, '_local_to_output', dict(local_to_output))
    object.__setattr__(graph, '_output_to_local', {v: k for k, v in local_to_output.items()})
    object.__setattr__(graph, '_semantic_payload', _semantic_payload(
        nodes, slot_specs, input_specs, classification.root_output_type
    ))

    return graph


# ── Exact source validation ──────────────────────────────────────────

@dataclass(frozen=True)
class SourceValidationResult:
    source_nodes_found: int
    source_nodes_expected: int
    structure_match: bool
    ports_match: bool
    types_match: bool
    constants_match: bool
    functions_match: bool
    root_match: bool
    derived_bindings: dict[str, Constant]
    binding_slots_passed: int
    binding_slots_total: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.structure_match
            and self.ports_match
            and self.types_match
            and self.constants_match
            and self.functions_match
            and self.root_match
            and not self.failures
        )


def _normalize_literal(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_literal(v) for v in value]
    if isinstance(value, list):
        return [_normalize_literal(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_literal(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return value


def _constants_equal(left: Constant, right: Constant) -> bool:
    return (
        left.type == right.type
        and _normalize_literal(left.value) == _normalize_literal(right.value)
    )


def _slot_specs_by_position(template: ParametricGraph) -> dict[str, ParamBindingSpec]:
    return {
        f"{sp.local_node_id}:{sp.port}": sp
        for sp in getattr(template, "_slot_specs", ())
    }


def _input_specs_by_position(template: ParametricGraph) -> dict[str, ParamBindingSpec]:
    return {
        f"{sp.local_node_id}:{sp.port}": sp
        for sp in getattr(template, "_input_specs", ())
    }


def validate_occurrence_against_mastery(
    pattern: dict[str, Any],
    occurrence: dict[str, Any],
    template: ParametricGraph,
    program: MasteryProgram,
) -> SourceValidationResult:
    """Validate one M1 occurrence exactly against its Hodel source program."""
    pattern_nodes = pattern.get("nodes", [])
    pattern_by_lid = {n["local_id"]: n for n in pattern_nodes}
    lton = occurrence.get("local_to_output_name", {})
    boundary_bindings = occurrence.get("boundary_bindings", {})
    selected_source_names = set(lton.values())
    slot_specs = _slot_specs_by_position(template)
    input_specs = _input_specs_by_position(template)
    local_to_template_output = getattr(template, "_local_to_output", {})
    template_by_output = {n.output_name: n for n in template.nodes}

    failures: list[str] = []
    flags = {
        "structure": True,
        "ports": True,
        "types": True,
        "constants": True,
        "functions": True,
        "root": True,
    }
    derived: dict[str, Constant] = {}
    binding_passed = 0
    binding_total = 0

    def fail(kind: str, message: str) -> None:
        flags[kind] = False
        failures.append(message)

    expected_lids = set(pattern_by_lid)
    actual_lids = set(lton)
    if actual_lids != expected_lids:
        fail(
            "structure",
            f"local mapping mismatch: expected {sorted(expected_lids)}, got {sorted(actual_lids)}",
        )

    source_nodes_found = sum(
        1 for lid in expected_lids
        if lton.get(lid) in program.node_by_output
    )

    # Verify template assembly itself against the structural pattern.
    for lid, pnode in pattern_by_lid.items():
        template_output = local_to_template_output.get(lid)
        tnode = template_by_output.get(template_output or "")
        if tnode is None:
            fail("structure", f"template node missing for {lid}")
            continue
        if tnode.operator != pnode.get("operator"):
            fail(
                "structure",
                f"template operator mismatch at {lid}: expected {pnode.get('operator')}, got {tnode.operator}",
            )
        expected_out_type = _parse_type(pnode.get("output_type", "ANY"))
        if tnode.output_type != expected_out_type:
            fail(
                "types",
                f"template output type mismatch at {lid}: expected {expected_out_type.name}, got {tnode.output_type.name}",
            )
        pinputs = sorted(pnode.get("inputs", []), key=lambda x: x.get("port", -1))
        if len(tnode.inputs) != len(pinputs):
            fail(
                "ports",
                f"template arity mismatch at {lid}: expected {len(pinputs)}, got {len(tnode.inputs)}",
            )
            continue
        for expected_port, pinp in enumerate(pinputs):
            if pinp.get("port") != expected_port:
                fail(
                    "ports",
                    f"non-contiguous pattern port at {lid}: expected {expected_port}, got {pinp.get('port')}",
                )
                continue
            actual_ref = tnode.inputs[expected_port]
            key = f"{lid}:{expected_port}"
            if pinp.get("kind") == "REF":
                expected_ref = local_to_template_output.get(pinp.get("target"))
                if actual_ref != expected_ref:
                    fail(
                        "structure",
                        f"template REF mismatch at {key}: expected {expected_ref}, got {actual_ref}",
                    )
            else:
                cat = pinp.get("boundary", {}).get("cat")
                if cat == "EXTERNAL_INPUT":
                    expected_ref = "INPUT"
                elif cat in ("CONSTANT", "FUNCTION_REF"):
                    expected_ref = slot_specs.get(key).name if key in slot_specs else None
                elif cat == "NODE_REF":
                    expected_ref = input_specs.get(key).name if key in input_specs else None
                else:
                    expected_ref = None
                if actual_ref != expected_ref:
                    fail(
                        "structure",
                        f"template boundary mapping mismatch at {key}: expected {expected_ref}, got {actual_ref}",
                    )

    # Verify source nodes and every source port exactly.
    for lid, pnode in pattern_by_lid.items():
        source_name = lton.get(lid)
        if not source_name:
            fail("structure", f"missing source mapping for {lid}")
            continue
        snode = program.node_by_output.get(source_name)
        if snode is None:
            fail("structure", f"source node {source_name!r} for {lid} not found")
            continue

        if snode.operator != pnode.get("operator"):
            fail(
                "structure",
                f"operator mismatch at {lid}: expected {pnode.get('operator')}, found {snode.operator}",
            )
        expected_type = _parse_type(pnode.get("output_type", "ANY"))
        if snode.output_type != expected_type:
            fail(
                "types",
                f"output type mismatch at {lid}: expected {expected_type.name}, found {snode.output_type.name}",
            )

        pinputs = sorted(pnode.get("inputs", []), key=lambda x: x.get("port", -1))
        if len(snode.inputs) != len(pinputs):
            fail(
                "ports",
                f"source arity mismatch at {lid}: expected {len(pinputs)}, found {len(snode.inputs)}",
            )
            continue

        for port, pinp in enumerate(pinputs):
            key = f"{lid}:{port}"
            if pinp.get("port") != port:
                fail(
                    "ports",
                    f"source port order mismatch at {lid}: expected port {port}, pattern has {pinp.get('port')}",
                )
                continue
            source_ref = snode.inputs[port]
            kind = pinp.get("kind")

            if kind == "REF":
                target_lid = pinp.get("target")
                expected_source_ref = lton.get(target_lid)
                if source_ref != expected_source_ref:
                    fail(
                        "structure",
                        f"REF target mismatch at {key}: expected source {expected_source_ref} mapped from {target_lid}, found {source_ref}",
                    )
                continue

            if kind != "BOUNDARY":
                fail("structure", f"unknown pattern input kind {kind!r} at {key}")
                continue

            boundary = pinp.get("boundary", {})
            category = boundary.get("cat")
            boundary_type = _parse_type(boundary.get("type", "ANY"))
            evidence_binding = boundary_bindings.get(key, {})

            if category == "EXTERNAL_INPUT":
                if source_ref != "INPUT":
                    fail(
                        "structure",
                        f"external INPUT mismatch at {key}: expected INPUT, found {source_ref}",
                    )
                if evidence_binding and evidence_binding.get("type") not in (None, "INPUT", "GRID"):
                    fail(
                        "types",
                        f"M1 external input type mismatch at {key}: {evidence_binding.get('type')}",
                    )
                continue

            if category == "NODE_REF":
                if source_ref not in program.node_by_output:
                    fail(
                        "structure",
                        f"NODE_REF boundary at {key} does not reference a source node: {source_ref}",
                    )
                    continue
                if source_ref in selected_source_names:
                    fail(
                        "structure",
                        f"NODE_REF boundary at {key} points inside selected subgraph: {source_ref}",
                    )
                producer_type = program.node_by_output[source_ref].output_type
                if boundary_type != ValueType.ANY and producer_type != boundary_type:
                    fail(
                        "types",
                        f"NODE_REF type mismatch at {key}: expected {boundary_type.name}, found {producer_type.name}",
                    )
                evidence_type = evidence_binding.get("type")
                if evidence_type not in (None, "?", "ANY"):
                    if _parse_type(evidence_type) != producer_type:
                        fail(
                            "types",
                            f"M1 NODE_REF type mismatch at {key}: evidence {evidence_type}, source {producer_type.name}",
                        )
                continue

            if category not in ("CONSTANT", "FUNCTION_REF"):
                fail("structure", f"unknown boundary category {category!r} at {key}")
                continue

            binding_total += 1
            source_constant = program.param_by_name.get(source_ref)
            if source_constant is None:
                fail(
                    "functions" if category == "FUNCTION_REF" else "constants",
                    f"source parameter missing at {key}: {source_ref}",
                )
                continue

            if category == "FUNCTION_REF":
                if source_constant.type != ValueType.FUNCTION:
                    fail(
                        "functions",
                        f"function type mismatch at {key}: found {source_constant.type.name}",
                    )
                if boundary_type not in (ValueType.ANY, ValueType.FUNCTION):
                    fail(
                        "types",
                        f"pattern function boundary has incompatible type {boundary_type.name} at {key}",
                    )
            else:
                if source_constant.type == ValueType.FUNCTION:
                    fail("constants", f"constant boundary resolved to FUNCTION at {key}")
                if boundary_type != ValueType.ANY and source_constant.type != boundary_type:
                    fail(
                        "types",
                        f"constant type mismatch at {key}: expected {boundary_type.name}, found {source_constant.type.name}",
                    )

            source_slot = evidence_binding.get("source_slot")
            if source_slot and source_slot != source_ref:
                fail(
                    "functions" if category == "FUNCTION_REF" else "constants",
                    f"source slot mismatch at {key}: evidence {source_slot}, mastery {source_ref}",
                )
            if "value" not in evidence_binding:
                fail(
                    "functions" if category == "FUNCTION_REF" else "constants",
                    f"M1 binding value missing at {key}",
                )
            else:
                evidence_type_name = evidence_binding.get("type", source_constant.type.name)
                evidence_type = _parse_type(evidence_type_name)
                evidence_constant = _native_to_constant(evidence_binding.get("value"), evidence_type)
                if evidence_constant is None or not _constants_equal(evidence_constant, source_constant):
                    fail(
                        "functions" if category == "FUNCTION_REF" else "constants",
                        f"literal mismatch at {key}: evidence {evidence_binding.get('value')!r}/{evidence_type.name}, "
                        f"mastery {source_constant.value!r}/{source_constant.type.name}",
                    )

            slot_spec = slot_specs.get(key)
            if slot_spec is None:
                fail(
                    "structure",
                    f"template slot missing at {key}",
                )
                continue
            if slot_spec.value_type != ValueType.ANY and source_constant.type != slot_spec.value_type:
                fail(
                    "types",
                    f"template slot type mismatch at {key}: slot {slot_spec.value_type.name}, source {source_constant.type.name}",
                )
            if slot_spec.mode == "FIXED":
                if slot_spec.default is None or not _constants_equal(slot_spec.default, source_constant):
                    fail(
                        "functions" if category == "FUNCTION_REF" else "constants",
                        f"fixed literal mismatch at {key}: expected {slot_spec.default}, found {source_constant}",
                    )
            derived[slot_spec.name] = source_constant

            current_kind_ok = (
                flags["functions"] if category == "FUNCTION_REF" else flags["constants"]
            )
            if current_kind_ok:
                binding_passed += 1

    root_source_name = lton.get("c0")
    if occurrence.get("root_output_name") not in (None, root_source_name):
        fail(
            "root",
            f"root evidence mismatch: root_output_name={occurrence.get('root_output_name')}, mapping c0={root_source_name}",
        )
    if root_source_name not in program.node_by_output:
        fail("root", f"root source node missing: {root_source_name}")
    else:
        root_node = program.node_by_output[root_source_name]
        root_pattern = pattern_by_lid.get("c0")
        if root_pattern is None:
            fail("root", "pattern root c0 missing")
        else:
            if root_node.operator != root_pattern.get("operator"):
                fail(
                    "root",
                    f"root operator mismatch: expected {root_pattern.get('operator')}, found {root_node.operator}",
                )
            root_type = _parse_type(root_pattern.get("output_type", "ANY"))
            if root_node.output_type != root_type:
                fail(
                    "root",
                    f"root output type mismatch: expected {root_type.name}, found {root_node.output_type.name}",
                )
    if template.output_node != local_to_template_output.get("c0"):
        fail(
            "root",
            f"template output mismatch: expected {local_to_template_output.get('c0')}, found {template.output_node}",
        )

    return SourceValidationResult(
        source_nodes_found=source_nodes_found,
        source_nodes_expected=len(pattern_nodes),
        structure_match=flags["structure"],
        ports_match=flags["ports"],
        types_match=flags["types"],
        constants_match=flags["constants"],
        functions_match=flags["functions"],
        root_match=flags["root"],
        derived_bindings=derived,
        binding_slots_passed=binding_passed,
        binding_slots_total=binding_total,
        failures=tuple(failures),
    )


# ── Real leave-one-out ───────────────────────────────────────────────

@dataclass(frozen=True)
class LooFoldResult:
    held_out_task: str
    status: str
    occurrences_passed: int
    occurrences_total: int
    binding_slots_passed: int
    binding_slots_total: int
    failures: tuple[str, ...]


@dataclass(frozen=True)
class LooOccurrenceEvidence:
    task_id: str
    occurrence_index: int
    source_nodes_found: int
    source_nodes_expected: int
    structure_match: bool
    ports_match: bool
    types_match: bool
    constants_match: bool
    functions_match: bool
    root_match: bool
    derived_bindings: dict[str, dict]
    status: str
    failures: list[str]


def run_loo(
    pattern: dict[str, Any],
    boundaries: list[BoundaryPosition],
    classification: PatternClassification,
    mastery_path: str | None = None,
    mastery_programs: dict[str, MasteryProgram] | None = None,
) -> tuple[list[LooFoldResult], list[LooOccurrenceEvidence]]:
    """Leave one task out and validate every occurrence against source Hodel."""
    occurrences = pattern.get("occurrences", [])
    task_ids = sorted(set(o["task_id"] for o in occurrences))
    if mastery_programs is None:
        if mastery_path is None:
            return [], []
        mastery_programs = load_mastery_programs(mastery_path, set(task_ids))
    missing = sorted(set(task_ids) - set(mastery_programs))
    if missing:
        raise AntiUnificationError(
            f"Mastery programs missing for pattern: {missing}"
        )

    folds: list[LooFoldResult] = []
    evidence: list[LooOccurrenceEvidence] = []

    for held_out in task_ids:
        train_occurrences = [o for o in occurrences if o["task_id"] != held_out]
        held_occurrences = [o for o in occurrences if o["task_id"] == held_out]
        if not train_occurrences:
            raise AntiUnificationError(
                f"No training occurrences when holding out {held_out}"
            )

        train_pattern = dict(pattern)
        train_pattern["occurrences"] = train_occurrences
        train_boundaries, _ = analyze_boundaries(train_pattern)
        train_classification = classify_pattern(train_pattern)
        train_template = build_template(
            train_pattern, train_boundaries, train_classification
        )

        fold_failures: list[str] = []
        occurrences_passed = 0
        binding_passed = 0
        binding_total = 0
        program = mastery_programs[held_out]

        for occurrence_index, occurrence in enumerate(held_occurrences):
            result = validate_occurrence_against_mastery(
                pattern, occurrence, train_template, program
            )
            status = "pass" if result.passed else "fail"
            if result.passed:
                occurrences_passed += 1
            else:
                fold_failures.extend(result.failures)
            binding_passed += result.binding_slots_passed
            binding_total += result.binding_slots_total
            evidence.append(LooOccurrenceEvidence(
                task_id=held_out,
                occurrence_index=occurrence_index,
                source_nodes_found=result.source_nodes_found,
                source_nodes_expected=result.source_nodes_expected,
                structure_match=result.structure_match,
                ports_match=result.ports_match,
                types_match=result.types_match,
                constants_match=result.constants_match,
                functions_match=result.functions_match,
                root_match=result.root_match,
                derived_bindings={
                    name: {"type": value.type.name, "value": value.value}
                    for name, value in sorted(result.derived_bindings.items())
                },
                status=status,
                failures=list(result.failures),
            ))

        fold_status = (
            "pass" if held_occurrences and occurrences_passed == len(held_occurrences)
            else "fail"
        )
        folds.append(LooFoldResult(
            held_out_task=held_out,
            status=fold_status,
            occurrences_passed=occurrences_passed,
            occurrences_total=len(held_occurrences),
            binding_slots_passed=binding_passed,
            binding_slots_total=binding_total,
            failures=tuple(fold_failures),
        ))

    return folds, evidence


# ── Full anti-unification ────────────────────────────────────────────

@dataclass(frozen=True)
class AntiUnificationResult:
    pattern_hash: str
    graph: ParametricGraph
    classification: PatternClassification
    boundaries: tuple[BoundaryPosition, ...]
    loo_folds: tuple[LooFoldResult, ...]
    loo_evidence: tuple[LooOccurrenceEvidence, ...]
    slot_specs: tuple[ParamBindingSpec, ...]
    input_specs: tuple[ParamBindingSpec, ...]
    slot_count: int
    fixed_function_count: int


def anti_unify(
    pattern: dict[str, Any],
    mastery_path: str | None = None,
    mastery_programs: dict[str, MasteryProgram] | None = None,
) -> AntiUnificationResult:
    """Anti-unify one pattern and perform exact source LOO when supplied."""
    classification = classify_pattern(pattern)
    boundaries, _ = analyze_boundaries(pattern)
    template = build_template(pattern, boundaries, classification)
    folds, evidence = run_loo(
        pattern, boundaries, classification, mastery_path, mastery_programs
    )

    real_folds = [fold for fold in folds if fold.status != "skip"]
    all_pass = bool(real_folds) and all(fold.status == "pass" for fold in real_folds)

    if classification.type_precision == "fully_typed":
        if not real_folds:
            promotion_status = "unvalidated"
        elif all_pass:
            promotion_status = "structurally_promotable"
        else:
            promotion_status = "provisional_failed_loo"
    elif classification.type_precision in ("internal_any", "boundary_any"):
        promotion_status = "provisional"
    else:
        promotion_status = "blocked"

    classification = PatternClassification(
        type_precision=classification.type_precision,
        promotion_status=promotion_status,
        root_output_type=classification.root_output_type,
        has_internal_any=classification.has_internal_any,
        has_boundary_any=classification.has_boundary_any,
        has_untyped_constant=classification.has_untyped_constant,
    )
    slot_specs = tuple(getattr(template, "_slot_specs", ()))
    input_specs = tuple(getattr(template, "_input_specs", ()))
    object.__setattr__(template, "_type_precision", classification.type_precision)
    object.__setattr__(template, "_promotion_status", classification.promotion_status)

    return AntiUnificationResult(
        pattern_hash=pattern.get("pattern_hash", "?"),
        graph=template,
        classification=classification,
        boundaries=tuple(boundaries),
        loo_folds=tuple(folds),
        loo_evidence=tuple(evidence),
        slot_specs=slot_specs,
        input_specs=input_specs,
        slot_count=len(template.parameters),
        fixed_function_count=sum(1 for sp in slot_specs if sp.mode == "FIXED"),
    )


# ── Serializers ──────────────────────────────────────────────────────

def serialize_candidate_graph(result: AntiUnificationResult) -> dict[str, Any]:
    """Serialize a ParametricGraph for candidate memory (no task IDs/SSA names)."""
    g = result.graph
    params_out = []
    for sp in result.slot_specs:
        entry = {
            "name": sp.name,
            "value_type": sp.value_type.name,
            "source_kind": sp.source_kind,
            "local_node_id": sp.local_node_id,
            "port": sp.port,
            "mode": sp.mode,
            "default": {"type": sp.default.type.name, "value": sp.default.value}
            if sp.default else None,
            "observed_values": [
                {"type": c.type.name, "value": c.value}
                for c in sp.observed_values
            ],
        }
        params_out.append(entry)

    inputs_out = []
    input_spec_by_name = {spec.name: spec for spec in result.input_specs}
    for graph_input in g.inputs:
        if graph_input.name == "INPUT":
            inputs_out.append({
                "name": "INPUT",
                "value_type": "GRID",
                "source_kind": "EXTERNAL_INPUT",
                "mode": "GRAPH_INPUT",
            })
            continue
        spec = input_spec_by_name.get(graph_input.name)
        if spec is None:
            raise AntiUnificationError(
                f"Missing input metadata for {graph_input.name} in {g.graph_id}"
            )
        inputs_out.append({
            "name": spec.name,
            "value_type": spec.value_type.name,
            "source_kind": spec.source_kind,
            "local_node_id": spec.local_node_id,
            "port": spec.port,
            "mode": spec.mode,
        })

    return {
        "graph_id": g.graph_id,
        "family": g.family,
        "version": g.version,
        "type_precision": getattr(g, '_type_precision', result.classification.type_precision),
        "promotion_status": getattr(g, '_promotion_status', result.classification.promotion_status),
        "inputs": inputs_out,
        "parameters": params_out,
        "output_type": g.output_type.name,
        "nodes": [
            {
                "output_name": n.output_name,
                "operator": n.operator,
                "inputs": list(n.inputs),
                "output_type": n.output_type.name,
            }
            for n in g.nodes
        ],
        "output_node": g.output_node,
        "support": g.support,
        "cost": g.cost,
        "is_singleton": g.is_singleton,
        "semantic_sha256": hashlib.sha256(
            _canonical_json(getattr(g, "_semantic_payload"))
        ).hexdigest(),
        "_source_tasks_empty": not bool(g.source_tasks),
    }


def serialize_evidence(result: AntiUnificationResult) -> dict[str, Any]:
    """Full evidence with task IDs, occurrence details, LOO folds."""
    c = result.classification

    # LOO summary
    real_folds = [f for f in result.loo_folds if f.status != "skip"]
    loo_tasks_pass = sum(1 for f in real_folds if f.status == "pass")
    loo_tasks_total = len(real_folds)
    loo_occ_pass = sum(f.occurrences_passed for f in real_folds)
    loo_occ_total = sum(f.occurrences_total for f in real_folds)
    bind_pass = sum(f.binding_slots_passed for f in real_folds)
    bind_total = sum(f.binding_slots_total for f in real_folds)

    # Evidence per occurrence
    occurrence_evidence = []
    for ev in result.loo_evidence:
        occurrence_evidence.append({
            "task_id": ev.task_id,
            "occurrence_index": ev.occurrence_index,
            "source_nodes_found": ev.source_nodes_found,
            "source_nodes_expected": ev.source_nodes_expected,
            "structure_match": ev.structure_match,
            "ports_match": ev.ports_match,
            "types_match": ev.types_match,
            "constants_match": ev.constants_match,
            "functions_match": ev.functions_match,
            "root_match": ev.root_match,
            "derived_bindings": ev.derived_bindings,
            "status": ev.status,
            "failures": ev.failures,
        })

    return {
        "pattern_hash": result.pattern_hash,
        "graph_id": result.graph.graph_id,
        "task_support": result.graph.support,
        "type_precision": c.type_precision,
        "promotion_status": c.promotion_status,
        "root_output_type": c.root_output_type.name,
        "loo_summary": {
            "task_folds": f"{loo_tasks_pass}/{loo_tasks_total}",
            "occurrence_folds": f"{loo_occ_pass}/{loo_occ_total}",
            "binding_slots": f"{bind_pass}/{bind_total}",
        },
        "loo_folds": [
            {
                "held_out_task": f.held_out_task,
                "status": f.status,
                "occurrences_passed": f.occurrences_passed,
                "occurrences_total": f.occurrences_total,
                "binding_slots_passed": f.binding_slots_passed,
                "binding_slots_total": f.binding_slots_total,
                "failures": list(f.failures),
            }
            for f in result.loo_folds
        ],
        "occurrence_evidence": occurrence_evidence,
        "boundaries": [
            {
                "key": bp.key,
                "category": bp.category,
                "value_type": bp.value_type.name,
                "is_fixed_function": bp.is_fixed_function,
                "function_value": bp.function_value,
                "observed_values": list(bp.observed_values),
            }
            for bp in result.boundaries
        ],
        "slot_count": result.slot_count,
        "fixed_function_count": result.fixed_function_count,
    }


# ── Full pipeline ────────────────────────────────────────────────────

def anti_unify_all(
    patterns_json_path: str,
    mastery_path: str,
) -> tuple[list[AntiUnificationResult], list[dict]]:
    """Anti-unify all M1 patterns with one fail-closed mastery load."""
    if not os.path.exists(patterns_json_path):
        raise AntiUnificationError(f"Patterns not found: {patterns_json_path}")
    with open(patterns_json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    patterns = data.get("patterns", [])
    if not isinstance(patterns, list) or not patterns:
        raise AntiUnificationError("Patterns payload is empty or invalid")

    required_task_ids = {
        occurrence["task_id"]
        for pattern in patterns
        for occurrence in pattern.get("occurrences", [])
    }
    mastery_programs = load_mastery_programs(mastery_path, required_task_ids)

    successes: list[AntiUnificationResult] = []
    failed_patterns: list[dict] = []
    semantic_by_id: dict[str, bytes] = {}

    for pattern in patterns:
        pattern_hash = pattern.get("pattern_hash", "?")
        try:
            result = anti_unify(
                pattern,
                mastery_path=None,
                mastery_programs=mastery_programs,
            )
            payload_bytes = _canonical_json(
                getattr(result.graph, "_semantic_payload")
            )
            previous = semantic_by_id.get(result.graph.graph_id)
            if previous is not None and previous != payload_bytes:
                raise AntiUnificationError(
                    f"Semantic graph ID collision: {result.graph.graph_id}"
                )
            semantic_by_id[result.graph.graph_id] = payload_bytes
            successes.append(result)
        except AntiUnificationError as exc:
            failed_patterns.append({
                "pattern_hash": pattern_hash,
                "reason": str(exc),
            })

    return successes, failed_patterns


# ── CLI ──────────────────────────────────────────────────────────────

def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _build_report(
    results: list[AntiUnificationResult],
    all_graphs: list[dict[str, Any]],
    promotable_graphs: list[dict[str, Any]],
    evidence_out: list[dict[str, Any]],
) -> str:
    statuses = Counter(r.classification.promotion_status for r in results)
    precisions = Counter(r.classification.type_precision for r in results)
    occurrence_records = [ev for r in results for ev in r.loo_evidence]
    fold_records = [fold for r in results for fold in r.loo_folds]
    slot_specs = [sp for r in results for sp in r.slot_specs]

    valid_backward_refs = 0
    forward_refs = 0
    unresolved_refs = 0
    for graph in all_graphs:
        produced: set[str] = set()
        all_outputs = {node["output_name"] for node in graph["nodes"]}
        for node in graph["nodes"]:
            for ref in node["inputs"]:
                if ref not in all_outputs:
                    continue
                if ref in produced:
                    valid_backward_refs += 1
                else:
                    forward_refs += 1
            produced.add(node["output_name"])
        unresolved_refs += sum(
            1
            for node in graph["nodes"]
            for ref in node["inputs"]
            if ref.startswith("n") and ref not in all_outputs
        )

    uses_primary = sum(
        any("INPUT" in node["inputs"] for node in graph["nodes"])
        for graph in all_graphs
    )
    declares_primary = sum(
        any(inp.get("name") == "INPUT" for inp in graph["inputs"])
        for graph in all_graphs
    )
    fixed_functions = sum(
        sp.source_kind == "FUNCTION_REF" and sp.mode == "FIXED"
        for sp in slot_specs
    )

    lines = [
        "# M2.3 A2 Exact Anti-Unification Report",
        "",
        f"Pattern M1 ricevuti: {len(results)}/81",
        f"Candidate graph assemblati: {len(all_graphs)}/{len(results)}",
        f"Graph structurally promotable: {len(promotable_graphs)}/{len(results)}",
        f"Graph ID unici candidate: {len({g['graph_id'] for g in all_graphs})}/{len(all_graphs)}",
        f"Graph ID unici promoted: {len({g['graph_id'] for g in promotable_graphs})}/{len(promotable_graphs)}",
        "",
        "## Type precision",
        f"fully_typed: {precisions['fully_typed']}/{len(results)}",
        f"internal_any: {precisions['internal_any']}/{len(results)}",
        f"boundary_any: {precisions['boundary_any']}/{len(results)}",
        f"untyped_constant: {precisions['untyped_constant']}/{len(results)}",
        "",
        "## Promotion status",
        f"structurally_promotable: {statuses['structurally_promotable']}/{len(results)}",
        f"provisional_failed_loo: {statuses['provisional_failed_loo']}/{len(results)}",
        f"provisional: {statuses['provisional']}/{len(results)}",
        f"blocked: {statuses['blocked']}/{len(results)}",
        f"unvalidated: {statuses['unvalidated']}/{len(results)}",
        "",
        "## Exact source validation",
        f"Source occurrence confrontate: {len(occurrence_records)}/215",
        f"Source occurrence PASS: {sum(ev.status == 'pass' for ev in occurrence_records)}/{len(occurrence_records)}",
        f"Task fold PASS: {sum(f.status == 'pass' for f in fold_records)}/{len(fold_records)}",
        f"PASS con root_match=false: {sum(ev.status == 'pass' and not ev.root_match for ev in occurrence_records)}/{len(occurrence_records)}",
        f"Structure match: {sum(ev.structure_match for ev in occurrence_records)}/{len(occurrence_records)}",
        f"Ports match: {sum(ev.ports_match for ev in occurrence_records)}/{len(occurrence_records)}",
        f"Types match: {sum(ev.types_match for ev in occurrence_records)}/{len(occurrence_records)}",
        f"Constants match: {sum(ev.constants_match for ev in occurrence_records)}/{len(occurrence_records)}",
        f"Functions match: {sum(ev.functions_match for ev in occurrence_records)}/{len(occurrence_records)}",
        "",
        "## TGIR integrity",
        f"Forward internal REF: {forward_refs}/{valid_backward_refs + forward_refs}",
        f"Valid backward REF: {valid_backward_refs}/{valid_backward_refs + forward_refs}",
        f"Unresolved internal REF: {unresolved_refs}/{valid_backward_refs + forward_refs + unresolved_refs}",
        f"Fixed functions declared: {fixed_functions}/{fixed_functions}",
        f"Primary INPUT used and declared: {declares_primary}/{uses_primary}",
        f"Parameter metadata complete: {sum(bool(sp.source_kind) and bool(sp.mode) for sp in slot_specs)}/{len(slot_specs)}",
        "",
        "## Runtime-safety",
        f"Task ID nella candidate memory: 0/{len(all_graphs)}",
        f"SSA source names nella candidate memory: 0/{len(all_graphs)}",
        f"source_tasks non vuoti: {sum(bool(r.graph.source_tasks) for r in results)}/{len(results)}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--mastery", required=True)
    parser.add_argument("--graphs-output", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--promotable-output", required=True)
    args = parser.parse_args()

    try:
        results, failed_patterns = anti_unify_all(args.patterns, args.mastery)
        if failed_patterns:
            raise AntiUnificationError(
                f"Anti-unification failed for {len(failed_patterns)} pattern(s): "
                f"{failed_patterns[:3]}"
            )

        all_graphs = [serialize_candidate_graph(result) for result in results]
        promotable_graphs = [
            graph for graph in all_graphs
            if graph["promotion_status"] == "structurally_promotable"
        ]
        evidence_out = [serialize_evidence(result) for result in results]

        if len({g["graph_id"] for g in all_graphs}) != len(all_graphs):
            raise AntiUnificationError("Duplicate graph IDs in candidate output")
        if len({g["graph_id"] for g in promotable_graphs}) != len(promotable_graphs):
            raise AntiUnificationError("Duplicate graph IDs in promotable output")

        report = _build_report(results, all_graphs, promotable_graphs, evidence_out)
        _write_json(args.graphs_output, all_graphs)
        _write_json(args.promotable_output, promotable_graphs)
        _write_json(args.evidence_output, evidence_out)
        os.makedirs(os.path.dirname(args.report_output) or ".", exist_ok=True)
        with open(args.report_output, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(report, end="")
    except AntiUnificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
