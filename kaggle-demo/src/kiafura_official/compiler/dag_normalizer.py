"""Deterministic DAG normalization for Hodel SSA graphs.

M1 scope: stable topological order, canonical node IDs, port-labeled
input classification and structural hashing with literal-type normalization.

V2: NormalizedNode.inputs carries (ref, category, type_hint) triple so that
constants with the same type but different values produce the same structural
identity.  COLOR(2) == COLOR(5) structurally; COLOR(2) != INT(2) structurally.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import Any

from schema.typed_graph_ir import (
    ParametricGraph,
    GraphNode,
    ValueType,
    ParamSlot,
    Constant,
)


class DagNormalizationError(RuntimeError):
    """Raised when a DAG cannot be normalized."""


class ForwardReferenceError(DagNormalizationError):
    """Node input references a later SSA definition."""


class UnresolvedReferenceError(DagNormalizationError):
    """Node input references a non-existent node."""


class CyclicDagError(DagNormalizationError):
    """DAG contains a cycle."""


class UnresolvedOutputError(DagNormalizationError):
    """Declared output node does not exist."""


# ── Input classification ──────────────────────────────────────────────

class InputCategory:
    NODE_REF = "NODE_REF"
    EXTERNAL_INPUT = "EXTERNAL_INPUT"
    CONSTANT = "CONSTANT"
    FUNCTION_REF = "FUNCTION_REF"
    UNKNOWN = "UNKNOWN"


def classify_input(
    name: str,
    node_output_names: set[str],
    param_map: dict[str, ValueType],
    function_names: set[str],
) -> tuple[str, str | None]:
    """Classify an input and return (category, type_hint).

    Returns (category, type_hint) where type_hint is the ValueType name
    for constants, None otherwise.
    """
    if name in node_output_names:
        return (InputCategory.NODE_REF, None)
    if name == "INPUT":
        return (InputCategory.EXTERNAL_INPUT, None)
    if name in function_names:
        return (InputCategory.FUNCTION_REF, None)
    if name in param_map:
        vt = param_map[name]
        return (InputCategory.CONSTANT, vt.name)
    return (InputCategory.UNKNOWN, None)


# ── Topological sort ──────────────────────────────────────────────────

def _topological_sort(
    nodes: dict[str, GraphNode],
    node_output_names: set[str],
) -> list[str]:
    """Kahn + stable tie-breaking."""
    in_degree: dict[str, int] = {}
    adjacency: dict[str, list[str]] = {}

    for name, node in nodes.items():
        if name not in in_degree:
            in_degree[name] = 0
        for inp in node.inputs:
            if inp in node_output_names:
                if inp not in adjacency:
                    adjacency[inp] = []
                adjacency[inp].append(name)
                in_degree[name] = in_degree.get(name, 0) + 1

    queue = deque(sorted([n for n, d in in_degree.items() if d == 0]))
    order: list[str] = []

    while queue:
        queue = deque(sorted(queue))
        name = queue.popleft()
        order.append(name)
        for successor in adjacency.get(name, []):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if len(order) != len(nodes):
        remaining = set(nodes) - set(order)
        raise CyclicDagError(
            f"Cycle involving {len(remaining)} node(s): {sorted(remaining)[:5]}"
        )
    return order


# ── Helper: producer / consumer maps ──────────────────────────────────

def _build_producer_map(
    dag: NormalizedDag,
) -> dict[str, list[tuple[int, str]]]:
    """node_id → [(port_index, producer_id), ...] for NODE_REF inputs."""
    pmap: dict[str, list[tuple[int, str]]] = {}
    for node in dag.nodes:
        pmap[node.canonical_id] = []
        for i, (ref, cat, _th) in enumerate(node.inputs):
            if cat == InputCategory.NODE_REF:
                pmap[node.canonical_id].append((i, ref))
    return pmap


def _build_consumer_map(
    dag: NormalizedDag,
) -> dict[str, list[tuple[str, int]]]:
    """node_id → [(consumer_id, port_index), ...]."""
    cmap: dict[str, list[tuple[str, int]]] = {}
    for node in dag.nodes:
        if node.canonical_id not in cmap:
            cmap[node.canonical_id] = []
    for node in dag.nodes:
        for i, (ref, cat, _th) in enumerate(node.inputs):
            if cat == InputCategory.NODE_REF:
                if ref not in cmap:
                    cmap[ref] = []
                cmap[ref].append((node.canonical_id, i))
    return cmap


# ── Canonical normalization ───────────────────────────────────────────


@dataclass(frozen=True)
class NormalizedNode:
    """A node in a canonically-renamed DAG.

    ``inputs`` is a tuple of (ref, category, type_hint) triples.
    For CONSTANT inputs, ``type_hint`` is the ValueType name (``COLOR``, ``INT``, etc.).
    For NODE_REF inputs, ``type_hint`` is the producer node's output type name.
    For others, ``type_hint`` is ``None``.
    """

    canonical_id: str
    original_name: str
    operator: str
    output_type: ValueType
    inputs: tuple[tuple[str, str, str | None], ...]


@dataclass(frozen=True)
class NormalizedDag:
    """Canonically-renamed DAG ready for structural hashing."""

    original_task_id: str
    nodes: tuple[NormalizedNode, ...]
    output_node_id: str
    node_map: dict[str, str]       # original SSA name → canonical ID
    reverse_map: dict[str, str]    # canonical ID → original SSA name
    param_types: dict[str, ValueType]
    audit: dict[str, Any]


def normalize_dag(graph: ParametricGraph) -> NormalizedDag:
    """Normalize a ParametricGraph into a canonically-renamed DAG.

    Returns NormalizedDag with canonical node IDs (n000, n001, ...) and
    type-normalized constant references.
    """
    # 1. Build node map and parameter map
    node_map: dict[str, GraphNode] = {}
    param_slot_map: dict[str, ValueType] = {}
    function_names: set[str] = set()

    for p in graph.parameters:
        param_slot_map[p.name] = p.value_type
        if p.value_type == ValueType.FUNCTION:
            function_names.add(p.name)

    for node in graph.nodes:
        if node.output_name in node_map:
            raise DagNormalizationError(f"Duplicate SSA name '{node.output_name}'")
        node_map[node.output_name] = node

    node_output_names: set[str] = set(node_map.keys())

    # 2. Topological sort
    topo_order_names = _topological_sort(node_map, node_output_names)
    topo_pos = {name: i for i, name in enumerate(topo_order_names)}

    # 3. Verify forward references
    for name in topo_order_names:
        node = node_map[name]
        for inp in node.inputs:
            if inp in node_output_names and topo_pos[inp] >= topo_pos[name]:
                raise ForwardReferenceError(
                    f"Node '{name}' forward-refs '{inp}'"
                )

    # 4. Verify output node
    output_original = graph.output_node
    if output_original not in node_map:
        if "OUTPUT" in node_map:
            output_original = "OUTPUT"
        else:
            raise UnresolvedOutputError(
                f"Output '{graph.output_node}' not in {sorted(node_output_names)}"
            )

    # 5. Alpha-rename
    canonical_ids: dict[str, str] = {}
    reverse_ids: dict[str, str] = {}
    for i, orig_name in enumerate(topo_order_names):
        can_id = f"n{i:03d}"
        canonical_ids[orig_name] = can_id
        reverse_ids[can_id] = orig_name

    # We need output types of producers to annotate NODE_REF inputs.
    # Build producer output types map.
    producer_types: dict[str, str] = {}
    for orig_name in topo_order_names:
        node = node_map[orig_name]
        producer_types[orig_name] = node.output_type.name

    normal_nodes: list[NormalizedNode] = []
    for orig_name in topo_order_names:
        node = node_map[orig_name]
        can_id = canonical_ids[orig_name]

        classified_inputs: list[tuple[str, str, str | None]] = []
        for inp in node.inputs:
            cat, type_hint = classify_input(
                inp, node_output_names, param_slot_map, function_names,
            )
            if cat == InputCategory.NODE_REF:
                resolved = canonical_ids.get(inp, inp)
                prod_type = producer_types.get(inp, "?")
                classified_inputs.append((resolved, cat, prod_type))
            elif cat == InputCategory.CONSTANT:
                # type_hint is the ValueType name (set by classify_input)
                classified_inputs.append((inp, cat, type_hint))
            elif cat == InputCategory.FUNCTION_REF:
                classified_inputs.append((inp, cat, "FUNC"))
            else:
                classified_inputs.append((inp, cat, type_hint))

        normal_nodes.append(NormalizedNode(
            canonical_id=can_id,
            original_name=orig_name,
            operator=node.operator,
            output_type=node.output_type,
            inputs=tuple(classified_inputs),
        ))

    output_canonical = canonical_ids[output_original]

    return NormalizedDag(
        original_task_id=graph.graph_id,
        nodes=tuple(normal_nodes),
        output_node_id=output_canonical,
        node_map=canonical_ids,
        reverse_map=reverse_ids,
        param_types=param_slot_map,
        audit={
            "original_nodes": len(graph.nodes),
            "normalized_nodes": len(normal_nodes),
            "param_slots": len(graph.parameters),
        },
    )


# ── Structural hashing ────────────────────────────────────────────────

def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def node_base_label(node: NormalizedNode) -> str:
    """Structural label excluding literal values.

    For CONSTANT inputs, encodes the VALUE TYPE name (COLOR, INT, ...)
    but NOT the constant name or value.
    """
    port_info = []
    for i, (ref, cat, th) in enumerate(node.inputs):
        if cat == InputCategory.CONSTANT:
            # Use the type hint (e.g., "COLOR") — NOT the constant name/value
            port_info.append((i, cat, th or "CONST"))
        elif cat == InputCategory.FUNCTION_REF:
            port_info.append((i, cat, "FUNC"))
        elif cat == InputCategory.NODE_REF:
            port_info.append((i, cat, th or "REF"))
        elif cat == InputCategory.EXTERNAL_INPUT:
            port_info.append((i, cat, "INPUT"))
        else:
            port_info.append((i, cat, "?"))

    payload = {
        "op": node.operator,
        "out_type": node.output_type.name,
        "ports": port_info,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def compute_wl_signatures(
    dag: NormalizedDag,
    rounds: int = 4,
) -> dict[str, list[str]]:
    """Bidirectional WL signatures.

    Returns {canonical_id: [sig_r0, sig_r1, ..., sig_rR]}
    """
    producer_map = _build_producer_map(dag)
    consumer_map = _build_consumer_map(dag)

    node_index = {n.canonical_id: n for n in dag.nodes}
    topo_order = [n.canonical_id for n in dag.nodes]

    # Base labels
    base_labels: dict[str, str] = {}
    for node in dag.nodes:
        base_labels[node.canonical_id] = node_base_label(node)

    sigs: dict[str, list[str]] = {}
    for node in dag.nodes:
        sigs[node.canonical_id] = [base_labels[node.canonical_id]]

    for r in range(1, rounds + 1):
        # Bottom-up
        up_sigs: dict[str, str] = {}
        for cid in topo_order:
            pred_sigs = []
            for port_idx, pred_id in producer_map.get(cid, []):
                if r - 1 < len(sigs.get(pred_id, [])):
                    pred_sigs.append((port_idx, sigs[pred_id][r - 1]))
            payload = {"base": base_labels[cid], "up": pred_sigs}
            up_sigs[cid] = hashlib.sha256(_canonical_json(payload)).hexdigest()

        # Top-down
        down_sigs: dict[str, str] = {}
        for cid in reversed(topo_order):
            cons_sigs = []
            for cons_id, port_idx in consumer_map.get(cid, []):
                if r - 1 < len(sigs.get(cons_id, [])):
                    cons_sigs.append((port_idx, sigs[cons_id][r - 1]))
            payload = {"base": base_labels[cid], "down": cons_sigs}
            down_sigs[cid] = hashlib.sha256(_canonical_json(payload)).hexdigest()

        # Combine
        for cid in topo_order:
            combined = hashlib.sha256(_canonical_json({
                "up": up_sigs[cid],
                "down": down_sigs[cid],
            })).hexdigest()
            sigs[cid].append(combined)

    return sigs


def compute_structural_hash(dag: NormalizedDag) -> str:
    """Deterministic hash of the entire DAG structure.

    For CONSTANT inputs, uses only the category + type hint, never the
    constant name or value.
    """
    parts = []
    for node in dag.nodes:
        inputs_normalized = []
        for ref, cat, th in node.inputs:
            if cat == InputCategory.CONSTANT:
                # Normalize constant to its type only
                inputs_normalized.append((cat, th or "CONST"))
            elif cat == InputCategory.NODE_REF:
                # Use canonical reference
                inputs_normalized.append((cat, ref))
            else:
                inputs_normalized.append((cat, ref))
        parts.append({
            "op": node.operator,
            "out_type": node.output_type.name,
            "inputs": inputs_normalized,
        })
    payload = {"nodes": parts, "output": dag.output_node_id}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()
