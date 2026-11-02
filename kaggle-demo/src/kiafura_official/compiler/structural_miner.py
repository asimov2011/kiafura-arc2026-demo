"""M1.2 (final): Exact structural DAG miner for Hodel SSA graphs.

V6 fixes from PM audit v3:
1. MinedPattern stores real CanonicalCone nodes (no placeholders)
2. NODE_REF boundary preserves producer output type (explicit branch)
3. Constant bindings: source_slot preserved in occurrences, stripped from hash
4. FUNCTION_REF: source_slot in occurrences
5. promotion_status field (promotable / blocked_untyped_constant)
6. pattern_to_dict serializes real graph structure
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from schema.typed_graph_ir import ParametricGraph, ValueType, Constant
from kiafura_official.compiler.hodel_parser import parse_hodel_tokens
from kiafura_official.compiler.dag_normalizer import (
    NormalizedDag,
    NormalizedNode,
    normalize_dag,
    InputCategory,
    DagNormalizationError,
)


@dataclass(frozen=True)
class CanonicalCone:
    nodes: tuple[dict[str, Any], ...]
    node_count: int
    edge_count: int
    exact_hash: str
    root_global_id: str
    ordered_global_node_ids: tuple[str, ...]
    local_to_global: dict[str, str]


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _build_producer_map(dag: NormalizedDag) -> dict[str, list[tuple[int, str]]]:
    pmap: dict[str, list[tuple[int, str]]] = {}
    for node in dag.nodes:
        pmap[node.canonical_id] = []
        for i, (ref, cat, _th) in enumerate(node.inputs):
            if cat == InputCategory.NODE_REF:
                pmap[node.canonical_id].append((i, ref))
    return pmap


def _collect_ancestors(
    root: str, pmap: dict[str, list[tuple[int, str]]],
    all_nodes: dict[str, NormalizedNode], max_depth: int,
) -> set[str]:
    selected = {root}
    frontier = [(root, 0)]
    visited: set[str] = set()
    while frontier:
        cid, depth = frontier.pop(0)
        if cid in visited:
            continue
        visited.add(cid)
        if depth >= max_depth:
            continue
        for _pi, pred_id in pmap.get(cid, []):
            if pred_id in all_nodes:
                selected.add(pred_id)
                frontier.append((pred_id, depth + 1))
    return selected


def canonicalize_cone(
    root_id: str, selected: set[str],
    dag: NormalizedDag, param_values: dict[str, Any],
) -> CanonicalCone | None:
    all_nodes: dict[str, NormalizedNode] = {n.canonical_id: n for n in dag.nodes}
    pmap = _build_producer_map(dag)

    if root_id not in all_nodes or root_id not in selected or len(selected) < 2:
        return None

    # c0 = root (assigned BEFORE DFS)
    local_of_global: dict[str, str] = {root_id: "c0"}
    next_local = 1
    visited_global: set[str] = set()

    def assign(gid: str) -> str:
        nonlocal next_local
        if gid not in local_of_global:
            local_of_global[gid] = f"c{next_local}"
            next_local += 1
        return local_of_global[gid]

    def dfs(gid: str) -> None:
        if gid in visited_global:
            return
        visited_global.add(gid)
        node = all_nodes[gid]
        for i, (ref, cat, _th) in enumerate(node.inputs):
            if cat == InputCategory.NODE_REF and ref in selected:
                dfs(ref)
        assign(gid)

    dfs(root_id)

    collected = list(local_of_global.keys())
    topo_order = _topological_sort_local(collected, all_nodes, pmap)
    if topo_order is None:
        return None

    global_of_local = {v: k for k, v in local_of_global.items()}
    node_by_lid: dict[str, NormalizedNode] = {}
    for gid in collected:
        lid = local_of_global[gid]
        node_by_lid[lid] = all_nodes[gid]

    internal_edges = 0
    canonical_nodes: list[dict[str, Any]] = []
    sorted_lids = sorted(local_of_global.values(), key=lambda x: int(x[1:]))

    for lid in sorted_lids:
        node = node_by_lid[lid]
        inputs: list[dict[str, Any]] = []

        for i, (ref, cat, th) in enumerate(node.inputs):
            inp: dict[str, Any] = {"port": i}

            if cat == InputCategory.NODE_REF and ref in selected:
                target_lid = local_of_global.get(ref)
                if target_lid is not None:
                    inp["kind"] = "REF"
                    inp["target"] = target_lid
                    internal_edges += 1
                else:
                    inp["kind"] = "BOUNDARY"
                    inp["boundary"] = {"cat": cat, "type": th or "ANY"}
            elif cat == InputCategory.NODE_REF:
                # NODE_REF outside selected — explicit branch for type
                inp["kind"] = "BOUNDARY"
                inp["boundary"] = {"cat": cat, "type": th or "ANY"}
            elif cat == InputCategory.CONSTANT:
                val = param_values.get(ref)
                inp["kind"] = "BOUNDARY"
                inp["boundary"] = {
                    "cat": cat,
                    "type": th or "ANY",
                    "source_slot": ref,
                    "constant_value": _native_value(val),
                }
            elif cat == InputCategory.FUNCTION_REF:
                fn_val = param_values.get(ref)
                inp["kind"] = "BOUNDARY"
                inp["boundary"] = {
                    "cat": cat,
                    "type": "FUNC",
                    "fn": str(fn_val.value) if isinstance(fn_val, Constant) else str(ref),
                    "source_slot": ref,
                    "constant_value": str(fn_val.value) if isinstance(fn_val, Constant) else str(ref),
                }
            elif cat == InputCategory.EXTERNAL_INPUT:
                inp["kind"] = "BOUNDARY"
                inp["boundary"] = {"cat": cat, "type": "INPUT"}
            else:
                inp["kind"] = "BOUNDARY"
                inp["boundary"] = {"cat": cat, "type": "?"}

            inputs.append(inp)

        canonical_nodes.append({
            "local_id": lid,
            "operator": node.operator,
            "output_type": node.output_type.name,
            "inputs": inputs,
        })

    # Hash: strip constant_value and source_slot from boundary dicts
    def _hash_boundary(b: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in b.items() if k not in ("constant_value", "source_slot")}

    encoding_payload = {
        "nodes": [
            {
                "lid": n["local_id"],
                "op": n["operator"],
                "out": n["output_type"],
                "in": [
                    (inp["port"], inp["kind"],
                     _hash_boundary(inp["boundary"]) if inp["kind"] == "BOUNDARY" else inp.get("target", ""))
                    for inp in n["inputs"]
                ],
            }
            for n in canonical_nodes
        ],
    }
    exact_hash = hashlib.sha256(_canonical_json(encoding_payload)).hexdigest()

    l2g = {v: k for k, v in local_of_global.items()}

    return CanonicalCone(
        nodes=tuple(canonical_nodes),
        node_count=len(canonical_nodes),
        edge_count=internal_edges,
        exact_hash=exact_hash,
        root_global_id=root_id,
        ordered_global_node_ids=tuple(topo_order),
        local_to_global=l2g,
    )


def _native_value(val: Any) -> Any:
    """Convert a Constant or other value to a native JSON value."""
    if isinstance(val, Constant):
        v = val.value
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return v
        if isinstance(v, (list, tuple)):
            return list(v)
        return v
    return val


def _topological_sort_local(
    nids: list[str], all_nodes: dict[str, NormalizedNode],
    pmap: dict[str, list[tuple[int, str]]],
) -> list[str] | None:
    nset = set(nids)
    indeg: dict[str, int] = {}
    adj: dict[str, list[str]] = {}
    for nid in nids:
        if nid not in indeg:
            indeg[nid] = 0
        for _pi, pred_id in pmap.get(nid, []):
            if pred_id in nset:
                if pred_id not in adj:
                    adj[pred_id] = []
                adj[pred_id].append(nid)
                indeg[nid] = indeg.get(nid, 0) + 1
    from collections import deque
    q = deque(sorted([n for n, d in indeg.items() if d == 0]))
    order: list[str] = []
    while q:
        q = deque(sorted(q))
        nid = q.popleft()
        order.append(nid)
        for succ in adj.get(nid, []):
            indeg[succ] -= 1
            if indeg[succ] == 0:
                q.append(succ)
    if len(order) != len(nids):
        return None
    return order


@dataclass(frozen=True)
class StructuralOccurrence:
    task_id: str
    root_output_name: str
    node_output_names: tuple[str, ...]
    local_to_output_name: dict[str, str]
    boundary_bindings: dict[str, dict[str, Any]]
    exact_hash: str


@dataclass
class MinedPattern:
    pattern_hash: str
    root_local_id: str
    nodes: tuple[dict[str, Any], ...]
    node_count: int
    edge_count: int
    support_tasks: int
    occurrence_count: int
    root_operator: str
    operators: tuple[str, ...]
    promotion_status: str
    occurrences: list[StructuralOccurrence]


@dataclass
class MiningReport:
    task_count: int
    task_failures: list[str]
    total_nodes: int
    total_cones: int
    roots_with_candidates: int
    structural_buckets: int
    patterns_with_support_ge2: int
    tasks_covered: int
    operators_covered: int
    distinct_operators: int
    patterns: list[MinedPattern]
    wall_seconds: float


def mine_cohort(
    graphs: list[tuple[str, ParametricGraph]],
    min_support: int = 2, min_nodes: int = 2,
    max_nodes: int = 12, max_depth: int = 6,
) -> MiningReport:
    import time
    start = time.time()

    task_failures: list[str] = []
    normalized_dags: list[tuple[str, NormalizedDag]] = []
    param_vals_map: dict[str, dict[str, Any]] = {}

    for task_id, graph in graphs:
        try:
            dag = normalize_dag(graph)
            normalized_dags.append((task_id, dag))
            pv: dict[str, Any] = {}
            for p in graph.parameters:
                pv[p.name] = p.default
            param_vals_map[task_id] = pv
        except DagNormalizationError as e:
            task_failures.append(f"{task_id}: {e}")

    total_nodes = sum(len(dag.nodes) for _, dag in normalized_dags)
    all_ops: set[str] = set()
    node_maps: dict[str, dict[str, NormalizedNode]] = {}
    p_maps: dict[str, dict[str, list[tuple[int, str]]]] = {}

    for tid, dag in normalized_dags:
        node_maps[tid] = {n.canonical_id: n for n in dag.nodes}
        p_maps[tid] = _build_producer_map(dag)
        for n in dag.nodes:
            all_ops.add(n.operator)

    cone_registry: dict[str, list[tuple[str, CanonicalCone]]] = defaultdict(list)
    seen_keys: dict[str, set] = defaultdict(set)

    for tid, dag in normalized_dags:
        nm = node_maps[tid]
        pm = p_maps[tid]
        pv = param_vals_map.get(tid, {})

        for node in dag.nodes:
            cid = node.canonical_id
            prev_sel: frozenset[str] | None = None

            for depth in range(1, max_depth + 1):
                sel = _collect_ancestors(cid, pm, nm, depth)
                if len(sel) < min_nodes:
                    continue
                if len(sel) > max_nodes:
                    break
                fs = frozenset(sel)
                if prev_sel is not None and fs == prev_sel:
                    break
                prev_sel = fs
                cone = canonicalize_cone(cid, sel, dag, pv)
                if cone is None:
                    continue
                key = (tid, cid, fs)
                if key in seen_keys[cone.exact_hash]:
                    continue
                seen_keys[cone.exact_hash].add(key)
                cone_registry[cone.exact_hash].append((tid, cone))

    patterns: list[MinedPattern] = []
    for exact_hash, occurrences in cone_registry.items():
        distinct_tasks = {tid for tid, _ in occurrences}
        if len(distinct_tasks) < min_support:
            continue

        occ_list: list[StructuralOccurrence] = []
        for tid, cone in occurrences:
            dag_ref = next(d for t, d in normalized_dags if t == tid)
            root_orig = dag_ref.reverse_map.get(cone.root_global_id, cone.root_global_id)
            node_names = tuple(
                dag_ref.reverse_map.get(gid, gid)
                for gid in cone.ordered_global_node_ids
            )

            l2o: dict[str, str] = {}
            for lid, gid in cone.local_to_global.items():
                l2o[lid] = dag_ref.reverse_map.get(gid, gid)

            bb: dict[str, dict[str, Any]] = {}
            for n in cone.nodes:
                lid = n.get("local_id", "")
                for inp in n.get("inputs", []):
                    if inp.get("kind") == "BOUNDARY":
                        b = inp.get("boundary", {})
                        key = f"{lid}:{inp['port']}"
                        entry: dict[str, Any] = {"type": b.get("type", "?")}

                        if b.get("cat") == InputCategory.CONSTANT:
                            entry["source_slot"] = b.get("source_slot", "?")
                            entry["value"] = b.get("constant_value")
                        elif b.get("cat") == InputCategory.FUNCTION_REF:
                            entry["source_slot"] = b.get("source_slot", "?")
                            entry["value"] = b.get("constant_value")
                            entry["type"] = "FUNCTION"
                        elif b.get("cat") == InputCategory.NODE_REF:
                            entry["type"] = b.get("type", "?")

                        bb[key] = entry

            occ_list.append(StructuralOccurrence(
                task_id=tid, root_output_name=root_orig,
                node_output_names=node_names,
                local_to_output_name=l2o, boundary_bindings=bb,
                exact_hash=exact_hash,
            ))

        ref_cone = occurrences[0][1]

        # Root operator from c0
        root_op = ""
        for n in ref_cone.nodes:
            if n.get("local_id") == "c0":
                root_op = n.get("operator", "")
                break

        ops = tuple(sorted(set(n.get("operator", "") for n in ref_cone.nodes)))

        # Promotion status
        has_any = any(
            inp.get("boundary", {}).get("type") == "ANY"
            and inp.get("boundary", {}).get("cat") == InputCategory.CONSTANT
            for n in ref_cone.nodes
            for inp in n.get("inputs", [])
            if inp.get("kind") == "BOUNDARY"
        )
        promo = "blocked_untyped_constant" if has_any else "promotable"

        patterns.append(MinedPattern(
            pattern_hash=exact_hash,
            root_local_id="c0",
            nodes=ref_cone.nodes,
            node_count=ref_cone.node_count,
            edge_count=ref_cone.edge_count,
            support_tasks=len(distinct_tasks),
            occurrence_count=len(occurrences),
            root_operator=root_op,
            operators=ops,
            promotion_status=promo,
            occurrences=occ_list,
        ))

    patterns.sort(key=lambda p: (-p.support_tasks, -p.node_count, p.pattern_hash))

    covered_tasks = set()
    covered_ops = set()
    for p in patterns:
        for o in p.occurrences:
            covered_tasks.add(o.task_id)
        covered_ops.update(p.operators)

    wall = time.time() - start
    rwc = len(set((tid, cone.root_global_id) for h, occs in cone_registry.items() for tid, cone in occs))
    tc = sum(len(v) for v in cone_registry.values())

    return MiningReport(
        task_count=len(graphs) - len(task_failures),
        task_failures=task_failures, total_nodes=total_nodes,
        total_cones=tc, roots_with_candidates=rwc,
        structural_buckets=len(cone_registry),
        patterns_with_support_ge2=len(patterns),
        tasks_covered=len(covered_tasks),
        operators_covered=len(covered_ops),
        distinct_operators=len(all_ops),
        patterns=patterns, wall_seconds=wall,
    )


# ── Serialization ─────────────────────────────────────────────────────

def _serialize_input(inp: dict[str, Any]) -> dict[str, Any]:
    """Serialize a single input descriptor for JSON output."""
    result: dict[str, Any] = {"port": inp["port"], "kind": inp["kind"]}
    if inp["kind"] == "REF":
        result["target"] = inp.get("target", "")
    elif inp["kind"] == "BOUNDARY":
        b = inp.get("boundary", {})
        # Strip internal fields from JSON output
        out_b: dict[str, Any] = {"cat": b.get("cat", "?")}
        if "type" in b:
            out_b["type"] = b["type"]
        if "fn" in b:
            out_b["fn"] = b["fn"]
        result["boundary"] = out_b
    return result


def pattern_to_dict(p: MinedPattern) -> dict[str, Any]:
    return {
        "pattern_hash": p.pattern_hash,
        "root_local_id": p.root_local_id,
        "node_count": p.node_count,
        "edge_count": p.edge_count,
        "support_tasks": p.support_tasks,
        "occurrence_count": p.occurrence_count,
        "root_operator": p.root_operator,
        "operators": sorted(p.operators),
        "promotion_status": p.promotion_status,
        "nodes": [
            {
                "local_id": n["local_id"],
                "operator": n["operator"],
                "output_type": n["output_type"],
                "inputs": [_serialize_input(inp) for inp in n.get("inputs", [])],
            }
            for n in p.nodes
        ],
        "occurrences": [
            {
                "task_id": o.task_id,
                "root_output_name": o.root_output_name,
                "node_output_names": list(o.node_output_names),
                "local_to_output_name": o.local_to_output_name,
                "boundary_bindings": o.boundary_bindings,
            }
            for o in p.occurrences
        ],
    }


def report_to_dict(report: MiningReport) -> dict[str, Any]:
    return {
        "task_count": report.task_count,
        "task_failures": report.task_failures,
        "total_nodes": report.total_nodes,
        "total_cones": report.total_cones,
        "roots_with_candidates": report.roots_with_candidates,
        "structural_buckets": report.structural_buckets,
        "patterns_with_support_ge2": report.patterns_with_support_ge2,
        "tasks_covered": report.tasks_covered,
        "operators_covered": report.operators_covered,
        "distinct_operators": report.distinct_operators,
        "wall_seconds": round(report.wall_seconds, 3),
        "patterns": [pattern_to_dict(p) for p in report.patterns],
    }


def mine_a2(
    mastery_path: str, manifest_path: str,
    min_support: int = 2, min_nodes: int = 2, max_nodes: int = 12,
) -> MiningReport:
    import json as _json
    with open(manifest_path) as f:
        manifest = _json.load(f)
    tids: list[str] = manifest.get("a2_ids", manifest.get("a_ids", []))
    graphs: list[tuple[str, ParametricGraph]] = []
    with open(mastery_path) as f:
        for line in f:
            d = _json.loads(line.strip())
            if d.get("backend") == "hodel-ssa" and d["task_id"] in tids:
                graphs.append((d["task_id"], parse_hodel_tokens(d["task_id"], d["tokens"])))
    return mine_cohort(graphs, min_support=min_support, min_nodes=min_nodes, max_nodes=max_nodes)
