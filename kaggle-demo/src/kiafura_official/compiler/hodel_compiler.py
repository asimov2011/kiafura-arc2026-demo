"""Compilatore Hodel: tokens → Typed Graph IR → fattorizzazione → graph dichiarativi.

Flusso:
  1. Parse: tokens Hodel → ParametricGraph per ogni task
  2. Fattorizzazione: scopri sottografi ricorrenti tra task
  3. Anti-unificazione: parametrizza sottografi in graph con binding slots
  4. Genera memoria operativa: manifest con graph, costi, witness
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Sequence

from schema.typed_graph_ir import (
    Constant, ValueType, ParamSlot, GraphNode, ParametricGraph,
    OPERATOR_REGISTRY, MemoryManifest, OperatorSchema,
)
from src.kiafura_official.compiler.hodel_parser import parse_hodel_tokens


# ──────────────────────────────────────────────
# STEP 1: PARSE COORTE
# ──────────────────────────────────────────────

def parse_cohort(task_ids: list[str], mastery_path: str) -> dict[str, ParametricGraph]:
    """Parsa tutti i task di una coorte in ParametricGraph."""
    import json as _json
    result = {}
    with open(mastery_path) as f:
        for line in f:
            d = _json.loads(line.strip())
            if d.get('backend') == 'hodel-ssa' and d['task_id'] in task_ids:
                result[d['task_id']] = parse_hodel_tokens(d['task_id'], d['tokens'])
    return result


# ──────────────────────────────────────────────
# STEP 2: FATTORIZZAZIONE
# ──────────────────────────────────────────────

def _operator_sequence(graph: ParametricGraph) -> tuple[str, ...]:
    """Estrae la sequenza di operatori da un grafo."""
    return tuple(n.operator for n in graph.nodes)


def _factorize(graphs: dict[str, ParametricGraph]) -> dict[tuple[str, ...], list[str]]:
    """Trova sottosequenze di operatori condivise tra più task.
    
    Ritorna: {sottosequenza: [task_id, ...]}
    Solo sequenze con support ≥ 2.
    """
    # Per ogni task, estrai la sequenza di operatori
    sequences = {}
    for tid, g in graphs.items():
        sequences[tid] = _operator_sequence(g)
    
    # Cerca sottosequenze comuni di lunghezza 2-5
    pattern_tasks = defaultdict(list)
    
    for tid, seq in sequences.items():
        # Sottosequenze di lunghezza 2-5
        for L in range(2, min(6, len(seq) + 1)):
            for i in range(len(seq) - L + 1):
                sub = seq[i:i+L]
                pattern_tasks[sub].append(tid)
    
    # Filtra support ≥ 2
    return {pat: sorted(set(tids)) for pat, tids in pattern_tasks.items() if len(set(tids)) >= 2}


# ──────────────────────────────────────────────
# STEP 3: ANTI-UNIFICAZIONE
# ──────────────────────────────────────────────

def _anti_unify(pattern: tuple[str, ...], task_ids: list[str],
                graphs: dict[str, ParametricGraph]) -> ParametricGraph:
    """Anti-unifica una sequenza di operatori in un graph parametrico.
    
    Prende un pattern condiviso (es. objects → colorfilter → first)
    e crea un ParametricGraph con slot di binding per i parametri.
    """
    # Determina i parametri da ogni occorrenza del pattern
    param_slots = set()
    for tid in task_ids:
        if tid not in graphs:
            continue
        g = graphs[tid]
        # Trova la posizione del pattern nella sequenza
        seq = _operator_sequence(g)
        for i in range(len(seq) - len(pattern) + 1):
            if seq[i:i+len(pattern)] == pattern:
                # Estrai i parametri costanti da questa posizione
                for j in range(len(pattern)):
                    if i + j < len(g.nodes):
                        node = g.nodes[i + j]
                        # Cerca parametri costanti associati a questo nodo
                        for p in g.parameters:
                            if p.name in node.inputs:
                                param_slots.add(p)
    
    # Crea nodi del grafo
    nodes = []
    for j, op in enumerate(pattern):
        schema = OPERATOR_REGISTRY.get(op)
        output_type = schema.output_type if schema else ValueType.GRID
        inputs = ('INPUT',) if j == 0 else (f"D{j}",)
        node = GraphNode(
            operator=op,
            inputs=inputs,
            output_type=output_type,
            output_name=f"D{j+1}"
        )
        nodes.append(node)
    
    # Genera ID univoco
    graph_id = hashlib.sha256(
        json.dumps({"pattern": pattern, "support": len(task_ids)},
                   sort_keys=True).encode()
    ).hexdigest()[:16]
    
    return ParametricGraph(
        graph_id=f"fact_{graph_id}",
        family="compiled",
        version="v1",
        inputs=(ParamSlot("input", ValueType.GRID),),
        parameters=tuple(sorted(set(param_slots), key=lambda x: x.name))[:5],
        output_type=ValueType.GRID,
        nodes=tuple(nodes),
        output_node=nodes[-1].output_name if nodes else "INPUT",
        support=len(task_ids),
        cost=sum(OPERATOR_REGISTRY.get(n.operator, OperatorSchema(n.operator, (), ValueType.ANY, cost=1)).cost for n in nodes),
        source_tasks=tuple(task_ids),
        is_singleton=len(task_ids) <= 1,
    )


# ──────────────────────────────────────────────
# STEP 4: GENERAZIONE MEMORIA
# ──────────────────────────────────────────────

def compile_cohort(
    task_ids: list[str],
    mastery_path: str,
    release_id: str = "",
    parent_memory: MemoryManifest | None = None,
) -> MemoryManifest:
    """Compila una coorte di task Hodel in memoria operativa.
    
    1. Parser ogni task Hodel → ParametricGraph
    2. Fattorizza: scopri sottosequenze condivise
    3. Anti-unifica: crea graph parametrici
    4. Genera MemoryManifest
    """
    # Step 1: parse
    graphs = parse_cohort(task_ids, mastery_path)
    if not graphs:
        raise ValueError("Nessun task Hodel parsato")
    
    print(f"  Parsati {len(graphs)}/{len(task_ids)} task Hodel")
    
    # Step 2: fattorizzazione
    patterns = _factorize(graphs)
    print(f"  Trovati {len(patterns)} pattern condivisi (support≥2)")
    
    # Step 3: anti-unificazione
    param_graphs = []
    for pattern, tids in sorted(patterns.items(), key=lambda x: -len(x[1])):
        g = _anti_unify(pattern, tids, graphs)
        param_graphs.append(g)
    
    # Limita a graph con support ≥ 2
    promoted = [g for g in param_graphs if g.support >= 2]
    singletons = [g for g in param_graphs if g.support == 1]
    
    print(f"  Promossi: {len(promoted)} graph (support≥2)")
    print(f"  Singleton: {len(singletons)} graph (support=1)")
    
    # Step 4: genera memoria
    memory = MemoryManifest(
        format="kiafura-tgir-memory-v1",
        release_id=release_id,
        parent_release_id=parent_memory.release_id if parent_memory else None,
        graphs=tuple(promoted),
        singletons=tuple(singletons),
        budget=2_000_000,
    )
    memory.memory_sha256 = memory.freeze_hash()
    
    return memory


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-ids', nargs='+', help='Task IDs da compilare')
    parser.add_argument('--task-list', help='File con lista di task ID (uno per riga)')
    parser.add_argument('--mastery', required=True, help='Path al mastery .jsonl')
    parser.add_argument('--release-id', default='v1', help='Identificatore release')
    parser.add_argument('--output', default='memory_manifest.json', help='Output file')
    args = parser.parse_args()
    
    if args.task_list:
        with open(args.task_list) as f:
            task_ids = [line.strip() for line in f if line.strip()]
    elif args.task_ids:
        task_ids = args.task_ids
    else:
        parser.error('Specificare --task-ids o --task-list')
    
    memory = compile_cohort(task_ids, args.mastery, args.release_id)
    
    import json as _json
    # Serializza
    output = {
        'format': memory.format,
        'release_id': memory.release_id,
        'parent_release_id': memory.parent_release_id,
        'memory_sha256': memory.memory_sha256,
        'budget': memory.budget,
        'contract': memory.contract,
        'graphs': [{
            'graph_id': g.graph_id,
            'family': g.family,
            'support': g.support,
            'is_singleton': g.is_singleton,
            'cost': g.cost,
            'parameters': [(p.name, p.value_type.name) for p in g.parameters],
            'nodes': [(n.operator, list(n.inputs)) for n in g.nodes],
            'output_node': g.output_node,
        } for g in memory.graphs],
        'singletons': [{
            'graph_id': g.graph_id,
            'family': g.family,
            'support': g.support,
            'cost': g.cost,
            'parameters': [(p.name, p.value_type.name) for p in g.parameters],
            'nodes': [(n.operator, list(n.inputs)) for n in g.nodes],
        } for g in memory.singletons],
    }
    
    with open(args.output, 'w') as f:
        _json.dump(output, f, indent=2)
    
    print(f"\nMemoria scritta in {args.output}")
    print(f"  Graph promossi: {len(memory.graphs)}")
    print(f"  Singleton: {len(memory.singletons)}")
    print(f"  Hash: {memory.memory_sha256[:20]}")
    
    # Verifica gate: tutti i task hanno almeno un graph che li copre?
    covered_tasks = set()
    for g in memory.graphs:
        covered_tasks.update(g.source_tasks)
    for g in memory.singletons:
        covered_tasks.update(g.source_tasks)
    
    uncovered = set(task_ids) - covered_tasks
    if uncovered:
        print(f"\n⚠️ Task non industrializzati: {len(uncovered)}/{len(task_ids)}")
    else:
        print(f"\n✅ Gate sorgente: {len(task_ids)}/{len(task_ids)} coperti")


if __name__ == '__main__':
    main()
