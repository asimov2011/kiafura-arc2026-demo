"""Protocol validation: compile Hodel cohort → M0A gate → M1 → M2 → transfer.

Per coorte Ai (100 Hodel tasks):
  1. M0A gate: tutti i 100/100 devono parseare come Hodel-ssa validi
  2. M1 mining + M2 anti-unification
  3. Solo se M0A gate pass: transfer strutturale su Bi (900 task)
  4. Report

Nessuna memoria cumulativa. Ogni run riparte da zero.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from schema.typed_graph_ir import (
    Constant, ValueType, GraphNode, ParametricGraph, MemoryManifest,
)
from src.kiafura_official.compiler.hodel_parser import parse_hodel_tokens
from src.kiafura_official.compiler.structural_miner import mine_cohort, report_to_dict
from src.kiafura_official.compiler.structural_anti_unifier import (
    anti_unify_all, serialize_candidate_graph, serialize_evidence,
)

from src.kiafura_official.protocol_validation.cohorts import (
    build_disjoint_cohorts, verify_disjoint,
)


CORPUS_ROOT = Path("/root/projects/kiafura/legacy/data/external/ARC-AGI-2/data/training")


@dataclass
class TaskScore:
    task_id: str
    exact: bool = False
    false: bool = False
    abstain: bool = False
    graph_id: str | None = None
    error: str | None = None
    work: int = 0


def aggregate_scores(scores: list[TaskScore]) -> dict[str, Any]:
    return {
        "exact": sum(1 for s in scores if s.exact),
        "false": sum(1 for s in scores if s.false),
        "abstain": sum(1 for s in scores if s.abstain),
        "total": len(scores),
    }


def load_hodel_graphs(
    mastery_path: str, task_ids: list[str],
) -> tuple[dict[str, ParametricGraph], list[str]]:
    """Load Hodel-ssa programs from mastery. Returns (graphs, parse_errors)."""
    tid_set = set(task_ids)
    graphs: dict[str, ParametricGraph] = {}
    errors: list[str] = []
    with open(mastery_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("backend") == "hodel-ssa" and d["task_id"] in tid_set:
                try:
                    graphs[d["task_id"]] = parse_hodel_tokens(d["task_id"], d["tokens"])
                except Exception as exc:
                    errors.append(f"{d['task_id']}: {exc}")
    return graphs, errors


def run_cohort(
    cohort: dict[str, Any],
    mastery_path: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run one cohort through the full protocol."""
    cohort_id = cohort["cohort_id"]
    a_ids = cohort["a_ids"]
    b_ids = cohort["b_ids"]
    out = Path(output_dir) / cohort_id
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    print(f"\n{'='*60}")
    print(f"  Cohort {cohort_id}: |Ai|={len(a_ids)} |Bi|={len(b_ids)}")
    print(f"{'='*60}")

    # Manifest
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(cohort, indent=2), encoding="utf-8")
    print(f"\n[1] Manifest saved")

    # Load & M0A parse gate
    print(f"\n[2] M0A parse gate ({len(a_ids)} tasks)...")
    graphs, parse_errors = load_hodel_graphs(mastery_path, a_ids)
    valid_count = len(graphs)
    print(f"  Valid: {valid_count}/{len(a_ids)}, parse errors: {len(parse_errors)}")

    source_gate_pass = valid_count == len(a_ids)
    source_gate = {
        "cohort_id": cohort_id,
        "phase": "source_gate_m0a",
        "total": len(a_ids),
        "valid": valid_count,
        "parse_errors": len(parse_errors),
        "gate_pass": source_gate_pass,
        "parse_error_tasks": parse_errors[:5] if parse_errors else [],
    }
    source_path = out / "source_gate.json"
    source_path.write_text(json.dumps(source_gate, indent=2), encoding="utf-8")

    base_result = None
    final_result = None
    m1_elapsed = 0.0
    m2_elapsed = 0.0
    report = None
    results = []
    promoted_graphs = ()
    singletons = ()

    if not source_gate_pass:
        print(f"  ❌ M0A gate BLOCKED — {len(parse_errors)} tasks failed parse")
    else:
        # M1 mining
        print(f"\n[3] M1 mining ({valid_count} tasks)...")
        m1_start = time.monotonic()
        graph_list = [(tid, g) for tid, g in graphs.items()]
        report = mine_cohort(graph_list, min_support=2, min_nodes=2, max_nodes=12)
        m1_elapsed = time.monotonic() - m1_start
        print(f"  {len(report.patterns)} patterns, {report.total_nodes} nodes, "
              f"{report.tasks_covered}/{report.task_count} tasks — {m1_elapsed:.1f}s")

        m1_dict = report_to_dict(report)
        m1_path = out / "hodel-m1-patterns.json"
        m1_path.write_text(json.dumps(m1_dict, indent=2), encoding="utf-8")

        # M2 anti-unification
        print(f"\n[4] M2 anti-unification...")
        m2_start = time.monotonic()
        m1_for_m2 = out / "hodel-m1-for-m2.json"
        m1_for_m2.write_text(json.dumps(m1_dict, indent=2), encoding="utf-8")
        if m1_dict.get("patterns"):
            results, failed_patterns = anti_unify_all(str(m1_for_m2), mastery_path)
        m2_elapsed = time.monotonic() - m2_start
        print(f"  {len(results)} graphs anti-unified — {m2_elapsed:.1f}s")

        promoted_graphs = tuple(r.graph for r in results if r.graph.support >= 2)
        singletons = tuple(r.graph for r in results if r.graph.support < 2)

        graphs_data = [serialize_candidate_graph(r) for r in results]
        graphs_path = out / "hodel-m2-parametric-graphs.json"
        graphs_path.write_text(json.dumps(graphs_data, indent=2), encoding="utf-8")

        # Transfer (structural baseline)
        print(f"\n[5] Transfer base on Bi ({len(b_ids)} tasks)...")
        base_start = time.monotonic()
        base_scores = [TaskScore(task_id=t, abstain=True) for t in b_ids]
        base_agg = aggregate_scores(base_scores)
        base_elapsed = time.monotonic() - base_start
        base_result = {
            "cohort_id": cohort_id, "phase": "transfer_base_structural",
            "total": len(b_ids),
            "exact": base_agg["exact"], "false": base_agg["false"],
            "abstain": base_agg["abstain"],
            "elapsed_seconds": round(base_elapsed, 2),
        }
        base_path = out / "b900_base_score.json"
        base_path.write_text(json.dumps(base_result, indent=2), encoding="utf-8")

        final_result = {
            "cohort_id": cohort_id, "phase": "transfer_final",
            "total": len(b_ids),
            "exact_final": base_agg["exact"], "false_final": base_agg["false"],
            "abstain_final": base_agg["abstain"],
            "additional_exact": 0, "lost_exact": 0, "additional_false": 0,
            "dsd_status": "not_implemented",
        }
        final_path = out / "b900_final_score.json"
        final_path.write_text(json.dumps(final_result, indent=2), encoding="utf-8")

    # Build summary
    elapsed = round(time.monotonic() - started, 1)
    summary = [
        f"# Protocol Validation — Cohort {cohort_id}",
        "",
        f"## M0A Source Gate",
        f"Valid: {valid_count}/{len(a_ids)} → {'PASS' if source_gate_pass else 'BLOCKED'}",
        f"Parse errors: {len(parse_errors)}",
    ]
    if parse_errors:
        for e in parse_errors[:3]:
            summary.append(f"- {e}")

    if report:
        summary.extend([
            "",
            f"## M1 Mining",
            f"Patterns: {len(report.patterns)}",
            f"Tasks covered: {report.tasks_covered}/{report.task_count}",
            f"Total nodes: {report.total_nodes}",
            f"Roots with candidates: {report.roots_with_candidates}",
        ])

    if base_result:
        summary.extend([
            "",
            f"## Transfer B900 ({len(b_ids)} tasks)",
            f"Base: exact={base_result['exact']} false={base_result['false']} abstain={base_result['abstain']}",
            f"Final: exact_final={final_result['exact_final']} false_final={final_result['false_final']}",
        ])

    if results:
        summary.extend([
            "",
            f"## M2 Anti-unification",
            f"Graphs: {len(promoted_graphs)} promoted, {len(singletons)} singletons",
            f"Fully typed: {sum(1 for r in results if r.classification.type_precision == 'fully_typed')}",
            f"Internal ANY: {sum(1 for r in results if r.classification.type_precision == 'internal_any')}",
            f"Boundary ANY: {sum(1 for r in results if r.classification.type_precision == 'boundary_any')}",
            f"Untyped constant: {sum(1 for r in results if r.classification.type_precision == 'untyped_constant')}",
        ])

    summary.extend([
        "",
        f"## Timing",
        f"M1: {m1_elapsed:.1f}s",
        f"M2: {m2_elapsed:.1f}s",
        f"Total: {elapsed}s",
    ])
    summary_path = out / "summary.md"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"\n  Summary → {summary_path}")

    metadata = {
        "cohort_id": cohort_id,
        "command": "python -m src.kiafura_official.protocol_validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "a_count": len(a_ids), "b_count": len(b_ids),
        "source_gate": source_gate,
        "transfer_base": base_result, "transfer_final": final_result,
        "m1_patterns": len(report.patterns) if report else 0,
        "m1_tasks_covered": report.tasks_covered if report else 0,
        "m2_graphs": len(promoted_graphs) + len(singletons),
        "m2_promoted": len(promoted_graphs),
        "elapsed_seconds": elapsed,
    }
    meta_path = out / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    import argparse
    parser = argparse.ArgumentParser(description="KIAFURA Protocol Validation")
    parser.add_argument("command", choices=["all", "cohort", "list-cohorts"])
    parser.add_argument("--mastery", default="/root/projects/kiafura/kiafura-a100-repro/data/memory/arc2-mastery-v4.jsonl")
    parser.add_argument("--cohort-size", type=int, default=100)
    parser.add_argument("--seed", default="kiafura-protocol-validation-20260806")
    parser.add_argument("--first-manifest", default="data/splits/ct-2-random-a2-v1.json")
    parser.add_argument("--output", default="experiments/protocol-validation")
    parser.add_argument("--cohort-id")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.command == "list-cohorts":
        cohorts = build_disjoint_cohorts(
            args.mastery, args.cohort_size, args.seed, args.first_manifest
        )
        verify_disjoint(cohorts)
        print("\nAvailable cohorts:")
        for c in cohorts:
            print(f"  {c['cohort_id']}: {len(c['a_ids'])} Ai / {len(c['b_ids'])} Bi")
        return

    if args.command == "cohort":
        if not args.cohort_id:
            parser.error("--cohort-id required")
        cohorts = build_disjoint_cohorts(
            args.mastery, args.cohort_size, args.seed, args.first_manifest
        )
        cohort = next((c for c in cohorts if c["cohort_id"] == args.cohort_id), None)
        if cohort is None:
            parser.error(f"Cohort {args.cohort_id} not found")
        run_cohort(cohort, args.mastery, output_dir)
        return

    if args.command == "all":
        cohorts = build_disjoint_cohorts(
            args.mastery, args.cohort_size, args.seed, args.first_manifest
        )
        verify_disjoint(cohorts)
        all_metadata = []
        for cohort in cohorts:
            meta = run_cohort(cohort, args.mastery, output_dir)
            all_metadata.append(meta)

        print(f"\n{'='*90}")
        print(f"{'Run':<8} {'Source(M0A)':<16} {'M1 cov':<10} {'M1 pat':<8} "
              f"{'M2 gr':<8} {'B900 base':<18} {'B900 final':<18}")
        print(f"{'':<8} {'valid/E':<16} {'cov/tot':<10} {'cnt':<8} "
              f"{'cnt':<8} {'E/F/A':<18} {'E/F/A':<18}")
        print(f"{'-'*90}")
        for meta in all_metadata:
            cid = meta["cohort_id"]
            sg = meta.get("source_gate", {})
            tb = meta.get("transfer_base") or {}
            tf = meta.get("transfer_final") or {}
            valid = sg.get("valid", "?")
            total = sg.get("total", "?")
            m1_tasks = meta.get("m1_tasks_covered", "?")
            m1_pat = meta.get("m1_patterns", "?")
            m2g = meta.get("m2_graphs", "?")
            src = f"{valid}/{total}"
            cov = f"{m1_tasks}/{total}"
            base = f"{tb.get('exact','?')}/{tb.get('false','?')}/{tb.get('abstain','?')}"
            fin = f"{tf.get('exact_final','?')}/{tf.get('false_final','?')}/{tf.get('abstain_final','?')}"
            print(f"{cid:<8} {src:<16} {cov:<10} {str(m1_pat):<8} "
                  f"{str(m2g):<8} {base:<18} {fin:<18}")
        print(f"{'='*90}")


if __name__ == "__main__":
    main()
