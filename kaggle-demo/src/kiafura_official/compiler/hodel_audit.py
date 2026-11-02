"""Deterministic structural audit for the Hodel -> TGIR M0A gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from schema.typed_graph_ir import OPERATOR_REGISTRY, ParametricGraph, ValueType
from src.kiafura_official.compiler.hodel_parser import (
    HodelParseError,
    HodelValidationError,
    parse_hodel_tokens,
    validate_graph_ssa,
)
from src.kiafura_official.compiler.hodel_runtime import EXECUTOR_REGISTRY


@dataclass
class AuditFailure:
    task_id: str
    error_type: str
    message: str


@dataclass
class HodelAuditReport:
    requested_tasks: int = 0
    mastery_records_found: int = 0
    parsed_tasks: int = 0
    structurally_valid_tasks: int = 0
    total_nodes: int = 0
    total_node_inputs: int = 0
    total_ssa_references: int = 0
    unresolved_ssa_references: int = 0
    registered_operator_nodes: int = 0
    unregistered_operator_nodes: int = 0
    executable_operator_nodes: int = 0
    unavailable_executor_nodes: int = 0
    invoke_nodes: int = 0
    numeric_literal_slots: int = 0
    numeric_literals_typed_color: int = 0
    numeric_literals_typed_int: int = 0
    numeric_literals_unresolved: int = 0
    distinct_operators: int = 0
    distinct_unavailable_executors: int = 0
    operator_occurrences: dict[str, int] = field(default_factory=dict)
    unavailable_executor_occurrences: dict[str, int] = field(default_factory=dict)
    failures: list[AuditFailure] = field(default_factory=list)

    @property
    def structural_gate_passed(self) -> bool:
        return (
            self.mastery_records_found == self.requested_tasks
            and self.parsed_tasks == self.requested_tasks
            and self.structurally_valid_tasks == self.requested_tasks
            and self.unresolved_ssa_references == 0
            and self.unregistered_operator_nodes == 0
        )

    @property
    def executor_gate_passed(self) -> bool:
        return self.unavailable_executor_nodes == 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["structural_gate_passed"] = self.structural_gate_passed
        payload["executor_gate_passed"] = self.executor_gate_passed
        return payload


def _load_task_ids(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if not isinstance(payload, dict):
        raise ValueError("Task manifest must be a JSON list or object")
    for key in ("a2_ids", "task_ids", "ids"):
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    raise ValueError("Could not find a2_ids/task_ids/ids in task manifest")


def _load_mastery(path: str | Path, requested: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = record.get("task_id")
            if task_id in requested and record.get("backend") == "hodel-ssa":
                if task_id in records:
                    raise ValueError(
                        f"Duplicate hodel-ssa mastery record for {task_id} at line {line_number}"
                    )
                records[task_id] = record
    return records


def _audit_graph(graph: ParametricGraph, report: HodelAuditReport) -> None:
    validate_graph_ssa(graph)
    report.structurally_valid_tasks += 1
    report.total_nodes += len(graph.nodes)

    produced = {"INPUT", *(slot.name for slot in graph.parameters)}
    operator_counts = Counter(report.operator_occurrences)
    unavailable_counts = Counter(report.unavailable_executor_occurrences)

    for node in graph.nodes:
        report.total_node_inputs += len(node.inputs)
        for reference in node.inputs:
            if reference.startswith("D"):
                report.total_ssa_references += 1
            if reference not in produced:
                report.unresolved_ssa_references += 1

        if node.operator == "invoke":
            report.invoke_nodes += 1
            report.registered_operator_nodes += 1
        elif node.operator in OPERATOR_REGISTRY:
            report.registered_operator_nodes += 1
            operator_counts[node.operator] += 1
        else:
            report.unregistered_operator_nodes += 1
            operator_counts[node.operator] += 1

        if node.operator == "invoke" or node.operator in EXECUTOR_REGISTRY:
            report.executable_operator_nodes += 1
        else:
            report.unavailable_executor_nodes += 1
            unavailable_counts[node.operator] += 1

        produced.add(node.output_name)

    for slot in graph.parameters:
        if slot.default is None or not isinstance(slot.default.value, int):
            continue
        # bool is an int subclass; exclude it from numeric literal accounting.
        if isinstance(slot.default.value, bool):
            continue
        report.numeric_literal_slots += 1
        if slot.value_type == ValueType.COLOR:
            report.numeric_literals_typed_color += 1
        elif slot.value_type == ValueType.INT:
            report.numeric_literals_typed_int += 1
        elif slot.value_type == ValueType.ANY:
            report.numeric_literals_unresolved += 1

    report.operator_occurrences = dict(sorted(operator_counts.items()))
    report.unavailable_executor_occurrences = dict(sorted(unavailable_counts.items()))


def audit_cohort(
    *,
    task_ids: Iterable[str],
    mastery_path: str | Path,
) -> HodelAuditReport:
    ordered_ids = list(dict.fromkeys(str(task_id) for task_id in task_ids))
    report = HodelAuditReport(requested_tasks=len(ordered_ids))
    records = _load_mastery(mastery_path, set(ordered_ids))
    report.mastery_records_found = len(records)

    for task_id in ordered_ids:
        record = records.get(task_id)
        if record is None:
            report.failures.append(
                AuditFailure(task_id, "MissingMasteryRecord", "No hodel-ssa record found")
            )
            continue
        try:
            graph = parse_hodel_tokens(task_id, record["tokens"], strict_registry=True)
            report.parsed_tasks += 1
            _audit_graph(graph, report)
        except (HodelParseError, HodelValidationError, KeyError, TypeError, ValueError) as exc:
            report.failures.append(
                AuditFailure(task_id, type(exc).__name__, str(exc))
            )

    report.distinct_operators = len(report.operator_occurrences)
    report.distinct_unavailable_executors = len(report.unavailable_executor_occurrences)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON containing a2_ids/task_ids")
    parser.add_argument("--mastery", required=True, help="Hodel mastery JSONL")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    task_ids = _load_task_ids(args.manifest)
    report = audit_cohort(task_ids=task_ids, mastery_path=args.mastery)
    payload = report.to_dict()

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")

    # M0A structural gate is the CLI success condition.  Executor completeness
    # is reported separately because it belongs to the execution milestone.
    return 0 if report.structural_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
