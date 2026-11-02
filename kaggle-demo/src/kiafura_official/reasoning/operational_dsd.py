"""Verified operational union with DSD-controlled family search.

DSD changes the order in which bindings are tested.  The region proof remains
complete: after an exact member is found, every still-unobserved candidate is
checked before an output can be emitted.  This keeps the verified-only
contract while making the live control loop explicit and measurable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import importlib
from typing import Any, Mapping, Sequence

from kiafura_official.arc_perception import as_grid
from kiafura_official.reasoning.causal_residual import CausalResidual, causal_residual
from kiafura_official.reasoning.ct2_operational import resolve as solve_ct2
from kiafura_official.reasoning.ct6_causal_ssa import resolve as solve_ssa
from kiafura_official.reasoning.dsd_runtime import (
    DsdCandidate,
    MinimalSupportPolicy,
    run_dsd,
)


Grid = tuple[tuple[int, ...], ...]
EXECUTION_ERRORS = (
    IndexError,
    KeyError,
    RuntimeError,
    StopIteration,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class FamilySpec:
    key: str
    name: str
    module: str
    selector: str = "region_consensus"
    candidate_module: str | None = None
    candidate_function: str = "candidates"


FAMILY_SPECS = (
    FamilySpec(
        "count",
        "count_render",
        "count_render_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="count_render_candidates",
    ),
    FamilySpec(
        "component",
        "component_relation",
        "component_relation_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="component_relation_candidates",
    ),
    FamilySpec(
        "selection",
        "selection_extraction",
        "selection_extraction_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="selection_extraction_candidates",
    ),
    FamilySpec(
        "region",
        "region_topology",
        "region_topology_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="region_topology_candidates",
    ),
    FamilySpec(
        "connection",
        "connection_topology",
        "connection_topology_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="connection_topology_candidates",
    ),
    FamilySpec(
        "local",
        "local_pattern",
        "local_pattern_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="local_pattern_candidates",
    ),
    FamilySpec(
        "pattern",
        "pattern_recolor",
        "pattern_recolor_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="pattern_recolor_candidates",
    ),
    FamilySpec(
        "panel",
        "panel_composition",
        "panel_composition_algebra",
        candidate_module="panel_composition_witness",
    ),
    FamilySpec(
        "geometry",
        "geometric_pattern",
        "geometric_pattern_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="geometric_pattern_candidates",
    ),
    FamilySpec(
        "spatial",
        "spatial_transform",
        "spatial_transform_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="spatial_transform_candidates",
    ),
    FamilySpec(
        "ray",
        "ray_pattern",
        "ray_pattern_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="ray_pattern_candidates",
    ),
    FamilySpec("frame", "frame_completion", "frame_completion_algebra"),
    FamilySpec(
        "object",
        "object_structure",
        "object_structure_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="object_structure_candidates",
    ),
    FamilySpec(
        "hierarchical",
        "hierarchical_panel",
        "hierarchical_panel_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="hierarchical_panel_candidates",
    ),
    FamilySpec(
        "advanced",
        "advanced_spatial",
        "advanced_spatial_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="advanced_spatial_candidates",
    ),
    FamilySpec(
        "repair",
        "relational_repair",
        "relational_repair_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="relational_repair_candidates",
    ),
    FamilySpec(
        "analogue",
        "analogue_repair",
        "relational_repair_algebra",
        candidate_module="inverse_witnesses",
        candidate_function="relational_repair_candidates",
    ),
    FamilySpec("conditioned", "conditioned_singleton", "conditioned_singleton_algebra"),
    FamilySpec(
        "structural",
        "structural_induction",
        "structural_induction_algebra",
        "unique_minimum_description_region",
    ),
    FamilySpec("a1_g1", "a1_g1_family", "a1_g1_algebra"),
    FamilySpec("a1_g2", "a1_g2_family", "a1_g2_algebra"),
    FamilySpec("a1_g4", "a1_g4_family", "a1_g4_algebra"),
    FamilySpec("a1_g5", "a1_g5_family", "a1_g5_algebra"),
    FamilySpec("a1_g6", "a1_g6_family", "a1_g6_algebra"),
    FamilySpec("a1_g7", "a1_g7_family", "a1_g7_algebra"),
    FamilySpec("a1_g8", "a1_g8_family", "a1_g8_algebra"),
    FamilySpec("a300_g1", "a300_g1_family", "a1_g1_algebra"),
    FamilySpec("a300_g2", "a300_g2_family", "a1_g2_algebra"),
    FamilySpec("a300_g4", "a300_g4_family", "a1_g4_algebra"),
    FamilySpec("a300_g5", "a300_g5_family", "a1_g5_algebra"),
    FamilySpec("a300_g6", "a300_g6_family", "a1_g6_algebra"),
    FamilySpec("a300_g7", "a300_g7_family", "a1_g7_algebra"),
    FamilySpec("a300_g8", "a300_g8_family", "a1_g8_algebra"),
    FamilySpec("a300_residual", "a300_residual", "a300_residual_algebra"),
    FamilySpec("a2_move", "a2_move_gravity", "a2_move_gravity_algebra"),
    FamilySpec("a2_color", "a2_color_permutation", "a2_color_permutation_algebra"),
    FamilySpec("a2_half", "a2_half_panel", "a2_half_panel_algebra"),
    FamilySpec("a2_canvas", "a2_classifier_canvas", "a2_classifier_canvas_algebra"),
    FamilySpec("a2_extract", "a2_extract_subgrid", "a2_extract_subgrid_algebra"),
    FamilySpec("a2_paint", "a2_paint_recolor", "a2_paint_recolor_algebra"),
    FamilySpec("a2_rays", "a2_rays_neighbors", "a2_rays_neighbors_algebra"),
    FamilySpec("a2_scale", "a2_scale_period", "a2_scale_period_algebra"),
    FamilySpec("a2_frames", "a2_frames_canvas", "a2_frames_canvas_algebra"),
    FamilySpec("a2_connections", "a2_connections", "a2_connections_algebra"),
)


def _module(spec: FamilySpec):
    return importlib.import_module(
        f"kiafura_official.reasoning.{spec.module}"
    )


def _candidate_module(spec: FamilySpec):
    if spec.candidate_module is None:
        return _module(spec)
    return importlib.import_module(
        f"kiafura_official.reasoning.{spec.candidate_module}"
    )


def _candidates(spec: FamilySpec, demonstrations: Sequence[tuple[Grid, Grid]]):
    return getattr(_candidate_module(spec), spec.candidate_function)(demonstrations)


def _demo_cells(demonstrations: Sequence[tuple[Grid, Grid]]) -> int:
    return sum(
        len(source) * len(source[0]) + len(expected) * len(expected[0])
        for source, expected in demonstrations
    )


def _candidate_cost(candidate: Any, demonstrations: Sequence[tuple[Grid, Grid]]) -> int:
    if hasattr(candidate, "cost"):
        return sum(
            int(candidate.cost(source)) + int(candidate.cost(expected))
            for source, expected in demonstrations
        )
    return _demo_cells(demonstrations)


def _observe(
    candidate: Any,
    demonstrations: Sequence[tuple[Grid, Grid]],
    expected: tuple[Grid, ...],
) -> CausalResidual:
    outputs: list[Grid | None] = []
    errors: list[str | None] = []
    for source, _ in demonstrations:
        try:
            outputs.append(candidate.execute(source))
            errors.append(None)
        except EXECUTION_ERRORS as error:
            outputs.append(None)
            errors.append(type(error).__name__)
    return causal_residual(tuple(outputs), expected, tuple(errors))


def _query(candidate: Any, queries: Sequence[Grid]) -> tuple[Grid, ...] | None:
    try:
        return tuple(candidate.execute(query) for query in queries)
    except EXECUTION_ERRORS:
        return None


def _solve_structural(
    spec: FamilySpec,
    demonstrations: Sequence[tuple[Grid, Grid]],
    queries: Sequence[Grid],
    *,
    budget: int,
    allowed_graphs: Sequence[str],
) -> dict[str, Any]:
    result = _module(spec).solve(
        demonstrations,
        queries,
        budget=budget,
        allowed_graphs=allowed_graphs,
    )
    return {
        **result,
        "control": {
            "function": "DSD",
            "policy": "delegate_exact_candidate_generation_then_lambda",
            "selector": spec.selector,
            "verification_complete": True,
        },
    }


def solve_family(
    spec: FamilySpec,
    demonstrations: Sequence[tuple[Grid, Grid]],
    queries: Sequence[Grid],
    *,
    budget: int,
    allowed_graphs: Sequence[str],
) -> dict[str, Any]:
    """Search one family using DSD, then complete the region proof."""

    if spec.selector != "region_consensus":
        return _solve_structural(
            spec,
            demonstrations,
            queries,
            budget=budget,
            allowed_graphs=allowed_graphs,
        )
    allowed = set(allowed_graphs)
    if not allowed:
        return {
            "status": "abstain",
            "outputs": None,
            "work": 0,
            "control": {
                "function": "DSD",
                "policy": "minimal_support_complete_v2",
                "verification_complete": True,
            },
        }
    pool = tuple(
        candidate
        for candidate in _candidates(spec, demonstrations)
        if candidate.graph in allowed
    )
    costs = tuple(_candidate_cost(candidate, demonstrations) for candidate in pool)
    total_work = sum(costs)
    if total_work > budget:
        return {
            "status": "abstain_budget",
            "outputs": None,
            "work": total_work,
            "control": {
                "function": "DSD",
                "policy": "minimal_support_complete_v2",
                "verification_complete": False,
            },
        }
    expected = tuple(target for _, target in demonstrations)
    residuals: dict[int, CausalResidual] = {}
    by_graph: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(pool):
        by_graph[candidate.graph].append(index)

    dsd_evaluations = 0
    dsd_work = 0
    dsd_exact_graphs = 0
    traces = 0
    for graph in sorted(by_graph):
        indices = sorted(by_graph[graph], key=lambda index: repr(pool[index].bindings))
        states = tuple(
            DsdCandidate(str(index), tuple(pool[index].bindings)) for index in indices
        )

        def evaluate(state: DsdCandidate) -> CausalResidual:
            index = int(state.candidate_id)
            if index not in residuals:
                residuals[index] = _observe(pool[index], demonstrations, expected)
            return residuals[index]

        outcome = run_dsd(
            states,
            evaluate,
            budget=sum(costs[index] for index in indices),
            cost=lambda state: costs[int(state.candidate_id)],
            policy=MinimalSupportPolicy(
                maximum_support=max(
                    1,
                    max(len(state.coordinates) for state in states),
                )
            ),
        )
        dsd_evaluations += outcome.evaluations
        dsd_work += outcome.work
        dsd_exact_graphs += int(outcome.exact)
        traces += len(outcome.trace)

    # Verified-only completion: DSD may find a valid path, but it is not a
    # proof that an unobserved exact member cannot yield another query region.
    # Materialize the remainder once, reusing every DSD observation.
    for index, candidate in enumerate(pool):
        if index not in residuals:
            residuals[index] = _observe(candidate, demonstrations, expected)
    exact = [pool[index] for index, residual in residuals.items() if residual.exact]
    regions: dict[tuple[Grid, ...], list[Any]] = defaultdict(list)
    for candidate in exact:
        outputs = _query(candidate, queries)
        if outputs is not None:
            regions[outputs].append(candidate)
    control = {
        "candidate_count": len(pool),
        "dsd_evaluations": dsd_evaluations,
        "dsd_exact_graphs": dsd_exact_graphs,
        "dsd_work": dsd_work,
        "function": "DSD",
        "policy": "minimal_support_complete_v2",
        "selector": spec.selector,
        "trace_steps": traces,
        "verification_complete": len(residuals) == len(pool),
        "verification_evaluations": len(residuals),
        "verification_work": total_work,
    }
    if len(regions) != 1:
        return {
            "status": "abstain_no_consensus",
            "outputs": None,
            "work": total_work,
            "regions": len(regions),
            "control": control,
        }
    outputs, members = next(iter(regions.items()))
    return {
        "status": "verified",
        "outputs": outputs,
        "work": total_work,
        "regions": 1,
        "graphs": sorted({member.graph for member in members}),
        "control": control,
    }


def resolve(
    demonstrations: Sequence[tuple[Grid, Grid]],
    queries: Sequence[Grid],
    memories: Mapping[str, Any],
    budget: int,
) -> dict[str, Any]:
    """Run the frozen union with DSD as its explicit live controller."""

    ct2 = solve_ct2(
        demonstrations,
        queries,
        tuple(tool["tool_id"] for tool in memories["ct2"]["tools"]),
        int(memories["ct2"]["budget"]["work_per_task"]),
    )
    ssa = solve_ssa(demonstrations, queries, memories["ssa"])
    sources: list[tuple[str, Any]] = [("ct2", ct2.outputs), ("ssa", ssa.outputs)]
    work: dict[str, Any] = {"ct2": ct2.work, "ssa": ssa.work.as_dict()}
    control: dict[str, Any] = {}
    for spec in FAMILY_SPECS:
        if spec.key not in memories:
            continue
        result = solve_family(
            spec,
            demonstrations,
            queries,
            budget=budget,
            allowed_graphs=memories[spec.key]["graphs"],
        )
        sources.append((spec.name, result.get("outputs")))
        work[spec.name] = result["work"]
        control[spec.name] = result.get("control", {})
    visible = [(name, tuple(outputs)) for name, outputs in sources if outputs is not None]
    regions = {outputs for _, outputs in visible}
    outputs = next(iter(regions)) if len(regions) == 1 else None
    return {
        "outputs": outputs,
        "sources": [name for name, _ in visible],
        "status": (
            "verified"
            if outputs is not None
            else "abstain_conflict"
            if len(regions) > 1
            else "abstain"
        ),
        "work": work,
        "control": control,
    }


def case_from_json(
    case: Mapping[str, Any],
) -> tuple[tuple[tuple[Grid, Grid], ...], tuple[Grid, ...]]:
    if set(case) != {"demonstrations", "queries"}:
        raise ValueError("forbidden runtime fixture field")
    demonstrations = tuple(
        (as_grid(pair["input"]), as_grid(pair["output"]))
        for pair in case["demonstrations"]
    )
    queries = tuple(as_grid(value) for value in case["queries"])
    return demonstrations, queries
