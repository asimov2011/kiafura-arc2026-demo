"""Region-first composition of frozen operational memories."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from kiafura_official.arc_perception import as_grid
from kiafura_official.reasoning.advanced_spatial_algebra import solve as solve_advanced
from kiafura_official.reasoning.component_relation_algebra import solve as solve_component
from kiafura_official.reasoning.connection_topology_algebra import solve as solve_connection
from kiafura_official.reasoning.conditioned_singleton_algebra import solve as solve_conditioned
from kiafura_official.reasoning.count_render_algebra import solve as solve_count
from kiafura_official.reasoning.ct2_operational import resolve as solve_ct2
from kiafura_official.reasoning.ct6_causal_ssa import resolve as solve_ssa
from kiafura_official.reasoning.local_pattern_algebra import solve as solve_local
from kiafura_official.reasoning.object_structure_algebra import solve as solve_object
from kiafura_official.reasoning.pattern_recolor_algebra import solve as solve_pattern
from kiafura_official.reasoning.panel_composition_algebra import solve as solve_panel
from kiafura_official.reasoning.geometric_pattern_algebra import solve as solve_geometry
from kiafura_official.reasoning.frame_completion_algebra import solve as solve_frame
from kiafura_official.reasoning.hierarchical_panel_algebra import solve as solve_hierarchical
from kiafura_official.reasoning.region_topology_algebra import solve as solve_region
from kiafura_official.reasoning.ray_pattern_algebra import solve as solve_ray
from kiafura_official.reasoning.relational_repair_algebra import solve as solve_repair
from kiafura_official.reasoning.selection_extraction_algebra import solve as solve_selection
from kiafura_official.reasoning.spatial_transform_algebra import solve as solve_spatial
from kiafura_official.reasoning.structural_induction_algebra import solve as solve_structural


Grid = tuple[tuple[int, ...], ...]

FAMILY_SOLVERS = (
    ("count", "count_render", solve_count),
    ("component", "component_relation", solve_component),
    ("selection", "selection_extraction", solve_selection),
    ("region", "region_topology", solve_region),
    ("connection", "connection_topology", solve_connection),
    ("local", "local_pattern", solve_local),
    ("pattern", "pattern_recolor", solve_pattern),
    ("panel", "panel_composition", solve_panel),
    ("geometry", "geometric_pattern", solve_geometry),
    ("spatial", "spatial_transform", solve_spatial),
    ("ray", "ray_pattern", solve_ray),
    ("frame", "frame_completion", solve_frame),
    ("object", "object_structure", solve_object),
    ("hierarchical", "hierarchical_panel", solve_hierarchical),
    ("advanced", "advanced_spatial", solve_advanced),
    ("repair", "relational_repair", solve_repair),
    ("analogue", "analogue_repair", solve_repair),
    ("conditioned", "conditioned_singleton", solve_conditioned),
    ("structural", "structural_induction", solve_structural),
)


def resolve(
    demonstrations: Sequence[tuple[Grid, Grid]],
    queries: Sequence[Grid],
    memories: Mapping[str, Any],
    budget: int,
) -> dict[str, Any]:
    ct2 = solve_ct2(
        demonstrations,
        queries,
        tuple(tool["tool_id"] for tool in memories["ct2"]["tools"]),
        int(memories["ct2"]["budget"]["work_per_task"]),
    )
    ssa = solve_ssa(demonstrations, queries, memories["ssa"])
    sources: list[tuple[str, Any]] = [("ct2", ct2.outputs), ("ssa", ssa.outputs)]
    work: dict[str, Any] = {"ct2": ct2.work, "ssa": ssa.work.as_dict()}
    for key, name, solver in FAMILY_SOLVERS:
        if key not in memories:
            continue
        result = solver(
            demonstrations,
            queries,
            budget=budget,
            allowed_graphs=memories[key]["graphs"],
        )
        sources.append((name, result.get("outputs")))
        work[name] = result["work"]
    visible = [(name, tuple(outputs)) for name, outputs in sources if outputs is not None]
    regions = {outputs for _, outputs in visible}
    outputs = next(iter(regions)) if len(regions) == 1 else None
    return {
        "outputs": outputs,
        "sources": [name for name, _ in visible],
        "status": "verified" if outputs is not None else "abstain_conflict" if len(regions) > 1 else "abstain",
        "work": work,
    }


def case_from_json(case: Mapping[str, Any]) -> tuple[tuple[tuple[Grid, Grid], ...], tuple[Grid, ...]]:
    if set(case) != {"demonstrations", "queries"}:
        raise ValueError("forbidden runtime fixture field")
    demonstrations = tuple((as_grid(pair["input"]), as_grid(pair["output"])) for pair in case["demonstrations"])
    queries = tuple(as_grid(value) for value in case["queries"])
    return demonstrations, queries
