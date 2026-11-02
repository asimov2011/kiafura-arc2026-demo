"""Policy-independent DSD loop and a replaceable minimal-support policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from kiafura_official.reasoning.causal_residual import CausalResidual


@dataclass(frozen=True)
class DsdCandidate:
    candidate_id: str
    coordinates: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class DsdStep:
    coordinates: tuple[str, ...]
    from_residual: tuple[Any, ...]
    support: int
    to_residual: tuple[Any, ...]


@dataclass(frozen=True)
class DsdOutcome:
    candidate_id: str
    exact: bool
    evaluations: int
    residual: CausalResidual
    status: str
    trace: tuple[DsdStep, ...]
    work: int


class DsdPolicy(Protocol):
    def batches(
        self,
        current: int,
        candidates: Sequence[DsdCandidate],
    ) -> Sequence[tuple[int, tuple[int, ...]]]: ...


@dataclass(frozen=True)
class MinimalSupportPolicy:
    """Probe one coordinate first, then the smallest joint support required."""

    maximum_support: int = 3

    def batches(
        self,
        current: int,
        candidates: Sequence[DsdCandidate],
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        current_values = dict(candidates[current].coordinates)
        output = []
        maximum = min(self.maximum_support, len(current_values))
        for support in range(1, maximum + 1):
            neighbors = []
            for index, candidate in enumerate(candidates):
                if index == current:
                    continue
                values = dict(candidate.coordinates)
                distance = sum(
                    current_values.get(name) != values.get(name)
                    for name in set(current_values) | set(values)
                )
                if distance == support:
                    neighbors.append(index)
            if neighbors:
                output.append((support, tuple(neighbors)))
        return tuple(output)


def run_dsd(
    candidates: Sequence[DsdCandidate],
    evaluate: Callable[[DsdCandidate], CausalResidual],
    *,
    budget: int,
    cost: Callable[[DsdCandidate], int] | None = None,
    policy: DsdPolicy,
) -> DsdOutcome:
    """Execute feedback until exact, local exhaustion or budget exhaustion."""

    if not candidates:
        raise ValueError("DSD candidate space is empty")
    if budget < 1:
        raise ValueError("DSD budget must admit the initial evaluation")
    cache: dict[int, CausalResidual] = {}
    spent = 0
    candidate_cost = cost or (lambda candidate: 1)

    def observe(index: int) -> CausalResidual:
        nonlocal spent
        if index not in cache:
            charge = int(candidate_cost(candidates[index]))
            if charge < 1:
                raise ValueError("DSD candidate cost must be positive")
            if spent + charge > budget:
                raise RuntimeError("budget_exhausted")
            spent += charge
            cache[index] = evaluate(candidates[index])
        return cache[index]

    current = 0
    residual = observe(current)
    trace = []
    while not residual.exact:
        moved = False
        for support, indices in policy.batches(current, candidates):
            alternatives = []
            try:
                for index in indices:
                    observed = observe(index)
                    if observed.order_key < residual.order_key:
                        alternatives.append(
                            (
                                observed.order_key,
                                repr(candidates[index].coordinates),
                                index,
                                observed,
                            )
                        )
            except RuntimeError as error:
                if str(error) != "budget_exhausted":
                    raise
                return DsdOutcome(
                    candidate_id=candidates[current].candidate_id,
                    exact=False,
                    evaluations=len(cache),
                    residual=residual,
                    status="budget_exhausted",
                    trace=tuple(trace),
                    work=spent,
                )
            if not alternatives:
                continue
            _, _, next_index, next_residual = min(alternatives)
            current_values = dict(candidates[current].coordinates)
            next_values = dict(candidates[next_index].coordinates)
            selected = tuple(
                sorted(
                    name
                    for name in set(current_values) | set(next_values)
                    if current_values.get(name) != next_values.get(name)
                )
            )
            trace.append(
                DsdStep(
                    coordinates=selected,
                    from_residual=residual.order_key,
                    support=support,
                    to_residual=next_residual.order_key,
                )
            )
            current = next_index
            residual = next_residual
            moved = True
            break
        if not moved:
            break
    return DsdOutcome(
        candidate_id=candidates[current].candidate_id,
        exact=residual.exact,
        evaluations=len(cache),
        residual=residual,
        status="exact" if residual.exact else "local_exhaustion",
        trace=tuple(trace),
        work=spent,
    )
