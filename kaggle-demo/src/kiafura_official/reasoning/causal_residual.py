"""Exact and structural residuals for DSD over ARC demonstration behavior."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence


Grid = tuple[tuple[int, ...], ...]


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _geometry(points: set[tuple[int, int]], height: int, width: int) -> str:
    if not points:
        return "exact"
    if len(points) == 1:
        return "point"
    rows = {row for row, _ in points}
    columns = {column for _, column in points}
    if len(rows) == 1:
        return "row"
    if len(columns) == 1:
        return "column"
    top, bottom = min(rows), max(rows)
    left, right = min(columns), max(columns)
    area = (bottom - top + 1) * (right - left + 1)
    if len(points) == area:
        return "rectangle"
    if len(points) == height * width:
        return "full_grid"
    if all(
        row in {top, bottom} or column in {left, right}
        for row, column in points
    ):
        return "border"
    return "scattered"


@dataclass(frozen=True)
class DemoResidual:
    exception: str | None
    excess_roles: int
    expected_cells: int
    geometry: str
    height_delta: int
    mismatch_cells: int
    missing_roles: int
    width_delta: int

    @property
    def exact(self) -> bool:
        return (
            self.exception is None
            and self.height_delta == self.width_delta == self.mismatch_cells == 0
        )


@dataclass(frozen=True)
class CausalResidual:
    demos: tuple[DemoResidual, ...]

    @property
    def exact(self) -> bool:
        return all(item.exact for item in self.demos)

    @property
    def order_key(self) -> tuple[Any, ...]:
        """Explicit lexicographic order; no weighted score."""

        return (
            sum(item.exception is not None for item in self.demos),
            sum(item.height_delta != 0 or item.width_delta != 0 for item in self.demos),
            sum(abs(item.height_delta) + abs(item.width_delta) for item in self.demos),
            sum(item.mismatch_cells for item in self.demos),
            sum(item.missing_roles + item.excess_roles for item in self.demos),
        )

    @property
    def discriminant(self) -> tuple[Any, ...]:
        """Task-normalized observation used by replaceable DSD policies."""

        geometries = Counter(item.geometry for item in self.demos)
        shape_signs = Counter(
            (_sign(item.height_delta), _sign(item.width_delta)) for item in self.demos
        )
        palette = Counter(
            (_sign(item.missing_roles), _sign(item.excess_roles))
            for item in self.demos
        )
        exceptions = Counter(item.exception or "none" for item in self.demos)
        mismatches = sum(item.mismatch_cells for item in self.demos)
        cells = sum(item.expected_cells for item in self.demos)
        ratio = Fraction(mismatches, cells) if cells else Fraction(0, 1)
        return (
            tuple(sorted(exceptions.items())),
            tuple(sorted(shape_signs.items())),
            tuple(sorted(geometries.items())),
            tuple(sorted(palette.items())),
            (ratio.numerator, ratio.denominator),
        )


def demo_residual(observed: Grid | None, expected: Grid, exception: str | None = None) -> DemoResidual:
    expected_height = len(expected)
    expected_width = len(expected[0]) if expected else 0
    if observed is None:
        return DemoResidual(
            exception=exception or "execution_error",
            excess_roles=0,
            expected_cells=expected_height * expected_width,
            geometry="unavailable",
            height_delta=0,
            mismatch_cells=expected_height * expected_width,
            missing_roles=0,
            width_delta=0,
        )
    observed_height = len(observed)
    observed_width = len(observed[0]) if observed else 0
    height = min(observed_height, expected_height)
    width = min(observed_width, expected_width)
    mismatches = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if observed[row][column] != expected[row][column]
    }
    missing_extent = abs(observed_height * observed_width - expected_height * expected_width)
    expected_palette = {value for row in expected for value in row}
    observed_palette = {value for row in observed for value in row}
    shape_matches = observed_height == expected_height and observed_width == expected_width
    return DemoResidual(
        exception=exception,
        excess_roles=len(observed_palette - expected_palette),
        expected_cells=expected_height * expected_width,
        geometry=(
            _geometry(mismatches, expected_height, expected_width)
            if shape_matches
            else "shape"
        ),
        height_delta=observed_height - expected_height,
        mismatch_cells=len(mismatches) + missing_extent,
        missing_roles=len(expected_palette - observed_palette),
        width_delta=observed_width - expected_width,
    )


def causal_residual(
    observed: Sequence[Grid | None],
    expected: Sequence[Grid],
    exceptions: Sequence[str | None] | None = None,
) -> CausalResidual:
    if len(observed) != len(expected):
        raise ValueError("observed/expected demonstration count mismatch")
    errors = exceptions or (None,) * len(observed)
    if len(errors) != len(observed):
        raise ValueError("exception count mismatch")
    return CausalResidual(
        tuple(
            demo_residual(output, target, error)
            for output, target, error in zip(observed, expected, errors, strict=True)
        )
    )
