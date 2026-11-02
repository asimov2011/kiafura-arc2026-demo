"""Build disjoint Hodel task cohorts for protocol validation.

Each cohort Ai is a disjoint set of 100 Hodel-valid tasks.
Bi = all 1000 ARC-AGI-2 tasks - Ai  (always 900 tasks).
No cumulative memory across runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CORPUS_ROOT = Path("/root/projects/kiafura/legacy/data/external/ARC-AGI-2/data/training")


def _sha256_seeded_sort(task_ids: list[str], seed: str) -> list[str]:
    return sorted(
        task_ids,
        key=lambda tid: (
            hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest(),
            tid,
        ),
    )


def all_corpus_ids() -> list[str]:
    return sorted(p.stem for p in CORPUS_ROOT.glob("*.json"))


def hodel_task_ids(mastery_path: str | Path) -> list[str]:
    """Return all ARC-AGI-2 task IDs that have a Hodel-ssa mastery record."""
    result: list[str] = []
    with open(mastery_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("backend") == "hodel-ssa":
                tid = rec.get("task_id")
                if tid:
                    result.append(tid)
    corpus = set(all_corpus_ids())
    return sorted(t for t in result if t in corpus)


def build_disjoint_cohorts(
    mastery_path: str | Path,
    cohort_size: int = 100,
    seed: str = "kiafura-protocol-validation-20260806",
    first_manifest: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build disjoint cohorts of Hodel-valid tasks.

    For each cohort:
      - a_ids: the 100 Hodel-valid tasks in Ai (disjoint from other A sets)
      - b_ids: all 1000 corpus tasks - a_ids (always exactly 900)

    Returns list of cohort dicts.
    """
    all_hodel = hodel_task_ids(mastery_path)
    all_ids = all_corpus_ids()
    print(f"Hodel valid tasks in corpus: {len(all_hodel)}")

    used: set[str] = set()
    cohorts: list[dict[str, Any]] = []

    # First cohort: from first_manifest if provided
    if first_manifest:
        manifest = json.loads(Path(first_manifest).read_text())
        a_key = next(k for k in ("a2_ids", "a_ids", "task_ids", "ids") if k in manifest)
        first_a = manifest[a_key]
        if not isinstance(first_a, list):
            raise ValueError(f"Expected list for {a_key} in manifest")
        # Keep only Hodel-valid tasks from the manifest
        first_a = [t for t in first_a if t in all_hodel]
        if len(first_a) < cohort_size:
            raise ValueError(
                f"First manifest has only {len(first_a)} Hodel-valid tasks "
                f"(need {cohort_size})"
            )
        first_a = first_a[:cohort_size]
        used.update(first_a)
        a_set = set(first_a)
        b_ids = sorted(t for t in all_ids if t not in a_set)
        assert len(b_ids) == len(all_ids) - cohort_size
        cohorts.append({
            "cohort_id": "A2",
            "source": str(first_manifest),
            "a_ids": sorted(first_a),
            "b_ids": b_ids,
        })
        print(f"Cohort A2: {len(first_a)} Ai tasks, {len(b_ids)} Bi tasks")

    # Build remaining cohorts from unused Hodel-valid tasks
    remaining = sorted(t for t in all_hodel if t not in used)
    remaining = _sha256_seeded_sort(remaining, seed)

    full_cohorts = len(remaining) // cohort_size
    start_idx = 0
    for i in range(full_cohorts):
        a_ids = remaining[start_idx : start_idx + cohort_size]
        start_idx += cohort_size
        used.update(a_ids)
        a_set = set(a_ids)
        b_ids = sorted(t for t in all_ids if t not in a_set)
        assert len(b_ids) == len(all_ids) - cohort_size
        cohort_id = f"A{3 + i}" if first_manifest else f"A{1 + i}"
        cohorts.append({
            "cohort_id": cohort_id,
            "source": "auto_seeded",
            "a_ids": sorted(a_ids),
            "b_ids": b_ids,
        })
        print(f"Cohort {cohort_id}: {len(a_ids)} Ai tasks, {len(b_ids)} Bi tasks")

    unused = sorted(t for t in all_hodel if t not in used)
    if unused:
        print(f"Remaining unused Hodel tasks: {len(unused)} (skipped, will form incomplete cohort)")

    return cohorts


def verify_disjoint(cohorts: list[dict[str, Any]]) -> None:
    """Verify all cohorts are properly disjoint."""
    all_ids = all_corpus_ids()
    full_set = set(all_ids)

    seen_a: set[str] = set()
    for cohort in cohorts:
        a_set = set(cohort["a_ids"])
        b_set = set(cohort["b_ids"])

        # Disjoint Ai check
        overlap = a_set & seen_a
        if overlap:
            raise ValueError(
                f"Cohort {cohort['cohort_id']} overlaps with previous: {sorted(overlap)[:5]}"
            )
        seen_a.update(a_set)

        # |Ai| = 100
        if len(a_set) != 100:
            raise ValueError(f"Cohort {cohort['cohort_id']} |Ai|={len(a_set)}, expected 100")

        # |Bi| = 900
        if len(b_set) != 900:
            raise ValueError(f"Cohort {cohort['cohort_id']} |Bi|={len(b_set)}, expected 900")

        # Bi = corpus - Ai
        expected_b = full_set - a_set
        if b_set != expected_b:
            missing = expected_b - b_set
            extra = b_set - expected_b
            msg = []
            if missing:
                msg.append(f"missing {len(missing)} tasks")
            if extra:
                msg.append(f"extra {len(extra)} tasks")
            raise ValueError(f"Cohort {cohort['cohort_id']} Bi mismatch: {', '.join(msg)}")

    print(f"All {len(cohorts)} cohorts verified: |Ai|=100, |Bi|=900, disjoint A sets")


def save_manifest(cohort: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(cohort, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cohorts = build_disjoint_cohorts(
        "/root/projects/kiafura/kiafura-a100-repro/data/memory/arc2-mastery-v4.jsonl",
        first_manifest="data/splits/ct-2-random-a2-v1.json",
    )
    verify_disjoint(cohorts)
    print("\nCohort summary:")
    for c in cohorts:
        print(f"  {c['cohort_id']}: A={len(c['a_ids'])} B={len(c['b_ids'])}")
