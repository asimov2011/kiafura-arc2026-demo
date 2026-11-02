"""A2 passo 1a: baseline source-side della coorte A2 con l'unione A100.

Misura quanto del lavoro A2 e' gia' coperto dal vocabolario A100
(CT-2 + SSA + 19 famiglie), senza alcuna memoria nuova.

Denominatore: 100 (task a2_ids in splits/ct-2-random-a2-v1.json).

Uso:
    PYTHONPATH=.:src PYTHONHASHSEED=0 python experiments/a2_baseline.py
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kiafura_official.reasoning.operational_dsd import resolve  # noqa: E402

MEM = ROOT / "memoria"
SPLITS = ROOT / "splits"
ARC = ROOT / "data" / "arc-agi-2-training"
FAMILY_BUDGET = 2_000_000

FAMILY_FILES = {
    "count": "ct-2-star-count-render-algebra-v1.json",
    "component": "ct-2-star-component-relation-algebra-v1.json",
    "selection": "ct-2-star-selection-extraction-algebra-v1.json",
    "region": "ct-2-star-region-topology-algebra-v1.json",
    "connection": "ct-2-star-connection-topology-algebra-v1.json",
    "local": "ct-2-star-local-pattern-algebra-v1.json",
    "pattern": "ct-2-star-pattern-recolor-algebra-v1.json",
    "panel": "ct-2-star-panel-composition-algebra-v1.json",
    "geometry": "ct-2-star-geometric-pattern-algebra-v1.json",
    "spatial": "ct-2-star-spatial-transform-algebra-v1.json",
    "ray": "ct-2-star-ray-pattern-algebra-v1.json",
    "frame": "ct-2-star-frame-completion-algebra-v1.json",
    "object": "ct-2-star-object-structure-algebra-v1.json",
    "hierarchical": "ct-2-star-hierarchical-panel-algebra-v1.json",
    "advanced": "ct-2-star-advanced-spatial-algebra-v1.json",
    "repair": "ct-2-star-relational-repair-algebra-v1.json",
    "analogue": "ct-2-star-analogue-repair-algebra-v1.json",
    "conditioned": "ct-2-star-conditioned-singleton-algebra-v1.json",
    "structural": "ct-2-star-structural-induction-algebra-v1.json",
}


def as_grid(value):
    return tuple(tuple(int(c) for c in row) for row in value)


def load_task(tid):
    task = json.loads((ARC / f"{tid}.json").read_text(encoding="utf-8"))
    demos = [(as_grid(p["input"]), as_grid(p["output"]))
             for p in task["train"]]
    queries = [as_grid(p["input"]) for p in task["test"]]
    expected = tuple(as_grid(p["output"]) for p in task["test"])
    return demos, queries, expected


def build_memories():
    memories = {
        "ct2": json.loads((MEM / "ct-2-operational-memory-v1.json")
                          .read_text()),
        "ssa": json.loads((MEM / "ct6-causal-ssa-memory-v2.json")
                          .read_text()),
    }
    for key, fname in FAMILY_FILES.items():
        memories[key] = json.loads((MEM / fname).read_text())
    return memories


def main() -> int:
    split = json.loads((SPLITS / "ct-2-random-a2-v1.json").read_text())
    a_ids = list(split["a2_ids"])
    memories = build_memories()

    def run(tid):
        demos, queries, expected = load_task(tid)
        res = resolve(demos, queries, memories, FAMILY_BUDGET)
        if res.get("outputs") is None:
            return "abstain", [], res.get("status")
        ok = tuple(res["outputs"]) == expected
        return ("exact" if ok else "false"), res.get("sources", []), res.get("status")

    print("[A2 baseline] unione A100 sui 100 task A2 (denominatore 100)")
    counts = Counter()
    sources = Counter()
    uncovered = []
    false_ids = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(run, a_ids))
    for tid, (verdict, srcs, status) in zip(a_ids, results):
        counts[verdict] += 1
        if verdict == "exact":
            sources["+".join(sorted(srcs)) or "none"] += 1
        elif verdict == "false":
            false_ids.append((tid, sorted(srcs)))
        else:
            uncovered.append((tid, status))
    print(f"    esito: {dict(sorted(counts.items()))}")
    print(f"    sorgenti degli exact: {dict(sorted(sources.items()))}")
    print(f"    falsi ({len(false_ids)}): {false_ids}")
    print(f"    scoperti ({len(uncovered)}):")
    for tid, status in uncovered:
        print(f"        {tid}  {status}")
    out = ROOT / "experiments" / "a2_baseline_result.json"
    out.write_text(json.dumps({
        "denominator": 100,
        "counts": dict(counts),
        "sources": dict(sources),
        "uncovered": uncovered,
        "false": false_ids,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"    dettaglio scritto in {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
