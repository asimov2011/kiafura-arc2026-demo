"""A2 passo 4: valutazione source-side dell'unione A2.

Unione solo-A2 (coorti disgiunte): CT-2 A2 + SSA A2 + famiglie A2.
Denominatore: 100 (a2_ids). Gate: 100 esatti / 0 falsi / 0 astensioni.

Uso:
    PYTHONPATH=.:src PYTHONHASHSEED=0 python experiments/a2_union_eval.py
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

A2_FAMILY_FILES = {
    "a2_move": "a2/ct-2-star-a2-move-gravity-algebra-v1.json",
    "a2_color": "a2/ct-2-star-a2-color-permutation-algebra-v1.json",
    "a2_half": "a2/ct-2-star-a2-half-panel-algebra-v1.json",
    "a2_canvas": "a2/ct-2-star-a2-classifier-canvas-algebra-v1.json",
    "a2_extract": "a2/ct-2-star-a2-extract-subgrid-algebra-v1.json",
    "a2_paint": "a2/ct-2-star-a2-paint-recolor-algebra-v1.json",
    "a2_rays": "a2/ct-2-star-a2-rays-neighbors-algebra-v1.json",
    "a2_scale": "a2/ct-2-star-a2-scale-period-algebra-v1.json",
    "a2_frames": "a2/ct-2-star-a2-frames-canvas-algebra-v1.json",
    "a2_connections": "a2/ct-2-star-a2-connections-algebra-v1.json",
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
        "ct2": json.loads((MEM / "a2" / "ct-2-a2-operational-memory-v1.json")
                          .read_text()),
        "ssa": json.loads((MEM / "a2" / "ct-2-a2-causal-ssa-memory-v1.json")
                          .read_text()),
    }
    for key, fname in A2_FAMILY_FILES.items():
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

    print("[A2 union] source-side (denominatore 100)")
    counts = Counter()
    sources = Counter()
    problems = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(run, a_ids))
    for tid, (verdict, srcs, status) in zip(a_ids, results):
        counts[verdict] += 1
        if verdict == "exact":
            sources["+".join(sorted(srcs)) or "none"] += 1
        else:
            problems.append((tid, verdict, status, sorted(srcs)))
    print(f"    esito: {dict(sorted(counts.items()))}")
    print(f"    sorgenti degli exact: {dict(sorted(sources.items()))}")
    print(f"    non esatti ({len(problems)}):")
    for item in problems:
        print(f"        {item}")
    gate = (counts.get("exact") == 100 and counts.get("false", 0) == 0
            and counts.get("abstain", 0) == 0)
    print("    gate source-side 100/100:", "PASS" if gate else "FAIL")
    out = ROOT / "experiments" / "a2_union_result.json"
    out.write_text(json.dumps({
        "denominator": 100,
        "counts": dict(counts),
        "sources": dict(sources),
        "problems": problems,
    }, indent=1) + "\n", encoding="utf-8")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
