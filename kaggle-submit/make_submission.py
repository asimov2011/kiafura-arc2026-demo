#!/usr/bin/env python3
"""Kaggle submission packer — LIGHT GLUE ONLY (additive; no frozen/demo code modified).

Runs the exact demonstration pipeline of `kaggle-demo/driver.py` (five frozen
memories + fail-closed emission gates + two-output contract) over the official
ARC-AGI-2 competition test tasks and emits `submission.json` in the venue
sample-submission format: one entry per test input with attempt_1/attempt_2.

Portable: all paths are CLI flags (defaults fit the local `kaggle-submit/`
layout); the same script runs on Kaggle as a subprocess of the notebook, with
`--package-dir /kaggle/input/<dataset>/kaggle-demo --data-dir <competition dir>`.
Competition filenames are resolved by glob (hyphen/underscore naming varies).

Boundary (execution/PAPER-2026-T1-PLAN.md §5, W5): packaging of existing code
plus light glue; no algorithm development; target-blind (test outputs do not
exist in the challenge file and are never read); the Kaggle score is not an
objective and is never computed here.

Abstention convention: a named abstention (no demo-verified emission) is
emitted as the placeholder [[0, 0], [0, 0]] — the venue's own submission-format
example fill for "a slot with no real prediction" ("Both attempt_1 and attempt_2
must be present, even if your submission doesn't have 2 predictions",
Evaluation page). It is never a grid this pipeline emitted; it can only score 0.

Wall-clock guard: `--time-budget-s` (default 36000 = 10h). When the budget is
exhausted, every remaining task is filled with the placeholder convention —
all task_ids stay present, so the submission stays valid (rules-legal).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PLACEHOLDER = [[0, 0], [0, 0]]  # venue example fill for a slot with no real prediction


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Kaggle submission packer (target-blind, fail-closed)")
    ap.add_argument("--package-dir", default=str(here.parent / "kaggle-demo"),
                    help="path of the kaggle-demo package (contains src/, experiments/, memoria/)")
    ap.add_argument("--data-dir", default=str(here / "data"),
                    help="dir containing the competition challenges + sample submission (glob-resolved)")
    ap.add_argument("--out", default=str(here / "outputs"), help="output dir")
    ap.add_argument("--time-budget-s", type=float, default=36000.0,
                    help="wall-clock budget; unfinished tasks are filled with the placeholder")
    return ap.parse_args()


def resolve_competition_files(data_dir: Path) -> tuple[Path, Path]:
    """Competition filenames vary (arc-agi_test_challenges.json vs arc_agi_*): resolve by glob."""
    chal = sorted(p for p in data_dir.glob("*test*challenges*.json"))
    samp = sorted(data_dir.glob("*sample*submission*.json"))
    assert len(chal) == 1, f"cannot resolve challenges file in {data_dir}: {chal}"
    assert len(samp) == 1, f"cannot resolve sample submission in {data_dir}: {samp}"
    return chal[0], samp[0]


def main() -> int:
    args = parse_args()
    package = Path(args.package_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    out = Path(args.out)
    (out / "_tmp").mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(package))
    sys.path.insert(0, str(package / "src"))
    sys.path.insert(0, str(package / "experiments"))
    import driver as d  # noqa: E402  (demo light glue: loaders, gate, contract formatting)
    from halfway_b1000 import FAMILY_BUDGET, MEMBER_TIMEOUT, RUNS, initialize  # noqa: E402

    chal_path, samp_path = resolve_competition_files(data_dir)
    challenges = json.loads(chal_path.read_text())
    sample = json.loads(samp_path.read_text())
    assert set(sample) == set(challenges), "sample/challenge task-id mismatch"

    initialize()  # frozen five-memory load (memoria/, read-only)

    submission: dict = {}
    counts: dict = {}
    abstained_slots = 0
    total_slots = 0
    placeholder_filled_tasks = 0
    started = time.time()
    deadline = started + args.time_budget_s

    with open(out / "trace.jsonl", "w") as trace, open(out / "predictions.jsonl", "w") as preds:
        for i, tid in enumerate(sorted(challenges), 1):
            t = challenges[tid]
            n_test = len(t["test"])

            if time.time() > deadline:  # wall-clock guard: legal placeholder fill, ids stay present
                placeholder_filled_tasks += 1
                submission[tid] = [{"attempt_1": PLACEHOLDER, "attempt_2": PLACEHOLDER}
                                   for _ in range(n_test)]
                total_slots += n_test
                abstained_slots += n_test
                preds.write(json.dumps({"task_id": tid, "status": "placeholder_time_budget",
                                        "attempts": [[PLACEHOLDER[0], PLACEHOLDER[1]]] * n_test,
                                        "named_abstention": list(range(n_test))}) + "\n")
                preds.flush()
                continue

            tmp = out / "_tmp" / f"{tid}.json"
            tmp.write_text(json.dumps(t))
            demos, queries, n_test = d.load_task_inputs(tmp)  # TARGET-BLIND loader
            r = d.resolve_union_traced(demos, queries)
            dist = d.classes(r["members"], RUNS, n_test)
            attempts = [d.fmt_pair((cs + [None, None])[:2]) for cs in dist]

            slots = []
            for att in attempts:
                total_slots += 1
                a, b = att
                if a is None:
                    abstained_slots += 1
                slots.append({
                    "attempt_1": a if a is not None else PLACEHOLDER,
                    "attempt_2": b if b is not None else PLACEHOLDER,
                })
            assert len(slots) == len(sample[tid]), f"{tid}: slot-count mismatch"
            submission[tid] = slots

            counts[r["status"]] = counts.get(r["status"], 0) + 1
            preds.write(json.dumps({
                "task_id": tid,
                "status": r["status"],
                "attempts": attempts,
                "named_abstention": [j for j, cs in enumerate(dist) if not cs],
            }) + "\n")
            preds.flush()
            trace.write(json.dumps({
                "task_id": tid,
                "status": r["status"],
                "emitting_runs": r["emitting_runs"],
                "n_test_inputs": n_test,
                "members": {n: {"status": m["status"], "elapsed_seconds": m["elapsed_seconds"],
                                "trace": m["trace"]} for n, m in r["members"].items()},
            }) + "\n")
            trace.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(challenges)}] {counts} elapsed={time.time()-started:.0f}s", flush=True)

    assert set(submission) == set(sample), "submission must contain every task_id"
    (out / "submission.json").write_text(json.dumps(submission) + "\n")
    manifest = {
        "label": "DEMONSTRATION OUTPUT, NOT CERTIFIED EVIDENCE; competition score is not an objective",
        "pipeline": "byte-for-byte kaggle-demo/driver.py chain (five frozen memories + fail-closed "
                    "emission gates + two-output contract); light glue = task packing + sample-format emit",
        "target_blind": True,
        "contract": "attempt_1/attempt_2 = first two distinct emission classes (A100,A2,A300,A4,A5 "
                    "order); sole emission duplicated; abstention = placeholder [[0,0],[0,0]]",
        "abstention_placeholder": PLACEHOLDER,
        "challenges_file": chal_path.name,
        "sample_file": samp_path.name,
        "tasks": len(submission),
        "test_input_slots": total_slots,
        "abstained_slots": abstained_slots,
        "placeholder_filled_tasks": placeholder_filled_tasks,
        "status_counts": counts,
        "wall_seconds": round(time.time() - started, 1),
        "time_budget_s": args.time_budget_s,
        "member_timeout_s": MEMBER_TIMEOUT,
        "family_budget": FAMILY_BUDGET,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
