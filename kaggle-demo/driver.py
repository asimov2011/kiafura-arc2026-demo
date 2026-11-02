#!/usr/bin/env python3
"""Kaggle minimal demonstration driver — LIGHT GLUE ONLY.

Demonstrates the approach: five frozen operational memories + fail-closed
emission gate + trace, emitting predictions-or-named-abstention under the
official two-output contract on public demonstration tasks.

DEMONSTRATION OUTPUT, NOT CERTIFIED EVIDENCE. The competition score is not an
objective and is never computed here: the driver is TARGET-BLIND — it reads
only `train` pairs and `test[].input` from each task file and never reads
`test[].output`.

Boundary (execution/PAPER-2026-T1-W5.md): packaging of existing code plus
light driver glue; no algorithm development; no 488-task residual in the task
set (membership exclusion only); no protected/private evaluation.

Frozen components used as-is (copied from kiafura-agosto26 branch
`work/issue-44-cumulative-pass2`, commit 2fa158980e14cd04456e9f98ba0b6042c0627d60):
- src/kiafura_official/... (inference core + emission gates)
- memoria/ (five frozen memories, read-only)
- experiments/{a2_baseline,a2_union_eval,a300_common,a4_freeze,a5_freeze,
  a4_source_inspect,a5_source_inspect,halfway_b1000}.py (loaders + consensus)

Light glue written for this demo:
- task loading (as_grid over train + test inputs only);
- resolve_union_traced(): control flow and gate semantics identical to
  experiments/halfway_b1000.py:resolve_union (blob e48f8819...), the only
  delta is retaining trace fields (work/control/regions/candidates) that the
  frozen milestone dropped from its records;
- two-output formatting: `classes()` first-two-distinct policy copied verbatim
  from experiments/cumulative_pass2.py (blob 804daafc...) and the sole-emission
  duplication rule of experiments/a100_a2_pass2.py:attempt_pairs (blob 01444a63...);
- output/trace/manifest serialization.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import halfway_b1000  # noqa: E402  (frozen; MEMORIES is rebound by initialize(), so read via module)
from halfway_b1000 import (  # noqa: E402  (frozen; import side effects are inert)
    FAMILY_BUDGET,
    MEMBER_TIMEOUT,
    RUNS,
    consensus,
    initialize,
)
from kiafura_official.arc_perception import as_grid  # noqa: E402
from kiafura_official.reasoning.operational_dsd import resolve as legacy_resolve  # noqa: E402
from kiafura_official.reasoning.a4_runtime import resolve as resolve_a4  # noqa: E402
from kiafura_official.reasoning.a5_runtime import resolve as resolve_a5  # noqa: E402


# --- frozen policy, copied verbatim (experiments/cumulative_pass2.py:classes) ---
def classes(members, runs, queries):
    """Target-blind; no task ID, source membership, vote or rank learning."""
    result = [[] for _ in range(queries)]
    for n in runs:
        outputs = members[n]['outputs']
        if outputs is None:
            continue
        assert len(outputs) == queries, 'query misalignment'
        for q, output in enumerate(outputs):
            assert output is not None
            if output not in result[q]:
                result[q].append(output)
    return result


# --- frozen gate semantics, trace-retaining (see module docstring) ---
def resolve_union_traced(demos, queries):
    members = {}

    def timeout(*_):
        raise TimeoutError('frozen member exceeded 120 seconds')

    signal.signal(signal.SIGALRM, timeout)
    for name in RUNS:
        started = time.time()
        signal.alarm(MEMBER_TIMEOUT)
        try:
            if name in ('A100', 'A2', 'A300'):
                r = legacy_resolve(demos, queries, halfway_b1000.MEMORIES[name], FAMILY_BUDGET)
            else:
                r = (resolve_a4 if name == 'A4' else resolve_a5)(demos, queries, halfway_b1000.MEMORIES[name])
            members[name] = {
                'outputs': r.get('outputs'),
                'status': r['status'],
                'trace': {
                    'work': r.get('work'),
                    'control': r.get('control'),
                    'regions': r.get('regions'),
                    'candidates': len(r.get('candidates') or []),
                    'sources': r.get('sources'),
                },
            }
        except TimeoutError:
            members[name] = {'outputs': None, 'status': 'timeout', 'trace': None}
        except Exception as exc:  # fail-closed: named error, never a guess
            members[name] = {'outputs': None, 'status': 'error',
                             'trace': {'error': type(exc).__name__ + ': ' + str(exc)}}
        finally:
            signal.alarm(0)
        members[name]['elapsed_seconds'] = round(time.time() - started, 3)
    return {**consensus(members), 'members': members}


def load_task_inputs(path: Path):
    """TARGET-BLIND: train pairs + test inputs only. test[].output is never read."""
    t = json.loads(path.read_text())
    demos = [(as_grid(p['input']), as_grid(p['output'])) for p in t['train']]
    queries = [as_grid(p['input']) for p in t['test']]
    return demos, queries, len(t['test'])


def to_json_grid(g):
    return [list(row) for row in g] if g is not None else None


def fmt_pair(pair):
    """Two-output contract: duplicate a sole emission into both slots;
    None = named abstention for that slot (a100_a2_pass2.py:attempt_pairs rule)."""
    a, b = pair[0], pair[1]
    if a is None and b is not None:
        a = b
    elif b is None and a is not None:
        b = a
    return [to_json_grid(a), to_json_grid(b)]


def run_task(tid: str):
    demos, queries, n_test = load_task_inputs(ROOT / 'data' / 'arc-agi-2-training' / f'{tid}.json')
    r = resolve_union_traced(demos, queries)
    dist = classes(r['members'], RUNS, n_test)
    attempts = [fmt_pair((cs + [None, None])[:2]) for cs in dist]
    abstained = [i for i, cs in enumerate(dist) if not cs]
    named = None
    if abstained:
        reasons = {n: m['status'] for n, m in r['members'].items() if m['outputs'] is None}
        named = {
            'abstained_test_inputs': abstained,
            'cross_run_status': r['status'],
            'member_non_emission_reasons': reasons,
            'meaning': 'no demo-verified emission reached a single class for these inputs; '
                       'fail-closed abstention, not a guess',
        }
    return {
        'task_id': tid,
        'n_test_inputs': n_test,
        'status': r['status'],
        'emitting_runs': r['emitting_runs'],
        'attempts': attempts,
        'named_abstention': named,
        'members': {n: {'status': m['status'], 'elapsed_seconds': m['elapsed_seconds'],
                        'trace': m['trace']} for n, m in r['members'].items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Kaggle minimal demonstration (target-blind, fail-closed)')
    ap.add_argument('--tasks', default=str(ROOT / 'demo_tasks.txt'))
    ap.add_argument('--out', default=str(ROOT / 'outputs'))
    ap.add_argument('--limit', type=int, default=0, help='first N tasks only (smoke runs)')
    args = ap.parse_args()

    tids = [x.strip() for x in Path(args.tasks).read_text().splitlines() if x.strip()]
    if args.limit:
        tids = tids[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    initialize()  # frozen five-memory load (memoria/ read-only)
    counts = {}
    started = time.time()
    with open(out / 'trace.jsonl', 'w') as trace, open(out / 'predictions.jsonl', 'w') as preds:
        for i, tid in enumerate(tids, 1):
            rec = run_task(tid)
            counts[rec['status']] = counts.get(rec['status'], 0) + 1
            preds.write(json.dumps({'task_id': tid, 'status': rec['status'],
                                    'attempts': rec['attempts'],
                                    'named_abstention': rec['named_abstention']}) + '\n')
            trace.write(json.dumps(rec) + '\n')
            if i % 25 == 0:
                print(f'[{i}/{len(tids)}] {counts}', flush=True)

    manifest = {
        'label': 'DEMONSTRATION OUTPUT, NOT CERTIFIED EVIDENCE',
        'contract': 'two retained predictions per test input (first two distinct classes in '
                    'A100,A2,A300,A4,A5 order; sole emission duplicated into both slots; '
                    'null slot = named abstention). Target-blind: no test output read.',
        'gate': 'frozen fail-closed emission gates; cross-run single-region consensus '
                '(experiments/halfway_b1000.py:consensus, unchanged)',
        'tasks': len(tids),
        'status_counts': counts,
        'wall_seconds': round(time.time() - started, 1),
        'member_timeout_s': MEMBER_TIMEOUT,
        'family_budget': FAMILY_BUDGET,
        'task_set': 'the 500 compiled source tasks + the 12 historically resolved external exacts; '
                    'the 488-task residual is excluded by membership (firewall) and was not run',
    }
    (out / 'run-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
