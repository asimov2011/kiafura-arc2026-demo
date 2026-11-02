# kaggle-demo — W5 inventory and runbook (frozen lineage → minimal runnable set)

**Status:** read-only inventory COMPLETE (2026-10-27); packaging COMPLETE (files on disk in this directory); **driver written but NEVER EXECUTED** (smoke run was interrupted when the PM platform was abandoned — the demo run itself remains owner-authorized, `execution/PAPER-2026-T1-W5.md`).
**Label:** demonstration packaging; every number it will produce is **demonstration output, not certified evidence**.

## 1. What this directory is

The smallest honest model that demonstrates the approach: **five frozen operational memories + fail-closed emission gates + trace**, emitting predictions-or-named-abstention under the official two-output contract on public demonstration tasks. Packaging of existing code + light driver glue only. **No algorithm modification** was needed or made: the frozen resolve/gate chain runs as-is once files are laid out and imported (verified by static import analysis; execution pending).

Source of truth: `kiafura-agosto26` branch **`work/issue-44-cumulative-pass2`** = commit `2fa158980e14cd04456e9f98ba0b6042c0627d60` (superset of `work/issue-39-a100-a2-pass2` = `8f0a74bfe…`). Every file here is a byte-for-byte copy from that branch, except `driver.py`, `README.md`, `INVENTORY.md` (new light glue/docs).

## 2. Layout (= the layout the frozen loaders expect)

```
kaggle-demo/
  driver.py                 LIGHT GLUE (see docstring; target-blind; fail-closed)
  demo_tasks.txt            512 task IDs: 500 compiled sources + 12 external exacts
  data/arc-agi-2-training/  512 public task JSONs (the 488-task residual EXCLUDED by membership)
  src/kiafura_official/     inference core + emission gates (82 .py, frozen)
  memoria/                  the five frozen memories (48 files, ~85 KB; read-only)
  experiments/              frozen loaders + consensus (8 .py, frozen):
                            a2_baseline.py (A100 loader), a2_union_eval.py (A2 loader),
                            a300_common.py (A300 loader; FAMILY_BUDGET=2_000_000),
                            a4_freeze.py / a5_freeze.py / a4_source_inspect.py /
                            a5_source_inspect.py (import-time only; verify() NOT called),
                            halfway_b1000.py (canonical five-run entry: initialize,
                            resolve_union, consensus; MEMBER_TIMEOUT=120s via SIGALRM)
  outputs/                  run record lands here (empty until the demo runs)
```

`memoria/a1/` was deliberately NOT packaged (different cohort, not one of the five).

## 3. Entry point chain (read from code; not yet executed)

Task JSON → `demos=[(as_grid(i),as_grid(o)) for train]`, `queries=[as_grid(i) for test]` (`kiafura_official.arc_perception.as_grid`) →

- A100/A2/A300: `kiafura_official.reasoning.operational_dsd.resolve(demos, queries, MEMORIES[name], 2_000_000)` → `{'outputs': tuple-of-grids|None, 'sources', 'status', 'work', 'control'}`. Statuses: `verified | abstain_conflict | abstain`.
- A4/A5: `kiafura_official.reasoning.{a4,a5}_runtime.resolve(demos, queries, memory)` → `{'outputs','regions','candidates','query_rejected_candidates','status': exact_demo_single_region|no_candidate|conflicting_query_regions}`.
- Cross-run: `experiments/halfway_b1000.py: consensus(members)` — emits only on ONE identical region across emitting runs, else `conflict` / `no_emission` (fail-closed). `driver.py:resolve_union_traced()` is a trace-retaining copy of `resolve_union` (same control flow/gate; the only delta keeps `work/control/regions/candidates` fields).
- Two-output contract in `driver.py`: `classes()` copied verbatim from `experiments/cumulative_pass2.py` (blob `804daafc…`) — first two distinct non-null outputs per query in A100,A2,A300,A4,A5 order — plus the sole-emission duplication rule of `experiments/a100_a2_pass2.py:attempt_pairs` (blob `01444a63…`). Null slot = named abstention. **Target-blind: `test[].output` is never read.**

## 4. Memory asset identity (from the audit manifests; spot-recomputed)

Per-memory `member_memory_sha256` pins: A100 `c57ae563…3880`, A2 `957961e6…b6c8`, A300 `62e3bf02…8211`, A4 `ed08cb09…2300`, A5 `21e08a77…eaeb`. Freeze digests verified: A300 `manifest.json` `freeze_sha256 239a13db…`, A4 `freeze_sha256 21207554…`, A5 `freeze_sha256 2a8ee73a…` (= the digests recorded in the transfer results). Audit identity pins: `experiments/a100_a2_pass2_v1/manifest.json` `c35c2bfc…8ebb` (self-digest `a496e134…`), `experiments/cumulative_pass2_v1/manifest.json` self-digest `29e98f95…b19f`. Full per-file SHA256 tables live in those manifests (`files_sha256`) in the lineage; the demo's `provenance.json` must re-list the packaged files' own hashes (NOT YET GENERATED — next step).

## 5. Runnability obstacles (all LIGHT GLUE; zero algorithm repair)

| # | Obstacle | Handling |
|---|---|---|
| R1 | Assets live on the frozen branch, not in the lineage worktree | extraction done (this directory) |
| R2 | Loaders expect sibling layout `src/`, `memoria/`, `data/arc-agi-2-training/` | layout reproduced (above); `driver.py` adds `src/` + `experiments/` to `sys.path` |
| R3 | Dependencies: **none** — pure Python stdlib (no numpy/torch/GPU; frozen under CPython 3.12.3; needs ≥3.9) | nothing to do |
| R4 | `signal.SIGALRM` timeouts are Linux-only | fine for WSL2/Kaggle CPU; a Windows run would need a threading-based cap (glue) |
| R5 | `a4_freeze.verify()`/`a5_freeze.verify()` hash external mastery files | NOT called; verify against `provenance.json` instead (glue) |
| R6 | The audit drivers (`a100_a2_pass2.py`, `cumulative_pass2.py`) entangle historical evidence | not needed; only their contract functions were copied into `driver.py` |
| R7 | transfer scripts import `a4/a5_source_inspect` for ROOT only | those tiny modules are packaged |

CPU-only: confirmed by import scan. Worst case five memories × 120 s/task; historical observed per-record times are single-digit seconds.

## 6. Runbook (for the next executor)

```sh
cd /root/projects/kiafura/reasoning-theory/kaggle-demo   # or the local checkout of RT
python3 driver.py --limit 3 --out outputs-smoke           # smoke first
python3 driver.py --out outputs                           # full run: 512 tasks
```

Expected acceptance (PLAN §5 / W5 packet): end-to-end run from a clean checkout of this package; `outputs/predictions.jsonl` (predictions-or-named-abstention), `outputs/trace.jsonl` (per-task trace), `outputs/run-manifest.json` (status counts, contract, boundary statement); then `provenance.json` (every packaged file SHA256 + source commit) and `papers/arc2026/C_APPENDIX_KAGGLE_DEMO.md` linked to it.

**Boundary reminders:** demo task set = the 500 compiled sources + the 12 historically resolved external exacts; the 488-task residual is excluded by membership and must not be run; no protected/private evaluation; no algorithm development; every output labelled demonstration-only; the Kaggle score is not an objective and must never be computed or reported.
