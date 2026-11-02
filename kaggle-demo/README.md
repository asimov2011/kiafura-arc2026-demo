# kaggle-demo — minimal Kaggle paper-track demonstration

**What it demonstrates:** the approach — frozen operational memory + fail-closed emission gate + trace, emitting predictions-or-named-abstention under the official two-output contract (two predictions per test input). Not a competitor: **the Kaggle score is not an objective and is never computed.**

**DEMONSTRATION OUTPUT, NOT CERTIFIED EVIDENCE.** Every number this package produces is labelled demonstration-only. Certified numbers live in `../evidence/` and `../analysis/` with provenance.

**Status (2026-10-27):** packaged from frozen lineage (branch `work/issue-44-cumulative-pass2` @ `2fa15898…`); driver written; **run pending** (owner-authorized; the first smoke attempt was interrupted by a platform switch, not by any boundary problem). See `INVENTORY.md` for the runbook, entry points, asset pins and boundary. See `../execution/PAPER-2026-T1-W5.md` for the binding boundary.

**Run:** `python3 driver.py --limit 3 --out outputs-smoke` then `python3 driver.py --out outputs`. Pure stdlib, CPU-only, no GPU.

**Boundary:** packaging of existing code + light driver glue only (no algorithm development); task set = 500 compiled source tasks + 12 historically resolved external exacts; the 488-task residual excluded by membership (firewall); no protected/private evaluation; frozen memories immutable and read-only.
