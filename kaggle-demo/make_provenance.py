#!/usr/bin/env python3
"""Generate kaggle-demo/provenance.json — LIGHT GLUE (packaging audit only).

Records, for every packaged file, its SHA-256 and origin: byte-for-byte copies
of the frozen lineage (kiafura-agosto26 branch work/issue-44-cumulative-pass2,
commit 2fa158980e14cd04456e9f98ba0b6042c0627d60) plus light glue/docs listed
explicitly. DEMONSTRATION PACKAGING, NOT CERTIFIED EVIDENCE.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = "2fa158980e14cd04456e9f98ba0b6042c0627d60"
SOURCE_BRANCH = "work/issue-44-cumulative-pass2"
SOURCE_REPO = "asimov2011/kiafura-agosto-26"
GLUE_FILES = {"driver.py", "README.md", "INVENTORY.md", "make_provenance.py", "provenance.json"}
EXCLUDED_DIRS = {"outputs", "outputs-smoke", "__pycache__"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    # frozen identity function over the packaged memories (read-only load)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "experiments"))
    import halfway_b1000
    halfway_b1000.initialize()
    member_hashes = {n: halfway_b1000.digest(m) for n, m in halfway_b1000.MEMORIES.items()}

    freeze_pins = {}
    for name, path in {
        "A300": ROOT / "memoria" / "a300" / "manifest.json",
        "A4": ROOT / "memoria" / "a4" / "a4_freeze.json",
        "A5": ROOT / "memoria" / "a5" / "a5_freeze.json",
    }.items():
        if path.exists():
            d = json.loads(path.read_text())
            if "freeze_sha256" in d:
                freeze_pins[name] = d["freeze_sha256"]

    files = {}
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if rel.parts[0] in EXCLUDED_DIRS or "__pycache__" in rel.parts:
            continue
        files[str(rel)] = {
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
            "origin": "light glue (this demo)" if str(rel) in GLUE_FILES or rel.parts[0] == "scripts" else "frozen lineage copy",
        }
    out = {
        "label": "DEMONSTRATION PACKAGING, NOT CERTIFIED EVIDENCE",
        "purpose": "Kaggle paper-track minimal demonstration: five frozen operational memories + "
                   "fail-closed emission gates + trace; every run number it produces is "
                   "demonstration output, never certified evidence or a competition score.",
        "source_of_truth": {
            "repo": SOURCE_REPO,
            "branch": SOURCE_BRANCH,
            "commit": SOURCE_COMMIT,
            "note": "superset of work/issue-39-a100-a2-pass2 = 8f0a74bfe…; every non-glue file "
                    "here is a byte-for-byte copy from that commit; the 488-task residual is "
                    "excluded by membership and was not packaged or run",
        },
        "member_memory_sha256": {
            **member_hashes,
            "method": "halfway_b1000.digest() over the packaged memories (same frozen identity "
                      "function asserted by cumulative_pass2_v1 verification)",
        },
        "freeze_sha256": freeze_pins,
        "frozen_identifiers": {
            "a100_a2_pass2_manifest_self_digest": "a496e134…8ebb (manifest c35c2bfc…)",
            "cumulative_pass2_manifest_self_digest": "29e98f95…b19f",
            "classes_policy_blob": "cumulative_pass2.py@804daafc…",
            "attempt_pairs_rule_blob": "a100_a2_pass2.py@01444a63…",
            "resolve_union_blob": "halfway_b1000.py@e48f8819…",
        },
        "boundary": {
            "task_set": "512 task IDs = 500 compiled sources + 12 historically resolved external exacts",
            "target_blind": "test[].output is never read by the driver",
            "contract": "two retained predictions per test input; null slot = named abstention",
            "no_algorithm_modification": True,
            "residual_488_firewall": "membership exclusion only; never run",
        },
        "files": files,
        "file_count": len(files),
    }
    (ROOT / "provenance.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"provenance.json written: {len(files)} files hashed; members {list(member_hashes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
