"""Shared loading for the frozen independent A300 operational memory."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARC = ROOT / "data" / "arc-agi-2-training"
SPLIT = ROOT / "splits" / "ct-2-independent-a300-v1.json"
MEMORY_ROOT = ROOT / "memoria" / "a300"
FAMILY_BUDGET = 2_000_000


def as_grid(value):
    return tuple(tuple(int(cell) for cell in row) for row in value)


def load_task(task_id):
    task = json.loads((ARC / f"{task_id}.json").read_text())
    demos = [(as_grid(pair["input"]), as_grid(pair["output"])) for pair in task["train"]]
    queries = [as_grid(pair["input"]) for pair in task["test"]]
    expected = tuple(as_grid(pair["output"]) for pair in task["test"])
    return demos, queries, expected


def load_memories():
    manifest = json.loads((MEMORY_ROOT / "manifest.json").read_text())
    return {
        key: json.loads((MEMORY_ROOT / filename).read_text())
        for key, filename in manifest["memory_files"].items()
    }
