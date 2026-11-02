# Kiafura — ARC-AGI-2 demonstration entry (ARC Prize 2026 Paper Track linkage)

**What this is.** The smallest honest model of the Kiafura approach: **five frozen operational
memories (A100, A2, A300, A4, A5) + frozen fail-closed emission gates + a full trace**, emitting
**predictions-or-named-abstention** under the official two-output contract (two predictions per test
input; a sole emission is duplicated into both slots; a slot with no real prediction carries the
venue's `[[0, 0], [0, 0]]` placeholder). It demonstrates the approach rather than competing on
score: every number it produces is demonstration output, not certified evidence, and the
competition score is not an objective of this work.

The memories *are* the model's knowledge — no training, no tuning, no selector development; the
pipeline is byte-for-byte the frozen inference/gate chain with light driver glue only (I/O, task
loading, two-output formatting, trace dump).

**Target-blind by construction:** only `train` pairs and `test[].input` are read; `test[].output` is
never read.

## Layout

- `kaggle-demo/` — the demonstration package: `src/` (inference core + emission gates),
  `memoria/` (five frozen memories, read-only), `experiments/` (frozen loaders + consensus),
  `driver.py` (light glue), `provenance.json` (SHA-256 of every packaged file; source commit
  `2fa158980e14cd04456e9f98ba0b6042c0627d60`; `no_algorithm_modification: true`).
- `kaggle-submit/make_submission.py` — submission packer (light glue): runs the driver chain over a
  set of challenge tasks and emits `submission.json` in the venue sample format. Portable paths via
  CLI flags; wall-clock guard fills unfinished tasks with the legal placeholder.
- `notebook/` — the Kaggle notebook entry (`kiafura-arc2026-demo.ipynb` + `kernel-metadata.json`).

## Run locally

```sh
python3 kaggle-submit/make_submission.py \
  --package-dir kaggle-demo --data-dir <dir-with-challenges> --out outputs
```

## Boundary

- The historical 488-task residual is excluded by membership only and is not part of any run here.
- No protected/private evaluation is performed by this code; no algorithm development, no training,
  no tuning to labels.
- Every number this pipeline produces is demonstration output, not certified evidence.

## Development and provenance

This package was **developed with AI assistance** under the direction of Carmelo Fiorilla (2026),
who takes responsibility for the work. The frozen core (`src/`, `experiments/`, `memoria/`) is
lineage code of the Kiafura program, copied byte-for-byte at the pinned commit; it retains its
original internal comments, including references to internal program documents that are not part
of this release. The driver and submission glue are light packaging code only.

## License

CC0 1.0 Universal (public domain dedication) — see `LICENSE`. This release covers the demonstration
code and the five frozen memories. The accompanying papers remain the author's work (CC BY as
published, with attribution) and are not covered by this license.
