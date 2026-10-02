# ADR-0005: Out-of-core backends and scale tiers

Status: Accepted

Owner: numerical/data-systems lead (TBD)

Decision evidence: direct Ali Sadeghi decision, 2026-08-20 -- the first domain
vertical is reliability + censoring + big-data; multipass work on a
`single_pass` source is rejected and spooling is explicit opt-in.

## Decision

Expose in-memory, chunked and parallel schedulers behind one data-source
protocol. The protocol declares `single_pass`, `replayable` or
`checkpoint_replayable`; a planner enforces `max_passes` before reading. Spool
is an explicit opt-in adapter with disk budget and cleanup policy. Classify each
operation, not merely each family: A-reg (one-pass,
fixed-state regular model), A-boundary (one-pass support/min/max statistics),
A-frequency (`O(unique)` state), B (adaptive multipass likelihood), C
(external-memory objective), or D (explicit approximation).

Every source declares stable identity, schema/version and redaction policy.
Adapters declare replayability plus ordering and offset guarantees. Chunks use
stable IDs and offset ranges; empty chunks are valid. Duplicate, lost or
out-of-order chunks fail with distinct typed errors unless declared semantics
and an idempotency key permit them. `chunk_bytes` and `max_inflight_bytes` are
hard observable budgets with backpressure and cancellation semantics.

## Consequences

Each matrix cell publishes accumulator state, pass budget, replayability,
chunk/row-ID semantics, memory/inflight-byte bounds, checkpoint/retry policy,
backpressure and cancellation behaviour. Missing/duplicate/out-of-order chunks
are typed failures; there is no unlabeled partial result. PKD informs regular
fixed-support models but does not license an iff claim for
parameter-dependent-support families. Tier B/C claims are about objective
evaluation and convergence contract, not a guarantee of global MLE.

Retries require idempotent updates. Checksum mismatch, source mutation, retry
exhaustion, cancellation and missing data are distinct typed causes.
Checkpoints retain accumulator schema/version and source range; incompatible
resume is rejected. Any partial result includes `complete=False`, exact missing
ranges and its typed cause. Provenance includes its schema/version, source
identity/hash or redaction, ranges/counts, adapter/library versions,
chunk/pass/budget policy, estimator/RNG/approximation and checkpoint metadata.

## Evidence and test implications

The first vertical slice must continuously test empty/ragged chunks, IDs and
offsets, duplicate/lost/out-of-order delivery, single-pass rejection of
multipass plans, explicit spool, pass enforcement, retry idempotency,
checkpoint compatibility, bounded state/inflight bytes, cancellation and
redacted provenance. The DS-01--DS-12 test harness specified in the v1 test
plan is **NOT IMPLEMENTED**; a scale claim is not deferred to a late
performance phase and cannot be advertised before the relevant tests exist.

## Implementation evidence addendum -- 2026-08-22

The preceding status sentence records the implementation state when this ADR
was accepted. DS-01--DS-12 contract tests are now present in commits `d846fc8`
through `1345666`. They exercise in-memory sources and test doubles for
planning, bounded delivery, pass enforcement, transactional retry/checkpoint
boundaries, typed failures, execution outcomes and closed redacted provenance.

This evidence does not establish an actual CSV, Parquet, Arrow, dataframe or
database adapter; a persistent checkpoint backend; a production orchestrator;
or an end-to-end statistical fit. The contract suite has not yet emitted a
retained adapter pass/byte trace or production-scale memory result. Therefore
no adapter, durability or scale tier is implemented or advertised by this
addendum.


## CSV exponential evidence addendum -- 2026-08-27

The preceding paragraph is historical. Commit `4490c9e` provides a strict
CSV lifetime adapter and the one-pass `fit_exponential_csv` orchestrator. Its
retained artifact, `python/evidence/scale-csv-exponential-v1.json`, is checked
fail-closed by `python/tools/check_scale_csv_exponential_evidence.py`.

The artifact records a CPython 3.14.6 / Windows 11 run at clean commit
`4490c9e`, over deterministic generated CSV inputs of 10,000, 100,000 and
1,000,000 logical rows and three configured logical chunk budgets (32,768,
65,536 and 131,072 bytes). Every matrix cell recorded one actual/allowed pass,
complete coverage of all logical rows, and a peak retained/inflight payload no
greater than its configured cap. The independent Decimal generator facts agree
with the fitted event count, total time and rate within the artifact's recorded
absolute and relative errors. It also records source bytes/SHA-256, chunk
counts, tracemalloc peaks and descriptive elapsed time; raw input paths are
absent. On this Windows run stdlib `resource` is unavailable, so RSS is
explicitly unknown rather than reported as zero.

This is evidence for this exact CSV/exponential matrix and CPython build only.
It establishes a bounded *internal retained-payload* observation, not a
portable process-memory ceiling, universal big-data throughput claim,
backpressure stress guarantee, Parquet/Arrow/dataframe/database support,
cancellation, retry, checkpoint durability or a general streaming-equivalence
theorem.

## Amendment (2026-10)

The schema has since moved past what `check_scale_csv_exponential_evidence.py`
accepts. The retained artifact records `schema_version: "1"`; the current
checker requires schema version 3 and rejects the file with "current evidence
requires schema version 3". The preceding addendum describes the artifact's
historical content, not a currently passing check. The historical snapshot
remains useful only for the independently reproducible source bytes, Decimal
fit facts, one-pass counts, and logical retained-payload observations it
records; it supports no current scale, throughput, or candidate-readiness
claim, matching `docs/evidence/scale-csv-exponential-v1.md` and
`docs/v1-readiness.md`.
