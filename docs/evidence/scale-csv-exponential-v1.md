# Historical SCALE-CSV-EXP-01 snapshot

The machine-readable artifact is retained for historical correctness and
logical-payload facts only:
[`python/evidence/scale-csv-exponential-v1.json`](../../python/evidence/scale-csv-exponential-v1.json).

It records a CPython 3.14.6 / Windows 11 run at commit
`4490c9eb08e9ed5e420a2b677d9de843fdf66a5d`. The deterministic generator uses
Decimal arithmetic independently of the production fit and writes raw CSV only
to a private temporary directory before measurement. The artifact stores source
byte counts and SHA-256 values, never raw input paths.

Schema v1 has no timing provenance and is deliberately quarantined: the current
checker rejects it, so it cannot establish candidate readiness, performance, or
a current scale claim. A new candidate must generate schema v3 evidence from a
clean checkout; v3 records the paired timing provenance, measures elapsed time
in a pass without `tracemalloc` and memory in a separate pass, requires exactly
one measurement worker, and binds both the run and candidate SHA to the
reviewed commit. `artifact_sha256` is an integrity digest, not a signature. The historical snapshot remains useful
only for independently reproducible source bytes, Decimal fit facts, one-pass
counts, and logical retained-payload observations.

| Logical rows | Chunk budgets | Passes | Internal payload result | Fit result |
| ---: | --- | --- | --- | --- |
| 10,000 | 32KiB, 64KiB, 128KiB | 1/1 in every cell | peak retained/inflight <= configured cap | Decimal facts agree |
| 100,000 | 32KiB, 64KiB, 128KiB | 1/1 in every cell | peak retained/inflight <= configured cap | Decimal facts agree |
| 1,000,000 | 32KiB, 64KiB, 128KiB | 1/1 in every cell | peak retained/inflight <= configured cap | Decimal facts agree |

No elapsed-time value from this historical snapshot is published as a
performance result. RSS is absent because this Windows stdlib environment has
no supported `resource` RSS interface; it must not be read as zero.

The snapshot does not establish a current bounded-payload claim, portable RSS
ceiling, generic large-data support, backpressure stress behavior, parallel
speed-up, cancellation, checkpoint/retry, or support for other input formats.
