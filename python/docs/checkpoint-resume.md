# Checkpoint and resume a local run

Use this path only when a compatible local lifetime reduction may be interrupted.
It persists sufficient statistics in a local SQLite file; it does not store raw
input rows and it is not a distributed checkpoint service.

Reopen the same local store and provide the remaining compatible input. A
changed source, an invalid checkpoint, or an incompatible reducer returns a
typed failure rather than advancing the checkpoint.

For `fit_exponential_checkpointed_csv`, the source revision is not a free-form
label: it must equal the CSV file's current SHA-256 digest (lowercase hex),
stream-hashed before any row is read. Reusing an old revision string against a
file that has since changed returns `SOURCE_REVISION_MISMATCH`, even if the
checkpoint itself still has that same old revision recorded -- the check is
against the file on disk, not only against the checkpoint. The checkpoint's
public source id and source schema must also match the call's, or the result
is `SOURCE_ID_MISMATCH` / `SOURCE_SCHEMA_MISMATCH`. Build the store with
`veridist.execution.create_checkpointed_csv_store(store_path, csv_path=...,
source_id=...)` rather than constructing the checkpoint record by hand; it
computes the correct revision for you and keeps it tied to that one file.

For `fit_exponential_checkpointed_chunks`, pass each chunk as the offset form
`(row_start, payload)` rather than bare `bytes`. Because `row_start` is then
supplied by the caller instead of read back from the live cursor, replaying an
already-committed chunk is recognized and skipped instead of being applied,
and counted, a second time. The bare-`bytes` form still works but is
deprecated: it cannot generally recognize a replay (its `row_start` always
comes from the current cursor), and it emits `DeprecationWarning`.

Run the complete example from the repository root:

```console
python python/examples/checkpoint_resume.py
```

It creates a store, commits one lifetime chunk, reopens that store, and applies
a second compatible chunk. Its verified output is:

```text
rows=2; events=1; total_time=3.75
```

Use a different architecture when multiple workers, a network filesystem, or a
distributed store is required. Review [known limits](../KNOWN_LIMITS.md) before
putting a resumed result on a production decision path.
