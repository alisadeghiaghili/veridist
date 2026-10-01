"""Continue a compatible local exponential reduction from a SQLite checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from veridist import fit_exponential_checkpointed_chunks
from veridist.engine import CheckpointRecord, SQLiteCheckpointStore

SOURCE_REVISION = "demo-revision-1"
INITIAL_STATE = (
    b'{"compensation":"0x0.0p+0","event_count":0,'
    b'"observation_count":0,"total_time":"0x0.0p+0"}'
)


def create_store(path: Path) -> SQLiteCheckpointStore:
    """Create the initial local store for the declared source and reducer contract."""

    initial = CheckpointRecord.create(
        format_version=1,
        source_id="demo-source",
        source_schema="exponential-v1",
        source_revision=SOURCE_REVISION,
        reducer_id="exponential-reduction-v1",
        accumulator_schema="exponential-reduction-v1",
        plan_digest=hashlib.sha256(b"demo-plan-v1").hexdigest(),
        cursor=0,
        committed_ranges=(),
        generation=0,
        operation_token=None,
        operation_digest=None,
        state=INITIAL_STATE,
    )
    return SQLiteCheckpointStore.create(path, initial)


with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.sqlite3"
    first_store = create_store(path)
    # Each chunk is the offset form `(row_start, payload)`: row_start is the
    # caller's own count of previously committed rows, not read back from the
    # store. That is what lets a replayed chunk be recognized and skipped
    # instead of being applied -- and its rows counted -- a second time.
    fit_exponential_checkpointed_chunks(
        store=first_store,
        source_revision=SOURCE_REVISION,
        chunks=((0, b"[[1.5,true]]"),),
    )
    resumed_fit = fit_exponential_checkpointed_chunks(
        store=SQLiteCheckpointStore(path),
        source_revision=SOURCE_REVISION,
        chunks=((1, b"[[2.25,false]]"),),
    )

assert resumed_fit.observation_count == 2
assert resumed_fit.event_count == 1
assert resumed_fit.total_time == 3.75
print("rows=2; events=1; total_time=3.75")
