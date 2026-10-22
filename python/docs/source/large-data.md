(veridist-large-data)=
# Large data with `veridist.scale`

`veridist.scale` fits the six families to exact observations that are read from files, dataframes or a database in bounded batches, so the memory a fit needs does not grow with the number of rows. This page explains how to choose a source, how often the data is read, how to read it in parallel, how to continue an interrupted run, and how far the results can be trusted. Import the engine with `from veridist import scale`; its names are not exported at the top level of the package.

## What the scale engine does and does not do

The engine takes exact observations only: finite numbers without censoring, weights or truncation. It supports all six families: `exponential`, `normal`, `gamma` and `lognormal` read the data once, and `weibull_min` and `gumbel_right` read it several times. Right censoring at scale is not supported yet; use the in-memory `fit` for censored data.

`scale.fit(source, family)` returns a `ScaleFitResult`. Its `fit` attribute is the same success or failure class that the in-memory fit of the family returns, and its `scale` attribute is a `ScaleProvenance`: a closed record of the partitions, the row count, the passes, the digest of the state the numbers were computed from, and the numeric environment. The record holds no data value.

Parquet, Arrow, Polars and DuckDB sources need an optional extra: `pip install "veridist[arrow]"`, `pip install "veridist[polars]"` or `pip install "veridist[duckdb]"`. Importing `veridist.scale` imports none of them, and a missing extra is a `ScaleSourceError` with the code `OPTIONAL_DEPENDENCY_MISSING` and a hint that names the extra.

## Choose a source

A source names the column it reads when you build it, and is cut into partitions that can be read independently. The class names are the constructors:

* `NumpySource(values, partition_rows=None)` reads a one-dimensional numpy array in place. It has one partition, or one per `partition_rows` rows, and every partition has a content fingerprint.
* `ParquetSource(paths, column, root=None, content_hash=False)` reads one column of Parquet files or directories, with one partition per file and row group. The fingerprint is a metadata fingerprint, or a content fingerprint with `content_hash=True`.
* `ArrowSource(data, column)` reads a pyarrow `Table`, `RecordBatch` or `RecordBatchReader`, or any object that exports `__arrow_c_stream__`. It has one partition and no fingerprint. A table or a batch can be read again; a reader or a stream can be read once.
* `PolarsSource(frame, column)` reads a Polars `DataFrame` or `LazyFrame` with one partition and no fingerprint, running a lazy plan with the streaming engine of Polars.
* `DuckDBSource(source, column)` reads a DuckDB relation, or a `(connection, sql)` pair, with one partition and no fingerprint.

A fingerprint says what a partition covers. A content fingerprint hashes the data bytes. A metadata fingerprint hashes what the storage reports (the file identity and size, the modification time and the statistics of the row group): it detects a replaced or rewritten file, but not an in-place edit that keeps every reported property. No fingerprint means that the source cannot be recognised again without being consumed, so it cannot be checkpointed. A fingerprint guards against accidental change; it is not authentication.

Every source applies the same value policy before a value reaches a statistic. A `float64` column is used as it is, a `float32` column is widened exactly, an integer column is accepted only when every value has magnitude at most `2**53`, and every other type is refused. A null is refused, never dropped, and so are NaN and infinity. A refusal is a `ScaleSourceError` with a closed code (`COLUMN_NOT_FOUND`, `UNSUPPORTED_DTYPE`, `LOSSY_CAST`, `NULL_VALUE`, `NAN_VALUE` or `NON_FINITE_VALUE`) raised before any value is used, and it never contains a data value.

A value outside the support of the family, such as a negative time, is a `ScaleStateError` with the code `VALUE_OUT_OF_SUPPORT`. As in memory, `gamma`, `lognormal` and `weibull_min` return it instead as the fit failure `INVALID_SUPPORT`.

```python
import numpy as np

from veridist import scale

p = (np.arange(10_000) + 0.5) / 10_000
hours = 100.0 * (-np.log1p(-p)) ** (1 / 1.5)  # a Weibull sample: shape 1.5, scale 100

source = scale.NumpySource(hours, partition_rows=2_500)
result = scale.fit(source, "weibull_min")

shape = result.fit.parameters["shape"]
print(f"shape={shape:.3f}; rows={result.scale.row_count}; partitions={result.scale.partition_count}")
assert 3 <= result.scale.passes.pass_count <= 17
```

```text
shape=1.500; rows=10000; partitions=4
```

## Families, passes and the pass budget

`weibull_min` and `gumbel_right` have no fixed set of sufficient statistics, so the source is read several times: at least 3 and at most 17 times, as a root search plans it. `pass_budget` (default 8) is the number of reads you allow. A budget below 3 is refused before anything is read (`PASS_BUDGET_INSUFFICIENT`), and a run that would need a pass beyond the budget is refused before that pass is read (`PASS_BUDGET_EXCEEDED`). Both are a `PassBudgetError`, and no estimate is returned. Raise the budget for data that is hard to fit. The passes that were read are in `result.scale.passes`.

Data problems that the in-memory fit also reports, such as `DEGENERATE_SAMPLE` or `BOUNDARY_SOLUTION`, are returned as a failure in `result.fit` and are not raised: check `isinstance(result.fit, FitSuccess)` before you read the parameters.

`max_rows` (default 262144) bounds the rows of one batch, which is the unit of memory. It changes no result.

## Read the partitions in parallel

By default the partitions of a pass are read one after the other. `executor=` reads them concurrently: `scale.ThreadExecutor(workers)` on threads of this process, `scale.ProcessExecutor(workers)` on worker processes, or `"serial"` to record a serial run. The estimate, the state and its digest are identical for every executor and every number of workers; only the time changes.

```python
import numpy as np

from veridist import scale

p = (np.arange(10_000) + 0.5) / 10_000
hours = 100.0 * (-np.log1p(-p)) ** (1 / 1.5)

source = scale.NumpySource(hours, partition_rows=2_500)
serial = scale.fit(source, "weibull_min")
threaded = scale.fit(source, "weibull_min", executor=scale.ThreadExecutor(2))

assert threaded.fit.parameters == serial.fit.parameters
assert threaded.scale.state == serial.scale.state
print(threaded.scale.execution.executor, threaded.scale.execution.requested_workers)
```

```text
threads 2
```

Choose the executor by where the time goes:

* Threads start at once and share the memory of the process. numpy releases the interpreter lock only in parts of its work, so the gain is modest; use threads for data that lives in this process, such as a `NumpySource`, or when reading is dominated by input and output.
* Processes give real parallelism for the arithmetic, but a pool has a start-up cost: every worker is a new interpreter, started with `spawn` on every platform, that imports numpy and the library of the source before its first task. The cost is paid once per fit and shared by all its passes, so processes pay off for many large partitions (many Parquet files or row groups) and for a multi-pass family, and not for a small file. Put the entry point of a script that uses `scale.ProcessExecutor(4)` under `if __name__ == "__main__":`.
* A pass is read serially, never silently, when the request cannot be honoured. The reason is in `downgrade` of the pass record of `result.scale.execution`: `single_partition` (fewer than two partitions), `single_use_source` (a reader or a stream), `engine_managed` (Polars and DuckDB run their own threads), `in_memory_data` (processes would copy data held in memory) or `spool_first_pass` (the first pass of a spooled fit writes the spool in order).

The number of workers is the request, capped by 64, by the number of cores and by the number of partitions. With an executor, a partition whose read fails with a transient input or output error is read again, at most twice; the default `None` never repeats a read.

## A source that can be read only once: the spool

A multi-pass family reads its source several times. An `ArrowSource` over a reader or a stream can be read once, and a Polars plan or a DuckDB query would run again for every pass. Without a spool, a multi-pass fit of such a source is refused before anything is read (`SPOOL_REQUIRED`).

`spool=scale.ScaleSpoolPolicy(max_bytes=..., directory=...)` is the explicit remedy. The first pass writes the validated values to a private directory, every later pass reads that copy and checks it against its digests, and the directory is deleted when the fit ends, however it ends. The result is bit for bit the one the same fit without a spool gives. A one-pass family reads its source once and ignores the spool.

`max_bytes` bounds the space the spool may use (`SPOOL_LIMIT_EXCEEDED`). `directory` must be an existing directory; pass `allow_system_temp=True` instead to use the system temporary directory. A spool file that changed or went missing is `SPOOL_CHANGED`. The spool needs the `arrow` extra, and `result.scale.spool` records its row count and one digest per partition, never a path.

```python
import tempfile

import numpy as np
import pyarrow as pa

from veridist import scale

p = (np.arange(4_000) + 0.5) / 4_000
table = pa.table({"hours": 100.0 * (-np.log1p(-p)) ** (1 / 1.5)})
reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches(max_chunksize=1_000))
source = scale.ArrowSource(reader, "hours")

try:
    scale.fit(source, "weibull_min")
except scale.ScaleSourceError as error:
    print(error.code.value)

with tempfile.TemporaryDirectory() as directory:
    policy = scale.ScaleSpoolPolicy(max_bytes=10_000_000, directory=directory)
    result = scale.fit(source, "weibull_min", spool=policy)
print(f"spooled rows={result.scale.spool.row_count}")
```

```text
SPOOL_REQUIRED
spooled rows=4000
```

## Checkpoint, resume and incremental fitting

`checkpoint=` takes a path or a `scale.SQLitePartitionStore` and keeps the exact state of every partition, and the merged result of every completed pass, in one local SQLite file. A run that was interrupted continues with `resume=True` and reads only the partitions that have no committed state. The result, its state digest and the provenance, except the checkpoint record, are bit for bit those of an uninterrupted run. Only a source whose partitions have a fingerprint can be checkpointed: `NumpySource` and `ParquetSource`. An `ArrowSource`, a `PolarsSource` or a `DuckDBSource` is refused (`SOURCE_REVISION_UNAVAILABLE`).

A store belongs to one job: the family, the column, the kind of source, `max_rows`, the value policy, the fingerprint levels, the state schema and the numeric environment. Without `resume=True`, a store that already holds a job is refused (`JOB_EXISTS`) and is never overwritten. With it, a store that holds another job is refused (`JOB_MISMATCH`), and the `components` of the error name what differs. The executor, the number of workers, the spool and the pass budget are not part of the job.

Before anything is read, the partitions of the source are compared with the stored ones. A partition whose fingerprint changed is refused (`PARTITION_CHANGED`), a partition that is gone is `PARTITION_MISSING`, and a new one is `PARTITION_ADDED`; a refusal changes nothing. `recompute_changed=True` reads a changed partition again and replaces its state; a multi-pass fit then starts again from its first pass. `incremental=True`, together with `resume=True` and a one-pass family, fits a source that has gained partitions by reading only the new ones, and `allow_removed=True` also excludes partitions that are gone. A multi-pass family refuses `incremental` (`INCREMENTAL_UNSUPPORTED`), because a new partition changes the sums of every pass.

The store is for one host and a local file system. Every row carries a checksum, so a damaged or unknown store is a typed `ScaleCheckpointError` (`STORE_DECODE_FAILED`, `STORE_DIGEST_MISMATCH`, `STORE_FORMAT_UNSUPPORTED`, `STORE_STORAGE_FAILED` or `STORE_COMMIT_UNCERTAIN`) and never a state that is used. This is an integrity check against accidental damage, not encryption or authentication. `result.scale.checkpoint` counts the partitions that were reused and read, and holds no path or fingerprint.

```python
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from veridist import scale

values = np.linspace(1.0, 100.0, 5_000)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "data").mkdir()
    for index in range(2):
        table = pa.table({"x": values[index::2][:2_000]})
        pq.write_table(table, root / "data" / f"part-{index}.parquet", row_group_size=1_000)
    store = root / "job.sqlite"

    source = scale.ParquetSource(root / "data", "x")
    first = scale.fit(source, "normal", checkpoint=store, resume=True)
    again = scale.fit(source, "normal", checkpoint=store, resume=True)
    assert again.scale.state == first.scale.state
    print(f"reused={again.scale.checkpoint.reused_partitions}; read={again.scale.checkpoint.read_partitions}")

    try:
        scale.fit(source, "normal", checkpoint=store)
    except scale.ScaleCheckpointError as error:
        print(error.code.value)

    pq.write_table(pa.table({"x": values[:1_000]}), root / "data" / "part-2.parquet")
    grown = scale.ParquetSource(root / "data", "x")
    try:
        scale.fit(grown, "normal", checkpoint=store, resume=True)
    except scale.ScaleCheckpointError as error:
        print(error.code.value)
    result = scale.fit(grown, "normal", checkpoint=store, resume=True, incremental=True)
    done = result.scale.checkpoint
    print(f"added={done.added_partitions}; read={done.read_partitions}; rows={result.scale.row_count}")
```

```text
reused=4; read=0
JOB_EXISTS
PARTITION_ADDED
added=1; read=1; rows=5000
```

## Uncertainty and profile-likelihood intervals

A successful result reports `result.fit.uncertainty()` as an in-memory fit does, without reading the data again. For `normal`, `gamma` and `lognormal` it is computed from the exact sufficient statistics, and for `exponential` from the count and the total time. For `weibull_min` and `gumbel_right` it is the Wald uncertainty (covariance, standard errors and Wald intervals) from the observed information of the last pass; `method="profile"` raises `NotImplementedError` there, because a profile likelihood needs more passes.

`scale.profile_intervals(source, result, level=0.95, quantities=[...])` reads the source again and returns likelihood-ratio intervals for the parameters of a `weibull_min` or `gumbel_right` fit. For a Weibull fit it also returns intervals for derived quantities: `scale.mean()`, `scale.quantile(p)` and `scale.survival(t)`; for a Gumbel fit those entries are a typed `ProfileUnavailable`. It reads at least 2 and at most 9 passes, within its own `inference_pass_budget` (default 6), which is separate from the budget of the fit. `source` must be the data the fit was made from; another source is refused (`PARTITION_CHANGED`).

A failure is a `ScaleInferenceError` with the code `FIT_NOT_PROFILABLE`, `PROFILE_NOT_CONVERGED`, `PROFILE_UNBOUNDED` (an end of the interval lies beyond the search domain of the fit) or `PROFILE_NOT_COMPUTABLE`. An end is never silently replaced by a Wald end or an unbounded one.

```python
import numpy as np

from veridist import scale

p = (np.arange(10_000) + 0.5) / 10_000
hours = 100.0 * (-np.log1p(-p)) ** (1 / 1.5)

source = scale.NumpySource(hours, partition_rows=2_500)
result = scale.fit(source, "weibull_min")
intervals = scale.profile_intervals(source, result, quantities=[scale.quantile(0.1)])

shape = intervals.parameters["shape"]
b10 = intervals.derived["quantile:0.1"]
print(f"shape {shape.lower:.3f} to {shape.upper:.3f} at level {intervals.level}")
print(f"B10 {b10.lower:.2f} to {b10.upper:.2f}")
```

```text
shape 1.477 to 1.523 at level 0.95
B10 21.64 to 22.99
```

## Goodness of fit at large n

At millions of rows a p-value answers whether the model is exactly right, which no model is. `scale.large_n_gof(source, result, threshold=0.01)` instead reports how far the data are from the fitted distribution, and lets you say how far is acceptable. It reads the source once more, at the fitted parameters of any of the six families, and returns a `LargeNGof`.

The report holds an interval `[ks_lower, ks_upper]` that is certain to contain the Kolmogorov-Smirnov distance between the empirical and the fitted distribution function, certified up to `cdf_error`, the stated rounding error of the fitted distribution function. It also holds an interval for a binned total-variation distance and, if you gave a `threshold`, a `GofDecision`: `PASS` when `ks_upper` is at most the threshold, `FAIL` when `ks_lower` is above it, and `INCONCLUSIVE` when the interval contains it. There is no p-value and no asymptotic approximation, and none of the three decisions says that the model is true or false; the threshold is yours.

`bin_exponent` (default 12, from 1 to 20) sets `2**k` bins: more bins narrow the interval. The report is one pass, so a source that can be read once is enough. A failure is a `ScaleInferenceError` with the code `GOF_FIT_NOT_AVAILABLE` (the fit is a failure), `GOF_NOT_REPRESENTABLE` or `GOF_RESOLUTION_UNSUPPORTED`.

The refit Monte Carlo tests of `veridist.inference` (`refit_monte_carlo_gof` and `assess_families`) refit the sample for every replicate. They refuse a sample of more than 100000 observations by default and refuse a scale source, with `RefitGofRefused` (code `SCALE_REFIT_REFUSED`); they never subsample. For large data use `large_n_gof`.

```python
import numpy as np

from veridist import scale

p = (np.arange(10_000) + 0.5) / 10_000
hours = 100.0 * (-np.log1p(-p)) ** (1 / 1.5)

source = scale.NumpySource(hours, partition_rows=2_500)
result = scale.fit(source, "weibull_min")
report = scale.large_n_gof(source, result, threshold=0.01)

assert 0.0 <= report.ks_lower <= report.ks_upper <= 0.01
print(report.decision.value, report.bins)
```

```text
PASS 4096
```

## Reproducibility and the numeric environment

Within one numeric environment, the result of a fit, its state and the digest of the state are bit-identical for every batch size, partition order, way of splitting the rows into partitions or files, executor, number of workers, use of a spool, and point at which a run was interrupted and resumed. This is guaranteed and tested. Compare `result.scale.state.digest_sha256` of two runs to check it.

The numeric environment is the CPU architecture, the operating system, the Python build and the numpy version. The logarithm and exponential of a platform are not correctly rounded, so the engine records a digest: the SHA-256 of the numpy version and of the results of `numpy.log` and `numpy.exp` over a fixed reference sample, in `result.scale.environment`. Equal digests mean that the two runs evaluated those functions identically. A checkpoint is tied to the digest: resuming in another environment is refused (`JOB_MISMATCH`, with `environment` among the components).

Across environments, bit identity is not promised. Results agree to within the rounding error of the `log`, `exp` and `pow` of the platforms, and the digest tells you which case you are in. For the families whose statistics are exact sums, equal bits are expected but not claimed. A scale fit also differs in the last bits from the in-memory fit of the same data, which uses another algorithm and a less exact sum.

## Memory, limits and stability

A worker holds one batch (at most `max_rows` values, 2 MiB of `float64` at the default) and a state whose size does not depend on the number of rows. This is a logical bound and not a promise about the memory of the process, which also holds the buffers of the libraries; with several workers the batches add up. A `NumpySource` reads an array that you already hold in memory.

Not supported at scale: right censoring, weights, truncation, covariates, distributed executors, checkpoint stores other than local SQLite, source classes of your own, and bootstrap or refit Monte Carlo tests. The <a href="../../KNOWN_LIMITS.md">known limits</a> record the exact boundary.

Every name in `veridist.scale.__all__` is public API. It is not removed or changed incompatibly within a major version, and a deprecated form keeps working with a `DeprecationWarning` until the next major version, as everywhere in Veridist. The private modules `veridist.scale._*` and the byte formats of states, stores and spools are not API.

## Retained measurements

The `veridist-scale-evidence` workflow measured the scale engine on 2026-10-21 from candidate commit `e316d125e0ae434beaf0c69ebeee3238c1076f6b`, with one measurement worker, on a GitHub Actions Linux runner (CPython 3.11.17) and a Windows runner (CPython 3.11.9), each with 4 vCPUs. The data are exponential observations in Parquet files, fitted with `max_rows` 262144 and a `pass_budget` of 8, once with the serial executor and once with four processes. The ladder is 1 million, 10 million and 100 million rows on both runners, and 1 billion rows on Linux only, where just the four families that read the data once were measured. The files `python/evidence/scale-engine-2.2.0-linux.json` and `python/evidence/scale-engine-2.2.0-windows.json` are kept in the source repository, not in the distribution archives, and `tools/check_scale_engine_evidence.py` accepts them for that commit.

* At 100 million rows on Linux, the four one-pass families took 1.3 to 5.4 seconds with the serial executor and 1.8 to 3.7 seconds with four processes, `exponential` being the fastest and `lognormal` the slowest. `weibull_min` took 57.9 seconds serially and 26.4 seconds with four processes, in 6 passes; `gumbel_right` took 39.8 and 18.7 seconds, in 4 passes.
* At 100 million rows on Windows, the four one-pass families took 1.9 to 7.6 seconds serially and 1.8 to 4.5 seconds with four processes. `weibull_min` took 83.1 and 45.6 seconds, in 6 passes, and `gumbel_right` 59.4 and 32.8 seconds, in 4 passes.
* At 1 billion rows on Linux, the four one-pass families took 13.1 to 57.5 seconds serially and 12.3 to 30.1 seconds with four processes: `exponential` took 13.1 and 12.3 seconds and `lognormal` 57.5 and 30.1 seconds.
* The peak resident set size (RSS) of the measuring process, which does not include the worker processes, was 221.6 MiB in every Linux cell at 100 million and at 1 billion rows: it was the same at both sizes. On Windows at 100 million rows it was 78.4 to 97.8 MiB with the serial executor and 56.1 to 59.7 MiB with four processes.
* A one-pass family read the data once at every size. `weibull_min` took 6 passes at every size up to 100 million rows, and `gumbel_right` took 6 passes up to 10 million rows and 4 passes at 100 million rows, within the `pass_budget` of 8.
* For every family at every size, the digests of the result, the state and the passes were identical in four fits of the same rows: with the serial executor, with four processes, with a second partition layout, and in a run that was interrupted part-way and resumed from its checkpoint. That holds for all 22 combinations of family and size on Linux and all 18 on Windows.

These are descriptive measurements of those runs, on one kind of data and one Python build per platform, and not guarantees of speed or memory on other machines, data or settings.

## Errors and their codes

Mistakes in the call are `TypeError` or `ValueError`. Every refusal that depends on the data, a source or a store is a `VeridistError` and a `ValueError` with a closed `code` and no data value, path or fingerprint:

* `ScaleSourceError`: the codes `OPTIONAL_DEPENDENCY_MISSING`, `COLUMN_NOT_FOUND`, `UNSUPPORTED_DTYPE`, `LOSSY_CAST`, `NULL_VALUE`, `NAN_VALUE`, `NON_FINITE_VALUE`, `SOURCE_NOT_FOUND`, `SOURCE_UNREADABLE`, `SOURCE_CONSUMED`, `PARTITION_CHANGED`, `SPOOL_REQUIRED`, `SPOOL_LIMIT_EXCEEDED`, `SPOOL_CHANGED` and `SPOOL_UNAVAILABLE`.
* `ScaleStateError`: the codes `NON_FINITE_VALUE`, `VALUE_OUT_OF_SUPPORT`, `OBSERVATION_LIMIT_EXCEEDED`, `TOTAL_NOT_REPRESENTABLE`, `INCOMPATIBLE_STATE`, `STATE_BYTES_INVALID`, `STATE_TAG_MISMATCH` and `STATE_VERSION_UNSUPPORTED`.
* `ScaleCheckpointError`: the codes `SOURCE_REVISION_UNAVAILABLE`, `INCREMENTAL_UNSUPPORTED`, `JOB_EXISTS`, `JOB_MISMATCH`, `PARTITION_MISSING`, `PARTITION_ADDED`, `PARTITION_STATE_CONFLICT`, `LEDGER_MISMATCH`, `STORE_FORMAT_UNSUPPORTED`, `STORE_DECODE_FAILED`, `STORE_DIGEST_MISMATCH`, `STORE_STORAGE_FAILED` and `STORE_COMMIT_UNCERTAIN`.
* `ScaleInferenceError`: the codes `FIT_NOT_PROFILABLE`, `PROFILE_NOT_CONVERGED`, `PROFILE_UNBOUNDED`, `PROFILE_NOT_COMPUTABLE`, `GOF_FIT_NOT_AVAILABLE`, `GOF_NOT_REPRESENTABLE` and `GOF_RESOLUTION_UNSUPPORTED`.
* `PassBudgetError`: the codes `PASS_BUDGET_INSUFFICIENT` and `PASS_BUDGET_EXCEEDED`.

For the other entry points of the package, see the <a href="api.html#veridist-api">API guide</a>.
