"""The scale engine: fits and inference for data that does not fit in memory.

``veridist.scale`` fits the six registered families to exact (uncensored) observations that
are read from a source in bounded batches, so the memory a fit needs does not grow with the
number of rows. The fit, its uncertainty and its goodness-of-fit report are computed from
exact, mergeable statistics: the estimate does not depend on the batch size, the partition
layout, the order of the rows, the executor, the number of workers, a spool or the point at
which a run was interrupted and resumed. Within one numeric environment the result is
bit-identical for every such choice; across environments it agrees within a tolerance (see
the large-data guide of the documentation for the two reproducibility tiers).

Entry points. :func:`fit` fits a family to a source; :func:`profile_intervals` adds
profile-likelihood intervals to a Weibull-minimum or right-Gumbel fit; :func:`large_n_gof`
reports a certified Kolmogorov-Smirnov enclosure of a fit, with no p-value.

Sources. Every source names its column when it is built and is one of
:class:`NumpySource`, :class:`ParquetSource`, :class:`ArrowSource`, :class:`PolarsSource`
and :class:`DuckDBSource`. The class names are the constructors: a source is a value that
can be inspected, passed to a process worker and named in a type annotation, and a second
spelling (a ``from_parquet`` function) would add nothing to it. Parquet, Arrow, Polars and
DuckDB need the optional extras ``veridist[arrow]``, ``veridist[polars]`` and
``veridist[duckdb]``; importing this package imports none of them, and a missing extra is
a :class:`ScaleSourceError` (``OPTIONAL_DEPENDENCY_MISSING``) when the source is built or
first read. Writing a source of your own is not part of the public API in this release.

Execution. ``executor=`` takes ``None`` or ``"serial"`` (one partition after the other),
:class:`ThreadExecutor` or :class:`ProcessExecutor`; ``spool=`` takes a :class:`ScaleSpoolPolicy`
for a source that can be read only once; ``checkpoint=`` takes a path or a
:class:`SQLitePartitionStore` and, with ``resume``, ``incremental``, ``recompute_changed``
and ``allow_removed``, continues or extends an earlier run.

Results and provenance. :func:`fit` returns a :class:`ScaleFitResult`: ``fit`` is the
success or failure of the in-memory family and ``scale`` a :class:`ScaleProvenance`, a closed
record of the partitions, the row count, the passes, the digest of the canonical state and
the numeric environment. A provenance record holds no data value; the partition identifier
of a file source names its file.

Errors. Programming mistakes are ``TypeError`` and ``ValueError``. Refusals that depend on
the data, a source, a stored state or a checkpoint are :class:`ScaleSourceError`,
:class:`ScaleStateError`, :class:`ScaleCheckpointError` and :class:`ScaleInferenceError`,
each a ``VeridistError`` and a ``ValueError`` with a closed ``code`` (the matching
``...ErrorCode`` enum), and a pass budget that is too small or exhausted is a
:class:`PassBudgetError`. No message or attribute holds a data value.

Stability. Every name in ``__all__`` is public API. It is not removed or changed
incompatibly within a major version; a deprecated name or form keeps working, emits a
``DeprecationWarning`` and is removed no earlier than the next major version, as everywhere
in Veridist, and each change is listed in the changelog. The private modules
(``veridist.scale._*``) are not API. The byte formats of states, of the partition store and
of the spool are not API either: a store written in another format is refused, never
silently read, and the numbers a fit returns are specified by the reproducibility tiers
and not by a format. The ``pool_factory`` field of the executors is a test hook and is not
part of the contract. The scale engine takes exact observations only: right censoring,
weights and truncation at scale are not supported.
"""

from veridist.engine.pass_budget import PassBudgetError
from veridist.engine.provenance import (
    ScaleCheckpointPassProvenance,
    ScaleCheckpointProvenance,
    ScaleEnvironmentProvenance,
    ScaleExecutionPassProvenance,
    ScaleExecutionProvenance,
    ScalePartitionProvenance,
    ScalePassProvenance,
    ScalePassRecord,
    ScaleProvenance,
    ScaleSpoolPartitionProvenance,
    ScaleSpoolProvenance,
    ScaleStateProvenance,
)
from veridist.scale._arrow import ArrowSource, ParquetSource
from veridist.scale._duckdb import DuckDBSource
from veridist.scale._errors import (
    ScaleCheckpointError,
    ScaleCheckpointErrorCode,
    ScaleInferenceError,
    ScaleInferenceErrorCode,
    ScaleSourceError,
    ScaleSourceErrorCode,
    ScaleStateError,
    ScaleStateErrorCode,
)
from veridist.scale._executors import ProcessExecutor, ThreadExecutor
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._gof import GofDecision, LargeNGof, large_n_gof
from veridist.scale._partition_store import SQLitePartitionStore
from veridist.scale._polars import PolarsSource
from veridist.scale._profile_intervals import (
    ProfileIntervals,
    ProfileResolution,
    ProfileUnavailable,
    ProfileUnavailableReason,
    Quantity,
    QuantityKind,
    mean,
    profile_intervals,
    quantile,
    survival,
)
from veridist.scale._sources import NumpySource
from veridist.scale._spool import ScaleSpoolPolicy

__all__ = [
    "ArrowSource",
    "DuckDBSource",
    "GofDecision",
    "LargeNGof",
    "NumpySource",
    "ParquetSource",
    "PassBudgetError",
    "PolarsSource",
    "ProcessExecutor",
    "ProfileIntervals",
    "ProfileResolution",
    "ProfileUnavailable",
    "ProfileUnavailableReason",
    "Quantity",
    "QuantityKind",
    "SQLitePartitionStore",
    "ScaleCheckpointError",
    "ScaleCheckpointErrorCode",
    "ScaleCheckpointPassProvenance",
    "ScaleCheckpointProvenance",
    "ScaleEnvironmentProvenance",
    "ScaleExecutionPassProvenance",
    "ScaleExecutionProvenance",
    "ScaleFitResult",
    "ScaleInferenceError",
    "ScaleInferenceErrorCode",
    "ScalePartitionProvenance",
    "ScalePassProvenance",
    "ScalePassRecord",
    "ScaleProvenance",
    "ScaleSourceError",
    "ScaleSourceErrorCode",
    "ScaleSpoolPartitionProvenance",
    "ScaleSpoolPolicy",
    "ScaleSpoolProvenance",
    "ScaleStateError",
    "ScaleStateErrorCode",
    "ScaleStateProvenance",
    "ThreadExecutor",
    "fit",
    "large_n_gof",
    "mean",
    "profile_intervals",
    "quantile",
    "survival",
]
