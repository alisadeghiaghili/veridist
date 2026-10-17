"""Large-n goodness of fit of a scale fit: a certified Kolmogorov-Smirnov enclosure, no p-value.

At very large ``n`` a p-value answers whether the model is exactly right, which no model is, so the
report states how far the data are from the fitted distribution and lets the caller say how far is
acceptable. ``large_n_gof(source, result, threshold=...)`` reads the source once more, at the fitted
parameters of ``result`` (any of the six families), and reports the Kolmogorov-Smirnov distance
``D = sup |F_n - F|`` of the empirical distribution function ``F_n`` from the fitted ``F`` as an
interval ``[ks_lower, ks_upper]`` that is certain to contain it, a binned total variation distance,
and, when a threshold is given, a typed decision. It reports no p-value and uses no asymptotic
result.

Counts. With ``u = F(x)`` the probability integral transform of an observation and ``B = 2**k``
bins, ``F_n`` and ``F`` are compared on the grid of edges ``e_j = j / B`` of the ``u`` scale. Since
``F`` is continuous, ``D = sup_t |G(t) - t|`` with ``G`` the empirical distribution function of the
``u`` values. A pass counts, exactly and as integers, how many rows have a computed ``u`` in each of
``2 B - 1`` regions: around every interior edge a strip of half-width ``epsilon`` and, between the
strips, the rest of the bins. The regions are told apart by comparing the computed ``u`` with
precomputed binary64 cut points (``numpy.searchsorted``, no arithmetic on the data), so the counts
are exact for the computed values, mergeable by addition and independent of batching, layout and
order.

Evaluation error. ``epsilon`` bounds ``|u_computed - u_true|`` for every row, where ``u_true`` is
the exact value of the fitted distribution function at the observation and the fitted binary64
parameters: :func:`cdf_error_bound` states it per family from the rounding analysis of the kernels
and is checked in the reference tests against ``mpmath``. Every claim below is certified up to that
stated error and no further; it is part of the report (``cdf_error``).

The enclosure. Let ``A_j`` be the number of rows with computed ``u`` below the strip of edge ``j``
and ``Z_j`` the number below the end of that strip (``A_0 = 0``, ``Z_B = n``). A row below the strip
has true ``u < e_j``, a row at or above it has true ``u >= e_j``, hence the number ``N_j`` of rows
with true ``u < e_j`` satisfies ``A_j <= N_j <= Z_j``. ``G`` never decreases, so for ``t`` in
``[e_j, e_{j+1})``

    G(t) - t <= N_{j+1} / n - e_j <= Z_{j+1} / n - e_j,
    t - G(t) <= e_{j+1} - N_j / n <= e_{j+1} - A_j / n.

Every point of ``[0, 1)`` lies in such a bin and ``G(1) = 1``, so the largest of these bounds is an
upper bound ``ks_upper`` of ``D``. The left limit of ``G`` at an edge is ``N_j / n``, so
``|N_j / n - e_j| <= D``, which gives the lower bound ``ks_lower`` as the largest of
``A_j / n - e_j`` and ``e_j - Z_j / n`` over the interior edges. All of it is integer arithmetic
until one division that is rounded outward. Writing
``N_{j+1} / n - e_j = (N_{j+1} / n - e_{j+1}) + 1 / B``, the interval is never wider than ``1 / B``
plus the share of the rows that lie in the strips, so the width falls as the number of bins grows.

Effect size. ``total_variation`` is half the sum over the bins of ``|c_j / n - 1 / B|`` for the
rows' bin counts ``c_j``, as an interval because the rows in a strip may belong to either neighbour.
It is a statistic of the histogram of ``u``: at fewer rows per bin than a few hundred, sampling
noise dominates it, and it is reported without a reference value.

Decision. ``PASS``: ``ks_upper <= threshold`` (the distance is certainly within it); ``FAIL``:
``ks_lower > threshold`` (certainly beyond it); ``INCONCLUSIVE``: the interval contains the
threshold, which a larger number of bins can resolve. No wording of acceptance or rejection of the
model follows from any of them; the threshold is the caller's.

Cost. One pass; per row one evaluation of the fitted distribution function, which is vectorised for
the exponential, Weibull and Gumbel families and is a scalar kernel per element for the others.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from hmac import compare_digest
from math import fsum, nextafter
from typing import Any, Final

from veridist.engine.provenance import ScaleProvenance
from veridist.families.registry import FamilyId
from veridist.families.results import FitSuccess
from veridist.scale._errors import (
    ScaleInferenceError,
    ScaleInferenceErrorCode,
    ScaleSourceError,
    ScaleSourceErrorCode,
    ScaleStateError,
    ScaleStateErrorCode,
)
from veridist.scale._executors import PassEngine
from veridist.scale._fit import ScaleFitResult
from veridist.scale._inference_run import (
    StoreArgument,
    check_options,
    inference_bounds,
    run_inference,
)
from veridist.scale._kernels import float64_batch, numpy_module
from veridist.scale._partition_store import SQLitePartitionStore
from veridist.scale._planner import PassLedger, PassRole
from veridist.scale._sources import DEFAULT_MAX_ROWS, BatchReader, Partition, Source

GOF_PASS_BUDGET_DEFAULT: Final = 1
GOF_PASSES: Final = 1
DEFAULT_BIN_EXPONENT: Final = 12
MIN_BIN_EXPONENT: Final = 1
MAX_BIN_EXPONENT: Final = 20
"""The number of bins is ``2**k`` for ``k`` from ``MIN_BIN_EXPONENT`` to ``MAX_BIN_EXPONENT``."""

MAX_PRODUCT: Final = 1 << 62
"""``rows * bins`` must stay below this, so that the integer arithmetic of the enclosure is exact
in 64 bits."""

PIT_TAG: Final = "scale.pit_counts"
PIT_SCHEMA_VERSION: Final = 1
_MAGIC: Final = b"VDSP"
_CHECKSUM_BYTES: Final = 32
_JOB_KIND: Final = "gof"
_SCHEMA: Final = f"{PIT_TAG}:{PIT_SCHEMA_VERSION}"
_UNIT: Final = 2.0**-53


class GofDecision(StrEnum):
    """What the enclosure says about a threshold on the Kolmogorov-Smirnov distance."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


def cdf_error_bound(family: FamilyId, parameters: Mapping[str, float]) -> float:
    """A bound on ``|F_computed(x) - F(x)|`` for the vectorised or scalar kernel of ``family``.

    ``F`` is the exact distribution function at the binary64 ``parameters`` and ``x`` any finite
    observation. The bound follows from the rounding of the formulas (the division ``x / scale``
    carries a relative error of ``2**-53`` which the power of a Weibull shape multiplies by the
    shape, the logarithm of a lognormal is divided by its sigma, the incomplete gamma ratio loses
    about ``shape`` units in the exponent of its prefactor) with a margin of more than ten, and the
    reference tests check it against ``mpmath`` on grids of parameters and of probabilities.
    """

    if (
        family is FamilyId.EXPONENTIAL
        or family is FamilyId.NORMAL
        or family is FamilyId.GUMBEL_RIGHT
    ):
        return 2.0**-48
    if family is FamilyId.WEIBULL_MIN:
        return (parameters["shape"] + 8.0) * 2.0**-49
    if family is FamilyId.LOGNORMAL:
        sigma = parameters["sigma_log"]
        return 2.0**-52 * (16.0 + (abs(parameters["mu_log"]) + 1.0) / sigma)
    return (parameters["shape"] + 8.0) * 2.0**-46


# ------------------------------------------------------------------------------ counts


def bin_count(exponent: int) -> int:
    return 1 << exponent


def edge_cuts(exponent: int, error: float) -> Any:
    """The cut points of the regions: for each interior edge ``e`` the pair ``e - error``,
    ``e + error`` rounded outward, ascending (``2 (B - 1)`` of them).

    ``ScaleInferenceError`` with ``GOF_RESOLUTION_UNSUPPORTED`` when the strips of neighbouring
    edges would overlap.
    """

    np = numpy_module()
    bins = bin_count(exponent)
    if not 4.0 * error < 1.0 / bins:
        raise ScaleInferenceError(ScaleInferenceErrorCode.GOF_RESOLUTION_UNSUPPORTED)
    cuts = np.empty(2 * (bins - 1), dtype=np.float64)
    for index in range(1, bins):
        edge = index / bins
        cuts[2 * index - 2] = nextafter(edge - error, -1.0)
        cuts[2 * index - 1] = nextafter(edge + error, 2.0)
    return cuts


class PitCounts:
    """The integer counts of the ``2 B - 1`` regions of some computed ``u`` values."""

    __slots__ = ("count", "counts", "exponent")

    def __init__(self, exponent: int) -> None:
        np = numpy_module()
        self.exponent = exponent
        self.count = 0
        self.counts = np.zeros(2 * bin_count(exponent) - 1, dtype=np.int64)

    def update(self, probabilities: Any, cuts: Any) -> None:
        """Count one batch of computed probabilities.

        ``ScaleInferenceError`` with ``GOF_NOT_REPRESENTABLE`` for a value that is not finite.
        """

        np = numpy_module()
        if not np.isfinite(probabilities).all():
            raise ScaleInferenceError(ScaleInferenceErrorCode.GOF_NOT_REPRESENTABLE)
        regions = np.searchsorted(cuts, probabilities, side="right")
        self.counts += np.bincount(regions, minlength=self.counts.shape[0])
        self.count += int(probabilities.shape[0])

    def merge(self, other: PitCounts) -> PitCounts:
        """A new accumulator holding the counts of both."""

        if other.exponent != self.exponent:
            raise ScaleStateError(ScaleStateErrorCode.INCOMPATIBLE_STATE)
        merged = PitCounts(self.exponent)
        merged.count = self.count + other.count
        merged.counts = self.counts + other.counts
        return merged

    def to_bytes(self) -> bytes:
        """Canonical bytes: tag, schema version, exponent, row count, the counts, a checksum."""

        tag = PIT_TAG.encode("ascii")
        body: bytes = (
            _MAGIC
            + bytes([len(tag)])
            + tag
            + PIT_SCHEMA_VERSION.to_bytes(2, "big")
            + bytes([self.exponent])
            + self.count.to_bytes(8, "big")
            + self.counts.astype(">u8").tobytes()
        )
        return body + sha256(body).digest()

    @classmethod
    def from_bytes(cls, data: object, exponent: int) -> PitCounts:
        """Restore the counts of ``2**exponent`` bins from :meth:`to_bytes` output.

        ``TypeError`` for a non-bytes argument, otherwise ``ScaleStateError`` with
        ``STATE_BYTES_INVALID``, ``STATE_TAG_MISMATCH`` or ``STATE_VERSION_UNSUPPORTED``.
        """

        if not isinstance(data, bytes | bytearray):
            raise TypeError("state data must be bytes")
        np = numpy_module()
        raw = bytes(data)
        tag = PIT_TAG.encode("ascii")
        header = len(_MAGIC) + 1 + len(tag) + 2
        invalid = ScaleStateError(ScaleStateErrorCode.STATE_BYTES_INVALID)
        if raw[: len(_MAGIC)] != _MAGIC or len(raw) < header + _CHECKSUM_BYTES:
            raise invalid
        if raw[len(_MAGIC) : header - 2] != bytes([len(tag)]) + tag:
            raise ScaleStateError(ScaleStateErrorCode.STATE_TAG_MISMATCH)
        if raw[header - 2 : header] != PIT_SCHEMA_VERSION.to_bytes(2, "big"):
            raise ScaleStateError(ScaleStateErrorCode.STATE_VERSION_UNSUPPORTED)
        end = len(raw) - _CHECKSUM_BYTES
        if not compare_digest(sha256(raw[:end]).digest(), raw[end:]):
            raise invalid
        regions = 2 * bin_count(exponent) - 1
        if raw[header] != exponent or end != header + 1 + 8 + 8 * regions:
            raise invalid
        restored = cls(exponent)
        restored.count = int.from_bytes(raw[header + 1 : header + 9], "big")
        restored.counts = np.frombuffer(raw[header + 9 : end], dtype=">u8").astype(np.int64)
        if int(restored.counts.sum()) != restored.count:
            raise invalid
        return restored


@dataclass(frozen=True, slots=True)
class PitJob:
    """Fold a partition into the region counts of the fitted distribution function's values."""

    family: str
    names: tuple[str, ...]
    values: tuple[float, ...]
    exponent: int
    error: float

    def run(self, source: BatchReader, partition: Partition, max_rows: int) -> bytes:
        from veridist.statistics.distributions import cdf

        np = numpy_module()
        cuts = edge_cuts(self.exponent, self.error)
        parameters = dict(zip(self.names, self.values, strict=True))
        part = PitCounts(self.exponent)
        for batch in source.batches(partition, max_rows=max_rows):
            try:
                with np.errstate(all="ignore"):
                    probabilities = cdf(self.family, float64_batch(batch), **parameters)
            except (ArithmeticError, ValueError) as error:
                raise ScaleInferenceError(ScaleInferenceErrorCode.GOF_NOT_REPRESENTABLE) from error
            part.update(probabilities, cuts)
        return part.to_bytes()

    def restore(self, data: bytes) -> PitCounts:
        return PitCounts.from_bytes(data, self.exponent)

    def identity(self) -> PitCounts:
        return PitCounts(self.exponent)

    def checkpoint_key(self) -> tuple[object, ...]:
        return (
            "pit",
            PIT_TAG,
            PIT_SCHEMA_VERSION,
            self.family,
            self.names,
            tuple(value.hex() for value in self.values),
            self.exponent,
            self.error.hex(),
        )


# ---------------------------------------------------------------------------- enclosure


@dataclass(frozen=True, slots=True)
class Enclosure:
    """The Kolmogorov-Smirnov enclosure and the total-variation enclosure of some counts."""

    ks_lower: float
    ks_upper: float
    tv_lower: float
    tv_upper: float
    strip_rows: int


def _outward(numerator: int, denominator: int, *, up: bool) -> float:
    value = numerator / denominator
    return min(nextafter(value, 2.0), 1.0) if up else max(nextafter(value, -1.0), 0.0)


def enclose(counts: Any, rows: int, exponent: int) -> Enclosure:
    """The enclosure of the Kolmogorov-Smirnov distance and of the binned total variation.

    ``counts`` are the ``2 B - 1`` region counts of ``rows`` rows. The arithmetic of the
    Kolmogorov-Smirnov bounds is in exact integers (``rows * B`` is below ``MAX_PRODUCT``); each
    bound is one correctly rounded division pushed outward by one unit in the last place.
    ``ValueError`` for fewer than two bins.
    """

    if exponent < MIN_BIN_EXPONENT:
        raise ValueError("exponent must be at least MIN_BIN_EXPONENT")
    np = numpy_module()
    bins = bin_count(exponent)
    cumulative = np.cumsum(counts)
    strips = counts[1::2]
    # A[j]: rows under the strip of edge j; Z[j]: rows under its end (A[0] = 0, Z[B] = rows).
    under = np.zeros(bins + 1, dtype=np.int64)
    under[1:bins] = cumulative[0 : 2 * bins - 2 : 2]
    over = np.zeros(bins + 1, dtype=np.int64)
    over[1:bins] = cumulative[1 : 2 * bins - 1 : 2]
    over[bins] = rows
    edges = np.arange(bins + 1, dtype=np.int64)
    # Numerators over rows * B of Z_{j+1} / n - e_j and e_{j+1} - A_j / n, for the bins j < B.
    upper = max(
        0,
        int((over[1:] * bins - edges[:bins] * rows).max()),
        int((edges[1:] * rows - under[:bins] * bins).max()),
    )
    # ... and of A_j / n - e_j and e_j - Z_j / n for the interior edges.
    lower = max(
        0,
        int((under[1:bins] * bins - edges[1:bins] * rows).max()),
        int((edges[1:bins] * rows - over[1:bins] * bins).max()),
    )
    denominator = rows * bins
    fewest = counts[0::2]
    most = fewest + np.concatenate(([0], strips)) + np.concatenate((strips, [0]))
    expected = 1.0 / bins
    first = fewest / rows - expected
    last = most / rows - expected
    nearest = np.maximum(0.0, np.maximum(first, -last))
    farthest = np.maximum(np.abs(first), np.abs(last))
    return Enclosure(
        _outward(lower, denominator, up=False),
        _outward(upper, denominator, up=True),
        0.5 * fsum(nearest.tolist()),
        0.5 * fsum(farthest.tolist()),
        int(strips.sum()),
    )


def decide(enclosure: Enclosure, threshold: float | None) -> GofDecision | None:
    """The decision on ``threshold``: ``None`` without one."""

    if threshold is None:
        return None
    if enclosure.ks_upper <= threshold:
        return GofDecision.PASS
    if enclosure.ks_lower > threshold:
        return GofDecision.FAIL
    return GofDecision.INCONCLUSIVE


@dataclass(frozen=True, slots=True)
class LargeNGof:
    """The large-n goodness-of-fit report of one scale fit.

    ``ks_lower`` and ``ks_upper`` enclose the Kolmogorov-Smirnov distance (certified up to
    ``cdf_error``, the declared evaluation error of the fitted distribution function), ``tv_lower``
    and ``tv_upper`` the binned total variation distance of the histogram of the probability
    integral transform from the uniform one, ``strip_rows`` the number of rows within
    ``cdf_error`` of a bin edge, ``counts_digest`` the SHA-256 of the canonical counts. ``decision``
    is ``None`` unless a ``threshold`` was given.
    """

    family: FamilyId
    rows: int
    bins: int
    cdf_error: float
    ks_lower: float
    ks_upper: float
    tv_lower: float
    tv_upper: float
    strip_rows: int
    counts_digest: str
    threshold: float | None
    decision: GofDecision | None
    scale: ScaleProvenance


def _checked_threshold(threshold: object) -> float | None:
    if threshold is None:
        return None
    if type(threshold) is not float:
        raise TypeError("threshold must be a float or None")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return threshold


def large_n_gof(
    source: Source,
    result: ScaleFitResult,
    *,
    bin_exponent: int = DEFAULT_BIN_EXPONENT,
    threshold: float | None = None,
    inference_pass_budget: int = GOF_PASS_BUDGET_DEFAULT,
    max_rows: int = DEFAULT_MAX_ROWS,
    executor: Any = None,
    checkpoint: StoreArgument | SQLitePartitionStore = None,
    resume: bool = False,
    recompute_changed: bool = False,
) -> LargeNGof:
    """The large-n goodness-of-fit report of the successful scale fit ``result`` of ``source``.

    ``bin_exponent`` is ``k`` of the ``2**k`` bins (``MIN_BIN_EXPONENT`` to ``MAX_BIN_EXPONENT``),
    ``threshold`` the caller's bound on the Kolmogorov-Smirnov distance. The pass is a single read;
    a source that can be read once is enough. ``inference_pass_budget``, ``max_rows``,
    ``executor``, ``checkpoint``, ``resume`` and ``recompute_changed`` are as for
    :func:`~veridist.scale._profile_intervals.profile_intervals`. ``ScaleInferenceError`` with
    ``GOF_FIT_NOT_AVAILABLE`` for a failed fit, ``GOF_NOT_REPRESENTABLE`` when the fitted
    distribution function is not a probability at some observation (including a kernel that cannot
    be evaluated there) and ``GOF_RESOLUTION_UNSUPPORTED`` when its error is too large for that many
    bins; ``ScaleSourceError`` with ``PARTITION_CHANGED`` when the source has another number of rows
    than the fit read.
    """

    fit = getattr(result, "fit", None)
    if not isinstance(fit, FitSuccess):
        raise ScaleInferenceError(ScaleInferenceErrorCode.GOF_FIT_NOT_AVAILABLE)
    if type(bin_exponent) is not int:
        raise TypeError("bin_exponent must be a built-in integer")
    if not MIN_BIN_EXPONENT <= bin_exponent <= MAX_BIN_EXPONENT:
        raise ValueError("bin_exponent is outside the supported range")
    checked = _checked_threshold(threshold)
    options = check_options(
        source,
        max_rows=max_rows,
        budget=inference_pass_budget,
        spool=None,
        executor=executor,
        checkpoint=checkpoint,
        resume=resume,
        recompute_changed=recompute_changed,
    )
    rows = result.scale.row_count
    bins = bin_count(bin_exponent)
    if rows * bins >= MAX_PRODUCT:
        raise ScaleStateError(ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)
    parameters: Mapping[str, float] = fit.parameters
    error = cdf_error_bound(fit.family, parameters)
    edge_cuts(bin_exponent, error)
    job = PitJob(
        fit.family.value,
        tuple(parameters),
        tuple(parameters.values()),
        bin_exponent,
        error,
    )
    bounds = inference_bounds(options.budget, GOF_PASSES, GOF_PASSES)
    state = result.scale.state

    def work(ledger: PassLedger, reader: Source, engine: PassEngine) -> PitCounts:
        partitions = tuple(ledger.begin(PassRole.GOF, 1))
        total: PitCounts = engine.run(reader, partitions, job)
        if total.count != rows:
            raise ScaleSourceError(ScaleSourceErrorCode.PARTITION_CHANGED)
        return total

    total, record = run_inference(
        source,
        options,
        bounds,
        kind=_JOB_KIND,
        state=state,
        rows=rows,
        schema=_SCHEMA,
        plan=(*job.checkpoint_key(), "" if state is None else state.digest_sha256),
        rereads=False,
        work=work,
    )
    enclosure = enclose(total.counts, rows, bin_exponent)
    return LargeNGof(
        fit.family,
        rows,
        bins,
        error,
        enclosure.ks_lower,
        enclosure.ks_upper,
        enclosure.tv_lower,
        enclosure.tv_upper,
        enclosure.strip_rows,
        sha256(total.to_bytes()).hexdigest(),
        checked,
        decide(enclosure, checked),
        record,
    )


__all__ = [
    "DEFAULT_BIN_EXPONENT",
    "GOF_PASS_BUDGET_DEFAULT",
    "MAX_BIN_EXPONENT",
    "MIN_BIN_EXPONENT",
    "Enclosure",
    "GofDecision",
    "LargeNGof",
    "PitCounts",
    "PitJob",
    "cdf_error_bound",
    "decide",
    "edge_cuts",
    "enclose",
    "large_n_gof",
]
