"""Profile-likelihood intervals of a ``weibull_min`` or ``gumbel_right`` scale fit.

``profile_intervals(source, result)`` returns the likelihood-ratio intervals that the in-memory
fit reports for ``method="profile"``, computed from extra passes over the source of a scale fit
(``result``, the value ``fit`` returned for it). The likelihood of both families depends on the
data through the exact sums ``S0(k) = sum exp(k * w)`` only, so each pass evaluates ``S0`` for a
vector of candidate values ``k`` at once, and everything else is a closed form
(:mod:`veridist.scale._profile_math`).

The plan, in passes. The window of ``t = log k`` is the Wald interval of ``k`` widened by
``WINDOW_FACTOR``; its Chebyshev-Lobatto nodes (degree 8, and the parameter the fit ended on, whose
sum must equal the one the fit recorded: a different source is ``PARTITION_CHANGED``) are read in
the first pass. A window whose ends are not both beyond the level is doubled, at most
``MAX_WINDOW_EXPANSIONS`` times. The polynomial through the nodes is refined by nested degrees
(8, 16, 32, 64, 128, one pass each, reading only the new nodes) until two degrees agree to
``SURROGATE_TOLERANCE`` of deviance. The ends of the intervals and the extremes of every other
quantity along the boundary of the likelihood region are then located on the polynomial and read
exactly in one last pass, the confirmation, which also checks the polynomial where it was used. What
is reported is evaluated from those exact sums; the polynomial decides only where to look. A
confirmation that differs from the polynomial by more than ``CONFIRMATION_TOLERANCE`` of deviance is
``PROFILE_NOT_CONVERGED``. At least ``PROFILE_MIN_PASSES`` and at most ``PROFILE_MAX_PASSES`` are
read, within ``inference_pass_budget`` (default ``PROFILE_PASS_BUDGET_DEFAULT``): a budget below
the least is ``PASS_BUDGET_INSUFFICIENT`` before any read, a run that would need a pass beyond
the budget is ``PASS_BUDGET_EXCEEDED`` before that pass is read, and neither returns an interval
or anything that stands for one.

Quantities. ``shape`` and ``scale`` (Weibull) or ``location`` and ``scale`` (Gumbel) are intervals
of the parameters, and ``quantities`` adds the mean, quantiles (B-lives) and survival
probabilities of the Weibull family, which the in-memory code has a profile for. For the Gumbel
family it has none, and the entry is a typed :class:`ProfileUnavailable`.

Every decision is a function of exact sums, so the result is bit for bit the same for every batch
cap, partition layout, executor, spool and resume point of a checkpointed run within one numeric
environment. An interval with an end beyond the search domain of the fit is
``PROFILE_UNBOUNDED``; no side is silently replaced by a Wald or an unbounded end.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import exp, gamma, isfinite, lgamma, log, log1p, sqrt
from types import MappingProxyType
from typing import Any, Final

from veridist.engine.provenance import ScaleProvenance
from veridist.families._reliability import exp_or_inf
from veridist.families.gumbel import GumbelFitSuccess
from veridist.families.registry import FamilyId
from veridist.families.uncertainty import (
    DerivedEstimate,
    ParameterInterval,
    UncertaintyUnavailable,
    _chi_square_1,
    _validated_level,
)
from veridist.families.weibull import WeibullFitSuccess
from veridist.scale._errors import (
    ScaleInferenceError,
    ScaleInferenceErrorCode,
    ScaleSourceError,
    ScaleSourceErrorCode,
)
from veridist.scale._executors import PassEngine
from veridist.scale._fit import ScaleFitResult
from veridist.scale._inference_run import (
    StoreArgument,
    check_options,
    inference_bounds,
    run_inference,
)
from veridist.scale._partition_store import SQLitePartitionStore
from veridist.scale._planner import PassLedger, PassRole
from veridist.scale._power_sums import (
    POWER_SUM_SCHEMA_VERSION,
    POWER_SUM_TAG,
    PowerSumJob,
)
from veridist.scale._profile_basis import ProfileBasis
from veridist.scale._profile_math import (
    CONFIRMATION_TOLERANCE,
    LOG_DOMAIN_LIMIT,
    SURROGATE_DEGREES,
    SURROGATE_TOLERANCE,
    LikelihoodForm,
    Surrogate,
    boundary_at,
    extremum,
    lobatto_logs,
    noise_allowance,
    root_between,
    surrogate_deviance,
)
from veridist.scale._sources import DEFAULT_MAX_ROWS, Source
from veridist.scale._spool import SpoolPolicy
from veridist.statistics.distributions import ppf, sf

PROFILE_PASS_BUDGET_DEFAULT: Final = 6
"""The inference pass budget unless the caller gives one."""

WINDOW_FACTOR: Final = 4.0
"""The first window is this many Wald half-widths (in ``log k``) on each side of the estimate."""

WINDOW_GROWTH: Final = 2.0
"""A window whose ends are not both beyond the level is multiplied by this."""

MAX_WINDOW_EXPANSIONS: Final = 3
MAX_QUANTITIES: Final = 64
"""The most derived quantities one call accepts (each needs at most two confirmation points)."""

PROFILE_MIN_PASSES: Final = 2
"""The first window pass and the confirmation."""

PROFILE_MAX_PASSES: Final = 1 + MAX_WINDOW_EXPANSIONS + (len(SURROGATE_DEGREES) - 1) + 1
"""A window pass and its expansions, every refinement of the polynomial, the confirmation."""

_JOB_KIND: Final = "profile"
_SCHEMA: Final = f"{POWER_SUM_TAG}:{POWER_SUM_SCHEMA_VERSION}"


def profile_plan_identity() -> tuple[object, ...]:
    """The constants that fix the passes of a profile run, as a canonical record."""

    return (
        PROFILE_PASS_BUDGET_DEFAULT,
        WINDOW_FACTOR.hex(),
        WINDOW_GROWTH.hex(),
        MAX_WINDOW_EXPANSIONS,
        LOG_DOMAIN_LIMIT.hex(),
        SURROGATE_DEGREES,
        SURROGATE_TOLERANCE.hex(),
        CONFIRMATION_TOLERANCE.hex(),
    )


class QuantityKind(StrEnum):
    """The derived quantities a profile interval can be asked for."""

    MEAN = "mean"
    QUANTILE = "quantile"
    SURVIVAL = "survival"


@dataclass(frozen=True, slots=True)
class Quantity:
    """A derived quantity of the fitted distribution: its kind and, if it has one, its argument."""

    kind: QuantityKind
    argument: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, QuantityKind):
            raise TypeError("kind must be a QuantityKind")
        if self.kind is QuantityKind.MEAN:
            if self.argument is not None:
                raise ValueError("the mean has no argument")
            return
        if type(self.argument) is not float:
            raise TypeError("the argument of a quantile or a survival must be a float")
        if self.kind is QuantityKind.QUANTILE and not 0.0 < self.argument < 1.0:
            raise ValueError("probability must be strictly between 0 and 1")
        if self.kind is QuantityKind.SURVIVAL and not isfinite(self.argument):
            raise ValueError("time must be finite")

    @property
    def name(self) -> str:
        """The key of the quantity in a result: ``mean``, ``quantile:0.1`` or ``survival:30.0``."""

        if self.argument is None:
            return self.kind.value
        return f"{self.kind.value}:{self.argument!r}"


def mean() -> Quantity:
    """The distribution mean."""

    return Quantity(QuantityKind.MEAN)


def quantile(probability: float) -> Quantity:
    """The ``probability`` quantile (the B-life: ``quantile(0.1)`` is B10)."""

    return Quantity(QuantityKind.QUANTILE, probability)


def survival(time: float) -> Quantity:
    """The survival probability at ``time``."""

    return Quantity(QuantityKind.SURVIVAL, time)


class ProfileUnavailableReason(StrEnum):
    """Why a derived quantity has no profile interval."""

    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    NOT_REPRESENTABLE = "NOT_REPRESENTABLE"


@dataclass(frozen=True, slots=True)
class ProfileUnavailable:
    """The typed absence of a profile interval of a derived quantity."""

    reason: ProfileUnavailableReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ProfileResolution:
    """How well the polynomial stood for the likelihood, in units of deviance.

    ``surrogate_degree`` is the degree that was used, ``surrogate_gap`` the difference between it
    and the next lower one, ``confirmation_residual`` the largest difference between an exact
    deviance and the polynomial's at a point the intervals were evaluated at, and ``tolerance`` the
    allowance for rounding of the exact sums that applies to this count of rows.
    """

    surrogate_degree: int
    surrogate_gap: float
    confirmation_residual: float
    tolerance: float


@dataclass(frozen=True, slots=True)
class ProfileIntervals:
    """Profile-likelihood intervals of one scale fit.

    ``parameters`` maps each parameter name to its :class:`ParameterInterval` (``method`` is
    ``"profile"``), ``derived`` each requested quantity (:attr:`Quantity.name`) to its
    :class:`DerivedEstimate` or a :class:`ProfileUnavailable`. ``deviance`` is the chi-square
    quantile of ``level`` the intervals were cut at. ``scale`` is the provenance of the passes
    that were read.
    """

    family: FamilyId
    level: float
    deviance: float
    parameters: Mapping[str, ParameterInterval]
    derived: Mapping[str, DerivedEstimate | ProfileUnavailable]
    resolution: ProfileResolution
    scale: ScaleProvenance


# ----------------------------------------------------------------------- the quantities


@dataclass(frozen=True, slots=True)
class _Target:
    """A function of ``(t, a)`` that is monotone in ``a``, and what to report from its range."""

    value: Any
    increasing: bool
    finish: Any
    estimate: float = 0.0


def _basis_of(result: object) -> ProfileBasis:
    from veridist.scale._evidence import ProfileEvidence

    fit = getattr(result, "fit", None)
    if not isinstance(fit, WeibullFitSuccess | GumbelFitSuccess):
        raise ScaleInferenceError(ScaleInferenceErrorCode.FIT_NOT_PROFILABLE)
    evidence = getattr(fit, "_evidence", None)
    if not isinstance(evidence, ProfileEvidence) or evidence.basis is None:
        raise ScaleInferenceError(ScaleInferenceErrorCode.FIT_NOT_PROFILABLE)
    return evidence.basis


def _relative_error(result: ScaleFitResult, basis: ProfileBasis) -> float:
    """The relative standard error of ``k`` from the information of the fit, or a stand-in."""

    uncertainty = result.fit.uncertainty()  # type: ignore[union-attr]
    stand_in = 1.0 / sqrt(basis.count)
    if isinstance(uncertainty, UncertaintyUnavailable):
        return stand_in
    name = "shape" if basis.family is FamilyId.WEIBULL_MIN else "scale"
    estimate = uncertainty.estimates[uncertainty.parameter_names.index(name)]
    relative = uncertainty.standard_errors[name] / estimate
    return relative if isfinite(relative) and relative > 0.0 else stand_in


def _weibull_targets(
    basis: ProfileBasis, parameters: Mapping[str, float], quantities: Sequence[Quantity]
) -> tuple[_Target, dict[str, _Target | ProfileUnavailable | DerivedEstimate]]:
    """The scale and the derived quantities of a Weibull fit as targets."""

    anchor = basis.anchor

    def log_scale(point: float, a: float) -> float:
        return anchor + a / exp(point)

    scale = _Target(log_scale, True, exp_or_inf, parameters["scale"])
    derived: dict[str, _Target | ProfileUnavailable | DerivedEstimate] = {}
    shape_hat, scale_hat = parameters["shape"], parameters["scale"]
    for item in quantities:
        try:
            derived[item.name] = _weibull_quantity(item, anchor, shape_hat, scale_hat)
        except (ArithmeticError, ValueError):
            derived[item.name] = ProfileUnavailable(
                ProfileUnavailableReason.NOT_REPRESENTABLE,
                "the quantity is not representable at the fitted parameters",
            )
    return scale, derived


def _weibull_quantity(
    item: Quantity, anchor: float, shape: float, scale: float
) -> _Target | DerivedEstimate:
    """The target of one Weibull quantity, or its degenerate estimate (a survival of 0 or 1)."""

    if item.kind is QuantityKind.MEAN:
        estimate = scale * gamma(1.0 + 1.0 / shape)

        def log_mean(point: float, a: float) -> float:
            k = exp(point)
            return anchor + a / k + lgamma(1.0 + 1.0 / k)

        return _checked_target(_Target(log_mean, True, exp_or_inf, estimate))
    assert item.argument is not None
    if item.kind is QuantityKind.QUANTILE:
        estimate = float(ppf("weibull_min", item.argument, shape=shape, scale=scale))
        level = log(-log1p(-item.argument))

        def log_quantile(point: float, a: float) -> float:
            k = exp(point)
            return anchor + a / k + level / k

        return _checked_target(_Target(log_quantile, True, exp_or_inf, estimate))
    estimate = float(sf("weibull_min", item.argument, shape=shape, scale=scale))
    if not 0.0 < estimate < 1.0:
        return DerivedEstimate(estimate, estimate, estimate, "profile", 0.0)
    log_time = log(item.argument)

    def survival_at(point: float, a: float) -> float:
        return exp(-exp_or_inf(exp(point) * (log_time - anchor) - a))

    return _checked_target(_Target(survival_at, True, float, estimate))


def _checked_target(target: _Target) -> _Target:
    if not isfinite(target.estimate):
        raise ArithmeticError("the estimate is not finite")
    return target


def _gumbel_location(basis: ProfileBasis) -> _Target:
    anchor, spread, offset = basis.anchor, basis.spread, basis.offset

    def location(point: float, a: float) -> float:
        return anchor + spread * (offset - a / exp(point))

    return _Target(location, False, float)


# ------------------------------------------------------------------------------ passes


def _read(
    ledger: PassLedger,
    engine: PassEngine,
    source: Source,
    basis: ProfileBasis,
    role: PassRole,
    logs: Sequence[float],
    parameters: Sequence[float],
) -> list[float]:
    """One pass: the exact ``S0`` of every candidate, in order.

    ``ScaleSourceError`` with ``PARTITION_CHANGED`` if the pass does not read the rows of the fit.
    """

    partitions = tuple(ledger.begin(role, len(parameters), min(logs), max(logs)))
    total = engine.run(source, partitions, PowerSumJob(basis.kernel, tuple(parameters)))
    if total.count != basis.count:
        raise ScaleSourceError(ScaleSourceErrorCode.PARTITION_CHANGED)
    sums: list[float] = total.totals()
    if not all(isfinite(item) and item > 0.0 for item in sums):
        raise ScaleInferenceError(ScaleInferenceErrorCode.PROFILE_NOT_COMPUTABLE)
    return sums


def _psi(basis: ProfileBasis, total: float) -> float:
    return log(total / basis.count)


def _window(center: float, half: float) -> tuple[float, float]:
    """The window ``center +- half`` cut to the search domain, as ``(low, high)``."""

    return max(center - half, -LOG_DOMAIN_LIMIT), min(center + half, LOG_DOMAIN_LIMIT)


@dataclass(frozen=True, slots=True)
class _Fitted:
    """The surrogate that was settled on, with the figures that justify it."""

    surrogate: Surrogate
    gap: float


def _first_window(
    ledger: PassLedger,
    engine: PassEngine,
    source: Source,
    basis: ProfileBasis,
    form: LikelihoodForm,
    delta: float,
    half: float,
) -> Surrogate:
    """Read windows until both ends are beyond the level; the degree-8 surrogate of the last."""

    center = log(basis.rate)
    for expansion in range(MAX_WINDOW_EXPANSIONS + 1):
        low, high = _window(center, half)
        window_center, window_half = 0.5 * (low + high), 0.5 * (high - low)
        logs = lobatto_logs(window_center, window_half, SURROGATE_DEGREES[0])
        parameters = (basis.iterate, *(exp(item) for item in logs))
        sums = _read(
            ledger,
            engine,
            source,
            basis,
            PassRole.PROFILE_WINDOW,
            (log(basis.iterate), *logs),
            parameters,
        )
        if sums[0] != basis.iterate_s0:
            raise ScaleSourceError(ScaleSourceErrorCode.PARTITION_CHANGED)
        values = tuple(_psi(basis, item) for item in sums[1:])
        reaches = [form.deviance(logs[end], values[end]) >= delta for end in (0, len(logs) - 1)]
        if all(reaches):
            return Surrogate(window_center, window_half, SURROGATE_DEGREES[0], values)
        at_edge = (not reaches[0] and low <= -LOG_DOMAIN_LIMIT) or (
            not reaches[1] and high >= LOG_DOMAIN_LIMIT
        )
        if at_edge:
            raise ScaleInferenceError(ScaleInferenceErrorCode.PROFILE_UNBOUNDED)
        half *= WINDOW_GROWTH
    raise ScaleInferenceError(ScaleInferenceErrorCode.PROFILE_NOT_CONVERGED)


def _refined(
    ledger: PassLedger,
    engine: PassEngine,
    source: Source,
    basis: ProfileBasis,
    surrogate: Surrogate,
) -> _Fitted:
    """Double the degree until the two highest degrees agree to the tolerance."""

    allowed = SURROGATE_TOLERANCE + noise_allowance(basis.count)
    while True:
        gap = surrogate.gap()
        if 2.0 * basis.count * gap <= allowed:
            return _Fitted(surrogate, gap)
        position = SURROGATE_DEGREES.index(surrogate.degree)
        if position + 1 == len(SURROGATE_DEGREES):
            raise ScaleInferenceError(ScaleInferenceErrorCode.PROFILE_NOT_CONVERGED)
        degree = SURROGATE_DEGREES[position + 1]
        fresh = lobatto_logs(surrogate.center, surrogate.half_width, degree)[1::2]
        sums = _read(
            ledger,
            engine,
            source,
            basis,
            PassRole.PROFILE_REFINEMENT,
            fresh,
            tuple(exp(item) for item in fresh),
        )
        values: list[float] = []
        for index, old in enumerate(surrogate.values):
            values.append(old)
            if index < len(sums):
                values.append(_psi(basis, sums[index]))
        surrogate = Surrogate(surrogate.center, surrogate.half_width, degree, tuple(values))


# --------------------------------------------------------------------------- the run


@dataclass(frozen=True, slots=True)
class _Located:
    """A target's two extremes on the polynomial: the points where they are attained."""

    lower: float
    upper: float


def _locate(
    target: _Target,
    form: LikelihoodForm,
    delta: float,
    surrogate: Surrogate,
    span: tuple[float, float],
) -> _Located:
    value = target.value

    def along(branch: str) -> Any:
        def at(point: float) -> float:
            bound = boundary_at(form, delta, point, surrogate.value(point))
            return float(value(point, getattr(bound, branch)))

        return at

    upper_branch, lower_branch = ("upper", "lower") if target.increasing else ("lower", "upper")
    high, _ = extremum(along(upper_branch), *span, maximize=True)
    low, _ = extremum(along(lower_branch), *span, maximize=False)
    return _Located(low, high)


def _end(
    target: _Target, form: LikelihoodForm, delta: float, point: float, psi: float, upper: bool
) -> float:
    """The reported end of ``target`` at an exactly evaluated ``point``."""

    value = target.value
    bound = boundary_at(form, delta, point, psi)
    use_upper = upper == target.increasing
    return float(value(point, bound.upper if use_upper else bound.lower))


def _compute(
    ledger: PassLedger,
    source: Source,
    engine: PassEngine,
    *,
    result: ScaleFitResult,
    basis: ProfileBasis,
    level: float,
    quantities: Sequence[Quantity],
) -> tuple[
    Mapping[str, ParameterInterval],
    Mapping[str, DerivedEstimate | ProfileUnavailable],
    ProfileResolution,
]:
    delta = _chi_square_1(level)
    form = LikelihoodForm(basis.count, basis.sum_w, basis.reduced_maximum)
    half = min(WINDOW_FACTOR * sqrt(delta) * _relative_error(result, basis), LOG_DOMAIN_LIMIT)
    fitted = _refined(
        ledger,
        engine,
        source,
        basis,
        _first_window(ledger, engine, source, basis, form, delta, half),
    )
    surrogate = fitted.surrogate
    center = log(basis.rate)

    def excess(point: float) -> float:
        return surrogate_deviance(form, surrogate, point) - delta

    if not excess(center) < 0.0:
        raise ScaleInferenceError(ScaleInferenceErrorCode.PROFILE_NOT_CONVERGED)
    shape_low = root_between(excess, surrogate.low, center)
    shape_high = root_between(excess, surrogate.high, center)
    span = (shape_low, shape_high)

    fit: Any = result.fit
    parameters: Mapping[str, float] = fit.parameters
    if basis.family is FamilyId.WEIBULL_MIN:
        spread_target, derived = _weibull_targets(basis, parameters, quantities)
    else:
        spread_target = _gumbel_location(basis)
        derived = {
            item.name: ProfileUnavailable(
                ProfileUnavailableReason.NOT_IMPLEMENTED,
                "the in-memory fit has no profile likelihood of this quantity for the "
                "right Gumbel family",
            )
            for item in quantities
        }
    targets = {"": spread_target} | {
        name: item for name, item in derived.items() if isinstance(item, _Target)
    }
    located = {name: _locate(item, form, delta, surrogate, span) for name, item in targets.items()}

    points = sorted(
        {shape_low, shape_high, *(p for item in located.values() for p in (item.lower, item.upper))}
    )
    sums = _read(
        ledger,
        engine,
        source,
        basis,
        PassRole.PROFILE_CONFIRMATION,
        points,
        tuple(exp(item) for item in points),
    )
    exact = {point: _psi(basis, total) for point, total in zip(points, sums, strict=True)}
    residual = max(
        abs(form.deviance(point, psi) - surrogate_deviance(form, surrogate, point))
        for point, psi in exact.items()
    )
    tolerance = CONFIRMATION_TOLERANCE + noise_allowance(basis.count)
    if residual > tolerance:
        raise ScaleInferenceError(ScaleInferenceErrorCode.PROFILE_NOT_CONVERGED)

    def interval(name: str) -> tuple[float, float]:
        item, spot = targets[name], located[name]
        low = _end(item, form, delta, spot.lower, exact[spot.lower], False)
        high = _end(item, form, delta, spot.upper, exact[spot.upper], True)
        return low, high

    shape_interval = (exp(shape_low), exp(shape_high))
    low, high = interval("")
    if basis.family is FamilyId.WEIBULL_MIN:
        reported = {
            "shape": ParameterInterval(*shape_interval, level, "profile"),
            "scale": ParameterInterval(
                spread_target.finish(low), spread_target.finish(high), level, "profile"
            ),
        }
    else:
        reported = {
            "location": ParameterInterval(low, high, level, "profile"),
            "scale": ParameterInterval(
                basis.spread / shape_interval[1], basis.spread / shape_interval[0], level, "profile"
            ),
        }
    answers: dict[str, DerivedEstimate | ProfileUnavailable] = {}
    for item in quantities:
        entry = derived[item.name]
        if isinstance(entry, _Target):
            lower, upper = interval(item.name)
            answers[item.name] = DerivedEstimate(
                entry.estimate, entry.finish(lower), entry.finish(upper), "profile", level
            )
        elif isinstance(entry, DerivedEstimate):
            answers[item.name] = DerivedEstimate(
                entry.estimate, entry.lower, entry.upper, "profile", level
            )
        else:
            answers[item.name] = entry
    return (
        MappingProxyType(reported),
        MappingProxyType(answers),
        ProfileResolution(
            fitted.surrogate.degree,
            fitted.gap,
            residual,
            noise_allowance(basis.count),
        ),
    )


def profile_intervals(
    source: Source,
    result: ScaleFitResult,
    *,
    level: float = 0.95,
    quantities: Sequence[Quantity] = (),
    inference_pass_budget: int = PROFILE_PASS_BUDGET_DEFAULT,
    max_rows: int = DEFAULT_MAX_ROWS,
    spool: SpoolPolicy | None = None,
    executor: Any = None,
    checkpoint: StoreArgument | SQLitePartitionStore = None,
    resume: bool = False,
    recompute_changed: bool = False,
) -> ProfileIntervals:
    """Profile-likelihood intervals of the successful scale fit ``result`` of ``source``.

    ``source`` must be the data the fit was made from (a different one is ``PARTITION_CHANGED``
    in the first pass); ``level`` is the confidence level and ``quantities`` the derived quantities
    to report (:func:`mean`, :func:`quantile`, :func:`survival`). ``inference_pass_budget``,
    ``max_rows``, ``spool``, ``executor``, ``checkpoint``, ``resume`` and ``recompute_changed`` are
    those of :func:`~veridist.scale._fit.fit`; a checkpoint holds the job of these passes only,
    so it is a store of its own, not the fit's. ``ScaleInferenceError`` with
    ``FIT_NOT_PROFILABLE`` for a fit that is not a successful Weibull or Gumbel scale fit, and
    the refusals named in the module documentation.
    """

    basis = _basis_of(result)
    checked = _validated_level(level)
    if isinstance(quantities, str) or not isinstance(quantities, Sequence):
        raise TypeError("quantities must be a sequence of Quantity")
    if any(type(item) is not Quantity for item in quantities):
        raise TypeError("quantities must be a sequence of Quantity")
    if len(quantities) > MAX_QUANTITIES:
        raise ValueError("too many quantities")
    if len({item.name for item in quantities}) != len(quantities):
        raise ValueError("quantities must not repeat")
    options = check_options(
        source,
        max_rows=max_rows,
        budget=inference_pass_budget,
        spool=spool,
        executor=executor,
        checkpoint=checkpoint,
        resume=resume,
        recompute_changed=recompute_changed,
    )
    bounds = inference_bounds(options.budget, PROFILE_MIN_PASSES, PROFILE_MAX_PASSES)
    state = result.scale.state
    plan = (
        *profile_plan_identity(),
        checked.hex(),
        "" if state is None else state.digest_sha256,
        tuple(item.name for item in quantities),
    )
    outcome, record = run_inference(
        source,
        options,
        bounds,
        kind=_JOB_KIND,
        state=state,
        rows=basis.count,
        schema=_SCHEMA,
        plan=plan,
        rereads=True,
        work=lambda ledger, reader, engine: _compute(
            ledger,
            reader,
            engine,
            result=result,
            basis=basis,
            level=checked,
            quantities=tuple(quantities),
        ),
    )
    parameters, derived, resolution = outcome
    return ProfileIntervals(
        basis.family, checked, _chi_square_1(checked), parameters, derived, resolution, record
    )


__all__ = [
    "MAX_QUANTITIES",
    "MAX_WINDOW_EXPANSIONS",
    "PROFILE_MAX_PASSES",
    "PROFILE_MIN_PASSES",
    "PROFILE_PASS_BUDGET_DEFAULT",
    "ProfileIntervals",
    "ProfileResolution",
    "ProfileUnavailable",
    "ProfileUnavailableReason",
    "Quantity",
    "QuantityKind",
    "mean",
    "profile_intervals",
    "profile_plan_identity",
    "quantile",
    "survival",
]
