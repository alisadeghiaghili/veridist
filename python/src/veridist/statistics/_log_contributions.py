"""Log-likelihood contributions of exact, censored and truncated observations.

For an observation with censoring set ``C`` and truncation window ``W = (a, b]`` the
contribution is ``log P(X in C and X in W) - log P(X in W)``; without a window the second term
is absent.  Written with ``alpha`` and ``beta`` for the ends of ``C`` read inside the window:

* exact ``x``: ``log f(x)``;
* right-censored at ``c``: ``log P(c, b)``, which is ``log S(c)`` without a window;
* left-censored at ``u``: ``log P(a, u)``, which is ``log F(u)`` without a window;
* interval ``(l, u]``: ``log P(l, u)``;

all minus ``log P(a, b)`` when the observation carries a window, where ``log P(x1, x2)`` is
``log(F(x2) - F(x1))`` from :func:`~veridist.statistics._logprob.kernel_log_window` (one stable
form per side of the distribution, an 8-node quadrature for narrow windows).  Nothing here forms
a probability and then takes its logarithm.

* Exact and right-censored observations without a window of the types of the earlier releases
  keep their own scalar evaluators, so their terms are bit for bit what they were.
* Where the numerator and the denominator share an end and a closed ratio exists, the ratio is
  used so that nothing cancels: for the exponential and the Weibull minimum with a lower
  truncation point ``a`` and no upper one, an exact row is ``log f(x) - log S(a)`` and a
  right-censored row ``log S(c) - log S(a)``, evaluated as ``log(rate) - rate (x - a)`` and
  ``-rate (c - a)`` for the exponential and with ``expm1`` of the log of the ratio of the
  positions for the Weibull.
* Elsewhere the contribution is the difference of two stable log-probabilities, so its absolute
  error is at most the sum of theirs (about ``1e-12`` times the larger of ``1`` and the
  magnitude of the larger term); no protection beyond that bound is claimed.

A term that is not a finite binary64 number is never returned: it is the typed
:class:`~veridist.statistics.log_density.LogDensityErrorCode` of the failure (``-inf`` is
``NUMERICAL_OVERFLOW``, a refused kernel request or NaN ``NONFINITE_LOG_DENSITY``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from math import inf, isfinite, log
from operator import attrgetter
from typing import Any, Final

from veridist.domain._combinations import CENSORING_KINDS, CensoringKind
from veridist.domain.lifetimes import (
    ExactLifetime,
    IntervalCensoredLifetime,
    LeftCensoredLifetime,
    RightCensoredLifetime,
)
from veridist.domain.values import (
    ExactValue,
    IntervalCensoredValue,
    LeftCensoredValue,
    RightCensoredValue,
)
from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.statistics.log_density import (
    LogDensityErrorCode,
    LogDensityFailure,
    LogDensityResult,
    LogDensitySuccess,
    _evaluate_validated_log_density,
)


def _numpy() -> Any:
    """numpy, imported on first use so that importing the package loads no array library."""

    return import_module("numpy")


#: Rows evaluated together; a row's result does not depend on the others in its batch.
BATCH_SIZE: Final = 512
_LOG_TWO: Final = log(2.0)
#: ``|(x - a) / a|`` up to which the log of a quotient is ``log1p`` of that relative difference.
_NEAR: Final = 0.5

KIND: Final[Mapping[type, CensoringKind]] = CENSORING_KINDS
#: The types whose untruncated terms keep the evaluators of the earlier releases.
LEGACY_TYPES: Final = frozenset(
    {ExactLifetime, RightCensoredLifetime, ExactValue, RightCensoredValue}
)
_POINT: Final[Mapping[type, Callable[[Any], float]]] = {
    ExactLifetime: attrgetter("time"),
    RightCensoredLifetime: attrgetter("time"),
    LeftCensoredLifetime: attrgetter("time"),
    IntervalCensoredLifetime: attrgetter("lower"),
    ExactValue: attrgetter("value"),
    RightCensoredValue: attrgetter("value"),
    LeftCensoredValue: attrgetter("value"),
    IntervalCensoredValue: attrgetter("lower"),
}

_Outcome = float | LogDensityErrorCode


def is_new_data(observation: Any) -> bool:
    """Whether ``observation`` is of a kind, or carries a window, that earlier releases lacked."""

    return (
        type(observation) not in LEGACY_TYPES
        or observation.truncation_lower is not None
        or observation.truncation_upper is not None
    )


def _log_quotient(x: Any, a: Any) -> Any:
    """``log(x / a)`` for positive arrays, from the exact relative difference near ``1``.

    Away from ``1`` the binary exponents are split off first, so neither a subnormal input nor
    ``x / a`` overflowing can spoil it.
    """

    np = _numpy()
    with np.errstate(all="ignore"):
        delta = (x - a) / a
        x_mantissa, x_exponent = np.frexp(x)
        a_mantissa, a_exponent = np.frexp(a)
        far = np.log(x_mantissa / a_mantissa) + (x_exponent - a_exponent) * _LOG_TWO
        return np.where(np.abs(delta) <= _NEAR, np.log1p(delta), far)


class _Pending:
    """A row whose term needs the kernels or the closed ratios.

    ``kind`` and ``point`` (the observation, the censoring point or the lower bound), the
    numerator window ``(alpha, beta)`` of a censored row, the truncation ``window`` (``None``
    without one) and, for an exact row, its ``density`` term.
    """

    __slots__ = ("density", "index", "kind", "numerator", "point", "window")

    def __init__(
        self,
        index: int,
        kind: CensoringKind,
        point: float,
        numerator: tuple[float, float] | None,
        window: tuple[float, float] | None,
        density: float | None = None,
    ) -> None:
        self.index = index
        self.kind = kind
        self.point = point
        self.numerator = numerator
        self.window = window
        self.density = density


class ContributionEvaluator:
    """Evaluate the terms of observations of one family at fixed, validated parameters.

    ``legacy_survival(family, parameters, point)`` is the scalar log-survival evaluator of the
    earlier releases, used for right-censored rows without a window; it returns a
    :class:`~veridist.statistics.log_density.LogDensityResult`.
    """

    def __init__(
        self,
        family: FamilyId,
        parameters: Mapping[str, float],
        legacy_survival: Callable[[FamilyId, Mapping[str, float], float], LogDensityResult],
    ) -> None:
        self.family = family
        self.parameters = parameters
        self.legacy_survival = legacy_survival
        self.specification = FAMILY_REGISTRY.families[family]
        self._kernel: Any = None

    def kernel(self) -> Any:
        """The kernel of the family at the parameters, built on first use."""

        if self._kernel is None:
            from veridist.statistics._logprob import build_kernel

            self._kernel = build_kernel(self.family, **self.parameters)
        return self._kernel

    def evaluate(self, observations: Sequence[object]) -> list[_Outcome]:
        """The term of each observation, or the code of its failure, in order.

        Every observation must be of one of the eight observation types.
        """

        outcomes: list[_Outcome | None] = [None] * len(observations)
        pending: list[_Pending] = []
        for index, observation in enumerate(observations):
            outcomes[index] = self._classify(index, observation, pending)
        for index, outcome in self._finish(pending):
            outcomes[index] = outcome
        settled = [outcome for outcome in outcomes if outcome is not None]
        assert len(settled) == len(observations)
        return settled

    def _density(self, point: float) -> float | LogDensityErrorCode:
        evaluated = _evaluate_validated_log_density(self.family, self.parameters, point)
        if type(evaluated) is LogDensityFailure:
            return evaluated.code
        assert isinstance(evaluated, LogDensitySuccess)
        return evaluated.log_density

    def _survival(self, point: float) -> float | LogDensityErrorCode:
        evaluated = self.legacy_survival(self.family, self.parameters, point)
        if type(evaluated) is LogDensityFailure:
            return evaluated.code
        assert isinstance(evaluated, LogDensitySuccess)
        return evaluated.log_density

    def _classify(self, index: int, observation: Any, pending: list[_Pending]) -> _Outcome | None:
        """Settle a row now (a number or a code), or queue it and return ``None``."""

        kind = KIND[type(observation)]
        point = float(_POINT[type(observation)](observation))
        low = observation.truncation_lower
        high = observation.truncation_upper
        window = (
            None
            if low is None and high is None
            else (-inf if low is None else low, inf if high is None else high)
        )
        if kind is CensoringKind.EXACT:
            density = self._density(point)
            if isinstance(density, LogDensityErrorCode) or window is None:
                return density
            pending.append(_Pending(index, kind, point, None, window, density))
            return None
        if kind is CensoringKind.RIGHT:
            if not self.specification.contains(point):
                return LogDensityErrorCode.SUPPORT_VIOLATION
            if window is None:
                return self._survival(point)
            pending.append(_Pending(index, kind, point, (point, window[1]), window))
            return None
        if kind is CensoringKind.LEFT:
            numerator = (-inf if window is None else window[0], point)
        else:
            numerator = (point, float(observation.upper))
        pending.append(_Pending(index, kind, point, numerator, window))
        return None

    def _finish(self, pending: list[_Pending]) -> list[tuple[int, _Outcome]]:
        """The terms of the queued rows; a refused batch is retried row by row."""

        if not pending:
            return []
        try:
            return self._evaluate_pending(pending)
        except ArithmeticError:
            if len(pending) == 1:
                return [(pending[0].index, LogDensityErrorCode.NONFINITE_LOG_DENSITY)]
            settled: list[tuple[int, _Outcome]] = []
            for row in pending:
                settled.extend(self._finish([row]))
            return settled

    def _evaluate_pending(self, pending: list[_Pending]) -> list[tuple[int, _Outcome]]:
        values = self._ratio_rows(pending)
        generic = [row for row in pending if row.index not in values]
        censored = [row for row in generic if row.numerator is not None]
        truncated = [row for row in generic if row.window is not None]
        numerators = _windows([row.numerator for row in censored if row.numerator], self.kernel)
        denominators = _windows([row.window for row in truncated if row.window], self.kernel)
        for row, term in zip(censored, numerators, strict=True):
            values[row.index] = term
        for row, term in zip(truncated, denominators, strict=True):
            base = row.density if row.density is not None else values[row.index]
            values[row.index] = base - term
        return [(row.index, _settle(values[row.index])) for row in pending]

    def _ratio_rows(self, pending: list[_Pending]) -> dict[int, float]:
        """Closed-ratio terms of the exponential and Weibull rows truncated below only."""

        if self.family not in (FamilyId.EXPONENTIAL, FamilyId.WEIBULL_MIN):
            return {}
        rows = [
            row
            for row in pending
            if row.window is not None
            and row.window[0] > 0.0
            and row.window[1] == inf
            and row.kind in (CensoringKind.EXACT, CensoringKind.RIGHT)
        ]
        if not rows:
            return {}
        np = _numpy()
        a = np.array([row.window[0] for row in rows if row.window])
        point = np.array([row.point for row in rows])
        exact = np.array([row.kind is CensoringKind.EXACT for row in rows])
        if self.family is FamilyId.EXPONENTIAL:
            terms = self._exponential_ratio(point, a, exact)
        else:
            terms = self._weibull_ratio(point, a, exact)
        return {row.index: float(term) for row, term in zip(rows, terms.tolist(), strict=True)}

    def _exponential_ratio(self, point: Any, a: Any, exact: Any) -> Any:
        rate = self.parameters["rate"]
        np = _numpy()
        with np.errstate(all="ignore"):
            return np.where(exact, log(rate), 0.0) - rate * (point - a)

    def _weibull_ratio(self, point: Any, a: Any, exact: Any) -> Any:
        from veridist.statistics._logmath import log_ratio

        shape = self.parameters["shape"]
        scale = self.parameters["scale"]
        np = _numpy()
        with np.errstate(all="ignore"):
            moved = -np.exp(shape * log_ratio(a, scale)) * np.expm1(shape * _log_quotient(point, a))
            density = log(shape) - log(scale) + (shape - 1.0) * log_ratio(point, scale)
            return np.where(exact, density, 0.0) + moved


def _windows(ends: list[tuple[float, float]], kernel: Callable[[], Any]) -> list[float]:
    """``log P`` of each window ``(low, high)``, in one kernel call."""

    if not ends:
        return []
    from veridist.statistics._logprob import kernel_log_window

    np = _numpy()
    low = np.array([end[0] for end in ends], dtype=np.float64)
    high = np.array([end[1] for end in ends], dtype=np.float64)
    terms: list[float] = kernel_log_window(kernel(), low, high).tolist()
    return terms


def _settle(value: float) -> _Outcome:
    """A finite term, or the code of why ``value`` is none."""

    if isfinite(value):
        return value
    if value == -inf:
        return LogDensityErrorCode.NUMERICAL_OVERFLOW
    return LogDensityErrorCode.NONFINITE_LOG_DENSITY


__all__ = [
    "BATCH_SIZE",
    "KIND",
    "LEGACY_TYPES",
    "ContributionEvaluator",
    "is_new_data",
]
