"""Maximum-likelihood estimates of the one-pass families from the exact statistics of a state.

Every function takes a folded state and returns the result type of the in-memory fit
of the same family: the same success class with the same fields, or the same failure
class with the same code. The estimates are computed from the exact rational
statistics of the state and rounded once; nothing here sees an observation.

* exponential: ``rate = n / sum t`` and the checks of the in-memory fit, which are
  functions of ``n`` and ``sum t`` only, by building the same reduction state.
* normal: ``mu = sum x / n`` rounded once and ``sigma`` the correctly rounded square root
  of the exact rational ``(sum x*x - (sum x)**2 / n) / n``. The log-likelihood is
  evaluated from the exact sum of squares about ``mu``, so it is finite for any
  magnitude of the data.
* lognormal: the same on ``log x``.
* gamma: the shape is the root of ``log k - digamma(k) = log(mean x) - mean(log x)``
  and the scale is ``mean x / k``. The right-hand side is evaluated from the exact
  means. The root is found by a bisection in the shape on a fixed bracket with a
  constant iteration cap; the left side is strictly decreasing in ``k``.

Degenerate samples. The centred sum of squares of the normal and lognormal states is
exact (the sums of squares hold the true squares), and it equals
``(1 / n) * sum over pairs of (x_i - x_j)**2``, so it is zero exactly when all values (all
logarithms, for the lognormal) are equal; that and nothing else is degenerate, and a sample
with any spread, however small relative to the size of the values, is estimated. The gamma
family compares ``log(mean x) - mean(log x)`` with the noise of the platform logarithm:
identical values give a difference of a few units in the last place of ``log x``, and
a sample whose gap is that small has a shape beyond the range the in-memory fit searches
(``exp(20)``) unless its values are identical.
"""

from __future__ import annotations

from fractions import Fraction
from math import exp, fsum, isqrt, log, log1p, pi, sqrt
from typing import Final

from veridist.families.exponential import (
    ExponentialFit,
    ExponentialFitFailure,
    ExponentialFitFailureCode,
    fit_exponential_reduction_state,
)
from veridist.families.gamma import (
    GammaFit,
    GammaFitFailure,
    GammaFitFailureCode,
    GammaFitSuccess,
)
from veridist.families.lognormal import (
    LognormalFit,
    LognormalFitFailure,
    LognormalFitFailureCode,
    LognormalFitSuccess,
)
from veridist.families.normal import (
    NormalFit,
    NormalFitFailure,
    NormalFitFailureCode,
    NormalFitSuccess,
)
from veridist.scale._errors import ScaleStateError
from veridist.scale._evidence import StateEvidence, model_of, squares_about
from veridist.scale._one_pass import (
    ExactStatistic,
    ExponentialState,
    GammaState,
    LognormalState,
    NormalState,
)
from veridist.statistics._special import digamma
from veridist.statistics.exponential import ExponentialReductionState

_HALF_LOG_2PI: Final = 0.5 * log(2.0 * pi)

_LOG_NOISE: Final = 2.0**-49
"""Relative noise allowed in ``log(mean) - mean(log)`` of identical values."""

MAX_SHAPE_BISECTIONS: Final = 100
"""Iteration cap of the shape bisection; 62 halvings reach binary64 resolution."""

LOG_SHAPE_LIMIT: Final = 20.0
"""The in-memory fit searches log-shape within plus and minus this limit."""

_SHAPE_UPPER: Final = exp(LOG_SHAPE_LIMIT)
"""The largest shape searched, as the in-memory fit evaluates it."""

_SERIES_FROM: Final = 10.0
_LOG_MINUS_DIGAMMA_SERIES: Final = (
    1.0 / 12.0,
    -1.0 / 120.0,
    1.0 / 252.0,
    -1.0 / 240.0,
    1.0 / 132.0,
    -691.0 / 32760.0,
    1.0 / 12.0,
    -3617.0 / 8160.0,
)


def rounded_sqrt(value: Fraction) -> float:
    """The square root of a positive rational, correctly rounded to binary64.

    The root is bracketed by an integer square root of ``value * 4**shift`` with at
    least 140 significant bits, so no binary64 rounding boundary lies strictly inside the
    bracket; an inexact root is represented by the midpoint of its bracket, which rounds
    like the true root, and an exact one by itself (ties to even). ``OverflowError`` if
    the root is beyond the finite range; it may round to zero below the subnormal range.
    """

    numerator, denominator = value.numerator, value.denominator
    shift = 72 - (numerator.bit_length() - denominator.bit_length()) // 2
    scaled_numerator = numerator << (2 * shift) if shift > 0 else numerator
    scaled_denominator = denominator if shift > 0 else denominator << (-2 * shift)
    root = isqrt(scaled_numerator // scaled_denominator)
    inexact = root * root * scaled_denominator != scaled_numerator
    return float(Fraction(2 * root + inexact, 2) * Fraction(2) ** (-shift))


def gaussian_log_likelihood(
    count: int, sigma: float, squares: Fraction, constant: float
) -> float:
    """``constant - count*log(sigma) - count*log(sqrt(2*pi)) - squares / (2*sigma**2)``.

    ``squares`` is the exact sum of squared deviations about the fitted mean and the last
    term is formed as an exact rational before it is rounded, so it cannot overflow or
    underflow however large or small ``sigma`` is. The four terms are added with ``fsum``.
    """

    ratio = float(squares / Fraction(sigma) ** 2)
    return fsum((constant, -count * log(sigma), -count * _HALF_LOG_2PI, -0.5 * ratio))


def gap_unresolved(gap: float, magnitude: float) -> bool:
    """Whether ``log(mean x) - mean(log x)`` cannot be told from that of identical values.

    ``magnitude`` is ``|log(mean x)| + |mean(log x)|``. For identical values the gap
    is the difference of two evaluations of the same logarithm (the platform's and the
    library's), a few units in the last place of it, far inside ``2**-49 * magnitude``.
    """

    return gap <= _LOG_NOISE * magnitude


def log_minus_digamma(shape: float) -> float:
    """``log(k) - digamma(k)`` for ``k > 0``, accurate also where the two terms nearly cancel.

    From ``k = 10`` on, the asymptotic series ``1/(2k) + sum B_2n / (2n k**2n)`` is
    used, which has no cancellation; below, the difference is formed from the
    library's digamma, whose terms differ by an amount that is not small there.
    """

    if shape < _SERIES_FROM:
        return log(shape) - digamma(shape)
    inverse_square = 1.0 / (shape * shape)
    series = 0.0
    for coefficient in reversed(_LOG_MINUS_DIGAMMA_SERIES):
        series = coefficient + series * inverse_square
    return 0.5 / shape + series * inverse_square


def solve_shape(gap: float) -> float | None:
    """The shape ``k`` with ``log(k) - digamma(k) == gap`` for ``gap > 0``, or ``None``.

    ``None`` when the root lies at or beyond the largest shape the in-memory fit
    searches (``exp(20)``). The smaller limit cannot bind: the gap of finite binary64
    data is below 1500 while the left side at ``exp(-20)`` is about ``5e8``. The
    function is strictly decreasing, so the bracket ``[exp(-20), exp(20)]`` is halved
    geometrically, at most ``MAX_SHAPE_BISECTIONS`` times, until it cannot be split
    in binary64.
    """

    if log_minus_digamma(_SHAPE_UPPER) >= gap:
        return None
    low, high = 1.0 / _SHAPE_UPPER, _SHAPE_UPPER
    for _ in range(MAX_SHAPE_BISECTIONS):
        middle = sqrt(low * high)
        if not low < middle < high:
            break
        if log_minus_digamma(middle) > gap:
            low = middle
        else:
            high = middle
    return sqrt(low * high)


def profile_gap(mean_x: ExactStatistic, mean_log_x: ExactStatistic) -> tuple[float, float]:
    """``log(mean x) - mean(log x)`` and the magnitude its rounding noise scales with.

    The means are exact rationals. Each is rounded once and the rounding residual is
    added back as a correction, so the only error left is that of the platform
    logarithm of the rounded mean. ``fsum`` adds the four parts without rounding
    between them.
    """

    mean = mean_x.rounded
    relative = float((mean_x.exact - Fraction(mean)) / Fraction(mean))
    residual = float(mean_log_x.exact - Fraction(mean_log_x.rounded))
    log_mean = log(mean)
    gap = fsum((log_mean, log1p(relative), -mean_log_x.rounded, -residual))
    return gap, abs(log_mean) + abs(mean_log_x.rounded)


def fit_exponential_state(state: ExponentialState) -> ExponentialFit:
    """The exponential fit of a folded state, by the checks of the in-memory fit."""

    try:
        statistics = state.finalize()
    except ScaleStateError:
        # The only refusal of this state's finalization is a total beyond the float range.
        return ExponentialFitFailure(
            ExponentialFitFailureCode.NUMERICAL_OVERFLOW, state.count, state.count, None
        )
    return fit_exponential_reduction_state(
        ExponentialReductionState(
            statistics.count, statistics.count, statistics.total_time.rounded, 0.0
        )
    )


def fit_normal_state(state: NormalState) -> NormalFit:
    """The normal fit of a folded state."""

    count = state.count

    def failure(code: NormalFitFailureCode) -> NormalFitFailure:
        return NormalFitFailure(code, count, count, 0)

    if count == 0:
        return failure(NormalFitFailureCode.EMPTY_SAMPLE)
    try:
        statistics = state.finalize()
    except ScaleStateError:
        return failure(NormalFitFailureCode.OPTIMIZER_EXHAUSTED)
    assert statistics.mean is not None and statistics.centered_sum_of_squares is not None
    centered = statistics.centered_sum_of_squares.exact
    if centered == 0:
        return failure(NormalFitFailureCode.DEGENERATE_SAMPLE)
    try:
        mu = statistics.mean.rounded
        sigma = rounded_sqrt(centered / count)
        squares = squares_about(
            count, statistics.sum_x.exact, statistics.sum_x_squared.exact, mu
        )
        likelihood = gaussian_log_likelihood(count, sigma, squares, 0.0)
        return NormalFitSuccess(
            mu, sigma, likelihood, count, count, 0, _evidence=StateEvidence(statistics)
        )
    except (ArithmeticError, ValueError):
        return failure(NormalFitFailureCode.OPTIMIZER_EXHAUSTED)


def fit_lognormal_state(state: LognormalState) -> LognormalFit:
    """The lognormal fit of a folded state; identical logarithms are ``OPTIMIZER_EXHAUSTED``."""

    count = state.count

    def failure(code: LognormalFitFailureCode) -> LognormalFitFailure:
        return LognormalFitFailure(code, count, count, 0)

    if count == 0:
        return failure(LognormalFitFailureCode.EMPTY_SAMPLE)
    # Finalization cannot be refused here: a logarithm of a finite positive float has
    # magnitude below 745, so every sum is far below the float range even at 2**64 rows.
    statistics = state.finalize()
    assert (
        statistics.mean_log_x is not None and statistics.centered_sum_of_squares_log_x is not None
    )
    centered = statistics.centered_sum_of_squares_log_x.exact
    if centered == 0:
        return failure(LognormalFitFailureCode.OPTIMIZER_EXHAUSTED)
    try:
        mu_log = statistics.mean_log_x.rounded
        sigma_log = rounded_sqrt(centered / count)
        squares = squares_about(
            count, statistics.sum_log_x.exact, statistics.sum_log_x_squared.exact, mu_log
        )
        likelihood = gaussian_log_likelihood(
            count, sigma_log, squares, -statistics.sum_log_x.rounded
        )
        return LognormalFitSuccess(
            mu_log, sigma_log, likelihood, count, count, 0, _evidence=StateEvidence(statistics)
        )
    except (ArithmeticError, ValueError):
        return failure(LognormalFitFailureCode.OPTIMIZER_EXHAUSTED)


def fit_gamma_state(state: GammaState) -> GammaFit:
    """The gamma fit of a folded state."""

    count = state.count

    def failure(code: GammaFitFailureCode) -> GammaFitFailure:
        return GammaFitFailure(code, count, count, 0)

    if count == 0:
        return failure(GammaFitFailureCode.EMPTY_SAMPLE)
    try:
        statistics = state.finalize()
    except ScaleStateError:
        return failure(GammaFitFailureCode.OPTIMIZER_EXHAUSTED)
    assert statistics.mean_x is not None and statistics.mean_log_x is not None
    gap, magnitude = profile_gap(statistics.mean_x, statistics.mean_log_x)
    if gap_unresolved(gap, magnitude):
        return failure(GammaFitFailureCode.DEGENERATE_SAMPLE)
    shape = solve_shape(gap)
    if shape is None:
        return failure(GammaFitFailureCode.BOUNDARY_SOLUTION)
    try:
        scale = float(statistics.mean_x.exact / Fraction(shape))
        likelihood = model_of(statistics).log_likelihood((shape, scale))
        return GammaFitSuccess(
            shape, scale, likelihood, count, count, 0, _evidence=StateEvidence(statistics)
        )
    except (ArithmeticError, ValueError):
        return failure(GammaFitFailureCode.OPTIMIZER_EXHAUSTED)


__all__ = [
    "LOG_SHAPE_LIMIT",
    "MAX_SHAPE_BISECTIONS",
    "fit_exponential_state",
    "fit_gamma_state",
    "fit_lognormal_state",
    "fit_normal_state",
    "gap_unresolved",
    "gaussian_log_likelihood",
    "log_minus_digamma",
    "profile_gap",
    "solve_shape",
    "rounded_sqrt",
]
