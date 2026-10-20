"""Independent references and shared data for the tests of inference from a scale fit.

Nothing here uses the code under test. The profile-likelihood intervals are rebuilt from the
definition at 40 digits with ``mpmath`` on the exact binary64 data: the endpoints of the interval of
the shape are the roots of the profile deviance, and the endpoints of every other quantity are the
extremes of that quantity over the likelihood region, found as the extremes of its value on the
boundary curve.
"""

from __future__ import annotations

from collections.abc import Callable
from math import fsum
from typing import Any

import mpmath
import numpy as np

mpmath.mp.dps = 40


def chi_square_quantile(level: float) -> Any:
    """The ``level`` quantile of chi-square with one degree of freedom, at 40 digits."""

    target = mpmath.mpf(level)
    return mpmath.findroot(
        lambda x: mpmath.erf(mpmath.sqrt(x / 2)) - target,
        mpmath.mpf(3.84),
        tol=mpmath.mpf(10) ** -35,
    )


class Likelihood:
    """The two-parameter log-likelihood of exact data, with its maximum and boundary curves.

    ``kind`` is ``"weibull"`` (parameters ``k`` and ``s``, the scale) or ``"gumbel"`` (``sigma``
    and ``mu``, the location); ``coordinate`` is the first parameter (``k`` or ``sigma``) and the
    second is profiled out in closed form.
    """

    def __init__(self, values: np.ndarray, kind: str) -> None:
        self.kind = kind
        self.n = len(values)
        raw = [mpmath.mpf(float(v)) for v in values]
        if kind == "weibull":
            self.logs = [mpmath.log(v) for v in raw]
            self.total_log = mpmath.fsum(self.logs)
            self.center = self.total_log / self.n
            self.data = [item - self.center for item in self.logs]
        else:
            unit = max(abs(v) for v in raw)
            self.unit = unit
            self.points = [v / unit for v in raw]
            self.total = mpmath.fsum(self.points)
            self.data = self.points

    # -- weibull: k is the shape, the scale is s -------------------------------------

    def _power_sum(self, k: Any) -> Any:
        if self.kind == "weibull":
            return mpmath.fsum(mpmath.exp(k * item) for item in self.data)
        low = min(self.data)
        return mpmath.fsum(mpmath.exp(-(item - low) / k) for item in self.data)

    def profile(self, k: Any) -> Any:
        """The profile log-likelihood of ``k`` (``sigma`` for the Gumbel family)."""

        s0 = self._power_sum(k)
        if self.kind == "weibull":
            return (
                self.n * mpmath.log(k)
                + (k - 1) * self.total_log
                - self.n * k * self.center
                - self.n * mpmath.log(s0 / self.n)
                - self.n
            )
        low = min(self.data)
        mu = low - k * mpmath.log(s0 / self.n)
        return self.log_likelihood(k, mu)

    def log_likelihood(self, k: Any, second: Any) -> Any:
        if self.kind == "weibull":
            scale = second
            return (
                self.n * mpmath.log(k)
                - self.n * k * mpmath.log(scale)
                + (k - 1) * self.total_log
                - mpmath.fsum(
                    mpmath.exp(k * (item + self.center - mpmath.log(scale))) for item in self.data
                )
            )
        sigma, mu = k, second
        return -self.n * mpmath.log(sigma) - mpmath.fsum(
            (p - mu) / sigma + mpmath.exp(-(p - mu) / sigma) for p in self.points
        )

    def maximum(self, start: float) -> tuple[Any, Any]:
        """``(k, l)`` at the maximum: the root of the derivative of the profile."""

        k = mpmath.findroot(
            lambda x: mpmath.diff(self.profile, x), mpmath.mpf(start), tol=mpmath.mpf(10) ** -25
        )
        return k, self.profile(k)

    def second_at(self, k: Any) -> Any:
        """The profiled-out parameter at ``k``: the scale (Weibull) or the location (Gumbel)."""

        s0 = self._power_sum(k)
        if self.kind == "weibull":
            return mpmath.exp(self.center + mpmath.log(s0 / self.n) / k)
        return min(self.data) - k * mpmath.log(s0 / self.n)

    def boundary(self, k: Any, floor: Any, upper: bool) -> Any:
        """The profiled-out parameter on the boundary ``l = floor`` at ``k`` (either side)."""

        best = self.second_at(k)
        if self.kind == "weibull":
            s0 = self._power_sum(k)
            total = s0 * mpmath.exp(k * self.center)
            c = floor - self.n * mpmath.log(k) - (k - 1) * self.total_log
            peak = self.n / total

            def f(v: Any) -> Any:
                return self.n * mpmath.log(v) - total * v - c

            return self._side(f, peak, lambda v: v ** (-1 / k), upper, best, invert=True)
        s0 = mpmath.fsum(mpmath.exp(-p / k) for p in self.points)
        c = floor + self.n * mpmath.log(k) + self.total / k

        def g(t: Any) -> Any:
            return self.n * mpmath.log(t) - t * s0 - c

        return self._side(g, self.n / s0, lambda t: k * mpmath.log(t), upper, best, invert=False)

    def _side(
        self,
        f: Callable[[Any], Any],
        peak: Any,
        convert: Callable[[Any], Any],
        upper: bool,
        best: Any,
        invert: bool,
    ) -> Any:
        """The root of ``f`` on the side of ``peak`` that gives the larger (or smaller) value."""

        # the Weibull scale decreases with v, the Gumbel location increases with t
        go_down = (upper and invert) or (not upper and not invert)
        step = mpmath.mpf(1) / 1000
        edge = peak
        for _ in range(200):
            edge = edge * (1 - step) if go_down else edge * (1 + step)
            if f(edge) < 0:
                break
            step *= 2
        root = mpmath.findroot(f, (peak, edge), solver="anderson", tol=mpmath.mpf(10) ** -30)
        return convert(root)

    def interval_of_k(
        self, start: float, delta: Any, lower_start: float, upper_start: float
    ) -> tuple[Any, Any, Any]:
        """The interval of ``k`` and the maximum: roots of ``2 (l_max - profile(k)) = delta``."""

        k_hat, maximum = self.maximum(start)

        def deviance(k: Any) -> Any:
            return 2 * (maximum - self.profile(k)) - delta

        low = mpmath.findroot(deviance, mpmath.mpf(lower_start), tol=mpmath.mpf(10) ** -25)
        high = mpmath.findroot(deviance, mpmath.mpf(upper_start), tol=mpmath.mpf(10) ** -25)
        return low, high, maximum

    def extreme(
        self,
        quantity: Callable[[Any, Any], Any],
        k_low: Any,
        k_high: Any,
        maximum: Any,
        delta: Any,
        upper: bool,
        side_upper: bool,
        iterations: int = 90,
    ) -> Any:
        """The largest (``upper``) or smallest value of ``quantity(k, second)`` along one side
        of the boundary of the likelihood region, over ``[k_low, k_high]`` by ``iterations``
        steps of golden section (the interval is cut to ``0.618**iterations`` of its width, and
        the value is stationary at an interior extreme and read exactly at the ends)."""

        floor = maximum - delta / 2
        sign = 1 if upper else -1

        def along(k: Any) -> Any:
            return sign * quantity(k, self.boundary(k, floor, side_upper))

        ratio = (mpmath.sqrt(5) - 1) / 2
        left, right = mpmath.mpf(k_low), mpmath.mpf(k_high)
        first = right - ratio * (right - left)
        second = left + ratio * (right - left)
        first_value, second_value = along(first), along(second)
        for _ in range(iterations):
            if first_value >= second_value:
                right, second, second_value = second, first, first_value
                first = right - ratio * (right - left)
                first_value = along(first)
            else:
                left, first, first_value = first, second, second_value
                second = left + ratio * (right - left)
                second_value = along(second)
        best = max(first_value, second_value, along(mpmath.mpf(k_low)), along(mpmath.mpf(k_high)))
        return sign * best


def weibull_data(shape: float, scale: float, size: int, seed: int) -> np.ndarray:
    return scale * np.random.default_rng(seed).weibull(shape, size)


def gumbel_data(location: float, scale: float, size: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).gumbel(location, scale, size)


def relative(value: float, reference: Any) -> float:
    return float(abs(mpmath.mpf(value) - reference) / abs(reference))


def fsum_deviation(values: list[float]) -> float:
    return fsum(values)
