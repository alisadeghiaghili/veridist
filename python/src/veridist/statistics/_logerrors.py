"""Typed refusals of the log-probability kernels.

The kernels of :mod:`veridist.statistics._logprob` and
:mod:`veridist.statistics._loggamma` never return a silent NaN.  A request they cannot
serve is an exception that carries a closed, locale-neutral code and, where it helps, the
region of the parameter space it concerns (a range of the gamma shape), never a value
taken from the data.
"""

from __future__ import annotations

from enum import StrEnum

from veridist.engine.errors import VeridistError


class LogProbabilityErrorCode(StrEnum):
    """Closed codes of a refused log-probability evaluation."""

    NONFINITE_INPUT = "nonfinite_input"
    INVALID_WINDOW = "invalid_window"
    GAMMA_SHAPE_REFUSED = "gamma_shape_refused"
    GAMMA_NONCONVERGENCE = "gamma_nonconvergence"


class LogProbabilityError(VeridistError, ArithmeticError):
    """A log-probability evaluation that is refused, with its code and shape region."""

    def __init__(self, code: LogProbabilityErrorCode, region: str | None = None) -> None:
        if type(code) is not LogProbabilityErrorCode:
            raise TypeError("code must be a LogProbabilityErrorCode")
        message = code.value if region is None else f"{code.value} ({region})"
        super().__init__(message)
        self.code = code
        self.region = region


__all__ = ["LogProbabilityError", "LogProbabilityErrorCode"]
