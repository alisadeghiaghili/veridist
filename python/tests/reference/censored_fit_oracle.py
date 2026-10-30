"""An independent maximum-likelihood oracle: the log-likelihood and its maximum in ``mpmath``.

Used only by tests.  The log-likelihood of a sample is the sum of the contributions of its rows
at working precision (the densities, cumulative values and window probabilities of
:mod:`tests.reference.censored_oracle`, which share nothing with the library's kernels), and
its maximum is found by Newton iteration in unconstrained coordinates (the logarithm of every
positive parameter) with central-difference gradients and Hessians at working precision.  The
iteration starts at the point it is given (the library's estimate, or any nearby point), so it
certifies that this point is, to the printed digits, a stationary point of the true likelihood
with a negative definite Hessian; it does not search for other maxima.
"""

from __future__ import annotations

import math

import mpmath

from tests.reference import censored_oracle as oracle

NAMES = {
    "exponential": ("rate",),
    "weibull_min": ("shape", "scale"),
    "lognormal": ("mu_log", "sigma_log"),
    "gamma": ("shape", "scale"),
    "normal": ("mu", "sigma"),
    "gumbel_right": ("location", "scale"),
}
#: Which parameters are positive (searched on the log scale).
POSITIVE = {
    "exponential": (True,),
    "weibull_min": (True, True),
    "lognormal": (False, True),
    "gamma": (True, True),
    "normal": (False, True),
    "gumbel_right": (False, True),
}

Row = tuple[str, float, float | None, tuple[float, float] | None]


def _row_term(family: str, parameters: tuple[mpmath.mpf, ...], row: Row) -> mpmath.mpf:
    """The contribution of one row at the working precision."""

    kind, point, upper, window = row
    if kind == "exact":
        numerator = oracle._density(family, parameters)(mpmath.mpf(point))
    else:
        low, high = {
            "right": (point, math.inf),
            "left": (-math.inf, point),
            "interval": (point, upper),
        }[kind]
        if window is not None:
            low, high = max(low, window[0]), min(high, window[1])
        found = oracle.log_probability(family, parameters, low, high)  # type: ignore[arg-type]
        assert found is not None
        numerator = found
    if window is not None:
        found = oracle.log_probability(family, parameters, window[0], window[1])
        assert found is not None
        numerator = numerator - found
    return numerator


def log_likelihood(family: str, parameters: tuple[mpmath.mpf, ...], rows: list[Row]) -> mpmath.mpf:
    """The log-likelihood of ``rows`` at ``parameters``, at the current working precision."""

    return mpmath.fsum(_row_term(family, parameters, row) for row in rows)


def _to_parameters(family: str, coordinates: list[mpmath.mpf]) -> tuple[mpmath.mpf, ...]:
    return tuple(
        mpmath.exp(value) if positive else value
        for value, positive in zip(coordinates, POSITIVE[family], strict=True)
    )


def maximise(
    family: str,
    rows: list[Row],
    start: tuple[float, ...],
    *,
    digits: int = 30,
    iterations: int = 6,
) -> tuple[tuple[mpmath.mpf, ...], mpmath.mpf, list[mpmath.mpf]]:
    """``(parameters, log-likelihood, gradient)`` of the stationary point reached from ``start``.

    Newton iteration at ``digits`` digits in coordinates ``log`` of each positive parameter;
    the gradient returned is the final gradient in those coordinates (it should be at the
    working-precision noise level).
    """

    with mpmath.workdps(digits + 15):
        dimension = len(start)
        coordinates = [
            mpmath.log(mpmath.mpf(value)) if positive else mpmath.mpf(value)
            for value, positive in zip(start, POSITIVE[family], strict=True)
        ]

        def f(point: list[mpmath.mpf]) -> mpmath.mpf:
            return log_likelihood(family, _to_parameters(family, point), rows)

        first = mpmath.mpf(10) ** (-(digits // 3))
        second = mpmath.mpf(10) ** (-(digits // 5))
        gradient = [mpmath.mpf(0)] * dimension
        for _ in range(iterations):
            centre = f(coordinates)
            gradient = []
            for i in range(dimension):
                up, down = list(coordinates), list(coordinates)
                up[i] += first
                down[i] -= first
                gradient.append((f(up) - f(down)) / (2 * first))
            hessian = mpmath.matrix(dimension, dimension)
            for i in range(dimension):
                up, down = list(coordinates), list(coordinates)
                up[i] += second
                down[i] -= second
                hessian[i, i] = (f(up) - 2 * centre + f(down)) / second**2
                for j in range(i + 1, dimension):
                    pp, pm, mp_, mm = (list(coordinates) for _ in range(4))
                    pp[i] += second
                    pp[j] += second
                    pm[i] += second
                    pm[j] -= second
                    mp_[i] -= second
                    mp_[j] += second
                    mm[i] -= second
                    mm[j] -= second
                    hessian[i, j] = hessian[j, i] = (f(pp) - f(pm) - f(mp_) + f(mm)) / (
                        4 * second**2
                    )
            step = mpmath.lu_solve(hessian, mpmath.matrix(gradient))
            coordinates = [c - step[i] for i, c in enumerate(coordinates)]
            if max(abs(step[i]) for i in range(dimension)) < mpmath.mpf(10) ** (-(digits - 4)):
                break
        parameters = _to_parameters(family, coordinates)
        value = f(coordinates)
        eigenvalues = mpmath.eigsy(hessian, eigvals_only=True)
        assert all(value_ < 0 for value_ in eigenvalues), "the Hessian is not negative definite"
        return parameters, value, gradient


def relative_difference(found: tuple[float, ...], reference: tuple[mpmath.mpf, ...]) -> float:
    """The largest ``|found - reference| / |reference|`` over the parameters."""

    return max(
        float(abs(mpmath.mpf(a) - b) / abs(b)) for a, b in zip(found, reference, strict=True)
    )


def generate(path: str, digits: int = 30) -> None:
    """Write the constants of every cell to ``path``: the stationary point and its value.

    The point is found at ``digits`` digits starting from the library's estimate; run as
    ``python -m tests.reference.censored_fit_oracle`` from the package root.
    """

    from tests.reference import censored_fit_cells as cells
    from veridist import fit

    lines = [
        '"""Constants of the fit reference tests: the maximum of each cell at 30 digits.',
        "",
        "Generated by ``python -m tests.reference.censored_fit_oracle`` (see",
        "``tests/reference/censored_fit_oracle.py``): for every cell of",
        "``tests/reference/censored_fit_cells.py`` the stationary point of the log-likelihood",
        "found by Newton iteration at 30 digits in ``mpmath``, rounded to binary64, and the",
        "log-likelihood there.  Not to be edited by hand.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "MAXIMA: dict[str, tuple[tuple[float, ...], float]] = {",
    ]
    for cell in cells.cells():
        result = fit(cell.family, cells.observations(cell))
        assert hasattr(result, "parameters"), (cell.key, result)
        rows = cells.rows(cell)
        start = tuple(result.parameters.values())
        parameters, value, gradient = maximise(cell.family, rows, start, digits=digits)
        lines.append(
            f'    "{cell.key}": ({tuple(float(p) for p in parameters)!r}, {float(value)!r}),'
        )
    lines.append("}")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    generate("tests/reference/censored_fit_constants.py")
