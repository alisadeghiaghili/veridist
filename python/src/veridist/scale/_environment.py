"""The numeric environment digest: what a libm-dependent state's bytes are tied to.

The logarithm in the gamma and lognormal states is the platform ``numpy.log``. It is
exact in the sums but not correctly rounded per element, so a state built on one
platform is bit-identical to a state of the same values on another only when both
platforms evaluate ``log`` (and ``exp``, which the likelihood of every fit uses)
identically. The digest records that fact without assuming it: it is the SHA-256 of
the numpy version and of the results of ``numpy.log`` and ``numpy.exp`` over a fixed
reference sample, so two environments agree on the digest exactly when they agree on
every one of those results. A difference tells a reader that the bit-level claim of
the result does not extend across the two environments, and that only the declared
tolerance does.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Final

from veridist.scale._kernels import numpy_module

REFERENCE_SAMPLE_SIZE: Final = 1024
"""Number of values in the reference sample."""

DIGEST_DOMAIN: Final = b"veridist.scale.environment.v1"
_MULTIPLIER: Final = 2654435761
_MODULUS: Final = 1 << 32
_EXPONENT_PERIOD: Final = 61
_EXPONENT_OFFSET: Final = 30
_LENGTH_BYTES: Final = 8


def reference_sample() -> Any:
    """The fixed reference sample: positive float64 values over about 18 decades.

    Value ``i`` is ``(1 + r_i / 2**32) * 2**e_i`` with ``r_i = (i * 2654435761) mod 2**32``
    and ``e_i = (i mod 61) - 30``. Both factors are computed in exact integer and
    power-of-two arithmetic, so the sample is the same on every platform.
    """

    np = numpy_module()
    values = [
        (1.0 + ((index * _MULTIPLIER) % _MODULUS) / _MODULUS)
        * 2.0 ** ((index % _EXPONENT_PERIOD) - _EXPONENT_OFFSET)
        for index in range(REFERENCE_SAMPLE_SIZE)
    ]
    return np.array(values, dtype=np.float64)


def numpy_version() -> str:
    """The version string of the numpy that evaluates the kernels."""

    return str(numpy_module().__version__)


def _framed(chunk: bytes) -> bytes:
    return len(chunk).to_bytes(_LENGTH_BYTES, "big") + chunk


def numeric_environment_digest(sample: Any = None) -> str:
    """The lowercase hex SHA-256 of the numpy version and the log and exp reference results.

    ``sample`` is a one-dimensional positive float64 array; ``None`` selects
    :func:`reference_sample`. The digest covers the numpy version, the sample, the
    results of ``numpy.log`` over it and the results of ``numpy.exp`` over those
    logarithms, each length-prefixed in binary64 little-endian bytes, so a change of
    any of them changes the digest. ``TypeError`` for a sample that is not a float64
    ``numpy.ndarray``; ``ValueError`` for one that is not one-dimensional, non-empty, finite
    and positive.
    """

    np = numpy_module()
    values = reference_sample() if sample is None else sample
    if not isinstance(values, np.ndarray) or values.dtype != np.float64:
        raise TypeError("sample must be a float64 numpy.ndarray")
    if values.ndim != 1:
        raise ValueError("sample must be one-dimensional")
    if values.size == 0 or not (values > 0.0).all() or not np.isfinite(values).all():
        raise ValueError("sample must be non-empty with finite positive values")
    logarithms = np.log(values)
    exponentials = np.exp(logarithms)
    digest = sha256(DIGEST_DOMAIN)
    digest.update(_framed(numpy_version().encode("ascii")))
    for array in (values, logarithms, exponentials):
        digest.update(_framed(np.ascontiguousarray(array, dtype="<f8").tobytes()))
    return digest.hexdigest()


__all__ = [
    "DIGEST_DOMAIN",
    "REFERENCE_SAMPLE_SIZE",
    "numeric_environment_digest",
    "numpy_version",
    "reference_sample",
]
