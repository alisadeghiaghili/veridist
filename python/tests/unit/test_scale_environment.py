"""The numeric environment digest: a fixed sample, and a digest that tracks what it covers."""

from __future__ import annotations

import hashlib
import re
import unittest
from collections.abc import Callable
from typing import Any
from unittest import mock

import numpy as np

from veridist.scale import _environment
from veridist.scale._environment import (
    DIGEST_DOMAIN,
    REFERENCE_SAMPLE_SIZE,
    numeric_environment_digest,
    numpy_version,
    reference_sample,
)

PINNED_SAMPLE_SHA256 = "3c8d3fa1d43224a23a5b6b6e6514ed1991ad4e4854f36d151504885dcb1be116"


def framed(chunk: bytes) -> bytes:
    return len(chunk).to_bytes(8, "big") + chunk


def expected_digest(sample: np.ndarray, version: str) -> str:
    logarithms = np.log(sample)
    exponentials = np.exp(logarithms)
    digest = hashlib.sha256(DIGEST_DOMAIN)
    digest.update(framed(version.encode("ascii")))
    for array in (sample, logarithms, exponentials):
        digest.update(framed(array.astype("<f8").tobytes()))
    return digest.hexdigest()


class ReferenceSampleTests(unittest.TestCase):
    def test_the_sample_is_a_fixed_float64_array(self) -> None:
        sample = reference_sample()
        self.assertEqual(REFERENCE_SAMPLE_SIZE, 1024)
        self.assertEqual(sample.shape, (1024,))
        self.assertEqual(sample.dtype, np.float64)
        self.assertTrue((sample > 0.0).all())
        self.assertTrue(np.isfinite(sample).all())

    def test_the_sample_is_the_same_bytes_on_every_platform(self) -> None:
        digest = hashlib.sha256(reference_sample().astype("<f8").tobytes()).hexdigest()
        self.assertEqual(digest, PINNED_SAMPLE_SHA256)

    def test_the_values_follow_the_definition(self) -> None:
        sample = reference_sample()
        self.assertEqual(sample[0], 2.0**-30)
        self.assertEqual(sample[1], (1.0 + 2654435761 / 2**32) * 2.0**-29)
        index = 777
        residue = (index * 2654435761) % 2**32
        self.assertEqual(sample[index], (1.0 + residue / 2**32) * 2.0 ** (index % 61 - 30))
        self.assertEqual(float(sample.min()), 2.0**-30)
        self.assertLess(float(sample.max()), 2.0**31)

    def test_the_sample_spans_many_binary_exponents(self) -> None:
        exponents = np.floor(np.log2(reference_sample()))
        self.assertEqual(int(exponents.min()), -30)
        self.assertEqual(int(exponents.max()), 30)


class DigestTests(unittest.TestCase):
    def test_it_is_a_lowercase_sha256_hex_string(self) -> None:
        self.assertRegex(numeric_environment_digest(), re.compile(r"[0-9a-f]{64}"))

    def test_it_is_stable_within_a_run(self) -> None:
        first = numeric_environment_digest()
        self.assertEqual(numeric_environment_digest(), first)
        self.assertEqual(numeric_environment_digest(reference_sample()), first)

    def test_it_is_the_documented_hash(self) -> None:
        self.assertEqual(
            numeric_environment_digest(), expected_digest(reference_sample(), numpy_version())
        )

    def test_the_version_is_numpys(self) -> None:
        self.assertEqual(numpy_version(), str(np.__version__))

    def test_a_changed_value_changes_the_digest(self) -> None:
        sample = reference_sample()
        changed = sample.copy()
        changed[500] = np.nextafter(changed[500], np.inf)
        self.assertNotEqual(numeric_environment_digest(changed), numeric_environment_digest())
        self.assertEqual(
            numeric_environment_digest(changed), expected_digest(changed, numpy_version())
        )

    def test_a_shorter_sample_changes_the_digest(self) -> None:
        sample = reference_sample()
        self.assertNotEqual(numeric_environment_digest(sample[:-1]), numeric_environment_digest())

    def test_a_reordered_sample_changes_the_digest(self) -> None:
        sample = reference_sample()
        self.assertNotEqual(numeric_environment_digest(sample[::-1]), numeric_environment_digest())

    def test_a_different_numpy_version_changes_the_digest(self) -> None:
        baseline = numeric_environment_digest()
        with mock.patch.object(_environment, "numpy_version", return_value="0.0.0"):
            patched = numeric_environment_digest()
        self.assertNotEqual(patched, baseline)
        self.assertEqual(patched, expected_digest(reference_sample(), "0.0.0"))

    def _perturbed(self, name: str) -> str:
        original: Callable[[Any], Any] = getattr(np, name)

        def shifted(values: Any) -> Any:
            result = original(values)
            result[0] = np.nextafter(result[0], np.inf)
            return result

        with mock.patch.object(np, name, shifted):
            return numeric_environment_digest()

    def test_a_different_logarithm_changes_the_digest(self) -> None:
        self.assertNotEqual(self._perturbed("log"), numeric_environment_digest())

    def test_a_different_exponential_changes_the_digest(self) -> None:
        self.assertNotEqual(self._perturbed("exp"), numeric_environment_digest())

    def test_the_arguments_are_checked(self) -> None:
        sample = reference_sample()
        with self.assertRaises(TypeError):
            numeric_environment_digest(sample.tolist())
        with self.assertRaises(TypeError):
            numeric_environment_digest(sample.astype(np.float32))
        with self.assertRaises(ValueError):
            numeric_environment_digest(sample.reshape(32, 32))
        with self.assertRaises(ValueError):
            numeric_environment_digest(sample[:0])

    def test_non_positive_and_non_finite_samples_are_refused(self) -> None:
        for bad in (0.0, -1.0, np.nan, np.inf):
            sample = reference_sample()
            sample[10] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                numeric_environment_digest(sample)


if __name__ == "__main__":
    unittest.main()
