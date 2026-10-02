"""Every exception class defined by veridist shares one catchable base."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import unittest

import veridist
import veridist.engine as engine
from veridist.engine.checkpoint import CheckpointCommitUncertain
from veridist.engine.errors import (
    CapabilityCode,
    CapabilityError,
    EngineContractError,
    FailureCode,
    VeridistError,
)


def _defined_exception_classes() -> dict[str, type[BaseException]]:
    """Return every non-private exception class defined inside the package."""

    found: dict[str, type[BaseException]] = {}
    for module_info in pkgutil.walk_packages(veridist.__path__, prefix="veridist."):
        module = importlib.import_module(module_info.name)
        for name, value in vars(module).items():
            if (
                inspect.isclass(value)
                and issubclass(value, BaseException)
                and value.__module__.startswith("veridist")
                and not value.__name__.startswith("_")
            ):
                found[f"{value.__module__}.{value.__qualname__}"] = value
    return found


class ErrorHierarchyTests(unittest.TestCase):
    def test_scan_finds_the_known_public_error_classes(self) -> None:
        names = {qualified.rsplit(".", 1)[1] for qualified in _defined_exception_classes()}
        self.assertTrue(
            {
                "VeridistError",
                "EngineContractError",
                "CapabilityError",
                "StreamSourceError",
                "DeliveryContractError",
                "DataSourceCapabilityError",
                "PassBudgetError",
                "CsvLifetimeAdapterError",
                "CheckpointCommitUncertain",
            }
            <= names
        )

    def test_every_public_error_class_derives_from_veridist_error(self) -> None:
        offenders = [
            qualified
            for qualified, cls in _defined_exception_classes().items()
            if not issubclass(cls, VeridistError)
        ]
        self.assertEqual(offenders, [])

    def test_internal_signal_classes_are_private_and_excluded(self) -> None:
        scanned = {cls.__name__ for cls in _defined_exception_classes().values()}
        for module_name in (
            "veridist.statistics.exponential",
            "veridist.statistics.log_density",
            "veridist.statistics.log_likelihood",
        ):
            module = importlib.import_module(module_name)
            for name, value in vars(module).items():
                if (
                    inspect.isclass(value)
                    and issubclass(value, Exception)
                    and value.__module__ == module_name
                ):
                    self.assertTrue(name.startswith("_"), name)
                    self.assertNotIn(name, scanned)

    def test_engine_and_capability_errors_are_catchable_as_veridist_error(self) -> None:
        with self.assertRaises(VeridistError):
            raise EngineContractError(FailureCode.RANGE_MISMATCH)
        with self.assertRaises(VeridistError):
            raise CapabilityError(CapabilityCode.TRUNCATION_UNSUPPORTED)

    def test_commit_uncertainty_is_a_veridist_error_and_still_a_runtime_error(self) -> None:
        error = CheckpointCommitUncertain("acknowledgement lost")
        self.assertIsInstance(error, VeridistError)
        self.assertIsInstance(error, RuntimeError)
        self.assertNotIsInstance(error, EngineContractError)
        self.assertEqual(str(error), "acknowledgement lost")

    def test_veridist_error_is_exported_from_engine_but_not_top_level(self) -> None:
        self.assertIn("VeridistError", engine.__all__)
        self.assertIs(engine.VeridistError, VeridistError)
        self.assertNotIn("VeridistError", veridist.__all__)
        self.assertFalse(hasattr(veridist, "VeridistError"))
