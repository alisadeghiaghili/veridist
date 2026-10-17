"""The provenance of inference from a scale fit: pass roles, ranges and the canonical record."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from veridist.engine.provenance import (
    ScalePassProvenance,
    ScalePassRecord,
    scale_provenance_to_canonical_json_bytes,
)
from veridist.scale import _gof as gof
from veridist.scale import _profile_intervals as pi
from veridist.scale._executors import ThreadExecutor
from veridist.scale._fit import fit
from veridist.scale._planner import PassRole
from veridist.scale._sources import NumpySource

RANGED = ("grid", "expansion", "profile_window", "profile_refinement", "profile_confirmation")
UNRANGED = ("iteration", "gof")


class RoleTests(unittest.TestCase):
    def test_the_roles_of_inference_are_pass_roles_of_the_planner(self) -> None:
        self.assertEqual(PassRole.PROFILE_WINDOW.value, "profile_window")
        self.assertEqual(PassRole.PROFILE_REFINEMENT.value, "profile_refinement")
        self.assertEqual(PassRole.PROFILE_CONFIRMATION.value, "profile_confirmation")
        self.assertEqual(PassRole.GOF.value, "gof")

    def test_a_pass_over_a_range_of_candidates_has_a_range(self) -> None:
        for role in RANGED:
            with self.subTest(role=role):
                record = ScalePassRecord(role, 9, -1.5, 2.5)
                self.assertEqual((record.log_low, record.log_high), (-1.5, 2.5))
                with self.assertRaisesRegex(ValueError, "logarithmic range"):
                    ScalePassRecord(role, 9, None, None)

    def test_the_other_passes_have_none(self) -> None:
        for role in UNRANGED:
            with self.subTest(role=role):
                self.assertEqual(ScalePassRecord(role, 1, None, None).role, role)
                with self.assertRaisesRegex(ValueError, "logarithmic range"):
                    ScalePassRecord(role, 1, -1.0, 1.0)

    def test_an_inference_pass_has_candidates(self) -> None:
        for role in (*RANGED[2:], "gof"):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "no candidate"):
                ScalePassRecord(
                    role, 0, None if role == "gof" else -1.0, None if role == "gof" else 1.0
                )

    def test_an_unknown_role_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "pass role"):
            ScalePassRecord("histogram", 1, None, None)


class RecordTests(unittest.TestCase):
    def setUp(self) -> None:
        values = 40.0 * np.random.default_rng(3).weibull(1.7, 600)
        self.source = NumpySource(values, partition_rows=200)
        self.result = fit(self.source, "weibull_min")

    def test_the_profile_record_is_valid_deterministic_and_has_no_data(self) -> None:
        out = pi.profile_intervals(self.source, self.result, executor=ThreadExecutor(2))
        again = pi.profile_intervals(self.source, self.result, executor=ThreadExecutor(2))
        encoded = scale_provenance_to_canonical_json_bytes(out.scale)
        self.assertEqual(encoded, scale_provenance_to_canonical_json_bytes(again.scale))
        document = json.loads(encoded)
        roles = [item["role"] for item in document["passes"]["records"]]
        self.assertEqual(roles[0], "profile_window")
        self.assertEqual(roles[-1], "profile_confirmation")
        self.assertEqual(document["passes"]["pass_budget"], 6)
        self.assertEqual(document["passes"]["minimum_passes"], 2)
        self.assertEqual(document["passes"]["maximum_passes"], 9)
        self.assertEqual(document["row_count"], 600)
        self.assertEqual(document["state"]["digest_sha256"], self.result.scale.state.digest_sha256)  # type: ignore[union-attr]
        self.assertEqual(len(document["execution"]["passes"]), len(roles))
        self.assertEqual(document["partition_count"], 3)
        self.assertNotIn("checkpoint", document)
        self.assertNotIn("spool", document)

    def test_the_gof_record_has_one_pass(self) -> None:
        out = gof.large_n_gof(self.source, self.result)
        document = json.loads(scale_provenance_to_canonical_json_bytes(out.scale))
        self.assertEqual(
            document["passes"]["records"],
            [{"log_high": None, "log_low": None, "points": 1, "role": "gof"}],
        )
        self.assertEqual(document["passes"]["pass_count"], 1)

    def test_a_checkpointed_run_records_every_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = pi.profile_intervals(
                self.source, self.result, checkpoint=Path(directory) / "job.db"
            )
        assert out.scale.passes is not None and out.scale.checkpoint is not None
        self.assertEqual(len(out.scale.checkpoint.passes), out.scale.passes.pass_count)
        document = json.loads(scale_provenance_to_canonical_json_bytes(out.scale))
        self.assertEqual(len(document["checkpoint"]["passes"]), out.scale.passes.pass_count)

    def test_the_pass_record_of_a_fit_is_unchanged(self) -> None:
        passes = self.result.scale.passes
        assert passes is not None
        self.assertEqual([item.role for item in passes.records][:2], ["statistics", "grid"])
        self.assertIsInstance(passes, ScalePassProvenance)
        self.assertEqual(replace(passes).records, passes.records)


if __name__ == "__main__":
    unittest.main()
