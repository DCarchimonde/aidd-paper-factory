import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("legacy_audit", Path(__file__).with_name("audit_legacy_quantiles.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class LegacyAuditTests(unittest.TestCase):
    def test_tiny_class_has_infinite_corrected_threshold(self):
        old, new = audit.thresholds(np.array([0.1, 0.2]), 0.1)
        self.assertEqual(old, 0.2)
        self.assertTrue(math.isinf(new))
        self.assertEqual(audit.thresholds(np.arange(9) / 10, 0.1), (0.8, 0.8))

    def test_membership_change_differs_from_true_coverage_change(self):
        rows = []
        for i, (role, y, p) in enumerate([("calibration", 0, .1), ("calibration", 1, .9), ("test", 0, .95), ("test", 1, .95)]):
            rows.append(dict(endpoint="toy", split_col="split_confirm_random_seed101", model="m", task_type="classification", role=role, row_index=i, y_true=y, y_output=p))
        result = audit.audit_predictions(pd.DataFrame(rows), [.1], "confirmatory")
        zero = next(r for r in result if r["calibration_class"] == 0)
        self.assertEqual(zero["candidate_memberships_changed"], 2)
        self.assertEqual(zero["true_labels_newly_covered"], 1)
        self.assertEqual(zero["coverage_delta"], 1)
        duplicate = pd.DataFrame(rows + [rows[0]])
        with self.assertRaises(ValueError):
            audit.audit_predictions(duplicate, [.1], "confirmatory")

    def test_counts_only_cannot_claim_actual_change(self):
        frame = pd.DataFrame([dict(endpoint="toy", split_col="split_confirm_random_seed101", model="m", alpha=.1, calibration_negative_count=100, calibration_positive_count=2)])
        rows = audit.audit_summary(frame, "confirmatory")
        self.assertTrue(rows[1]["rank_exceeds_n"])
        self.assertIsNone(rows[1]["candidate_memberships_changed"])

    def test_missing_files_are_not_zero_impact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = audit.run(root / "missing", None, root / "out", [.1], "confirmatory", "*.csv")
            self.assertEqual(report["status"], "missing_inputs_cannot_determine_historical_impact")
            self.assertFalse(report["historical_published_numbers_modified"])
            with self.assertRaises(FileExistsError):
                audit.run(root / "missing", None, root / "out", [.1], "confirmatory", "*.csv")


if __name__ == "__main__":
    unittest.main()
