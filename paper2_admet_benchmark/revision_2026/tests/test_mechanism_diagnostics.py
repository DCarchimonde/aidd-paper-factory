import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from paper2_admet_benchmark.revision_2026 import mechanism_diagnostics as mech


def frame(missing=False):
    rows = []
    for seed in [211, 212]:
        for method, deferred in [(mech.DEFAULT_BASELINE, set()), (mech.DEFAULT_METHOD, {"a"}), (mech.DEFAULT_REFERENCE, {"b"})]:
            for i, (sid, y, p) in enumerate([("a", 1, .8), ("b", np.nan if missing else 0, .7), ("c", 0, .1), ("d", 1, .9)]):
                rows.append(dict(phase="retrospective_epa", endpoint="toy", seed=seed, model="stack", alpha=.1,
                                 method=method, sample_id=sid, structure_id=sid, y=y, eligible=1,
                                 include_0=1 if sid != "d" or sid in deferred else 0,
                                 include_1=int(sid == "d" or sid in deferred), probability_1=p,
                                 max_train_tanimoto=.1 * i))
    return pd.DataFrame(rows)


class MechanismTests(unittest.TestCase):
    def test_discordant_rescue_and_exact_identities(self):
        result = mech.analyze_predictions(frame())
        row = next(c for c in result["cells"] if c["seed"] == 211 and c["baseline_class"] == 0)
        self.assertEqual(row["method_only_true_active"], 1)
        self.assertEqual(row["reference_only_true_active"], 0)
        self.assertEqual(row["false_decision_exposure_delta"], -.5)
        self.assertEqual(row["correct_singleton_yield_delta"], .5)
        self.assertTrue(row["simplified_efficiency_identity_applicable"])
        self.assertEqual(result["endpoints"][0]["n_seed_cells"], 2)
        self.assertTrue(result["report"]["similarity_stratification_available"])
        for column in ("probability_1", "max_train_tanimoto"):
            self.assertTrue(any(s["feature"] == column for s in result["strata"]))

    def test_missing_labels_preserve_budget_but_change_efficiency_identity(self):
        row = next(c for c in mech.analyze_predictions(frame(True))["cells"] if c["seed"] == 211 and c["baseline_class"] == 0)
        self.assertEqual(row["defer_budget_method"], row["defer_budget_reference"])
        self.assertEqual(row["observed_budget_correction_reference_minus_method"], -1)
        self.assertEqual(row["false_decision_exposure_delta"], -.5)
        self.assertEqual(row["correct_singleton_yield_delta"], 0)
        self.assertFalse(row["simplified_efficiency_identity_applicable"])

    def test_row_order_does_not_change_results(self):
        first = mech.analyze_predictions(frame())
        second = mech.analyze_predictions(frame().sample(frac=1, random_state=13))
        self.assertEqual(first, second)

    def test_categorical_structure_ids_match_plain_input(self):
        data = frame()
        expected = mech.analyze_predictions(data)
        data["structure_id"] = data.structure_id.astype("category")
        self.assertEqual(mech.analyze_predictions(data), expected)
        invalid = data[data.sample_id == "d"].copy()
        invalid["sample_id"] = "invalid"
        invalid["structure_id"] = np.nan
        invalid["eligible"] = 0
        invalid["probability_1"] = np.nan
        combined = pd.concat([data, invalid], ignore_index=True)
        combined["structure_id"] = combined.structure_id.astype("category")
        self.assertEqual(mech.analyze_predictions(combined), expected)

    def test_excluded_invalid_structure_keeps_full_manifest_identity(self):
        original = frame()
        invalid = original[original.sample_id == "d"].copy()
        invalid["sample_id"] = "invalid"
        invalid["structure_id"] = np.nan
        invalid["eligible"] = 0
        invalid["probability_1"] = np.nan
        result = mech.analyze_predictions(pd.concat([original, invalid], ignore_index=True))
        self.assertEqual(result, mech.analyze_predictions(original))
        invalid.loc[invalid.method == mech.DEFAULT_METHOD, "sample_id"] = "other_invalid"
        with self.assertRaisesRegex(ValueError, "full target IDs"):
            mech.analyze_predictions(pd.concat([original, invalid], ignore_index=True))

    def test_class_budget_mismatch_rejected_even_if_total_equal(self):
        data = frame()
        mask = (data.method == mech.DEFAULT_REFERENCE) & (data.sample_id == "b")
        data.loc[mask, "include_1"] = 0
        data.loc[(data.method == mech.DEFAULT_REFERENCE) & (data.sample_id == "d"), "include_0"] = 1
        with self.assertRaisesRegex(ValueError, "class-specific"):
            mech.analyze_predictions(data)

    def test_mismatched_identity_probabilities_and_nonnested_sets_rejected(self):
        for column, value in [("y", 0), ("structure_id", "other"), ("probability_1", .4)]:
            data = frame()
            data.loc[(data.method == mech.DEFAULT_METHOD) & (data.sample_id == "a"), column] = value
            with self.assertRaises(ValueError):
                mech.analyze_predictions(data)
        data = frame()
        data.loc[(data.method == mech.DEFAULT_METHOD) & (data.sample_id == "a"), "include_0"] = 0
        with self.assertRaisesRegex(ValueError, "preserve"):
            mech.analyze_predictions(data)

    def test_file_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame(True).to_csv(root / "predictions.csv", index=False)
            data, paths = mech.load_prediction_files(root, [mech.DEFAULT_METHOD, mech.DEFAULT_REFERENCE, mech.DEFAULT_BASELINE])
            result = mech.analyze_predictions(data)
            report = mech.write_diagnostics(result, root / "output", paths)
            self.assertEqual(report["n_prediction_cells"], 2)
            self.assertTrue((root / "output/discordant_members.csv").is_file())
            with self.assertRaises(FileExistsError):
                mech.write_diagnostics(result, root / "output", paths)


if __name__ == "__main__":
    unittest.main()
