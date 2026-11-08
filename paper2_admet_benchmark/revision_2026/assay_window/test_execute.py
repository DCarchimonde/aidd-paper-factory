"""Integration and information-boundary tests for the paired-response runner."""
from __future__ import annotations

from collections import defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from . import execute
from .targets import DEFAULT_EDGES, DEFAULT_MARGIN_LOG10


class ExecutionTests(unittest.TestCase):
    def test_split_keeps_all_repeats_and_protocols_together(self):
        rows = execute.synthetic_records()
        other = [{**r, "protocol": "second_protocol"} for r in rows]
        rows += other
        for split in execute.SPLITS:
            train, test = execute.split_indices(rows, split)
            train_groups = {rows[int(i)]["group_id"] for i in train}
            test_groups = {rows[int(i)]["group_id"] for i in test}
            self.assertFalse(train_groups & test_groups)
            self.assertEqual(len(train) + len(test), len(rows))
            for group in test_groups:
                self.assertEqual(sum(rows[int(i)]["group_id"] == group for i in test), 6)
            if split == "scaffold":
                self.assertFalse({rows[int(i)]["scaffold_id"] for i in train} &
                                 {rows[int(i)]["scaffold_id"] for i in test})
        changed = copy.deepcopy(rows)
        for r in changed:
            r["target_responses_pct"] = [-100] * 12
            r["official_outcomes"] = {"invented": "active"}
        for split in execute.SPLITS:
            original_split = execute.split_indices(rows, split)
            changed_split = execute.split_indices(changed, split)
            for original, changed_part in zip(original_split, changed_split):
                np.testing.assert_array_equal(original, changed_part)

    def test_features_ignore_outcomes_and_test_protocol_vocab(self):
        rows = execute.synthetic_records()
        train, test = execute.split_indices(rows, "random")
        first, info = execute.build_features(rows, train, synthetic=True)
        changed = copy.deepcopy(rows)
        for i in test:
            changed[int(i)]["target_responses_pct"] = [-999] * 12
            changed[int(i)]["viability_responses_pct"] = [999] * 12
            changed[int(i)]["official_outcomes"] = {"AC50": 1e-12}
        second, _ = execute.build_features(changed, train, synthetic=True)
        np.testing.assert_array_equal(first, second)
        for i in test:
            changed[int(i)]["protocol"] = "unseen_protocol"
        third, info = execute.build_features(changed, train, synthetic=True)
        self.assertNotIn("unseen_protocol", info["protocol_vocabulary"])
        self.assertTrue(np.all(third[test, -1] == 0))

    def test_weights_and_reference_disagreement_respect_protocol(self):
        rows, _ = execute.derive_records(execute.synthetic_records())
        totals = defaultdict(float)
        for row, weight in zip(rows, execute.molecule_weights(rows)):
            totals[row["group_id"]] += weight
        self.assertTrue(all(abs(v-1) < 1e-12 for v in totals.values()))
        summary = execute.describe_records(rows)
        self.assertEqual(summary["group_protocol_units_with_repeats"], 48)
        self.assertTrue(all(r["complete_three_repeats"] for r in summary["sample_protocol_repeat_profiles"]))

    def test_point_imputation_keeps_unobserved_uninformative(self):
        mask = np.zeros((2, 4, 4), dtype=bool)
        mask[0, 2:, 1:] = True
        mask[1] = True
        imputed = execute.point_impute_masks(mask)
        self.assertEqual(imputed[0].sum(), 1)
        self.assertTrue(imputed[0, 3, 2])
        self.assertTrue(imputed[1].all())

    def test_uncertain_reference_is_retained_in_loss_bounds(self):
        rows = [{"group_id": "a", "window_reference": {"status": "definite_no"}},
                {"group_id": "b", "window_reference": {"status": "unresolved"}}]
        report = execute.probability_metrics(rows, [.2, .8], [.2, .8])
        self.assertEqual(report["resolved_rows"], 1)
        self.assertAlmostEqual(report["brier"], .04)
        self.assertAlmostEqual(report["all_reference_brier_lower"], .04)
        self.assertAlmostEqual(report["all_reference_brier_upper"], .34)

    def test_resolved_metrics_reweight_mixed_unresolved_repeats(self):
        rows = [{"group_id": "a", "window_reference": {"status": status}}
                for status in ("definite_yes", "unresolved", "unresolved")]
        rows.append({"group_id": "b", "window_reference": {"status": "definite_no"}})
        q = np.asarray([.1, .8, .6, .2])
        report = execute.probability_metrics(rows, q, q)
        self.assertAlmostEqual(report["brier"], (.81 + .04) / 2)
        self.assertAlmostEqual(report["resolved_fraction_molecule_weighted"], 2 / 3)
        np.testing.assert_allclose(execute.resolved_molecule_weights(rows), [1, 0, 0, 1])
        np.testing.assert_allclose(execute.status_molecule_weights(rows), [1, .5, .5, 1])
        # All-reference bounds retain each molecule's full-record total weight of one.
        self.assertAlmostEqual(report["all_reference_brier_lower"], ((.81 + .04 + .16) / 3 + .04) / 2)

    def test_all_uninformative_training_is_not_evaluable(self):
        rows, masks = execute.derive_records(execute.synthetic_records())
        train, test = execute.split_indices(rows, "random")
        X, _ = execute.build_features(rows, train, synthetic=True)
        masks[train] = True
        metrics, predictions = execute.fit_evaluate("joint_censored", X, rows, masks, train, test,
                            edges=DEFAULT_EDGES, margin=DEFAULT_MARGIN_LOG10, max_iter=10)
        self.assertEqual(metrics["convergence"]["status"], "not_evaluable")
        self.assertIsNone(metrics["masked_nll_molecule_weighted"])
        self.assertTrue(all(p["q_midpoint"] is None for p in predictions))

    def test_cli_pipeline_and_resume_without_retraining(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "synthetic_pipeline"
            arguments = ["--mode", "pilot", "--output", str(output), "--max-iter", "30", "--jobs", "1"]
            execute.main(arguments)
            receipt = json.loads((output / "experiment_complete.json").read_text())
            self.assertEqual(receipt["completed_jobs"], 8)
            self.assertEqual(receipt["kind"], "synthetic_software_check")
            paired = json.loads(gzip.decompress((output / "paired_records.json.gz").read_bytes()))
            self.assertEqual(len(paired), 144)
            runs = sorted((output / "runs").glob("*.json"))
            self.assertEqual(len(runs), 8)
            for path in runs:
                run = json.loads(path.read_text())
                self.assertEqual(set(run["metrics"]["training_reference_counts"]),
                                 {"definite_yes", "definite_no", "unresolved"})
                if run["job"]["method"] != "binary_window":
                    row_loss = np.asarray([p["masked_nll"] for p in run["predictions"]])
                    row_weight = np.asarray([p["test_molecule_weight"] for p in run["predictions"]])
                    self.assertAlmostEqual(float(np.average(row_loss, weights=row_weight)),
                                           run["metrics"]["masked_nll_molecule_weighted"])
                else:
                    self.assertIsNone(run["metrics"]["all_repeats"]["mean_probability_resolution_width"])
                resolved_predictions = [p for p in run["predictions"] if p["window_reference"] != "unresolved"]
                reconstructed_brier = np.average(
                    [(p["q_midpoint"] - int(p["window_reference"] == "definite_yes")) ** 2
                     for p in resolved_predictions],
                    weights=[p["resolved_molecule_weight"] for p in resolved_predictions])
                self.assertAlmostEqual(reconstructed_brier, run["metrics"]["all_repeats"]["brier"])
            before = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in runs}
            for split in execute.SPLITS:
                manifest = json.loads((output / f"split_{split}.json").read_text())
                self.assertFalse(set(manifest["train_group_ids"]) & set(manifest["test_group_ids"]))
            with patch.object(execute, "build_features", side_effect=AssertionError("resume refit features")), \
                 patch.object(execute, "fit_evaluate", side_effect=AssertionError("resume refit model")), \
                 patch.object(execute, "synthetic_records", side_effect=AssertionError("resume regenerated data")):
                execute.main(arguments + ["--resume"])
            after = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in runs}
            self.assertEqual(before, after)
            with zipfile.ZipFile(output.with_name(output.name + "_review.zip")) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn("test_predictions.csv", archive.namelist())
            # A changed cached input must not silently combine with old fits.
            paired[0]["purity"] = "changed"
            execute.write_gzip_json(output / "paired_records.json.gz", paired)
            with self.assertRaisesRegex(ValueError, "prepared records"):
                execute.main(arguments + ["--resume"])

    def test_prepare_real_interface_has_no_fit(self):
        from . import data
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "mock_prepare_not_molecular"
            with patch.object(data, "prepare_protocol", return_value=(execute.synthetic_records(), {"fixture": True})), \
                 patch.object(execute, "fit_evaluate", side_effect=AssertionError("prepare trained model")):
                execute.main(["--mode", "prepare", "--output", str(output)])
            receipt = json.loads((output / "experiment_complete.json").read_text())
            self.assertEqual(receipt["model_fits"], 0)
            self.assertEqual(receipt["paired_rows"], 144)
            self.assertTrue((output / "READOUT_CN.md").exists())


if __name__ == "__main__":
    unittest.main()
