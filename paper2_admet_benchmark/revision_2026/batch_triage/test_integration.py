"""Mock-data CLI integration: real oracle, strategies, certificates and resume."""
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import data, execute


class MockScreenIntegrationTests(unittest.TestCase):
    def test_screen_real_data_api_and_resume_without_preparation_or_queries(self):
        spec = {"phase": "independent_random", "endpoint": "AMES", "split": "random", "fit_seed": 311}
        case_id = "independent_random__AMES__fit311"
        rows = tuple({"sample_id": f"fixture_{i:03d}", "structure_id": f"fixture_structure_{i:03d}",
                      "standardized_smiles": "C"} for i in range(32))
        scores = np.linspace(.01, .99, 32)
        support = np.tile([.2, .8], 16)
        labels = tuple(int(x > .5) for x in scores)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "mock_screen_not_molecular"
            case = data.PreparedCase(case_id, rows, {"stack": scores}, support,
                {"kind": "synthetic_fixture_with_real_data_oracle", "n_pool": 32}, root, labels)
            args = ["--mode", "screen", "--jobs", "1", "--cache", str(root / "cache"), "--output", str(output)]
            first_log, resume_log = io.StringIO(), io.StringIO()
            with patch.object(data, "case_specs", return_value=[spec]), \
                 patch.object(data, "prepare_case", return_value=case) as prepare, \
                 contextlib.redirect_stdout(first_log):
                execute.main(args)
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.args[0]["case_id"], case_id)
            manifest = json.loads((output / "required_jobs.json").read_text())
            self.assertEqual(len(manifest["required_jobs"]), 24)
            self.assertEqual({j["method"] for j in manifest["required_jobs"]}, set(execute.METHODS))
            self.assertEqual({j["audit_seed"] for j in manifest["required_jobs"]}, {81001, 81002, 81003})
            files = sorted((output / "runs").glob("*.json"))
            self.assertEqual(len(files), 24)
            before = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in files}
            total_queries = 0
            for path in files:
                saved = json.loads(path.read_text())
                points = saved["checkpoints"]
                self.assertEqual([p["audit_budget"] for p in points], [0, 1, 3, 6, 12])
                total_queries += points[-1]["queried"]
                for point in points:
                    metric = point["metrics"]
                    self.assertEqual(metric["class_0_n"], 16)
                    self.assertEqual(metric["class_1_n"], 16)
                    self.assertEqual(metric["fn_rate"], metric["false_negative_count"] / 16)
                    self.assertEqual(metric["fp_rate"], metric["false_positive_count"] / 16)
                    self.assertEqual(metric["n_total_assays"], point["total_assays_required"])
                    self.assertEqual(metric["n_audit_queries"], point["queried"])
                    self.assertEqual(point["saved_items"] + metric["n_total_assays"], 32)
                    self.assertEqual(metric["n_remaining_assays"] + point["queried"], metric["n_total_assays"])
                    self.assertEqual(len(point["query_history"]), point["queried"])
                    self.assertTrue(all(point["actions"][int(row["sample_id"].split("_")[-1])] == 2
                                        for row in point["query_history"]))
            self.assertGreater(total_queries, 0)
            with (output / "all_checkpoint_metrics.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 120)
            with patch.object(data, "case_specs", return_value=[spec]), \
                 patch.object(data, "prepare_case", side_effect=AssertionError("completed resume prepared data")), \
                 patch.object(data.FrozenLabelOracle, "query", side_effect=AssertionError("completed resume queried labels")), \
                 contextlib.redirect_stdout(resume_log):
                execute.main(args + ["--resume"])
            after = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in files}
            self.assertEqual(before, after)
            self.assertNotIn("completed " + case_id, resume_log.getvalue())
            completion = json.loads((output / "experiment_complete.json").read_text())
            self.assertEqual(completion["jobs"], 24)
            self.assertEqual(completion["checkpoint_rows"], 120)
            with zipfile.ZipFile(output.with_name(output.name + "_review.zip")) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn("experiment_complete.json", archive.namelist())
            print("MOCK-DATA CLI CHECK ONLY — no download or predictor training")
            print(first_log.getvalue(), end="")
            print("RESUME CHECK: no data preparation, no queries, 24 run receipts unchanged")
            print(resume_log.getvalue(), end="")


if __name__ == "__main__":
    unittest.main()
