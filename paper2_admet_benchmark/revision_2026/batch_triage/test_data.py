"""No downloads or estimator training: test role, cache and oracle contracts."""
import copy
import csv
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from . import data


def fixture():
    source = [{"structure_id": f"source{i}", "standardized_smiles": "CC"} for i in range(12)]
    target = [{"sample_id": f"target{i}", "structure_id": f"structure{i}",
               "standardized_smiles": "CCC", "structure_status": "pass"} for i in range(7)]
    target[4]["structure_id"] = "source0"
    target[5]["structure_status"] = "invalid"
    labels = {row["sample_id"]: {"AMES": i % 2} for i, row in enumerate(target)}
    labels["target6"]["AMES"] = float("nan")
    return {"source_rows": source, "source_y": np.tile([0, 1], 6), "target_rows": target,
            "labels": labels, "input_hashes": {"raw": "fixture"},
            "split_metadata": {"outer_seed": 43021, "target_label_counts": {"0": 4, "1": 2}}}


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = fixture()
        self.roles = (np.arange(6), np.arange(6, 9), np.arange(9, 12))
        self.spec = {"phase": "independent_random", "endpoint": "AMES", "split": "random", "fit_seed": 311}
        self.patches = ExitStack()
        self.patches.enter_context(patch.object(data, "_load_raw_case", side_effect=lambda *_: copy.deepcopy(self.raw)))
        self.patches.enter_context(patch.object(data, "_source_roles", return_value=self.roles))
        self.patches.enter_context(patch.object(data, "_training_dependencies", return_value={"source_model": "frozen"}))
        self.patches.enter_context(patch.object(data, "_runtime_versions", return_value={"fixture_runtime": "1"}))
        self.original_builder = data._build_score_arrays
        def build(source_rows, source_y, targets, spec, predictors, jobs):
            self.assertFalse(any("y" in row or "target" in row for row in targets))
            self.assertEqual([row["sample_id"] for row in targets], [f"target{i}" for i in range(4)])
            return self.roles, {name: np.array([.1, .3, .7, .9]) for name in predictors}, np.array([.2, .4, .6, .8])
        self.build = self.patches.enter_context(patch.object(data, "_build_score_arrays", side_effect=build))

    def tearDown(self):
        self.patches.close()
        self.temp.cleanup()

    def prepare(self, folder="scores", **kwargs):
        return data.prepare_case(self.spec, self.root / "raw", self.root / folder, jobs=1, **kwargs)

    def test_fixed_eight_case_catalog(self):
        specs = data.case_specs()
        self.assertEqual(len(specs), 8)
        self.assertEqual({s["fit_seed"] for s in specs}, {311})
        self.assertEqual({s["endpoint"] for s in specs}, set(data.ENDPOINTS))
        with self.assertRaises(ValueError):
            data.case_specs(endpoints=["AMES", "AMES"])

    def test_eligibility_public_view_and_no_target_label_counts(self):
        case = self.prepare()
        public = case.public_view("stack")
        self.assertEqual(len(public["rows"]), 4)
        self.assertEqual(public["metadata"]["n_excluded"], 3)
        self.assertEqual(public["metadata"]["support_reference"], "source_fit_only")
        self.assertNotIn("target_label_counts", public["metadata"]["outer_split_metadata"])
        self.assertEqual(set(public), {"case_id", "predictor", "rows", "score", "max_source_tanimoto", "metadata"})
        self.assertFalse(public["score"].flags.writeable)
        with (case.cache_path / "target_manifest_stack.csv").open() as handle:
            fields = csv.DictReader(handle).fieldnames
        self.assertFalse(set(fields).intersection({"y", "target", "label"}))
        exclusions = json.loads((case.cache_path / "exclusions.json").read_text())
        self.assertEqual({x["reasons"][0] for x in exclusions},
                         {"source_structure_overlap", "invalid_structure", "unobserved_endpoint_outcome"})

    def test_budget_runs_reuse_scores_but_get_fresh_oracles(self):
        first = self.prepare()
        first.new_oracle().query(0)
        second = self.prepare()
        self.build.assert_called_once()
        self.assertEqual(first.cache_path, second.cache_path)
        self.assertEqual(second.new_oracle().query_count, 0)
        np.testing.assert_array_equal(first.public_view()["score"], second.public_view()["score"])

    def test_support_reference_is_source_fit_only_without_real_estimators(self):
        import paper2_admet_benchmark.revision_2026 as package
        target = data._public_rows(self.raw["target_rows"][:4])
        x = np.arange(24).reshape(12, 2)
        q = np.zeros((4, 2))
        calls = []
        def support(reference, query):
            calls.append(reference.copy())
            np.testing.assert_array_equal(reference, x[self.roles[0]])
            np.testing.assert_array_equal(query, q)
            return np.zeros(4)
        fake = SimpleNamespace(
            LEGACY_LOCK=data.ROOT / "paper2_admet_benchmark/configs/racer_c4/prospective_lock_v1.yaml",
            fingerprints=lambda rows: x if len(rows) == 12 else q,
            fit_predictors=lambda *args: (self.roles, {"stack": (np.ones(3) * .5, np.ones(4) * .5)}),
            max_tanimoto=support)
        # The real data builder is called, but all estimator/chemistry operations
        # are stand-ins. This catches accidental use of router/calibration rows.
        with patch.object(package, "runner", fake, create=True):
            self.original_builder(self.raw["source_rows"], self.raw["source_y"], target,
                                  self.spec, ("stack",), 1)
        self.assertEqual(len(calls), 1)

    def test_cache_tamper_is_not_silently_retrained(self):
        case = self.prepare()
        (case.cache_path / "oracle_labels.json").write_text('{"y": [1, 1, 1, 1]}')
        with self.assertRaisesRegex(RuntimeError, "missing or changed"):
            self.prepare()
        self.build.assert_called_once()

    def test_hidden_label_change_does_not_change_score_input_or_public_output(self):
        first = self.prepare()
        inputs_before = self.build.call_args
        for i in range(4):
            self.raw["labels"][f"target{i}"]["AMES"] = 1 - self.raw["labels"][f"target{i}"]["AMES"]
        second = self.prepare()
        np.testing.assert_array_equal(first.public_view()["score"], second.public_view()["score"])
        self.assertEqual(inputs_before.args[0], self.build.call_args.args[0])
        np.testing.assert_array_equal(inputs_before.args[1], self.build.call_args.args[1])
        self.assertEqual(inputs_before.args[2:], self.build.call_args.args[2:])
        self.assertEqual(first.new_oracle().query(0), 1 - second.new_oracle().query(0))

    def test_oracle_without_replacement_and_all_assay_costs(self):
        oracle = data.FrozenLabelOracle(["a", "b", "c", "d"], [0, 1, 1, 0])
        self.assertEqual(oracle.query("a"), 0)
        self.assertEqual(oracle.query(2), 1)
        with self.assertRaisesRegex(ValueError, "second query"):
            oracle.query(0)
        with self.assertRaises(KeyError):
            oracle.query(-1)
        with self.assertRaisesRegex(ValueError, "cannot also"):
            oracle.evaluate([0, 0, 2, 1])
        result = oracle.evaluate([2, 0, 2, 1])
        self.assertEqual(result["n_total_assays"], 2)
        self.assertEqual(result["n_audit_queries"], 2)
        self.assertEqual(result["n_remaining_assays"], 0)
        self.assertEqual(result["false_clearance"], .5)
        self.assertEqual(result["false_discard"], .5)
        self.assertEqual(result["assay_savings"], .5)
        ledger = oracle.ledger
        ledger[0]["y"] = 9
        self.assertEqual(oracle.ledger[0]["y"], 0)

    def test_import_exact_known_source_scores_without_fit(self):
        original = self.prepare()
        imported = self.prepare("imported", score_import=original.manifest_import())
        self.build.assert_called_once()
        self.assertEqual(imported.metadata["score_origin"], "known_source_manifest_import")
        np.testing.assert_array_equal(original.support, imported.support)
        self.assertEqual(original.new_oracle().query(1), imported.new_oracle().query(1))

    def test_import_rejects_unknown_provenance(self):
        original = self.prepare()
        bundle = original.manifest_import()
        bad = data.ManifestImport(bundle.paths, bundle.sha256, "unknown")
        with self.assertRaisesRegex(ValueError, "different data/roles"):
            self.prepare("imported", score_import=bad)

    def test_import_rejects_manifest_labels_even_if_checksum_matches(self):
        original = self.prepare()
        bundle = original.manifest_import()
        path = Path(bundle.paths["stack"])
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["y"] = "1"
        data._csv(path, rows, list(rows[0]))
        hashes = dict(bundle.sha256)
        hashes["stack"] = data._file_hash(path)
        bad = data.ManifestImport(bundle.paths, hashes, bundle.identity_sha256)
        with self.assertRaisesRegex(ValueError, "must not contain target labels"):
            self.prepare("imported", score_import=bad)

    def test_bad_inputs_and_training_role_drift_fail(self):
        with self.assertRaises(ValueError):
            self.prepare(predictors=["stack", "typo"])
        self.raw["labels"]["target0"]["AMES"] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            self.prepare()
        self.raw = fixture()
        self.build.side_effect = None
        self.build.return_value = ((self.roles[1], self.roles[0], self.roles[2]),
            {name: np.ones(4) * .5 for name in data.PREDICTORS}, np.zeros(4))
        with self.assertRaisesRegex(RuntimeError, "role allocation"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
