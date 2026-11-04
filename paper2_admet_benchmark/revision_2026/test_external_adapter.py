"""Input-integrity tests for external comparison joins (synthetic fixtures only)."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from paper2_admet_benchmark.revision_2026.external_adapter import import_predictions, sha256


class ExternalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "manifest.csv"
        self.predictions = self.root / "external.csv"
        self.metadata = self.root / "meta.json"
        self.manifest.write_text("sample_id,structure_id,phase,endpoint,seed,model\na,A,development,NR-AR,17,logistic\nb,B,development,NR-AR,17,logistic\n")
        self.predictions.write_text("sample_id,alpha,include_0,include_1,selected\nb,0.1,0,0,0\na,0.1,1,0,1\n")
        self.meta = {"method": "external_test", "evaluation_model": "external_model",
                     "source_url": "https://example.org/paper", "source_revision": "synthetic-fixture-only",
                     "adapter_version": "1", "implementation_kind": "component_only",
                     "protocol_sha256": "a" * 64, "target_manifest_sha256": sha256(self.manifest),
                     "target_labels_used": False, "target_features_used": True,
                     "fitted_label_roles": ["fit", "conformal"], "dependencies": {"python": "test"}}
        self.metadata.write_text(json.dumps(self.meta))

    def run_import(self):
        return import_predictions(self.manifest, self.predictions, self.metadata, "a" * 64, [0.1], self.root / "out")

    def test_aligns_by_id_and_preserves_unselected_original(self):
        result = self.run_import()
        self.assertEqual(result["rows"], 2)
        with (self.root / "out" / "predictions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["sample_id"] for r in rows], ["a", "b"])
        self.assertEqual((rows[1]["include_0"], rows[1]["include_1"]), ("1", "1"))
        self.assertEqual((rows[1]["external_original_include_0"], rows[1]["external_original_include_1"]), ("0", "0"))
        self.assertEqual(rows[1]["external_selected"], "0")
        self.assertEqual(rows[0]["model"], "external_model")
        self.assertEqual(rows[0]["reference_model"], "logistic")

    def test_rejects_duplicate_missing_and_unexpected_ids(self):
        header = "sample_id,alpha,include_0,include_1\n"
        for body in ("a,0.1,1,0\na,0.1,0,1\n", "a,0.1,1,0\n", "a,0.1,1,0\nc,0.1,0,1\n"):
            with self.subTest(body=body):
                self.predictions.write_text(header + body)
                with self.assertRaises(ValueError):
                    self.run_import()

    def test_rejects_target_labels_even_when_blank(self):
        self.predictions.write_text("sample_id,alpha,include_0,include_1,y\na,0.1,1,0,\nb,0.1,0,1,\n")
        with self.assertRaisesRegex(ValueError, "Unexpected columns"):
            self.run_import()

    def test_rejects_bad_flags_and_alpha(self):
        for row in ("a,0.1,2,0", "a,nan,1,0", "a,0.2,1,0"):
            with self.subTest(row=row):
                self.predictions.write_text("sample_id,alpha,include_0,include_1\n" + row + "\nb,0.1,0,1\n")
                with self.assertRaises(ValueError):
                    self.run_import()

    def test_rejects_provenance_mismatch_or_label_use(self):
        for key, value in (("protocol_sha256", "b" * 64), ("target_manifest_sha256", "b" * 64),
                           ("target_labels_used", True), ("fitted_label_roles", ["test"]),
                           ("evaluation_model", "logistic")):
            with self.subTest(key=key):
                self.metadata.write_text(json.dumps({**self.meta, key: value}))
                with self.assertRaises(ValueError):
                    self.run_import()

    def test_does_not_overwrite_previous_import(self):
        self.run_import()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_import()


if __name__ == "__main__":
    unittest.main()
