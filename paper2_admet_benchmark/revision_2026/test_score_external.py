"""Synthetic tests for post-prediction label joining and selection denominators."""
import csv
import json
import unittest

from paper2_admet_benchmark.revision_2026.score_external import score_external
from paper2_admet_benchmark.revision_2026 import test_external_adapter as adapter_fixtures


class ExternalScoringTests(adapter_fixtures.ExternalAdapterTests):
    def setUp(self):
        super().setUp()
        self.run_import()
        self.reference = self.root / "reference.csv"
        self.reference.write_text("phase,endpoint,seed,model,method,alpha,sample_id,structure_id,y,eligible\n"
                                  "development,NR-AR,17,logistic,mondrian,0.1,a,A,0,1\n"
                                  "development,NR-AR,17,logistic,mondrian,0.1,b,B,1,1\n")

    def evaluate(self):
        return score_external(self.root / "out", self.reference, self.root / "scored")

    def test_full_and_selected_denominators(self):
        self.evaluate()
        with (self.root / "scored" / "metrics_full.csv").open() as h:
            full = next(csv.DictReader(h))
        with (self.root / "scored" / "metrics_selected.csv").open() as h:
            selected = next(csv.DictReader(h))
        self.assertEqual(full["n"], "2")
        self.assertEqual(full["n_deferred"], "1")
        self.assertEqual(full["overall_coverage"], "1.0")
        self.assertEqual(full["class_1_retention"], "0.0")
        self.assertEqual(selected["n"], "1")
        self.assertEqual(selected["class_1_n"], "0")
        self.assertEqual(selected["minimum_class_coverage"], "nan")

    def test_hash_checked_before_reference_is_read(self):
        path = self.root / "out" / "predictions.csv"
        path.write_text(path.read_text() + "\n")
        self.reference.unlink()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.evaluate()

    def test_rejects_structure_or_phase_mismatch(self):
        old = self.reference.read_text()
        for changed in (old.replace(",b,B,", ",b,WRONG,"), old.replace("development", "retrospective_epa")):
            with self.subTest(changed=changed):
                self.reference.write_text(changed)
                with self.assertRaises(ValueError):
                    self.evaluate()

    def test_rejects_conflicting_labels_across_methods(self):
        with self.reference.open("a") as handle:
            handle.write("development,NR-AR,17,logistic,other,0.1,a,A,1,1\n")
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.evaluate()

    def test_zero_selected_cohort_is_reported(self):
        pred = self.root / "out" / "predictions.csv"
        with pred.open() as h:
            rows = list(csv.DictReader(h))
        for row in rows:
            row.update({"external_selected": "0", "include_0": "1", "include_1": "1"})
        with pred.open("w", newline="") as h:
            writer = csv.DictWriter(h, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        # This fixture emulates a valid zero-selection import with an updated import receipt.
        from paper2_admet_benchmark.revision_2026.external_adapter import sha256
        receipt_path = self.root / "out" / "import_report.json"
        receipt = json.loads(receipt_path.read_text()); receipt["output_sha256"] = sha256(pred)
        receipt_path.write_text(json.dumps(receipt))
        self.evaluate()
        with (self.root / "scored" / "metrics_selected.csv").open() as h:
            selected = next(csv.DictReader(h))
        self.assertEqual(selected["n"], "0")
        self.assertEqual(selected["overall_coverage"], "nan")

    # Inherited importer tests belong to the importer suite, not this fixture.
    test_aligns_by_id_and_preserves_unselected_original = None
    test_rejects_duplicate_missing_and_unexpected_ids = None
    test_rejects_target_labels_even_when_blank = None
    test_rejects_bad_flags_and_alpha = None
    test_rejects_provenance_mismatch_or_label_use = None
    test_does_not_overwrite_previous_import = None


if __name__ == "__main__":
    unittest.main()
