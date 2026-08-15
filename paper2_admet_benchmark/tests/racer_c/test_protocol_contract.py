from __future__ import annotations

import csv
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"


class ProtocolContractTests(unittest.TestCase):
    def test_protocol_records_v1_freeze_approval(self) -> None:
        text = (P2 / "protocols/paper2_reliability_extension_protocol_2026.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("1.0 (freeze candidate approved; frozen by tag)", text)
        self.assertIn("user approves the formal protocol tag", text)
        self.assertIn("paper2-racer-protocol-freeze-v1.0", text)
        self.assertIn("2026-08-07", text)

    def test_endpoint_manifest_has_required_columns_and_unique_endpoints(self) -> None:
        path = P2 / "protocols/endpoint_candidate_manifest.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {
            "endpoint",
            "task_type",
            "data_source",
            "raw_sha256",
            "class_0_n",
            "class_1_n",
            "critical_class",
            "label_definition",
            "license",
            "eligibility_status",
            "eligibility_reason",
        }
        self.assertTrue(rows)
        self.assertTrue(required.issubset(rows[0]))
        names = [row["endpoint"] for row in rows]
        self.assertEqual(len(names), len(set(names)))

    def test_method_access_never_allows_test_labels(self) -> None:
        path = P2 / "protocols/method_access_manifest.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertTrue(all(row["test_labels"] == "no" for row in rows))
        self.assertTrue(any(row["access_class"] == "transductive" for row in rows))
        self.assertTrue(any(row["primary_rankable"] == "yes" for row in rows))

    def test_phase1_contracts_are_present_and_pre_freeze(self) -> None:
        provenance = P2 / "protocols/data_provenance_license_manifest.csv"
        standardization = P2 / "protocols/chemical_standardization_contract.yaml"
        runbook = P2 / "protocols/data_acquisition_and_cleaning_runbook.md"
        self.assertTrue(provenance.is_file())
        self.assertIn("status: draft_pre_freeze", standardization.read_text(encoding="utf-8"))
        self.assertIn("creates no model predictions", runbook.read_text(encoding="utf-8"))
        with provenance.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 29)
        self.assertTrue(any(row["analysis_use_status"] == "pending_original_terms" for row in rows))

    def test_frozen_headline_tables_match_integrity_manifest(self) -> None:
        asset_root = P2 / "results/manuscript_assets"
        manifest = asset_root / "final_results_integrity_manifest.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 11)
        for row in rows:
            path = P2 / row["file"]
            raw = path.read_bytes()
            raw_digest = hashlib.sha256(raw).hexdigest()
            # The frozen manifest was generated on Windows and records CRLF bytes,
            # while Git's clean checkout contains LF text.  Verify the historical
            # hash without rewriting either the frozen tables or their manifest.
            lf = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            crlf_digest = hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()
            self.assertIn(row["sha256"], {raw_digest, crlf_digest}, path.as_posix())
            with path.open(newline="", encoding="utf-8") as handle:
                observed_rows = sum(1 for _ in csv.reader(handle)) - 1
            self.assertEqual(observed_rows, int(row["rows"]), path.as_posix())
            self.assertEqual(row["row_count_valid"], "True")

    def test_frozen_figure_files_match_integrity_manifest(self) -> None:
        asset_root = P2 / "results/manuscript_assets"
        manifest = asset_root / "figures/main_figure_integrity_manifest.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        for row in rows:
            path = P2 / row["file"]
            self.assertEqual(path.stat().st_size, int(row["bytes"]), path.as_posix())
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                row["sha256"],
                path.as_posix(),
            )


if __name__ == "__main__":
    unittest.main()
