"""Parser checks include two real matched NCATS curve rows (Nitrofurazone)."""
import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from . import data

PROTOCOL = "tox21-ar-mda-kb2-luc-antagonist-p1"
DOSES = [1.18e-9, 2.638e-9, 5.8989999999e-9, 1.319e-8, 2.9489999999e-8,
         6.5949999999e-8, 1.475e-7, 3.297e-7, 7.3729999999e-7, 1.649e-6,
         3.6869999999e-6, 8.244e-6, 1.8429999999e-5, 4.1219999999e-5, 9.217e-5]
TARGET = [-9.51200358, 13.38308097, 2.78173374, -1.35206729, 7.29560965,
          -.64921595, 7.99301431, -2.63702871, -12.60750318, -6.80200633,
          -.18404792, -6.53223936, -8.44575007, -30.16887527, -71.48245184]
VIABILITY = [-.34270022, 3.95403144, 4.43516775, -.06008404, .15810338,
             .62539555, 3.56904129, 3.21261397, -2.10401915, -1.01020442,
             -1.3003137, -3.70776842, -2.85731726, -15.13098512, -47.00840546]


def actual_pair():
    common = {key: "" for key in data.REQUIRED}
    common.update(SAMPLE_ID="NCGC00090686-07", PROTOCOL_NAME=PROTOCOL,
                  SMILES="O=[N+]([O-])c1ccc(/C=N/NC(N)=O)o1", PURITY_RATING="A",
                  PURITY_RATING_4M="A", TOX21_ID="Tox21_400035",
                  ASSAY_OUTCOME="inconclusive antagonist (cytotoxic)",
                  CHANNEL_OUTCOME="active antagonist")
    rows = []
    for kind, curve_id, ys in (("antagonist3", "153069292", TARGET),
                                ("viability3", "153083496", VIABILITY)):
        row = dict(common, SAMPLE_DATA_TYPE=kind, SAMPLE_DATA_ID=curve_id)
        for i, (c, y) in enumerate(zip(DOSES, ys)):
            row[f"CONC{i}"] = str(c)
            row[f"DATA{i}"] = str(y)
        rows.append(row)
    return rows


def fake_structure(smiles):
    group = "parent:" + smiles.replace(".[Cl-]", "")
    return {"smiles": smiles.replace(".[Cl-]", ""), "compound_id": group,
            "group_id": group, "structure_id": hashlib.sha256(smiles.encode()).hexdigest(),
            "scaffold_id": "fixture-scaffold"}


def archive_bytes(rows, protocol=PROTOCOL, overflow=""):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(fields)
    for row in rows:
        writer.writerow([row.get(key, "") for key in fields] + [overflow])
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{protocol}.txt", output.getvalue())
    return archive.getvalue()


class PairedDataTests(unittest.TestCase):
    def parse(self, rows, protocol=PROTOCOL, purity=("A", "B"), overflow=""):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.zip"
            path.write_bytes(archive_bytes(rows, protocol, overflow))
            return data.parse_protocol_zip(path, protocol, purity, fake_structure)

    def test_real_paired_fixture_preserves_units_and_inconclusive_labels(self):
        records, diagnostics = self.parse(actual_pair())
        self.assertEqual(len(records), 1)
        row = records[0]
        self.assertEqual(row["target_curve_id"], "153069292")
        self.assertEqual(row["viability_curve_id"], "153083496")
        self.assertEqual(row["replicate"], 3)
        self.assertEqual(row["concentrations_m"], DOSES)
        self.assertEqual(row["target_responses_pct"], TARGET)
        self.assertEqual(row["viability_responses_pct"], VIABILITY)
        self.assertEqual(row["official_outcomes"]["target"]["ASSAY_OUTCOME"], "inconclusive antagonist (cytotoxic)")
        self.assertFalse(diagnostics["outcome_labels_used_for_selection"])
        self.assertEqual(diagnostics["raw_rows"], 2)

    def test_duplicate_channel_is_excluded_not_arbitrarily_selected(self):
        rows = actual_pair()
        records, diag = self.parse(rows + [dict(rows[0])])
        self.assertEqual(records, [])
        self.assertEqual(diag["pair_exclusion_reasons"]["duplicate_target_channel"], 1)

    def test_unpaired_replicate_is_reported(self):
        rows = actual_pair()
        rows[1]["SAMPLE_DATA_TYPE"] = "viability2"
        records, diag = self.parse(rows)
        self.assertEqual(records, [])
        self.assertEqual(diag["pair_exclusion_reasons"]["missing_target_channel"], 1)
        self.assertEqual(diag["pair_exclusion_reasons"]["missing_viability_channel"], 1)

    def test_concentration_mismatch_is_excluded(self):
        rows = actual_pair()
        rows[1]["CONC0"] = "1e-10"
        records, diag = self.parse(rows)
        self.assertEqual(records, [])
        self.assertEqual(diag["pair_exclusion_reasons"]["paired_concentration_mismatch"], 1)

    def test_missing_response_stays_missing(self):
        rows = actual_pair()
        rows[0]["DATA7"] = ""
        records, diag = self.parse(rows)
        self.assertIsNone(records[0]["target_responses_pct"][7])
        self.assertEqual(records[0]["missing_target_responses"], 1)
        self.assertEqual(len(records[0]["concentrations_m"]), 15)

    def test_purity_filter_and_explicit_sensitivity(self):
        rows = actual_pair()
        rows[0]["PURITY_RATING"] = "F"
        records, diag = self.parse(rows)
        self.assertFalse(records)
        self.assertEqual(diag["pair_exclusion_reasons"]["purity_not_in_declared_set"], 1)
        records, diag = self.parse(rows, purity=None)
        self.assertEqual(len(records), 1)
        self.assertIsNone(diag["purity_filter"])

    def test_nonempty_extra_field_rejected(self):
        with self.assertRaisesRegex(ValueError, "overflow"):
            self.parse(actual_pair(), overflow="unexpected-content")

    def test_beta_lactamase_registry_does_not_use_raw_channel(self):
        protocol = "tox21-ar-bla-antagonist-p1"
        rows = actual_pair()
        for row in rows:
            row["PROTOCOL_NAME"] = protocol
        rows[0]["SAMPLE_DATA_TYPE"] = "ratio_3"
        rows[1]["SAMPLE_DATA_TYPE"] = "viability_3"
        extra = dict(rows[0], SAMPLE_DATA_TYPE="ch1_3", SAMPLE_DATA_ID="999")
        records, diag = self.parse(rows + [extra], protocol)
        self.assertEqual(len(records), 1)
        self.assertEqual(diag["row_exclusions"]["nonprimary_channel"], 1)

    def test_parent_group_repeats_retained_without_aggregation(self):
        rows = actual_pair()
        other = []
        for i, row in enumerate(rows):
            copy = dict(row, SAMPLE_ID="different-salt-sample", SAMPLE_DATA_ID=f"salt-{i}",
                        SMILES=row["SMILES"] + ".[Cl-]")
            other.append(copy)
        records, diag = self.parse(rows + other)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["group_id"], records[1]["group_id"])
        self.assertNotEqual(records[0]["structure_id"], records[1]["structure_id"])
        self.assertEqual(diag["retained_compound_groups"], 1)

    def test_reused_curve_id_across_samples_is_rejected(self):
        rows = actual_pair()
        other = [dict(row, SAMPLE_ID="other-sample") for row in rows]
        records, diag = self.parse(rows + other)
        self.assertEqual(records, [])
        self.assertEqual(diag["pair_exclusion_reasons"]["curve_id_reused_across_pairs"], 2)

    def test_download_and_cache_digest(self):
        payload = archive_bytes(actual_pair())
        manifest = data.source_manifest()
        manifest["protocols"][PROTOCOL]["expected_bytes"] = len(payload)
        manifest["protocols"][PROTOCOL]["sha256"] = hashlib.sha256(payload).hexdigest()
        class Response(io.BytesIO):
            headers = {"ETag": "fixture", "Content-Type": "application/zip"}
        with tempfile.TemporaryDirectory() as tmp, patch.object(data, "source_manifest", return_value=manifest):
            with patch.object(data.urllib.request, "urlopen", return_value=Response(payload)) as fetch:
                path, receipt = data.download_protocol(PROTOCOL, Path(tmp))
                self.assertEqual(receipt["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(fetch.call_count, 1)
            with patch.object(data.urllib.request, "urlopen", side_effect=AssertionError("no redownload")):
                cached, repeated = data.download_protocol(PROTOCOL, Path(tmp))
                self.assertEqual(cached, path)
                self.assertEqual(repeated, receipt)
                path.write_bytes(payload[:-1] + b"X")
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    data.download_protocol(PROTOCOL, Path(tmp))

    def test_unknown_protocol_rejected(self):
        with self.assertRaisesRegex(ValueError, "undeclared"):
            self.parse(actual_pair(), protocol="invented-protocol")


if __name__ == "__main__":
    unittest.main()
