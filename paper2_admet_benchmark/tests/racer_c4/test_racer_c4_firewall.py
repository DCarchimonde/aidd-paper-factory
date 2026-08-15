from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "paper2_admet_benchmark" / "scripts" / "racer_c4"
sys.path.insert(0, str(SCRIPT_DIR))

import racer_c4_io as io_contract


ENDPOINTS = list(io_contract.ENDPOINT_PROPERTIES)


class RacerC4FirewallTests(unittest.TestCase):
    def label_text(self) -> str:
        headers = ["Sample ID", *[io_contract.ENDPOINT_PROPERTIES[key] for key in ENDPOINTS]]
        row = ["NCGC0001", *[str(index % 2) for index in range(len(ENDPOINTS))]]
        return "\t".join(headers) + "\n" + "\t".join(row) + "\n"

    def test_final_label_parser_accepts_official_style_header(self) -> None:
        observed = io_contract.parse_final_label_text(self.label_text(), ENDPOINTS)
        self.assertEqual(set(observed), {"NCGC0001"})
        self.assertEqual(observed["NCGC0001"][ENDPOINTS[0]], 0.0)
        self.assertEqual(observed["NCGC0001"][ENDPOINTS[1]], 1.0)

    def test_final_label_open_is_blocked_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label = root / "labels.txt"
            label.write_text(self.label_text(), encoding="utf-8")
            with self.assertRaises(PermissionError):
                io_contract.open_final_labels_after_promotion(
                    label,
                    root / "missing.json",
                    hashlib.sha256(label.read_bytes()).hexdigest(),
                    ENDPOINTS,
                )

    def test_final_label_open_requires_bound_prediction_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label = root / "labels.txt"
            label.write_text(self.label_text(), encoding="utf-8")
            label_hash = hashlib.sha256(label.read_bytes()).hexdigest()
            promotion = root / "promotion.json"
            promotion.write_text(
                json.dumps(
                    {
                        "status": "predictions_sealed_before_final_labels",
                        "development_gate_passed": True,
                        "final_labels_opened": False,
                        "expected_final_label_sha256": label_hash,
                        "sealed_predictions_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            observed = io_contract.open_final_labels_after_promotion(
                label, promotion, label_hash, ENDPOINTS
            )
            self.assertIn("NCGC0001", observed)
            record = json.loads(promotion.read_text(encoding="utf-8"))
            record["sealed_predictions_sha256"] = "not-a-hash"
            promotion.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(PermissionError):
                io_contract.open_final_labels_after_promotion(
                    label, promotion, label_hash, ENDPOINTS
                )


if __name__ == "__main__":
    unittest.main()
