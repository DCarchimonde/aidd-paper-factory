"""Structural leakage/annotation tests; optional frozen real-data integration."""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from paper2_admet_benchmark.revision_2026.independent_data import (
    ENDPOINTS, OUTER_SEEDS, SEEDS, SPLITS, _download_locked, case_specs,
    clean_rows, load_case, outer_split, prepare_endpoint, source_manifest,
)


class IndependentDataTests(unittest.TestCase):
    def test_source_identity_and_complete_case_matrix(self):
        manifest = source_manifest()
        self.assertEqual(manifest["sources"]["hERG"]["dataverse_file_id"], 4259588)
        self.assertEqual(manifest["sources"]["hERG"]["raw_rows"], 655)
        specs = case_specs()
        self.assertEqual(manifest["sources"]["CYP2D6_Veith"]["dataverse_file_id"], 4259580)
        self.assertEqual(len(specs), 40)
        self.assertEqual(len({(c["endpoint"], c["split"], c["seed"]) for c in specs}), 40)
        self.assertEqual({c["endpoint"] for c in specs}, set(ENDPOINTS))
        self.assertEqual({c["seed"] for c in specs}, set(SEEDS))
        for c in specs:
            self.assertEqual(c["phase"], "independent_" + c["split"])

    def test_annotation_conflict_excludes_whole_structure_not_majority(self):
        raw = [
            {"Drug_ID": "z", "Drug": "CCO", "Y": "1"},
            {"Drug_ID": "a", "Drug": "OCC", "Y": "1.0"},
            {"Drug_ID": "b", "Drug": "CCN", "Y": "0"},
            {"Drug_ID": "c", "Drug": "NCC", "Y": "1"},
            {"Drug_ID": "d", "Drug": "CCN", "Y": "1"},
            {"Drug_ID": "e", "Drug": "CCC", "Y": "0.5"},
            {"Drug_ID": "f", "Drug": "CCC", "Y": "nan"},
            {"Drug_ID": "g", "Drug": "invalid_smiles", "Y": "0"},
        ]
        rows, excluded, counts = clean_rows(raw, "AMES")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_record_id"], "a")
        self.assertEqual(rows[0]["target"], 1)
        self.assertEqual(rows[0]["structure_status"], "pass")
        self.assertEqual(counts["conflicting_structure_groups"], 1)
        self.assertEqual(counts["exclusion_reasons"]["conflicting_structure_labels"], 3)
        self.assertEqual(counts["exclusion_reasons"]["same_label_structure_duplicate"], 1)
        self.assertEqual(len(excluded) + len(rows), len(raw))
        other, _, _ = clean_rows(raw, "DILI")
        self.assertEqual(other[0]["structure_id"], rows[0]["structure_id"])
        self.assertNotEqual(other[0]["sample_id"], rows[0]["sample_id"])

    def test_outer_selection_cannot_read_labels_and_is_order_invariant(self):
        # No labels are present in structural-only input. Adding adversarial
        # labels and reversing row order must not affect either outer split.
        structures = [dict(structure_id=f"m{i:03}", scaffold_id=f"s{i // 3:02}")
                      for i in range(60)]
        changed = [dict(row, target=i % 2) for i, row in enumerate(structures[::-1])]
        for split in SPLITS:
            a, b = outer_split(structures, split)
            self.assertEqual((a, b), outer_split(changed, split))
            self.assertEqual(len(b), 12)
            self.assertFalse(set(a) & set(b))
            self.assertEqual(set(a) | set(b), {r["structure_id"] for r in structures})
            if split == "scaffold":
                sa = {r["scaffold_id"] for r in structures if r["structure_id"] in a}
                sb = {r["scaffold_id"] for r in structures if r["structure_id"] in b}
                self.assertFalse(sa & sb)

    def test_infeasible_scaffold_partition_does_not_search_alternative_seeds(self):
        rows = [dict(structure_id=str(i), scaffold_id="one_family") for i in range(10)]
        with self.assertRaisesRegex(RuntimeError, "empty role"):
            outer_split(rows, "scaffold")

    def test_duplicate_structure_or_unknown_regime_fails(self):
        with self.assertRaises(ValueError):
            outer_split([{"structure_id": "a"}, {"structure_id": "a"}], "random")
        with self.assertRaises(ValueError):
            outer_split([{"structure_id": "a"}, {"structure_id": "b"}], "optimistic")

    def test_changed_cached_raw_bytes_fail_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.tab"
            path.write_bytes(b"changed")
            source = {"sha256": hashlib.sha256(b"original").hexdigest()}
            with patch("urllib.request.urlopen", side_effect=AssertionError("must not fetch")):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    _download_locked(source, path)

    def test_cases_outside_locked_matrix_are_rejected_before_download(self):
        with self.assertRaises(ValueError):
            load_case(Path("does_not_exist"), "AMES", "random", 999)
        with self.assertRaises(ValueError):
            load_case(Path("does_not_exist"), "AMES", "selected_scaffold", 311)


@unittest.skipUnless(os.environ.get("INDEPENDENT_DATA_CACHE"),
                     "set INDEPENDENT_DATA_CACHE to a prepared frozen real-data cache")
class IndependentRealDataTests(unittest.TestCase):
    def test_frozen_raw_cleaning_roles_and_target_identity_all_40_cases(self):
        from paper2_admet_benchmark.scripts.racer_c4.run_prospective_racer_c4 import split_internal_roles
        cache = Path(os.environ["INDEPENDENT_DATA_CACHE"])
        for endpoint in ENDPOINTS:
            prepared = prepare_endpoint(cache, endpoint)
            self.assertEqual(prepared["counts"]["raw_rows"],
                             source_manifest()["sources"][endpoint]["raw_rows"])
            for split in SPLITS:
                reference_ids = None
                for seed in SEEDS:
                    case = load_case(cache, endpoint, split, seed)
                    source, target = case["source_rows"], case["target_rows"]
                    source_ids = {r["structure_id"] for r in source}
                    target_ids = {r["structure_id"] for r in target}
                    self.assertFalse(source_ids & target_ids)
                    self.assertEqual(len(source) + len(target), prepared["counts"]["clean_structures"])
                    self.assertTrue(case["eligible"].all())
                    self.assertTrue(all("target" not in row for row in target))
                    self.assertEqual(set(case["labels"]), {r["sample_id"] for r in target})
                    self.assertEqual(case["split_metadata"]["outer_seed"], OUTER_SEEDS[split])
                    if reference_ids is not None:
                        self.assertEqual(target_ids, reference_ids)
                    reference_ids = target_ids
                    roles = split_internal_roles(case["source_y"], seed)
                    role_arrays = [np.asarray(role) for role in roles]
                    self.assertEqual(len(role_arrays[2]), case["split_metadata"]["n_source_calibration_expected"])
                    self.assertEqual(len(set(np.concatenate(role_arrays))), len(source))
                    self.assertEqual(sum(map(len, role_arrays)), len(source))
                    for indices in role_arrays:
                        self.assertEqual(set(case["source_y"][indices]), {0, 1})
                    if endpoint in ("hERG", "DILI"):
                        self.assertLess(len(role_arrays[2]), 100)
                    else:
                        self.assertGreaterEqual(len(role_arrays[2]), 100)
                    if split == "scaffold":
                        self.assertFalse({r["scaffold_id"] for r in source} &
                                         {r["scaffold_id"] for r in target})
                    if endpoint == "DILI":
                        self.assertLess(len(target), 100)
                        self.assertTrue(case["split_metadata"]["target_n_below_historical_100_gate"])


if __name__ == "__main__":
    unittest.main()
