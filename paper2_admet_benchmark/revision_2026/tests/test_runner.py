"""Integration contracts with synthetic inputs; no molecular outcomes asserted."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from argparse import Namespace

import numpy as np
import yaml

from paper2_admet_benchmark.revision_2026 import runner


class RunnerTests(unittest.TestCase):
    def test_rdkit_contract_checked_before_download(self):
        from rdkit import rdBase
        with tempfile.TemporaryDirectory() as td, \
             patch.object(rdBase, "rdkitVersion", "9999.01.1"), \
             patch.object(runner, "acquire_unlabeled_sources") as download:
            with self.assertRaisesRegex(RuntimeError, "requires RDKit 2026.03.4"):
                runner.prepare(Path(td) / "cache", {})
            download.assert_not_called()
        with patch.object(rdBase, "rdkitVersion", "2026.03.4"):
            runner.check_processing_environment()

    def test_resume_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "run"
            runner.ensure_run(out, {"protocol": "a", "code": "b"}, False)
            runner.ensure_run(out, {"protocol": "a", "code": "b"}, True)
            with self.assertRaisesRegex(RuntimeError, "provenance differs"):
                runner.ensure_run(out, {"protocol": "a", "code": "changed"}, True)
            with self.assertRaisesRegex(RuntimeError, "already contains"):
                runner.ensure_run(out, {"protocol": "a", "code": "b"}, False)

    def test_resume_checks_actual_completed_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            cell = Path(td)
            payload = cell / "predictions.csv"
            payload.write_text("include_0,include_1\n1,0\n")
            (cell / "complete.json").write_text(json.dumps({"files": {"predictions.csv": runner.sha256_file(payload)}}))
            self.assertTrue(runner.validate_cell_receipt(cell))
            payload.write_text("include_0,include_1\n0,1\n")
            with self.assertRaisesRegex(RuntimeError, "changed cached input"):
                runner.validate_cell_receipt(cell)

    def test_output_rejects_legacy_results(self):
        with self.assertRaises(ValueError):
            runner.protect_output(runner.P2 / "results/racer_c4_independent_final")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "unrelated.txt").write_text("keep")
            with self.assertRaisesRegex(RuntimeError, "nonempty"):
                runner.ensure_run(out, {}, False)

    def test_exact_tanimoto_and_zero_fingerprints(self):
        train = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]])
        query = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1]])
        np.testing.assert_allclose(runner.max_tanimoto(train, query, block=1), [2 / 3, 1, 1])

    def test_model_probabilities_share_roles_and_equal_logits(self):
        rng = np.random.default_rng(19)
        x = rng.integers(0, 2, (200, 16))
        qx = rng.integers(0, 2, (12, 16))
        y = np.tile([0, 1], 100)
        lock = yaml.safe_load(runner.LEGACY_LOCK.read_text())
        for item in lock["base_predictor"]["components"]:
            if "n_estimators" in item:
                item["n_estimators"] = 5
        roles, result = runner.fit_predictors(x, y, qx, lock, 11, 1, ["logistic", "stack", "equal_logit"])
        self.assertEqual([len(r) for r in roles], [120, 40, 40])
        self.assertEqual(len(set(np.concatenate(roles))), len(y))
        for name, (cp, qp, cl, ql) in result.items():
            self.assertEqual(len(cp), len(roles[2]))
            self.assertEqual(len(qp), len(qx))
            self.assertTrue(np.isfinite(cp).all())
        np.testing.assert_allclose(result["equal_logit"][1], runner.expit(result["equal_logit"][3][:, :4].mean(axis=1)))

    def test_writer_retains_missing_labels_and_exclusions(self):
        with tempfile.TemporaryDirectory() as td:
            rows = [dict(sample_id=f"s{i}", structure_id=f"z{i}", standardized_smiles="C",
                         structure_status="pass", excluded_structure_overlap=int(i == 2)) for i in range(3)]
            labels = {"s0": {"ep": 1.0}, "s1": {"ep": float("nan")}, "s2": {"ep": 0.0}}
            cal = [dict(structure_id="cal0", standardized_smiles="CC"), dict(structure_id="cal1", standardized_smiles="CCC")]
            methods = {0.1: {"test": np.array([[1, 0], [0, 1]], dtype=bool)}}
            runner.write_cell(Path(td), "retrospective_epa", "ep", 1, "logistic", rows, labels,
                np.array([True, True, False]), np.array([.1, .9, np.nan]), cal, np.array([0, 1]),
                np.array([.2, .8]), [{"role": "fit"}], methods, {})
            saved = runner.read_csv(Path(td) / "predictions.csv")
            self.assertEqual(saved[1]["y"], "")
            self.assertEqual(saved[2]["eligible"], "0")
            self.assertEqual((saved[2]["include_0"], saved[2]["include_1"]), ("1", "1"))
            metric = runner.read_csv(Path(td) / "metrics.csv")[0]
            self.assertEqual(metric["n"], "1")
            self.assertEqual(metric["n_missing_labels"], "1")
            self.assertTrue(runner.validate_cell_receipt(Path(td)))
            self.assertNotIn("y", runner.read_csv(Path(td) / "target_manifest.csv")[0])
            import pandas as pd
            from paper2_admet_benchmark.revision_2026.summarize import validate_frame
            validated = validate_frame(pd.read_csv(Path(td) / "predictions.csv"))
            self.assertEqual(len(validated), 1)

    def test_mock_sources_end_to_end_and_verified_resume(self):
        """Train real small estimators, emit files, validate summary, resume."""
        rng = np.random.default_rng(91)
        source = [dict(structure_id=f"src{i}", standardized_smiles="CC") for i in range(200)]
        targets = [dict(sample_id=f"s{i}", structure_id=f"target{i}", standardized_smiles="CCC",
                        structure_status="pass") for i in range(40)]
        targets[0]["structure_id"] = "src0"  # overlap must remain full set and excluded.
        targets[1]["structure_status"] = "excluded_synthetic_invalid"
        endpoint = "Tox21_NR_AhR"
        outcomes = {f"s{i}": {endpoint: float(i % 2)} for i in range(40)}
        outcomes["s2"][endpoint] = float("nan")
        y = np.tile([0, 1], 100)
        all_ids = {r["structure_id"] for r in source + targets}
        features = {key: rng.integers(0, 2, 16) for key in sorted(all_ids)}
        descriptors = {key: rng.normal(size=9) for key in sorted(all_ids)}
        protocol = json.loads(runner.DEFAULT_PROTOCOL.read_text())
        lock = yaml.safe_load(runner.LEGACY_LOCK.read_text())
        with tempfile.TemporaryDirectory() as td:
            cache, output = Path(td) / "cache", Path(td) / "run"
            cache.mkdir()
            (cache / "preparation.json").write_text("synthetic-source-only")
            label_path = cache / "outcomes.txt"
            label_path.write_text("synthetic-outcomes-only")
            args = Namespace(cache=cache, output=output, phase="development", endpoints=[endpoint],
                             seeds=[101], models=["logistic"], alphas=[0.1], jobs=1,
                             sensitivities=False, resume=False, protocol=runner.DEFAULT_PROTOCOL)
            with patch.object(runner, "prepare", return_value=({}, cache)), \
                 patch.object(runner, "load_target", return_value=(targets, outcomes, label_path)), \
                 patch.object(runner, "read_clean_endpoint", return_value=(source, y)), \
                 patch.object(runner, "fingerprints", side_effect=lambda rows: np.array([features[r["structure_id"]] for r in rows])), \
                 patch.object(runner, "physchem_features", side_effect=lambda rows: np.array([descriptors[r["structure_id"]] for r in rows])):
                runner.run(args, protocol, lock)
                cell = output / "cells" / f"development__{endpoint}__101__logistic"
                before = runner.sha256_file(cell / "predictions.csv")
                import pandas as pd
                from paper2_admet_benchmark.revision_2026.summarize import validate_frame
                validated = validate_frame(pd.read_csv(cell / "predictions.csv"))
                self.assertEqual(validated.sample_id.nunique(), 37)
                args.resume = True
                with patch.object(runner, "fit_predictors", side_effect=AssertionError("resume retrained completed cell")):
                    runner.run(args, protocol, lock)
                self.assertEqual(before, runner.sha256_file(cell / "predictions.csv"))
                receipt = json.loads((output / "run_complete.json").read_text())
                self.assertEqual(receipt["cells"], 1)
                self.assertFalse(receipt["epa_is_new_blinded_validation"])

    def test_optional_density_components_are_explicit_and_alpha_aligned(self):
        rng = np.random.default_rng(24)
        cx, qx = rng.normal(size=(20, 3)), rng.normal(size=(8, 3))
        cp, qp = rng.uniform(.05, .95, 20), rng.uniform(.05, .95, 8)
        y = np.tile([0, 1], 10)
        protocol = json.loads(runner.DEFAULT_PROTOCOL.read_text())
        outputs, diagnostics = runner.density_components(cx, qx, cp, y, qp,
            [0.1, 0.2], ["kde", "kmm"], protocol)
        self.assertEqual(set(outputs), {0.1, 0.2})
        for methods in outputs.values():
            self.assertEqual(set(methods), {"kde_ratio_mondrian_physchem_component", "kmm_mondrian_physchem_component"})
            self.assertTrue(all(member.shape == (8, 2) for member in methods.values()))
        self.assertFalse(diagnostics["kde"]["full_codrug_reproduction"])
        self.assertFalse(diagnostics["kmm"]["full_kmm_cp_reproduction"])


if __name__ == "__main__":
    unittest.main()
