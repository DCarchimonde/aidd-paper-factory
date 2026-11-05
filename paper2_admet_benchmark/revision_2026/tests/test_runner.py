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
    def test_cuda_preflight_runs_tensor_backward_and_rejects_unsupported_gpu(self):
        from unittest.mock import MagicMock
        from paper2_admet_benchmark.revision_2026.execute_revision import resolve_device
        import torch
        probe = MagicMock()
        probe.__matmul__.return_value = probe
        probe.square.return_value = probe
        probe.mean.return_value = probe
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch, 'randn', return_value=probe) as create, \
             patch.object(torch.cuda, 'synchronize') as sync:
            self.assertEqual(resolve_device('auto'), 'cuda')
            create.assert_called_once_with((4, 4), device='cuda', requires_grad=True)
            probe.backward.assert_called_once()
            sync.assert_called_once()
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch, 'randn', side_effect=RuntimeError('no kernel image available')):
            with self.assertRaisesRegex(RuntimeError, 'before data acquisition'):
                resolve_device('auto')
        with patch.object(torch.cuda, 'is_available', return_value=False), \
             patch.object(torch, 'randn') as create:
            self.assertEqual(resolve_device('auto'), 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'cannot use a CUDA device'):
                resolve_device('cuda')
            create.assert_not_called()

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

    def test_declared_method_catalog_matches_runtime(self):
        from paper2_admet_benchmark.revision_2026 import execute_revision as execution
        from paper2_admet_benchmark.revision_2026.full_baselines import codrug, kmm_cp
        from paper2_admet_benchmark.revision_2026 import independent_data
        protocol = json.loads(runner.DEFAULT_PROTOCOL.read_text())
        specs = execution.case_specs(protocol, independent_data.case_specs())
        jobs = execution.build_required_jobs(protocol, specs, codrug.METHOD_NAMES, kmm_cp.METHOD_NAMES)
        self.assertEqual(len(specs), 160)
        self.assertEqual(len(jobs), 960)
        self.assertEqual(len(set(j["job_id"] for j in jobs)), 960)
        self.assertTrue(all("codrug_margin_swap" in {x["model"] for x in jobs if x["phase"] == phase}
                            for phase in {j["phase"] for j in jobs}))
        self.assertEqual(len(runner.expected_method_names(protocol, True)), 116)

    def test_every_gate_ablation_has_own_nontrivial_class_budget(self):
        from paper2_admet_benchmark.scripts.racer_c4.racer_c4_core import DensityRatioResult
        protocol = json.loads(runner.DEFAULT_PROTOCOL.read_text())
        lock = yaml.safe_load(runner.LEGACY_LOCK.read_text())
        y = np.repeat([0, 1], 100)
        cp = np.concatenate([np.linspace(.01, .2, 100), np.linspace(.8, .99, 100)])
        qp = np.linspace(.001, .999, 200)
        ratios = {view: DensityRatioResult(np.ones(200), np.roll(np.linspace(.1, 30, 200), index * 27),
                     .7, 200., 100., 100., 0., 0., .2, .1)
                  for index, view in enumerate((*runner.VIEWS, 'ecfp_bits'))}
        audits = {view: runner.audit_density_ratio(view, ratio, y, len(qp), **lock['transport']['audit'])
                  for view, ratio in ratios.items()}
        self.assertTrue(all(a.active for a in audits.values()))
        methods = runner.generate_methods(cp, y, qp, [f'id{i}' for i in range(200)], np.linspace(0, 1, 200),
            ratios, audits, lock, protocol, 19, .1, sensitivities=True)
        self.assertEqual(set(methods), set(runner.expected_method_names(protocol, True)))
        base = methods['mondrian_lac_empty_to_full']
        names = [name for name in methods if name.startswith('matched_tame_')]
        self.assertEqual(len(names), 18)
        nonzero = 0
        for name in names:
            reference = name[len('matched_'):-len('_omitted_label_probability')]
            for label in (0, 1):
                pool = (base.sum(1) == 1) & base[:, label]
                actual = int((methods[reference][pool].sum(1) == 2).sum())
                matched = int((methods[name][pool].sum(1) == 2).sum())
                self.assertEqual(actual, matched, (reference, label))
                nonzero += actual
        self.assertGreater(nonzero, 0, 'a trivial all-fallback test does not validate matched budgets')

    def test_pilot_subset_is_marked_and_uses_no_target_labels(self):
        from paper2_admet_benchmark.revision_2026.execute_revision import reduce_pilot_case
        source = [dict(structure_id=f"src{i:03d}", standardized_smiles="C") for i in range(200)]
        target = [dict(sample_id=f"s{i:03d}", structure_id=f"tar{i:03d}", standardized_smiles="CC") for i in range(50)]
        labels = {r["sample_id"]: {"ep": i % 2} for i, r in enumerate(target)}
        case = {"source_rows": source, "source_y": np.tile([0, 1], 100), "target_rows": target, "labels": labels}
        a = reduce_pilot_case(case)
        permuted = {**case, "labels": {k: {"ep": 1-v["ep"]} for k, v in labels.items()}}
        b = reduce_pilot_case(permuted)
        self.assertEqual([r["sample_id"] for r in a["target_rows"]], [r["sample_id"] for r in b["target_rows"]])
        self.assertEqual(len(a["source_rows"]), 160)
        self.assertEqual(len(a["target_rows"]), 32)
        self.assertTrue(a["split_metadata"]["pilot_only"])

    def test_result_package_contains_verified_manifest(self):
        import zipfile
        from paper2_admet_benchmark.revision_2026.execute_revision import package_results
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "run"
            output.mkdir()
            (output / "result.json").write_text('{"real_result": false}')
            package = Path(td) / "result.zip"
            receipt = package_results(output, package)
            self.assertEqual(receipt["sha256"], runner.sha256_file(package))
            with zipfile.ZipFile(package) as archive:
                manifest = json.loads(archive.read("BUNDLE_CONTENT_SHA256.json"))
                self.assertEqual(manifest["result.json"], runner.sha256_file(output / "result.json"))
            with self.assertRaises(FileExistsError):
                package_results(output, package)
            rebuilt = package_results(output, package, resume=True)
            self.assertEqual(rebuilt["sha256"], runner.sha256_file(package))
            self.assertEqual(len(list((Path(td) / "run_bundle_history").glob("*.zip"))), 1)

    def test_reporting_stage_resumes_after_partial_and_rejects_corruption(self):
        from paper2_admet_benchmark.revision_2026.execute_revision import completed_stage
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            def incomplete(directory):
                directory.mkdir()
                (directory / "table.csv").write_text("partial")
                raise RuntimeError("interrupted")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                completed_stage(output, "report", incomplete)
            def complete(directory):
                directory.mkdir()
                (directory / "table.csv").write_text("complete")
                return {"status": "complete"}
            self.assertEqual(completed_stage(output, "report", complete)["status"], "complete")
            self.assertEqual(len(list((output / "incomplete_stage_history").iterdir())), 1)
            self.assertEqual(completed_stage(output, "report", lambda _: self.fail("recomputed"))["status"], "complete")
            (output / "report/table.csv").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "changed cached input"):
                completed_stage(output, "report", complete)

    def test_lightweight_bundle_omits_weights_rows_and_temporary_files(self):
        import zipfile
        from paper2_admet_benchmark.revision_2026.execute_revision import package_results
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'run'
            cell = root / 'full/cell/backbone'
            cell.mkdir(parents=True)
            for name in ('training_checkpoint.pt', '.training_checkpoint.pt.abcd', 'state.tmp'):
                (cell / name).write_text(name)
            (cell.parent / 'predictions.csv').write_text('private rows')
            (cell.parent / 'metrics.csv').write_text('aggregate')
            (cell.parent / 'complete.json').write_text('{}')
            review = package_results(root, Path(td) / 'review.zip', review=True)
            full = package_results(root, Path(td) / 'full.zip')
            with zipfile.ZipFile(review['path']) as archive:
                names = archive.namelist()
                self.assertFalse(any(name.endswith('.pt') or name.endswith('predictions.csv') for name in names))
                self.assertTrue(any(name.endswith('metrics.csv') for name in names))
            with zipfile.ZipFile(full['path']) as archive:
                self.assertTrue(any(name.endswith('training_checkpoint.pt') for name in archive.namelist()))
                self.assertFalse(any('.abcd' in name or name.endswith('.tmp') for name in archive.namelist()))
            self.assertTrue((cell / '.training_checkpoint.pt.abcd').exists())

    def test_transport_parts_reassemble_and_reject_tamper(self):
        import subprocess
        import sys
        from paper2_admet_benchmark.revision_2026.execute_revision import split_bundle
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / 'sample.zip'
            archive.write_bytes(bytes(range(256)) * 31)
            receipt = {'path': str(archive), 'sha256': runner.sha256_file(archive)}
            parts = split_bundle(receipt, part_size=1000)
            folder = Path(parts['directory'])
            rebuilt = Path(td) / 'rebuilt.zip'
            subprocess.run([sys.executable, str(folder / 'reassemble.py'), str(rebuilt)], check=True, capture_output=True)
            self.assertEqual(rebuilt.read_bytes(), archive.read_bytes())
            self.assertEqual(split_bundle(receipt, part_size=1000)['parts'], 8)
            first = sorted(folder.glob('*.part*'))[0]
            first.write_bytes(b'corrupted')
            with self.assertRaisesRegex(RuntimeError, 'changed cached input'):
                split_bundle(receipt, part_size=1000)
            result = subprocess.run([sys.executable, str(folder / 'reassemble.py'), str(Path(td) / 'bad.zip')], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_execution_log_keeps_failure_and_traceback(self):
        from paper2_admet_benchmark.revision_2026.execute_revision import execution_log
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(ValueError, 'fixture failure'):
                with execution_log(root):
                    print('fixture stage reached')
                    raise ValueError('fixture failure')
            self.assertIn('fixture stage reached', (root / 'execution.log').read_text())
            self.assertIn('Traceback', (root / 'execution.log').read_text())
            failures = list((root / 'failures').glob('*.json'))
            self.assertEqual(len(failures), 1)
            self.assertEqual(json.loads(failures[0].read_text())['status'], 'execution_failed_not_complete')


if __name__ == "__main__":
    unittest.main()
