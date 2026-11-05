import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from paper2_admet_benchmark.revision_2026.reporting import (
    PRIMARY_METHOD, PRIMARY_REFERENCE, _normalize, aggregate_metrics,
    build_report, gate_stratified_targeting, interpret_primary, recompute_metrics, result_text,
    verify_job, verify_matched_budgets,
)


def fixture():
    rows = []
    # Same class-specific budget, but different identities are expanded.
    # One expanded target has a missing label: this must not invalidate the
    # label-free budget match or force equal evaluated deferral rates.
    for method in ["mondrian_lac_empty_to_full", PRIMARY_METHOD, PRIMARY_REFERENCE]:
        for i in range(8):
            include = [int(i % 2 == 0), int(i % 2 == 1)]
            if (method == PRIMARY_METHOD and i == 0) or (method == PRIMARY_REFERENCE and i == 2):
                include = [1, 1]
            rows.append({"phase": "retrospective_epa", "endpoint": "A", "seed": 211,
                "model": "stack", "alpha": .1, "method": method, "sample_id": str(i),
                "structure_id": "structure-" + str(i), "y": np.nan if i == 2 else i % 2,
                "eligible": 1, "include_0": include[0], "include_1": include[1]})
    return _normalize(pd.DataFrame(rows))


class ReportingTests(unittest.TestCase):
    def test_matched_budget_uses_unlabeled_target_population(self):
        frame = fixture()
        verify_matched_budgets(frame)
        metrics = pd.DataFrame(recompute_metrics(frame))
        metrics = metrics[metrics.population == "full_eligible_labeled"].set_index("method")
        self.assertEqual(metrics.loc[PRIMARY_METHOD, "n_deferred"], 1)
        self.assertEqual(metrics.loc[PRIMARY_REFERENCE, "n_deferred"], 0)
        self.assertEqual(metrics.loc[PRIMARY_METHOD, "operational_defer_rate"], .125)
        self.assertEqual(metrics.loc[PRIMARY_REFERENCE, "operational_defer_rate"], .125)

    def test_mismatch_predicted_class_budget_is_rejected(self):
        frame = fixture()
        # Keep the total expansion count at one, change its original class.
        frame.loc[(frame.method == PRIMARY_REFERENCE) & (frame.sample_id == "2"), ["include_0", "include_1"]] = [1, 0]
        frame.loc[(frame.method == PRIMARY_REFERENCE) & (frame.sample_id == "3"), ["include_0", "include_1"]] = [1, 1]
        with self.assertRaisesRegex(ValueError, "class-specific"):
            verify_matched_budgets(frame)

    def test_selected_denominator_never_replaces_full_cohort(self):
        frame = fixture()[lambda x: x.method == PRIMARY_METHOD].copy()
        frame.loc[frame.sample_id == "1", "method_selected"] = 0
        frame.loc[frame.sample_id == "1", ["include_0", "include_1"]] = [1, 1]
        metrics = {r["population"]: r for r in recompute_metrics(frame)}
        self.assertEqual(metrics["full_eligible_labeled"]["n"], 7)
        self.assertEqual(metrics["selected_eligible_labeled"]["n"], 6)
        self.assertEqual(metrics["full_eligible_labeled"]["class_1_n"], 4)
        self.assertEqual(metrics["selected_eligible_labeled"]["class_1_n"], 3)

    def test_receipt_and_method_matrix_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "cell"
            cell.mkdir()
            prediction = cell / "predictions.csv"
            fixture().to_csv(prediction, index=False)
            receipt = {"status": "complete", "files": {"predictions.csv": hashlib.sha256(prediction.read_bytes()).hexdigest()}}
            (cell / "complete.json").write_text(json.dumps(receipt))
            job = {"job_id": "test", "phase": "retrospective_epa", "endpoint": "A", "seed": 211,
                "model": "stack", "alphas": [.1], "methods": ["mondrian_lac_empty_to_full", PRIMARY_METHOD, PRIMARY_REFERENCE],
                "predictions": "cell/predictions.csv", "completion": "cell/complete.json"}
            frame, _ = verify_job(root, job)
            self.assertEqual(len(frame), 24)
            bad = {**job, "methods": job["methods"] + ["missing_strong_baseline"]}
            with self.assertRaisesRegex(ValueError, "method/alpha matrix"):
                verify_job(root, bad)
            prediction.write_text(prediction.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                verify_job(root, job)

    def test_method_label_mismatch_rejected_even_with_valid_hash(self):
        frame = fixture()
        frame.loc[(frame.method == PRIMARY_REFERENCE) & (frame.sample_id == "0"), "y"] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame.to_csv(root / "predictions.csv", index=False)
            (root / "complete.json").write_text(json.dumps({"files": {"predictions.csv": hashlib.sha256((root / "predictions.csv").read_bytes()).hexdigest()}}))
            job = {"job_id": "badlabels", "phase": "retrospective_epa", "endpoint": "A", "seed": 211,
                "model": "stack", "alphas": [.1], "methods": sorted(set(frame.method)),
                "predictions": "predictions.csv", "completion": "complete.json"}
            with self.assertRaisesRegex(ValueError, "disagree on outcomes"):
                verify_job(root, job)

    def test_no_silent_nanmean_or_seed_probability_averaging(self):
        rows = []
        for seed, value in [(1, .2), (2, .8)]:
            rows.append({"phase": "retrospective_epa", "model": "stack", "alpha": .1,
                "method": PRIMARY_METHOD, "population": "full_eligible_labeled", "endpoint": "A",
                "seed": seed, "active_false_clearance": value, "macro_csy": np.nan if seed == 2 else .5})
        output = aggregate_metrics(pd.DataFrame(rows), {"panel": ["A"]}).iloc[0]
        self.assertAlmostEqual(output.active_false_clearance, .5)
        self.assertTrue(np.isnan(output.macro_csy))
        self.assertEqual(output.macro_csy_undefined_cells, 1)

    def test_primary_is_not_rescued_by_secondary_positive_result(self):
        rows = [{"role": "primary", "metric": "active_false_clearance", "comparison_id": "primary",
                 "delta": -.01, "ci_95_low": -.03, "ci_95_high": .01},
                {"role": "secondary", "metric": "active_false_clearance", "comparison_id": "secondary",
                 "delta": -.10, "ci_95_low": -.2, "ci_95_high": -.05}]
        result = interpret_primary(rows, matrix_complete=True, reference_complete=True)
        self.assertEqual(result["decision"], "NO_GO_ALGORITHM_ADVANTAGE")
        self.assertEqual(result["evidence"], "primary_advantage_not_established")
        rows[0]["ci_95_high"] = -.001
        self.assertEqual(interpret_primary(rows, matrix_complete=True, reference_complete=True)["decision"], "GO_TO_AUTHOR_REVIEW")
        self.assertEqual(interpret_primary(rows, matrix_complete=True, reference_complete=False)["decision"], "INCOMPLETE_DO_NOT_SUBMIT")

    def test_gate_strata_keep_undefined_class_cell_in_denominator(self):
        rows, gates = [], []
        for seed in [1, 2]:
            identity = {"phase": "retrospective_epa", "endpoint": "A", "seed": seed, "model": "stack"}
            gates.append(identity | {"original_gate_status": "both_pass"})
            for method in [PRIMARY_METHOD, PRIMARY_REFERENCE]:
                rows.append(identity | {"alpha": .1, "population": "full_eligible_labeled", "method": method,
                    "active_false_clearance": np.nan if seed == 2 else .1 if method == PRIMARY_METHOD else .2,
                    "defer_rate": .3})
        result = gate_stratified_targeting(pd.DataFrame(rows), gates).iloc[0]
        self.assertEqual(result.cells, 2)
        self.assertTrue(np.isnan(result.mean_excess_false_inactive))
        self.assertEqual(result.excess_false_inactive_undefined_cells, 1)
        self.assertEqual(result.mean_labeled_deferral_difference, 0)
        self.assertEqual(result.labeled_deferral_difference_undefined_cells, 0)

    def test_no_matrix_no_output_no_fabricated_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "required_jobs"):
                build_report(root, root / "report")
            self.assertFalse((root / "report").exists())

    def test_result_prose_uses_actual_values_and_negative_conclusion(self):
        paired = pd.DataFrame([{"role": "primary", "metric": "active_false_clearance", "method_value": .12,
            "reference_value": .08, "delta": .04, "ci_95_low": .01, "ci_95_high": .07,
            "phase": "retrospective_epa", "panel": "primary_six", "model": "stack", "alpha": .1,
            "method": PRIMARY_METHOD, "reference": PRIMARY_REFERENCE}])
        aggregate = pd.DataFrame([{"phase": "retrospective_epa", "panel": "primary_six", "alpha": .1,
            "population": "full_eligible_labeled", "model": "stack", "method": PRIMARY_METHOD,
            "active_false_clearance": .12, "minimum_class_coverage": .88, "macro_csy": .4, "defer_rate": .5}])
        sections, abstract, conclusion = result_text(aggregate, paired, {"evidence": "primary_simple_comparator_is_better"})
        self.assertIn("12.00", sections)
        self.assertIn("8.00", sections)
        self.assertIn("4.00", abstract)
        self.assertIn("favors the simpler", conclusion)
        self.assertNotIn("pending", sections.lower())

# This intentionally exercises the real execution manifest and complete matrix
# with artificial labels/sets only. No fixture output is scientific evidence.
# Opt in because 960 receipts, recomputation and LaTeX compilation are costly.
class CompleteExecutionContractTests(unittest.TestCase):
    @unittest.skipUnless(__import__('os').environ.get('RUN_FULL_REPORT_TEST') == '1',
                         'set RUN_FULL_REPORT_TEST=1 for complete synthetic contract')
    def test_actual_execution_manifest_to_figures_and_compiled_manuscript(self):
        import os
        from unittest.mock import patch
        from paper2_admet_benchmark.revision_2026 import reporting, execute_revision, independent_data, runner, synthetic_mechanisms
        from paper2_admet_benchmark.revision_2026.full_baselines import codrug, kmm_cp
        with tempfile.TemporaryDirectory(prefix='synthetic_reporting_contract_') as directory:
            root = Path(directory)
            protocol = json.loads((reporting.HERE / 'protocol.json').read_text())
            # Reduce bootstrap/simulation repetitions only in this artificial
            # protocol. The real molecular endpoint/seed/method matrix is intact.
            protocol['statistical_analysis']['bootstrap_replicates'] = 20
            protocol['synthetic_mechanisms'].update(replicates=2, n_calibration=100, n_target=100)
            specs = execute_revision.case_specs(protocol, independent_data.case_specs())
            jobs = execute_revision.build_required_jobs(protocol, specs, codrug.METHOD_NAMES, kmm_cp.METHOD_NAMES)
            self.assertEqual(len(jobs), 960)
            snapshot = root / 'protocol_snapshot.json'
            reporting.write_json(snapshot, protocol)
            manifest = {'execution_kind': 'formal_declared_matrix', 'protocol_sha256': reporting.sha256(snapshot),
                'phase_specs': execute_revision.build_phase_specs(specs), 'jobs': jobs,
                'test_fixture_only': True}
            reporting.write_json(root / 'required_jobs.json', manifest)
            for job in jobs:
                path = root / job['predictions']
                path.parent.mkdir(parents=True)
                rows = []
                for alpha in job['alphas']:
                    for method in job['methods']:
                        for i in range(8):
                            a, b = (1, 1) if method == 'full_set' else (int(i % 2 == 0), int(i % 2 == 1))
                            rows.append({k: job[k] for k in ['phase', 'endpoint', 'seed', 'model']} | {
                                'alpha': alpha, 'method': method, 'sample_id': f'fixture_{i}',
                                'structure_id': f'fixture_structure_{i}', 'y': (i // 2) % 2,
                                'eligible': 1, 'include_0': a, 'include_1': b,
                                'method_selected': 1, 'original_include_0': a, 'original_include_1': b,
                                'probability_1': .1 + i * .1, 'max_train_tanimoto': .2 + i * .05})
                pd.DataFrame(rows).to_csv(path, index=False)
                files = {'predictions.csv': reporting.sha256(path)}
                if job['family'] == 'shared':
                    audit = path.parent / 'audits.json'
                    reporting.write_json(audit, {v: {'active': bool(job['seed'] % 2), 'reason': 'synthetic_fixture'} for v in runner.VIEWS})
                    files['audits.json'] = reporting.sha256(audit)
                reporting.write_json(root / job['completion'], {'status': 'complete', 'files': files, 'synthetic_fixture_only': True})
            synthetic_mechanisms.run(root / 'synthetic_mechanisms', **protocol['synthetic_mechanisms'])
            template = root / 'fixture_template.tex'
            text = reporting.MANUSCRIPT.read_text()
            text = text.replace('\\title{Class-Specific Errors', '\\title{SYNTHETIC SOFTWARE TEST---NOT RESEARCH RESULTS. Class-Specific Errors')
            template.write_text(text)
            print('Synthetic reporting contract: 960 actual-builder jobs prepared', flush=True)
            with patch.object(reporting, 'MANUSCRIPT', template):
                report = reporting.build_report(root, root / 'report', compile_pdf=True)
            self.assertEqual(report['matrix_jobs'], 960)
            self.assertEqual(report['manuscript']['pdf_status'], 'compiled')
            self.assertEqual(report['decision']['decision'], 'NO_GO_ALGORITHM_ADVANTAGE')
            self.assertTrue((root / 'report/figures/additional_assay_families.pdf').is_file())
            self.assertTrue((root / 'report/gate_stratified_own_budget.csv').is_file())
            self.assertNotIn('\\pending{', (root / 'report/latex/manuscript_completed.tex').read_text())
            retain = os.environ.get('REPORT_TEST_ARTIFACT_DIR')
            if retain:
                import shutil
                artifact = Path(retain)
                artifact.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / 'report/latex/manuscript_completed.pdf', artifact / 'SYNTHETIC_SOFTWARE_TEST_NOT_RESEARCH.pdf')
                reporting.write_json(artifact / 'contract_test_receipt.json', {'synthetic_fixture_only': True,
                    'jobs': len(jobs), 'contrasts': len(reporting.declared_contrasts(protocol, jobs)),
                    'code_sha256': reporting.sha256(Path(reporting.__file__)), 'compiled': True})
            print('Synthetic reporting contract: complete report, figures, mechanism strata and PDF passed', flush=True)


if __name__ == "__main__":
    unittest.main()
