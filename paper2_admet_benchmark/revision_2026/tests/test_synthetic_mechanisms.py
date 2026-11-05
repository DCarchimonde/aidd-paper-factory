import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from paper2_admet_benchmark.revision_2026 import synthetic_mechanisms as sm


class SyntheticMechanismTests(unittest.TestCase):
    def test_constant_query_weight_has_no_ranking_gain_at_matched_budget(self):
        # Unequal, nonconstant source weights can move the threshold/budget.
        # Query-constant weights still produce only a p1 cutoff in each view.
        rng = np.random.default_rng(2301)
        cp = rng.uniform(.01, .99, 160)
        y = np.tile([0, 1], 80)
        # Include ties to exercise the exact selected-set equality, not just
        # agreement on aggregate budget or error metrics.
        qp = np.repeat(np.linspace(.001, .999, 80), 2)
        ids = [f"query_{i:04d}" for i in rng.permutation(len(qp))]
        cs, qs = sm.binary_nonconformity(cp), sm.binary_nonconformity(qp)
        total_expanded = 0
        for alpha in (.05, .1, .2):
            base = sm.core.repair_empty(sm.core.conformal_sets(cp, y, qp, alpha))
            for constants in ((.1, 1.), (5., 30.), (1000., 1000.)):
                weighted = [sm.weighted_mondrian_sets(cs, y, qs,
                    rng.lognormal(0, .8, len(cp)), np.full(len(qp), value), alpha)
                    for value in constants]
                for quorum in ("all", "any"):
                    audits = [SimpleNamespace(active=True, view=f"view_{i}") for i in range(2)]
                    member = sm.transport_envelope(base, weighted, audits,
                        protected_labels=[1], quorum=quorum, minimum_active_views=2).membership
                    matched = sm.core.matched_budget_deferral(base, member, qp, ids, match_classes=True)
                    np.testing.assert_array_equal(member, matched)
                    total_expanded += int(np.sum((base.sum(1) == 1) & (member.sum(1) == 2)))
        self.assertGreater(total_expanded, 0)  # Equality is not a no-op artifact.

    def test_oracle_ratio_matches_independent_density_calculation(self):
        x = np.array([[-2., .8], [0., 0.], [1., -3.], [4., 2.]])
        expected = multivariate_normal.pdf(x, mean=[1,0])/multivariate_normal.pdf(x, mean=[0,0])
        np.testing.assert_allclose(sm.gaussian_ratio(x), expected, rtol=1e-13)
        self.assertGreater(sm.gaussian_ratio(np.array([[6., 0.]]))[0], 20)

    def test_same_x_concept_shift_has_identical_audits_and_predictions(self):
        rows, audits = sm.run_replicate(seed=12, n_calibration=200, n_target=200)
        df, ad = pd.DataFrame(rows), pd.DataFrame(audits)
        pair = df[df.scenario.isin(["no_shift", "same_x_concept_shift"])]
        self.assertEqual(pair.label_free_fingerprint.nunique(), 1)
        for _, group in pair.groupby("method"):
            self.assertEqual(group.n_deferred.nunique(), 1)
            self.assertEqual(group.n_singleton.nunique(), 1)
        self.assertEqual(ad[ad.scenario.isin(["no_shift", "same_x_concept_shift"])].label_free_fingerprint.nunique(), 1)
        base = pair[pair.method == "mondrian_lac_empty_to_full"]
        self.assertGreater(base.overall_coverage.max()-base.overall_coverage.min(), .15)
        self.assertNotIn("target_y", inspect.signature(sm.predict_without_target_labels).parameters)

    def test_support_loss_never_fabricates_an_oracle(self):
        rows, _ = sm.run_replicate(seed=4, n_calibration=100, n_target=100)
        support = [row for row in rows if row["scenario"] == "support_loss"]
        self.assertTrue(all(row["source_support"] == "violated" for row in support))
        self.assertTrue(all(row["method"] != "oracle_x_ratio_mondrian" for row in support))
        self.assertTrue(all("unavailable" in row["oracle_status"] for row in support))

    def test_report_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"simulation"
            report = sm.run(path, replicates=1, n_calibration=100, n_target=100)
            self.assertTrue(report["same_x_concept_shift_prediction_and_audit_identity"])
            self.assertEqual(len(list(path.iterdir())), 4)
            with self.assertRaises(FileExistsError):
                sm.run(path, replicates=1)


if __name__ == "__main__":
    unittest.main()
