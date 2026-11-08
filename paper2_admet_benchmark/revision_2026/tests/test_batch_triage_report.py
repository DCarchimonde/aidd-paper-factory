import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paper2_admet_benchmark.revision_2026.batch_triage.report import (
    aggregate_rows, build_report, summarize_pairs,
)


def row(case, seed, saved, *, method="joint_lookahead", n=100, absent=False):
    return dict(case_id=case, predictor="stack", audit_seed=seed, method=method,
                eps_fn=.1, eps_fp=.1, audit_budget=int(.2 * n), n_target=n,
                queried=int(.2 * n), saved_fraction=saved, total_assays_required=n * (1-saved),
                fn_rate=None if absent else .02, fp_rate=.01,
                class_1_n=0 if absent else n // 2, class_0_n=n if absent else n // 2,
                violated_realized_class_risk=False)


class BatchReportTests(unittest.TestCase):
    def test_cases_are_not_pooled_and_absent_class_stays_nan(self):
        rows = [row("rare", 1, .2, absent=True), row("rare", 2, .4, absent=True),
                row("other", 1, .7), row("other", 2, .7)]
        result = aggregate_rows(rows).set_index("case_id")
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result.loc["rare", "mean_saved_fraction"], .3)
        self.assertAlmostEqual(result.loc["other", "mean_saved_fraction"], .7)
        self.assertTrue(np.isnan(result.loc["rare", "mean_fn_rate"]))
        self.assertEqual(result.loc["rare", "n_defined_fn_rate"], 0)
        self.assertEqual(result.loc["other", "n_defined_fn_rate"], 2)
        with self.assertRaises(ValueError):
            aggregate_rows([rows[0], rows[0]])

    def test_pair_standard_error_is_across_exact_audit_repeats(self):
        base = dict(case_id="one", predictor="stack", reference="joint_uniform", eps_fn=.1,
                    eps_fp=.1, audit_budget=20)
        result = summarize_pairs([dict(base, audit_seed=1, delta_saved_fraction=0),
                                  dict(base, audit_seed=2, delta_saved_fraction=.2)])
        self.assertAlmostEqual(result.iloc[0]["mean_delta_saved_fraction"], .1)
        self.assertAlmostEqual(result.iloc[0]["mc_se_delta_saved_fraction"], .1)
        single = summarize_pairs([dict(base, audit_seed=1, delta_saved_fraction=0)])
        self.assertTrue(np.isnan(single.iloc[0]["pointwise_mc_t95_low"]))

    def test_saved_artifacts_preserve_na_and_mark_pilot(self):
        rows = [row("synthetic_a", 1, 0, n=95, absent=True),
                row("synthetic_a", 1, 0, method="joint_uniform", n=95, absent=True)]
        pair = {k: rows[0][k] for k in ("case_id", "predictor", "audit_seed", "eps_fn", "eps_fp", "audit_budget")}
        pair.update(reference="joint_uniform", delta_saved_fraction=0)
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = build_report(rows, [pair], temporary, "pilot")
            main = pd.read_csv(artifacts["main_20pct"])
            self.assertEqual(len(main), 2)
            self.assertTrue(main["mean_fn_rate"].isna().all())
            self.assertTrue((main["audit_budget"] == 19).all())
            self.assertEqual(len(artifacts["figures"]), 2)
            self.assertTrue(all(Path(path).stat().st_size > 1000 for path in artifacts["figures"]))
            self.assertIn("不是分子实验结果", Path(artifacts["readout"]).read_text())


if __name__ == "__main__":
    unittest.main()
