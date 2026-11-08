"""Information-boundary and scientific accounting regression checks."""
import unittest
import numpy as np

from .data import FrozenLabelOracle, case_specs
from .execute import METHODS, SyntheticCase, normalized_evaluation, run_policy


CONFIG = dict(delta=.05, eps_fn=.2, eps_fp=.2, score_bins=2, support_bins=1,
              count_prior="edge_mixture", query_block=4, audit_fractions=[0., .2, .5, 1.])


class ExecutionTests(unittest.TestCase):
    def test_real_oracle_risk_fields_are_not_dropped(self):
        oracle = FrozenLabelOracle(["a", "b", "c", "d"], [1, 1, 0, 0])
        oracle.query("b")
        point = dict(actions=np.array([0, 2, 1, 2]), queried=1, total_assays_required=2)
        result = normalized_evaluation(oracle, point)
        self.assertEqual(result["fn_rate"], .5)
        self.assertEqual(result["fp_rate"], .5)
        self.assertEqual(result["class_1_n"], 2)

    def test_absent_class_is_marked_not_falsely_perfect(self):
        oracle = FrozenLabelOracle(["a", "b"], [0, 0])
        result = normalized_evaluation(oracle, dict(actions=np.array([0, 0]), queried=0,
                                                    total_assays_required=0))
        self.assertIsNone(result["fn_rate"])
        self.assertEqual(result["fp_rate"], 0.)

    def test_shared_audit_history_is_not_retrospectively_overwritten(self):
        case = SyntheticCase()
        public = case.public_view("stack")
        oracle = case.new_oracle()
        points = []
        for point in run_policy(public, oracle.query, method="joint_uniform", seed=77, config=CONFIG):
            point["metrics"] = normalized_evaluation(oracle, point)
            points.append(point)
            self.assertEqual(point["queried"], len(point["query_history"]))
            self.assertEqual(point["total_assays_required"] + point["saved_items"], 160)
            self.assertEqual(point["remaining_assays_required"] + point["queried"], point["total_assays_required"])
        self.assertEqual(points[0]["query_history"], [])
        self.assertEqual(points[-1]["queried"], 160)
        self.assertEqual(points[-1]["saved_items"], 0)

    def test_replay_queries_are_reproducible_and_unique(self):
        case = SyntheticCase()
        config = {**CONFIG, "audit_fractions": [0., .1]}
        histories = []
        for _ in range(2):
            oracle = case.new_oracle()
            points = list(run_policy(case.public_view("stack"), oracle.query,
                                     method="joint_neyman", seed=78, config=config))
            histories.append(points[-1]["query_history"])
        self.assertEqual(histories[0], histories[1])
        self.assertEqual(len({r["sample_id"] for r in histories[0]}), len(histories[0]))

    def test_declared_screen_pools_are_four_without_loading_labels(self):
        specs = case_specs(endpoints=["AMES", "CYP2D6_Veith"])
        ids = [f"{s['phase']}__{s['endpoint']}__fit{s['fit_seed']}" for s in specs]
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual(len(METHODS), 8)


if __name__ == "__main__":
    unittest.main()
