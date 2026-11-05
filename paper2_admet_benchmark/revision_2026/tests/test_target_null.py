import unittest
from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np
from paper2_admet_benchmark.revision_2026.target_null import query_permutation, generate_controls, method_names

@dataclass(frozen=True)
class Audit:
    view: str = "test"
    active: bool = True
    reason: str = "test"

class QueryNullTests(unittest.TestCase):
    def test_permutation_preserves_bins_and_is_id_order_invariant(self):
        p = np.linspace(0, 1, 40)
        names = np.array([f"id-{i}" for i in range(40)])
        perm, bins = query_permutation(p, names, seed=19)
        self.assertEqual(sorted(perm), list(range(40)))
        np.testing.assert_array_equal(bins, bins[perm])
        order = np.random.default_rng(5).permutation(40)
        other, _ = query_permutation(p[order], names[order], seed=19)
        self.assertEqual(dict(zip(names, names[perm])), dict(zip(names[order], names[order][other])))
        self.assertGreater(np.mean(perm != np.arange(40)), .5)

    def test_controls_keep_baseline_and_exact_class_budget(self):
        cp = np.linspace(.1, .9, 80)
        cy = np.tile([0, 1], 40)
        p = np.linspace(.01, .99, 40)
        ids = [str(i) for i in range(40)]
        base = np.column_stack((p <= .5, p > .5))
        ratios = {v: SimpleNamespace(source_weights=np.ones(80), target_weights=np.linspace(.1, 30, 40)) for v in ("a", "b")}
        audits = {v: Audit() for v in ratios}
        methods, diag = generate_controls(cp, cy, p, ids, base, ratios, audits, ("a", "b"), alpha=.1, seed=19, replicates=2)
        self.assertEqual(set(methods), set(method_names(2)))
        for name, membership in methods.items():
            self.assertTrue(np.all(~base | membership))
            if not name.startswith("matched_"):
                matched = methods[f"matched_{name}_omitted_label_probability"]
                for label in (0, 1):
                    pool = base[:, label] & (base.sum(axis=1) == 1)
                    self.assertEqual(int((membership[pool].sum(axis=1) == 2).sum()), int((matched[pool].sum(axis=1) == 2).sum()))
        self.assertEqual(diag["replicates"][0]["seed"], 79019)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError): query_permutation([.1, .2], ["same", "same"], seed=1)
        with self.assertRaises(ValueError): query_permutation([float("nan")], ["a"], seed=1)

if __name__ == "__main__": unittest.main()
