"""Small, fixed mechanism checks; not evidence of molecular performance.

The classifier is the known source posterior, fixed before calibration. Two
synthetic feature views feed the existing density estimator/audits. The no-shift
and concept-shift scenarios have literally identical source and target X,
predictions, and audit results; only target Y changes. Consequently label-free
transport audits cannot distinguish these two conditional-label mechanisms.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from paper2_admet_benchmark.revision_2026 import core
from paper2_admet_benchmark.scripts.racer_c4.racer_c4_core import (
    audit_density_ratio, binary_nonconformity, cross_fitted_density_ratio,
    transport_envelope, weighted_mondrian_sets,
)

SCENARIOS = ("no_shift", "gaussian_covariate_shift", "support_loss", "same_x_concept_shift")


def posterior(x):
    """Known source P(Y=1|X), shared by all covariate-shift scenarios."""
    return expit(2*np.asarray(x)[:, 0] + .5*np.asarray(x)[:, 1] - 1)


def gaussian_ratio(x, shift=1.0):
    """Exact N((shift,0),I)/N((0,0),I); never clipped."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("Gaussian ratio requires finite two-dimensional features")
    if not np.isfinite(shift):
        raise ValueError("shift must be finite")
    return np.exp(shift*x[:, 0] - shift*shift/2)


def _views(x):
    return {"linear": x, "quadratic": np.column_stack((x, x*x))}


def predict_without_target_labels(cal_x, cal_y, target_x, *, alpha=.1, seed=0,
                                  oracle_weights=None):
    """No target outcomes enter this API; all predictions precede scoring."""
    cp, tp = posterior(cal_x), posterior(target_x)
    base = core.repair_empty(core.conformal_sets(cp, cal_y, tp, alpha))
    ids = [f"synthetic_{i:06d}" for i in range(len(tp))]
    priority = np.where(base[:, 0], tp, 1-tp)
    audits, weighted, ratios = [], [], {}
    for view in ("linear", "quadratic"):
        ratio = cross_fitted_density_ratio(_views(cal_x)[view], _views(target_x)[view],
                                          cal_y, seed=int(seed), folds=5)
        ratios[view] = ratio
        audits.append(audit_density_ratio(view, ratio, cal_y, len(tp)))
        weighted.append(weighted_mondrian_sets(binary_nonconformity(cp), cal_y,
            binary_nonconformity(tp), ratio.source_weights, ratio.target_weights, alpha))
    methods = {"mondrian_lac_empty_to_full": base}
    for name, active in (("tame_label1_active", audits),
                         ("tame_label1_ungated", [replace(a, active=True, reason="synthetic_ungated") for a in audits])):
        methods[name] = transport_envelope(base, weighted, active, protected_labels=[1],
                                           quorum="all", minimum_active_views=2).membership
        methods[f"matched_{name}_omitted_label_probability"] = core.matched_budget_deferral(
            base, methods[name], priority, ids, match_classes=True)
    for view, member in zip(("linear", "quadratic"), weighted):
        methods[f"estimated_ratio_{view}_mondrian"] = member
    if oracle_weights is not None:
        sw, tw = oracle_weights
        methods["oracle_x_ratio_mondrian"] = weighted_mondrian_sets(
            binary_nonconformity(cp), cal_y, binary_nonconformity(tp), sw, tw, alpha)
    digest = hashlib.sha256()
    for array in (cal_x, cal_y, target_x, cp, tp):
        digest.update(np.ascontiguousarray(array).tobytes())
    for name in sorted(methods):
        digest.update(name.encode()); digest.update(methods[name].tobytes())
    for view in sorted(ratios):
        digest.update(ratios[view].source_weights.tobytes())
        digest.update(ratios[view].target_weights.tobytes())
    audit_values = [asdict(a) for a in audits]
    digest.update(json.dumps(audit_values, sort_keys=True).encode())
    return methods, {"views": audit_values, "fingerprint": digest.hexdigest(),
                     "active_views": sum(a.active for a in audits)}


def run_replicate(seed=0, n_calibration=400, n_target=800, alpha=.1):
    """Four prespecified scenarios with no result-dependent parameter search."""
    for value in (n_calibration, n_target):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 100:
            raise ValueError("each cohort requires at least 100 rows")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    rng = np.random.default_rng(int(seed))
    cal = rng.normal(size=(n_calibration, 2))
    cal_y = (rng.random(n_calibration) < posterior(cal)).astype(int)
    target = rng.normal(size=(n_target, 2))
    uniform = rng.random(n_target)
    common, audit = predict_without_target_labels(cal, cal_y, target, alpha=alpha, seed=seed,
                    oracle_weights=(np.ones(n_calibration), np.ones(n_target)))
    contexts = [
        ("no_shift", common, audit, (uniform < posterior(target)).astype(int),
         "holds", "holds", "available_and_joint_ratio_valid"),
        ("same_x_concept_shift", common, audit, (uniform < 1-posterior(target)).astype(int),
         "violated", "holds", "available_x_ratio_but_joint_ratio_invalid"),
    ]
    shifted = rng.normal(size=(n_target, 2)) + np.array([1., 0.])
    methods, details = predict_without_target_labels(cal, cal_y, shifted, alpha=alpha, seed=seed,
                    oracle_weights=(gaussian_ratio(cal), gaussian_ratio(shifted)))
    contexts.append(("gaussian_covariate_shift", methods, details,
        (rng.random(n_target) < posterior(shifted)).astype(int),
        "holds", "holds", "available_and_joint_ratio_valid"))
    bounded_cal = rng.uniform(-1, 1, size=(n_calibration, 2))
    bounded_y = (rng.random(n_calibration) < posterior(bounded_cal)).astype(int)
    unsupported = rng.uniform(-1, 1, size=(n_target, 2))
    # Exactly half of the target rows are outside source support in x[0].
    unsupported[n_target//2:, 0] = rng.uniform(2, 3, size=n_target-n_target//2)
    methods, details = predict_without_target_labels(bounded_cal, bounded_y, unsupported,
                                                      alpha=alpha, seed=seed)
    contexts.append(("support_loss", methods, details,
        (rng.random(n_target) < posterior(unsupported)).astype(int),
        "holds", "violated", "unavailable_target_not_absolutely_continuous"))
    rows, audit_rows = [], []
    for scenario, methods, details, y, conditional, support, oracle in contexts:
        common_fields = {"scenario": scenario, "seed": int(seed), "alpha": float(alpha),
                         "conditional_invariance": conditional, "source_support": support,
                         "oracle_status": oracle, "active_views": details["active_views"],
                         "label_free_fingerprint": details["fingerprint"]}
        for method, member in methods.items():
            rows.append({**common_fields, "method": method, **core.decision_metrics(y, member)})
        audit_rows.extend({**common_fields, **view} for view in details["views"])
    paired = [c for c in contexts if c[0] in {"no_shift", "same_x_concept_shift"}]
    assert paired[0][2] == paired[1][2]
    return rows, audit_rows


def run(output: Path, *, replicates=30, seed=73000, n_calibration=400, n_target=800, alpha=.1):
    """Write reproducible descriptive synthetic results; never overwrite a run."""
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows, audits = [], []
    for index in range(replicates):
        one, details = run_replicate(seed+index, n_calibration, n_target, alpha)
        rows.extend(one); audits.extend(details)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"replicate_metrics.csv", index=False)
    pd.DataFrame(audits).to_csv(output/"transport_audits.csv", index=False)
    metrics = ["minimum_class_coverage", "class_1_coverage", "false_clearance",
               "macro_csy", "deferred_rate", "active_views"]
    frame.groupby(["scenario", "method"])[metrics].agg(["mean", "std"]).to_csv(output/"descriptive_summary.csv")
    report = {"status": "synthetic_mechanism_checks_completed_not_molecular_evidence",
        "replicates": replicates, "first_seed": seed, "n_calibration": n_calibration,
        "n_target": n_target, "alpha": alpha,
        "scenario_names": list(SCENARIOS), "classifier": "frozen_known_source_posterior",
        "views": ["raw_two_dimensional_x", "raw_x_and_elementwise_squared_x"],
        "gates": "unchanged_racer_c4_core_defaults",
        "same_x_concept_shift_prediction_and_audit_identity": True,
        "oracle_definition": "exact_unclipped_p_target_x_over_p_source_x",
        "support_loss_oracle": "unavailable_due_to_absolute_continuity_failure",
        "concept_shift_oracle": "x_ratio_known_but_cannot_correct_changed_P_Y_given_X",
        "interpretation": "Descriptive repeated simulations; no portability or new coverage theorem. A passing audit cannot certify conditional invariance.",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--seed", type=int, default=73000)
    parser.add_argument("--n-calibration", type=int, default=400)
    parser.add_argument("--n-target", type=int, default=800)
    parser.add_argument("--alpha", type=float, default=.1)
    args = parser.parse_args()
    print(json.dumps(run(args.output, replicates=args.replicates, seed=args.seed,
        n_calibration=args.n_calibration, n_target=args.n_target, alpha=args.alpha), indent=2))


if __name__ == "__main__":
    main()
