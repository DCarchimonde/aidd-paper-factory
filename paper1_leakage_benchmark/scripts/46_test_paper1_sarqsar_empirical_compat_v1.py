from __future__ import annotations

import pandas as pd

from paper1_sarqsar_finalizer_empirical_compat_v1 import audit_empirical_frames


def make_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_rows: list[dict] = []
    for dataset in ["BACE", "BBBP", "ClinTox", "HIV"]:
        for model in ["LR", "RF", "XGB"]:
            primary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "primary_metric": "roc_auc",
                    "mean_size_metric": 0.80,
                    "mean_balanced_metric": 0.79,
                    "mean_effect": -0.01,
                    "bootstrap_ci_low": -0.02,
                    "bootstrap_ci_high": 0.01,
                    "p_raw": 0.20,
                    "p_holm": 1.0,
                    "inference_label": "inconclusive",
                }
            )
    for dataset in ["ESOL", "FreeSolv"]:
        for model in ["Ridge", "RF", "XGB"]:
            primary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "primary_metric": "rmse",
                    "mean_size_metric": 2.0,
                    "mean_balanced_metric": 1.5,
                    "mean_effect": 0.5,
                    "bootstrap_ci_low": 0.1,
                    "bootstrap_ci_high": 0.8,
                    "p_raw": 0.001,
                    "p_holm": 0.01,
                    "inference_label": "target_balanced_better",
                }
            )
    primary = pd.DataFrame(primary_rows)

    singleton_rows: list[dict] = []
    for dataset, sign in [("ESOL", 1.0), ("FreeSolv", -1.0)]:
        for model in ["Ridge", "RF", "XGB"]:
            singleton_rows.append(
                {
                    "analysis_role": "acyclic_singleton_sensitivity",
                    "dataset": dataset,
                    "model": model,
                    "primary_metric": "rmse",
                    "n_unique_partition_pairs": 20,
                    "mean_size_rmse": 2.0,
                    "mean_balanced_rmse": 1.9,
                    "mean_effect_positive_is_balanced_better": 0.05 * sign,
                    "median_effect_positive_is_balanced_better": 0.02 * sign,
                    "bootstrap_ci_low": -0.1,
                    "bootstrap_ci_high": 0.2,
                    "wilcoxon_statistic_descriptive": 95.0,
                    "p_raw_descriptive": 0.7,
                }
            )
    singleton = pd.DataFrame(singleton_rows)

    mean_only = pd.DataFrame(
        [
            {
                "freeze_label": "main_regression",
                "dataset": "ESOL",
                "mean_effect_size_minus_balanced_rmse": 0.4,
                "bootstrap_ci_low": 0.2,
                "bootstrap_ci_high": 0.5,
            },
            {
                "freeze_label": "main_regression",
                "dataset": "FreeSolv",
                "mean_effect_size_minus_balanced_rmse": 1.8,
                "bootstrap_ci_low": 1.5,
                "bootstrap_ci_high": 2.0,
            },
            {
                "freeze_label": "acyclic_singleton_sensitivity",
                "dataset": "ESOL",
                "mean_effect_size_minus_balanced_rmse": 0.1,
                "bootstrap_ci_low": 0.0,
                "bootstrap_ci_high": 0.2,
            },
            {
                "freeze_label": "acyclic_singleton_sensitivity",
                "dataset": "FreeSolv",
                "mean_effect_size_minus_balanced_rmse": -0.05,
                "bootstrap_ci_low": -0.4,
                "bootstrap_ci_high": 0.3,
            },
        ]
    )
    return mean_only, primary, singleton


def main() -> None:
    audit_empirical_frames(*make_frames())
    print("EMPIRICAL COMPATIBILITY UNIT TEST: PASS")


if __name__ == "__main__":
    main()
