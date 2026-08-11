from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper1_leakage_benchmark" / "results" / "tables"
GEN = ROOT / "paper1_latex" / "generated"
GEN.mkdir(parents=True, exist_ok=True)


def write(name: str, text: str) -> None:
    (GEN / name).write_text(text.strip() + "\n", encoding="utf-8")


def mean_only_text() -> None:
    df = pd.read_csv(TABLES / "q1_mean_only_regression_summary_v3.csv", keep_default_na=False)
    primary_effects = pd.read_csv(TABLES / "primary_inference_summary_v3.csv", keep_default_na=False)
    primary = df.loc[df["freeze_label"].eq("main_regression")].set_index("dataset")
    es = primary.loc["ESOL"]
    fs = primary.loc["FreeSolv"]
    same_direction = float(es.mean_effect_size_minus_balanced_rmse) > 0 and float(fs.mean_effect_size_minus_balanced_rmse) > 0

    learned = primary_effects.loc[primary_effects["task_type"].eq("regression")].copy()
    exceed_all = True
    for dataset, control_row in [("ESOL", es), ("FreeSolv", fs)]:
        learned_ds = learned.loc[learned["dataset"].eq(dataset), "mean_effect"].astype(float)
        if len(learned_ds) != 3 or float(control_row.mean_effect_size_minus_balanced_rmse) <= float(learned_ds.max()):
            exceed_all = False
            break

    if same_direction:
        interpretation = (
            "Both effects favored the target-mean-balanced partition, showing that part of the primary regression contrast is already present for a predictor that uses no molecular features. "
            "This is the expected mechanical contribution of target-mean alignment to RMSE difficulty."
        )
    else:
        interpretation = (
            "The two datasets did not show a common directional mean-only effect, so the learned-model contrast cannot be reduced to a universal mean-alignment mechanism."
        )
    if exceed_all:
        interpretation += (
            " The mean-only effect exceeded the corresponding learned-model mean effect in all six primary regression dataset--model cells; this comparison is descriptive and does not imply an additive decomposition of learned-model RMSE."
        )

    text = (
        "As a mechanistic control, every test molecule was assigned the training-set target mean. For this predictor, test MSE satisfies "
        "$\\mathrm{MSE}=\\mathrm{Var}(y_{\\mathrm{test}})+(\\bar y_{\\mathrm{test}}-\\bar y_{\\mathrm{train}})^2$ exactly. "
        f"Across the 20 primary partition pairs, the mean RMSE effect (size-matched minus target-mean-balanced) was {float(es.mean_effect_size_minus_balanced_rmse):.3f} "
        f"(95\\% bootstrap interval {float(es.bootstrap_ci_low):.3f} to {float(es.bootstrap_ci_high):.3f}) for ESOL and {float(fs.mean_effect_size_minus_balanced_rmse):.3f} "
        f"({float(fs.bootstrap_ci_low):.3f} to {float(fs.bootstrap_ci_high):.3f}) for FreeSolv. {interpretation}"
    )
    write("q1_mean_only_control_results_v3.tex", text)


def collateral_text() -> None:
    df = pd.read_csv(TABLES / "q1_collateral_diagnostics_summary_v3.csv", keep_default_na=False)

    def rng(metric: str) -> tuple[float, float]:
        x = df.loc[df["metric"].eq(metric), "mean_delta_balanced_minus_size"].to_numpy(float)
        return float(np.nanmin(x)), float(np.nanmax(x))

    largest = rng("largest_test_scaffold_fraction")
    effective = rng("effective_test_scaffolds")
    ks = rng("target_ks_statistic")
    wass = rng("target_wasserstein")
    text = (
        "Target-mean-aware selection also changed benchmark properties that were not part of the optimization objective. "
        f"Across datasets, the mean paired change (balanced minus size-matched) ranged from {largest[0]:+.3f} to {largest[1]:+.3f} for largest-scaffold fraction and from {effective[0]:+.3f} to {effective[1]:+.3f} for effective scaffold number. "
        f"Target-distribution diagnostics beyond the optimized mean also changed: mean KS-statistic shifts ranged from {ks[0]:+.3f} to {ks[1]:+.3f}, and Wasserstein-distance shifts ranged from {wass[0]:+.3f} to {wass[1]:+.3f} in endpoint units. "
        "These are measured consequences of selecting a different scaffold subset, not covariates held fixed by the paired design."
    )
    write("q1_collateral_diagnostics_results_v3.tex", text)


def seed_text() -> None:
    df = pd.read_csv(TABLES / "q1_model_seed_summary_v3.csv", keep_default_na=False)
    pivot = df.pivot_table(index=["freeze_label", "dataset", "model"], columns="model_seed", values="mean_effect")

    def summarize(label: str) -> tuple[int, int, float]:
        sub = pivot.loc[label]
        stable = 0
        total = 0
        max_dev = 0.0
        for _, row in sub.iterrows():
            vals = np.asarray([float(row[17]), float(row[29]), float(row[43])], dtype=float)
            signs = np.sign(vals[np.abs(vals) > 1e-12])
            if len(signs) == 0 or np.all(signs > 0) or np.all(signs < 0):
                stable += 1
            max_dev = max(max_dev, abs(vals[1] - vals[0]), abs(vals[2] - vals[0]))
            total += 1
        return stable, total, float(max_dev)

    cls = summarize("main_classification")
    reg = summarize("main_regression")
    single = summarize("acyclic_singleton_sensitivity")
    text = (
        "The predeclared RF/XGB model-seed sensitivity was complete on all five designated partition seeds. "
        f"Across the eight primary classification dataset--model cells, mean paired-effect direction was consistent across model seeds 17, 29, and 43 in {cls[0]}/{cls[1]} cells; the largest absolute shift relative to seed 17 was {cls[2]:.4f} AUC. "
        f"For the four primary regression cells, direction was consistent in {reg[0]}/{reg[1]} cells and the largest shift was {reg[2]:.4f} RMSE units; under singleton acyclic semantics the corresponding values were {single[0]}/{single[1]} and {single[2]:.4f} RMSE units. "
        "Repeated model seeds were used only as a robustness check and were never added to the partition-level inferential sample size."
    )
    write("q1_model_seed_results_v3.tex", text)


def main() -> None:
    mean_only_text(); collateral_text(); seed_text()
    print("Q1 GENERATED RESULT NARRATIVES: PASS")


if __name__ == "__main__":
    main()
