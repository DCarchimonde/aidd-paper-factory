from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
FROZEN_DIR = PAPER_DIR / "results" / "frozen_v3"
TABLE_DIR = PAPER_DIR / "results" / "tables"
MODEL_DIR = PAPER_DIR / "results" / "model_rerun_v3"
CLEAN_DIR = PAPER_DIR / "results" / "split_rebuild_v2"
GEN_DIR = ROOT / "paper1_latex" / "generated"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
GEN_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260810

SPECS = {
    "main_classification": {
        "task_type": "classification",
        "datasets": ("BACE", "BBBP", "ClinTox", "HIV"),
        "manifest": "split_manifest_v3_BACE-BBBP-ClinTox-HIV_single_group_20s_300c.csv",
    },
    "main_regression": {
        "task_type": "regression",
        "datasets": ("ESOL", "FreeSolv"),
        "manifest": "split_manifest_v3_ESOL-FreeSolv_single_group_20s_20000c.csv",
    },
    "acyclic_singleton_sensitivity": {
        "task_type": "regression",
        "datasets": ("ESOL", "FreeSolv"),
        "manifest": "split_manifest_v3_ESOL-FreeSolv_singleton_20s_5000c.csv",
    },
}

PRIMARY_PROTOCOLS = ("size_matched_scaffold", "target_balanced_scaffold")
PRIMARY_SEEDS = {"42", "123", "2024", "2026", "3407", "7", "19", "71", "101", "211", "307", "401", "503", "601", "701", "809", "907", "1009", "1201", "1429"}
SEED_SENSITIVITY_PARTITIONS = {"42", "123", "2024", "2026", "3407"}
SEED_SENSITIVITY_MODELS = {"RF", "XGB"}


def normalize_seed(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(BOOTSTRAP_REPS, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def signed_rank(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or np.allclose(x, 0.0, atol=1e-15, rtol=0.0):
        return 1.0
    return float(wilcoxon(x, zero_method="wilcox", correction=False, alternative="two-sided", method="auto").pvalue)


def load_manifest(label: str) -> pd.DataFrame:
    spec = SPECS[label]
    path = require(FROZEN_DIR / label / str(spec["manifest"]))
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    frame["partition_seed_key"] = frame["partition_seed"].map(normalize_seed)
    frame["target_numeric"] = pd.to_numeric(frame["target"], errors="raise")
    return frame


def scaffold_metrics(test: pd.DataFrame) -> dict[str, float]:
    counts = test["scaffold"].astype(str).value_counts(dropna=False)
    fractions = counts.to_numpy(dtype=float) / float(len(test))
    hhi = float(np.sum(fractions ** 2)) if len(fractions) else math.nan
    acyclic = test["scaffold"].astype(str).str.startswith("__ACYCLIC__").mean()
    return {
        "n_test_scaffolds": float(len(counts)),
        "largest_test_scaffold_fraction": float(fractions.max()) if len(fractions) else math.nan,
        "top5_test_scaffold_fraction": float(np.sort(fractions)[-5:].sum()) if len(fractions) else math.nan,
        "test_scaffold_hhi": hhi,
        "effective_test_scaffolds": float(1.0 / hhi) if hhi > 0 else math.nan,
        "acyclic_test_fraction": float(acyclic),
    }


def partition_diagnostics(group: pd.DataFrame) -> dict[str, float]:
    train = group.loc[group["assignment"].eq("train")].copy()
    test = group.loc[group["assignment"].eq("test")].copy()
    train_y = train["target_numeric"].to_numpy(dtype=float)
    test_y = test["target_numeric"].to_numpy(dtype=float)
    out = {
        "n_train": float(len(train)),
        "n_test": float(len(test)),
        "train_target_mean": float(np.mean(train_y)),
        "test_target_mean": float(np.mean(test_y)),
        "abs_target_mean_gap": float(abs(np.mean(train_y) - np.mean(test_y))),
        "target_ks_statistic": float(ks_2samp(train_y, test_y, alternative="two-sided").statistic),
        "target_wasserstein": float(wasserstein_distance(train_y, test_y)),
        "test_target_sd": float(np.std(test_y, ddof=0)),
    }
    out.update(scaffold_metrics(test))
    return out


def build_collateral_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for label in ("main_classification", "main_regression"):
        manifest = load_manifest(label)
        for dataset in SPECS[label]["datasets"]:
            ds = manifest.loc[manifest["dataset"].eq(dataset)].copy()
            for seed in sorted(PRIMARY_SEEDS, key=lambda x: int(x)):
                base: dict[str, object] = {
                    "freeze_label": label,
                    "task_type": SPECS[label]["task_type"],
                    "dataset": dataset,
                    "partition_seed": seed,
                }
                metrics_by_protocol: dict[str, dict[str, float]] = {}
                for protocol in PRIMARY_PROTOCOLS:
                    subset = ds.loc[
                        ds["protocol"].eq(protocol) & ds["partition_seed_key"].eq(seed)
                    ].copy()
                    if subset.empty:
                        raise AssertionError(f"Missing frozen partition: {label}/{dataset}/{protocol}/{seed}")
                    metrics_by_protocol[protocol] = partition_diagnostics(subset)
                if int(metrics_by_protocol[PRIMARY_PROTOCOLS[0]]["n_test"]) != int(metrics_by_protocol[PRIMARY_PROTOCOLS[1]]["n_test"]):
                    raise AssertionError(f"Exact-size pairing broken: {label}/{dataset}/{seed}")
                row = dict(base)
                for metric in metrics_by_protocol[PRIMARY_PROTOCOLS[0]]:
                    s = metrics_by_protocol[PRIMARY_PROTOCOLS[0]][metric]
                    b = metrics_by_protocol[PRIMARY_PROTOCOLS[1]][metric]
                    row[f"size_{metric}"] = s
                    row[f"balanced_{metric}"] = b
                    row[f"delta_balanced_minus_size_{metric}"] = b - s
                    if np.isfinite(s) and s > 0:
                        row[f"ratio_balanced_over_size_{metric}"] = b / s
                rows.append(row)
    paired = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    metrics = [
        "abs_target_mean_gap",
        "target_ks_statistic",
        "target_wasserstein",
        "largest_test_scaffold_fraction",
        "top5_test_scaffold_fraction",
        "test_scaffold_hhi",
        "effective_test_scaffolds",
        "acyclic_test_fraction",
    ]
    for (task_type, dataset), group in paired.groupby(["task_type", "dataset"], sort=False):
        for metric in metrics:
            s = group[f"size_{metric}"].to_numpy(float)
            b = group[f"balanced_{metric}"].to_numpy(float)
            delta = b - s
            ratio_mask = np.isfinite(s) & np.isfinite(b) & (s > 0)
            ratios = b[ratio_mask] / s[ratio_mask]
            summary_rows.append({
                "task_type": task_type,
                "dataset": dataset,
                "metric": metric,
                "n_pairs": int(len(group)),
                "mean_size": float(np.nanmean(s)),
                "mean_balanced": float(np.nanmean(b)),
                "mean_delta_balanced_minus_size": float(np.nanmean(delta)),
                "median_delta_balanced_minus_size": float(np.nanmedian(delta)),
                "mean_ratio_balanced_over_size": float(np.nanmean(ratios)) if len(ratios) else math.nan,
            })
    summary = pd.DataFrame(summary_rows)
    return paired, summary


def mean_only_record(group: pd.DataFrame) -> dict[str, float]:
    train = group.loc[group["assignment"].eq("train"), "target_numeric"].to_numpy(float)
    test = group.loc[group["assignment"].eq("test"), "target_numeric"].to_numpy(float)
    train_mean = float(np.mean(train))
    test_mean = float(np.mean(test))
    mse = float(np.mean((test - train_mean) ** 2))
    rmse = float(np.sqrt(mse))
    test_var = float(np.mean((test - test_mean) ** 2))
    gap_sq = float((test_mean - train_mean) ** 2)
    return {
        "intercept_rmse": rmse,
        "intercept_mse": mse,
        "test_variance": test_var,
        "target_mean_gap_squared": gap_sq,
        "decomposition_residual": float(mse - test_var - gap_sq),
        "abs_target_mean_gap": float(abs(test_mean - train_mean)),
    }


def build_mean_only_control() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for label in ("main_regression", "acyclic_singleton_sensitivity"):
        manifest = load_manifest(label)
        for dataset in SPECS[label]["datasets"]:
            ds = manifest.loc[manifest["dataset"].eq(dataset)].copy()
            for seed in sorted(PRIMARY_SEEDS, key=lambda x: int(x)):
                rec: dict[str, object] = {
                    "freeze_label": label,
                    "dataset": dataset,
                    "partition_seed": seed,
                }
                vals = {}
                for protocol in PRIMARY_PROTOCOLS:
                    subset = ds.loc[
                        ds["protocol"].eq(protocol) & ds["partition_seed_key"].eq(seed)
                    ].copy()
                    if subset.empty:
                        raise AssertionError(f"Missing regression manifest group: {label}/{dataset}/{protocol}/{seed}")
                    vals[protocol] = mean_only_record(subset)
                for metric in vals[PRIMARY_PROTOCOLS[0]]:
                    s = vals[PRIMARY_PROTOCOLS[0]][metric]
                    b = vals[PRIMARY_PROTOCOLS[1]][metric]
                    rec[f"size_{metric}"] = s
                    rec[f"balanced_{metric}"] = b
                    rec[f"effect_size_minus_balanced_{metric}"] = s - b
                rows.append(rec)
    paired = pd.DataFrame(rows)
    if float(paired[["size_decomposition_residual", "balanced_decomposition_residual"]].abs().to_numpy().max()) > 1e-10:
        raise AssertionError("Mean-only MSE decomposition identity failed")

    summary_rows: list[dict] = []
    counter = 0
    for (label, dataset), group in paired.groupby(["freeze_label", "dataset"], sort=False):
        effects = group["effect_size_minus_balanced_intercept_rmse"].to_numpy(float)
        lo, hi = bootstrap_ci(effects, BOOTSTRAP_SEED + counter)
        summary_rows.append({
            "freeze_label": label,
            "dataset": dataset,
            "n_pairs": int(len(effects)),
            "mean_size_intercept_rmse": float(group["size_intercept_rmse"].mean()),
            "mean_balanced_intercept_rmse": float(group["balanced_intercept_rmse"].mean()),
            "mean_effect_size_minus_balanced_rmse": float(np.mean(effects)),
            "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi,
            "wilcoxon_p_descriptive": signed_rank(effects),
            "mean_size_gap_squared": float(group["size_target_mean_gap_squared"].mean()),
            "mean_balanced_gap_squared": float(group["balanced_target_mean_gap_squared"].mean()),
            "max_abs_decomposition_residual": float(group[["size_decomposition_residual", "balanced_decomposition_residual"]].abs().to_numpy().max()),
        })
        counter += 1
    return paired, pd.DataFrame(summary_rows)


def build_cleaning_accounting() -> pd.DataFrame:
    summary = pd.read_csv(require(CLEAN_DIR / "cleaning_summary_v2.csv"), keep_default_na=False)
    decisions = pd.read_csv(require(CLEAN_DIR / "cleaning_group_decisions_v2.csv"), keep_default_na=False)
    rows: list[dict] = []
    for row in summary.itertuples(index=False):
        ds = str(row.dataset)
        d = decisions.loc[decisions["dataset"].eq(ds)].copy()
        duplicate = d.loc[d["n_source_rows"].astype(int) > 1].copy()
        duplicate_rows_beyond_first = int((duplicate["n_source_rows"].astype(int) - 1).sum())
        consistent = d.loc[d["decision"].isin([
            "collapse_consistent_classification_duplicates",
            "aggregate_regression_duplicates_by_mean",
        ])].copy()
        consistent_duplicate_rows_beyond_first = int((consistent["n_source_rows"].astype(int) - 1).sum()) if len(consistent) else 0
        rows.append({
            "dataset": ds,
            "task_type": str(row.task_type),
            "raw_rows": int(row.raw_n_rows),
            "invalid_or_missing_rows": int(row.invalid_or_missing_rows),
            "valid_unique_canonical_groups": int(row.valid_unique_canonical_groups),
            "duplicate_groups_total": int(len(duplicate)),
            "duplicate_rows_beyond_first": duplicate_rows_beyond_first,
            "consistent_duplicate_groups_collapsed_or_aggregated": int(len(consistent)),
            "consistent_duplicate_rows_beyond_first": consistent_duplicate_rows_beyond_first,
            "conflicting_groups_excluded": int(row.excluded_conflicting_classification_groups),
            "conflicting_rows_excluded": int(row.excluded_conflicting_classification_rows),
            "final_clean_unique_molecules": int(row.clean_v2_n_rows),
        })
    return pd.DataFrame(rows)


def build_model_seed_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = pd.read_csv(require(MODEL_DIR / "model_job_index_v3.csv"), keep_default_na=False, low_memory=False)
    jobs["partition_seed_key"] = jobs["partition_seed"].map(normalize_seed)
    jobs["model_seed_int"] = pd.to_numeric(jobs["model_seed"], errors="raise").astype(int)
    keep_labels = set(SPECS)
    rows: list[dict] = []
    for label in keep_labels:
        task_type = SPECS[label]["task_type"]
        metric = "roc_auc" if task_type == "classification" else "rmse"
        subset = jobs.loc[
            jobs["freeze_label"].eq(label)
            & jobs["dataset"].isin(SPECS[label]["datasets"])
            & jobs["model"].isin(SEED_SENSITIVITY_MODELS)
            & jobs["partition_seed_key"].isin(SEED_SENSITIVITY_PARTITIONS)
            & jobs["protocol"].isin(PRIMARY_PROTOCOLS)
            & jobs["model_seed_int"].isin([17, 29, 43])
        ].copy()
        for (dataset, model, seed, model_seed), group in subset.groupby(
            ["dataset", "model", "partition_seed_key", "model_seed_int"], sort=True
        ):
            if set(group["protocol"].tolist()) != set(PRIMARY_PROTOCOLS):
                continue
            size = group.loc[group["protocol"].eq(PRIMARY_PROTOCOLS[0])].iloc[0]
            bal = group.loc[group["protocol"].eq(PRIMARY_PROTOCOLS[1])].iloc[0]
            sv = float(size[metric]); bv = float(bal[metric])
            effect = bv - sv if task_type == "classification" else sv - bv
            rows.append({
                "freeze_label": label,
                "task_type": task_type,
                "dataset": dataset,
                "model": model,
                "partition_seed": seed,
                "model_seed": int(model_seed),
                "metric": metric,
                "size_value": sv,
                "balanced_value": bv,
                "effect_positive_is_balanced_better": effect,
            })
    paired = pd.DataFrame(rows)
    expected_cells = sum(len(SPECS[label]["datasets"]) * 2 for label in SPECS)
    expected_rows = expected_cells * 5 * 3
    if len(paired) != expected_rows:
        raise AssertionError(
            f"Model-seed sensitivity incomplete: expected {expected_rows} paired model-seed effects, found {len(paired)}. "
            "Run the Q1 final runner, which executes the predeclared sensitivity jobs."
        )
    summary = (
        paired.groupby(["freeze_label", "task_type", "dataset", "model", "model_seed"], as_index=False)
        .agg(
            n_partition_pairs=("partition_seed", "size"),
            mean_effect=("effect_positive_is_balanced_better", "mean"),
            sd_effect=("effect_positive_is_balanced_better", "std"),
        )
    )
    return paired, summary


def latex_escape(text: object) -> str:
    out = str(text)
    for a, b in [("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#")]:
        out = out.replace(a, b)
    return out


def write_generated_tex(
    mean_summary: pd.DataFrame,
    collateral_summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    cleaning: pd.DataFrame,
) -> None:
    main_reg = mean_summary.loc[mean_summary["freeze_label"].eq("main_regression")].set_index("dataset")
    es = main_reg.loc["ESOL"]
    fs = main_reg.loc["FreeSolv"]
    if es["mean_effect_size_minus_balanced_rmse"] > 0 and fs["mean_effect_size_minus_balanced_rmse"] > 0:
        control_interpretation = (
            "Thus, part of the apparent regression advantage is already present for a predictor that uses no molecular features, "
            "consistent with a mechanical contribution from target-mean alignment to RMSE difficulty."
        )
    else:
        control_interpretation = (
            "The mean-only control did not improve in the same direction for both regression datasets, so the learned-model effects cannot be reduced to a universal mean-alignment mechanism."
        )
    controls = (
        "As a mechanistic control, we evaluated a mean-only predictor that assigns every test molecule the training-set target mean. "
        "For such a predictor, test MSE satisfies $\\mathrm{MSE}=\\mathrm{Var}(y_{\\mathrm{test}})+(\\bar y_{\\mathrm{test}}-\\bar y_{\\mathrm{train}})^2$ exactly. "
        f"Across the 20 primary partition pairs, the mean RMSE effect (size-matched minus target-balanced) was {es['mean_effect_size_minus_balanced_rmse']:.3f} "
        f"(95\\% bootstrap interval {es['bootstrap_ci_low']:.3f} to {es['bootstrap_ci_high']:.3f}) for ESOL and {fs['mean_effect_size_minus_balanced_rmse']:.3f} "
        f"({fs['bootstrap_ci_low']:.3f} to {fs['bootstrap_ci_high']:.3f}) for FreeSolv. {control_interpretation}"
    )
    (GEN_DIR / "q1_mean_only_control_results_v3.tex").write_text(controls + "\n", encoding="utf-8")

    primary_collateral = collateral_summary.copy()
    lines = []
    for metric, label in [
        ("largest_test_scaffold_fraction", "largest-scaffold fraction"),
        ("effective_test_scaffolds", "effective scaffold number"),
        ("acyclic_test_fraction", "test-set acyclic fraction"),
    ]:
        sub = primary_collateral.loc[primary_collateral["metric"].eq(metric)]
        deltas = sub["mean_delta_balanced_minus_size"].to_numpy(float)
        lines.append(f"{label} mean paired changes ranged from {np.nanmin(deltas):+.3f} to {np.nanmax(deltas):+.3f} across datasets")
    collateral_text = (
        "Target-aware selection also changed aspects of test-set composition that were not part of the optimization objective. "
        + "; ".join(lines)
        + ". These collateral changes are therefore treated as measured consequences of the selection rule, not as controlled covariates."
    )
    (GEN_DIR / "q1_collateral_diagnostics_results_v3.tex").write_text(collateral_text + "\n", encoding="utf-8")

    pivot = seed_summary.pivot_table(index=["freeze_label", "dataset", "model"], columns="model_seed", values="mean_effect")
    stable = 0
    max_dev = 0.0
    total = 0
    for _, row in pivot.iterrows():
        if not {17, 29, 43}.issubset(set(row.dropna().index.astype(int))):
            continue
        vals = np.asarray([float(row[17]), float(row[29]), float(row[43])])
        nonzero = vals[np.abs(vals) > 1e-12]
        if len(nonzero) and (np.all(nonzero > 0) or np.all(nonzero < 0)):
            stable += 1
        elif len(nonzero) == 0:
            stable += 1
        max_dev = max(max_dev, float(max(abs(vals[1] - vals[0]), abs(vals[2] - vals[0]))))
        total += 1
    seed_text = (
        f"The predeclared RF/XGB model-seed sensitivity was complete for all {total} dataset--model--protocol families on the five designated partition seeds. "
        f"Mean paired-effect direction was consistent across model seeds 17, 29, and 43 in {stable}/{total} cells; the largest absolute shift of a cell mean relative to model seed 17 was {max_dev:.4f} in the metric's native effect units. "
        "Repeated model seeds were used only as a robustness check and were not added to the partition-level inferential sample size."
    )
    (GEN_DIR / "q1_model_seed_results_v3.tex").write_text(seed_text + "\n", encoding="utf-8")

    table_lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Closed accounting of the audited raw-to-clean molecular-data construction. Duplicate rows beyond the first are reported separately from conflicting-label exclusions so that each raw-to-clean transition is auditable.}",
        "\\label{tab:si-cleaning-accounting}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Dataset & Task & Raw & Invalid/missing & Duplicate groups & Duplicate rows beyond first & Conflict groups & Conflict rows & Final unique \\\\",
        "\\midrule",
    ]
    for r in cleaning.itertuples(index=False):
        table_lines.append(
            f"{latex_escape(r.dataset)} & {latex_escape(r.task_type)} & {r.raw_rows} & {r.invalid_or_missing_rows} & "
            f"{r.duplicate_groups_total} & {r.duplicate_rows_beyond_first} & {r.conflicting_groups_excluded} & {r.conflicting_rows_excluded} & {r.final_clean_unique_molecules} \\\\" 
        )
    table_lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    (GEN_DIR / "q1_cleaning_accounting_table_v3.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    seed_table = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Predeclared stochastic-model seed sensitivity. Values are mean paired effects over the five designated partition seeds; positive values favor target balancing.}",
        "\\label{tab:si-model-seed}",
        "\\scriptsize",
        "\\begin{tabular}{llllrrr}",
        "\\toprule",
        "Analysis & Dataset & Model & Metric & Seed 17 & Seed 29 & Seed 43 \\\\",
        "\\midrule",
    ]
    p = seed_summary.pivot_table(index=["freeze_label", "dataset", "model"], columns="model_seed", values="mean_effect").reset_index()
    for r in p.itertuples(index=False):
        task = "AUC" if "classification" in str(r.freeze_label) else "RMSE"
        seed_table.append(
            f"{latex_escape(r.freeze_label)} & {latex_escape(r.dataset)} & {latex_escape(r.model)} & {task} & {getattr(r, '_4'):.4f} & {getattr(r, '_5'):.4f} & {getattr(r, '_6'):.4f} \\\\" 
        )
    seed_table += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    # Pandas namedtuple field names for integer columns are implementation-specific; rebuild safely if necessary.
    seed_table = seed_table[:7]
    for _, rr in p.iterrows():
        task = "AUC" if "classification" in str(rr["freeze_label"]) else "RMSE"
        seed_table.append(
            f"{latex_escape(rr['freeze_label'])} & {latex_escape(rr['dataset'])} & {latex_escape(rr['model'])} & {task} & {float(rr[17]):.4f} & {float(rr[29]):.4f} & {float(rr[43]):.4f} \\\\" 
        )
    seed_table += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (GEN_DIR / "q1_model_seed_table_v3.tex").write_text("\n".join(seed_table) + "\n", encoding="utf-8")


def main() -> None:
    collateral_paired, collateral_summary = build_collateral_diagnostics()
    mean_paired, mean_summary = build_mean_only_control()
    cleaning = build_cleaning_accounting()
    seed_paired, seed_summary = build_model_seed_sensitivity()

    outputs = {
        "collateral_paired": TABLE_DIR / "q1_collateral_partition_diagnostics_v3.csv",
        "collateral_summary": TABLE_DIR / "q1_collateral_diagnostics_summary_v3.csv",
        "mean_only_paired": TABLE_DIR / "q1_mean_only_regression_control_v3.csv",
        "mean_only_summary": TABLE_DIR / "q1_mean_only_regression_summary_v3.csv",
        "cleaning": TABLE_DIR / "q1_cleaning_accounting_v3.csv",
        "seed_paired": TABLE_DIR / "q1_model_seed_partition_effects_v3.csv",
        "seed_summary": TABLE_DIR / "q1_model_seed_summary_v3.csv",
    }
    collateral_paired.to_csv(outputs["collateral_paired"], index=False)
    collateral_summary.to_csv(outputs["collateral_summary"], index=False)
    mean_paired.to_csv(outputs["mean_only_paired"], index=False)
    mean_summary.to_csv(outputs["mean_only_summary"], index=False)
    cleaning.to_csv(outputs["cleaning"], index=False)
    seed_paired.to_csv(outputs["seed_paired"], index=False)
    seed_summary.to_csv(outputs["seed_summary"], index=False)
    write_generated_tex(mean_summary, collateral_summary, seed_summary, cleaning)

    print("\nQ1 SCIENTIFIC CONTROLS")
    print("Mean-only regression control:")
    print(mean_summary.to_string(index=False))
    print("\nModel-seed sensitivity rows:", len(seed_paired))
    print("Collateral paired diagnostic rows:", len(collateral_paired))
    print("Cleaning accounting rows:", len(cleaning))
    print("\nSaved:")
    for p in outputs.values():
        print(p)
    print("\nQ1 SCIENTIFIC CONTROL AUDIT: PASS")


if __name__ == "__main__":
    main()
