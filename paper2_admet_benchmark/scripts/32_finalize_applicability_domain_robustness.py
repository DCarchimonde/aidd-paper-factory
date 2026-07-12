from __future__ import annotations

"""Finalize applicability-domain robustness analyses for frozen confirmatory results.

This script uses the already generated marginal conformal test predictions at alpha=0.10
and the frozen ECFP/Tanimoto applicability details. It does not retrain models, alter
splits, or tune thresholds. It evaluates whether lower similarity to the training set is
associated with:

1. larger predictive risk (classification error or regression absolute error);
2. higher marginal-conformal miscoverage;
3. consistent risk differences across a prespecified threshold grid; and
4. monotonic changes across similarity quartiles.

Threshold sensitivity is descriptive. Sparse below/above-threshold groups are retained
with explicit valid-seed counts rather than silently discarded.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"
APPLICABILITY_DIR = PAPER_DIR / "results" / "applicability"
TABLE_DIR = PAPER_DIR / "results" / "tables"

SPLIT_PATTERN = re.compile(r"split_confirm_(random|scaffold|cluster)_seed(\d+)$")
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
QUARTILES = [1, 2, 3, 4]
MIN_GROUP_N = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize continuous and threshold-sensitivity applicability-domain analyses."
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(x) for x in THRESHOLDS),
        help="Comma-separated max-Tanimoto thresholds.",
    )
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds or any(not 0 < x < 1 for x in thresholds):
        raise ValueError(f"Invalid thresholds: {thresholds}")
    if thresholds != sorted(set(thresholds)):
        raise ValueError("Thresholds must be unique and sorted in ascending order.")
    return thresholds


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise RuntimeError(f"Required table is empty: {path}")
    return frame


def to_bool(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    if mapped.notna().all():
        return mapped.astype(bool)
    return series.astype(bool)


def add_split_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    extracted = out["split_col"].astype(str).str.extract(SPLIT_PATTERN)
    out["split_type"] = extracted[0]
    out["seed"] = pd.to_numeric(extracted[1], errors="coerce").astype("Int64")
    if out["split_type"].isna().any() or out["seed"].isna().any():
        bad = out.loc[
            out["split_type"].isna() | out["seed"].isna(), "split_col"
        ].drop_duplicates()
        raise ValueError(f"Unrecognized confirmatory split columns: {bad.tolist()}")
    return out


def finite_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return math.nan
    x_valid = x[mask]
    y_valid = y[mask]
    if len(np.unique(x_valid)) < 2 or len(np.unique(y_valid)) < 2:
        return math.nan
    result = spearmanr(x_valid, y_valid)
    return float(result.statistic)


def finite_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return math.nan
    x_valid = x[mask]
    y_valid = y[mask]
    if len(np.unique(x_valid)) < 2 or len(np.unique(y_valid)) < 2:
        return math.nan
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def binary_ood_auc(outcome: np.ndarray, similarity: np.ndarray) -> float:
    mask = np.isfinite(outcome) & np.isfinite(similarity)
    y = outcome[mask].astype(int)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y, 1 - similarity[mask]))


def prepare_merged(alpha: float) -> pd.DataFrame:
    conformal = read_required(
        CONFORMAL_DIR / "conformal_predictions_confirmatory_full.csv"
    )
    conformal = conformal[
        np.isclose(pd.to_numeric(conformal["alpha"], errors="coerce"), alpha)
    ].copy()
    if conformal.empty:
        raise RuntimeError(f"No marginal conformal predictions found for alpha={alpha}")

    applicability = read_required(
        APPLICABILITY_DIR / "applicability_details_confirmatory_full.csv"
    )
    applicability = applicability[applicability["role"] == "test"].copy()

    merge_cols = ["endpoint", "task_type", "split_col", "row_index"]
    ad_cols = merge_cols + [
        "max_tanimoto_to_train",
        "unseen_scaffold",
        "domain_bin",
    ]
    merged = conformal.merge(
        applicability[ad_cols],
        on=merge_cols,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise RuntimeError("Marginal conformal/applicability merge produced zero rows.")

    merged = add_split_metadata(merged)
    merged["covered"] = to_bool(merged["covered"])
    merged["miscoverage"] = (~merged["covered"]).astype(float)
    merged["max_tanimoto_to_train"] = pd.to_numeric(
        merged["max_tanimoto_to_train"], errors="coerce"
    )

    classification = merged["task_type"] == "classification"
    y_true_class = pd.to_numeric(merged.loc[classification, "y_true"], errors="coerce").astype(int)
    y_score_class = pd.to_numeric(merged.loc[classification, "y_output"], errors="coerce")
    merged.loc[classification, "risk_value"] = (
        (y_score_class >= 0.5).astype(int).to_numpy() != y_true_class.to_numpy()
    ).astype(float)
    merged.loc[classification, "risk_name"] = "classification_error_rate"

    regression = merged["task_type"] == "regression"
    y_true_reg = pd.to_numeric(merged.loc[regression, "y_true"], errors="coerce")
    y_pred_reg = pd.to_numeric(merged.loc[regression, "y_output"], errors="coerce")
    merged.loc[regression, "risk_value"] = np.abs(
        y_true_reg.to_numpy() - y_pred_reg.to_numpy()
    )
    merged.loc[regression, "risk_name"] = "absolute_error"

    group_count = merged[
        ["endpoint", "task_type", "split_col", "model"]
    ].drop_duplicates()
    if len(group_count) != 600:
        raise RuntimeError(f"Expected 600 endpoint/split/model groups, found {len(group_count)}")
    return merged


def continuous_rows(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["endpoint", "task_type", "split_col", "split_type", "seed", "model"]
    for keys, group in merged.groupby(group_cols, sort=True):
        endpoint, task_type, split_col, split_type, seed, model = keys
        similarity = group["max_tanimoto_to_train"].to_numpy(float)
        risk = group["risk_value"].to_numpy(float)
        miscoverage = group["miscoverage"].to_numpy(float)
        row: dict[str, object] = {
            "endpoint": endpoint,
            "task_type": task_type,
            "split_col": split_col,
            "split_type": split_type,
            "seed": int(seed),
            "model": model,
            "risk_name": str(group["risk_name"].iloc[0]),
            "n_test": int(len(group)),
            "mean_similarity": float(np.mean(similarity)),
            "median_similarity": float(np.median(similarity)),
            "unseen_scaffold_rate": float(to_bool(group["unseen_scaffold"]).mean()),
            "mean_risk": float(np.mean(risk)),
            "miscoverage_rate": float(np.mean(miscoverage)),
            "risk_similarity_spearman": finite_spearman(similarity, risk),
            "risk_similarity_pearson": finite_pearson(similarity, risk),
            "miscoverage_similarity_spearman": finite_spearman(similarity, miscoverage),
            "miscoverage_similarity_pearson": finite_pearson(similarity, miscoverage),
            "miscoverage_ood_auc": binary_ood_auc(miscoverage, similarity),
        }
        row["risk_ood_auc"] = (
            binary_ood_auc(risk, similarity)
            if task_type == "classification"
            else math.nan
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) != 600:
        raise RuntimeError(f"Expected 600 continuous rows, found {len(out)}")
    return out


def threshold_rows(merged: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["endpoint", "task_type", "split_col", "split_type", "seed", "model"]
    for keys, group in merged.groupby(group_cols, sort=True):
        endpoint, task_type, split_col, split_type, seed, model = keys
        similarity = group["max_tanimoto_to_train"].to_numpy(float)
        risk = group["risk_value"].to_numpy(float)
        miscoverage = group["miscoverage"].to_numpy(float)
        for threshold in thresholds:
            low = similarity < threshold
            high = similarity >= threshold
            n_low = int(low.sum())
            n_high = int(high.sum())
            valid = n_low > 0 and n_high > 0
            low_risk = float(np.mean(risk[low])) if n_low else math.nan
            high_risk = float(np.mean(risk[high])) if n_high else math.nan
            low_misc = float(np.mean(miscoverage[low])) if n_low else math.nan
            high_misc = float(np.mean(miscoverage[high])) if n_high else math.nan
            rows.append(
                {
                    "endpoint": endpoint,
                    "task_type": task_type,
                    "split_col": split_col,
                    "split_type": split_type,
                    "seed": int(seed),
                    "model": model,
                    "risk_name": str(group["risk_name"].iloc[0]),
                    "threshold": threshold,
                    "n_low_similarity": n_low,
                    "n_high_similarity": n_high,
                    "low_fraction": float(n_low / len(group)),
                    "valid_two_group_comparison": valid,
                    "small_low_group_warning": n_low < MIN_GROUP_N,
                    "small_high_group_warning": n_high < MIN_GROUP_N,
                    "low_similarity_risk": low_risk,
                    "high_similarity_risk": high_risk,
                    "delta_risk_low_minus_high": (
                        low_risk - high_risk if valid else math.nan
                    ),
                    "low_similarity_miscoverage": low_misc,
                    "high_similarity_miscoverage": high_misc,
                    "delta_miscoverage_low_minus_high": (
                        low_misc - high_misc if valid else math.nan
                    ),
                }
            )
    out = pd.DataFrame(rows)
    expected = 600 * len(thresholds)
    if len(out) != expected:
        raise RuntimeError(f"Expected {expected} threshold rows, found {len(out)}")
    return out


def quartile_rows(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["endpoint", "task_type", "split_col", "split_type", "seed", "model"]
    for keys, group in merged.groupby(group_cols, sort=True):
        endpoint, task_type, split_col, split_type, seed, model = keys
        ordered = group.sort_values("max_tanimoto_to_train", kind="mergesort").reset_index(drop=True)
        index_chunks = np.array_split(np.arange(len(ordered)), 4)
        for quartile, indices in zip(QUARTILES, index_chunks):
            subset = ordered.iloc[indices]
            rows.append(
                {
                    "endpoint": endpoint,
                    "task_type": task_type,
                    "split_col": split_col,
                    "split_type": split_type,
                    "seed": int(seed),
                    "model": model,
                    "risk_name": str(group["risk_name"].iloc[0]),
                    "similarity_quartile": quartile,
                    "quartile_definition": "Q1=lowest similarity; Q4=highest similarity",
                    "n": int(len(subset)),
                    "mean_similarity": float(subset["max_tanimoto_to_train"].mean()),
                    "median_similarity": float(subset["max_tanimoto_to_train"].median()),
                    "mean_risk": float(subset["risk_value"].mean()),
                    "miscoverage_rate": float(subset["miscoverage"].mean()),
                    "unseen_scaffold_rate": float(to_bool(subset["unseen_scaffold"]).mean()),
                }
            )
    out = pd.DataFrame(rows)
    if len(out) != 2400:
        raise RuntimeError(f"Expected 2400 quartile rows, found {len(out)}")
    return out


def summarize_values(values: pd.Series) -> dict[str, float | int]:
    x = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    n = int(len(x))
    if n == 0:
        return {
            "n_splits": 0,
            "mean": math.nan,
            "std": math.nan,
            "se": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else math.nan
    se = std / math.sqrt(n) if n > 1 else math.nan
    critical = float(t.ppf(0.975, df=n - 1)) if n > 1 else math.nan
    margin = critical * se if n > 1 else math.nan
    return {
        "n_splits": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": mean - margin if n > 1 else math.nan,
        "ci95_high": mean + margin if n > 1 else math.nan,
    }


def continuous_by_model(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "risk_similarity_spearman",
        "risk_similarity_pearson",
        "miscoverage_similarity_spearman",
        "miscoverage_similarity_pearson",
        "risk_ood_auc",
        "miscoverage_ood_auc",
    ]
    rows: list[dict[str, object]] = []
    group_cols = ["endpoint", "task_type", "split_type", "model", "risk_name"]
    for keys, group in runs.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        expected = 5 if base["split_type"] == "cluster" else 10
        row: dict[str, object] = {**base, "expected_splits": expected}
        for metric in metrics:
            summary = summarize_values(group[metric])
            for key, value in summary.items():
                row[f"{metric}_{key}"] = value
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) != 72:
        raise RuntimeError(f"Expected 72 continuous model rows, found {len(out)}")
    return out


def continuous_headline(runs: pd.DataFrame, by_model: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["endpoint", "task_type", "split_type", "risk_name"]
    for keys, group in runs.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        model_group = by_model[
            (by_model["endpoint"] == base["endpoint"])
            & (by_model["split_type"] == base["split_type"])
        ].copy()
        expected_models = 8 if base["task_type"] == "classification" else 4
        if model_group["model"].nunique() != expected_models:
            raise RuntimeError(f"Incomplete continuous headline group: {base}")
        rows.append(
            {
                **base,
                "models": expected_models,
                "seeds": int(group["seed"].nunique()),
                "model_seed_cells": int(len(group)),
                "mean_risk_similarity_spearman": float(
                    pd.to_numeric(group["risk_similarity_spearman"], errors="coerce").mean()
                ),
                "mean_miscoverage_similarity_spearman": float(
                    pd.to_numeric(group["miscoverage_similarity_spearman"], errors="coerce").mean()
                ),
                "mean_risk_ood_auc": float(
                    pd.to_numeric(group["risk_ood_auc"], errors="coerce").mean()
                ),
                "mean_miscoverage_ood_auc": float(
                    pd.to_numeric(group["miscoverage_ood_auc"], errors="coerce").mean()
                ),
                "models_risk_spearman_ci_strictly_negative": int(
                    (
                        pd.to_numeric(
                            model_group["risk_similarity_spearman_ci95_high"],
                            errors="coerce",
                        )
                        < 0
                    ).sum()
                ),
                "models_miscoverage_spearman_ci_strictly_negative": int(
                    (
                        pd.to_numeric(
                            model_group["miscoverage_similarity_spearman_ci95_high"],
                            errors="coerce",
                        )
                        < 0
                    ).sum()
                ),
                "models_risk_ood_auc_ci_strictly_above_half": int(
                    (
                        pd.to_numeric(
                            model_group["risk_ood_auc_ci95_low"], errors="coerce"
                        )
                        > 0.5
                    ).sum()
                ),
                "models_miscoverage_ood_auc_ci_strictly_above_half": int(
                    (
                        pd.to_numeric(
                            model_group["miscoverage_ood_auc_ci95_low"],
                            errors="coerce",
                        )
                        > 0.5
                    ).sum()
                ),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) != 12:
        raise RuntimeError(f"Expected 12 continuous headline rows, found {len(out)}")
    return out.sort_values(["endpoint", "split_type"]).reset_index(drop=True)


def threshold_by_model(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = [
        "endpoint",
        "task_type",
        "split_type",
        "model",
        "risk_name",
        "threshold",
    ]
    for keys, group in runs.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        expected = 5 if base["split_type"] == "cluster" else 10
        risk_summary = summarize_values(group["delta_risk_low_minus_high"])
        misc_summary = summarize_values(group["delta_miscoverage_low_minus_high"])
        rows.append(
            {
                **base,
                "expected_splits": expected,
                "valid_risk_splits": int(risk_summary["n_splits"]),
                "mean_delta_risk_low_minus_high": risk_summary["mean"],
                "risk_delta_ci95_low": risk_summary["ci95_low"],
                "risk_delta_ci95_high": risk_summary["ci95_high"],
                "valid_miscoverage_splits": int(misc_summary["n_splits"]),
                "mean_delta_miscoverage_low_minus_high": misc_summary["mean"],
                "miscoverage_delta_ci95_low": misc_summary["ci95_low"],
                "miscoverage_delta_ci95_high": misc_summary["ci95_high"],
            }
        )
    out = pd.DataFrame(rows)
    expected_rows = 72 * runs["threshold"].nunique()
    if len(out) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} threshold model rows, found {len(out)}")
    return out


def threshold_headline(runs: pd.DataFrame, by_model: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["endpoint", "task_type", "split_type", "risk_name", "threshold"]
    for keys, group in runs.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        model_group = by_model[
            (by_model["endpoint"] == base["endpoint"])
            & (by_model["split_type"] == base["split_type"])
            & np.isclose(by_model["threshold"].astype(float), float(base["threshold"]))
        ].copy()
        expected_models = 8 if base["task_type"] == "classification" else 4
        rows.append(
            {
                **base,
                "models": expected_models,
                "model_seed_cells": int(len(group)),
                "valid_two_group_cells": int(
                    to_bool(group["valid_two_group_comparison"]).sum()
                ),
                "mean_low_fraction": float(group["low_fraction"].mean()),
                "mean_delta_risk_low_minus_high": float(
                    pd.to_numeric(group["delta_risk_low_minus_high"], errors="coerce").mean()
                ),
                "mean_delta_miscoverage_low_minus_high": float(
                    pd.to_numeric(
                        group["delta_miscoverage_low_minus_high"], errors="coerce"
                    ).mean()
                ),
                "models_risk_delta_mean_positive": int(
                    (
                        pd.to_numeric(
                            model_group["mean_delta_risk_low_minus_high"],
                            errors="coerce",
                        )
                        > 0
                    ).sum()
                ),
                "models_risk_delta_ci_strictly_positive": int(
                    (
                        pd.to_numeric(
                            model_group["risk_delta_ci95_low"], errors="coerce"
                        )
                        > 0
                    ).sum()
                ),
                "models_miscoverage_delta_mean_positive": int(
                    (
                        pd.to_numeric(
                            model_group["mean_delta_miscoverage_low_minus_high"],
                            errors="coerce",
                        )
                        > 0
                    ).sum()
                ),
                "models_miscoverage_delta_ci_strictly_positive": int(
                    (
                        pd.to_numeric(
                            model_group["miscoverage_delta_ci95_low"], errors="coerce"
                        )
                        > 0
                    ).sum()
                ),
                "models_with_complete_seed_support": int(
                    (
                        pd.to_numeric(model_group["valid_risk_splits"], errors="coerce")
                        == pd.to_numeric(model_group["expected_splits"], errors="coerce")
                    ).sum()
                ),
            }
        )
    out = pd.DataFrame(rows)
    expected_rows = 12 * runs["threshold"].nunique()
    if len(out) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} threshold headline rows, found {len(out)}")
    return out.sort_values(["endpoint", "split_type", "threshold"]).reset_index(drop=True)


def quartile_headline(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = [
        "endpoint",
        "task_type",
        "split_type",
        "risk_name",
        "similarity_quartile",
    ]
    for keys, group in runs.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        rows.append(
            {
                **base,
                "model_seed_cells": int(len(group)),
                "mean_similarity": float(group["mean_similarity"].mean()),
                "mean_risk": float(group["mean_risk"].mean()),
                "mean_miscoverage": float(group["miscoverage_rate"].mean()),
                "mean_unseen_scaffold_rate": float(group["unseen_scaffold_rate"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) != 48:
        raise RuntimeError(f"Expected 48 quartile headline rows, found {len(out)}")
    return out.sort_values(["endpoint", "split_type", "similarity_quartile"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    thresholds = parse_thresholds(args.thresholds)
    alpha_tag = str(args.alpha).replace(".", "")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    merged = prepare_merged(args.alpha)
    continuous = continuous_rows(merged)
    thresholds_run = threshold_rows(merged, thresholds)
    quartiles = quartile_rows(merged)

    continuous_model = continuous_by_model(continuous)
    continuous_summary = continuous_headline(continuous, continuous_model)
    threshold_model = threshold_by_model(thresholds_run)
    threshold_summary = threshold_headline(thresholds_run, threshold_model)
    quartile_summary = quartile_headline(quartiles)

    outputs = {
        TABLE_DIR / f"ad_continuous_runs_alpha{alpha_tag}.csv": continuous,
        TABLE_DIR / f"ad_continuous_by_model_alpha{alpha_tag}.csv": continuous_model,
        TABLE_DIR / f"ad_continuous_headline_alpha{alpha_tag}.csv": continuous_summary,
        TABLE_DIR / f"ad_threshold_sensitivity_runs_alpha{alpha_tag}.csv": thresholds_run,
        TABLE_DIR / f"ad_threshold_sensitivity_by_model_alpha{alpha_tag}.csv": threshold_model,
        TABLE_DIR / f"ad_threshold_sensitivity_headline_alpha{alpha_tag}.csv": threshold_summary,
        TABLE_DIR / f"ad_similarity_quartiles_runs_alpha{alpha_tag}.csv": quartiles,
        TABLE_DIR / f"ad_similarity_quartiles_headline_alpha{alpha_tag}.csv": quartile_summary,
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)
        print("saved", path)

    print(
        "Applicability-domain robustness finalization complete. "
        f"continuous_runs={len(continuous)}, continuous_models={len(continuous_model)}, "
        f"continuous_headline={len(continuous_summary)}, threshold_runs={len(thresholds_run)}, "
        f"threshold_models={len(threshold_model)}, threshold_headline={len(threshold_summary)}, "
        f"quartile_runs={len(quartiles)}, quartile_headline={len(quartile_summary)}."
    )


if __name__ == "__main__":
    main()
