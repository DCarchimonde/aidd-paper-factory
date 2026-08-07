from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
OUT_DIR = PAPER_DIR / "results" / "model_rerun_v3"
TABLE_DIR = PAPER_DIR / "results" / "tables"
JOB_INDEX = OUT_DIR / "model_job_index_v3.csv"
COMPLETENESS = OUT_DIR / "model_completeness_detail_v3.csv"

BOOTSTRAP_REPS = 10000
BOOTSTRAP_BASE_SEED = 20260807

MAIN_LABELS = {
    "main_classification": "classification",
    "main_regression": "regression",
}

SUPPORTING_METRICS = {
    "classification": {
        "roc_auc": "higher",
        "average_precision": "higher",
        "f1": "higher",
        "accuracy": "higher",
        "balanced_accuracy": "higher",
        "brier_score": "lower",
    },
    "regression": {
        "rmse": "lower",
        "mae": "lower",
        "r2": "higher",
    },
}


def normalize_seed(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(BOOTSTRAP_REPS, len(x)))
    means = x[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def signed_rank(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if np.allclose(x, 0.0, rtol=0.0, atol=1e-15):
        return 0.0, 1.0
    result = wilcoxon(
        x,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy(dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        adjusted[idx] = min(running, 1.0)
    return pd.Series(adjusted, index=pvalues.index)


def effect_from_pair(task_type: str, size_value: float, balanced_value: float) -> float:
    if task_type == "classification":
        return float(balanced_value - size_value)
    return float(size_value - balanced_value)


def metric_effect(direction: str, size_value: float, balanced_value: float) -> float:
    if direction == "higher":
        return float(balanced_value - size_value)
    return float(size_value - balanced_value)


def inference_label(mean_effect: float, ci_low: float, ci_high: float, p_holm: float) -> str:
    if p_holm < 0.05 and ci_low > 0:
        return "target_balanced_better"
    if p_holm < 0.05 and ci_high < 0:
        return "target_balanced_worse"
    return "inconclusive"


def prepare_jobs() -> pd.DataFrame:
    if not JOB_INDEX.exists():
        raise FileNotFoundError(JOB_INDEX)
    if not COMPLETENESS.exists():
        raise FileNotFoundError(
            f"Run 11_audit_model_completeness_v3.py first: {COMPLETENESS}"
        )
    completeness = pd.read_csv(COMPLETENESS, keep_default_na=False)
    if not completeness["status"].eq("complete").all():
        raise AssertionError("Model completeness gate has not passed")

    jobs = pd.read_csv(JOB_INDEX, keep_default_na=False, low_memory=False)
    jobs = jobs.loc[jobs["run_type"].eq("production")].copy()
    jobs["partition_seed_key"] = jobs["partition_seed"].map(normalize_seed)
    return jobs


def main_primary(jobs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    partition_rows: list[dict] = []
    summary_rows: list[dict] = []
    cell_index = 0

    for freeze_label, task_type in MAIN_LABELS.items():
        metric = "roc_auc" if task_type == "classification" else "rmse"
        subset = jobs.loc[jobs["freeze_label"].eq(freeze_label)].copy()
        for (dataset, model), group in subset.groupby(["dataset", "model"], sort=True):
            size = group.loc[group["protocol"].eq("size_matched_scaffold")].copy()
            balanced = group.loc[group["protocol"].eq("target_balanced_scaffold")].copy()
            pair = size.merge(
                balanced,
                on=["dataset", "model", "partition_seed_key"],
                suffixes=("_size", "_balanced"),
                how="inner",
                validate="one_to_one",
            )
            if len(pair) != 20:
                raise AssertionError(
                    f"Expected 20 primary pairs for {dataset}/{model}; found {len(pair)}"
                )
            size_values = pd.to_numeric(pair[f"{metric}_size"], errors="raise").to_numpy(float)
            balanced_values = pd.to_numeric(pair[f"{metric}_balanced"], errors="raise").to_numpy(float)
            effects = np.asarray(
                [
                    effect_from_pair(task_type, s, b)
                    for s, b in zip(size_values, balanced_values)
                ],
                dtype=float,
            )
            for row, s, b, effect in zip(pair.itertuples(index=False), size_values, balanced_values, effects):
                partition_rows.append(
                    {
                        "analysis_role": "primary",
                        "freeze_label": freeze_label,
                        "dataset": dataset,
                        "task_type": task_type,
                        "model": model,
                        "partition_seed": row.partition_seed_key,
                        "size_partition_hash": row.partition_hash_size,
                        "balanced_partition_hash": row.partition_hash_balanced,
                        "primary_metric": metric,
                        "size_value": float(s),
                        "balanced_value": float(b),
                        "effect_positive_is_balanced_better": float(effect),
                    }
                )
            ci_low, ci_high = bootstrap_ci(effects, BOOTSTRAP_BASE_SEED + cell_index)
            statistic, p_raw = signed_rank(effects)
            summary_rows.append(
                {
                    "analysis_role": "primary",
                    "freeze_label": freeze_label,
                    "dataset": dataset,
                    "task_type": task_type,
                    "model": model,
                    "primary_metric": metric,
                    "n_unique_partition_pairs": int(len(effects)),
                    "mean_size_metric": float(np.mean(size_values)),
                    "mean_balanced_metric": float(np.mean(balanced_values)),
                    "mean_effect": float(np.mean(effects)),
                    "median_effect": float(np.median(effects)),
                    "sd_effect": float(np.std(effects, ddof=1)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "n_balanced_better": int(np.sum(effects > 0)),
                    "n_equal": int(np.sum(np.isclose(effects, 0.0, atol=1e-15, rtol=0.0))),
                    "n_balanced_worse": int(np.sum(effects < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_raw,
                }
            )
            cell_index += 1

    partition = pd.DataFrame(partition_rows)
    summary = pd.DataFrame(summary_rows)
    if len(summary) != 18:
        raise AssertionError(f"Expected 18 main dataset-model cells; found {len(summary)}")
    summary["p_holm"] = holm_adjust(summary["p_raw"])
    summary["inference_label"] = [
        inference_label(m, lo, hi, p)
        for m, lo, hi, p in zip(
            summary["mean_effect"],
            summary["bootstrap_ci_low"],
            summary["bootstrap_ci_high"],
            summary["p_holm"],
        )
    ]
    return partition, summary


def supporting_effects(jobs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for freeze_label, task_type in MAIN_LABELS.items():
        subset = jobs.loc[jobs["freeze_label"].eq(freeze_label)].copy()
        for (dataset, model), group in subset.groupby(["dataset", "model"], sort=True):
            size = group.loc[group["protocol"].eq("size_matched_scaffold")].copy()
            balanced = group.loc[group["protocol"].eq("target_balanced_scaffold")].copy()
            pair = size.merge(
                balanced,
                on=["dataset", "model", "partition_seed_key"],
                suffixes=("_size", "_balanced"),
                how="inner",
                validate="one_to_one",
            )
            for metric, direction in SUPPORTING_METRICS[task_type].items():
                s = pd.to_numeric(pair[f"{metric}_size"], errors="raise").to_numpy(float)
                b = pd.to_numeric(pair[f"{metric}_balanced"], errors="raise").to_numpy(float)
                effects = np.asarray(
                    [metric_effect(direction, sv, bv) for sv, bv in zip(s, b)],
                    dtype=float,
                )
                rows.append(
                    {
                        "freeze_label": freeze_label,
                        "dataset": dataset,
                        "task_type": task_type,
                        "model": model,
                        "metric": metric,
                        "metric_direction": direction,
                        "n_partition_pairs": int(len(effects)),
                        "mean_size": float(np.mean(s)),
                        "mean_balanced": float(np.mean(b)),
                        "mean_effect_positive_is_balanced_better": float(np.mean(effects)),
                        "median_effect_positive_is_balanced_better": float(np.median(effects)),
                    }
                )
    return pd.DataFrame(rows)


def protocol_reference_summary(jobs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for freeze_label, task_type in MAIN_LABELS.items():
        metric = "roc_auc" if task_type == "classification" else "rmse"
        subset = jobs.loc[jobs["freeze_label"].eq(freeze_label)].copy()
        for (dataset, model, protocol), group in subset.groupby(
            ["dataset", "model", "protocol"], sort=True
        ):
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(float)
            rows.append(
                {
                    "freeze_label": freeze_label,
                    "dataset": dataset,
                    "task_type": task_type,
                    "model": model,
                    "protocol": protocol,
                    "metric": metric,
                    "n_partitions": int(len(values)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def singleton_sensitivity(jobs: pd.DataFrame) -> pd.DataFrame:
    subset = jobs.loc[jobs["freeze_label"].eq("acyclic_singleton_sensitivity")].copy()
    rows: list[dict] = []
    seed_counter = 1000
    for (dataset, model), group in subset.groupby(["dataset", "model"], sort=True):
        size = group.loc[group["protocol"].eq("size_matched_scaffold")].copy()
        balanced = group.loc[group["protocol"].eq("target_balanced_scaffold")].copy()
        pair = size.merge(
            balanced,
            on=["dataset", "model", "partition_seed_key"],
            suffixes=("_size", "_balanced"),
            how="inner",
            validate="one_to_one",
        )
        if len(pair) != 20:
            raise AssertionError(f"Expected 20 singleton pairs for {dataset}/{model}")
        s = pd.to_numeric(pair["rmse_size"], errors="raise").to_numpy(float)
        b = pd.to_numeric(pair["rmse_balanced"], errors="raise").to_numpy(float)
        effects = s - b
        ci_low, ci_high = bootstrap_ci(effects, BOOTSTRAP_BASE_SEED + seed_counter)
        statistic, p_raw = signed_rank(effects)
        rows.append(
            {
                "analysis_role": "acyclic_singleton_sensitivity",
                "dataset": dataset,
                "model": model,
                "primary_metric": "rmse",
                "n_unique_partition_pairs": 20,
                "mean_size_rmse": float(np.mean(s)),
                "mean_balanced_rmse": float(np.mean(b)),
                "mean_effect_positive_is_balanced_better": float(np.mean(effects)),
                "median_effect_positive_is_balanced_better": float(np.median(effects)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "wilcoxon_statistic_descriptive": statistic,
                "p_raw_descriptive": p_raw,
            }
        )
        seed_counter += 1
    return pd.DataFrame(rows)


def main() -> None:
    jobs = prepare_jobs()
    partition, primary = main_primary(jobs)
    supporting = supporting_effects(jobs)
    references = protocol_reference_summary(jobs)
    singleton = singleton_sensitivity(jobs)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "partition_effects": TABLE_DIR / "partition_level_primary_effects_v3.csv",
        "primary_summary": TABLE_DIR / "primary_inference_summary_v3.csv",
        "supporting": TABLE_DIR / "supporting_metric_effects_v3.csv",
        "references": TABLE_DIR / "protocol_reference_summary_v3.csv",
        "singleton": TABLE_DIR / "acyclic_singleton_sensitivity_v3.csv",
    }
    partition.to_csv(paths["partition_effects"], index=False)
    primary.to_csv(paths["primary_summary"], index=False)
    supporting.to_csv(paths["supporting"], index=False)
    references.to_csv(paths["references"], index=False)
    singleton.to_csv(paths["singleton"], index=False)

    print("\nPrimary 18-cell inference summary:")
    display_cols = [
        "dataset", "model", "primary_metric", "mean_size_metric",
        "mean_balanced_metric", "mean_effect", "bootstrap_ci_low",
        "bootstrap_ci_high", "p_raw", "p_holm", "inference_label",
    ]
    print(primary[display_cols].to_string(index=False))
    print("\nInference counts:")
    print(primary["inference_label"].value_counts().to_string())
    print("\nAcyclic singleton sensitivity:")
    print(singleton.to_string(index=False))
    print("\nSaved:")
    for path in paths.values():
        print(path)
    print("\nPARTITION-LEVEL ANALYSIS V3 COMPLETED")


if __name__ == "__main__":
    main()
