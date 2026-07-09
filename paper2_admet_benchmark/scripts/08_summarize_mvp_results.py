from __future__ import annotations

"""Summarize Paper 2 MVP outputs into manuscript-ready tables.

Inputs:
- data/manifests/endpoint_registry.csv
- results/metrics/baseline_metrics_mvp.csv
- results/calibration/calibration_summary_mvp.csv
- results/conformal/conformal_summary_mvp.csv

Outputs:
- results/tables/table1_endpoint_summary_mvp.csv
- results/tables/table2_baseline_test_performance_mvp.csv
- results/tables/table3_calibration_test_summary_mvp.csv
- results/tables/table4_conformal_test_summary_mvp.csv
- results/tables/key_findings_mvp.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper2_admet_benchmark"
MANIFEST_DIR = PAPER_DIR / "data" / "manifests"
METRIC_DIR = PAPER_DIR / "results" / "metrics"
CALIBRATION_DIR = PAPER_DIR / "results" / "calibration"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"
TABLE_DIR = PAPER_DIR / "results" / "tables"

MVP_ENDPOINT_ORDER = ["bbbp", "clintox", "esol", "lipophilicity"]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def normalize_endpoint(value: str) -> str:
    return str(value).lower()


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "registry": pd.read_csv(require_file(MANIFEST_DIR / "endpoint_registry.csv")),
        "metrics": pd.read_csv(require_file(METRIC_DIR / "baseline_metrics_mvp.csv")),
        "calibration": pd.read_csv(require_file(CALIBRATION_DIR / "calibration_summary_mvp.csv")),
        "conformal": pd.read_csv(require_file(CONFORMAL_DIR / "conformal_summary_mvp.csv")),
    }


def table1_endpoint_summary(registry: pd.DataFrame) -> pd.DataFrame:
    registry = registry.copy()
    registry["endpoint_key"] = registry["endpoint"].map(normalize_endpoint)
    out = registry[registry["endpoint_key"].isin(MVP_ENDPOINT_ORDER)].copy()
    out["endpoint_key"] = pd.Categorical(out["endpoint_key"], categories=MVP_ENDPOINT_ORDER, ordered=True)
    out = out.sort_values("endpoint_key")
    cols = [
        "endpoint",
        "task_type",
        "admet_category",
        "n_raw",
        "n_clean",
        "duplicates_removed",
        "duplicate_conflict_count",
        "positive_ratio",
        "target_mean",
        "target_min",
        "target_max",
        "url",
    ]
    return out[[col for col in cols if col in out.columns]]


def table2_baseline_performance(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    test = metrics[metrics["role"] == "test"].copy()

    classification = test[test["task_type"] == "classification"].copy()
    regression = test[test["task_type"] == "regression"].copy()

    cls_cols = [
        "endpoint",
        "split_col",
        "model",
        "n_train",
        "n_eval",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "brier_score",
        "negative_log_likelihood",
    ]
    reg_cols = [
        "endpoint",
        "split_col",
        "model",
        "n_train",
        "n_eval",
        "rmse",
        "mae",
        "r2",
    ]
    out = pd.concat(
        [
            classification[[col for col in cls_cols if col in classification.columns]],
            regression[[col for col in reg_cols if col in regression.columns]],
        ],
        ignore_index=True,
        sort=False,
    )
    return out.sort_values(["endpoint", "split_col", "model"]).reset_index(drop=True)


def table3_calibration_summary(calibration: pd.DataFrame) -> pd.DataFrame:
    calibration = calibration.copy()
    test = calibration[calibration["role"] == "test"].copy()
    cols = [
        "endpoint",
        "task_type",
        "split_col",
        "model",
        "n",
        "brier_score",
        "negative_log_likelihood",
        "ece",
        "mce",
        "rmse",
        "mae",
        "bias",
        "median_abs_error",
    ]
    return test[[col for col in cols if col in test.columns]].sort_values(
        ["endpoint", "split_col", "model"]
    ).reset_index(drop=True)


def table4_conformal_summary(conformal: pd.DataFrame) -> pd.DataFrame:
    conformal = conformal.copy()
    cols = [
        "endpoint",
        "task_type",
        "split_col",
        "model",
        "alpha",
        "nominal_coverage",
        "empirical_coverage",
        "qhat",
        "n_calibration",
        "n_test",
        "mean_prediction_set_size",
        "empty_set_rate",
        "singleton_rate",
        "ambiguous_set_rate",
        "mean_interval_width",
        "mean_absolute_error",
        "median_absolute_error",
    ]
    return conformal[[col for col in cols if col in conformal.columns]].sort_values(
        ["endpoint", "split_col", "model", "alpha"]
    ).reset_index(drop=True)


def best_classification_models(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[(metrics["role"] == "test") & (metrics["task_type"] == "classification")].copy()
    if test.empty:
        return test
    idx = test.groupby(["endpoint", "split_col"])["roc_auc"].idxmax()
    return test.loc[idx, ["endpoint", "split_col", "model", "roc_auc", "pr_auc", "brier_score"]].sort_values(
        ["endpoint", "split_col"]
    )


def best_regression_models(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[(metrics["role"] == "test") & (metrics["task_type"] == "regression")].copy()
    if test.empty:
        return test
    idx = test.groupby(["endpoint", "split_col"])["rmse"].idxmin()
    return test.loc[idx, ["endpoint", "split_col", "model", "rmse", "mae", "r2"]].sort_values(
        ["endpoint", "split_col"]
    )


def scaffold_random_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["role"] == "test"].copy()
    rows = []
    for (endpoint, model), group in test.groupby(["endpoint", "model"]):
        random_row = group[group["split_col"] == "split_random_seed0"]
        scaffold_row = group[group["split_col"] == "split_scaffold"]
        if random_row.empty or scaffold_row.empty:
            continue
        task_type = str(group["task_type"].iloc[0])
        row = {"endpoint": endpoint, "task_type": task_type, "model": model}
        if task_type == "classification":
            row["random_roc_auc"] = float(random_row["roc_auc"].iloc[0])
            row["scaffold_roc_auc"] = float(scaffold_row["roc_auc"].iloc[0])
            row["scaffold_minus_random_roc_auc"] = row["scaffold_roc_auc"] - row["random_roc_auc"]
        else:
            row["random_rmse"] = float(random_row["rmse"].iloc[0])
            row["scaffold_rmse"] = float(scaffold_row["rmse"].iloc[0])
            row["scaffold_minus_random_rmse"] = row["scaffold_rmse"] - row["random_rmse"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["endpoint", "model"]).reset_index(drop=True)


def write_key_findings(metrics: pd.DataFrame, conformal: pd.DataFrame, out_path: Path) -> None:
    best_cls = best_classification_models(metrics)
    best_reg = best_regression_models(metrics)
    delta = scaffold_random_delta(metrics)

    lines = [
        "# Paper 2 MVP key findings",
        "",
        "This file is automatically generated by `08_summarize_mvp_results.py`.",
        "",
        "## Best test models by endpoint and split",
        "",
        "### Classification: best ROC-AUC",
        "",
    ]
    if not best_cls.empty:
        lines.append(best_cls.to_markdown(index=False))
    else:
        lines.append("No classification rows found.")

    lines.extend(["", "### Regression: best RMSE", ""])
    if not best_reg.empty:
        lines.append(best_reg.to_markdown(index=False))
    else:
        lines.append("No regression rows found.")

    lines.extend(["", "## Random vs scaffold shift summary", ""])
    if not delta.empty:
        lines.append(delta.to_markdown(index=False))
    else:
        lines.append("No paired random/scaffold rows found.")

    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- MVP results are pipeline-validation results, not final manuscript results.",
        "- Full experiments should include all planned random seeds and the final endpoint set.",
        "- Conformal results should be described as empirical coverage under the selected protocol, not as guaranteed coverage under chemical distribution shift.",
        "- Prediction files remain local and are not intended to be committed unless explicitly needed.",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()

    tables = {
        "table1_endpoint_summary_mvp.csv": table1_endpoint_summary(inputs["registry"]),
        "table2_baseline_test_performance_mvp.csv": table2_baseline_performance(inputs["metrics"]),
        "table3_calibration_test_summary_mvp.csv": table3_calibration_summary(inputs["calibration"]),
        "table4_conformal_test_summary_mvp.csv": table4_conformal_summary(inputs["conformal"]),
        "table5_random_vs_scaffold_delta_mvp.csv": scaffold_random_delta(inputs["metrics"]),
    }

    for filename, table in tables.items():
        path = TABLE_DIR / filename
        table.to_csv(path, index=False)
        print("saved", path)

    key_findings_path = TABLE_DIR / "key_findings_mvp.md"
    write_key_findings(inputs["metrics"], inputs["conformal"], key_findings_path)
    print("saved", key_findings_path)
    print("MVP result summarization complete.")


if __name__ == "__main__":
    main()
