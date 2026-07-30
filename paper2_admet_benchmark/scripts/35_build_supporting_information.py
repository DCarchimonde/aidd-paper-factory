from __future__ import annotations

"""Build the standalone Paper 2 Supporting Information tables.

The script reads only frozen confirmatory result tables. It performs no model fitting,
threshold selection, or statistical re-analysis. Model-specific means, SDs, and 95% t
intervals are reproduced from the already generated model-level CSV files.
"""

from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
ASSET_TABLE_DIR = PAPER_DIR / "results" / "manuscript_assets" / "tables"
LATEX_DIR = ROOT / "paper2_latex"
OUTPUT = LATEX_DIR / "generated_supplementary_tables.tex"


def read_required(path: Path, required: set[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen table is missing: {path}\n"
            "Regenerate the frozen comparison tables before building the SI."
        )
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise RuntimeError(f"Required frozen table is empty: {path}")
    if required:
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{path.name} missing columns: {sorted(missing)}")
    return frame


def tex(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return tex(value)
    if not math.isfinite(number):
        return "--"
    if abs(number) < 0.0005:
        number = 0.0
    return f"{number:.{digits}f}"


def ci_cell(row: pd.Series) -> str:
    return (
        f"{fmt(row['mean'])} "
        f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]"
    )


def longtable(
    rows: list[list[str]],
    headers: list[str],
    widths: str,
    caption: str,
    label: str,
    landscape: bool = True,
    font_command: str = r"\scriptsize",
) -> str:
    out: list[str] = []
    if landscape:
        out.append(r"\begin{landscape}")
    out.extend(
        [
            font_command,
            rf"\begin{{longtable}}{{{widths}}}",
            rf"\caption{{{caption}}}\label{{{label}}}\\",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            r"\endfirsthead",
            rf"\multicolumn{{{len(headers)}}}{{l}}{{\tablename\ \thetable\ (continued)}}\\",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            r"\endhead",
            r"\midrule",
            rf"\multicolumn{{{len(headers)}}}{{r}}{{Continued on next page}}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    )
    out.extend(" & ".join(row) + r" \\" for row in rows)
    out.append(r"\end{longtable}")
    if landscape:
        out.append(r"\end{landscape}")
    return "\n".join(out) + "\n"


def add_model_display(frame: pd.DataFrame) -> pd.DataFrame:
    if "model" not in frame.columns:
        raise KeyError("Model-specific table does not contain a model column")
    regime_columns = [
        column
        for column in [
            "imbalance_regime",
            "training_regime",
            "regime",
            "class_balance_regime",
        ]
        if column in frame.columns
    ]

    def build(row: pd.Series) -> str:
        pieces = [str(row["model"])]
        for column in regime_columns:
            value = row.get(column)
            if pd.notna(value) and str(value).strip() and str(value).lower() != "nan":
                pieces.append(str(value))
        return " / ".join(pieces)

    out = frame.copy()
    out["model_display"] = out.apply(build, axis=1)
    return out


def ci_pivot_table(
    frame: pd.DataFrame,
    metrics: list[str],
    metric_headers: list[str],
    caption: str,
    label: str,
    extra_index: list[str] | None = None,
) -> str:
    required = {
        "endpoint",
        "split_type",
        "model_display",
        "metric",
        "mean",
        "ci95_low",
        "ci95_high",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"CI table missing columns: {sorted(missing)}")
    extra_index = extra_index or []
    index_columns = ["endpoint", "split_type", "model_display", *extra_index]
    subset = frame[frame["metric"].isin(metrics)].copy()
    if subset.empty:
        raise RuntimeError(f"No rows found for metrics {metrics}")

    rows: list[list[str]] = []
    for keys, group in subset.groupby(index_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = [tex(value) for value in keys]
        for metric in metrics:
            metric_rows = group[group["metric"] == metric]
            if len(metric_rows) != 1:
                raise RuntimeError(
                    f"Expected one row for {dict(zip(index_columns, keys))}, "
                    f"metric={metric}; found {len(metric_rows)}"
                )
            row.append(ci_cell(metric_rows.iloc[0]))
        rows.append(row)

    headers = [
        "Endpoint",
        "Split",
        "Model / regime",
        *[tex(column.replace("_", " ").title()) for column in extra_index],
        *metric_headers,
    ]
    widths = "lll" + "l" * len(extra_index) + "p{3.0cm}" * len(metrics)
    return longtable(rows, headers, widths, caption, label)


def simple_table(
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    caption: str,
    label: str,
    digits: dict[str, int] | None = None,
) -> str:
    digits = digits or {}
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"Simple table missing columns: {sorted(missing)}")
    rows: list[list[str]] = []
    for _, record in frame[columns].iterrows():
        row: list[str] = []
        for column in columns:
            value = record[column]
            if column in digits:
                row.append(fmt(value, digits[column]))
            elif pd.api.types.is_number(value):
                row.append(fmt(value))
            else:
                row.append(tex(value))
        rows.append(row)
    widths = "l" * min(4, len(columns)) + "p{2.5cm}" * max(0, len(columns) - 4)
    return longtable(rows, headers, widths, caption, label)


def build_rq1_tables() -> str:
    aggregate = add_model_display(
        read_required(
            TABLE_DIR / "confirmatory_aggregate_long.csv",
            {
                "endpoint",
                "task_type",
                "split_type",
                "source_table",
                "model",
                "metric",
                "mean",
                "ci95_low",
                "ci95_high",
            },
        )
    )
    parts = [r"\section{Model-specific predictive performance and calibration}"]

    class_perf = aggregate[
        (aggregate["task_type"] == "classification")
        & (aggregate["source_table"] == "baseline_test")
    ]
    parts.append(
        ci_pivot_table(
            class_perf,
            ["roc_auc", "pr_auc", "balanced_accuracy"],
            ["ROC--AUC mean [95\% CI]", "PR--AUC mean [95\% CI]", "Balanced accuracy mean [95\% CI]"],
            "Model-specific classification performance across confirmatory seeds. Cluster splits use five seeds; random and scaffold splits use ten seeds.",
            "tab:s1_classification_performance",
        )
    )

    class_cal = aggregate[
        (aggregate["task_type"] == "classification")
        & (aggregate["source_table"] == "calibration_test")
    ]
    parts.append(
        ci_pivot_table(
            class_cal,
            ["brier_score", "negative_log_likelihood", "ece_probability", "ece_confidence"],
            ["Brier mean [95\% CI]", "NLL mean [95\% CI]", "Probability ECE mean [95\% CI]", "Confidence ECE mean [95\% CI]"],
            "Model-specific classification probability and confidence calibration across confirmatory seeds.",
            "tab:s2_classification_calibration",
        )
    )

    reg_perf = aggregate[
        (aggregate["task_type"] == "regression")
        & (aggregate["source_table"] == "baseline_test")
    ]
    parts.append(
        ci_pivot_table(
            reg_perf,
            ["rmse", "mae", "r2"],
            ["RMSE mean [95\% CI]", "MAE mean [95\% CI]", "$R^2$ mean [95\% CI]"],
            "Model-specific regression performance across confirmatory seeds.",
            "tab:s3_regression_performance",
        )
    )
    return "\n".join(parts)


def build_classification_conformal_tables() -> str:
    parts = [r"\section{Classification conformal prediction}"]
    headline = read_required(
        ASSET_TABLE_DIR / "table_rq3_rq4_classification_conformal.csv"
    ).sort_values(["endpoint", "split_type", "method"])
    parts.append(
        simple_table(
            headline,
            [
                "endpoint",
                "split_type",
                "method",
                "empirical_coverage",
                "positive_coverage",
                "negative_coverage",
                "class_conditional_coverage_gap",
                "mean_prediction_set_size",
                "empty_set_rate",
                "ambiguous_set_rate",
            ],
            [
                "Endpoint",
                "Split",
                "Method",
                "Coverage",
                "Positive coverage",
                "Negative coverage",
                "Class gap",
                "Mean set size",
                "Empty rate",
                "Ambiguous rate",
            ],
            "Descriptive classification conformal summaries at $\alpha=0.10$. Values average frozen model--regime and seed cells and are not inferential replicates.",
            "tab:s4_classification_conformal_headline",
            {column: 3 for column in [
                "empirical_coverage",
                "positive_coverage",
                "negative_coverage",
                "class_conditional_coverage_gap",
                "mean_prediction_set_size",
                "empty_set_rate",
                "ambiguous_set_rate",
            ]},
        )
    )

    paired = add_model_display(
        read_required(
            TABLE_DIR / "paired_classification_conformal_by_model_alpha01.csv",
            {
                "endpoint",
                "split_type",
                "model",
                "comparison",
                "metric",
                "mean",
                "ci95_low",
                "ci95_high",
            },
        )
    )
    parts.append(
        ci_pivot_table(
            paired,
            [
                "positive_coverage",
                "class_conditional_coverage_gap",
                "mean_prediction_set_size",
                "ambiguous_set_rate",
            ],
            [
                "$\Delta$ positive coverage [95\% CI]",
                "$\Delta$ class gap [95\% CI]",
                "$\Delta$ mean set size [95\% CI]",
                "$\Delta$ ambiguity [95\% CI]",
            ],
            "Model-specific paired classification conformal effects. Deltas are treatment minus marginal conformal prediction on matched endpoint, split, seed, model, and regime cells.",
            "tab:s5_classification_paired",
            ["comparison"],
        )
    )
    return "\n".join(parts)


def build_regression_conformal_tables() -> str:
    parts = [r"\section{Regression conformal prediction}"]
    by_model = add_model_display(
        read_required(
            TABLE_DIR / "regression_conformal_by_model_alpha01.csv",
            {
                "endpoint",
                "split_type",
                "model",
                "method",
                "metric",
                "mean",
                "ci95_low",
                "ci95_high",
            },
        )
    )
    parts.append(
        ci_pivot_table(
            by_model,
            [
                "empirical_coverage",
                "absolute_coverage_gap",
                "mean_interval_width",
                "width_error_spearman",
            ],
            [
                "Coverage mean [95\% CI]",
                "Absolute gap mean [95\% CI]",
                "Mean width [95\% CI]",
                "Width--error $\rho$ [95\% CI]",
            ],
            "Model-specific regression conformal summaries at $\alpha=0.10$. Width--error association is undefined for constant-width marginal intervals.",
            "tab:s6_regression_conformal",
            ["method"],
        )
    )

    paired = add_model_display(
        read_required(
            TABLE_DIR / "paired_regression_conformal_by_model_alpha01.csv",
            {
                "endpoint",
                "split_type",
                "model",
                "comparison",
                "metric",
                "mean",
                "ci95_low",
                "ci95_high",
            },
        )
    )
    parts.append(
        ci_pivot_table(
            paired,
            ["empirical_coverage", "absolute_coverage_gap", "mean_interval_width"],
            [
                "$\Delta$ coverage [95\% CI]",
                "$\Delta$ absolute gap [95\% CI]",
                "$\Delta$ mean width [95\% CI]",
            ],
            "Model-specific paired regression conformal effects. Deltas are treatment minus marginal intervals on matched endpoint, split, seed, and model cells.",
            "tab:s7_regression_paired",
            ["comparison"],
        )
    )
    return "\n".join(parts)


def build_domain_and_selective_tables() -> str:
    parts = [r"\section{Applicability-domain and selective-prediction sensitivity analyses}"]
    ad = read_required(ASSET_TABLE_DIR / "table_rq2_ad_continuous.csv").sort_values(
        ["endpoint", "split_type"]
    )
    parts.append(
        simple_table(
            ad,
            [
                "endpoint",
                "split_type",
                "mean_risk_similarity_spearman",
                "mean_miscoverage_similarity_spearman",
                "mean_risk_ood_auc",
                "mean_miscoverage_ood_auc",
                "models_risk_spearman_ci_strictly_negative",
                "models_miscoverage_spearman_ci_strictly_negative",
            ],
            [
                "Endpoint",
                "Split",
                "Risk--similarity $\rho$",
                "Miscoverage--similarity $\rho$",
                "Risk OOD AUC",
                "Miscoverage OOD AUC",
                "Models with risk CI $<0$",
                "Models with miscoverage CI $<0$",
            ],
            "Continuous applicability-domain diagnostics. Blank risk OOD AUC cells for regression reflect the prespecified output definition rather than missing runs.",
            "tab:s8_ad_continuous",
            {column: 3 for column in [
                "mean_risk_similarity_spearman",
                "mean_miscoverage_similarity_spearman",
                "mean_risk_ood_auc",
                "mean_miscoverage_ood_auc",
            ]},
        )
    )

    quartiles = read_required(
        ASSET_TABLE_DIR / "table_rq2_ad_similarity_quartiles.csv"
    ).sort_values(["endpoint", "split_type", "similarity_quartile"])
    parts.append(
        simple_table(
            quartiles,
            [
                "endpoint",
                "split_type",
                "similarity_quartile",
                "model_seed_cells",
                "mean_similarity",
                "mean_risk",
                "mean_miscoverage",
                "mean_unseen_scaffold_rate",
            ],
            [
                "Endpoint",
                "Split",
                "Quartile",
                "Model--seed cells",
                "Mean similarity",
                "Mean risk",
                "Mean miscoverage",
                "Unseen-scaffold rate",
            ],
            "Similarity-quartile sensitivity analysis. Quartile 1 is the lowest-similarity group and quartile 4 the highest-similarity group.",
            "tab:s9_ad_quartiles",
            {column: 3 for column in [
                "mean_similarity",
                "mean_risk",
                "mean_miscoverage",
                "mean_unseen_scaffold_rate",
            ]},
        )
    )

    selective = read_required(
        ASSET_TABLE_DIR / "table_rq2_rq3_selective_prediction.csv"
    ).sort_values(["endpoint", "split_type", "uncertainty_measure"])
    parts.append(
        simple_table(
            selective,
            [
                "endpoint",
                "task_type",
                "split_type",
                "uncertainty_measure",
                "primary_paurc_improvement_vs_random",
                "risk_improvement_at_05",
                "balanced_paurc_improvement_vs_random",
                "positive_retention_at_05",
                "negative_retention_at_05",
                "class_balance_shift_at_05",
            ],
            [
                "Endpoint",
                "Task",
                "Split",
                "Rejection score",
                "Primary pAURC improvement",
                "Risk improvement at 0.50",
                "Balanced pAURC improvement",
                "Positive retention at 0.50",
                "Negative retention at 0.50",
                "Class-balance shift at 0.50",
            ],
            "Selective-prediction summaries against matched random rejection. Blank class-specific cells are not applicable to regression.",
            "tab:s10_selective",
            {column: 3 for column in [
                "primary_paurc_improvement_vs_random",
                "risk_improvement_at_05",
                "balanced_paurc_improvement_vs_random",
                "positive_retention_at_05",
                "negative_retention_at_05",
                "class_balance_shift_at_05",
            ]},
        )
    )
    return "\n".join(parts)


def build_integrity_table() -> str:
    manifest = read_required(
        PAPER_DIR / "results" / "manuscript_assets" / "final_results_integrity_manifest.csv"
    ).sort_values("file")
    return "\n".join(
        [
            r"\section{Frozen-result integrity manifest}",
            simple_table(
                manifest,
                ["file", "rows", "expected_rows", "row_count_valid", "sha256"],
                ["File", "Rows", "Expected", "Valid", "SHA-256"],
                "Integrity manifest for the manuscript-ready frozen result tables.",
                "tab:s11_integrity",
            ),
        ]
    )


def main() -> None:
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    sections = [
        "% AUTO-GENERATED by paper2_admet_benchmark/scripts/35_build_supporting_information.py",
        "% Do not edit this file manually; edit the builder or frozen source tables.",
        build_rq1_tables(),
        build_classification_conformal_tables(),
        build_regression_conformal_tables(),
        build_domain_and_selective_tables(),
        build_integrity_table(),
    ]
    OUTPUT.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print("saved", OUTPUT)
    print("Supporting Information table generation complete.")


if __name__ == "__main__":
    main()
