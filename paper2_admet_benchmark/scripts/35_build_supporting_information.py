from __future__ import annotations

"""Build the standalone Paper 2 Supporting Information tables.

This builder reads only the small, versioned, integrity-checked manuscript asset
CSVs. It performs no model fitting, split construction, threshold estimation, or
statistical re-analysis. The resulting SI is therefore reproducible from a clean
clone of the repository and does not depend on local row-level prediction files.
"""

from pathlib import Path
import math
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
ASSET_DIR = PAPER_DIR / "results" / "manuscript_assets"
TABLE_DIR = ASSET_DIR / "tables"
LATEX_DIR = ROOT / "paper2_latex"
OUTPUT = LATEX_DIR / "generated_supplementary_tables.tex"


def read_required(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen manuscript table is missing: {path}\n"
            "Run script 33 to rebuild the integrity-checked manuscript package."
        )
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise RuntimeError(f"Required frozen manuscript table is empty: {path}")
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


def fmt_number(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return tex(value)
    if not math.isfinite(number):
        return "--"
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:.{digits}f}"


def longtable(
    rows: list[list[str]],
    headers: list[str],
    widths: str,
    caption: str,
    label: str,
    *,
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
    out.append(r"\normalsize")
    if landscape:
        out.append(r"\end{landscape}")
    return "\n".join(out) + "\n"


def table_from_frame(
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    widths: str,
    caption: str,
    label: str,
    *,
    digits: dict[str, int] | None = None,
    path_columns: Iterable[str] = (),
    landscape: bool = True,
    font_command: str = r"\scriptsize",
) -> str:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"{label} missing columns: {sorted(missing)}")
    digits = digits or {}
    path_columns = set(path_columns)
    rows: list[list[str]] = []
    for _, record in frame[columns].iterrows():
        row: list[str] = []
        for column in columns:
            value = record[column]
            if column in digits:
                row.append(fmt_number(value, digits[column]))
            elif column in path_columns:
                if pd.isna(value):
                    row.append("--")
                else:
                    row.append(r"\path{" + str(value) + "}")
            else:
                row.append(tex(value))
        rows.append(row)
    return longtable(
        rows,
        headers,
        widths,
        caption,
        label,
        landscape=landscape,
        font_command=font_command,
    )


def build_rq1_tables() -> str:
    parts = [r"\section{Predictive performance and calibration}"]

    classification = read_required("table_rq1_classification_performance.csv").sort_values(
        ["endpoint", "split_type"]
    )
    parts.append(
        table_from_frame(
            classification,
            ["endpoint", "split_type", "roc_auc", "pr_auc", "balanced_accuracy"],
            ["Endpoint", "Split", "ROC--AUC", "PR--AUC", "Balanced accuracy"],
            "llccc",
            "Descriptive classification performance means across the frozen model--regime and seed cells. Model families are not inferential replicates.",
            "tab:s1_classification_performance",
            digits={"roc_auc": 3, "pr_auc": 3, "balanced_accuracy": 3},
            landscape=False,
        )
    )

    calibration = read_required("table_rq1_classification_calibration.csv").sort_values(
        ["endpoint", "split_type"]
    )
    parts.append(
        table_from_frame(
            calibration,
            [
                "endpoint",
                "split_type",
                "brier_score",
                "negative_log_likelihood",
                "ece_probability",
                "ece_confidence",
            ],
            ["Endpoint", "Split", "Brier", "NLL", "Probability ECE", "Confidence ECE"],
            "llcccc",
            "Descriptive classification calibration means across the frozen model--regime and seed cells.",
            "tab:s2_classification_calibration",
            digits={
                "brier_score": 3,
                "negative_log_likelihood": 3,
                "ece_probability": 3,
                "ece_confidence": 3,
            },
            landscape=False,
        )
    )

    regression = read_required("table_rq1_regression_performance.csv").sort_values(
        ["endpoint", "split_type"]
    )
    parts.append(
        table_from_frame(
            regression,
            ["endpoint", "split_type", "rmse", "mae", "r2"],
            ["Endpoint", "Split", "RMSE", "MAE", "$R^2$"],
            "llccc",
            "Descriptive regression performance means across the frozen model and seed cells.",
            "tab:s3_regression_performance",
            digits={"rmse": 3, "mae": 3, "r2": 3},
            landscape=False,
        )
    )
    return "\n".join(parts)


def build_classification_conformal_tables() -> str:
    parts = [r"\section{Classification conformal prediction}"]
    headline = read_required("table_rq3_rq4_classification_conformal.csv").sort_values(
        ["endpoint", "split_type", "method"]
    )
    parts.append(
        table_from_frame(
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
                "Positive cov.",
                "Negative cov.",
                "Class gap",
                "Mean set size",
                "Empty rate",
                "Ambiguous rate",
            ],
            "llp{4.0cm}ccccccc",
            r"Classification conformal summaries at $\alpha=0.10$. Values are descriptive averages across frozen model--regime and seed cells.",
            "tab:s4_classification_conformal",
            digits={
                "empirical_coverage": 3,
                "positive_coverage": 3,
                "negative_coverage": 3,
                "class_conditional_coverage_gap": 3,
                "mean_prediction_set_size": 3,
                "empty_set_rate": 3,
                "ambiguous_set_rate": 3,
            },
        )
    )

    paired = read_required("table_rq3_rq4_classification_paired_effects.csv").sort_values(
        ["endpoint", "split_type", "comparison"]
    )
    parts.append(
        table_from_frame(
            paired,
            [
                "endpoint",
                "split_type",
                "comparison",
                "mean_delta_positive_coverage",
                "models_positive_coverage_ci_strictly_positive",
                "mean_delta_class_conditional_coverage_gap",
                "models_class_conditional_coverage_gap_ci_strictly_negative",
                "mean_delta_mean_prediction_set_size",
                "mean_delta_ambiguous_set_rate",
            ],
            [
                "Endpoint",
                "Split",
                "Comparison",
                r"$\Delta$ positive cov.",
                "Model CIs $>0$",
                r"$\Delta$ class gap",
                "Model CIs $<0$",
                r"$\Delta$ mean set size",
                r"$\Delta$ ambiguity",
            ],
            "llp{4.2cm}cccccc",
            r"Paired classification conformal effects. Deltas are treatment minus marginal conformal prediction on matched cells. CI-count columns report how many of the eight frozen model--regime cross-seed 95\% intervals exclude zero in the stated direction.",
            "tab:s5_classification_paired",
            digits={
                "mean_delta_positive_coverage": 3,
                "models_positive_coverage_ci_strictly_positive": 0,
                "mean_delta_class_conditional_coverage_gap": 3,
                "models_class_conditional_coverage_gap_ci_strictly_negative": 0,
                "mean_delta_mean_prediction_set_size": 3,
                "mean_delta_ambiguous_set_rate": 3,
            },
        )
    )
    return "\n".join(parts)


def build_regression_conformal_tables() -> str:
    parts = [r"\section{Regression conformal prediction}"]
    headline = read_required("table_rq4_regression_conformal.csv").sort_values(
        ["endpoint", "split_type", "method"]
    )
    parts.append(
        table_from_frame(
            headline,
            [
                "endpoint",
                "split_type",
                "method",
                "empirical_coverage",
                "absolute_coverage_gap",
                "mean_interval_width",
                "interval_width_cv",
                "width_error_spearman",
            ],
            [
                "Endpoint",
                "Split",
                "Method",
                "Coverage",
                "Absolute gap",
                "Mean width",
                "Width CV",
                r"Width--error $\rho$",
            ],
            "llp{5.0cm}ccccc",
            r"Regression conformal summaries at $\alpha=0.10$. Width--error association is undefined for constant-width marginal intervals.",
            "tab:s6_regression_conformal",
            digits={
                "empirical_coverage": 3,
                "absolute_coverage_gap": 3,
                "mean_interval_width": 3,
                "interval_width_cv": 3,
                "width_error_spearman": 3,
            },
        )
    )

    paired = read_required("table_rq4_regression_paired_effects.csv").sort_values(
        ["endpoint", "split_type", "comparison"]
    )
    parts.append(
        table_from_frame(
            paired,
            [
                "endpoint",
                "split_type",
                "comparison",
                "mean_delta_empirical_coverage",
                "mean_delta_absolute_coverage_gap",
                "models_absolute_coverage_gap_ci_strictly_negative",
                "mean_delta_mean_interval_width",
                "models_mean_interval_width_ci_strictly_positive",
            ],
            [
                "Endpoint",
                "Split",
                "Comparison",
                r"$\Delta$ coverage",
                r"$\Delta$ absolute gap",
                "Model CIs gap $<0$",
                r"$\Delta$ mean width",
                "Model CIs width $>0$",
            ],
            "llp{4.2cm}ccccc",
            "Paired regression conformal effects. Deltas are treatment minus marginal intervals on matched endpoint, split, seed, and model cells; CI counts are out of four frozen regressors.",
            "tab:s7_regression_paired",
            digits={
                "mean_delta_empirical_coverage": 3,
                "mean_delta_absolute_coverage_gap": 3,
                "models_absolute_coverage_gap_ci_strictly_negative": 0,
                "mean_delta_mean_interval_width": 3,
                "models_mean_interval_width_ci_strictly_positive": 0,
            },
        )
    )
    return "\n".join(parts)


def build_domain_tables() -> str:
    parts = [r"\section{Applicability-domain sensitivity analyses}"]
    continuous = read_required("table_rq2_ad_continuous.csv").sort_values(
        ["endpoint", "split_type"]
    )
    parts.append(
        table_from_frame(
            continuous,
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
                r"Risk--similarity $\rho$",
                r"Miscoverage--similarity $\rho$",
                "Risk OOD AUC",
                "Miscoverage OOD AUC",
                "Model CIs risk $<0$",
                "Model CIs miscov. $<0$",
            ],
            "llcccccc",
            "Continuous applicability-domain diagnostics. Blank regression risk-OOD AUC cells reflect the prespecified output definition rather than missing runs.",
            "tab:s8_ad_continuous",
            digits={
                "mean_risk_similarity_spearman": 3,
                "mean_miscoverage_similarity_spearman": 3,
                "mean_risk_ood_auc": 3,
                "mean_miscoverage_ood_auc": 3,
                "models_risk_spearman_ci_strictly_negative": 0,
                "models_miscoverage_spearman_ci_strictly_negative": 0,
            },
        )
    )

    quartiles = read_required("table_rq2_ad_similarity_quartiles.csv").sort_values(
        ["endpoint", "split_type", "similarity_quartile"]
    )
    parts.append(
        table_from_frame(
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
            "llcccccc",
            "Similarity-quartile sensitivity analysis. Quartile 1 is the lowest-similarity group and quartile 4 the highest-similarity group.",
            "tab:s9_ad_quartiles",
            digits={
                "similarity_quartile": 0,
                "model_seed_cells": 0,
                "mean_similarity": 3,
                "mean_risk": 3,
                "mean_miscoverage": 3,
                "mean_unseen_scaffold_rate": 3,
            },
        )
    )

    thresholds = read_required("table_rq2_ad_threshold_sensitivity.csv").sort_values(
        ["endpoint", "split_type", "threshold"]
    )
    parts.append(
        table_from_frame(
            thresholds,
            [
                "endpoint",
                "split_type",
                "threshold",
                "valid_two_group_cells",
                "mean_low_fraction",
                "mean_delta_risk_low_minus_high",
                "mean_delta_miscoverage_low_minus_high",
                "models_risk_delta_ci_strictly_positive",
                "models_miscoverage_delta_ci_strictly_positive",
            ],
            [
                "Endpoint",
                "Split",
                "Threshold",
                "Valid cells",
                "Low-domain fraction",
                r"$\Delta$ risk low--high",
                r"$\Delta$ miscov. low--high",
                "Model CIs risk $>0$",
                "Model CIs miscov. $>0$",
            ],
            "llccccccc",
            "Threshold sensitivity for low-domain minus high-domain risk and miscoverage. Threshold partitions are sensitivity analyses, not universal applicability-domain definitions.",
            "tab:s10_ad_thresholds",
            digits={
                "threshold": 2,
                "valid_two_group_cells": 0,
                "mean_low_fraction": 3,
                "mean_delta_risk_low_minus_high": 3,
                "mean_delta_miscoverage_low_minus_high": 3,
                "models_risk_delta_ci_strictly_positive": 0,
                "models_miscoverage_delta_ci_strictly_positive": 0,
            },
            font_command=r"\tiny",
        )
    )
    return "\n".join(parts)


def build_selective_table() -> str:
    selective = read_required("table_rq2_rq3_selective_prediction.csv").sort_values(
        ["endpoint", "split_type", "uncertainty_measure"]
    )
    return "\n".join(
        [
            r"\section{Selective prediction}",
            table_from_frame(
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
                    "Primary pAURC impr.",
                    "Risk impr. at 0.50",
                    "Balanced pAURC impr.",
                    "Positive retention",
                    "Negative retention",
                    "Class-balance shift",
                ],
                "lllp{4.5cm}cccccc",
                "Selective-prediction summaries against matched random rejection. Blank class-specific cells are not applicable to regression.",
                "tab:s11_selective",
                digits={
                    "primary_paurc_improvement_vs_random": 3,
                    "risk_improvement_at_05": 3,
                    "balanced_paurc_improvement_vs_random": 3,
                    "positive_retention_at_05": 3,
                    "negative_retention_at_05": 3,
                    "class_balance_shift_at_05": 3,
                },
            ),
        ]
    )


def build_integrity_table() -> str:
    path = ASSET_DIR / "final_results_integrity_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required integrity manifest is missing: {path}")
    manifest = pd.read_csv(path).sort_values("file")
    return "\n".join(
        [
            r"\section{Frozen-result integrity manifest}",
            table_from_frame(
                manifest,
                ["file", "rows", "expected_rows", "row_count_valid", "sha256"],
                ["File", "Rows", "Expected", "Valid", "SHA-256"],
                "p{8.0cm}cccp{10.0cm}",
                "Integrity manifest for the manuscript-ready frozen result tables.",
                "tab:s12_integrity",
                digits={"rows": 0, "expected_rows": 0},
                path_columns={"file", "sha256"},
                font_command=r"\tiny",
            ),
        ]
    )


def main() -> None:
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    sections = [
        "% AUTO-GENERATED by paper2_admet_benchmark/scripts/35_build_supporting_information.py",
        "% Source boundary: versioned integrity-checked manuscript asset CSVs only.",
        build_rq1_tables(),
        build_classification_conformal_tables(),
        build_regression_conformal_tables(),
        build_domain_tables(),
        build_selective_table(),
        build_integrity_table(),
    ]
    OUTPUT.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print("saved", OUTPUT)
    print("Supporting Information table generation complete from frozen manuscript assets.")


if __name__ == "__main__":
    main()
