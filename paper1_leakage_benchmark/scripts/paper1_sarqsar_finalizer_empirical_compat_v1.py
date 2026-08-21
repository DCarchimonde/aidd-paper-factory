from __future__ import annotations

import re
from types import ModuleType

import numpy as np
import pandas as pd

import paper1_sarqsar_finalizer_core_v1 as core

CLASSIFICATION_DATASETS = {"BACE", "BBBP", "ClinTox", "HIV"}
REGRESSION_DATASETS = {"ESOL", "FreeSolv"}
EXPECTED_DATASET_COUNTS = {
    "BACE": 3,
    "BBBP": 3,
    "ClinTox": 3,
    "HIV": 3,
    "ESOL": 3,
    "FreeSolv": 3,
}


def canonical_token(value: object) -> str:
    """Normalize spaces, hyphens, punctuation, and underscores to one stable token."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def resolve_column(frame: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise KeyError(
        f"Could not resolve {purpose}. Tried {candidates}; available columns={list(frame.columns)}"
    )


def audit_empirical_frames(
    mean_only: pd.DataFrame,
    primary: pd.DataFrame,
    singleton: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_mean_only = {
        "freeze_label",
        "dataset",
        "mean_effect_size_minus_balanced_rmse",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
    }
    missing = required_mean_only.difference(mean_only.columns)
    if missing:
        raise KeyError(f"Mean-only table missing columns: {sorted(missing)}")
    if len(mean_only) != 4:
        raise AssertionError(f"Mean-only summary must contain 4 rows, found {len(mean_only)}")
    if len(primary) != 18 or len(singleton) != 6:
        raise AssertionError(
            f"Empirical summary sizes changed: primary={len(primary)}, singleton={len(singleton)}"
        )

    dataset_col = resolve_column(primary, ["dataset"], "primary dataset column")
    model_col = resolve_column(primary, ["model"], "primary model column")
    metric_col = resolve_column(primary, ["primary_metric", "metric"], "primary metric column")
    inference_col = resolve_column(
        primary,
        ["inference_label", "inference", "decision"],
        "primary inference column",
    )
    effect_col = resolve_column(
        primary,
        ["mean_effect", "mean_effect_positive_is_balanced_better"],
        "primary mean-effect column",
    )
    p_holm_col = resolve_column(
        primary,
        ["p_holm", "holm_p", "p_holm_adjusted"],
        "primary Holm-adjusted P-value column",
    )
    ci_low_col = resolve_column(primary, ["bootstrap_ci_low"], "primary lower CI column")
    ci_high_col = resolve_column(primary, ["bootstrap_ci_high"], "primary upper CI column")

    datasets = primary[dataset_col].astype(str).str.strip()
    counts = datasets.value_counts().to_dict()
    if counts != EXPECTED_DATASET_COUNTS:
        raise AssertionError(
            f"Frozen primary dataset cell counts changed: observed={counts}, expected={EXPECTED_DATASET_COUNTS}"
        )

    unknown_datasets = set(datasets).difference(CLASSIFICATION_DATASETS | REGRESSION_DATASETS)
    if unknown_datasets:
        raise AssertionError(f"Unexpected primary datasets: {sorted(unknown_datasets)}")

    derived_tasks = pd.Series(
        np.where(datasets.isin(CLASSIFICATION_DATASETS), "classification", "regression"),
        index=primary.index,
    )
    if "task_type" in primary.columns:
        explicit_tasks = primary["task_type"].map(canonical_token)
        if not explicit_tasks.equals(derived_tasks):
            mismatch_rows = primary.loc[explicit_tasks.ne(derived_tasks), [dataset_col, "task_type"]]
            raise AssertionError(
                "Explicit task_type disagrees with the frozen dataset map: "
                + mismatch_rows.to_dict(orient="records").__repr__()
            )

    metrics = primary[metric_col].map(canonical_token)
    bad_class_metrics = primary.loc[
        derived_tasks.eq("classification") & metrics.ne("roc_auc"),
        [dataset_col, model_col, metric_col],
    ]
    bad_reg_metrics = primary.loc[
        derived_tasks.eq("regression") & metrics.ne("rmse"),
        [dataset_col, model_col, metric_col],
    ]
    if not bad_class_metrics.empty or not bad_reg_metrics.empty:
        raise AssertionError(
            "Frozen primary metric map changed: "
            f"classification={bad_class_metrics.to_dict(orient='records')}; "
            f"regression={bad_reg_metrics.to_dict(orient='records')}"
        )

    class_models = set(primary.loc[derived_tasks.eq("classification"), model_col].astype(str))
    reg_models = set(primary.loc[derived_tasks.eq("regression"), model_col].astype(str))
    if class_models != {"LR", "RF", "XGB"}:
        raise AssertionError(f"Frozen classification models changed: {sorted(class_models)}")
    if reg_models != {"Ridge", "RF", "XGB"}:
        raise AssertionError(f"Frozen regression models changed: {sorted(reg_models)}")

    labels = primary[inference_col].map(canonical_token)
    label_counts = labels.value_counts().to_dict()
    expected_label_counts = {"inconclusive": 12, "target_balanced_better": 6}
    if label_counts != expected_label_counts:
        raise AssertionError(
            f"Frozen empirical label counts changed: observed={label_counts}, expected={expected_label_counts}; "
            f"column={inference_col}"
        )

    class_labels = labels[derived_tasks.eq("classification")]
    reg_labels = labels[derived_tasks.eq("regression")]
    if not class_labels.eq("inconclusive").all():
        rows = primary.loc[
            derived_tasks.eq("classification") & labels.ne("inconclusive"),
            [dataset_col, model_col, metric_col, inference_col],
        ]
        raise AssertionError(
            f"Frozen classification decisions changed: {rows.to_dict(orient='records')}"
        )
    if not reg_labels.eq("target_balanced_better").all():
        rows = primary.loc[
            derived_tasks.eq("regression") & labels.ne("target_balanced_better"),
            [dataset_col, model_col, metric_col, inference_col],
        ]
        raise AssertionError(
            f"Frozen regression decisions changed: {rows.to_dict(orient='records')}"
        )

    numeric_columns = [effect_col, p_holm_col, ci_low_col, ci_high_col]
    numeric = primary[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise AssertionError(f"Non-finite primary inference values in {numeric_columns}")

    class_mask = derived_tasks.eq("classification")
    reg_mask = derived_tasks.eq("regression")
    if not (numeric.loc[class_mask, p_holm_col] >= 0.05 - 1e-12).all():
        raise AssertionError("A frozen classification Holm-adjusted P value fell below 0.05")
    if not (numeric.loc[reg_mask, effect_col] > 0.0).all():
        raise AssertionError("A frozen regression mean effect is not positive")
    if not (numeric.loc[reg_mask, ci_low_col] > 0.0).all():
        raise AssertionError("A frozen regression bootstrap interval is not wholly above zero")
    if not (numeric.loc[reg_mask, p_holm_col] < 0.05).all():
        raise AssertionError("A frozen regression Holm-adjusted P value is not below 0.05")

    singleton_dataset_col = resolve_column(singleton, ["dataset"], "singleton dataset column")
    singleton_model_col = resolve_column(singleton, ["model"], "singleton model column")
    singleton_metric_col = resolve_column(
        singleton,
        ["primary_metric", "metric"],
        "singleton metric column",
    )
    singleton_effect_col = resolve_column(
        singleton,
        ["mean_effect_positive_is_balanced_better", "mean_effect"],
        "singleton mean-effect column",
    )
    singleton_ci_low_col = resolve_column(
        singleton, ["bootstrap_ci_low"], "singleton lower CI column"
    )
    singleton_ci_high_col = resolve_column(
        singleton, ["bootstrap_ci_high"], "singleton upper CI column"
    )
    singleton_p_col = resolve_column(
        singleton,
        ["p_raw_descriptive", "wilcoxon_p_descriptive", "p_raw"],
        "singleton descriptive P-value column",
    )
    if "analysis_role" in singleton.columns:
        roles = singleton["analysis_role"].map(canonical_token)
        if not roles.eq("acyclic_singleton_sensitivity").all():
            raise AssertionError(
                f"Unexpected singleton analysis roles: {sorted(set(roles))}"
            )
    singleton_counts = singleton[singleton_dataset_col].astype(str).value_counts().to_dict()
    if singleton_counts != {"ESOL": 3, "FreeSolv": 3}:
        raise AssertionError(f"Frozen singleton dataset counts changed: {singleton_counts}")
    if set(singleton[singleton_model_col].astype(str)) != {"Ridge", "RF", "XGB"}:
        raise AssertionError("Frozen singleton model set changed")
    if not singleton[singleton_metric_col].map(canonical_token).eq("rmse").all():
        raise AssertionError("Frozen singleton metric is not uniformly RMSE")
    singleton_numeric = singleton[
        [singleton_effect_col, singleton_ci_low_col, singleton_ci_high_col, singleton_p_col]
    ].apply(pd.to_numeric, errors="coerce")
    if singleton_numeric.isna().any().any() or not np.isfinite(singleton_numeric.to_numpy(float)).all():
        raise AssertionError("Non-finite singleton sensitivity values")
    esol_effects = singleton_numeric.loc[
        singleton[singleton_dataset_col].astype(str).eq("ESOL"), singleton_effect_col
    ]
    freesolv_effects = singleton_numeric.loc[
        singleton[singleton_dataset_col].astype(str).eq("FreeSolv"), singleton_effect_col
    ]
    if not (esol_effects > 0).all() or not (freesolv_effects < 0).all():
        raise AssertionError(
            "Frozen singleton effect directions changed: "
            f"ESOL={esol_effects.tolist()}, FreeSolv={freesolv_effects.tolist()}"
        )

    mean_labels = mean_only["freeze_label"].map(canonical_token)
    mean_datasets = mean_only["dataset"].astype(str).str.strip()
    expected_mean_pairs = {
        ("main_regression", "ESOL"),
        ("main_regression", "FreeSolv"),
        ("acyclic_singleton_sensitivity", "ESOL"),
        ("acyclic_singleton_sensitivity", "FreeSolv"),
    }
    observed_mean_pairs = set(zip(mean_labels, mean_datasets))
    if observed_mean_pairs != expected_mean_pairs:
        raise AssertionError(
            f"Frozen mean-only analysis rows changed: observed={sorted(observed_mean_pairs)}"
        )

    print(
        "EMPIRICAL SCIENCE AUDIT: PASS "
        "(labels normalized from inference_label; 12/12 classification inconclusive; "
        "6/6 primary regression target_balanced_better; singleton directions preserved)",
        flush=True,
    )
    return mean_only, primary, singleton


def audit_empirical() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_only = pd.read_csv(
        core.require(core.EMP_TABLES / "q1_mean_only_regression_summary_v3.csv")
    )
    primary = pd.read_csv(
        core.require(core.EMP_TABLES / "primary_inference_summary_v3.csv")
    )
    singleton = pd.read_csv(
        core.require(core.EMP_TABLES / "acyclic_singleton_sensitivity_v3.csv")
    )
    return audit_empirical_frames(mean_only, primary, singleton)


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def display_inference(value: object) -> str:
    token = canonical_token(value)
    mapping = {
        "inconclusive": "Inconclusive",
        "target_balanced_better": "Target-balanced better",
        "target_balanced_worse": "Target-balanced worse",
    }
    return mapping.get(token, str(value).replace("_", " "))


def rewrite_empirical_tex_tables(primary: pd.DataFrame, singleton: pd.DataFrame) -> None:
    primary_metric = resolve_column(primary, ["primary_metric", "metric"], "primary metric")
    primary_size = resolve_column(primary, ["mean_size_metric", "mean_size"], "primary size mean")
    primary_bal = resolve_column(
        primary, ["mean_balanced_metric", "mean_balanced"], "primary response-aware mean"
    )
    primary_effect = resolve_column(
        primary,
        ["mean_effect", "mean_effect_positive_is_balanced_better"],
        "primary effect",
    )
    primary_p = resolve_column(primary, ["p_holm", "holm_p"], "primary Holm P")
    primary_inf = resolve_column(
        primary, ["inference_label", "inference", "decision"], "primary inference"
    )

    lines = [
        r"\begin{tabular}{lllrrrrrrl}",
        r"\toprule",
        r"Dataset & Model & Metric & Size mean & Response-aware mean & Effect & CI low & CI high & Holm $P$ & Inference \\",
        r"\midrule",
    ]
    for row in primary.itertuples(index=False):
        values = row._asdict()
        lines.append(
            " & ".join(
                [
                    tex_escape(values["dataset"]),
                    tex_escape(values["model"]),
                    tex_escape(str(values[primary_metric]).upper()),
                    f"{float(values[primary_size]):.6f}",
                    f"{float(values[primary_bal]):.6f}",
                    f"{float(values[primary_effect]):+.6f}",
                    f"{float(values['bootstrap_ci_low']):+.6f}",
                    f"{float(values['bootstrap_ci_high']):+.6f}",
                    f"{float(values[primary_p]):.6f}",
                    tex_escape(display_inference(values[primary_inf])),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (core.GEN / "primary_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    singleton_metric = resolve_column(
        singleton, ["primary_metric", "metric"], "singleton metric"
    )
    singleton_size = resolve_column(
        singleton, ["mean_size_rmse", "mean_size_metric", "mean_size"], "singleton size mean"
    )
    singleton_bal = resolve_column(
        singleton,
        ["mean_balanced_rmse", "mean_balanced_metric", "mean_balanced"],
        "singleton response-aware mean",
    )
    singleton_effect = resolve_column(
        singleton,
        ["mean_effect_positive_is_balanced_better", "mean_effect"],
        "singleton effect",
    )
    singleton_p = resolve_column(
        singleton,
        ["p_raw_descriptive", "wilcoxon_p_descriptive", "p_raw"],
        "singleton descriptive P",
    )
    lines = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Dataset & Model & Metric & Size mean & Response-aware mean & Effect & CI low & CI high & Descriptive $P$ \\",
        r"\midrule",
    ]
    for row in singleton.itertuples(index=False):
        values = row._asdict()
        lines.append(
            " & ".join(
                [
                    tex_escape(values["dataset"]),
                    tex_escape(values["model"]),
                    tex_escape(str(values[singleton_metric]).upper()),
                    f"{float(values[singleton_size]):.6f}",
                    f"{float(values[singleton_bal]):.6f}",
                    f"{float(values[singleton_effect]):+.6f}",
                    f"{float(values['bootstrap_ci_low']):+.6f}",
                    f"{float(values['bootstrap_ci_high']):+.6f}",
                    f"{float(values[singleton_p]):.6f}",
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (core.GEN / "singleton_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("EMPIRICAL SI TABLE SCHEMA GATE: PASS", flush=True)


def install(build_module: ModuleType) -> None:
    """Install compatibility/audit functions into the already imported finalizer build module."""
    original_write_generated = build_module.write_generated

    def write_generated_patched(
        summary: pd.DataFrame,
        quality: pd.DataFrame,
        bridge: pd.DataFrame,
        primary: pd.DataFrame,
        singleton: pd.DataFrame,
    ) -> None:
        original_write_generated(summary, quality, bridge, primary, singleton)
        rewrite_empirical_tex_tables(primary, singleton)

    build_module.audit_empirical = audit_empirical
    build_module.write_generated = write_generated_patched
