from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper1_leakage_benchmark" / "results" / "tables"
GEN = ROOT / "paper1_latex" / "generated"
GEN.mkdir(parents=True, exist_ok=True)


def esc(x: object) -> str:
    return str(x).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def fmt(x: object, n: int = 4) -> str:
    return f"{float(x):.{n}f}"


def write(name: str, lines: list[str]) -> None:
    (GEN / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_only_table() -> None:
    df = pd.read_csv(TABLES / "q1_mean_only_regression_summary_v3.csv", keep_default_na=False)
    lines = [
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Mean-only regression control. The predictor assigns every test molecule the training-set target mean. Positive RMSE effects indicate lower error for the target-mean-balanced partition.}",
        "\\label{tab:si-mean-only}", "\\begin{tabular}{llrrrrr}", "\\toprule",
        "Analysis & Dataset & Size RMSE & Balanced RMSE & Mean effect & 95\\% CI & Descriptive $P$ \\\\", "\\midrule",
    ]
    for r in df.itertuples(index=False):
        label = "Primary" if r.freeze_label == "main_regression" else "Singleton"
        lines.append(
            f"{label} & {r.dataset} & {fmt(r.mean_size_intercept_rmse)} & {fmt(r.mean_balanced_intercept_rmse)} & "
            f"{fmt(r.mean_effect_size_minus_balanced_rmse)} & [{fmt(r.bootstrap_ci_low)}, {fmt(r.bootstrap_ci_high)}] & {fmt(r.wilcoxon_p_descriptive)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("q1_mean_only_table_v3.tex", lines)


def collateral_table() -> None:
    df = pd.read_csv(TABLES / "q1_collateral_diagnostics_summary_v3.csv", keep_default_na=False)
    keep = ["abs_target_mean_gap", "target_ks_statistic", "target_wasserstein", "largest_test_scaffold_fraction", "effective_test_scaffolds", "acyclic_test_fraction"]
    labels = {
        "abs_target_mean_gap": "Target-mean gap", "target_ks_statistic": "KS statistic",
        "target_wasserstein": "Wasserstein", "largest_test_scaffold_fraction": "Largest-scaffold fraction",
        "effective_test_scaffolds": "Effective scaffolds", "acyclic_test_fraction": "Acyclic test fraction",
    }
    lines = [
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Collateral partition diagnostics. Values are means across 20 paired partitions; $\\Delta$ is target-mean-balanced minus size-matched. Only target-mean gap entered the selection objective.}",
        "\\label{tab:si-collateral}", "\\scriptsize", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llrrrr}", "\\toprule",
        "Dataset & Diagnostic & Size mean & Balanced mean & Mean $\\Delta$ & Balanced/size ratio \\\\", "\\midrule",
    ]
    for ds in ["BACE", "BBBP", "ClinTox", "HIV", "ESOL", "FreeSolv"]:
        for metric in keep:
            r = df[(df.dataset == ds) & (df.metric == metric)].iloc[0]
            ratio = r.mean_ratio_balanced_over_size
            ratio_text = "--" if str(ratio) == "" or pd.isna(ratio) else fmt(ratio, 3)
            lines.append(f"{ds} & {labels[metric]} & {fmt(r.mean_size,3)} & {fmt(r.mean_balanced,3)} & {fmt(r.mean_delta_balanced_minus_size,3)} & {ratio_text} \\\\")
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    write("q1_collateral_table_v3.tex", lines)


def seed_table() -> None:
    df = pd.read_csv(TABLES / "q1_model_seed_summary_v3.csv", keep_default_na=False)
    p = df.pivot_table(index=["freeze_label", "dataset", "model"], columns="model_seed", values="mean_effect").reset_index()
    lines = [
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Predeclared RF/XGB stochastic-model seed sensitivity. Values are mean paired effects over the five designated partition seeds; positive values favor target-mean-aware selection. Repeated model seeds are not inferential replicates.}",
        "\\label{tab:si-model-seed}", "\\scriptsize", "\\begin{tabular}{llllrrr}", "\\toprule",
        "Analysis & Dataset & Model & Metric & Seed 17 & Seed 29 & Seed 43 \\\\", "\\midrule",
    ]
    for _, r in p.iterrows():
        metric = "AUC" if r.freeze_label == "main_classification" else "RMSE"
        analysis = {"main_classification":"Primary classification", "main_regression":"Primary regression", "acyclic_singleton_sensitivity":"Singleton regression"}[r.freeze_label]
        lines.append(f"{analysis} & {r.dataset} & {r.model} & {metric} & {fmt(r[17])} & {fmt(r[29])} & {fmt(r[43])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("q1_model_seed_table_v3.tex", lines)


def environment_table() -> None:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib", "rdkit"]
    rows = [("Python", platform.python_version()), ("Platform", platform.platform())]
    for p in packages:
        try:
            v = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            v = "not installed"
        rows.append((p, v))
    lines = [
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Software environment captured during the final reproducibility build.}",
        "\\label{tab:si-environment}", "\\begin{tabular}{ll}", "\\toprule", "Component & Version \\\\", "\\midrule",
    ]
    for k,v in rows:
        lines.append(f"{esc(k)} & {esc(v)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("q1_environment_table_v3.tex", lines)
    (TABLES / "q1_environment_versions_v3.txt").write_text("\n".join(f"{k}: {v}" for k,v in rows) + "\n", encoding="utf-8")


def main() -> None:
    mean_only_table(); collateral_table(); seed_table(); environment_table()
    print("Q1 GENERATED LATEX TABLES: PASS")


if __name__ == "__main__":
    main()
