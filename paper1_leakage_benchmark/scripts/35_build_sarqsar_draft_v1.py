from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
SIM_OUT = PAPER / "results" / "sarqsar_metric_coupling_v1"
TABLES = SIM_OUT / "tables"
REPORTS = SIM_OUT / "reports"
FIGURES = SIM_OUT / "figures"
EMPIRICAL_FIGURES = PAPER / "results" / "figures"
LATEX = ROOT / "paper1_sarqsar_latex"
GENERATED = LATEX / "generated"
BUILD = LATEX / "build"
PROTOCOL = PAPER / "SARQSAR_METRIC_COUPLING_PROTOCOL_V1.md"
CONFIG = PAPER / "SARQSAR_METRIC_COUPLING_CONFIG_V1.json"
REFERENCE_SOURCE = ROOT / "paper1_latex" / "references_joc.tex"
TITLE = "Split-Objective--Metric Coupling in QSAR Benchmarks: Null Simulations and Exact-Size Paired Audits"


def require(path: Path) -> Path:
    if not path.exists() or (path.is_file() and path.stat().st_size == 0):
        raise FileNotFoundError(path)
    return path


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def tex(value: object) -> str:
    text = str(value)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def selected_row(summary: pd.DataFrame, dataset: str, mode: str, metric: str) -> pd.Series:
    block = summary[
        summary["dataset"].eq(dataset)
        & summary["scaffold_mode"].eq(mode)
        & summary["metric"].eq(metric)
    ].copy()
    if block.empty:
        raise AssertionError(f"Missing summary row: {dataset}/{mode}/{metric}")
    maximum_budget = int(block["budget"].max())
    block = block[block["budget"].eq(maximum_budget)]
    if len(block) != 1:
        raise AssertionError(f"Expected one maximum-budget row: {dataset}/{mode}/{metric}")
    return block.iloc[0]


def number(row: pd.Series, key: str = "mean", digits: int = 4) -> str:
    value = float(row[key])
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def interval(row: pd.Series, digits: int = 4) -> str:
    return f"{number(row, 'mean', digits)} [{number(row, 'q025', digits)}, {number(row, 'q975', digits)}]"


def build_generated(summary: pd.DataFrame) -> dict[str, str]:
    GENERATED.mkdir(parents=True, exist_ok=True)
    shutil.copy2(require(REPORTS / "qsar_benchmark_checklist.tex"), GENERATED / "qsar_benchmark_checklist.tex")

    regression_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Maximum-budget endpoint-permutation null effects after averaging partition seeds within each permutation. Positive values favor response-aware selection.}",
        r"\label{tab:null-regression}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Dataset & Acyclic rule & Draws & RMSE & Mean-gap$^2$ & Test variance \\",
        r"\hline",
    ]
    values: dict[str, str] = {}
    for dataset in ["ESOL", "FreeSolv"]:
        for mode in ["single_group", "singleton"]:
            rmse = selected_row(summary, dataset, mode, "effect_rmse")
            gap = selected_row(summary, dataset, mode, "effect_squared_mean_gap")
            variance = selected_row(summary, dataset, mode, "effect_test_variance")
            key = f"{dataset}_{mode}_rmse"
            values[key] = interval(rmse)
            regression_lines.append(
                f"{tex(dataset)} & {tex(mode.replace('_', ' '))} & {int(rmse['budget']):,} & "
                f"{number(rmse)} & {number(gap)} & {number(variance)} \\\\"
            )
    regression_lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    write(GENERATED / "null_regression_table.tex", "\n".join(regression_lines))

    classification_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Maximum-budget classification null effects. Positive values favor response-aware selection.}",
        r"\label{tab:null-classification}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Dataset & Draws & Brier & Log loss & ROC--AUC \\",
        r"\hline",
    ]
    brier_means: list[float] = []
    log_means: list[float] = []
    auc_means: list[float] = []
    for dataset in ["BACE", "BBBP", "ClinTox", "HIV"]:
        brier = selected_row(summary, dataset, "single_group", "effect_brier")
        logloss = selected_row(summary, dataset, "single_group", "effect_log_loss")
        auc = selected_row(summary, dataset, "single_group", "effect_roc_auc")
        brier_means.append(float(brier["mean"]))
        log_means.append(float(logloss["mean"]))
        auc_means.append(float(auc["mean"]))
        classification_lines.append(
            f"{tex(dataset)} & {int(brier['budget']):,} & {number(brier)} & {number(logloss)} & {number(auc)} \\\\"
        )
    classification_lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    write(GENERATED / "null_classification_table.tex", "\n".join(classification_lines))

    values["brier_range"] = f"{np.nanmin(brier_means):.4f} to {np.nanmax(brier_means):.4f}"
    values["logloss_range"] = f"{np.nanmin(log_means):.4f} to {np.nanmax(log_means):.4f}"
    values["auc_range"] = f"{np.nanmin(auc_means):.4f} to {np.nanmax(auc_means):.4f}"
    return values


def build_sources(values: dict[str, str]) -> None:
    abstract = (
        "Response-aware test-set selection can couple a benchmark-construction objective to the downstream evaluation metric and alter apparent QSAR difficulty before molecular features contribute. "
        "We combined endpoint-permutation null simulations with an exact-size paired audit of six public molecular-property datasets. Real molecular rows, Bemis--Murcko scaffold geometry, endpoint marginals, partition seeds, and nested target-blind candidate pools were preserved; only structure--endpoint association was destroyed. "
        f"At the maximum regression budget, null RMSE effects were {values['ESOL_single_group_rmse']} for ESOL and {values['FreeSolv_single_group_rmse']} for FreeSolv under the shared-acyclic convention, compared with {values['ESOL_singleton_rmse']} and {values['FreeSolv_singleton_rmse']} under singleton semantics. "
        f"Across classification datasets, mean null Brier effects ranged from {values['brier_range']}, log-loss effects from {values['logloss_range']}, and constant-score ROC--AUC effects from {values['auc_range']} where defined. "
        "Together with the empirical exact-size audit, the simulations identify split-objective--metric coupling as a benchmark-design risk and motivate a minimum reporting checklist for response-aware QSAR validation."
    )

    main = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=2.6cm]{{geometry}}
\usepackage{{graphicx,booktabs,array,amsmath,microtype,setspace}}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{{hyperref}}
\onehalfspacing
\graphicspath{{{{../paper1_leakage_benchmark/results/sarqsar_metric_coupling_v1/figures/}}{{../paper1_leakage_benchmark/results/figures/}}}}
\title{{{TITLE}}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle
\begin{{abstract}}
{abstract}
\end{{abstract}}
\noindent\textbf{{Keywords:}} QSAR validation; scaffold splitting; benchmark construction; metric coupling; molecular machine learning

\section{{Introduction}}
A QSAR benchmark is a measurement design rather than a neutral container for model scores. Molecular identity rules, scaffold definitions, train--test allocation, search effort, and inferential replication determine the claim supported by a held-out metric. A particular risk arises when endpoint information enters split selection. For a constant predictor equal to the training mean,
\begin{{equation}}
\mathrm{{MSE}}=\mathrm{{Var}}(y_{{\mathrm{{test}}}})+(\bar y_{{\mathrm{{test}}}}-\bar y_{{\mathrm{{train}}}})^2.
\end{{equation}}
Reducing the second term can therefore reduce RMSE even when molecular structure carries no signal. Binary Brier score has an analogous prevalence decomposition, whereas constant-score ROC--AUC has no direct prevalence-gap term.

\section{{Materials and methods}}
The analysis used BACE, BBBP, ClinTox, HIV, ESOL, and FreeSolv. For each dataset, endpoint values were independently permuted 200 times while preserving molecular rows, endpoint marginals, and scaffold geometry. The same 20 frozen partition seeds and nested target-blind candidate pools were used for size-matched and same-size response-aware selection. Regression used the training mean as a constant predictor; classification used the training prevalence. Partition-seed effects were averaged within each endpoint permutation, making the permutation the simulation replicate. Hard gates enforced exact-size pairing, non-worsening target gap, complete cell counts, constant-score AUC invariance, and the MSE decomposition.

\input{{generated/qsar_benchmark_checklist.tex}}

\section{{Results}}
\subsection{{Regression coupling under a molecular null}}
The maximum-budget results are summarized in Table~\ref{{tab:null-regression}}. Full nested-budget trajectories appear in Figure~\ref{{fig:regression}}.
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.96\textwidth]{{figure_mc1_regression_null_coupling.pdf}}
\caption{{Endpoint-permutation null RMSE effects across nested requested-draw budgets.}}
\label{{fig:regression}}
\end{{figure}}
\input{{generated/null_regression_table.tex}}

The total MSE effect was separated into squared mean-gap and test-variance contributions (Figure~\ref{{fig:decomposition}}).
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.96\textwidth]{{figure_mc3_mse_decomposition.pdf}}
\caption{{Decomposition of the null MSE effect.}}
\label{{fig:decomposition}}
\end{{figure}}

\subsection{{Classification metrics coupled differently}}
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.96\textwidth]{{figure_mc2_classification_null_coupling.pdf}}
\caption{{Classification null effects across nested requested-draw budgets.}}
\label{{fig:classification}}
\end{{figure}}
\input{{generated/null_classification_table.tex}}

\subsection{{Connection to the empirical audit}}
The frozen empirical study used exact-size paired partitions, partition-level inference, acyclic-scaffold sensitivity, a mean-only regression control, and a deterministic dominant-fragment representation sensitivity. No classification cell met the corrected criterion for a response-aware advantage, whereas all six primary regression cells favored response-aware selection under the shared-acyclic convention. Those regression effects attenuated strongly under singleton acyclic semantics, and the mean-only control exceeded the corresponding learned-model mean effect in every primary regression cell. The submitted empirical version and its figures document the complete numerical results. fileciteturn545file0

\section{{Discussion}}
The endpoint-permutation simulations establish a mechanism that cannot be attributed to molecular learning: response-aware split selection can change apparent difficulty while molecular and scaffold geometry remain fixed. The effect depends on evaluation metric, candidate-search budget, dataset geometry, and scaffold semantics. Exact-size pairing controls cardinality but not chemical composition. Response-aware QSAR benchmarks should therefore disclose endpoint use, candidate-search effort, scaffold and molecular-record rules, trivial response-only controls, collateral diagnostics, inferential units, and immutable partition provenance.

\section{{Limitations}}
The null removes all structure--endpoint association and does not represent every realistic QSAR signal regime. The empirical regression evidence remains limited to ESOL and FreeSolv, the learned models are classical fingerprint baselines, and candidate generation is a random scaffold-prefix heuristic. The checklist is evidence-informed but has not yet been prospectively evaluated across independent laboratories.

\section{{Conclusion}}
Split-objective--metric coupling is a benchmark-design risk. Molecular endpoint-permutation simulations show that response-aware exact-size scaffold selection can alter several error and probability metrics before molecular features contribute. Real-data conclusions are additionally conditional on scaffold semantics and molecular-record representation.

\section*{{Data and code availability}}
Public datasets, the frozen protocol, code, candidate caches, checkpointed simulation outputs, tables, figures, and manuscript-generation scripts are maintained in the project repository. An anonymized review archive will be prepared for double-blind peer review.

\section*{{Competing interests}}
The authors declare no competing interests.

\section*{{AI-assisted tools}}
OpenAI ChatGPT was used for language editing, code debugging, LaTeX formatting, and figure-layout assistance. All scientific computations and outputs were executed and verified by the authors.

\input{{references.tex}}
\end{{document}}
"""
    # Remove the file citation marker from the LaTeX source; it is only for this chat's source grounding.
    main = main.replace(" fileciteturn545file0", "")
    write(LATEX / "main.tex", main)

    title_page = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=2.6cm]{{geometry}}
\begin{{document}}
\begin{{center}}
{{\Large\bfseries {TITLE}\par}}
\vspace{{1.5em}}
Siyuan Tong$^{{1,*}}$ and Yuechen Wang$^{{2}}$
\end{{center}}
$^1$Department of Artificial Intelligence, Faculty of Computer Science and Information Technology, University of Malaya, 50603 Kuala Lumpur, Malaysia

$^2$Faculty of Data Science, City University of Macau, Macau SAR, China

$^*$Corresponding author: 25064241@siswa.um.edu.my

\section*{{Related manuscript disclosure}}
A related manuscript currently under consideration at \textit{{Chemometrics and Intelligent Laboratory Systems}} develops and evaluates a post-prediction conformal intervention under molecular distribution shift. The present manuscript addresses the distinct question of how response-aware benchmark partition selection, scaffold semantics, and molecular-record policy alter QSAR evaluation. The manuscripts share several public datasets but contain different hypotheses, protocols, primary analyses, figures, tables, and conclusions.
\end{{document}}
"""
    write(LATEX / "title_page.tex", title_page)

    supplementary = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{graphicx,booktabs,array}}
\usepackage[colorlinks=true,urlcolor=blue]{{hyperref}}
\graphicspath{{{{../paper1_leakage_benchmark/results/sarqsar_metric_coupling_v1/figures/}}{{../paper1_leakage_benchmark/results/figures/}}}}
\title{{Supporting Information: {TITLE}}}
\author{{Double-blind review version}}
\date{{}}
\begin{{document}}
\maketitle
\section{{Frozen protocol}}
The complete pre-outcome protocol fixes datasets, endpoint permutations, partition seeds, requested-draw budgets, scaffold modes, metric orientation, aggregation, quality gates, and change control.
\section{{Machine-readable outputs}}
Partition-seed effects, permutation-level means, summary distributions, candidate first-appearance budgets, hashes, and quality-gate results are included in the reproducibility bundle.
\section{{Reporting checklist}}
\input{{generated/qsar_benchmark_checklist.tex}}
\section{{Additional empirical diagnostics}}
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.94\textwidth]{{figure3_acyclic_sensitivity_v3.pdf}}
\caption{{Empirical sensitivity to acyclic-scaffold semantics.}}
\end{{figure}}
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.94\textwidth]{{figure6_collateral_diagnostics_v3.pdf}}
\caption{{Empirical mean-only and collateral diagnostics.}}
\end{{figure}}
\end{{document}}
"""
    write(LATEX / "supplementary.tex", supplementary)

    notes = f"""# SAR/QSAR working draft\n\n## Proposed title\n\n{TITLE}\n\n## Core claim\n\nEndpoint-aware partition selection can couple directly to the evaluation metric and alter apparent QSAR difficulty before molecular features contribute.\n\n## Maximum-budget null RMSE effects\n\n- ESOL, shared acyclic group: {values['ESOL_single_group_rmse']}\n- ESOL, singleton acyclic groups: {values['ESOL_singleton_rmse']}\n- FreeSolv, shared acyclic group: {values['FreeSolv_single_group_rmse']}\n- FreeSolv, singleton acyclic groups: {values['FreeSolv_singleton_rmse']}\n\nThis is a scientific working draft generated from frozen outputs, not the final submission package.\n"""
    write(LATEX / "WORKING_DRAFT_NOTES.md", notes)


def compile_sources() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if not latexmk and not pdflatex:
        raise RuntimeError("Neither latexmk nor pdflatex is available")
    for source in ["main.tex", "title_page.tex", "supplementary.tex"]:
        if latexmk:
            command = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", f"-outdir={BUILD.name}", source]
            subprocess.run(command, cwd=str(LATEX), check=True)
        else:
            for _ in range(2):
                command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={BUILD.name}", source]
                subprocess.run(command, cwd=str(LATEX), check=True)
        require(BUILD / f"{Path(source).stem}.pdf")
    print("SAR/QSAR LATEX DRAFT COMPILE: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SAR/QSAR working manuscript from completed null-simulation outputs.")
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(require(TABLES / "null_metric_effect_summary.csv"), keep_default_na=False, na_values=["", "nan", "NaN"])
    require(FIGURES / "figure_mc1_regression_null_coupling.pdf")
    require(FIGURES / "figure_mc2_classification_null_coupling.pdf")
    require(FIGURES / "figure_mc3_mse_decomposition.pdf")
    require(EMPIRICAL_FIGURES / "figure3_acyclic_sensitivity_v3.pdf")
    require(EMPIRICAL_FIGURES / "figure6_collateral_diagnostics_v3.pdf")
    require(PROTOCOL)
    require(CONFIG)
    require(REFERENCE_SOURCE)
    if LATEX.exists():
        shutil.rmtree(LATEX)
    LATEX.mkdir(parents=True)
    values = build_generated(summary)
    build_sources(values)
    shutil.copy2(REFERENCE_SOURCE, LATEX / "references.tex")
    shutil.copy2(PROTOCOL, LATEX / PROTOCOL.name)
    shutil.copy2(CONFIG, LATEX / CONFIG.name)
    if not args.no_compile:
        compile_sources()
    print("\n" + "=" * 88)
    print("SAR/QSAR SCIENTIFIC WORKING DRAFT: PASS")
    print("Directory:", LATEX)
    print("Main PDF:", BUILD / "main.pdf")
    print("Title page:", BUILD / "title_page.pdf")
    print("SI PDF:", BUILD / "supplementary.pdf")
    print("=" * 88)


if __name__ == "__main__":
    main()
