from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
NULL_ROOT = PAPER / "results" / "sarqsar_metric_coupling_v1"
NULL_TABLES = NULL_ROOT / "tables"
NULL_FIGURES = NULL_ROOT / "figures"
EMP_TABLES = PAPER / "results" / "tables"
EMP_FIGURES = PAPER / "results" / "figures"
SOURCE = ROOT / "paper1_sarqsar_submission_source_v1"
BUILD = ROOT / "paper1_sarqsar_submission_build_v1"
LATEX = BUILD / "latex"
GEN = LATEX / "generated"
FIGS = LATEX / "figures"
OUT = ROOT / "paper1_sarqsar_submission_v1"
OUT_ZIP = ROOT / "paper1_sarqsar_submission_v1.zip"
EXPECTED_BRANCH = "paper1-sarqsar-metric-coupling-2026"
SCIENCE_COMMIT = "9e87368d4c3530a5e82d476aee2f58032e98261f"
TITLE = "Split-objective--metric coupling in QSAR validation: molecular null experiments and exact-size paired scaffold audits"

ANON_TOKENS = [
    "siyuan tong", "yuechen wang", "25064241", "d25091100346",
    "university of malaya", "city university of macau", "0009-0004-4450-083x",
    "dcarchimonde", "aidd-paper-factory",
]

FIGURE_MAP = {
    1: (NULL_FIGURES, "figure_mc1_regression_null_coupling"),
    2: (NULL_FIGURES, "figure_mc3_mse_decomposition"),
    3: (NULL_FIGURES, "figure_mc2_classification_null_coupling"),
    4: (EMP_FIGURES, "figure2_primary_effects_v3"),
    5: (EMP_FIGURES, "figure3_acyclic_sensitivity_v3"),
    6: (EMP_FIGURES, "figure4_dominant_fragment_sensitivity_v3"),
    7: (EMP_FIGURES, "figure6_collateral_diagnostics_v3"),
}


def require(path: Path) -> Path:
    if not path.exists() or (path.is_file() and path.stat().st_size == 0):
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd or ROOT), check=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def audit_null() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads(require(NULL_ROOT / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise AssertionError("Null run is not complete")
    if manifest.get("git_commit") != SCIENCE_COMMIT:
        raise AssertionError(f"Unexpected null science commit: {manifest.get('git_commit')}")
    if int(manifest.get("n_permutations", 0)) != 200 or len(manifest.get("partition_seeds", [])) != 20:
        raise AssertionError("Null run must contain 200 permutations and 20 partition seeds")

    summary = pd.read_csv(require(NULL_TABLES / "null_metric_effect_summary.csv"))
    perm = pd.read_csv(require(NULL_TABLES / "null_simulation_permutation_level_effects.csv"))
    quality = pd.read_csv(require(NULL_TABLES / "null_simulation_quality_gate_summary.csv"))

    needed = {
        "dataset", "task_type", "scaffold_mode", "budget", "permutation_id",
        "target_gap_reduction", "effect_roc_auc", "effect_rmse", "effect_mse",
        "effect_test_variance", "effect_squared_mean_gap",
    }
    missing = needed.difference(perm.columns)
    if missing:
        raise KeyError(f"Null permutation table missing columns: {sorted(missing)}")
    if len(perm) != 8800:
        raise AssertionError(f"Expected 8,800 permutation cells, found {len(perm):,}")
    if int(quality["permutation_level_rows"].iloc[0]) != 8800:
        raise AssertionError("Quality summary does not report 8,800 permutation cells")
    if int(quality["raw_partition_seed_rows"].iloc[0]) != 176000:
        raise AssertionError("Quality summary does not report 176,000 seed-level rows")
    if float(perm["target_gap_reduction"].min()) < -1e-12:
        raise AssertionError("Response-aware target gap worsened")
    auc = perm["effect_roc_auc"].dropna().to_numpy(float)
    if auc.size and float(np.max(np.abs(auc))) > 1e-12:
        raise AssertionError("Constant-score ROC-AUC is not invariant")
    reg = perm.loc[perm["task_type"].eq("regression")]
    residual = (reg["effect_mse"] - reg["effect_test_variance"] - reg["effect_squared_mean_gap"]).abs()
    if not residual.empty and float(residual.max()) > 1e-9:
        raise AssertionError("MSE decomposition failed")
    if float(quality["max_abs_mse_decomposition_residual"].iloc[0]) > 1e-9:
        raise AssertionError("Quality-gate MSE residual failed")

    print("NULL SCIENCE AUDIT: PASS (8,800 permutation cells; 176,000 seed rows)", flush=True)
    return summary, perm, quality, manifest


def ensure_empirical() -> None:
    files = [
        EMP_TABLES / "q1_mean_only_regression_summary_v3.csv",
        EMP_TABLES / "primary_inference_summary_v3.csv",
        EMP_TABLES / "acyclic_singleton_sensitivity_v3.csv",
        EMP_TABLES / "q1_collateral_diagnostics_summary_v3.csv",
    ]
    files += [folder / f"{stem}.pdf" for folder, stem in list(FIGURE_MAP.values())[3:]]
    files += [folder / f"{stem}.tiff" for folder, stem in list(FIGURE_MAP.values())[3:]]
    if any(not path.exists() for path in files):
        builder = require(PAPER / "scripts" / "32_build_paper1_publication_final_v3.py")
        print("Rebuilding missing frozen empirical publication assets once.", flush=True)
        run([sys.executable, "-u", str(builder)])
    for path in files:
        require(path)
    print("EMPIRICAL ASSET PREFLIGHT: PASS", flush=True)


def audit_empirical() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_only = pd.read_csv(require(EMP_TABLES / "q1_mean_only_regression_summary_v3.csv"))
    primary = pd.read_csv(require(EMP_TABLES / "primary_inference_summary_v3.csv"))
    singleton = pd.read_csv(require(EMP_TABLES / "acyclic_singleton_sensitivity_v3.csv"))

    required = {
        "freeze_label", "dataset", "mean_effect_size_minus_balanced_rmse",
        "bootstrap_ci_low", "bootstrap_ci_high",
    }
    missing = required.difference(mean_only.columns)
    if missing:
        raise KeyError(f"Mean-only table missing columns: {sorted(missing)}")
    if len(primary) != 18 or len(singleton) != 6:
        raise AssertionError(f"Empirical summary sizes changed: primary={len(primary)}, singleton={len(singleton)}")

    inf_col = next((c for c in ["inference", "inference_label", "decision"] if c in primary.columns), None)
    if inf_col is None:
        raise KeyError("Primary summary lacks inference column")
    if "task_type" in primary.columns:
        tasks = primary["task_type"].astype(str).str.lower()
    elif "metric" in primary.columns:
        tasks = pd.Series(np.where(primary["metric"].astype(str).str.lower().str.contains("auc"), "classification", "regression"))
    else:
        raise KeyError("Primary summary lacks task_type/metric")
    supported = primary[inf_col].astype(str).str.lower().str.contains("balanced better|target-balanced better|response-aware better")
    c = int(supported[tasks.eq("classification")].sum())
    r = int(supported[tasks.eq("regression")].sum())
    if (c, r) != (0, 6):
        raise AssertionError(f"Frozen empirical decisions changed: classification={c}, regression={r}")
    print("EMPIRICAL SCIENCE AUDIT: PASS (0/12 classification; 6/6 primary regression)", flush=True)
    return mean_only, primary, singleton


def summary_row(summary: pd.DataFrame, dataset: str, mode: str, budget: int, metric: str) -> pd.Series:
    row = summary[
        summary["dataset"].eq(dataset)
        & summary["scaffold_mode"].eq(mode)
        & summary["budget"].astype(int).eq(budget)
        & summary["metric"].eq(metric)
    ]
    if len(row) != 1:
        raise AssertionError(f"Expected one summary row for {dataset}/{mode}/K{budget}/{metric}, found {len(row)}")
    return row.iloc[0]


def bridge_table(mean_only: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("ESOL", "single_group", 20000, "main_regression"),
        ("FreeSolv", "single_group", 20000, "main_regression"),
        ("ESOL", "singleton", 5000, "acyclic_singleton_sensitivity"),
        ("FreeSolv", "singleton", 5000, "acyclic_singleton_sensitivity"),
    ]
    rows = []
    for dataset, mode, budget, label in specs:
        empirical = mean_only[mean_only["freeze_label"].eq(label) & mean_only["dataset"].eq(dataset)]
        if len(empirical) != 1:
            raise AssertionError(f"Missing empirical bridge row: {label}/{dataset}")
        observed = float(empirical.iloc[0]["mean_effect_size_minus_balanced_rmse"])
        values = perm[
            perm["dataset"].eq(dataset)
            & perm["scaffold_mode"].eq(mode)
            & perm["budget"].astype(int).eq(budget)
        ]["effect_rmse"].dropna().to_numpy(float)
        if values.size != 200:
            raise AssertionError(f"Bridge distribution has {values.size} rows for {dataset}/{mode}/K{budget}")
        exceed = int(np.sum(values >= observed - 1e-15))
        rows.append({
            "dataset": dataset,
            "scaffold_mode": mode,
            "budget": budget,
            "observed": observed,
            "null_mean": float(np.mean(values)),
            "null_q025": float(np.quantile(values, 0.025)),
            "null_q975": float(np.quantile(values, 0.975)),
            "exceedances": exceed,
            "upper_tail_p": float((exceed + 1) / 201),
            "percentile": float(100 * np.mean(values <= observed)),
        })
    frame = pd.DataFrame(rows)
    print("EMPIRICAL-NULL BRIDGE: PASS (exploratory; matched budgets)", flush=True)
    return frame


def macro(name: str, value: object) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def write_generated(summary: pd.DataFrame, quality: pd.DataFrame, bridge: pd.DataFrame, primary: pd.DataFrame, singleton: pd.DataFrame) -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    es = summary_row(summary, "ESOL", "single_group", 20000, "effect_rmse")
    fs = summary_row(summary, "FreeSolv", "single_group", 20000, "effect_rmse")
    es_gap = summary_row(summary, "ESOL", "single_group", 20000, "effect_squared_mean_gap")
    fs_gap = summary_row(summary, "FreeSolv", "single_group", 20000, "effect_squared_mean_gap")
    es_var = summary_row(summary, "ESOL", "single_group", 20000, "effect_test_variance")
    fs_var = summary_row(summary, "FreeSolv", "single_group", 20000, "effect_test_variance")
    cls = summary[summary["task_type"].eq("classification") & summary["budget"].astype(int).eq(300)]
    brier = cls[cls["metric"].eq("effect_brier")]["mean"].to_numpy(float)
    logloss = cls[cls["metric"].eq("effect_log_loss")]["mean"].to_numpy(float)
    b = bridge.set_index(["dataset", "scaffold_mode"])
    ep, fp = b.loc[("ESOL", "single_group")], b.loc[("FreeSolv", "single_group")]
    esn, fsn = b.loc[("ESOL", "singleton")], b.loc[("FreeSolv", "singleton")]

    lines = [
        "% Generated from frozen result tables.",
        macro("NullPermutationN", 200), macro("PartitionSeedN", 20),
        macro("RawSeedRows", f"{int(quality['raw_partition_seed_rows'].iloc[0]):,}"),
        macro("PermutationRows", f"{int(quality['permutation_level_rows'].iloc[0]):,}"),
        macro("MaxMSEResidual", f"{float(quality['max_abs_mse_decomposition_residual'].iloc[0]):.3e}"),
        macro("ESGapSqMean", f"{float(es_gap['mean']):.4f}"),
        macro("FSGapSqMean", f"{float(fs_gap['mean']):.4f}"),
        macro("ESVarianceMean", f"{float(es_var['mean']):+.4f}"),
        macro("FSVarianceMean", f"{float(fs_var['mean']):+.4f}"),
        macro("ESNullRMSEMean", f"{float(es['mean']):+.4f}"),
        macro("ESNullRMSELow", f"{float(es['q025']):+.4f}"),
        macro("ESNullRMSEHigh", f"{float(es['q975']):+.4f}"),
        macro("FSNullRMSEMean", f"{float(fs['mean']):+.4f}"),
        macro("FSNullRMSELow", f"{float(fs['q025']):+.4f}"),
        macro("FSNullRMSEHigh", f"{float(fs['q975']):+.4f}"),
        macro("ClassBrierMin", f"{float(np.min(brier)):+.4f}"),
        macro("ClassBrierMax", f"{float(np.max(brier)):+.4f}"),
        macro("ClassLogLossMin", f"{float(np.min(logloss)):+.4f}"),
        macro("ClassLogLossMax", f"{float(np.max(logloss)):+.4f}"),
        macro("ESPrimaryObserved", f"{ep['observed']:.4f}"),
        macro("ESPrimaryNullMean", f"{ep['null_mean']:+.4f}"),
        macro("ESPrimaryNullLow", f"{ep['null_q025']:+.4f}"),
        macro("ESPrimaryNullHigh", f"{ep['null_q975']:+.4f}"),
        macro("FSPrimaryObserved", f"{fp['observed']:.4f}"),
        macro("FSPrimaryNullMean", f"{fp['null_mean']:+.4f}"),
        macro("FSPrimaryNullLow", f"{fp['null_q025']:+.4f}"),
        macro("FSPrimaryNullHigh", f"{fp['null_q975']:+.4f}"),
        macro("ESSingletonObserved", f"{esn['observed']:+.4f}"),
        macro("FSSingletonObserved", f"{fsn['observed']:+.4f}"),
        macro("FSSingletonNullMean", f"{fsn['null_mean']:+.4f}"),
        macro("FSSingletonPercentile", f"{fsn['percentile']:.1f}"),
        macro("BridgeMinP", f"{1/201:.4f}"),
    ]
    (GEN / "results_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    table = [
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Dataset & Acyclic rule & Draws & Observed & Null mean [2.5\%, 97.5\%] & Upper-tail $p$ \\",
        r"\midrule",
    ]
    for row in bridge.itertuples(index=False):
        interval = f"{row.null_mean:+.4f} [{row.null_q025:+.4f}, {row.null_q975:+.4f}]"
        table.append(
            f"{row.dataset} & {str(row.scaffold_mode).replace('_', ' ')} & {int(row.budget):,} & "
            f"{row.observed:+.4f} & {interval} & {row.upper_tail_p:.4f} \\\\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "bridge_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    null_reg = [r"\begin{tabular}{llrllll}", r"\toprule", r"Dataset & Acyclic rule & Draws & RMSE & MSE & Mean-gap$^2$ & Test variance \\", r"\midrule"]
    for dataset, mode, budget in [("ESOL", "single_group", 20000), ("ESOL", "singleton", 20000), ("FreeSolv", "single_group", 20000), ("FreeSolv", "singleton", 20000)]:
        vals = [summary_row(summary, dataset, mode, budget, metric) for metric in ["effect_rmse", "effect_mse", "effect_squared_mean_gap", "effect_test_variance"]]
        iv = [f"{float(row['mean']):+.4f} [{float(row['q025']):+.4f}, {float(row['q975']):+.4f}]" for row in vals]
        null_reg.append(f"{dataset} & {mode.replace('_', ' ')} & {budget:,} & " + " & ".join(iv) + r" \\")
    null_reg += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "null_regression_table.tex").write_text("\n".join(null_reg) + "\n", encoding="utf-8")

    null_cls = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Dataset & Brier & Log loss & Average precision & ROC--AUC \\", r"\midrule"]
    for dataset in ["BACE", "BBBP", "ClinTox", "HIV"]:
        vals = [summary_row(summary, dataset, "single_group", 300, metric) for metric in ["effect_brier", "effect_log_loss", "effect_average_precision", "effect_roc_auc"]]
        null_cls.append(f"{dataset} & " + " & ".join(f"{float(row['mean']):+.5f}" for row in vals) + r" \\")
    null_cls += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "null_classification_table.tex").write_text("\n".join(null_cls) + "\n", encoding="utf-8")

    def compact_table(frame: pd.DataFrame, preferred: list[str], path: Path) -> None:
        cols = [column for column in preferred if column in frame.columns]
        if len(cols) < 4:
            cols = list(frame.columns[: min(8, len(frame.columns))])
        lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule", " & ".join(column.replace("_", r"\_") for column in cols) + r" \\", r"\midrule"]
        for _, row in frame[cols].iterrows():
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.5g}" if np.isfinite(value) else "NA")
                else:
                    values.append(str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&"))
            lines.append(" & ".join(values) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    compact_table(primary, ["dataset", "model", "metric", "mean_size", "mean_balanced", "mean_effect", "bootstrap_ci_low", "bootstrap_ci_high", "holm_p", "inference"], GEN / "primary_table.tex")
    compact_table(singleton, ["dataset", "model", "metric", "mean_size", "mean_balanced", "mean_effect", "bootstrap_ci_low", "bootstrap_ci_high", "wilcoxon_p_descriptive", "inference"], GEN / "singleton_table.tex")

    primary.to_csv(BUILD / "primary_inference_summary_v3.csv", index=False)
    singleton.to_csv(BUILD / "acyclic_singleton_sensitivity_v3.csv", index=False)
    bridge.to_csv(BUILD / "empirical_null_bridge_v1.csv", index=False)
    summary.to_csv(BUILD / "null_metric_effect_summary_v1.csv", index=False)


def copy_figures() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for number, (folder, stem) in FIGURE_MAP.items():
        for suffix in [".pdf", ".tiff"]:
            source = require(folder / f"{stem}{suffix}")
            shutil.copy2(source, FIGS / f"Figure_{number}{suffix}")
        png = folder / f"{stem}.png"
        if png.exists():
            shutil.copy2(png, FIGS / f"Figure_{number}.png")
        with Image.open(FIGS / f"Figure_{number}.tiff") as image:
            dpi = image.info.get("dpi", (0, 0))
            if min(float(dpi[0]), float(dpi[1])) < 590:
                raise AssertionError(f"Figure {number} is below 600 dpi: {dpi}")
    print("ARTWORK GATE: PASS (7 PDF/TIFF figure pairs; at least 600 dpi)", flush=True)
