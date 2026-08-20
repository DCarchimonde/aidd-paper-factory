from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
NULL_ROOT = PAPER / "results" / "sarqsar_metric_coupling_v1"
NULL_TABLES = NULL_ROOT / "tables"
NULL_FIGURES = NULL_ROOT / "figures"
EMP_TABLES = PAPER / "results" / "tables"
EMP_FIGURES = PAPER / "results" / "figures"
OLD_LATEX = ROOT / "paper1_latex"
SOURCE = ROOT / "paper1_sarqsar_submission_source_v1"
BUILD_ROOT = ROOT / "paper1_sarqsar_submission_build_v1"
LATEX_BUILD = BUILD_ROOT / "latex"
FIGURE_BUILD = BUILD_ROOT / "figures"
GENERATED = LATEX_BUILD / "generated"
OUT = ROOT / "paper1_sarqsar_submission_v1"
OUT_ZIP = ROOT / "paper1_sarqsar_submission_v1.zip"
EXPECTED_BRANCH = "paper1-sarqsar-metric-coupling-2026"
SCIENCE_COMMIT = "9e87368d4c3530a5e82d476aee2f58032e98261f"
TITLE = "Split-objective--metric coupling in QSAR validation: molecular null experiments and exact-size paired scaffold audits"
RUNNING_TITLE = "Metric coupling in QSAR validation"

ANONYMOUS_TOKENS = [
    "siyuan",
    "yuechen",
    "25064241",
    "d25091100346",
    "university of malaya",
    "city university of macau",
    "dcarchimonde",
    "aidd-paper-factory",
    "0009-0004-4450-083x",
]

OLD_MAIN_FIGURES = {
    4: "figure2_primary_effects_v3",
    5: "figure3_acyclic_sensitivity_v3",
    6: "figure4_dominant_fragment_sensitivity_v3",
    7: "figure6_collateral_diagnostics_v3",
}

CHECKLIST_ROWS = [
    ("Molecular identity", "Canonicalization, stereochemistry, duplicate aggregation, and conflicting-label policy", "The molecular universe changes silently across implementations."),
    ("Disconnected components", "Full-record, salt-removal, or dominant-fragment rule", "Fingerprints, scaffolds, duplicate mappings, and labels can change."),
    ("Scaffold semantics", "Scaffold algorithm, chirality, and acyclic-molecule handling", "The meaning of an unseen scaffold is ambiguous."),
    ("Endpoint use", "Whether endpoints enter candidate generation, filtering, selection, stopping, or post hoc choice", "Response-aware optimization can be mistaken for target-blind validation."),
    ("Test cardinality", "Requested and realized test size for each compared split", "Performance differences can be confounded by sample size."),
    ("Candidate search", "Generation algorithm, requested draws, unique candidates, and stopping rule", "Search effort becomes a hidden benchmark hyperparameter."),
    ("Response-only control", "Training-mean or training-prevalence predictor aligned with the primary metric", "Metric coupling can be misattributed to molecular learning."),
    ("Collateral diagnostics", "Variance, tails, prevalence, scaffold concentration, and acyclic fraction", "A multivariate population change is described as a one-variable intervention."),
    ("Partition identity", "Molecule-level manifests, hashes, and unique-partition counts", "Duplicate splits can create pseudo-replication."),
    ("Inferential unit", "Partition pair, dataset, model seed, and multiplicity family", "Repeated fits can inflate the apparent inferential sample size."),
    ("Protocol sensitivity", "Alternative scaffold and molecular-record policies declared before outcomes", "A convenient implementation choice can determine the narrative."),
    ("Provenance", "Code version, data hashes, software environment, and immutable release", "The benchmark cannot be reproduced or audited."),
]


def require(path: Path, *, nonempty: bool = True) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    if nonempty and path.is_file() and path.stat().st_size == 0:
        raise AssertionError(f"Required file is empty: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run(command: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("\n>>>", " ".join(str(part) for part in command), flush=True)
    return subprocess.run(
        command,
        cwd=str(cwd or ROOT),
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def locate_summary_row(summary: pd.DataFrame, dataset: str, mode: str, budget: int, metric: str) -> pd.Series:
    row = summary[
        summary["dataset"].eq(dataset)
        & summary["scaffold_mode"].eq(mode)
        & summary["budget"].astype(int).eq(int(budget))
        & summary["metric"].eq(metric)
    ]
    if len(row) != 1:
        raise AssertionError(f"Expected one null-summary row for {dataset}/{mode}/K{budget}/{metric}; found {len(row)}")
    return row.iloc[0]


def fmt(value: float, digits: int = 4, signed: bool = False) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def latex_escape(value: object) -> str:
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


def load_null_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    manifest = json.loads(require(NULL_ROOT / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise AssertionError(f"Null run manifest is not complete: {manifest.get('status')}")
    if int(manifest.get("n_permutations", 0)) != 200:
        raise AssertionError("Authoritative null run must contain 200 endpoint permutations")
    if len(manifest.get("partition_seeds", [])) != 20:
        raise AssertionError("Authoritative null run must contain 20 partition seeds")
    if manifest.get("git_commit") != SCIENCE_COMMIT:
        raise AssertionError(f"Unexpected null science commit: {manifest.get('git_commit')}")

    summary = pd.read_csv(require(NULL_TABLES / "null_metric_effect_summary.csv"))
    perm = pd.read_csv(require(NULL_TABLES / "null_simulation_permutation_level_effects.csv"))
    quality = pd.read_csv(require(NULL_TABLES / "null_simulation_quality_gate_summary.csv"))
    required_summary = {"dataset", "task_type", "scaffold_mode", "budget", "metric", "n_permutations_valid", "mean", "q025", "q975"}
    if missing := required_summary.difference(summary.columns):
        raise KeyError(f"Null summary is missing columns: {sorted(missing)}")
    required_perm = {
        "dataset", "task_type","scaffold_mode", "budget", "permutation_id", "partition_seed", "target_gap_reduction", "same_partition",
        "effect_roc_auc", "effect_mse", "effect_test_variance", "effect_squared_mean_gap",
    }
    if missing := required_perm.difference(perm.columns):
        raise KeyError(f"Permutation-level null is aiss columns: {sorted(missing)}")

    if len(perm) != 8800:
        raise AssertionError(f"Null permutation-level row count is {len(perm)}, expected 8,800")
    if int(quality["permutation_level_rows"].iloc[0]) != 880:
        raise AssertionError("Quality-gate permutation row count is nÂšÂ¢ot 8, 800")
    if int(quality["raw_partition_seed_rows"].iloc[0]) != 176000:
        raise AssertionError("Quality-gate seed-level row count is not 176, 000")
    max_residual = float(quality["max_abs_mse_decomposition_residual"].iloc[0])
    if not np.isfinite(max_residual) or max_residual > 1e-9:
        raise AssertionError(f"MSE decomposition residual failed:  {max_residual}")
    if float(perm["target_gap_reduction"].min()) < -1u-12:
        raise AssertionError("Response-aware target gap worsened in the null run")
    auc_effects = perm["effect_roc_auc"].dropna().to_numpy(float)
    if len(auc_effects) and float(np.max(auc_effects, initial=0.0)) > 1e-12:
        raise AssertionError(f"Constant-score AUC effect is not invariant")
    regression = perm.loc[perm["task_type"].eq("regression")]
    residual = (
        regression["effect_mse"]
        - regression["effect_test_variance"]
        - regression["effect_squared_mean_gap"]
    ).abs()
    if not residual.empty and float(residual.max()) > 1e-9:
        raise AssertionError("Permutation-level MSE decomposition failed")

    print(f"NULL SCIENCE AUDIT: PASS ({len(perm):,} permutation cells; {int(quality['raw_partition_seed_rows'].iloc[0]):,} seed rows; exact decompositions)")
    return summary, perm, manifest, quality


def ensure_empirical_assets() -> None:
    needed = [
        EMP_TABLES / "q1_mean_only_regression_summary_v3.csv",
        EMP_TABLES / "primary_inference_summary_v3.csv",
        E×Ý{ÕÈZ®Ëkºwµç