"""Manifest and diagnostic helpers for audited molecular partitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from shared_utils.scaffold_identity import ACYCLIC_SCAFFOLD


def split_manifest_rows(
    df: pd.DataFrame,
    *,
    dataset: str,
    protocol: str,
    partition_seed: int | None,
    split_col: str,
    partition_hash_value: str,
    smiles_col: str = "canonical_smiles",
    target_col: str = "target",
) -> list[dict]:
    rows: list[dict] = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "partition_seed": partition_seed,
                "partition_hash": partition_hash_value,
                "row_index": int(idx),
                "canonical_smiles": str(row[smiles_col]),
                "scaffold": str(row["scaffold"]),
                "target": float(row[target_col]),
                "assignment": str(row[split_col]),
            }
        )
    return rows


def duplicate_target_audit(df: pd.DataFrame, dataset: str) -> dict:
    grouped = df.groupby("canonical_smiles", dropna=False)["target"]
    sizes = grouped.size()
    target_nunique = grouped.nunique(dropna=False)
    duplicate_groups = sizes[sizes > 1]
    conflicting = target_nunique[target_nunique > 1]
    return {
        "dataset": dataset,
        "n_rows": int(len(df)),
        "n_unique_canonical_smiles": int(
            df["canonical_smiles"].nunique(dropna=False)
        ),
        "n_duplicate_smiles_groups": int(len(duplicate_groups)),
        "n_duplicate_rows_beyond_first": (
            int((duplicate_groups - 1).sum())
            if len(duplicate_groups)
            else 0
        ),
        "n_conflicting_target_groups": int(len(conflicting)),
    }


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("nan")


def _test_scaffold_metrics(test: pd.DataFrame) -> dict:
    counts = test["scaffold"].value_counts(dropna=False)
    if counts.empty:
        return {
            "n_test_scaffolds": 0,
            "largest_test_scaffold_fraction": float("nan"),
            "top5_test_scaffold_fraction": float("nan"),
            "test_scaffold_hhi": float("nan"),
            "effective_test_scaffolds": float("nan"),
            "test_scaffold_gini": float("nan"),
        }
    fractions = counts.to_numpy(dtype=float) / float(counts.sum())
    sorted_counts = np.sort(counts.to_numpy(dtype=float))
    cumulative = np.cumsum(sorted_counts)
    n = len(sorted_counts)
    gini = (
        (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
        if n > 0 and cumulative[-1] > 0
        else float("nan")
    )
    hhi = float(np.sum(fractions ** 2))
    return {
        "n_test_scaffolds": int(len(counts)),
        "largest_test_scaffold_fraction": float(fractions.max()),
        "top5_test_scaffold_fraction": float(np.sort(fractions)[-5:].sum()),
        "test_scaffold_hhi": hhi,
        "effective_test_scaffolds": float(1.0 / hhi) if hhi > 0 else float("nan"),
        "test_scaffold_gini": float(gini),
    }


def summarize_partition(
    df: pd.DataFrame,
    *,
    dataset: str,
    task_type: str,
    protocol: str,
    partition_seed: int | None,
    split_col: str,
    meta: dict,
) -> dict:
    train = df.loc[df[split_col].eq("train")].copy()
    test = df.loc[df[split_col].eq("test")].copy()
    train_y = train["target"].to_numpy(dtype=float)
    test_y = test["target"].to_numpy(dtype=float)
    target_n = int(round(len(df) * 0.2))
    shared = set(train["scaffold"]).intersection(set(test["scaffold"]))
    row = {
        "dataset": dataset,
        "task_type": task_type,
        "protocol": protocol,
        "partition_seed": partition_seed,
        "partition_hash": meta["partition_hash"],
        "n_total": int(len(df)),
        "target_test_n": target_n,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_fraction": float(len(test) / len(df)),
        "size_deviation_from_target": float(
            abs(len(test) - target_n) / max(target_n, 1)
        ),
        "within_size_constraints": meta.get(
            "within_size_constraints",
            np.nan,
        ),
        "n_scaffolds_total": int(df["scaffold"].nunique(dropna=False)),
        "n_shared_scaffolds": int(len(shared)),
        "n_acyclic_total": int(
            df["scaffold"].astype(str).str.startswith(ACYCLIC_SCAFFOLD).sum()
        ),
        "n_acyclic_train": int(
            train["scaffold"].astype(str).str.startswith(ACYCLIC_SCAFFOLD).sum()
        ),
        "n_acyclic_test": int(
            test["scaffold"].astype(str).str.startswith(ACYCLIC_SCAFFOLD).sum()
        ),
        "train_target_mean": float(np.mean(train_y)),
        "test_target_mean": float(np.mean(test_y)),
        "abs_target_mean_gap": float(abs(np.mean(test_y) - np.mean(train_y))),
        "train_target_sd": float(np.std(train_y, ddof=0)),
        "test_target_sd": float(np.std(test_y, ddof=0)),
        "train_target_q05": _quantile(train_y, 0.05),
        "train_target_q25": _quantile(train_y, 0.25),
        "train_target_q50": _quantile(train_y, 0.50),
        "train_target_q75": _quantile(train_y, 0.75),
        "train_target_q95": _quantile(train_y, 0.95),
        "test_target_q05": _quantile(test_y, 0.05),
        "test_target_q25": _quantile(test_y, 0.25),
        "test_target_q50": _quantile(test_y, 0.50),
        "test_target_q75": _quantile(test_y, 0.75),
        "test_target_q95": _quantile(test_y, 0.95),
        "target_wasserstein": float(wasserstein_distance(train_y, test_y)),
        "target_ks_statistic": float(
            ks_2samp(train_y, test_y, alternative="two-sided").statistic
        ),
        "search_objective": meta.get("objective"),
        "search_objective_value": meta.get("objective_value", np.nan),
        "search_size_component": meta.get("size_component", np.nan),
        "search_mean_component": meta.get("mean_component", np.nan),
        "search_n_trials": meta.get("n_trials", np.nan),
    }
    row.update(_test_scaffold_metrics(test))
    return row
