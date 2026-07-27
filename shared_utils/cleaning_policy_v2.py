"""Audited raw-to-clean molecule policy for the Paper 1 rebuild."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shared_utils.raw_audit_v2 import canonicalize_smiles


@dataclass(frozen=True)
class CleaningResult:
    clean: pd.DataFrame
    group_decisions: pd.DataFrame
    row_lineage: pd.DataFrame
    summary: dict


def dataframe_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_clean_dataset_v2(
    raw: pd.DataFrame,
    *,
    dataset: str,
    smiles_col: str,
    target_col: str,
    task_type: str,
) -> CleaningResult:
    required = {smiles_col, target_col}
    missing = required.difference(raw.columns)
    if missing:
        raise KeyError(f"Missing raw columns for {dataset}: {sorted(missing)}")

    work = raw[[smiles_col, target_col]].copy()
    work.columns = ["source_smiles", "source_target"]
    work["raw_row_index"] = work.index.astype(int)
    work["target_numeric"] = pd.to_numeric(work["source_target"], errors="coerce")
    canonical = work["source_smiles"].map(canonicalize_smiles)
    work["canonical_smiles"] = canonical.map(lambda item: item[0])
    work["smiles_status"] = canonical.map(lambda item: item[1])

    valid_mask = work["canonical_smiles"].notna() & work["target_numeric"].notna()
    valid = work.loc[valid_mask].copy()

    clean_rows: list[dict] = []
    decision_rows: list[dict] = []
    lineage_rows: list[dict] = []

    for _, row in work.loc[~valid_mask].iterrows():
        reason = (
            "exclude_missing_or_invalid_smiles"
            if row["smiles_status"] != "valid"
            else "exclude_missing_or_nonnumeric_target"
        )
        lineage_rows.append(
            {
                "dataset": dataset,
                "raw_row_index": int(row["raw_row_index"]),
                "source_smiles": str(row["source_smiles"]),
                "source_target": str(row["source_target"]),
                "canonical_smiles": "",
                "action": reason,
                "final_target": np.nan,
            }
        )

    for canonical_smiles, group in valid.groupby("canonical_smiles", sort=True):
        targets = group["target_numeric"].to_numpy(dtype=float)
        unique_targets = np.unique(targets)
        n_rows = int(len(group))
        target_min = float(np.min(targets))
        target_max = float(np.max(targets))
        target_mean = float(np.mean(targets))
        target_std = float(np.std(targets, ddof=0))
        target_spread = float(np.ptp(targets))

        keep = True
        if task_type == "classification":
            if not set(unique_targets.tolist()).issubset({0.0, 1.0}):
                raise ValueError(
                    f"{dataset} has classification values outside 0/1 for {canonical_smiles}"
                )
            if len(unique_targets) > 1:
                keep = False
                action = "exclude_conflicting_classification_labels"
                final_target = np.nan
            elif n_rows > 1:
                action = "collapse_consistent_classification_duplicates"
                final_target = float(unique_targets[0])
            else:
                action = "keep_unique"
                final_target = float(unique_targets[0])
        elif task_type == "regression":
            final_target = target_mean
            action = "aggregate_regression_duplicates_by_mean" if n_rows > 1 else "keep_unique"
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        decision = {
            "dataset": dataset,
            "canonical_smiles": str(canonical_smiles),
            "task_type": task_type,
            "n_source_rows": n_rows,
            "raw_row_indices": ",".join(str(int(x)) for x in group["raw_row_index"]),
            "n_unique_targets": int(len(unique_targets)),
            "source_target_values": "|".join(f"{x:.12g}" for x in unique_targets),
            "target_min": target_min,
            "target_max": target_max,
            "target_mean": target_mean,
            "target_std": target_std,
            "target_spread": target_spread,
            "decision": action,
            "included_in_clean_v2": bool(keep),
            "final_target": final_target,
        }
        decision_rows.append(decision)

        if keep:
            clean_rows.append(
                {
                    "canonical_smiles": str(canonical_smiles),
                    "target": float(final_target),
                    "n_source_rows": n_rows,
                    "target_min": target_min,
                    "target_max": target_max,
                    "target_std": target_std,
                    "target_spread": target_spread,
                    "cleaning_decision": action,
                }
            )

        for _, source_row in group.iterrows():
            lineage_rows.append(
                {
                    "dataset": dataset,
                    "raw_row_index": int(source_row["raw_row_index"]),
                    "source_smiles": str(source_row["source_smiles"]),
                    "source_target": str(source_row["source_target"]),
                    "canonical_smiles": str(canonical_smiles),
                    "action": action,
                    "final_target": final_target,
                }
            )

    clean = pd.DataFrame(clean_rows).sort_values("canonical_smiles").reset_index(drop=True)
    decisions = pd.DataFrame(decision_rows).sort_values("canonical_smiles").reset_index(drop=True)
    lineage = pd.DataFrame(lineage_rows).sort_values("raw_row_index").reset_index(drop=True)

    if clean["canonical_smiles"].duplicated().any():
        raise AssertionError(f"Duplicate canonical molecules remain in {dataset} clean_v2")
    if clean[["canonical_smiles", "target"]].isna().any().any():
        raise AssertionError(f"Missing clean values remain in {dataset} clean_v2")
    if task_type == "classification" and not clean["target"].isin([0.0, 1.0]).all():
        raise AssertionError(f"Non-binary target remains in {dataset} clean_v2")

    excluded_conflicts = decisions["decision"].eq("exclude_conflicting_classification_labels")
    aggregated_regression = decisions["decision"].eq("aggregate_regression_duplicates_by_mean")
    collapsed_classification = decisions["decision"].eq("collapse_consistent_classification_duplicates")
    summary = {
        "dataset": dataset,
        "task_type": task_type,
        "raw_n_rows": int(len(work)),
        "valid_raw_rows": int(valid_mask.sum()),
        "invalid_or_missing_rows": int((~valid_mask).sum()),
        "valid_unique_canonical_groups": int(decisions.shape[0]),
        "excluded_conflicting_classification_groups": int(excluded_conflicts.sum()),
        "excluded_conflicting_classification_rows": int(decisions.loc[excluded_conflicts, "n_source_rows"].sum()),
        "collapsed_consistent_classification_groups": int(collapsed_classification.sum()),
        "aggregated_regression_groups": int(aggregated_regression.sum()),
        "clean_v2_n_rows": int(len(clean)),
        "clean_v2_sha256": dataframe_sha256(clean),
    }
    return CleaningResult(clean, decisions, lineage, summary)
