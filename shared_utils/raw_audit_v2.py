"""Raw molecular-data provenance and duplicate-target audit helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_smiles(value: object) -> tuple[str | None, str]:
    if pd.isna(value) or not str(value).strip():
        return None, "missing"
    mol = Chem.MolFromSmiles(str(value).strip())
    if mol is None:
        return None, "invalid"
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), "valid"


def duplicate_group_rows(
    frame: pd.DataFrame,
    *,
    dataset: str,
    canonical_col: str,
    target_col: str,
) -> list[dict]:
    rows: list[dict] = []
    valid = frame.loc[frame[canonical_col].notna()].copy()
    for canonical, group in valid.groupby(canonical_col, sort=True, dropna=False):
        if len(group) < 2:
            continue
        values = pd.to_numeric(group[target_col], errors="coerce").dropna().to_numpy(dtype=float)
        unique_values = sorted(set(float(value) for value in values))
        rows.append(
            {
                "dataset": dataset,
                "canonical_smiles": str(canonical),
                "n_raw_rows": int(len(group)),
                "raw_row_indices": ",".join(str(int(idx)) for idx in group.index),
                "n_numeric_targets": int(len(values)),
                "n_unique_targets": int(len(unique_values)),
                "target_values": "|".join(f"{value:.12g}" for value in unique_values),
                "target_min": float(np.min(values)) if len(values) else float("nan"),
                "target_max": float(np.max(values)) if len(values) else float("nan"),
                "target_spread": float(np.ptp(values)) if len(values) else float("nan"),
                "has_target_conflict": bool(len(unique_values) > 1),
            }
        )
    return rows


def processed_set_audit(
    path: Path,
    raw_valid: pd.DataFrame,
) -> dict:
    if not path.exists():
        return {
            "processed_file_exists": False,
            "processed_n_rows": np.nan,
            "processed_n_unique_canonical": np.nan,
            "processed_duplicate_canonical_groups": np.nan,
            "processed_conflicting_target_groups": np.nan,
            "raw_valid_unique_not_in_processed": np.nan,
            "processed_unique_not_in_raw_valid": np.nan,
            "processed_matches_raw_valid_unique_set": False,
        }
    processed = pd.read_csv(path, keep_default_na=False, low_memory=False)
    required = {"canonical_smiles", "target"}
    missing = required.difference(processed.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    processed["target_numeric"] = pd.to_numeric(processed["target"], errors="coerce")
    grouped = processed.groupby("canonical_smiles", dropna=False)["target_numeric"]
    sizes = grouped.size()
    conflicts = grouped.nunique(dropna=False)
    raw_set = set(raw_valid["canonical_smiles_v2"].dropna().astype(str))
    processed_set = set(processed["canonical_smiles"].astype(str))
    return {
        "processed_file_exists": True,
        "processed_n_rows": int(len(processed)),
        "processed_n_unique_canonical": int(len(processed_set)),
        "processed_duplicate_canonical_groups": int((sizes > 1).sum()),
        "processed_conflicting_target_groups": int((conflicts > 1).sum()),
        "raw_valid_unique_not_in_processed": int(len(raw_set - processed_set)),
        "processed_unique_not_in_raw_valid": int(len(processed_set - raw_set)),
        "processed_matches_raw_valid_unique_set": bool(raw_set == processed_set),
    }
