"""Canonical scaffold identity utilities for the Paper 1 rebuild."""

from __future__ import annotations

import hashlib
from typing import Literal

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ACYCLIC_SCAFFOLD = "__ACYCLIC__"
INVALID_SCAFFOLD = "__INVALID__"


def generate_scaffold_v2(smiles: str) -> str:
    """Return a non-empty Bemis--Murcko scaffold identifier."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return INVALID_SCAFFOLD
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False,
    )
    return scaffold if scaffold else ACYCLIC_SCAFFOLD


def prepare_scaffold_frame(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    *,
    acyclic_mode: Literal["single_group", "singleton"] = "single_group",
) -> pd.DataFrame:
    """Recompute scaffolds and fail on NaN, empty, or invalid identifiers."""
    if smiles_col not in df.columns:
        raise KeyError(f"Missing SMILES column: {smiles_col}")
    out = df.copy().reset_index(drop=True)
    out["scaffold"] = out[smiles_col].map(generate_scaffold_v2)
    if acyclic_mode == "singleton":
        mask = out["scaffold"].eq(ACYCLIC_SCAFFOLD)
        out.loc[mask, "scaffold"] = [
            f"{ACYCLIC_SCAFFOLD}:{idx}" for idx in out.index[mask]
        ]
    elif acyclic_mode != "single_group":
        raise ValueError(f"Unsupported acyclic_mode: {acyclic_mode}")
    if out["scaffold"].isna().any():
        raise AssertionError("Scaffold column contains NaN values.")
    if out["scaffold"].astype(str).str.len().eq(0).any():
        raise AssertionError("Scaffold column contains empty strings.")
    if out["scaffold"].eq(INVALID_SCAFFOLD).any():
        bad = int(out["scaffold"].eq(INVALID_SCAFFOLD).sum())
        raise ValueError(f"{bad} invalid molecules reached split generation.")
    return out


def partition_hash(
    df: pd.DataFrame,
    split_col: str,
    smiles_col: str = "canonical_smiles",
) -> str:
    """Hash a partition from sorted test-set row identities and SMILES."""
    test = df.loc[df[split_col].eq("test"), [smiles_col]]
    payload = "\n".join(
        f"{idx}\t{smiles}"
        for idx, smiles in sorted(
            zip(test.index.astype(int), test[smiles_col].astype(str)),
            key=lambda item: item[0],
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_valid_split(
    df: pd.DataFrame,
    split_col: str,
    *,
    require_scaffold_disjoint: bool,
) -> None:
    """Fail on incomplete assignment or scaffold leakage."""
    if split_col not in df.columns:
        raise KeyError(f"Missing split column: {split_col}")
    labels = set(df[split_col].astype(str).unique())
    if labels != {"train", "test"}:
        raise AssertionError(
            f"Unexpected or incomplete labels in {split_col}: {labels}"
        )
    assigned = int(df[split_col].isin(["train", "test"]).sum())
    if assigned != len(df):
        raise AssertionError("Every row must be assigned exactly once.")
    if require_scaffold_disjoint:
        train_scaffolds = set(
            df.loc[df[split_col].eq("train"), "scaffold"]
        )
        test_scaffolds = set(
            df.loc[df[split_col].eq("test"), "scaffold"]
        )
        shared = train_scaffolds.intersection(test_scaffolds)
        if shared:
            raise AssertionError(
                f"Scaffold leakage detected: {sorted(shared)[:5]}"
            )
