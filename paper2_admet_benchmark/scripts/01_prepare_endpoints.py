from __future__ import annotations

"""Prepare public ADMET / molecular property endpoints for Paper 2.

This MVP implementation intentionally avoids a hard dependency on PyTDC.
Instead, it uses direct public CSV/URL loaders where possible. This keeps the
Windows environment lightweight and makes the exact data sources explicit.

Outputs:
- paper2_admet_benchmark/data/raw/<endpoint>_raw.*
- paper2_admet_benchmark/data/processed/<endpoint>_clean.csv
- paper2_admet_benchmark/data/manifests/endpoint_registry.csv
- paper2_admet_benchmark/results/tables/endpoint_cleaning_summary.csv
"""

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.chem_features import canonicalize_smiles

RDLogger.DisableLog("rdApp.warning")

TaskType = Literal["classification", "regression"]

PAPER_DIR = ROOT / "paper2_admet_benchmark"
RAW_DIR = PAPER_DIR / "data" / "raw"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
MANIFEST_DIR = PAPER_DIR / "data" / "manifests"
TABLE_DIR = PAPER_DIR / "results" / "tables"


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    url: str
    smiles_col: str
    target_col: str
    task_type: TaskType
    admet_category: str
    source: str = "MoleculeNet/DeepChem public CSV"
    citation_key: str = "wu2018moleculenet"
    source_note: str = "Public MoleculeNet/DeepChem dataset URL."
    include_in_mvp: bool = True


ENDPOINTS: dict[str, EndpointSpec] = {
    "BBBP": EndpointSpec(
        name="BBBP",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        smiles_col="smiles",
        target_col="p_np",
        task_type="classification",
        admet_category="permeability",
        source_note="Blood-brain barrier penetration dataset distributed through DeepChem/MoleculeNet.",
    ),
    "ClinTox": EndpointSpec(
        name="ClinTox",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        smiles_col="smiles",
        target_col="CT_TOX",
        task_type="classification",
        admet_category="toxicity",
        source_note="Clinical toxicity endpoint distributed through DeepChem/MoleculeNet. CT_TOX is used for the MVP.",
    ),
    "BACE": EndpointSpec(
        name="BACE",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        smiles_col="mol",
        target_col="Class",
        task_type="classification",
        admet_category="bioactivity_comparator",
        source_note="BACE inhibitor classification dataset. Included as a molecular ML comparator, not a core ADMET endpoint.",
    ),
    "HIV": EndpointSpec(
        name="HIV",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        smiles_col="smiles",
        target_col="HIV_active",
        task_type="classification",
        admet_category="bioactivity_comparator",
        source_note="HIV activity classification dataset. Included as a molecular ML comparator, not a core ADMET endpoint.",
    ),
    "ESOL": EndpointSpec(
        name="ESOL",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        smiles_col="smiles",
        target_col="measured log solubility in mols per litre",
        task_type="regression",
        admet_category="solubility",
        source_note="Delaney aqueous solubility dataset distributed through DeepChem/MoleculeNet.",
    ),
    "FreeSolv": EndpointSpec(
        name="FreeSolv",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        smiles_col="smiles",
        target_col="expt",
        task_type="regression",
        admet_category="hydration_free_energy",
        source_note="FreeSolv/SAMPL hydration free-energy dataset distributed through DeepChem/MoleculeNet.",
    ),
    "Lipophilicity": EndpointSpec(
        name="Lipophilicity",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        smiles_col="smiles",
        target_col="exp",
        task_type="regression",
        admet_category="lipophilicity",
        source_note="Lipophilicity dataset distributed through DeepChem/MoleculeNet.",
    ),
}


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MANIFEST_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)
        print("ensured", path)


def raw_suffix(url: str) -> str:
    return ".csv.gz" if url.endswith(".gz") else ".csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw(spec: EndpointSpec) -> Path:
    out_path = RAW_DIR / f"{spec.name.lower()}_raw{raw_suffix(spec.url)}"
    if out_path.exists():
        print(f"loading cached raw file: {out_path}")
        return out_path

    print(f"downloading {spec.name}: {spec.url}")
    response = requests.get(spec.url, timeout=60)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    print(f"saved raw file: {out_path}")
    return out_path


def resolve_duplicate_classification(group: pd.DataFrame) -> pd.Series | None:
    labels = group["target"].dropna().astype(int)
    if labels.empty:
        return None

    counts = labels.value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        # Ambiguous tie: remove this molecule.
        return None

    chosen = int(counts.idxmax())
    row = group.iloc[0].copy()
    row["target"] = chosen
    row["n_duplicate_records"] = int(len(group))
    row["duplicate_conflict"] = bool(len(counts) > 1)
    return row


def resolve_duplicate_regression(group: pd.DataFrame) -> pd.Series | None:
    values = pd.to_numeric(group["target"], errors="coerce").dropna()
    if values.empty:
        return None

    row = group.iloc[0].copy()
    row["target"] = float(values.median())
    row["n_duplicate_records"] = int(len(group))
    row["duplicate_conflict"] = bool(values.nunique() > 1)
    return row


def clean_endpoint(spec: EndpointSpec, raw_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(raw_path)
    missing_cols = [col for col in [spec.smiles_col, spec.target_col] if col not in raw.columns]
    if missing_cols:
        raise KeyError(f"{spec.name} missing expected columns: {missing_cols}. Available columns: {list(raw.columns)}")

    clean = raw[[spec.smiles_col, spec.target_col]].copy()
    clean.columns = ["smiles", "target"]
    clean["endpoint"] = spec.name
    clean["task_type"] = spec.task_type
    clean["admet_category"] = spec.admet_category

    n_raw = len(clean)
    clean = clean.dropna(subset=["smiles", "target"]).copy()
    n_after_dropna = len(clean)

    clean["canonical_smiles"] = clean["smiles"].map(canonicalize_smiles)
    clean = clean.dropna(subset=["canonical_smiles"]).copy()
    n_after_valid_smiles = len(clean)

    if spec.task_type == "classification":
        clean["target"] = pd.to_numeric(clean["target"], errors="coerce")
        clean = clean.dropna(subset=["target"]).copy()
        clean["target"] = clean["target"].astype(int)
        resolver = resolve_duplicate_classification
    else:
        clean["target"] = pd.to_numeric(clean["target"], errors="coerce")
        clean = clean.dropna(subset=["target"]).copy()
        clean["target"] = clean["target"].astype(float)
        resolver = resolve_duplicate_regression

    duplicate_groups = clean.groupby("canonical_smiles", sort=False)
    duplicate_conflict_count = 0
    resolved_rows = []
    removed_tie_count = 0

    for _, group in duplicate_groups:
        if len(group) > 1 and group["target"].nunique(dropna=True) > 1:
            duplicate_conflict_count += 1
        row = resolver(group)
        if row is None:
            removed_tie_count += 1
            continue
        resolved_rows.append(row)

    if not resolved_rows:
        raise ValueError(f"{spec.name}: no valid molecules after duplicate handling.")

    cleaned = pd.DataFrame(resolved_rows).reset_index(drop=True)
    cleaned = cleaned[
        [
            "endpoint",
            "task_type",
            "admet_category",
            "smiles",
            "canonical_smiles",
            "target",
            "n_duplicate_records",
            "duplicate_conflict",
        ]
    ]

    n_clean = len(cleaned)
    duplicates_removed = n_after_valid_smiles - n_clean

    target = cleaned["target"]
    if spec.task_type == "classification":
        positive_ratio = float(target.mean())
        target_mean = positive_ratio
    else:
        positive_ratio = None
        target_mean = float(target.mean())

    summary = {
        "endpoint": spec.name,
        "source": spec.source,
        "task_type": spec.task_type,
        "admet_category": spec.admet_category,
        "smiles_col": spec.smiles_col,
        "target_col": spec.target_col,
        "url": spec.url,
        "raw_file": str(raw_path.relative_to(ROOT)),
        "raw_sha256": sha256_file(raw_path),
        "n_raw": int(n_raw),
        "after_dropna": int(n_after_dropna),
        "after_valid_smiles": int(n_after_valid_smiles),
        "n_clean": int(n_clean),
        "duplicates_removed": int(duplicates_removed),
        "duplicate_conflict_count": int(duplicate_conflict_count),
        "removed_ambiguous_tie_count": int(removed_tie_count),
        "positive_ratio": positive_ratio,
        "target_mean": target_mean,
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "license_or_source_note": spec.source_note,
        "citation_key": spec.citation_key,
        "include_in_mvp": bool(spec.include_in_mvp),
        "exclusion_reason": "",
    }
    return cleaned, summary


def main() -> None:
    ensure_directories()

    summaries = []
    for name, spec in ENDPOINTS.items():
        print("\n==========", name, "==========")
        raw_path = download_raw(spec)
        cleaned, summary = clean_endpoint(spec, raw_path)

        out_path = PROCESSED_DIR / f"{name.lower()}_clean.csv"
        cleaned.to_csv(out_path, index=False)
        print(f"saved clean endpoint: {out_path} {cleaned.shape}")
        summaries.append(summary)

    registry = pd.DataFrame(summaries)
    registry_path = MANIFEST_DIR / "endpoint_registry.csv"
    summary_path = TABLE_DIR / "endpoint_cleaning_summary.csv"
    registry.to_csv(registry_path, index=False)
    registry.to_csv(summary_path, index=False)

    print("\nsaved", registry_path)
    print("saved", summary_path)
    print("\nEndpoint preparation complete.")
    print("Next: implement 02_featurize.py and 03_make_splits.py.")


if __name__ == "__main__":
    main()
