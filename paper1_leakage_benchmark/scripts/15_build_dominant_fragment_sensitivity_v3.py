from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
SOURCE_DIR = PAPER_DIR / "data" / "processed_v2"
OUT_DIR = PAPER_DIR / "data" / "dominant_fragment_sensitivity_v3"
RESULT_DIR = PAPER_DIR / "results" / "dominant_fragment_sensitivity_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ("BACE", "BBBP", "ClinTox", "HIV")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fragment_key(mol: Chem.Mol) -> tuple[int, int, str]:
    heavy = int(mol.GetNumHeavyAtoms())
    carbon = int(sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()))
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return (heavy, carbon, smiles)


def dominant_fragment(smiles: str) -> tuple[str, int, str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Could not parse source canonical SMILES: {smiles}")
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if not frags:
        raise ValueError(f"No fragments found for: {smiles}")
    ranked = sorted(frags, key=fragment_key, reverse=True)
    selected = ranked[0]
    selected_smiles = Chem.MolToSmiles(
        selected, canonical=True, isomericSmiles=True
    )
    extras = [
        Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True)
        for frag in ranked[1:]
    ]
    return selected_smiles, len(frags), "|".join(extras)


def main() -> None:
    summary_rows: list[dict] = []
    group_rows: list[dict] = []
    lineage_rows: list[dict] = []

    for dataset in DATASETS:
        source_path = SOURCE_DIR / f"{dataset.lower()}_clean_v2.csv"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source = pd.read_csv(
            source_path,
            keep_default_na=False,
            low_memory=False,
        ).reset_index(drop=True)
        required = {"canonical_smiles", "target"}
        missing = required.difference(source.columns)
        if missing:
            raise KeyError(f"{source_path} missing {sorted(missing)}")
        source["target"] = pd.to_numeric(source["target"], errors="raise")
        if source["canonical_smiles"].duplicated().any():
            raise AssertionError(f"Source clean_v2 has duplicate canonical SMILES: {dataset}")
        labels = set(source["target"].astype(int).unique().tolist())
        if not labels.issubset({0, 1}):
            raise AssertionError(f"Non-binary labels in {dataset}: {sorted(labels)}")

        mapped_rows: list[dict] = []
        for idx, row in source.iterrows():
            dominant_smiles, n_fragments, extras = dominant_fragment(
                str(row["canonical_smiles"])
            )
            mapped_rows.append(
                {
                    "source_row_index": int(idx),
                    "source_canonical_smiles": str(row["canonical_smiles"]),
                    "target": int(row["target"]),
                    "dominant_fragment_smiles": dominant_smiles,
                    "n_fragments": int(n_fragments),
                    "extra_fragments": extras,
                    "representation_changed": bool(
                        dominant_smiles != str(row["canonical_smiles"])
                    ),
                }
            )
        mapped = pd.DataFrame(mapped_rows)

        clean_rows: list[dict] = []
        conflict_groups = 0
        conflict_source_rows = 0
        consistent_duplicate_groups = 0
        consistent_duplicate_source_rows = 0

        grouped = mapped.groupby("dominant_fragment_smiles", sort=True)
        for dominant_smiles, group in grouped:
            unique_labels = sorted(group["target"].astype(int).unique().tolist())
            if len(unique_labels) > 1:
                decision = "exclude_conflicting_labels"
                conflict_groups += 1
                conflict_source_rows += len(group)
                retained = False
            else:
                decision = (
                    "collapse_consistent_duplicates"
                    if len(group) > 1
                    else "retain_singleton"
                )
                if len(group) > 1:
                    consistent_duplicate_groups += 1
                    consistent_duplicate_source_rows += len(group)
                retained = True
                clean_rows.append(
                    {
                        "canonical_smiles": dominant_smiles,
                        "target": int(unique_labels[0]),
                        "source_row_count": int(len(group)),
                        "source_multifragment_count": int(
                            group["n_fragments"].gt(1).sum()
                        ),
                        "source_representation_changed_count": int(
                            group["representation_changed"].sum()
                        ),
                        "source_row_indices": "|".join(
                            str(value)
                            for value in group["source_row_index"].astype(int).tolist()
                        ),
                    }
                )

            group_rows.append(
                {
                    "dataset": dataset,
                    "dominant_fragment_smiles": dominant_smiles,
                    "n_source_rows": int(len(group)),
                    "labels": "|".join(str(value) for value in unique_labels),
                    "decision": decision,
                }
            )
            for _, item in group.iterrows():
                lineage_rows.append(
                    {
                        "dataset": dataset,
                        "source_row_index": int(item["source_row_index"]),
                        "source_canonical_smiles": item["source_canonical_smiles"],
                        "source_target": int(item["target"]),
                        "dominant_fragment_smiles": dominant_smiles,
                        "n_fragments": int(item["n_fragments"]),
                        "extra_fragments": item["extra_fragments"],
                        "representation_changed": bool(item["representation_changed"]),
                        "group_decision": decision,
                        "retained_in_sensitivity": retained,
                    }
                )

        clean = pd.DataFrame(clean_rows).sort_values(
            "canonical_smiles", kind="mergesort"
        ).reset_index(drop=True)
        if clean.empty:
            raise AssertionError(f"No sensitivity rows retained for {dataset}")
        if clean["canonical_smiles"].duplicated().any():
            raise AssertionError(f"Duplicate dominant fragments remain: {dataset}")
        if set(clean["target"].astype(int).unique()).difference({0, 1}):
            raise AssertionError(f"Invalid target after dominant-fragment cleaning: {dataset}")
        for smiles in clean["canonical_smiles"].astype(str):
            if Chem.MolFromSmiles(smiles) is None:
                raise AssertionError(f"Unparseable retained dominant fragment: {dataset}")

        output_path = OUT_DIR / f"{dataset.lower()}_dominant_fragment_clean_v3.csv"
        clean.to_csv(output_path, index=False)

        summary_rows.append(
            {
                "dataset": dataset,
                "n_source_clean_v2_rows": int(len(source)),
                "n_source_multifragment_rows": int(mapped["n_fragments"].gt(1).sum()),
                "n_source_representation_changed": int(mapped["representation_changed"].sum()),
                "n_unique_dominant_fragments_before_conflict_exclusion": int(
                    mapped["dominant_fragment_smiles"].nunique()
                ),
                "n_conflict_groups_excluded": int(conflict_groups),
                "n_conflict_source_rows_excluded": int(conflict_source_rows),
                "n_consistent_duplicate_groups_collapsed": int(consistent_duplicate_groups),
                "n_consistent_duplicate_source_rows": int(consistent_duplicate_source_rows),
                "n_final_sensitivity_rows": int(len(clean)),
                "n_final_positive": int(clean["target"].astype(int).sum()),
                "n_final_negative": int(len(clean) - clean["target"].astype(int).sum()),
                "source_sha256": sha256_file(source_path),
                "output_sha256": sha256_file(output_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    groups = pd.DataFrame(group_rows)
    lineage = pd.DataFrame(lineage_rows)

    summary_path = RESULT_DIR / "dominant_fragment_cleaning_summary_v3.csv"
    groups_path = RESULT_DIR / "dominant_fragment_group_decisions_v3.csv"
    lineage_path = RESULT_DIR / "dominant_fragment_row_lineage_v3.csv"
    metadata_path = RESULT_DIR / "dominant_fragment_cleaning_metadata_v3.json"

    summary.to_csv(summary_path, index=False)
    groups.to_csv(groups_path, index=False)
    lineage.to_csv(lineage_path, index=False)
    metadata = {
        "protocol": "paper1_leakage_benchmark/DOMINANT_FRAGMENT_SENSITIVITY_PROTOCOL_V3.md",
        "datasets": list(DATASETS),
        "selection_rule": "max heavy atoms, then max carbon atoms, then canonical isomeric SMILES",
        "classification_conflict_policy": "exclude entire dominant-fragment group",
        "classification_consistent_duplicate_policy": "collapse to one record",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nDominant-fragment sensitivity cleaning summary:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(summary_path)
    print(groups_path)
    print(lineage_path)
    print(metadata_path)
    for dataset in DATASETS:
        print(OUT_DIR / f"{dataset.lower()}_dominant_fragment_clean_v3.csv")

    if not bool((summary["n_final_sensitivity_rows"] > 0).all()):
        raise AssertionError("At least one sensitivity dataset is empty")
    if not bool((summary["n_final_positive"] > 0).all()):
        raise AssertionError("At least one sensitivity dataset lost all positives")
    if not bool((summary["n_final_negative"] > 0).all()):
        raise AssertionError("At least one sensitivity dataset lost all negatives")

    print("\nDOMINANT FRAGMENT SENSITIVITY CLEANING V3 PASSED")


if __name__ == "__main__":
    main()
