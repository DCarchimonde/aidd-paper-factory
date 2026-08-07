from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.scaffold_identity import generate_scaffold_v2

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
DATA_DIR = PAPER_DIR / "data" / "processed_v2"
OUT_DIR = PAPER_DIR / "results" / "model_rerun_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ("BACE", "BBBP", "ClinTox", "HIV", "ESOL", "FreeSolv")
CLASSIFICATION = {"BACE", "BBBP", "ClinTox", "HIV"}
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fragment_key(mol: Chem.Mol) -> tuple[int, int, str]:
    heavy = int(mol.GetNumHeavyAtoms())
    carbon = int(sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()))
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return (heavy, carbon, smiles)


def choose_parent(mol: Chem.Mol) -> tuple[Chem.Mol, list[str]]:
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if not frags:
        raise ValueError("No fragments returned")
    ranked = sorted(frags, key=fragment_key, reverse=True)
    parent = ranked[0]
    extras = [
        Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True)
        for frag in ranked[1:]
    ]
    return parent, extras


def target_conflict(dataset: str, values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(float)
    if dataset in CLASSIFICATION:
        return len(np.unique(numeric.astype(int))) > 1
    return bool(np.ptp(numeric) > 1e-12)


def main() -> None:
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    extra_counter: Counter[str] = Counter()

    for dataset in DATASETS:
        path = DATA_DIR / f"{dataset.lower()}_clean_v2.csv"
        frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
        frame["target"] = pd.to_numeric(frame["target"], errors="raise")
        dataset_details: list[dict] = []

        for idx, row in frame.iterrows():
            smiles = str(row["canonical_smiles"])
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise AssertionError(f"Unexpected parse failure: {dataset} row {idx}")
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            parent, extras = choose_parent(mol)
            parent_smiles = Chem.MolToSmiles(
                parent, canonical=True, isomericSmiles=True
            )
            full_scaffold = generate_scaffold_v2(smiles)
            parent_scaffold = generate_scaffold_v2(parent_smiles)
            fp_full = FP.GetFingerprint(mol)
            fp_parent = FP.GetFingerprint(parent)
            tanimoto = float(DataStructs.TanimotoSimilarity(fp_full, fp_parent))
            for extra in extras:
                extra_counter[extra] += 1
            record = {
                "dataset": dataset,
                "row_index": int(idx),
                "canonical_smiles": smiles,
                "target": float(row["target"]),
                "n_fragments": int(len(frags)),
                "parent_smiles": parent_smiles,
                "extra_fragments": "|".join(extras),
                "full_scaffold": full_scaffold,
                "parent_scaffold": parent_scaffold,
                "scaffold_changed": bool(full_scaffold != parent_scaffold),
                "morgan_tanimoto_full_vs_parent": tanimoto,
            }
            detail_rows.append(record)
            dataset_details.append(record)

        d = pd.DataFrame(dataset_details)
        multi = d.loc[d["n_fragments"].gt(1)].copy()
        parent_groups = d.groupby("parent_smiles", sort=False)
        duplicate_groups = [g for _, g in parent_groups if len(g) > 1]
        conflict_groups = [
            g for g in duplicate_groups
            if target_conflict(dataset, g["target"])
        ]
        similarities = multi["morgan_tanimoto_full_vs_parent"].to_numpy(float)
        summary_rows.append(
            {
                "dataset": dataset,
                "n_clean_rows": int(len(d)),
                "n_multifragment_rows": int(len(multi)),
                "n_scaffold_changed_after_parent_selection": int(multi["scaffold_changed"].sum()),
                "fraction_multifragment_scaffold_changed": float(multi["scaffold_changed"].mean()) if len(multi) else 0.0,
                "mean_morgan_tanimoto_full_vs_parent_multifragment": float(np.mean(similarities)) if len(similarities) else 1.0,
                "median_morgan_tanimoto_full_vs_parent_multifragment": float(np.median(similarities)) if len(similarities) else 1.0,
                "min_morgan_tanimoto_full_vs_parent_multifragment": float(np.min(similarities)) if len(similarities) else 1.0,
                "n_multifragment_tanimoto_lt_0_95": int(np.sum(similarities < 0.95)) if len(similarities) else 0,
                "n_multifragment_tanimoto_lt_0_90": int(np.sum(similarities < 0.90)) if len(similarities) else 0,
                "n_unique_parent_smiles": int(d["parent_smiles"].nunique()),
                "n_parent_duplicate_groups": int(len(duplicate_groups)),
                "n_parent_duplicate_rows": int(sum(len(g) for g in duplicate_groups)),
                "n_parent_target_conflict_groups": int(len(conflict_groups)),
            }
        )

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    extras = pd.DataFrame(
        [
            {"extra_fragment": frag, "count": int(count)}
            for frag, count in extra_counter.most_common()
        ]
    )

    detail_path = OUT_DIR / "multifragment_standardization_detail_v3.csv"
    summary_path = OUT_DIR / "multifragment_standardization_summary_v3.csv"
    extras_path = OUT_DIR / "multifragment_extra_fragment_counts_v3.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    extras.to_csv(extras_path, index=False)

    print("\nMulti-fragment / parent-fragment sensitivity summary:")
    print(summary.to_string(index=False))
    print("\nMost common removed fragments (top 30):")
    print(extras.head(30).to_string(index=False))
    print("\nSaved:")
    print(detail_path)
    print(summary_path)
    print(extras_path)

    if summary["n_clean_rows"].sum() <= 0:
        raise AssertionError("No rows audited")
    print("\nMULTIFRAGMENT STANDARDIZATION AUDIT V3 PASSED")


if __name__ == "__main__":
    main()
