from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
DATA_DIR = PAPER_DIR / "data" / "processed_v2"
OUT_DIR = PAPER_DIR / "results" / "model_rerun_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ("BACE", "BBBP", "ClinTox", "HIV", "ESOL", "FreeSolv")


def parse_without_h_removal(smiles: str):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(str(smiles), params)


def main() -> None:
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    for dataset in DATASETS:
        path = DATA_DIR / f"{dataset.lower()}_clean_v2.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
        if "canonical_smiles" not in frame.columns:
            raise KeyError(f"{path} missing canonical_smiles")

        n_isolated_h = 0
        n_multifragment = 0
        n_parse_fail = 0

        for row_index, smiles in enumerate(frame["canonical_smiles"].astype(str)):
            mol = parse_without_h_removal(smiles)
            if mol is None:
                n_parse_fail += 1
                detail_rows.append(
                    {
                        "dataset": dataset,
                        "row_index": row_index,
                        "canonical_smiles": smiles,
                        "parse_failed": True,
                        "n_isolated_hydrogen_atoms": None,
                        "n_fragments": None,
                    }
                )
                continue

            isolated_h = [
                atom.GetIdx()
                for atom in mol.GetAtoms()
                if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0
            ]
            n_fragments = len(Chem.GetMolFrags(mol))

            if isolated_h:
                n_isolated_h += 1
            if n_fragments > 1:
                n_multifragment += 1

            if isolated_h or n_fragments > 1:
                detail_rows.append(
                    {
                        "dataset": dataset,
                        "row_index": row_index,
                        "canonical_smiles": smiles,
                        "parse_failed": False,
                        "n_isolated_hydrogen_atoms": len(isolated_h),
                        "n_fragments": n_fragments,
                    }
                )

        summary_rows.append(
            {
                "dataset": dataset,
                "n_clean_rows": len(frame),
                "n_parse_failures": n_parse_fail,
                "n_rows_with_isolated_hydrogen": n_isolated_h,
                "n_rows_with_multiple_fragments": n_multifragment,
                "fraction_with_isolated_hydrogen": n_isolated_h / len(frame),
                "fraction_with_multiple_fragments": n_multifragment / len(frame),
            }
        )

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)

    detail_path = OUT_DIR / "rdkit_hydrogen_warning_detail_v3.csv"
    summary_path = OUT_DIR / "rdkit_hydrogen_warning_summary_v3.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nRDKit isolated-hydrogen / multi-fragment audit summary:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(detail_path)
    print(summary_path)

    if int(summary["n_parse_failures"].sum()) > 0:
        raise AssertionError("Some clean_v2 SMILES failed RDKit parsing during warning audit")

    print("\nRDKIT HYDROGEN WARNING AUDIT V3 PASSED")


if __name__ == "__main__":
    main()
