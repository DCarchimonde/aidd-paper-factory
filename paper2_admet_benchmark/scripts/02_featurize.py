from __future__ import annotations

"""Featurize cleaned Paper 2 endpoints with ECFP4/Morgan fingerprints.

Inputs:
- paper2_admet_benchmark/data/processed/<endpoint>_clean.csv

Outputs:
- paper2_admet_benchmark/data/processed/<endpoint>_features.npz
- paper2_admet_benchmark/results/tables/featurization_summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.chem_features import build_feature_matrix

RDLogger.DisableLog("rdApp.*")

PAPER_DIR = ROOT / "paper2_admet_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
TABLE_DIR = PAPER_DIR / "results" / "tables"

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(PROCESSED_DIR.glob("*_clean.csv"))
    if not clean_files:
        raise FileNotFoundError(
            f"No cleaned endpoint files found in {PROCESSED_DIR}. Run 01_prepare_endpoints.py first."
        )

    summary_rows = []
    for clean_path in clean_files:
        endpoint = clean_path.name.replace("_clean.csv", "")
        print("\n==========", endpoint, "==========")
        df = pd.read_csv(clean_path)

        if "canonical_smiles" not in df.columns or "target" not in df.columns:
            raise KeyError(f"{clean_path} must contain canonical_smiles and target columns.")

        smiles = df["canonical_smiles"].astype(str).tolist()
        X, valid_indices = build_feature_matrix(
            smiles,
            radius=FINGERPRINT_RADIUS,
            n_bits=FINGERPRINT_BITS,
        )

        valid_df = df.iloc[valid_indices].reset_index(drop=True)
        y = valid_df["target"].to_numpy()
        canonical_smiles = valid_df["canonical_smiles"].to_numpy()

        out_path = PROCESSED_DIR / f"{endpoint}_features.npz"
        np.savez_compressed(
            out_path,
            X=X,
            y=y,
            canonical_smiles=canonical_smiles,
            endpoint=endpoint,
            fingerprint_radius=FINGERPRINT_RADIUS,
            fingerprint_bits=FINGERPRINT_BITS,
        )

        n_input = len(df)
        n_valid = len(valid_df)
        n_invalid_after_cleaning = n_input - n_valid
        print(f"saved features: {out_path} X={X.shape} y={y.shape}")

        task_type = str(valid_df["task_type"].iloc[0]) if "task_type" in valid_df.columns else "unknown"
        admet_category = str(valid_df["admet_category"].iloc[0]) if "admet_category" in valid_df.columns else "unknown"

        summary_rows.append(
            {
                "endpoint": endpoint,
                "task_type": task_type,
                "admet_category": admet_category,
                "n_input_clean": int(n_input),
                "n_valid_features": int(n_valid),
                "n_invalid_after_cleaning": int(n_invalid_after_cleaning),
                "fingerprint": "ECFP4_Morgan",
                "radius": FINGERPRINT_RADIUS,
                "n_bits": FINGERPRINT_BITS,
                "feature_file": str(out_path.relative_to(ROOT)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = TABLE_DIR / "featurization_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nsaved", summary_path)
    print("Featurization complete.")
    print("Next: run 03_make_splits.py.")


if __name__ == "__main__":
    main()
