from __future__ import annotations

"""Prepare public ADMET endpoints for Paper 2.

This script is intentionally a scaffold for the next implementation step.
It will be expanded to:

1. load MoleculeNet/DeepChem and TDC endpoints;
2. canonicalize SMILES;
3. handle duplicate and conflicting labels;
4. save cleaned endpoint CSV files;
5. generate endpoint registry and cleaning-summary tables.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper2_admet_benchmark"
RAW_DIR = PAPER_DIR / "data" / "raw"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
MANIFEST_DIR = PAPER_DIR / "data" / "manifests"
TABLE_DIR = PAPER_DIR / "results" / "tables"


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MANIFEST_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)
        print("ensured", path)


def main() -> None:
    ensure_directories()
    print("Endpoint preparation scaffold is ready.")
    print("Next: implement MoleculeNet/DeepChem and TDC loaders.")


if __name__ == "__main__":
    main()
