from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "sklearn",
    "xgboost",
    "matplotlib",
    "rdkit",
]

OPTIONAL_MODULES = [
    "shap",
    "tdc",
    "yaml",
]


def check_module(name: str, required: bool = True) -> bool:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {name}: {version}")
        return True
    except Exception as exc:  # noqa: BLE001
        level = "ERROR" if required else "WARN"
        print(f"[{level}] {name}: {exc}")
        return False


def main() -> None:
    print("Paper 2 environment check")
    print("Repository root:", ROOT)
    print("Python:", sys.version.replace("\n", " "))
    print()

    required_ok = [check_module(name, required=True) for name in REQUIRED_MODULES]
    print()
    _ = [check_module(name, required=False) for name in OPTIONAL_MODULES]

    if not all(required_ok):
        raise SystemExit("Missing required dependencies. Install environment/requirements.txt first.")

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SystemExit("RDKit failed to parse test SMILES.")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = generator.GetFingerprint(mol)
    print()
    print("RDKit Morgan fingerprint test:", fp.GetNumBits(), "bits")
    print("Environment test passed.")


if __name__ == "__main__":
    main()
