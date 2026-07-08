from __future__ import annotations

"""Create random and scaffold train/calibration/test splits for Paper 2.

Inputs:
- paper2_admet_benchmark/data/processed/<endpoint>_clean.csv
- paper2_admet_benchmark/data/processed/<endpoint>_features.npz

Outputs:
- paper2_admet_benchmark/data/processed/<endpoint>_splits.csv
- paper2_admet_benchmark/data/manifests/split_manifest.csv
- paper2_admet_benchmark/results/tables/split_summary.csv
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RDLogger.DisableLog("rdApp.*")

PAPER_DIR = ROOT / "paper2_admet_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
MANIFEST_DIR = PAPER_DIR / "data" / "manifests"
TABLE_DIR = PAPER_DIR / "results" / "tables"

SEEDS = [0, 1, 2, 3, 4]
TRAIN_FRAC = 0.60
CALIBRATION_FRAC = 0.20
TEST_FRAC = 0.20


def generate_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold if scaffold else "NO_SCAFFOLD"


def split_col_name(setting: str, seed: int | None = None) -> str:
    if seed is None:
        return f"split_{setting}"
    return f"split_{setting}_seed{seed}"


def can_stratify(y: pd.Series, min_count: int = 2) -> bool:
    counts = y.value_counts(dropna=False)
    return len(counts) > 1 and int(counts.min()) >= min_count


def add_random_train_cal_test_split(
    df: pd.DataFrame,
    task_type: str,
    seed: int,
) -> pd.Series:
    idx = np.arange(len(df))
    y = df["target"]

    stratify_first = y if task_type == "classification" and can_stratify(y, min_count=2) else None
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=CALIBRATION_FRAC + TEST_FRAC,
        random_state=seed,
        stratify=stratify_first,
    )

    temp_y = y.iloc[temp_idx]
    stratify_second = temp_y if task_type == "classification" and can_stratify(temp_y, min_count=2) else None
    calibration_idx, test_idx = train_test_split(
        temp_idx,
        test_size=TEST_FRAC / (CALIBRATION_FRAC + TEST_FRAC),
        random_state=seed,
        stratify=stratify_second,
    )

    split = pd.Series("unused", index=df.index, dtype="object")
    split.iloc[train_idx] = "train"
    split.iloc[calibration_idx] = "calibration"
    split.iloc[test_idx] = "test"
    return split


def add_scaffold_train_cal_test_split(df: pd.DataFrame) -> pd.Series:
    scaffold_groups: dict[str, list[int]] = defaultdict(list)
    for idx, scaffold in enumerate(df["scaffold"].tolist()):
        scaffold_groups[scaffold].append(idx)

    groups = sorted(scaffold_groups.values(), key=len, reverse=True)
    n_total = len(df)
    target_train_n = int(round(n_total * TRAIN_FRAC))
    target_cal_n = int(round(n_total * CALIBRATION_FRAC))

    train_idx: list[int] = []
    calibration_idx: list[int] = []
    test_idx: list[int] = []

    # Greedy assignment by current fraction deficit. This keeps scaffolds intact.
    for group in groups:
        train_deficit = target_train_n - len(train_idx)
        cal_deficit = target_cal_n - len(calibration_idx)
        if train_deficit >= cal_deficit and train_deficit > 0:
            train_idx.extend(group)
        elif cal_deficit > 0:
            calibration_idx.extend(group)
        else:
            test_idx.extend(group)

    # If large early scaffolds overfilled train/calibration, remaining groups go to test.
    assigned = set(train_idx) | set(calibration_idx) | set(test_idx)
    for idx in range(n_total):
        if idx not in assigned:
            test_idx.append(idx)

    split = pd.Series("unused", index=df.index, dtype="object")
    split.iloc[train_idx] = "train"
    split.iloc[calibration_idx] = "calibration"
    split.iloc[test_idx] = "test"
    return split


def summarize_role(df: pd.DataFrame, split_col: str, endpoint: str, setting: str, seed: int | str) -> list[dict]:
    rows = []
    task_type = str(df["task_type"].iloc[0])
    for role in ["train", "calibration", "test"]:
        part = df[df[split_col] == role]
        row = {
            "endpoint": endpoint,
            "split_setting": setting,
            "seed": seed,
            "split_col": split_col,
            "role": role,
            "n": int(len(part)),
            "fraction": float(len(part) / len(df)) if len(df) else np.nan,
            "task_type": task_type,
            "target_mean": float(part["target"].mean()) if len(part) else np.nan,
            "target_min": float(part["target"].min()) if len(part) else np.nan,
            "target_max": float(part["target"].max()) if len(part) else np.nan,
            "n_scaffolds": int(part["scaffold"].nunique()) if "scaffold" in part.columns else np.nan,
        }
        if task_type == "classification" and len(part):
            row["positive_ratio"] = float(part["target"].mean())
            row["positive_count"] = int(part["target"].sum())
            row["negative_count"] = int(len(part) - part["target"].sum())
        else:
            row["positive_ratio"] = np.nan
            row["positive_count"] = np.nan
            row["negative_count"] = np.nan
        rows.append(row)
    return rows


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(PROCESSED_DIR.glob("*_clean.csv"))
    if not clean_files:
        raise FileNotFoundError(
            f"No cleaned endpoint files found in {PROCESSED_DIR}. Run 01_prepare_endpoints.py first."
        )

    manifest_rows = []
    summary_rows = []

    for clean_path in clean_files:
        endpoint = clean_path.name.replace("_clean.csv", "")
        print("\n==========", endpoint, "==========")
        df = pd.read_csv(clean_path)
        feature_path = PROCESSED_DIR / f"{endpoint}_features.npz"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing {feature_path}. Run 02_featurize.py first.")

        feature_data = np.load(feature_path, allow_pickle=True)
        if len(df) != int(feature_data["X"].shape[0]):
            raise ValueError(
                f"{endpoint}: clean rows ({len(df)}) do not match feature rows ({feature_data['X'].shape[0]})."
            )

        task_type = str(df["task_type"].iloc[0])
        df = df.copy()
        df["scaffold"] = df["canonical_smiles"].map(generate_scaffold)

        split_columns = []
        for seed in SEEDS:
            col = split_col_name("random", seed)
            df[col] = add_random_train_cal_test_split(df, task_type=task_type, seed=seed)
            split_columns.append(col)
            summary_rows.extend(summarize_role(df, col, endpoint, "random", seed))
            manifest_rows.append(
                {
                    "endpoint": endpoint,
                    "split_setting": "random",
                    "seed": seed,
                    "split_col": col,
                    "train_fraction_target": TRAIN_FRAC,
                    "calibration_fraction_target": CALIBRATION_FRAC,
                    "test_fraction_target": TEST_FRAC,
                    "split_file": str((PROCESSED_DIR / f"{endpoint}_splits.csv").relative_to(ROOT)),
                }
            )

        scaffold_col = split_col_name("scaffold")
        df[scaffold_col] = add_scaffold_train_cal_test_split(df)
        split_columns.append(scaffold_col)
        summary_rows.extend(summarize_role(df, scaffold_col, endpoint, "scaffold", "deterministic"))
        manifest_rows.append(
            {
                "endpoint": endpoint,
                "split_setting": "scaffold",
                "seed": "deterministic",
                "split_col": scaffold_col,
                "train_fraction_target": TRAIN_FRAC,
                "calibration_fraction_target": CALIBRATION_FRAC,
                "test_fraction_target": TEST_FRAC,
                "split_file": str((PROCESSED_DIR / f"{endpoint}_splits.csv").relative_to(ROOT)),
            }
        )

        split_path = PROCESSED_DIR / f"{endpoint}_splits.csv"
        df.to_csv(split_path, index=False)
        print(f"saved split file: {split_path}")
        print("split columns:", ", ".join(split_columns))

    manifest = pd.DataFrame(manifest_rows)
    summary = pd.DataFrame(summary_rows)

    manifest_path = MANIFEST_DIR / "split_manifest.csv"
    summary_path = TABLE_DIR / "split_summary.csv"
    manifest.to_csv(manifest_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nsaved", manifest_path)
    print("saved", summary_path)
    print("Split generation complete.")
    print("Next: implement 04_train_models.py.")


if __name__ == "__main__":
    main()
