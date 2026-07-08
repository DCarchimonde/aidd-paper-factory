from __future__ import annotations

"""Create random and scaffold train/calibration/test splits for Paper 2.

Inputs:
- paper2_admet_benchmark/data/processed/<endpoint>_clean.csv
- paper2_admet_benchmark/data/processed/<endpoint>_features.npz

Outputs:
- paper2_admet_benchmark/data/processed/<endpoint>_splits.csv
- paper2_admet_benchmark/data/manifests/split_manifest.csv
- paper2_admet_benchmark/results/tables/split_summary.csv

Important:
- Random splits are stratified for classification where feasible.
- Scaffold splits keep each scaffold in a single role.
- For classification endpoints, scaffold assignment is class-balance aware so that
  train/calibration/test splits remain usable for ROC-AUC, PR-AUC, calibration,
  and conformal analyses whenever the endpoint distribution permits it.
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
ROLE_FRACTIONS = {
    "train": TRAIN_FRAC,
    "calibration": CALIBRATION_FRAC,
    "test": TEST_FRAC,
}
ROLES = ["train", "calibration", "test"]


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


def make_scaffold_groups(df: pd.DataFrame) -> list[dict]:
    scaffold_groups: dict[str, list[int]] = defaultdict(list)
    for idx, scaffold in enumerate(df["scaffold"].tolist()):
        scaffold_groups[scaffold].append(idx)

    groups = []
    for scaffold, indices in scaffold_groups.items():
        target_values = df.iloc[indices]["target"]
        groups.append(
            {
                "scaffold": scaffold,
                "indices": indices,
                "n": int(len(indices)),
                "positive_count": int(target_values.sum()) if str(df["task_type"].iloc[0]) == "classification" else 0,
            }
        )

    # Sort large and label-pure scaffolds first because they are hardest to place.
    return sorted(
        groups,
        key=lambda g: (g["n"], abs(g["positive_count"] - (g["n"] - g["positive_count"]))),
        reverse=True,
    )


def scaffold_assignment_score(
    role_counts: dict[str, int],
    role_pos_counts: dict[str, int],
    candidate_group: dict,
    candidate_role: str,
    n_total: int,
    n_pos_total: int,
    task_type: str,
) -> float:
    projected_counts = role_counts.copy()
    projected_pos_counts = role_pos_counts.copy()
    projected_counts[candidate_role] += int(candidate_group["n"])
    projected_pos_counts[candidate_role] += int(candidate_group["positive_count"])

    score = 0.0
    for role, fraction in ROLE_FRACTIONS.items():
        target_n = n_total * fraction
        size_error = abs(projected_counts[role] - target_n) / max(target_n, 1.0)
        score += size_error

        if task_type == "classification":
            target_pos = n_pos_total * fraction
            target_neg = (n_total - n_pos_total) * fraction
            projected_neg = projected_counts[role] - projected_pos_counts[role]
            pos_error = abs(projected_pos_counts[role] - target_pos) / max(target_pos, 1.0)
            neg_error = abs(projected_neg - target_neg) / max(target_neg, 1.0)
            score += 2.0 * (pos_error + neg_error)

    # Soft penalty for making any classification role single-class after enough samples exist.
    if task_type == "classification":
        for role in ROLES:
            n_role = projected_counts[role]
            pos_role = projected_pos_counts[role]
            neg_role = n_role - pos_role
            expected_min_size = max(10, int(0.05 * n_total))
            if n_role >= expected_min_size and (pos_role == 0 or neg_role == 0):
                score += 100.0

    return score


def add_scaffold_train_cal_test_split(df: pd.DataFrame, task_type: str) -> pd.Series:
    groups = make_scaffold_groups(df)
    n_total = len(df)
    n_pos_total = int(df["target"].sum()) if task_type == "classification" else 0

    role_indices: dict[str, list[int]] = {role: [] for role in ROLES}
    role_counts: dict[str, int] = {role: 0 for role in ROLES}
    role_pos_counts: dict[str, int] = {role: 0 for role in ROLES}

    for group in groups:
        scores = {
            role: scaffold_assignment_score(
                role_counts=role_counts,
                role_pos_counts=role_pos_counts,
                candidate_group=group,
                candidate_role=role,
                n_total=n_total,
                n_pos_total=n_pos_total,
                task_type=task_type,
            )
            for role in ROLES
        }
        chosen_role = min(scores, key=scores.get)
        role_indices[chosen_role].extend(group["indices"])
        role_counts[chosen_role] += int(group["n"])
        role_pos_counts[chosen_role] += int(group["positive_count"])

    split = pd.Series("unused", index=df.index, dtype="object")
    for role, indices in role_indices.items():
        split.iloc[indices] = role
    return split


def summarize_role(df: pd.DataFrame, split_col: str, endpoint: str, setting: str, seed: int | str) -> list[dict]:
    rows = []
    task_type = str(df["task_type"].iloc[0])
    for role in ROLES:
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
            positives = int(part["target"].sum())
            row["positive_ratio"] = float(positives / len(part))
            row["positive_count"] = positives
            row["negative_count"] = int(len(part) - positives)
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
        df[scaffold_col] = add_scaffold_train_cal_test_split(df, task_type=task_type)
        split_columns.append(scaffold_col)
        summary_rows.extend(summarize_role(df, scaffold_col, endpoint, "scaffold", "deterministic_class_balance_aware"))
        manifest_rows.append(
            {
                "endpoint": endpoint,
                "split_setting": "scaffold",
                "seed": "deterministic_class_balance_aware",
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
    print("Next: run 03b_validate_splits.py, then implement 04_train_models.py.")


if __name__ == "__main__":
    main()
