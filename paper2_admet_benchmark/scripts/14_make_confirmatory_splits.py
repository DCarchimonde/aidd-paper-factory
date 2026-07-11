from __future__ import annotations

"""Create pre-specified confirmatory random and label-blind scaffold splits.

This script preserves all existing development split columns and appends:

- split_confirm_random_seed101 ... seed110
- split_confirm_scaffold_seed101 ... seed110

Random classification splits are stratified where feasible. Confirmatory scaffold
assignment is strictly label-blind: target labels/values are never read by the group
assignment algorithm. Scaffold identity and group size are the only assignment inputs.

The output split files are written in place after making a `.development_backup.csv`
copy the first time the script is run.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper2_admet_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
MANIFEST_DIR = PAPER_DIR / "data" / "manifests"
TABLE_DIR = PAPER_DIR / "results" / "tables"

DEFAULT_SEEDS = list(range(101, 111))
TRAIN_FRAC = 0.60
CALIBRATION_FRAC = 0.20
TEST_FRAC = 0.20
ROLE_FRACTIONS = {
    "train": TRAIN_FRAC,
    "calibration": CALIBRATION_FRAC,
    "test": TEST_FRAC,
}
ROLE_ORDER = ["train", "calibration", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create confirmatory random and label-blind scaffold splits.")
    parser.add_argument("--endpoints", default=None, help="Optional comma-separated endpoint names.")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated confirmatory seeds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing confirmatory split columns. Without this flag existing columns are retained.",
    )
    return parser.parse_args()


def parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Confirmatory seeds must be unique.")
    return seeds


def available_endpoints() -> list[str]:
    return sorted(path.name.replace("_splits.csv", "") for path in PROCESSED_DIR.glob("*_splits.csv"))


def can_stratify(y: pd.Series, min_count: int = 2) -> bool:
    counts = y.value_counts(dropna=False)
    return len(counts) > 1 and int(counts.min()) >= min_count


def random_train_cal_test_split(df: pd.DataFrame, task_type: str, seed: int) -> pd.Series:
    indices = np.arange(len(df))
    y = df["target"]
    first_stratify = y if task_type == "classification" and can_stratify(y) else None
    train_idx, temporary_idx = train_test_split(
        indices,
        test_size=CALIBRATION_FRAC + TEST_FRAC,
        random_state=seed,
        stratify=first_stratify,
    )

    temporary_y = y.iloc[temporary_idx]
    second_stratify = (
        temporary_y
        if task_type == "classification" and can_stratify(temporary_y)
        else None
    )
    calibration_idx, test_idx = train_test_split(
        temporary_idx,
        test_size=TEST_FRAC / (CALIBRATION_FRAC + TEST_FRAC),
        random_state=seed,
        stratify=second_stratify,
    )

    split = pd.Series("unused", index=df.index, dtype="object")
    split.iloc[train_idx] = "train"
    split.iloc[calibration_idx] = "calibration"
    split.iloc[test_idx] = "test"
    return split


def scaffold_groups(df: pd.DataFrame, seed: int) -> list[tuple[str, np.ndarray]]:
    if "scaffold" not in df.columns:
        raise KeyError("Split file must contain a scaffold column before confirmatory scaffold splitting.")

    rng = np.random.default_rng(seed)
    groups = [
        (str(scaffold), group.index.to_numpy(dtype=int))
        for scaffold, group in df.groupby("scaffold", sort=False)
    ]
    rng.shuffle(groups)
    # Stable sorting retains randomized order among equal-size groups.
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    return groups


def label_blind_scaffold_split(df: pd.DataFrame, seed: int) -> pd.Series:
    """Assign scaffold groups using only group identity and size.

    At each step, place the next scaffold into the role with the smallest projected
    fill ratio relative to its target size. Randomized tie order is determined only by
    the pre-specified seed. No target label or target value is consulted.
    """
    groups = scaffold_groups(df, seed=seed)
    n_total = len(df)
    target_sizes = {role: max(1.0, n_total * fraction) for role, fraction in ROLE_FRACTIONS.items()}
    role_counts = {role: 0 for role in ROLE_ORDER}
    role_indices: dict[str, list[int]] = {role: [] for role in ROLE_ORDER}

    rng = np.random.default_rng(seed + 10_000)
    role_tie_order = list(ROLE_ORDER)
    rng.shuffle(role_tie_order)
    tie_rank = {role: idx for idx, role in enumerate(role_tie_order)}

    for _scaffold, indices in groups:
        group_n = len(indices)
        candidate_roles = sorted(
            ROLE_ORDER,
            key=lambda role: (
                (role_counts[role] + group_n) / target_sizes[role],
                tie_rank[role],
            ),
        )
        chosen = candidate_roles[0]
        role_indices[chosen].extend(indices.tolist())
        role_counts[chosen] += group_n

    split = pd.Series("unused", index=df.index, dtype="object")
    for role, indices in role_indices.items():
        split.iloc[indices] = role
    return split


def summarize_split(
    df: pd.DataFrame,
    endpoint: str,
    split_col: str,
    split_type: str,
    seed: int,
) -> list[dict[str, object]]:
    task_type = str(df["task_type"].iloc[0])
    rows = []
    for role in ROLE_ORDER:
        part = df[df[split_col] == role]
        row: dict[str, object] = {
            "endpoint": endpoint,
            "task_type": task_type,
            "split_col": split_col,
            "split_type": split_type,
            "seed": seed,
            "assignment_uses_target": split_type == "confirm_random_stratified",
            "role": role,
            "n": int(len(part)),
            "fraction": float(len(part) / len(df)) if len(df) else np.nan,
            "target_mean": float(part["target"].mean()) if len(part) else np.nan,
            "target_min": float(part["target"].min()) if len(part) else np.nan,
            "target_max": float(part["target"].max()) if len(part) else np.nan,
            "n_scaffolds": int(part["scaffold"].nunique()) if len(part) else 0,
        }
        if task_type == "classification" and len(part):
            positives = int(part["target"].sum())
            row.update(
                {
                    "positive_count": positives,
                    "negative_count": int(len(part) - positives),
                    "positive_ratio": float(positives / len(part)),
                }
            )
        else:
            row.update(
                {
                    "positive_count": np.nan,
                    "negative_count": np.nan,
                    "positive_ratio": np.nan,
                }
            )
        rows.append(row)
    return rows


def validate_split(df: pd.DataFrame, split_col: str, require_scaffold_exclusion: bool) -> list[str]:
    issues: list[str] = []
    roles = set(df[split_col].dropna().unique())
    if roles != set(ROLE_ORDER):
        issues.append(f"roles={sorted(roles)}")
    if (df[split_col] == "unused").any():
        issues.append("contains unused rows")
    if require_scaffold_exclusion:
        role_counts = df.groupby("scaffold")[split_col].nunique()
        leakage_n = int((role_counts > 1).sum())
        if leakage_n:
            issues.append(f"{leakage_n} scaffolds cross roles")
    return issues


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    endpoints = parse_csv_list(args.endpoints) or available_endpoints()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for endpoint in endpoints:
        split_path = PROCESSED_DIR / f"{endpoint}_splits.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        print(f"\n========== {endpoint} ==========")
        df = pd.read_csv(split_path)
        task_type = str(df["task_type"].iloc[0])

        backup_path = PROCESSED_DIR / f"{endpoint}_splits.development_backup.csv"
        if not backup_path.exists():
            shutil.copy2(split_path, backup_path)
            print("saved development backup:", backup_path)

        for seed in seeds:
            random_col = f"split_confirm_random_seed{seed}"
            scaffold_col = f"split_confirm_scaffold_seed{seed}"

            if args.overwrite or random_col not in df.columns:
                df[random_col] = random_train_cal_test_split(df, task_type=task_type, seed=seed)
            if args.overwrite or scaffold_col not in df.columns:
                df[scaffold_col] = label_blind_scaffold_split(df, seed=seed)

            for split_col, split_type, scaffold_exclusion in [
                (random_col, "confirm_random_stratified", False),
                (scaffold_col, "confirm_scaffold_label_blind", True),
            ]:
                issues = validate_split(df, split_col, require_scaffold_exclusion=scaffold_exclusion)
                validation_rows.append(
                    {
                        "endpoint": endpoint,
                        "split_col": split_col,
                        "split_type": split_type,
                        "seed": seed,
                        "validation_passed": not issues,
                        "issues": "; ".join(issues),
                    }
                )
                summary_rows.extend(
                    summarize_split(
                        df=df,
                        endpoint=endpoint,
                        split_col=split_col,
                        split_type=split_type,
                        seed=seed,
                    )
                )
                manifest_rows.append(
                    {
                        "endpoint": endpoint,
                        "split_col": split_col,
                        "split_type": split_type,
                        "seed": seed,
                        "train_fraction_target": TRAIN_FRAC,
                        "calibration_fraction_target": CALIBRATION_FRAC,
                        "test_fraction_target": TEST_FRAC,
                        "label_blind_assignment": split_type == "confirm_scaffold_label_blind",
                        "split_file": str(split_path.relative_to(ROOT)),
                    }
                )

        df.to_csv(split_path, index=False)
        print("updated", split_path)

    manifest = pd.DataFrame(manifest_rows)
    summary = pd.DataFrame(summary_rows)
    validation = pd.DataFrame(validation_rows)

    manifest_path = MANIFEST_DIR / "confirmatory_split_manifest.csv"
    summary_path = TABLE_DIR / "confirmatory_split_summary.csv"
    validation_path = TABLE_DIR / "confirmatory_split_validation.csv"
    manifest.to_csv(manifest_path, index=False)
    summary.to_csv(summary_path, index=False)
    validation.to_csv(validation_path, index=False)

    failed = int((~validation["validation_passed"].astype(bool)).sum()) if not validation.empty else 0
    print("\nsaved", manifest_path)
    print("saved", summary_path)
    print("saved", validation_path)
    print(f"Confirmatory split generation complete. Validation failures: {failed}.")
    if failed:
        raise SystemExit("One or more confirmatory splits failed structural validation.")


if __name__ == "__main__":
    main()
