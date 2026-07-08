from __future__ import annotations

"""Validate Paper 2 endpoint, feature, and split outputs before model training.

This script checks:
- cleaned endpoint files exist;
- feature files exist and match cleaned row counts;
- split files exist;
- every split column contains train/calibration/test;
- random split proportions are close to 60/20/20;
- classification splits do not have empty positive/negative classes;
- scaffold splits keep each scaffold in a single role.

Outputs:
- paper2_admet_benchmark/results/tables/split_validation_report.csv
- paper2_admet_benchmark/results/tables/split_validation_issues.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper2_admet_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
TABLE_DIR = PAPER_DIR / "results" / "tables"

EXPECTED_ROLES = {"train", "calibration", "test"}
TARGET_FRACTIONS = {"train": 0.60, "calibration": 0.20, "test": 0.20}
FRACTION_TOLERANCE_RANDOM = 0.03
FRACTION_TOLERANCE_SCAFFOLD = 0.15


def get_split_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("split_")]


def add_issue(issues: list[dict], endpoint: str, split_col: str, severity: str, message: str) -> None:
    issues.append(
        {
            "endpoint": endpoint,
            "split_col": split_col,
            "severity": severity,
            "message": message,
        }
    )


def validate_endpoint(clean_path: Path, issues: list[dict]) -> list[dict]:
    endpoint = clean_path.name.replace("_clean.csv", "")
    rows = []
    df_clean = pd.read_csv(clean_path)

    feature_path = PROCESSED_DIR / f"{endpoint}_features.npz"
    split_path = PROCESSED_DIR / f"{endpoint}_splits.csv"

    if not feature_path.exists():
        add_issue(issues, endpoint, "", "error", f"Missing feature file: {feature_path}")
        return rows

    if not split_path.exists():
        add_issue(issues, endpoint, "", "error", f"Missing split file: {split_path}")
        return rows

    features = np.load(feature_path, allow_pickle=True)
    n_features = int(features["X"].shape[0])
    if len(df_clean) != n_features:
        add_issue(
            issues,
            endpoint,
            "",
            "error",
            f"Clean row count {len(df_clean)} does not match feature rows {n_features}.",
        )

    df = pd.read_csv(split_path)
    if len(df) != len(df_clean):
        add_issue(
            issues,
            endpoint,
            "",
            "error",
            f"Split row count {len(df)} does not match clean rows {len(df_clean)}.",
        )

    if "task_type" not in df.columns or "target" not in df.columns:
        add_issue(issues, endpoint, "", "error", "Split file must contain task_type and target columns.")
        return rows

    task_type = str(df["task_type"].iloc[0])
    split_cols = get_split_columns(df)
    if not split_cols:
        add_issue(issues, endpoint, "", "error", "No split columns found.")
        return rows

    for split_col in split_cols:
        roles_present = set(df[split_col].dropna().unique().tolist())
        missing_roles = EXPECTED_ROLES - roles_present
        unexpected_roles = roles_present - EXPECTED_ROLES
        if missing_roles:
            add_issue(issues, endpoint, split_col, "error", f"Missing roles: {sorted(missing_roles)}")
        if unexpected_roles:
            add_issue(issues, endpoint, split_col, "error", f"Unexpected roles: {sorted(unexpected_roles)}")

        setting = "scaffold" if split_col == "split_scaffold" else "random"
        tolerance = FRACTION_TOLERANCE_SCAFFOLD if setting == "scaffold" else FRACTION_TOLERANCE_RANDOM

        for role in ["train", "calibration", "test"]:
            part = df[df[split_col] == role]
            fraction = len(part) / len(df) if len(df) else np.nan
            target_fraction = TARGET_FRACTIONS[role]
            diff = abs(fraction - target_fraction)
            if diff > tolerance:
                add_issue(
                    issues,
                    endpoint,
                    split_col,
                    "warning" if setting == "scaffold" else "error",
                    f"{role} fraction {fraction:.3f} differs from target {target_fraction:.2f} by {diff:.3f}.",
                )

            row = {
                "endpoint": endpoint,
                "task_type": task_type,
                "split_col": split_col,
                "setting": setting,
                "role": role,
                "n": int(len(part)),
                "fraction": float(fraction),
                "target_mean": float(part["target"].mean()) if len(part) else np.nan,
                "target_min": float(part["target"].min()) if len(part) else np.nan,
                "target_max": float(part["target"].max()) if len(part) else np.nan,
            }

            if task_type == "classification" and len(part):
                positives = int(part["target"].sum())
                negatives = int(len(part) - positives)
                row["positive_count"] = positives
                row["negative_count"] = negatives
                row["positive_ratio"] = float(positives / len(part))
                if positives == 0 or negatives == 0:
                    add_issue(
                        issues,
                        endpoint,
                        split_col,
                        "error",
                        f"{role} split has positives={positives}, negatives={negatives}.",
                    )
            else:
                row["positive_count"] = np.nan
                row["negative_count"] = np.nan
                row["positive_ratio"] = np.nan

            if "scaffold" in df.columns and len(part):
                row["n_scaffolds"] = int(part["scaffold"].nunique())
            else:
                row["n_scaffolds"] = np.nan
            rows.append(row)

        if split_col == "split_scaffold" and "scaffold" in df.columns:
            scaffold_role_counts = df.groupby("scaffold")[split_col].nunique()
            leaking_scaffolds = scaffold_role_counts[scaffold_role_counts > 1]
            if len(leaking_scaffolds):
                add_issue(
                    issues,
                    endpoint,
                    split_col,
                    "error",
                    f"{len(leaking_scaffolds)} scaffolds appear in multiple scaffold-split roles.",
                )

    return rows


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(PROCESSED_DIR.glob("*_clean.csv"))
    if not clean_files:
        raise FileNotFoundError(f"No clean files found in {PROCESSED_DIR}.")

    issues: list[dict] = []
    report_rows: list[dict] = []

    for clean_path in clean_files:
        print("validating", clean_path.name)
        report_rows.extend(validate_endpoint(clean_path, issues))

    report = pd.DataFrame(report_rows)
    issues_df = pd.DataFrame(issues)

    report_path = TABLE_DIR / "split_validation_report.csv"
    issues_path = TABLE_DIR / "split_validation_issues.csv"
    report.to_csv(report_path, index=False)
    issues_df.to_csv(issues_path, index=False)

    print("\nsaved", report_path)
    print("saved", issues_path)

    if issues_df.empty:
        print("Validation passed: no issues found.")
        return

    print("\nValidation issues:")
    print(issues_df.to_string(index=False))

    if (issues_df["severity"] == "error").any():
        raise SystemExit("Validation failed with error-level issues. Fix before model training.")

    print("Validation completed with warnings only.")


if __name__ == "__main__":
    main()
