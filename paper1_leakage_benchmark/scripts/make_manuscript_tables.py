from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def require_csv(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


# Table 1: dataset summary.
dataset = require_csv("dataset_summary.csv")
table1_cols = [c for c in ["dataset", "task_type", "n_raw", "n_after_dropna", "n_valid_smiles", "n_unique_canonical_smiles", "n_duplicate_canonical_smiles"] if c in dataset.columns]
table1 = dataset[table1_cols].copy()
table1.to_csv(TABLE_DIR / "manuscript_table1_dataset_summary.csv", index=False)

# Table 2: split target shift and scaffold statistics.
shift = require_csv("paper1_split_shift_table_rounded.csv")
scaffold = require_csv("paper1_scaffold_stats_summary.csv")
balanced_scaffold_path = TABLE_DIR / "paper1_balanced_scaffold_stats_summary.csv"
if balanced_scaffold_path.exists():
    balanced_scaffold = pd.read_csv(balanced_scaffold_path)
    scaffold = pd.concat([scaffold, balanced_scaffold], ignore_index=True, sort=False)

scaffold_simple = scaffold[[
    "dataset", "task_type", "split", "largest_scaffold_fraction", "singleton_scaffold_fraction", "n_shared_scaffolds", "shared_scaffold_fraction_test", "target_mean_gap_test_minus_train"
]].copy()
for col in ["largest_scaffold_fraction", "singleton_scaffold_fraction", "shared_scaffold_fraction_test", "target_mean_gap_test_minus_train"]:
    scaffold_simple[col] = scaffold_simple[col].round(4)
scaffold_simple.to_csv(TABLE_DIR / "manuscript_table2_split_diagnostics.csv", index=False)

# Table 3: generalization gap.
gap = require_csv("paper1_main_gap_table_rounded.csv")
gap.to_csv(TABLE_DIR / "manuscript_table3_generalization_gap.csv", index=False)

# Table 4: similarity audit.
sim = require_csv("paper1_similarity_audit_compact_rounded.csv")
sim.to_csv(TABLE_DIR / "manuscript_table4_similarity_audit.csv", index=False)

# Optional appendix tables.
if (TABLE_DIR / "paper1_outlier_cases.csv").exists():
    require_csv("paper1_outlier_cases.csv").to_csv(TABLE_DIR / "appendix_table_outlier_cases.csv", index=False)
if (TABLE_DIR / "paper1_model_rankings_by_split.csv").exists():
    require_csv("paper1_model_rankings_by_split.csv").to_csv(TABLE_DIR / "appendix_table_model_rankings.csv", index=False)

print("saved manuscript_table1_dataset_summary.csv")
print("saved manuscript_table2_split_diagnostics.csv")
print("saved manuscript_table3_generalization_gap.csv")
print("saved manuscript_table4_similarity_audit.csv")
