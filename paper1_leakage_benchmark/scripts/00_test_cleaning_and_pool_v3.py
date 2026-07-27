from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.cleaning_policy_v2 import build_clean_dataset_v2
from shared_utils.scaffold_identity import assert_valid_split, prepare_scaffold_frame
from shared_utils.split_candidate_pool_v3 import (
    generate_candidate_pool,
    materialize_candidate_split,
    select_paired_candidates,
)


def main() -> None:
    classification_raw = pd.DataFrame(
        {
            "smiles": ["CCO", "OCC", "CCN", "NCC", "c1ccccc1", "not_a_smiles"],
            "label": [0, 0, 0, 1, 1, 0],
        }
    )
    classification = build_clean_dataset_v2(
        classification_raw,
        dataset="synthetic_classification",
        smiles_col="smiles",
        target_col="label",
        task_type="classification",
    )
    assert len(classification.clean) == 2
    assert classification.summary["excluded_conflicting_classification_groups"] == 1
    assert classification.summary["invalid_or_missing_rows"] == 1
    assert set(classification.clean["target"]) == {0.0, 1.0}

    regression_raw = pd.DataFrame(
        {
            "smiles": ["CCO", "OCC", "c1ccccc1"],
            "value": [1.0, 3.0, -2.0],
        }
    )
    regression = build_clean_dataset_v2(
        regression_raw,
        dataset="synthetic_regression",
        smiles_col="smiles",
        target_col="value",
        task_type="regression",
    )
    ethanol = regression.clean.loc[regression.clean["canonical_smiles"].eq("CCO")]
    assert len(ethanol) == 1
    assert np.isclose(float(ethanol["target"].iloc[0]), 2.0)
    assert regression.summary["aggregated_regression_groups"] == 1

    smiles = ["C" * length for length in range(1, 31)]
    split_input = pd.DataFrame(
        {
            "canonical_smiles": smiles,
            "target": [float(index % 2) for index in range(len(smiles))],
        }
    )
    base = prepare_scaffold_frame(split_input, acyclic_mode="singleton")
    candidates, groups, meta = generate_candidate_pool(
        base,
        seed=42,
        n_candidates=200,
    )
    size_candidate, balanced_candidate, _, pair_meta = select_paired_candidates(
        candidates,
        groups,
        seed=42,
        total_n=len(base),
        total_target_sum=float(base["target"].sum()),
    )
    assert meta["target_blind_generation"] is True
    assert pair_meta["exact_size_match"] is True

    size_df, _ = materialize_candidate_split(
        base,
        groups,
        size_candidate,
        split_col="split_size",
    )
    balanced_df, _ = materialize_candidate_split(
        base,
        groups,
        balanced_candidate,
        split_col="split_balanced",
    )
    assert_valid_split(size_df, "split_size", require_scaffold_disjoint=True)
    assert_valid_split(balanced_df, "split_balanced", require_scaffold_disjoint=True)
    assert int(size_df["split_size"].eq("test").sum()) == int(
        balanced_df["split_balanced"].eq("test").sum()
    )

    print("CLEANING AND CANDIDATE POOL V3 SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
