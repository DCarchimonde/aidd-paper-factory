from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.scaffold_identity import (
    ACYCLIC_SCAFFOLD,
    assert_valid_split,
    generate_scaffold_v2,
    prepare_scaffold_frame,
)
from shared_utils.split_search_v2 import (
    add_legacy_scaffold_split_v2,
    add_searched_scaffold_split_v2,
)


def synthetic_frame() -> pd.DataFrame:
    smiles = [
        "c1ccccc1",
        "Cc1ccccc1",
        "Oc1ccccc1",
        "C1CCCCC1",
        "CC1CCCCC1",
        "C1CCCC1",
        "CC1CCCC1",
        "CCO",
        "CCCO",
        "CCCC",
        "CCN",
        "CCCN",
    ]
    return pd.DataFrame(
        {
            "canonical_smiles": smiles,
            "target": [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
        }
    )


def main() -> None:
    assert generate_scaffold_v2("CCO") == ACYCLIC_SCAFFOLD
    base = prepare_scaffold_frame(synthetic_frame())
    assert base["scaffold"].notna().all()
    assert (base["scaffold"].astype(str).str.len() > 0).all()
    assert int(base["scaffold"].eq(ACYCLIC_SCAFFOLD).sum()) == 5

    legacy, _ = add_legacy_scaffold_split_v2(base, test_size=0.25)
    assert_valid_split(
        legacy,
        "split_legacy_scaffold_v2",
        require_scaffold_disjoint=True,
    )

    size_a, size_meta_a = add_searched_scaffold_split_v2(
        base,
        seed=42,
        objective="size_only",
        test_size=0.25,
        n_trials=30,
        candidate_pool=10,
    )
    size_b, size_meta_b = add_searched_scaffold_split_v2(
        base,
        seed=42,
        objective="size_only",
        test_size=0.25,
        n_trials=30,
        candidate_pool=10,
    )
    assert size_meta_a["partition_hash"] == size_meta_b["partition_hash"]
    assert size_a["split_size_matched_scaffold_v2"].equals(
        size_b["split_size_matched_scaffold_v2"]
    )

    balanced, balanced_meta = add_searched_scaffold_split_v2(
        base,
        seed=123,
        objective="target_balanced",
        test_size=0.25,
        n_trials=30,
        candidate_pool=10,
    )
    assert_valid_split(
        balanced,
        "split_target_balanced_scaffold_v2",
        require_scaffold_disjoint=True,
    )
    assert balanced_meta["actual_test_n"] == int(
        balanced["split_target_balanced_scaffold_v2"].eq("test").sum()
    )

    singleton = prepare_scaffold_frame(
        synthetic_frame(),
        acyclic_mode="singleton",
    )
    assert singleton["scaffold"].nunique() > base["scaffold"].nunique()

    print("SPLITTING V2 SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
