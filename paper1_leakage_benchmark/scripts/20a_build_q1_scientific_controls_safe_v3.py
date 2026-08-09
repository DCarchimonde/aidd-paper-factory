from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "paper1_leakage_benchmark" / "scripts"
BROKEN_ORIGINAL = SCRIPT_DIR / "20_build_q1_scientific_controls_v3.py"


def load_original():
    spec = importlib.util.spec_from_file_location("paper1_q1_controls_original", BROKEN_ORIGINAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BROKEN_ORIGINAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    q1 = load_original()

    collateral_paired, collateral_summary = q1.build_collateral_diagnostics()
    mean_paired, mean_summary = q1.build_mean_only_control()
    cleaning = q1.build_cleaning_accounting()
    seed_paired, seed_summary = q1.build_model_seed_sensitivity()

    outputs = {
        "collateral_paired": q1.TABLE_DIR / "q1_collateral_partition_diagnostics_v3.csv",
        "collateral_summary": q1.TABLE_DIR / "q1_collateral_diagnostics_summary_v3.csv",
        "mean_only_paired": q1.TABLE_DIR / "q1_mean_only_regression_control_v3.csv",
        "mean_only_summary": q1.TABLE_DIR / "q1_mean_only_regression_summary_v3.csv",
        "cleaning": q1.TABLE_DIR / "q1_cleaning_accounting_v3.csv",
        "seed_paired": q1.TABLE_DIR / "q1_model_seed_partition_effects_v3.csv",
        "seed_summary": q1.TABLE_DIR / "q1_model_seed_summary_v3.csv",
    }

    collateral_paired.to_csv(outputs["collateral_paired"], index=False)
    collateral_summary.to_csv(outputs["collateral_summary"], index=False)
    mean_paired.to_csv(outputs["mean_only_paired"], index=False)
    mean_summary.to_csv(outputs["mean_only_summary"], index=False)
    cleaning.to_csv(outputs["cleaning"], index=False)
    seed_paired.to_csv(outputs["seed_paired"], index=False)
    seed_summary.to_csv(outputs["seed_summary"], index=False)

    # TeX generation is deliberately delegated to the dedicated, robust
    # 20b/20c scripts. The legacy writer in 20_build_q1_scientific_controls_v3.py
    # used pandas itertuples() on integer pivot columns and could address them as
    # _4/_5/_6 differently across pandas versions, causing the observed crash.
    # Keeping this stage data-only removes that version-sensitive failure mode.

    required = list(outputs.values())
    missing = [path for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Q1 scientific-control outputs missing: " + ", ".join(map(str, missing)))

    print("\nQ1 SCIENTIFIC CONTROLS (SAFE WRITER)")
    print("Mean-only regression control:")
    print(mean_summary.to_string(index=False))
    print("\nModel-seed sensitivity rows:", len(seed_paired))
    print("Collateral paired diagnostic rows:", len(collateral_paired))
    print("Cleaning accounting rows:", len(cleaning))
    print("\nSaved:")
    for path in required:
        print(path)
    print("\nQ1 SCIENTIFIC CONTROL AUDIT: PASS")


if __name__ == "__main__":
    main()
