from __future__ import annotations

"""Run the frozen technical smoke test for confirmatory cluster splits at seed 99.

This wrapper deliberately keeps seed 99 separate from confirmatory seeds 101-110 and
writes a split-specific metric file so it cannot overwrite random/scaffold smoke results.
"""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve().parent / "16_train_confirmatory_models.py"
PAPER_DIR = ROOT / "paper2_admet_benchmark"
METRIC_DIR = PAPER_DIR / "results" / "metrics"


def load_training_module():
    spec = importlib.util.spec_from_file_location("confirmatory_training", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    training = load_training_module()
    endpoints = ["bbbp", "clintox", "esol", "lipophilicity"]
    split_columns = ["split_confirm_cluster_seed99"]
    requested_models = ["linear", "random_forest", "xgboost", "mlp_ecfp"]
    regimes = ["unweighted", "balanced_oversample"]

    print("Seed-99 cluster smoke training")
    print("endpoints:", ", ".join(endpoints))
    print("split columns:", ", ".join(split_columns))
    print("classification regimes:", ", ".join(regimes))
    print("models:", ", ".join(requested_models))

    rows: list[dict[str, object]] = []
    for endpoint in endpoints:
        print(f"\n========== {endpoint} ==========")
        rows.extend(
            training.train_endpoint(
                endpoint=endpoint,
                split_columns=split_columns,
                requested_models=requested_models,
                classification_regimes=regimes,
            )
        )

    if not rows:
        raise RuntimeError(
            "Cluster smoke training produced zero metric rows. Confirm that "
            "split_confirm_cluster_seed99 exists in each endpoint split file."
        )

    metrics = pd.DataFrame(rows)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRIC_DIR / "baseline_metrics_confirmatory_smoke_cluster_seed99.csv"
    metrics.to_csv(output_path, index=False)
    print("\nsaved", output_path)
    print(f"Cluster smoke training complete. Metric rows: {len(metrics)}.")


if __name__ == "__main__":
    main()
