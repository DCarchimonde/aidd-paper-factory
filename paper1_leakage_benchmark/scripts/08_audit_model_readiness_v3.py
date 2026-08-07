from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
FROZEN_DIR = PAPER_DIR / "results" / "frozen_v3"
OUT_DIR = PAPER_DIR / "results" / "model_rerun_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ReadinessSpec:
    label: str
    task_type: str
    datasets: tuple[str, ...]
    suffix: str
    protocols: tuple[str, ...]


SPECS = (
    ReadinessSpec(
        label="main_classification",
        task_type="classification",
        datasets=("BACE", "BBBP", "ClinTox", "HIV"),
        suffix="BACE-BBBP-ClinTox-HIV_single_group_20s_300c",
        protocols=(
            "legacy_scaffold",
            "random_observation",
            "size_matched_scaffold",
            "target_balanced_scaffold",
        ),
    ),
    ReadinessSpec(
        label="main_regression",
        task_type="regression",
        datasets=("ESOL", "FreeSolv"),
        suffix="ESOL-FreeSolv_single_group_20s_20000c",
        protocols=(
            "legacy_scaffold",
            "random_observation",
            "size_matched_scaffold",
            "target_balanced_scaffold",
        ),
    ),
    ReadinessSpec(
        label="acyclic_singleton_sensitivity",
        task_type="regression",
        datasets=("ESOL", "FreeSolv"),
        suffix="ESOL-FreeSolv_singleton_20s_5000c",
        protocols=("size_matched_scaffold", "target_balanced_scaffold"),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(spec: ReadinessSpec) -> Path:
    return (
        FROZEN_DIR
        / spec.label
        / f"split_manifest_v3_{spec.suffix}.csv"
    )


def main() -> None:
    metadata_path = FROZEN_DIR / "frozen_metadata_v3.json"
    partition_registry_path = FROZEN_DIR / "frozen_partition_registry_v3.csv"
    if not metadata_path.exists() or not partition_registry_path.exists():
        raise FileNotFoundError(
            "Frozen v3 metadata/partition registry missing. Run 07_freeze_manifests_v3.py first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_partition_sha = sha256_file(partition_registry_path)
    expected_partition_sha = metadata.get("partition_registry_sha256")
    if actual_partition_sha != expected_partition_sha:
        raise AssertionError(
            "Frozen partition registry SHA256 mismatch; do not train on modified manifests."
        )

    rows: list[dict] = []
    for spec in SPECS:
        path = manifest_path(spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen manifest: {path}")
        manifest = pd.read_csv(
            path,
            dtype={
                "dataset": str,
                "protocol": str,
                "partition_seed": str,
                "partition_hash": str,
                "canonical_smiles": str,
                "assignment": str,
            },
            keep_default_na=False,
            low_memory=False,
        )
        manifest["target_numeric"] = pd.to_numeric(manifest["target"], errors="raise")

        for dataset in spec.datasets:
            dataset_frame = manifest.loc[manifest["dataset"].eq(dataset)].copy()
            if dataset_frame.empty:
                raise AssertionError(f"Missing {dataset} in {spec.label}")
            for protocol in spec.protocols:
                protocol_frame = dataset_frame.loc[
                    dataset_frame["protocol"].eq(protocol)
                ].copy()
                if protocol_frame.empty:
                    raise AssertionError(
                        f"Missing protocol {protocol} for {dataset}/{spec.label}"
                    )
                grouped = protocol_frame.groupby(
                    ["partition_seed", "partition_hash"],
                    dropna=False,
                    sort=False,
                )
                expected_partitions = 1 if protocol == "legacy_scaffold" else 20
                if grouped.ngroups != expected_partitions:
                    raise AssertionError(
                        f"{spec.label}/{dataset}/{protocol}: "
                        f"{grouped.ngroups} partitions, expected {expected_partitions}"
                    )
                if protocol_frame["partition_hash"].nunique() != expected_partitions:
                    raise AssertionError(
                        f"Duplicate partition hashes remain in {spec.label}/{dataset}/{protocol}"
                    )

                for (partition_seed, partition_hash), group in grouped:
                    train = group.loc[group["assignment"].eq("train")]
                    test = group.loc[group["assignment"].eq("test")]
                    if train.empty or test.empty:
                        raise AssertionError(
                            f"Empty train/test in {dataset}/{protocol}/{partition_seed}"
                        )
                    y_train = train["target_numeric"].to_numpy(dtype=float)
                    y_test = test["target_numeric"].to_numpy(dtype=float)
                    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
                        raise AssertionError(
                            f"Non-finite target in {dataset}/{protocol}/{partition_seed}"
                        )

                    row = {
                        "freeze_label": spec.label,
                        "dataset": dataset,
                        "task_type": spec.task_type,
                        "protocol": protocol,
                        "partition_seed": partition_seed,
                        "partition_hash": partition_hash,
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                    }
                    if spec.task_type == "classification":
                        train_labels = set(np.unique(y_train).tolist())
                        test_labels = set(np.unique(y_test).tolist())
                        if train_labels != {0.0, 1.0}:
                            raise AssertionError(
                                f"Training split lacks both classes: {dataset}/{protocol}/{partition_seed}"
                            )
                        if test_labels != {0.0, 1.0}:
                            raise AssertionError(
                                f"Test split lacks both classes: {dataset}/{protocol}/{partition_seed}"
                            )
                        row.update(
                            {
                                "n_train_positive": int(np.sum(y_train == 1)),
                                "n_train_negative": int(np.sum(y_train == 0)),
                                "n_test_positive": int(np.sum(y_test == 1)),
                                "n_test_negative": int(np.sum(y_test == 0)),
                            }
                        )
                    rows.append(row)

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(
            ["freeze_label", "dataset", "task_type", "protocol"],
            as_index=False,
        )
        .agg(
            n_partitions=("partition_hash", "size"),
            n_unique_partitions=("partition_hash", "nunique"),
            min_n_train=("n_train", "min"),
            max_n_train=("n_train", "max"),
            min_n_test=("n_test", "min"),
            max_n_test=("n_test", "max"),
        )
    )
    classification = detail.loc[detail["task_type"].eq("classification")]
    if not classification.empty:
        class_summary = (
            classification.groupby(
                ["freeze_label", "dataset", "protocol"], as_index=False
            )
            .agg(
                min_train_positive=("n_train_positive", "min"),
                min_train_negative=("n_train_negative", "min"),
                min_test_positive=("n_test_positive", "min"),
                min_test_negative=("n_test_negative", "min"),
            )
        )
        summary = summary.merge(
            class_summary,
            on=["freeze_label", "dataset", "protocol"],
            how="left",
        )

    detail_path = OUT_DIR / "model_readiness_detail_v3.csv"
    summary_path = OUT_DIR / "model_readiness_summary_v3.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nModel-readiness summary:")
    print(summary.to_string(index=False))
    print("\nFrozen partition registry SHA256:")
    print(actual_partition_sha)
    print("\nSaved:")
    print(detail_path)
    print(summary_path)
    print("\nMODEL READINESS V3 PASSED")


if __name__ == "__main__":
    main()
