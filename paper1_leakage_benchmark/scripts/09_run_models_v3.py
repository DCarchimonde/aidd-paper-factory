from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.modeling_v3 import (
    MODEL_SEED_SENSITIVITY_PARTITION_SEEDS,
    build_model,
    classification_metrics,
    load_or_build_morgan_matrix,
    model_names,
    production_model_seed,
    regression_metrics,
    sensitivity_model_seeds,
)

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
DATA_DIR = PAPER_DIR / "data" / "processed_v2"
FROZEN_DIR = PAPER_DIR / "results" / "frozen_v3"
OUT_DIR = PAPER_DIR / "results" / "model_rerun_v3"
JOB_DIR = OUT_DIR / "jobs"
CACHE_DIR = OUT_DIR / "fingerprint_cache"
PRED_DIR = OUT_DIR / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunSpec:
    label: str
    task_type: str
    datasets: tuple[str, ...]
    suffix: str
    production_protocols: tuple[str, ...]


SPECS = {
    "main_classification": RunSpec(
        label="main_classification",
        task_type="classification",
        datasets=("BACE", "BBBP", "ClinTox", "HIV"),
        suffix="BACE-BBBP-ClinTox-HIV_single_group_20s_300c",
        production_protocols=(
            "legacy_scaffold",
            "random_observation",
            "size_matched_scaffold",
            "target_balanced_scaffold",
        ),
    ),
    "main_regression": RunSpec(
        label="main_regression",
        task_type="regression",
        datasets=("ESOL", "FreeSolv"),
        suffix="ESOL-FreeSolv_single_group_20s_20000c",
        production_protocols=(
            "legacy_scaffold",
            "random_observation",
            "size_matched_scaffold",
            "target_balanced_scaffold",
        ),
    ),
    "acyclic_singleton_sensitivity": RunSpec(
        label="acyclic_singleton_sensitivity",
        task_type="regression",
        datasets=("ESOL", "FreeSolv"),
        suffix="ESOL-FreeSolv_singleton_20s_5000c",
        production_protocols=(
            "size_matched_scaffold",
            "target_balanced_scaffold",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Paper 1 models from frozen v3 manifests with separated partition/model seeds."
    )
    parser.add_argument(
        "--freeze-label",
        required=True,
        choices=sorted(SPECS),
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--protocols", default="default")
    parser.add_argument("--models", default="all")
    parser.add_argument("--partition-seeds", default="all")
    parser.add_argument(
        "--run-type",
        choices=["production", "model_seed_sensitivity"],
        default="production",
    )
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_seed(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def runtime_metadata() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit_learn": package_version("scikit-learn"),
        "scipy": package_version("scipy"),
        "rdkit": package_version("rdkit"),
        "xgboost": package_version("xgboost"),
    }


def frozen_manifest_path(spec: RunSpec) -> Path:
    return (
        FROZEN_DIR
        / spec.label
        / f"split_manifest_v3_{spec.suffix}.csv"
    )


def parse_selection(raw: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if raw in {"all", "default"}:
        return allowed
    selected = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown selections {unknown}; allowed={list(allowed)}")
    return selected


def selected_partition_seed(seed_key: str, raw: str, *, run_type: str) -> bool:
    if run_type == "model_seed_sensitivity":
        return seed_key in {str(value) for value in MODEL_SEED_SENSITIVITY_PARTITION_SEEDS}
    if raw == "all":
        return True
    requested = {normalize_seed(item) for item in raw.split(",")}
    return seed_key in requested


def model_seed_plan(model_name: str, *, run_type: str) -> tuple[int, ...]:
    if run_type == "production":
        return (production_model_seed(model_name),)
    return sensitivity_model_seeds(model_name)


def job_path(
    *,
    freeze_label: str,
    dataset: str,
    protocol: str,
    partition_seed: str,
    partition_hash: str,
    model_name: str,
    model_seed: int,
    run_type: str,
) -> Path:
    seed_token = partition_seed if partition_seed else "legacy"
    return (
        JOB_DIR
        / run_type
        / freeze_label
        / dataset
        / protocol
        / f"p{seed_token}_{partition_hash[:12]}"
        / model_name
        / f"m{model_seed}.json"
    )


def prediction_path_from_job(path: Path) -> Path:
    relative = path.relative_to(JOB_DIR)
    return (PRED_DIR / relative).with_suffix(".npz")


def load_clean_dataset(dataset: str) -> pd.DataFrame:
    path = DATA_DIR / f"{dataset.lower()}_clean_v2.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing clean_v2 dataset: {path}")
    frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
    required = {"canonical_smiles", "target"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path} missing columns {sorted(missing)}")
    if frame["canonical_smiles"].duplicated().any():
        raise AssertionError(f"Duplicate canonical molecules in {dataset} clean_v2")
    frame["target"] = pd.to_numeric(frame["target"], errors="raise")
    return frame.reset_index(drop=True)


def verify_manifest_targets(manifest: pd.DataFrame, clean: pd.DataFrame, dataset: str) -> None:
    manifest_unique = (
        manifest[["canonical_smiles", "target_numeric"]]
        .drop_duplicates("canonical_smiles")
        .set_index("canonical_smiles")["target_numeric"]
    )
    clean_series = clean.set_index("canonical_smiles")["target"]
    if set(manifest_unique.index.astype(str)) != set(clean_series.index.astype(str)):
        raise AssertionError(f"Frozen manifest molecule universe mismatch for {dataset}")
    aligned = manifest_unique.reindex(clean_series.index)
    if not np.allclose(
        aligned.to_numpy(dtype=float),
        clean_series.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(f"Frozen manifest target mismatch for {dataset}")


def collect_job_index() -> pd.DataFrame:
    rows: list[dict] = []
    if not JOB_DIR.exists():
        return pd.DataFrame()
    for path in JOB_DIR.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["job_file"] = str(path.relative_to(ROOT))
        rows.append(payload)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    spec = SPECS[args.freeze_label]
    protocol_file = PAPER_DIR / "MODEL_PROTOCOL_V3.md"
    metadata_file = FROZEN_DIR / "frozen_metadata_v3.json"
    readiness_file = OUT_DIR / "model_readiness_summary_v3.csv"
    if not protocol_file.exists():
        raise FileNotFoundError(protocol_file)
    if not metadata_file.exists():
        raise FileNotFoundError(metadata_file)
    if not readiness_file.exists():
        raise FileNotFoundError(
            f"Run 08_audit_model_readiness_v3.py before model fitting: {readiness_file}"
        )

    manifest_path = frozen_manifest_path(spec)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(
        manifest_path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    manifest["target_numeric"] = pd.to_numeric(manifest["target"], errors="raise")
    manifest["partition_seed_key"] = manifest["partition_seed"].map(normalize_seed)

    datasets = parse_selection(args.datasets, spec.datasets)
    if args.run_type == "model_seed_sensitivity":
        default_protocols = ("size_matched_scaffold", "target_balanced_scaffold")
    else:
        default_protocols = spec.production_protocols
    protocols = parse_selection(
        args.protocols,
        default_protocols if args.protocols == "default" else spec.production_protocols,
    )
    allowed_models = model_names(spec.task_type)
    models = parse_selection(args.models, allowed_models)
    if args.run_type == "model_seed_sensitivity":
        models = tuple(model for model in models if model in {"RF", "XGB"})
        if not models:
            raise ValueError("Model-seed sensitivity is defined only for RF and XGB")
        protocols = tuple(
            protocol for protocol in protocols
            if protocol in {"size_matched_scaffold", "target_balanced_scaffold"}
        )
        if not protocols:
            raise ValueError("Model-seed sensitivity requires size/balanced scaffold protocols")

    runtime = runtime_metadata()
    run_meta = {
        "run_type": args.run_type,
        "freeze_label": spec.label,
        "task_type": spec.task_type,
        "datasets": list(datasets),
        "protocols": list(protocols),
        "models": list(models),
        "partition_seed_filter": args.partition_seeds,
        "model_protocol_sha256": sha256_file(protocol_file),
        "frozen_metadata_sha256": sha256_file(metadata_file),
        "frozen_manifest_sha256": sha256_file(manifest_path),
        "runtime": runtime,
    }
    run_meta_path = OUT_DIR / f"last_run_metadata_{spec.label}_{args.run_type}.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8")

    completed = 0
    skipped = 0
    attempted = 0
    stop = False

    for dataset in datasets:
        if stop:
            break
        print(f"\n========== {dataset} ==========")
        clean = load_clean_dataset(dataset)
        clean_smiles = clean["canonical_smiles"].astype(str).tolist()
        clean_targets = clean["target"].to_numpy(dtype=float)
        smiles_to_index = {
            smiles: idx for idx, smiles in enumerate(clean_smiles)
        }
        dataset_manifest = manifest.loc[manifest["dataset"].eq(dataset)].copy()
        verify_manifest_targets(dataset_manifest, clean, dataset)
        X, cache_meta = load_or_build_morgan_matrix(
            clean_smiles,
            cache_dir=CACHE_DIR,
            dataset=dataset,
        )
        print(
            f"fingerprints={X.shape}, nnz={X.nnz}, cache={cache_meta['cache_status']}"
        )

        for protocol in protocols:
            if stop:
                break
            protocol_manifest = dataset_manifest.loc[
                dataset_manifest["protocol"].eq(protocol)
            ].copy()
            grouped = protocol_manifest.groupby(
                ["partition_seed_key", "partition_hash"],
                dropna=False,
                sort=False,
            )
            for (partition_seed, partition_hash), group in grouped:
                if stop:
                    break
                partition_seed = normalize_seed(partition_seed)
                if not selected_partition_seed(
                    partition_seed,
                    args.partition_seeds,
                    run_type=args.run_type,
                ):
                    continue
                train_smiles = group.loc[
                    group["assignment"].eq("train"), "canonical_smiles"
                ].astype(str)
                test_smiles = group.loc[
                    group["assignment"].eq("test"), "canonical_smiles"
                ].astype(str)
                train_idx = np.asarray(
                    [smiles_to_index[value] for value in train_smiles], dtype=int
                )
                test_idx = np.asarray(
                    [smiles_to_index[value] for value in test_smiles], dtype=int
                )
                if len(train_idx) + len(test_idx) != len(clean):
                    raise AssertionError(
                        f"Incomplete partition coverage: {dataset}/{protocol}/{partition_seed}"
                    )
                if len(np.intersect1d(train_idx, test_idx)):
                    raise AssertionError(
                        f"Train/test overlap: {dataset}/{protocol}/{partition_seed}"
                    )
                X_train = X[train_idx]
                X_test = X[test_idx]
                y_train = clean_targets[train_idx]
                y_test = clean_targets[test_idx]

                if spec.task_type == "classification":
                    if set(np.unique(y_train).tolist()) != {0.0, 1.0}:
                        raise AssertionError("Classification training split lost a class")
                    if set(np.unique(y_test).tolist()) != {0.0, 1.0}:
                        raise AssertionError("Classification test split lost a class")

                for model_name in models:
                    if stop:
                        break
                    seeds = model_seed_plan(model_name, run_type=args.run_type)
                    if not seeds:
                        continue
                    for model_seed in seeds:
                        if args.max_jobs > 0 and attempted >= args.max_jobs:
                            stop = True
                            break
                        path = job_path(
                            freeze_label=spec.label,
                            dataset=dataset,
                            protocol=protocol,
                            partition_seed=partition_seed,
                            partition_hash=str(partition_hash),
                            model_name=model_name,
                            model_seed=model_seed,
                            run_type=args.run_type,
                        )
                        attempted += 1
                        if path.exists() and not args.force:
                            existing = json.loads(path.read_text(encoding="utf-8"))
                            if (
                                existing.get("partition_hash") == str(partition_hash)
                                and int(existing.get("model_seed")) == int(model_seed)
                            ):
                                skipped += 1
                                print(
                                    f"SKIP {dataset} {protocol} p={partition_seed or 'legacy'} "
                                    f"{model_name} m={model_seed}"
                                )
                                continue
                            raise AssertionError(f"Existing job metadata mismatch: {path}")

                        model = build_model(
                            task_type=spec.task_type,
                            model_name=model_name,
                            model_seed=model_seed,
                            y_train=y_train,
                        )
                        fit_start = time.perf_counter()
                        model.fit(X_train, y_train)
                        fit_seconds = time.perf_counter() - fit_start
                        predict_start = time.perf_counter()
                        if spec.task_type == "classification":
                            predictions = model.predict_proba(X_test)[:, 1]
                            metrics = classification_metrics(y_test, predictions)
                            metrics.update(
                                {
                                    "n_train_positive": int(np.sum(y_train == 1)),
                                    "n_train_negative": int(np.sum(y_train == 0)),
                                }
                            )
                        else:
                            predictions = model.predict(X_test)
                            metrics = regression_metrics(y_test, predictions)
                        predict_seconds = time.perf_counter() - predict_start

                        payload = {
                            "run_type": args.run_type,
                            "freeze_label": spec.label,
                            "dataset": dataset,
                            "task_type": spec.task_type,
                            "protocol": protocol,
                            "partition_seed": partition_seed,
                            "partition_hash": str(partition_hash),
                            "model": model_name,
                            "model_seed": int(model_seed),
                            "n_train": int(len(train_idx)),
                            "n_test": int(len(test_idx)),
                            "fit_seconds": float(fit_seconds),
                            "predict_seconds": float(predict_seconds),
                            "feature_radius": 2,
                            "feature_bits": 2048,
                            "clean_smiles_sha256": cache_meta["canonical_smiles_sha256"],
                            "model_protocol_sha256": run_meta["model_protocol_sha256"],
                            "frozen_manifest_sha256": run_meta["frozen_manifest_sha256"],
                            **metrics,
                        }
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(
                            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
                            encoding="utf-8",
                        )
                        if args.save_predictions:
                            pred_path = prediction_path_from_job(path)
                            pred_path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(
                                pred_path,
                                test_index=test_idx,
                                y_true=y_test,
                                prediction=np.asarray(predictions, dtype=float),
                            )
                        completed += 1
                        primary = metrics["roc_auc"] if spec.task_type == "classification" else metrics["rmse"]
                        primary_name = "ROC-AUC" if spec.task_type == "classification" else "RMSE"
                        print(
                            f"DONE {dataset} {protocol} p={partition_seed or 'legacy'} "
                            f"{model_name} m={model_seed} {primary_name}={primary:.6f} "
                            f"fit={fit_seconds:.2f}s"
                        )

    index = collect_job_index()
    index_path = OUT_DIR / "model_job_index_v3.csv"
    if not index.empty:
        preferred = [
            "run_type", "freeze_label", "dataset", "task_type", "protocol",
            "partition_seed", "partition_hash", "model", "model_seed",
            "n_train", "n_test", "roc_auc", "average_precision", "f1",
            "accuracy", "balanced_accuracy", "brier_score", "rmse", "mae", "r2",
            "fit_seconds", "predict_seconds", "job_file",
        ]
        columns = [column for column in preferred if column in index.columns]
        columns += [column for column in index.columns if column not in columns]
        index = index[columns]
        index.to_csv(index_path, index=False)

    print("\nRun summary:")
    print(f"attempted={attempted}")
    print(f"completed={completed}")
    print(f"skipped_existing={skipped}")
    print(f"job_index={index_path}")
    print("\nMODEL RUNNER V3 COMPLETED")


if __name__ == "__main__":
    main()
