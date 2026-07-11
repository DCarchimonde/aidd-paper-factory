from __future__ import annotations

"""Train frozen confirmatory baseline models under unified imbalance regimes.

Classification models are evaluated in two separate regimes:

1. unweighted: every model is trained on the original training distribution with no
   class weights;
2. balanced_oversample: the minority class is randomly oversampled within the
   training set to match the majority class, using the split seed. The same resampled
   training data is supplied to every model family.

Regression models are trained once with regime `not_applicable`.

This script consumes only confirmatory split columns created by scripts 14 and 15.
It never uses calibration or test labels during fitting or resampling.
"""

import argparse
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper2_admet_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
METRIC_DIR = PAPER_DIR / "results" / "metrics"
PREDICTION_DIR = PAPER_DIR / "results" / "predictions"

CORE_ENDPOINTS = ["bbbp", "clintox", "esol", "lipophilicity"]
DEFAULT_CONFIRMATORY_SEEDS = list(range(101, 111))
DEFAULT_CLUSTER_SEEDS = list(range(101, 106))
DEFAULT_SPLIT_TYPES = ["random", "scaffold", "cluster"]
DEFAULT_CLASSIFICATION_REGIMES = ["unweighted", "balanced_oversample"]
MODEL_NAMES = ["linear", "random_forest", "xgboost", "mlp_ecfp"]

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen confirmatory molecular baselines.")
    parser.add_argument("--scope", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--endpoints",
        default=",".join(CORE_ENDPOINTS),
        help="Comma-separated endpoints, or 'all'.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds. Defaults to 101 for smoke and 101-110 for full.",
    )
    parser.add_argument(
        "--split-types",
        default=",".join(DEFAULT_SPLIT_TYPES),
        help="Subset of random,scaffold,cluster.",
    )
    parser.add_argument(
        "--regimes",
        default=",".join(DEFAULT_CLASSIFICATION_REGIMES),
        help="Classification regimes: unweighted,balanced_oversample.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_NAMES),
        help="Subset of linear,random_forest,xgboost,mlp_ecfp.",
    )
    return parser.parse_args()


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def available_endpoints() -> list[str]:
    return sorted(path.name.replace("_features.npz", "") for path in PROCESSED_DIR.glob("*_features.npz"))


def resolve_endpoints(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return available_endpoints()
    return [item.lower() for item in parse_list(value)]


def resolve_seeds(scope: str, value: str | None) -> list[int]:
    if value:
        return [int(item) for item in parse_list(value)]
    return [101] if scope == "smoke" else DEFAULT_CONFIRMATORY_SEEDS


def resolve_split_columns(split_types: list[str], seeds: list[int]) -> list[str]:
    invalid = set(split_types) - set(DEFAULT_SPLIT_TYPES)
    if invalid:
        raise ValueError(f"Unknown split types: {sorted(invalid)}")
    columns: list[str] = []
    for seed in seeds:
        if "random" in split_types:
            columns.append(f"split_confirm_random_seed{seed}")
        if "scaffold" in split_types:
            columns.append(f"split_confirm_scaffold_seed{seed}")
        if "cluster" in split_types and seed in DEFAULT_CLUSTER_SEEDS:
            columns.append(f"split_confirm_cluster_seed{seed}")
    return columns


def infer_seed(split_col: str) -> int:
    return int(split_col.rsplit("seed", 1)[1])


def classification_models(seed: int) -> dict[str, Any]:
    return {
        "linear": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="liblinear",
                        class_weight=None,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            class_weight=None,
            random_state=seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        ),
        "mlp_ecfp": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        alpha=1e-4,
                        max_iter=300,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def regression_models(seed: int) -> dict[str, Any]:
    return {
        "linear": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        ),
        "mlp_ecfp": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        alpha=1e-4,
                        max_iter=300,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def filter_models(models: dict[str, Any], requested: list[str]) -> dict[str, Any]:
    invalid = set(requested) - set(models)
    if invalid:
        raise ValueError(f"Unknown model names: {sorted(invalid)}")
    return {name: models[name] for name in requested}


def oversample_binary_training(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_int = y.astype(int)
    classes, counts = np.unique(y_int, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"Balanced oversampling requires two training classes, got {classes.tolist()}")

    target_n = int(counts.max())
    rng = np.random.default_rng(seed)
    sampled_indices: list[np.ndarray] = []
    for class_label, count in zip(classes, counts):
        class_idx = np.flatnonzero(y_int == class_label)
        if count < target_n:
            extra = rng.choice(class_idx, size=target_n - int(count), replace=True)
            class_idx = np.concatenate([class_idx, extra])
        sampled_indices.append(class_idx)

    combined = np.concatenate(sampled_indices)
    rng.shuffle(combined)
    return X[combined], y[combined]


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_score = np.clip(y_score.astype(float), 1e-7, 1 - 1e-7)
    y_pred = (y_score >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": math.nan,
        "pr_auc": math.nan,
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "negative_log_likelihood": math.nan,
    }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out["negative_log_likelihood"] = float(log_loss(y_true, y_score, labels=[0, 1]))
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else math.nan,
    }


def classification_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=float)
    raw = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=float)))


def safe_name(value: str) -> str:
    return value.replace("split_", "").replace("/", "_").replace("\\", "_")


def save_predictions(
    endpoint: str,
    task_type: str,
    split_col: str,
    role: str,
    model_key: str,
    base_model: str,
    imbalance_regime: str,
    split_df: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_output: np.ndarray,
) -> Path:
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "endpoint": endpoint,
            "task_type": task_type,
            "split_col": split_col,
            "role": role,
            "model": model_key,
            "base_model": base_model,
            "imbalance_regime": imbalance_regime,
            "row_index": indices,
            "canonical_smiles": split_df.iloc[indices]["canonical_smiles"].to_numpy(),
            "y_true": y_true,
            "y_output": y_output,
        }
    )
    if task_type == "classification":
        out["y_pred_label"] = (np.asarray(y_output) >= 0.5).astype(int)
    path = PREDICTION_DIR / (
        f"{endpoint}_{safe_name(split_col)}_{model_key}_{role}_predictions.csv"
    )
    out.to_csv(path, index=False)
    return path


def train_endpoint(
    endpoint: str,
    split_columns: list[str],
    requested_models: list[str],
    classification_regimes: list[str],
) -> list[dict[str, object]]:
    feature_path = PROCESSED_DIR / f"{endpoint}_features.npz"
    split_path = PROCESSED_DIR / f"{endpoint}_splits.csv"
    if not feature_path.exists() or not split_path.exists():
        raise FileNotFoundError(f"Missing feature or split file for endpoint={endpoint}")

    feature_data = np.load(feature_path, allow_pickle=True)
    X = feature_data["X"].astype(np.float32)
    y = feature_data["y"]
    split_df = pd.read_csv(split_path)
    task_type = str(split_df["task_type"].iloc[0])
    if len(split_df) != len(X):
        raise ValueError(f"{endpoint}: split rows {len(split_df)} != feature rows {len(X)}")

    metric_rows: list[dict[str, object]] = []
    for split_col in split_columns:
        if split_col not in split_df.columns:
            print(f"[SKIP] {endpoint}: missing {split_col}")
            continue

        seed = infer_seed(split_col)
        train_idx = split_df.index[split_df[split_col] == "train"].to_numpy()
        calibration_idx = split_df.index[split_df[split_col] == "calibration"].to_numpy()
        test_idx = split_df.index[split_df[split_col] == "test"].to_numpy()
        if min(len(train_idx), len(calibration_idx), len(test_idx)) == 0:
            print(f"[SKIP] {endpoint} {split_col}: an empty role was produced")
            continue

        X_train_raw = X[train_idx]
        y_train_raw = y[train_idx]
        if task_type == "classification" and len(np.unique(y_train_raw.astype(int))) < 2:
            print(f"[SKIP] {endpoint} {split_col}: training role is single-class")
            continue

        regimes = classification_regimes if task_type == "classification" else ["not_applicable"]
        for regime in regimes:
            if task_type == "classification" and regime == "balanced_oversample":
                X_train, y_train = oversample_binary_training(X_train_raw, y_train_raw, seed=seed)
            else:
                X_train, y_train = X_train_raw, y_train_raw

            model_dict = (
                classification_models(seed)
                if task_type == "classification"
                else regression_models(seed)
            )
            model_dict = filter_models(model_dict, requested_models)

            for base_model, model in model_dict.items():
                model_key = f"{base_model}__{regime}"
                print(
                    f"training endpoint={endpoint} split={split_col} "
                    f"model={base_model} regime={regime} n_train={len(y_train)}"
                )
                t0 = time.time()
                model.fit(X_train, y_train)
                fit_seconds = time.time() - t0

                for role, indices in [("calibration", calibration_idx), ("test", test_idx)]:
                    X_eval = X[indices]
                    y_eval = y[indices]
                    if task_type == "classification":
                        output = classification_scores(model, X_eval)
                        metrics = classification_metrics(y_eval, output)
                    else:
                        output = np.asarray(model.predict(X_eval), dtype=float)
                        metrics = regression_metrics(y_eval.astype(float), output)

                    prediction_path = save_predictions(
                        endpoint=endpoint,
                        task_type=task_type,
                        split_col=split_col,
                        role=role,
                        model_key=model_key,
                        base_model=base_model,
                        imbalance_regime=regime,
                        split_df=split_df,
                        indices=indices,
                        y_true=y_eval,
                        y_output=output,
                    )
                    row: dict[str, object] = {
                        "endpoint": endpoint,
                        "task_type": task_type,
                        "split_col": split_col,
                        "role": role,
                        "model": model_key,
                        "base_model": base_model,
                        "imbalance_regime": regime,
                        "seed": seed,
                        "n_train_original": int(len(y_train_raw)),
                        "n_train_effective": int(len(y_train)),
                        "n_eval": int(len(indices)),
                        "fit_seconds": float(fit_seconds),
                        "prediction_file": str(prediction_path.relative_to(ROOT)),
                    }
                    row.update(metrics)
                    metric_rows.append(row)
    return metric_rows


def main() -> None:
    args = parse_args()
    endpoints = resolve_endpoints(args.endpoints)
    seeds = resolve_seeds(args.scope, args.seeds)
    split_types = parse_list(args.split_types)
    split_columns = resolve_split_columns(split_types, seeds)
    requested_models = parse_list(args.models)
    regimes = parse_list(args.regimes)

    invalid_regimes = set(regimes) - set(DEFAULT_CLASSIFICATION_REGIMES)
    if invalid_regimes:
        raise ValueError(f"Unknown classification regimes: {sorted(invalid_regimes)}")

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    print("Confirmatory baseline training")
    print("scope:", args.scope)
    print("endpoints:", ", ".join(endpoints))
    print("split columns:", ", ".join(split_columns))
    print("classification regimes:", ", ".join(regimes))
    print("models:", ", ".join(requested_models))

    rows: list[dict[str, object]] = []
    for endpoint in endpoints:
        print(f"\n========== {endpoint} ==========")
        rows.extend(
            train_endpoint(
                endpoint=endpoint,
                split_columns=split_columns,
                requested_models=requested_models,
                classification_regimes=regimes,
            )
        )

    metrics = pd.DataFrame(rows)
    output_path = METRIC_DIR / f"baseline_metrics_confirmatory_{args.scope}.csv"
    metrics.to_csv(output_path, index=False)
    print("\nsaved", output_path)
    print(f"Confirmatory training complete. Metric rows: {len(metrics)}.")


if __name__ == "__main__":
    main()
