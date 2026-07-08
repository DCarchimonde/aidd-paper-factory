from __future__ import annotations

"""Train baseline models for Paper 2.

This script is intentionally staged to avoid overwhelming a local Windows laptop.

Modes:
- smoke: quick bug check on BBBP + ESOL, random seed 0, linear + random forest.
- mvp: core endpoints, random seed 0 + scaffold, all baseline model families.
- full: all endpoints, all random seeds + scaffold, all baseline model families.

Inputs:
- paper2_admet_benchmark/data/processed/<endpoint>_features.npz
- paper2_admet_benchmark/data/processed/<endpoint>_splits.csv

Outputs:
- paper2_admet_benchmark/results/metrics/baseline_metrics_<mode>.csv
- paper2_admet_benchmark/results/predictions/*_predictions.csv
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
TABLE_DIR = PAPER_DIR / "results" / "tables"

SMOKE_ENDPOINTS = ["bbbp", "esol"]
MVP_ENDPOINTS = ["bbbp", "clintox", "esol", "lipophilicity"]
ALL_RANDOM_SPLITS = [f"split_random_seed{i}" for i in range(5)]
SMOKE_SPLITS = ["split_random_seed0"]
MVP_SPLITS = ["split_random_seed0", "split_scaffold"]

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Paper 2 baseline models.")
    parser.add_argument(
        "--mode",
        choices=["smoke", "mvp", "full"],
        default="smoke",
        help="Training scope. Start with smoke before mvp or full.",
    )
    parser.add_argument(
        "--endpoints",
        default=None,
        help="Optional comma-separated endpoint names, e.g. bbbp,esol. Overrides mode endpoint defaults.",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Optional comma-separated split columns. Overrides mode split defaults.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated model names. Overrides mode model defaults.",
    )
    return parser.parse_args()


def available_endpoints() -> list[str]:
    return sorted(path.name.replace("_features.npz", "") for path in PROCESSED_DIR.glob("*_features.npz"))


def resolve_endpoints(mode: str, endpoints_arg: str | None) -> list[str]:
    if endpoints_arg:
        return [item.strip().lower() for item in endpoints_arg.split(",") if item.strip()]
    if mode == "smoke":
        return SMOKE_ENDPOINTS
    if mode == "mvp":
        return MVP_ENDPOINTS
    return available_endpoints()


def resolve_splits(mode: str, splits_arg: str | None) -> list[str]:
    if splits_arg:
        return [item.strip() for item in splits_arg.split(",") if item.strip()]
    if mode == "smoke":
        return SMOKE_SPLITS
    if mode == "mvp":
        return MVP_SPLITS
    return ALL_RANDOM_SPLITS + ["split_scaffold"]


def classification_models(random_state: int, mode: str) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200 if mode == "smoke" else 500,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if mode != "smoke":
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        models["mlp_ecfp"] = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        alpha=1e-4,
                        max_iter=300,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return models


def regression_models(random_state: int, mode: str) -> dict[str, Any]:
    models: dict[str, Any] = {
        "ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200 if mode == "smoke" else 500,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if mode != "smoke":
        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        models["mlp_ecfp"] = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        alpha=1e-4,
                        max_iter=300,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return models


def filter_models(model_dict: dict[str, Any], models_arg: str | None) -> dict[str, Any]:
    if not models_arg:
        return model_dict
    requested = {item.strip() for item in models_arg.split(",") if item.strip()}
    missing = requested - set(model_dict.keys())
    if missing:
        raise KeyError(f"Requested models are not available in this mode/task: {sorted(missing)}")
    return {name: model for name, model in model_dict.items() if name in requested}


def infer_seed_from_split(split_col: str) -> int:
    if "seed" not in split_col:
        return 0
    try:
        return int(split_col.rsplit("seed", maxsplit=1)[1])
    except ValueError:
        return 0


def safe_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": math.nan,
        "pr_auc": math.nan,
        "brier_score": math.nan,
        "negative_log_likelihood": math.nan,
    }
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    if len(y_score):
        clipped = np.clip(y_score, 1e-7, 1 - 1e-7)
        metrics["brier_score"] = float(brier_score_loss(y_true, clipped))
        if len(np.unique(y_true)) >= 2:
            metrics["negative_log_likelihood"] = float(log_loss(y_true, clipped, labels=[0, 1]))
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else math.nan,
    }


def predict_classification(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            score = proba[:, 1]
        else:
            score = proba.max(axis=1)
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        score = 1.0 / (1.0 + np.exp(-raw))
    else:
        score = model.predict(X).astype(float)
    pred = (score >= 0.5).astype(int)
    return score, pred


def save_predictions(
    endpoint: str,
    task_type: str,
    split_col: str,
    role: str,
    model_name: str,
    split_df: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_output: np.ndarray,
    y_pred_label: np.ndarray | None = None,
) -> Path:
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "endpoint": endpoint,
            "task_type": task_type,
            "split_col": split_col,
            "role": role,
            "model": model_name,
            "row_index": indices,
            "canonical_smiles": split_df.iloc[indices]["canonical_smiles"].to_numpy(),
            "y_true": y_true,
            "y_output": y_output,
        }
    )
    if y_pred_label is not None:
        out["y_pred_label"] = y_pred_label

    safe_split = split_col.replace("split_", "")
    out_path = PREDICTION_DIR / f"{endpoint}_{safe_split}_{model_name}_{role}_predictions.csv"
    out.to_csv(out_path, index=False)
    return out_path


def train_one_endpoint(
    endpoint: str,
    split_cols: list[str],
    mode: str,
    models_arg: str | None,
) -> list[dict]:
    feature_path = PROCESSED_DIR / f"{endpoint}_features.npz"
    split_path = PROCESSED_DIR / f"{endpoint}_splits.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    data = np.load(feature_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"]
    split_df = pd.read_csv(split_path)
    task_type = str(split_df["task_type"].iloc[0])

    if len(split_df) != X.shape[0]:
        raise ValueError(f"{endpoint}: split rows {len(split_df)} != feature rows {X.shape[0]}")

    metric_rows: list[dict] = []
    for split_col in split_cols:
        if split_col not in split_df.columns:
            print(f"[SKIP] {endpoint}: missing split column {split_col}")
            continue

        seed = infer_seed_from_split(split_col)
        if task_type == "classification":
            model_dict = classification_models(random_state=seed, mode=mode)
        else:
            model_dict = regression_models(random_state=seed, mode=mode)
        model_dict = filter_models(model_dict, models_arg)

        train_idx = split_df.index[split_df[split_col] == "train"].to_numpy()
        calibration_idx = split_df.index[split_df[split_col] == "calibration"].to_numpy()
        test_idx = split_df.index[split_df[split_col] == "test"].to_numpy()

        if min(len(train_idx), len(calibration_idx), len(test_idx)) == 0:
            raise ValueError(f"{endpoint} {split_col}: train/calibration/test must all be non-empty.")

        X_train = X[train_idx]
        y_train = y[train_idx]

        for model_name, model in model_dict.items():
            print(f"training endpoint={endpoint} split={split_col} model={model_name}")
            t0 = time.time()
            model.fit(X_train, y_train)
            fit_seconds = time.time() - t0

            for role, indices in [("calibration", calibration_idx), ("test", test_idx)]:
                X_eval = X[indices]
                y_eval = y[indices]
                row = {
                    "endpoint": endpoint,
                    "task_type": task_type,
                    "split_col": split_col,
                    "role": role,
                    "model": model_name,
                    "mode": mode,
                    "n_train": int(len(train_idx)),
                    "n_eval": int(len(indices)),
                    "fit_seconds": float(fit_seconds),
                }

                if task_type == "classification":
                    y_score, y_pred = predict_classification(model, X_eval)
                    row.update(safe_binary_metrics(y_eval.astype(int), y_score, y_pred))
                    pred_path = save_predictions(
                        endpoint=endpoint,
                        task_type=task_type,
                        split_col=split_col,
                        role=role,
                        model_name=model_name,
                        split_df=split_df,
                        indices=indices,
                        y_true=y_eval.astype(int),
                        y_output=y_score,
                        y_pred_label=y_pred,
                    )
                else:
                    y_pred = model.predict(X_eval)
                    row.update(regression_metrics(y_eval.astype(float), y_pred.astype(float)))
                    pred_path = save_predictions(
                        endpoint=endpoint,
                        task_type=task_type,
                        split_col=split_col,
                        role=role,
                        model_name=model_name,
                        split_df=split_df,
                        indices=indices,
                        y_true=y_eval.astype(float),
                        y_output=y_pred.astype(float),
                    )
                row["prediction_file"] = str(pred_path.relative_to(ROOT))
                metric_rows.append(row)

    return metric_rows


def main() -> None:
    args = parse_args()
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    endpoints = resolve_endpoints(args.mode, args.endpoints)
    split_cols = resolve_splits(args.mode, args.splits)

    print("Paper 2 baseline training")
    print("mode:", args.mode)
    print("endpoints:", ", ".join(endpoints))
    print("splits:", ", ".join(split_cols))
    print()

    all_rows: list[dict] = []
    for endpoint in endpoints:
        print("\n==========", endpoint, "==========")
        rows = train_one_endpoint(endpoint, split_cols, mode=args.mode, models_arg=args.models)
        all_rows.extend(rows)

    metrics = pd.DataFrame(all_rows)
    out_path = METRIC_DIR / f"baseline_metrics_{args.mode}.csv"
    metrics.to_csv(out_path, index=False)

    # Also write/update a latest file for convenience.
    latest_path = METRIC_DIR / "baseline_metrics_latest.csv"
    metrics.to_csv(latest_path, index=False)

    print("\nsaved", out_path)
    print("saved", latest_path)
    print("Training complete.")
    print("Next: inspect baseline metrics, then implement calibration and conformal scripts.")


if __name__ == "__main__":
    main()
