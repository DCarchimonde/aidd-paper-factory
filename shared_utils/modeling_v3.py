"""Model and fingerprint helpers for the frozen Paper 1 v3 rerun."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

FP_RADIUS = 2
FP_BITS = 2048
STOCHASTIC_MODEL_SEEDS = (17, 29, 43)
DETERMINISTIC_MODEL_SEED = 0


def canonical_smiles_hash(smiles: list[str]) -> str:
    payload = "\n".join(str(value) for value in smiles).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_morgan_matrix(smiles: list[str]) -> sparse.csr_matrix:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_BITS,
    )
    row_indices: list[int] = []
    col_indices: list[int] = []
    for row_index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES reached feature generation: {value}")
        fp = generator.GetFingerprint(mol)
        bits = list(fp.GetOnBits())
        row_indices.extend([row_index] * len(bits))
        col_indices.extend(int(bit) for bit in bits)
    data = np.ones(len(row_indices), dtype=np.uint8)
    matrix = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(smiles), FP_BITS),
        dtype=np.uint8,
    )
    return matrix


def load_or_build_morgan_matrix(
    smiles: list[str],
    *,
    cache_dir: Path,
    dataset: str,
) -> tuple[sparse.csr_matrix, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / f"{dataset.lower()}_morgan_r{FP_RADIUS}_{FP_BITS}.npz"
    meta_path = cache_dir / f"{dataset.lower()}_morgan_r{FP_RADIUS}_{FP_BITS}.json"
    expected_hash = canonical_smiles_hash(smiles)
    if matrix_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("canonical_smiles_sha256") == expected_hash
            and int(meta.get("n_rows", -1)) == len(smiles)
            and int(meta.get("radius", -1)) == FP_RADIUS
            and int(meta.get("n_bits", -1)) == FP_BITS
        ):
            matrix = sparse.load_npz(matrix_path).tocsr()
            if matrix.shape == (len(smiles), FP_BITS):
                return matrix, {**meta, "cache_status": "hit"}
    matrix = build_morgan_matrix(smiles)
    sparse.save_npz(matrix_path, matrix, compressed=True)
    meta = {
        "dataset": dataset,
        "n_rows": len(smiles),
        "radius": FP_RADIUS,
        "n_bits": FP_BITS,
        "canonical_smiles_sha256": expected_hash,
        "matrix_nnz": int(matrix.nnz),
        "cache_status": "built",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return matrix, meta


def model_names(task_type: str) -> tuple[str, ...]:
    if task_type == "classification":
        return ("LR", "RF", "XGB")
    if task_type == "regression":
        return ("Ridge", "RF", "XGB")
    raise ValueError(f"Unsupported task type: {task_type}")


def model_seeds(model_name: str) -> tuple[int, ...]:
    if model_name in {"LR", "Ridge"}:
        return (DETERMINISTIC_MODEL_SEED,)
    if model_name in {"RF", "XGB"}:
        return STOCHASTIC_MODEL_SEEDS
    raise ValueError(f"Unknown model: {model_name}")


def _xgb_classifier(*, model_seed: int, scale_pos_weight: float):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required for the frozen v3 model protocol. Install/import xgboost in the active environment."
        ) from exc
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=int(model_seed),
        scale_pos_weight=float(scale_pos_weight),
        verbosity=0,
    )


def _xgb_regressor(*, model_seed: int):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required for the frozen v3 model protocol. Install/import xgboost in the active environment."
        ) from exc
    return XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=int(model_seed),
        verbosity=0,
    )


def build_model(
    *,
    task_type: str,
    model_name: str,
    model_seed: int,
    y_train: np.ndarray,
):
    if task_type == "classification":
        if model_name == "LR":
            return LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=0,
            )
        if model_name == "RF":
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=int(model_seed),
            )
        if model_name == "XGB":
            y = np.asarray(y_train, dtype=int)
            n_pos = int(np.sum(y == 1))
            n_neg = int(np.sum(y == 0))
            if n_pos <= 0 or n_neg <= 0:
                raise ValueError("XGBoost classification requires both classes in training data")
            return _xgb_classifier(
                model_seed=model_seed,
                scale_pos_weight=n_neg / n_pos,
            )
    elif task_type == "regression":
        if model_name == "Ridge":
            return Ridge(alpha=1.0, solver="lsqr")
        if model_name == "RF":
            return RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=int(model_seed),
            )
        if model_name == "XGB":
            return _xgb_regressor(model_seed=model_seed)
    raise ValueError(f"Unsupported task/model combination: {task_type}/{model_name}")


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    labels = (p >= 0.5).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Classification test partition contains only one class")
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, labels, zero_division=0)),
        "accuracy": float(accuracy_score(y, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y, labels)),
        "brier_score": float(brier_score_loss(y, p)),
        "n_test_positive": int(np.sum(y == 1)),
        "n_test_negative": int(np.sum(y == 0)),
    }


def regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }
