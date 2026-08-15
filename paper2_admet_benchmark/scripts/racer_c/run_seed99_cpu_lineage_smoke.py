from __future__ import annotations

"""Development-only CPU smoke for nested OOF lineage; never scores outer roles."""

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import rdkit
import sklearn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from lineage_contract import FitNode, PredictionNode, validate_prediction_lineage
from role_feasibility import allocate_groups, read_csv, validate_role_input


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
DEFAULT_ROLE_INPUT = (
    P2 / "data" / "processed" / "racer_c" / "role_inputs" / "Tox21_NR_ER_role_input.csv"
)
DEFAULT_CLEAN = P2 / "data" / "processed" / "racer_c" / "Tox21_NR_ER_clean.csv"
DEFAULT_OUTPUT = P2 / "results" / "racer_c_phase2_preflight" / "seed99_cpu_lineage_smoke.json"
FRACTIONS = {"dev": 0.50, "policy": 0.20, "conformal": 0.15, "test": 0.15}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_clean_dev(path: Path, dev_ids: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["structure_id"] in dev_ids]
    if {row["structure_id"] for row in rows} != dev_ids:
        raise ValueError("clean file does not contain every allocated development row")
    return sorted(rows, key=lambda row: row["structure_id"])


def fingerprints(smiles: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    output = np.empty((len(smiles), 2048), dtype=np.uint8)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"standardized SMILES failed to parse at development row {index}")
        output[index] = generator.GetFingerprintAsNumPy(molecule)
    return output


def fit_logistic(x: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    if len(np.unique(y)) != 2:
        raise ValueError("smoke fold contains only one class")
    return LogisticRegression(
        solver="liblinear", max_iter=500, random_state=seed
    ).fit(x, y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-input", type=Path, default=DEFAULT_ROLE_INPUT)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    role_rows = validate_role_input(read_csv(args.role_input))
    allocation = allocate_groups(
        role_rows,
        "murcko_scaffold_id",
        FRACTIONS,
        99,
        use_labels_for_assignment=False,
    )
    role_by_id = {
        row["structure_id"]: allocation.assignment[row["murcko_scaffold_id"]]
        for row in role_rows
    }
    dev_ids = {structure_id for structure_id, role in role_by_id.items() if role == "dev"}
    dev_rows = read_clean_dev(args.clean, dev_ids)
    y = np.asarray([int(row["target"]) for row in dev_rows], dtype=np.int8)
    groups = np.asarray([row["murcko_scaffold_id"] for row in dev_rows])
    structure_ids = np.asarray([row["structure_id"] for row in dev_rows])
    feature_started = time.perf_counter()
    x = fingerprints([row["standardized_smiles"] for row in dev_rows])
    feature_seconds = time.perf_counter() - feature_started

    outer = GroupKFold(n_splits=3)
    oof = np.full(len(dev_rows), np.nan, dtype=float)
    fit_nodes: list[FitNode] = []
    prediction_nodes: list[PredictionNode] = []
    fold_sizes: list[dict[str, int]] = []
    fit_started = time.perf_counter()
    for outer_fold, (outer_train, outer_valid) in enumerate(
        outer.split(x, y, groups), start=1
    ):
        inner_oof = np.full(len(outer_train), np.nan, dtype=float)
        inner_groups = groups[outer_train]
        inner_dependencies: list[str] = []
        for inner_fold, (inner_train_local, inner_valid_local) in enumerate(
            GroupKFold(n_splits=2).split(
                x[outer_train], y[outer_train], inner_groups
            ),
            start=1,
        ):
            train_idx = outer_train[inner_train_local]
            valid_idx = outer_train[inner_valid_local]
            node_id = f"outer{outer_fold}_inner{inner_fold}_base"
            model = fit_logistic(x[train_idx], y[train_idx], 9900 + 10 * outer_fold + inner_fold)
            inner_oof[inner_valid_local] = model.predict_proba(x[valid_idx])[:, 1]
            fit_nodes.append(
                FitNode(node_id, "ecfp_logistic", frozenset(structure_ids[train_idx]))
            )
            inner_dependencies.append(node_id)
        if not np.isfinite(inner_oof).all():
            raise AssertionError("inner OOF predictions are incomplete")
        clipped = np.clip(inner_oof, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        calibrator = fit_logistic(logits, y[outer_train], 9950 + outer_fold)
        calibrator_id = f"outer{outer_fold}_platt"
        fit_nodes.append(
            FitNode(
                calibrator_id,
                "platt_on_inner_oof",
                frozenset(structure_ids[outer_train]),
                tuple(inner_dependencies),
            )
        )
        final_base_id = f"outer{outer_fold}_final_base"
        final_base = fit_logistic(x[outer_train], y[outer_train], 9990 + outer_fold)
        fit_nodes.append(
            FitNode(
                final_base_id,
                "ecfp_logistic",
                frozenset(structure_ids[outer_train]),
            )
        )
        raw = np.clip(final_base.predict_proba(x[outer_valid])[:, 1], 1e-6, 1 - 1e-6)
        calibrated = calibrator.predict_proba(
            np.log(raw / (1 - raw)).reshape(-1, 1)
        )[:, 1]
        oof[outer_valid] = calibrated
        for index in outer_valid:
            prediction_nodes.append(
                PredictionNode(
                    f"dev_oof_{structure_ids[index]}",
                    str(structure_ids[index]),
                    "dev",
                    (final_base_id, calibrator_id),
                )
            )
        fold_sizes.append(
            {"outer_fold": outer_fold, "train_n": len(outer_train), "valid_n": len(outer_valid)}
        )
    fit_seconds = time.perf_counter() - fit_started
    if not np.isfinite(oof).all() or not np.logical_and(oof > 0, oof < 1).all():
        raise AssertionError("outer OOF probabilities are incomplete or non-finite")
    resolved = validate_prediction_lineage(fit_nodes, prediction_nodes, role_by_id)
    for node in prediction_nodes:
        if node.row_id in resolved[node.prediction_id]:
            raise AssertionError(f"self leakage survived validation: {node.row_id}")

    result = {
        "status": "pass_development_only_cpu_smoke",
        "endpoint": "Tox21_NR_ER",
        "seed": 99,
        "track": "strict_scaffold",
        "allocation": "50_20_15_15",
        "outer_role_labels_used_by_trainer": ["dev"],
        "policy_conformal_test_predictions_generated": False,
        "performance_metrics_computed": False,
        "dev_n": len(dev_rows),
        "dev_class_0_n": int((y == 0).sum()),
        "dev_class_1_n": int((y == 1).sum()),
        "outer_oof_prediction_count": int(np.isfinite(oof).sum()),
        "outer_folds": fold_sizes,
        "fit_node_count": len(fit_nodes),
        "prediction_lineage_node_count": len(prediction_nodes),
        "feature_seconds": round(feature_seconds, 6),
        "fit_seconds": round(fit_seconds, 6),
        "total_seconds": round(time.perf_counter() - started, 6),
        "role_input_sha256": sha256_file(args.role_input),
        "clean_input_sha256": sha256_file(args.clean),
        "script_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "interpretation": "technical lineage integration only; not a GPU benchmark or scientific model result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
