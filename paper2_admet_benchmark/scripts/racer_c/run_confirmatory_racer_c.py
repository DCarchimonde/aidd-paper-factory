from __future__ import annotations

"""Fail-closed, resumable RACER-C v1.0 confirmatory orchestrator.

The runner has two irreversible scientific stages after the protocol tag:
1. honest development OOF plus deployment predictions for all 60 primary cells;
2. one global development-only attenuation choice, followed by policy selection,
   conformal calibration, and a single test evaluation.

Cell failures are retained and never removed from the denominator.  A rerun only
skips artifacts whose manifest hashes and row-count contracts still validate.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from molformer_token_contract import filter_model_eligible_rows, verify_runtime_tokenizer
from policy_selection import PolicyConstraints, select_policy
from prepare_seed99_gpu_benchmark import label_blind_group_folds
from racer_c_production_core import (
    apply_platt,
    attenuate,
    bri_predict,
    class_midrank_percentiles,
    conformal_thresholds,
    fit_bri,
    fit_platt,
    fit_rcp_transform,
    fit_stacker,
    logits,
    metric_record,
    prediction_sets,
    reliability_features,
    stack_probability,
    state_labels,
    tanimoto_topk_local_loss,
)
from role_feasibility import (
    TRACK_GROUP_COLUMN,
    allocate_groups,
    audit_one,
    read_csv,
    validate_role_input,
)


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
DEFAULT_PRODUCTION_LOCK = P2 / "configs" / "racer_c" / "production_lock_v1.yaml"
DEFAULT_GPU_LOCK = P2 / "configs" / "racer_c" / "gpu_environment_lock.yaml"
DEFAULT_REVIEW = (
    P2 / "results" / "racer_c_phase4_freeze_review" / "formal_freeze_review_windows_rtx4060.json"
)
DEFAULT_OUTPUT = P2 / "results" / "racer_c_confirmatory_v1"
ROLE_INPUT_DIR = P2 / "data" / "processed" / "racer_c" / "role_inputs"
PROCESSED_DIR = P2 / "data" / "processed" / "racer_c"
MANIFEST_DIR = P2 / "data" / "manifests" / "racer_c"
ENDPOINT_MANIFEST = P2 / "protocols" / "endpoint_candidate_manifest.csv"
EXPECTED_TAG = "paper2-racer-protocol-freeze-v1.0"
RAW_FIELDS = [
    "structure_id", "role", "target", "meta_fold", "ecfp_p", "dmpnn_p",
    "molformer_p", "heterogeneous_p", "stack_p", "unrestricted_p", "bri",
    "risk_percentile", "absolute_margin", "disagreement", "ecfp_distance",
    "local_oof_brier_loss",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.npy")
    np.save(temporary, values, allow_pickle=False)
    os.replace(temporary, path)


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def verify_protocol_tag(allow_unfrozen_for_tests: bool) -> dict[str, str]:
    head = git_output("rev-parse", "HEAD")
    if allow_unfrozen_for_tests:
        return {"head": head, "tag": "test-bypass", "tag_commit": head}
    tags = set(git_output("tag", "--points-at", "HEAD").splitlines())
    if EXPECTED_TAG not in tags:
        raise RuntimeError(
            f"HEAD {head} is not the frozen protocol tag {EXPECTED_TAG}; refusing predictions"
        )
    tag_commit = git_output("rev-list", "-n", "1", EXPECTED_TAG)
    if tag_commit != head:
        raise RuntimeError("protocol tag does not resolve to the checked-out commit")
    return {"head": head, "tag": EXPECTED_TAG, "tag_commit": tag_commit}


def verify_freeze_review(path: Path, lock: Mapping[str, object]) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"formal freeze review is missing: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    required_endpoints = list(lock["primary_endpoints"])
    failures: list[str] = []
    if row.get("status") != lock["formal_review_status"]:
        failures.append("status")
    if row.get("scientific_predictions_generated") is not False:
        failures.append("scientific_predictions_generated")
    if row.get("performance_metrics_computed") is not False:
        failures.append("performance_metrics_computed")
    if row.get("selection_uses_model_outputs") is not False:
        failures.append("selection_uses_model_outputs")
    if int(row.get("track_seed_cell_count", -1)) != int(lock["formal_review_required_cells"]):
        failures.append("track_seed_cell_count")
    if row.get("primary_endpoints") != required_endpoints:
        failures.append("primary_endpoints")
    if row.get("tracks") != list(lock["tracks"]):
        failures.append("tracks")
    if row.get("main_split_seeds") != list(lock["main_split_seeds"]):
        failures.append("main_split_seeds")
    expected_allocation = "_".join(
        str(round(100 * float(lock["outer_roles"][role])))
        for role in ("dev", "policy", "conformal", "test")
    )
    if row.get("allocation") != expected_allocation:
        failures.append("allocation")
    cell_hash = str(row.get("track_seed_cells_sha256", ""))
    if len(cell_hash) != 64 or any(value not in "0123456789abcdef" for value in cell_hash):
        failures.append("track_seed_cells_sha256")
    endpoint_reviews = row.get("endpoint_reviews", [])
    if len(endpoint_reviews) != len(required_endpoints) or any(
        item.get("eligibility_status_after_model_domain") != "primary_freeze_ready"
        or int(item.get("passing_track_seed_cells", -1)) != 15
        for item in endpoint_reviews
    ):
        failures.append("endpoint_reviews")
    if failures:
        raise RuntimeError("formal freeze review contract failed: " + ", ".join(failures))
    return row


def endpoint_critical_classes(path: Path, endpoints: Sequence[str]) -> dict[str, int]:
    rows = read_table(path)
    selected = {row["endpoint"]: row for row in rows if row["endpoint"] in endpoints}
    if set(selected) != set(endpoints):
        raise RuntimeError("endpoint manifest does not contain the frozen primary panel")
    output: dict[str, int] = {}
    for endpoint in endpoints:
        row = selected[endpoint]
        if row["eligibility_status"] != "primary_candidate":
            raise RuntimeError(f"{endpoint} is no longer a primary candidate")
        output[endpoint] = int(row["critical_class"])
    return output


def load_endpoint(endpoint: str, tokenizer: object, gpu_lock: Mapping[str, object]):
    role_path = ROLE_INPUT_DIR / f"{endpoint}_role_input.csv"
    clean_path = PROCESSED_DIR / f"{endpoint}_clean.csv"
    manifest_path = MANIFEST_DIR / f"{endpoint}_cleaning.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path, key in ((role_path, "role_input_byte_sha256"), (clean_path, "cleaned_byte_sha256")):
        observed = sha256_file(path)
        if observed != manifest[key]:
            raise RuntimeError(f"{endpoint} locked input mismatch: {path} expected={manifest[key]} observed={observed}")
    role_rows = validate_role_input(read_csv(role_path))
    by_id = {row["structure_id"]: row for row in read_table(clean_path)}
    if set(by_id) != {row["structure_id"] for row in role_rows}:
        raise RuntimeError(f"{endpoint} clean/role identity mismatch")
    clean_rows = [by_id[row["structure_id"]] for row in role_rows]
    verify_runtime_tokenizer(clean_rows, tokenizer, str(gpu_lock["molformer"]["input_column"]))
    role_rows, clean_rows, eligibility = filter_model_eligible_rows(role_rows, clean_rows, gpu_lock)
    merged: list[dict[str, str]] = []
    clean_by_id = {row["structure_id"]: row for row in clean_rows}
    for role_row in role_rows:
        merged.append({**clean_by_id[role_row["structure_id"]], **role_row})
    return merged, eligibility, {"role_sha256": sha256_file(role_path), "clean_sha256": sha256_file(clean_path)}


def fingerprints(rows: Sequence[Mapping[str, str]], n_bits: int) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits, includeChirality=False)
    output = np.empty((len(rows), n_bits), dtype=np.uint8)
    for index, row in enumerate(rows):
        molecule = Chem.MolFromSmiles(row["standardized_smiles"])
        if molecule is None:
            raise ValueError(f"locked standardized SMILES failed to parse: {row['structure_id']}")
        output[index] = generator.GetFingerprintAsNumPy(molecule)
    return output


def molformer_embeddings(rows: Sequence[Mapping[str, str]], tokenizer: object, model: object, config: Mapping[str, object]) -> np.ndarray:
    import torch

    batches: list[np.ndarray] = []
    size = int(config["inference_batch_size"])
    for start in range(0, len(rows), size):
        encoded = tokenizer(
            [row["standardized_smiles"] for row in rows[start : start + size]],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if int(encoded["attention_mask"].sum(dim=1).max()) > int(config["max_tokens_including_special_tokens"]):
            raise RuntimeError("overlength structure survived the frozen eligibility filter")
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            pooled = model(**encoded).pooler_output
        batches.append(pooled.detach().to(dtype=torch.float32).cpu().numpy())
    values = np.concatenate(batches, axis=0)
    if values.shape[0] != len(rows) or not np.isfinite(values).all():
        raise RuntimeError("MoLFormer embedding cache is incomplete or non-finite")
    return values


def feature_cache(endpoint: str, rows: Sequence[Mapping[str, str]], tokenizer: object, model: object, lock: Mapping[str, object], output: Path):
    cache = output / "feature_cache" / endpoint
    metadata_path = cache / "manifest.json"
    ids = [row["structure_id"] for row in rows]
    contract = {
        "endpoint": endpoint,
        "row_ids_sha256": stable_sha256(ids),
        "n": len(ids),
        "production_lock_sha256": sha256_file(DEFAULT_PRODUCTION_LOCK),
        "gpu_lock_sha256": sha256_file(DEFAULT_GPU_LOCK),
    }
    ecfp_path, mol_path = cache / "ecfp.npy", cache / "molformer.npy"
    if metadata_path.is_file() and ecfp_path.is_file() and mol_path.is_file():
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(prior.get(key) == value for key, value in contract.items()):
            if prior.get("ecfp_sha256") == sha256_file(ecfp_path) and prior.get("molformer_sha256") == sha256_file(mol_path):
                return np.load(ecfp_path, allow_pickle=False), np.load(mol_path, allow_pickle=False)
    cache.mkdir(parents=True, exist_ok=True)
    ecfp = fingerprints(rows, int(lock["ecfp"]["n_bits"]))
    mol = molformer_embeddings(rows, tokenizer, model, yaml.safe_load(DEFAULT_GPU_LOCK.read_text(encoding="utf-8"))["molformer"])
    atomic_npy(ecfp_path, ecfp)
    atomic_npy(mol_path, mol)
    contract.update({"ecfp_sha256": sha256_file(ecfp_path), "molformer_sha256": sha256_file(mol_path)})
    atomic_json(metadata_path, contract)
    return ecfp, mol


def fit_ecfp_components(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_predict: np.ndarray,
    seed: int,
    lock: Mapping[str, object],
) -> np.ndarray:
    from xgboost import XGBClassifier

    ecfp = lock["ecfp"]
    models = [
        LogisticRegression(**ecfp["logistic"], random_state=seed),
        RandomForestClassifier(**ecfp["random_forest"], random_state=seed),
        XGBClassifier(
            **ecfp["xgboost"], random_state=seed, eval_metric="logloss",
            objective="binary:logistic",
        ),
        MLPClassifier(
            **{**ecfp["mlp"], "hidden_layer_sizes": tuple(ecfp["mlp"]["hidden_layer_sizes"])},
            random_state=seed,
        ),
    ]
    predictions: list[np.ndarray] = []
    for model in models:
        model.fit(x_fit, y_fit)
        predictions.append(model.predict_proba(x_predict)[:, 1])
    # The protocol calibrates each constituent honestly before averaging; keep
    # the four raw streams separate until their six component calibrators exist.
    return np.clip(np.column_stack(predictions), 1.0e-6, 1.0 - 1.0e-6)


def fit_molformer_head(x_fit: np.ndarray, y_fit: np.ndarray, x_predict: np.ndarray, seed: int) -> np.ndarray:
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
    model.fit(x_fit, y_fit)
    return np.clip(model.predict_proba(x_predict)[:, 1], 1.0e-6, 1.0 - 1.0e-6)


def chemprop_predict(
    fit_rows: Sequence[Mapping[str, str]],
    predict_rows: Sequence[Mapping[str, str]],
    group_column: str,
    seed: int,
    work: Path,
    gpu_lock: Mapping[str, object],
) -> np.ndarray:
    prediction_path = work / "predictions.csv"
    contract_path = work / "job_contract.json"
    expected_ids = [row["structure_id"] for row in predict_rows]
    fit_ids = [row["structure_id"] for row in fit_rows]
    if set(fit_ids) & set(expected_ids):
        raise RuntimeError("Chemprop fit/predict row overlap violates honest lineage")
    expected_contract = {
        "status": "complete_dmpnn_job",
        "fit_ids_sha256": stable_sha256(sorted(fit_ids)),
        "predict_ids_sha256": stable_sha256(expected_ids),
        "fit_n": len(fit_ids),
        "predict_n": len(expected_ids),
        "group_column": group_column,
        "training_seed": seed,
        "gpu_lock_sha256": sha256_file(DEFAULT_GPU_LOCK),
    }
    if prediction_path.is_file() and contract_path.is_file():
        prior = json.loads(contract_path.read_text(encoding="utf-8"))
        if all(prior.get(key) == value for key, value in expected_contract.items()):
            if prior.get("predictions_sha256") == sha256_file(prediction_path):
                table = read_table(prediction_path)
                values = [float(row[[key for key in row if key not in {"smiles", "structure_id"}][0]]) for row in table]
                if len(values) == len(expected_ids) and np.isfinite(values).all():
                    return np.asarray(values)
    if work.exists():
        archived = work.with_name(work.name + f"_partial_{time.time_ns()}")
        shutil.move(str(work), str(archived))
    work.mkdir(parents=True)
    assignment = label_blind_group_folds(fit_rows, group_column, 10, seed * 1000 + 17)
    training = []
    for row in fit_rows:
        training.append({
            "smiles": row["standardized_smiles"], "target": row["target"],
            "split": "val" if assignment[row[group_column]] == 0 else "train",
            "structure_id": row["structure_id"],
        })
    train_path, predict_path, model_dir = work / "train.csv", work / "predict.csv", work / "model"
    write_csv(train_path, training, ["smiles", "target", "split", "structure_id"])
    write_csv(predict_path, ({"smiles": row["standardized_smiles"], "structure_id": row["structure_id"]} for row in predict_rows), ["smiles", "structure_id"])
    cp = gpu_lock["chemprop"]
    train_command = [
        "chemprop", "train", "--data-path", str(train_path), "--smiles-columns", "smiles",
        "--target-columns", "target", "--splits-column", "split", "--task-type", str(cp["task_type"]),
        "--loss-function", str(cp["loss_function"]), "--message-hidden-dim", str(cp["message_hidden_dim"]),
        "--depth", str(cp["depth"]), "--dropout", str(cp["dropout"]), "--aggregation", str(cp["aggregation"]),
        "--ffn-hidden-dim", str(cp["ffn_hidden_dim"]), "--ffn-num-layers", str(cp["ffn_num_layers"]),
        "--batch-size", str(cp["batch_size"]), "--epochs", str(cp["epochs"]), "--patience", str(cp["patience"]),
        "--init-lr", str(cp["init_lr"]), "--max-lr", str(cp["max_lr"]), "--final-lr", str(cp["final_lr"]),
        "--pytorch-seed", str(seed), "--accelerator", "gpu", "--devices", "1", "--output-dir", str(model_dir),
    ]
    predict_command = [
        "chemprop", "predict", "--test-path", str(predict_path), "--smiles-columns", "smiles",
        "--model-paths", str(model_dir), "--preds-path", str(prediction_path),
        "--accelerator", "gpu", "--devices", "1",
    ]
    subprocess.run(train_command, check=True)
    subprocess.run(predict_command, check=True)
    table = read_table(prediction_path)
    if [row.get("structure_id") for row in table] != expected_ids:
        raise RuntimeError("Chemprop prediction identity/order mismatch")
    fields = [key for key in table[0] if key not in {"smiles", "structure_id"}]
    if len(fields) != 1:
        raise RuntimeError(f"unexpected Chemprop prediction fields: {fields}")
    values = np.asarray([float(row[fields[0]]) for row in table])
    if len(values) != len(expected_ids) or not np.isfinite(values).all():
        raise RuntimeError("Chemprop predictions are incomplete or non-finite")
    expected_contract["predictions_sha256"] = sha256_file(prediction_path)
    atomic_json(contract_path, expected_contract)
    return values


def raw_components(
    fit_indices: np.ndarray,
    predict_indices: np.ndarray,
    rows: Sequence[Mapping[str, str]],
    ecfp: np.ndarray,
    mol: np.ndarray,
    y: np.ndarray,
    group_column: str,
    seed: int,
    work: Path,
    lock: Mapping[str, object],
    gpu_lock: Mapping[str, object],
) -> np.ndarray:
    ecfp_components = fit_ecfp_components(
        ecfp[fit_indices], y[fit_indices], ecfp[predict_indices], seed, lock
    )
    return np.column_stack(
        [
            ecfp_components,
            chemprop_predict(
                [rows[i] for i in fit_indices],
                [rows[i] for i in predict_indices],
                group_column,
                seed,
                work,
                gpu_lock,
            ),
            fit_molformer_head(
                mol[fit_indices], y[fit_indices], mol[predict_indices], seed
            ),
        ]
    )


def calibrated_components_to_blocks(values: np.ndarray) -> np.ndarray:
    components = np.asarray(values, dtype=float)
    if components.ndim != 2 or components.shape[1] != 6:
        raise ValueError("expected four ECFP plus D-MPNN and MoLFormer components")
    return np.column_stack(
        [np.mean(components[:, :4], axis=1), components[:, 4], components[:, 5]]
    )


def fit_chain(
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    rows: Sequence[Mapping[str, str]],
    ecfp: np.ndarray,
    mol: np.ndarray,
    y: np.ndarray,
    group_column: str,
    fold_seed: int,
    training_seed: int,
    work: Path,
    lock: Mapping[str, object],
    gpu_lock: Mapping[str, object],
) -> dict[str, np.ndarray]:
    train_rows = [rows[i] for i in train_indices]
    inner = label_blind_group_folds(train_rows, group_column, 2, fold_seed)
    raw_oof = np.full((len(train_indices), 6), np.nan)
    for fold in range(2):
        local_fit = np.asarray([j for j, row in enumerate(train_rows) if inner[row[group_column]] != fold])
        local_valid = np.asarray([j for j, row in enumerate(train_rows) if inner[row[group_column]] == fold])
        raw_oof[local_valid] = raw_components(
            train_indices[local_fit], train_indices[local_valid], rows, ecfp, mol, y,
            group_column, training_seed, work / f"inner_{fold}_dmpnn", lock, gpu_lock,
        )
    if not np.isfinite(raw_oof).all():
        raise RuntimeError("inner OOF block predictions are incomplete")
    calibrators = [
        fit_platt(raw_oof[:, component], y[train_indices], training_seed)
        for component in range(6)
    ]
    calibrated_components_oof = np.column_stack(
        [apply_platt(calibrators[c], raw_oof[:, c]) for c in range(6)]
    )
    calibrated_oof = calibrated_components_to_blocks(calibrated_components_oof)
    stacker = fit_stacker(calibrated_oof, y[train_indices], training_seed)
    stack_oof = stack_probability(stacker, calibrated_oof)
    losses = (stack_oof - y[train_indices]) ** 2
    groups = [str(rows[i][group_column]) for i in train_indices]
    distance, local = tanimoto_topk_local_loss(
        ecfp[train_indices], ecfp[train_indices], losses, int(lock["local_loss_neighbors"]),
        query_groups=groups, reference_groups=groups,
    )
    features_oof = reliability_features(calibrated_oof, stack_oof, distance, local)
    bri_model = fit_bri(features_oof, losses, y[train_indices], training_seed)
    unrestricted = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=training_seed)
    unrestricted.fit(np.column_stack([logits(calibrated_oof), features_oof]), y[train_indices])

    raw = raw_components(
        train_indices, predict_indices, rows, ecfp, mol, y, group_column, training_seed,
        work / "final_dmpnn", lock, gpu_lock,
    )
    calibrated_components = np.column_stack(
        [apply_platt(calibrators[c], raw[:, c]) for c in range(6)]
    )
    calibrated = calibrated_components_to_blocks(calibrated_components)
    stack = stack_probability(stacker, calibrated)
    q_distance, q_local = tanimoto_topk_local_loss(
        ecfp[predict_indices], ecfp[train_indices], losses, int(lock["local_loss_neighbors"])
    )
    features = reliability_features(calibrated, stack, q_distance, q_local)
    return {
        "blocks": calibrated,
        "stack": stack,
        "unrestricted": np.clip(unrestricted.predict_proba(np.column_stack([logits(calibrated), features]))[:, 1], 1e-6, 1-1e-6),
        "bri": bri_predict(bri_model, features),
        "features": features,
    }


def cell_id(endpoint: str, track: str, seed: int) -> str:
    return f"{endpoint}__{track}__seed{seed}"


def valid_cell_artifact(directory: Path, expected_n: int, lock_sha: str) -> bool:
    manifest_path = directory / "raw_manifest.json"
    table_path = directory / "raw_predictions.csv"
    lineage_path = directory / "lineage_manifest.json"
    if not manifest_path.is_file() or not table_path.is_file() or not lineage_path.is_file():
        return False
    row = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        row.get("status") == "complete_raw_predictions"
        and row.get("n") == expected_n
        and row.get("production_lock_sha256") == lock_sha
        and row.get("raw_predictions_sha256") == sha256_file(table_path)
        and row.get("lineage_manifest_sha256") == sha256_file(lineage_path)
        and len(read_table(table_path)) == expected_n
    )


def run_cell_raw(
    endpoint: str,
    track: str,
    seed: int,
    rows: Sequence[Mapping[str, str]],
    ecfp: np.ndarray,
    mol: np.ndarray,
    lock: Mapping[str, object],
    gpu_lock: Mapping[str, object],
    output: Path,
) -> None:
    directory = output / "cells" / cell_id(endpoint, track, seed)
    lock_sha = sha256_file(DEFAULT_PRODUCTION_LOCK)
    if valid_cell_artifact(directory, len(rows), lock_sha):
        print(f"RESUME raw complete: {directory.name}", flush=True)
        return
    group_column = TRACK_GROUP_COLUMN[track]
    allocation = allocate_groups(
        rows, group_column, lock["outer_roles"], seed,
        use_labels_for_assignment=(track == "random_grouped"),
    )
    roles = np.asarray([allocation.assignment[row[group_column]] for row in rows])
    y = np.asarray([int(row["target"]) for row in rows], dtype=np.int8)
    dev_indices = np.flatnonzero(roles == "dev")
    nondev_indices = np.flatnonzero(roles != "dev")
    dev_rows = [rows[i] for i in dev_indices]
    outer = label_blind_group_folds(dev_rows, group_column, int(lock["development_meta_folds"]), seed)
    dev_fold = {int(index): outer[rows[index][group_column]] for index in dev_indices}
    containers = {
        key: np.full((len(rows), 3) if key == "blocks" else (len(rows), 4) if key == "features" else len(rows), np.nan)
        for key in ("blocks", "stack", "unrestricted", "bri", "features")
    }
    nondev_chains: list[dict[str, np.ndarray]] = []
    lineage_folds: list[dict[str, object]] = []
    for fold in range(int(lock["development_meta_folds"])):
        train = np.asarray([i for i in dev_indices if dev_fold[int(i)] != fold])
        valid = np.asarray([i for i in dev_indices if dev_fold[int(i)] == fold])
        predict = np.concatenate([valid, nondev_indices])
        train_ids = [rows[i]["structure_id"] for i in train]
        valid_ids = [rows[i]["structure_id"] for i in valid]
        nondev_ids = [rows[i]["structure_id"] for i in nondev_indices]
        if set(train_ids) & (set(valid_ids) | set(nondev_ids)):
            raise RuntimeError("outer-chain fit/prediction lineage overlap")
        lineage_folds.append(
            {
                "outer_fold": fold,
                "split_seed": seed,
                "inner_fold_seed": seed * 100 + fold,
                "training_seed": int(lock["training_seed"]),
                "fit_ids_sha256": stable_sha256(sorted(train_ids)),
                "heldout_dev_ids_sha256": stable_sha256(sorted(valid_ids)),
                "nondev_prediction_ids_sha256": stable_sha256(sorted(nondev_ids)),
                "fit_n": len(train_ids),
                "heldout_dev_n": len(valid_ids),
                "nondev_n": len(nondev_ids),
                "fit_outer_roles": ["dev"],
            }
        )
        chain = fit_chain(
            train, predict, rows, ecfp, mol, y, group_column,
            seed * 100 + fold,
            int(lock["training_seed"]),
            directory / "jobs" / f"outer_{fold}", lock, gpu_lock,
        )
        for key, values in chain.items():
            containers[key][valid] = values[: len(valid)]
        nondev_chains.append({key: values[len(valid) :] for key, values in chain.items()})
    for key in containers:
        containers[key][nondev_indices] = np.mean([chain[key] for chain in nondev_chains], axis=0)
        if not np.isfinite(containers[key]).all():
            raise RuntimeError(f"{directory.name} contains incomplete {key} predictions")
    predicted = (containers["stack"] >= 0.5).astype(np.int8)
    risk = class_midrank_percentiles(
        containers["bri"][dev_indices], predicted[dev_indices], containers["bri"], predicted
    )
    table = []
    for index, row in enumerate(rows):
        features = containers["features"][index]
        table.append({
            "structure_id": row["structure_id"], "role": roles[index],
            # Keep D_test labels out of every pre-selection artifact. They are
            # joined by structure_id only inside the final evaluation stage.
            "target": "" if roles[index] == "test" else int(y[index]),
            "meta_fold": dev_fold.get(index, ""), "ecfp_p": containers["blocks"][index, 0],
            "dmpnn_p": containers["blocks"][index, 1], "molformer_p": containers["blocks"][index, 2],
            "heterogeneous_p": float(np.mean(containers["blocks"][index])), "stack_p": containers["stack"][index],
            "unrestricted_p": containers["unrestricted"][index], "bri": containers["bri"][index],
            "risk_percentile": risk[index], "absolute_margin": features[0], "disagreement": features[1],
            "ecfp_distance": features[2], "local_oof_brier_loss": features[3],
        })
    table_path = directory / "raw_predictions.csv"
    write_csv(table_path, table, RAW_FIELDS)
    lineage_path = directory / "lineage_manifest.json"
    atomic_json(
        lineage_path,
        {
            "status": "pass_transitive_outer_lineage",
            "endpoint": endpoint,
            "track": track,
            "split_seed": seed,
            "training_seed": int(lock["training_seed"]),
            "folds": lineage_folds,
            "test_targets_written_to_raw_artifact": False,
        },
    )
    role_counts = Counter(roles)
    atomic_json(directory / "raw_manifest.json", {
        "status": "complete_raw_predictions", "endpoint": endpoint, "track": track, "seed": seed,
        "n": len(rows), "role_counts": dict(role_counts), "production_lock_sha256": lock_sha,
        "raw_predictions_sha256": sha256_file(table_path), "trainer_label_roles": ["dev"],
        "lineage_manifest_sha256": sha256_file(lineage_path),
        "policy_used_for_gate_only": True, "conformal_used_for_quantiles_only": True,
        "test_metrics_computed": False,
    })


def leave_fold_dev_metrics(rows: list[dict[str, str]], t_max: float, alpha: float) -> dict[str, float]:
    dev = [row for row in rows if row["role"] == "dev"]
    y = np.asarray([int(row["target"]) for row in dev])
    folds = np.asarray([int(row["meta_fold"]) for row in dev])
    stack = np.asarray([float(row["stack_p"]) for row in dev])
    risk = np.asarray([float(row["risk_percentile"]) for row in dev])
    racer = attenuate(stack, risk, t_max)
    out: dict[str, list[str]] = {"baseline": [""] * len(dev), "candidate": [""] * len(dev)}
    for fold in sorted(set(folds)):
        fit, query = folds != fold, folds == fold
        for name, probabilities in (("baseline", stack), ("candidate", racer)):
            thresholds = conformal_thresholds(probabilities[fit], y[fit], alpha, class_conditional=True)
            states = state_labels(prediction_sets(probabilities[query], thresholds))
            for index, value in zip(np.flatnonzero(query), states):
                out[name][index] = value
    baseline = metric_record(y, out["baseline"])
    candidate = metric_record(y, out["candidate"])
    return {
        "baseline_macro_csy": float(baseline["macro_csy"]),
        "candidate_macro_csy": float(candidate["macro_csy"]),
        "baseline_class_0_coverage": float(baseline["class_0_coverage"]),
        "baseline_class_1_coverage": float(baseline["class_1_coverage"]),
        "candidate_class_0_coverage": float(candidate["class_0_coverage"]),
        "candidate_class_1_coverage": float(candidate["class_1_coverage"]),
    }


def select_global_tmax(lock: Mapping[str, object], output: Path, cell_directories: Sequence[Path]) -> dict[str, object]:
    candidates = [float(value) for value in lock["attenuation"]["t_max_candidates"]]
    margin = float(lock["attenuation"]["coverage_shortfall_margin"])
    evaluations = []
    for candidate in candidates:
        per_cell = [leave_fold_dev_metrics(read_table(path / "raw_predictions.csv"), candidate, float(lock["conformal"]["alpha"])) for path in cell_directories]
        feasible = all(
            row[f"candidate_class_{label}_coverage"] >= row[f"baseline_class_{label}_coverage"] - margin
            for row in per_cell for label in (0, 1)
        )
        evaluations.append({
            "t_max": candidate, "feasible": feasible,
            "mean_candidate_macro_csy": float(np.mean([row["candidate_macro_csy"] for row in per_cell])),
            "mean_baseline_macro_csy": float(np.mean([row["baseline_macro_csy"] for row in per_cell])),
            "cell_count": len(per_cell),
        })
    feasible = [row for row in evaluations if row["feasible"]]
    if not feasible:
        raise RuntimeError("no global T_max candidate satisfies the frozen development coverage constraint")
    chosen = min(feasible, key=lambda row: (-row["mean_candidate_macro_csy"], row["t_max"]))
    record = {"status": "selected_development_only", "selected_t_max": chosen["t_max"], "evaluations": evaluations}
    atomic_json(output / "global_tmax_selection.json", record)
    return record


def evaluate_cell(
    directory: Path,
    endpoint: str,
    track: str,
    seed: int,
    critical_class: int,
    t_max: float,
    lock: Mapping[str, object],
    test_targets_by_id: Mapping[str, int],
) -> None:
    result_path = directory / "final_manifest.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete_final_evaluation" and prior.get("selected_t_max") == t_max:
            print(f"RESUME final complete: {directory.name}", flush=True)
            return
    rows = read_table(directory / "raw_predictions.csv")
    by_role = {role: [row for row in rows if row["role"] == role] for role in ("dev", "policy", "conformal", "test")}
    for row in by_role["test"]:
        if row["target"]:
            raise RuntimeError("D_test target leaked into the pre-selection raw artifact")
        if row["structure_id"] not in test_targets_by_id:
            raise RuntimeError(f"missing sealed test target: {row['structure_id']}")
        row["target"] = str(int(test_targets_by_id[row["structure_id"]]))
    def arrays(role: str, field: str, dtype=float):
        return np.asarray([dtype(row[field]) for row in by_role[role]])
    policy_rows = [{
        "structure_id": row["structure_id"], "true_class": int(row["target"]),
        "predicted_class": int(float(row["stack_p"]) >= 0.5), "risk_percentile": float(row["risk_percentile"]),
    } for row in by_role["policy"]]
    policy_status, chosen, grid = select_policy(
        policy_rows,
        PolicyConstraints(
            critical_class=critical_class,
            retention_floor=float(lock["gate"]["class_retention_lower_bound_minimum"]),
            critical_error_ceiling=float(lock["gate"]["critical_selected_base_error_upper_bound_maximum"]),
            familywise_alpha=float(lock["gate"]["familywise_alpha"]),
            minimum_class_n=int(lock["gate"]["minimum_policy_count_per_true_class"]),
            minimum_selected_critical_n=int(lock["gate"]["minimum_selected_critical_count"]),
        ),
        tuple(float(value) for value in lock["gate"]["threshold_grid"]),
    )
    thresholds = {0: float(chosen["threshold_0"]), 1: float(chosen["threshold_1"])} if chosen else {0: 0.0, 1: 0.0}
    write_csv(directory / "policy_grid.csv", grid, list(grid[0]))
    atomic_json(directory / "policy_selection.json", {"status": policy_status, "chosen": chosen, "critical_class": critical_class})

    alpha = float(lock["conformal"]["alpha"])
    conf_y, test_y = arrays("conformal", "target", int), arrays("test", "target", int)
    conf_stack, test_stack = arrays("conformal", "stack_p"), arrays("test", "stack_p")
    conf_risk, test_risk = arrays("conformal", "risk_percentile"), arrays("test", "risk_percentile")
    conf_pred, test_pred = (conf_stack >= 0.5).astype(int), (test_stack >= 0.5).astype(int)
    conf_selected = np.asarray([risk <= thresholds[int(label)] for risk, label in zip(conf_risk, conf_pred)])
    test_selected = np.asarray([risk <= thresholds[int(label)] for risk, label in zip(test_risk, test_pred)])
    conf_racer, test_racer = attenuate(conf_stack, conf_risk, t_max), attenuate(test_stack, test_risk, t_max)
    methods: dict[str, tuple[np.ndarray, np.ndarray, bool, np.ndarray | None]] = {
        "ECFP_marginal_LAC": (arrays("conformal", "ecfp_p"), arrays("test", "ecfp_p"), False, None),
        "ECFP_Mondrian_LAC": (arrays("conformal", "ecfp_p"), arrays("test", "ecfp_p"), True, None),
        "heterogeneous_Mondrian": (arrays("conformal", "heterogeneous_p"), arrays("test", "heterogeneous_p"), True, None),
        "stacking_Mondrian": (conf_stack, test_stack, True, None),
        "unrestricted_reliability_stacking_Mondrian": (arrays("conformal", "unrestricted_p"), arrays("test", "unrestricted_p"), True, None),
        "RACER_score_no_gate": (conf_racer, test_racer, True, None),
    }
    prediction_rows, metric_rows = [], []
    for method, (conf_p, test_p, mondrian, _) in methods.items():
        q = conformal_thresholds(conf_p, conf_y, alpha, class_conditional=mondrian)
        states = state_labels(prediction_sets(test_p, q))
        metric_rows.append({"endpoint": endpoint, "track": track, "seed": seed, "method": method, "policy_status": "not_applicable", **metric_record(test_y, states)})
        prediction_rows.extend({"structure_id": row["structure_id"], "target": int(row["target"]), "method": method, "state": state} for row, state in zip(by_role["test"], states))
    for method, conf_p, test_p in (
        ("stacking_RACER_gate_selected_Mondrian", conf_stack, test_stack),
        ("full_RACER_C", conf_racer, test_racer),
    ):
        q = conformal_thresholds(conf_p, conf_y, alpha, selected=conf_selected, class_conditional=True)
        states = state_labels(prediction_sets(test_p, q), selected=test_selected)
        metric_rows.append({"endpoint": endpoint, "track": track, "seed": seed, "method": method, "policy_status": policy_status, **metric_record(test_y, states)})
        prediction_rows.extend({"structure_id": row["structure_id"], "target": int(row["target"]), "method": method, "state": state} for row, state in zip(by_role["test"], states))
    dev_features = np.asarray([[float(row[key]) for key in ("absolute_margin", "disagreement", "ecfp_distance", "local_oof_brier_loss")] for row in by_role["dev"]])
    conf_features = np.asarray([[float(row[key]) for key in ("absolute_margin", "disagreement", "ecfp_distance", "local_oof_brier_loss")] for row in by_role["conformal"]])
    test_features = np.asarray([[float(row[key]) for key in ("absolute_margin", "disagreement", "ecfp_distance", "local_oof_brier_loss")] for row in by_role["test"]])
    rcp = fit_rcp_transform(dev_features, arrays("dev", "stack_p"), arrays("dev", "target", int), alpha, seed)
    conf_multiplier, test_multiplier = rcp.multipliers(conf_features), rcp.multipliers(test_features)
    conf_true_multiplier = conf_multiplier[np.arange(len(conf_y)), conf_y]
    q = conformal_thresholds(conf_stack, conf_y, alpha, class_conditional=True, score_multiplier=conf_true_multiplier)
    states = state_labels(prediction_sets(test_stack, q, score_multiplier=test_multiplier))
    method = "RCP_class_conditional"
    metric_rows.append({"endpoint": endpoint, "track": track, "seed": seed, "method": method, "policy_status": "not_applicable", **metric_record(test_y, states)})
    prediction_rows.extend({"structure_id": row["structure_id"], "target": int(row["target"]), "method": method, "state": state} for row, state in zip(by_role["test"], states))
    expected_methods = list(lock["core_methods"])
    if {row["method"] for row in metric_rows} != set(expected_methods):
        raise RuntimeError("core method set is incomplete")
    write_csv(directory / "test_predictions.csv", prediction_rows, ["structure_id", "target", "method", "state"])
    metric_fields = list(metric_rows[0])
    write_csv(directory / "metrics.csv", metric_rows, metric_fields)
    atomic_json(result_path, {
        "status": "complete_final_evaluation", "endpoint": endpoint, "track": track, "seed": seed,
        "selected_t_max": t_max, "policy_status": policy_status, "method_count": len(metric_rows),
        "test_n": len(test_y), "metrics_sha256": sha256_file(directory / "metrics.csv"),
        "test_predictions_sha256": sha256_file(directory / "test_predictions.csv"),
    })


def aggregate(output: Path, lock: Mapping[str, object], directories: Sequence[Path], failures: list[dict[str, object]]) -> dict[str, object]:
    if failures:
        record = {"status": "incomplete_retained_failures", "failure_count": len(failures), "failures": failures}
        atomic_json(output / "run_summary.json", record)
        return record
    metrics: list[dict[str, str]] = []
    for directory in directories:
        final = json.loads((directory / "final_manifest.json").read_text(encoding="utf-8"))
        if final.get("status") != "complete_final_evaluation":
            raise RuntimeError(f"missing final cell: {directory.name}")
        metrics.extend(read_table(directory / "metrics.csv"))
    expected_cells = len(lock["primary_endpoints"]) * len(lock["tracks"]) * len(lock["main_split_seeds"])
    expected_rows = expected_cells * len(lock["core_methods"])
    if len(directories) != expected_cells or len(metrics) != expected_rows:
        raise RuntimeError(f"aggregate completeness failure: cells={len(directories)}/{expected_cells}, metrics={len(metrics)}/{expected_rows}")
    fields = list(metrics[0])
    write_csv(output / "all_cell_metrics.csv", metrics, fields)
    record = {
        "status": "complete_confirmatory_primary_study", "primary_cell_count": expected_cells,
        "method_cell_count": expected_rows, "failed_cell_count": 0,
        "all_cell_metrics_sha256": sha256_file(output / "all_cell_metrics.csv"),
        "production_lock_sha256": sha256_file(DEFAULT_PRODUCTION_LOCK),
        "test_results_generated": True,
    }
    atomic_json(output / "run_summary.json", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-lock", type=Path, default=DEFAULT_PRODUCTION_LOCK)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--freeze-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-unfrozen-for-tests", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = yaml.safe_load(args.production_lock.read_text(encoding="utf-8"))
    gpu_lock = yaml.safe_load(args.gpu_lock.read_text(encoding="utf-8"))
    tag = verify_protocol_tag(args.allow_unfrozen_for_tests)
    review = verify_freeze_review(args.freeze_review, lock)
    endpoints, tracks, seeds = list(lock["primary_endpoints"]), list(lock["tracks"]), [int(value) for value in lock["main_split_seeds"]]
    critical = endpoint_critical_classes(ENDPOINT_MANIFEST, endpoints)
    expected_cells = len(endpoints) * len(tracks) * len(seeds)
    if expected_cells != int(lock["formal_review_required_cells"]):
        raise RuntimeError("production lock no longer defines exactly 60 primary cells")
    print(json.dumps({"status": "production_contract_pass", "tag": tag, "review_sha256": sha256_file(args.freeze_review), "cells": expected_cells}, sort_keys=True), flush=True)
    if args.validate_only:
        return 0

    if not args.resume and args.output_dir.exists():
        raise FileExistsError("confirmatory output exists; --resume is required and never implies overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.output_dir / "run_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {
        "status": "running", "protocol": tag, "formal_review_sha256": sha256_file(args.freeze_review),
        "production_lock_sha256": sha256_file(args.production_lock), "cells": {}, "failures": [],
    }
    if registry["production_lock_sha256"] != sha256_file(args.production_lock):
        raise RuntimeError("existing registry belongs to a different production lock")

    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        gpu_lock["molformer"]["model_id"], revision=gpu_lock["molformer"]["revision"],
        trust_remote_code=bool(gpu_lock["molformer"]["trust_remote_code"]),
    )
    endpoint_data = {}
    reproduced_cells: list[dict[str, object]] = []
    for endpoint in endpoints:
        rows, eligibility, hashes = load_endpoint(endpoint, tokenizer, gpu_lock)
        endpoint_data[endpoint] = (rows, None, None)
        registry.setdefault("endpoint_eligibility", {})[endpoint] = {**eligibility, **hashes}
        atomic_json(registry_path, registry)
        for track in tracks:
            for seed in seeds:
                summary, _ = audit_one(
                    rows,
                    track,
                    lock["outer_roles"],
                    seed,
                    minimum_retention=float(lock["gate"]["class_retention_lower_bound_minimum"]),
                    alpha=float(lock["conformal"]["alpha"]),
                )
                reproduced_cells.append(dict(summary))
    reproduced_hash = stable_sha256(reproduced_cells)
    if reproduced_hash != review.get("track_seed_cells_sha256"):
        raise RuntimeError(
            "current 60-cell role/count audit differs from the approved formal review: "
            f"expected={review.get('track_seed_cells_sha256')} observed={reproduced_hash}"
        )

    model = AutoModel.from_pretrained(
        gpu_lock["molformer"]["model_id"], revision=gpu_lock["molformer"]["revision"],
        trust_remote_code=bool(gpu_lock["molformer"]["trust_remote_code"]), deterministic_eval=True,
    ).to("cuda")
    model.eval()
    for endpoint in endpoints:
        rows = endpoint_data[endpoint][0]
        ecfp, mol = feature_cache(endpoint, rows, tokenizer, model, lock, args.output_dir)
        endpoint_data[endpoint] = (rows, ecfp, mol)
    del model
    import torch
    torch.cuda.empty_cache()

    directories: list[Path] = []
    failures: list[dict[str, object]] = []
    attempts = int(lock["failure_policy"]["attempts_per_cell"])
    for endpoint in endpoints:
        rows, ecfp, mol = endpoint_data[endpoint]
        for track in tracks:
            for seed in seeds:
                identifier = cell_id(endpoint, track, seed)
                directory = args.output_dir / "cells" / identifier
                directories.append(directory)
                for attempt in range(1, attempts + 1):
                    try:
                        registry["cells"][identifier] = {"status": "running_raw", "attempt": attempt}
                        atomic_json(registry_path, registry)
                        run_cell_raw(endpoint, track, seed, rows, ecfp, mol, lock, gpu_lock, args.output_dir)
                        registry["cells"][identifier] = {"status": "raw_complete", "attempt": attempt}
                        atomic_json(registry_path, registry)
                        break
                    except Exception as error:
                        failure = {"cell": identifier, "stage": "raw", "attempt": attempt, "error": repr(error), "traceback": traceback.format_exc()}
                        atomic_json(directory / f"failure_raw_attempt_{attempt}.json", failure)
                        if attempt == attempts:
                            failures.append(failure)
                            registry["cells"][identifier] = {"status": "failed_raw_retained", "attempt": attempt, "error": repr(error)}
                            atomic_json(registry_path, registry)
    if failures:
        registry["status"] = "incomplete_raw_failures_retained"
        registry["failures"] = failures
        atomic_json(registry_path, registry)
        aggregate(args.output_dir, lock, directories, failures)
        return 2

    tmax = select_global_tmax(lock, args.output_dir, directories)
    for endpoint in endpoints:
        for track in tracks:
            for seed in seeds:
                identifier = cell_id(endpoint, track, seed)
                directory = args.output_dir / "cells" / identifier
                try:
                    registry["cells"][identifier] = {"status": "running_final"}
                    atomic_json(registry_path, registry)
                    rows = endpoint_data[endpoint][0]
                    sealed_targets = {
                        row["structure_id"]: int(row["target"]) for row in rows
                    }
                    evaluate_cell(
                        directory, endpoint, track, seed, critical[endpoint],
                        float(tmax["selected_t_max"]), lock, sealed_targets,
                    )
                    registry["cells"][identifier] = {"status": "complete"}
                    atomic_json(registry_path, registry)
                except Exception as error:
                    failure = {"cell": identifier, "stage": "final", "attempt": 1, "error": repr(error), "traceback": traceback.format_exc()}
                    failures.append(failure)
                    atomic_json(directory / "failure_final.json", failure)
                    registry["cells"][identifier] = {"status": "failed_final_retained", "error": repr(error)}
                    atomic_json(registry_path, registry)
    summary = aggregate(args.output_dir, lock, directories, failures)
    registry["status"] = summary["status"]
    registry["failures"] = failures
    atomic_json(registry_path, registry)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "complete_confirmatory_primary_study" else 3


if __name__ == "__main__":
    raise SystemExit(main())
