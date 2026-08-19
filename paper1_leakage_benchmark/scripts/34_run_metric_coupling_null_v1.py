from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.dataset_registry import DATASETS
from shared_utils.scaffold_identity import prepare_scaffold_frame
from shared_utils.split_candidate_pool_v3 import _hash_scaffolds, _random_prefix_candidate, _seeded_rank
from shared_utils.split_search_v2 import build_scaffold_groups

PAPER = ROOT / "paper1_leakage_benchmark"
DATA_DIR = PAPER / "data" / "processed_v2"
PROTOCOL = PAPER / "SARQSAR_METRIC_COUPLING_PROTOCOL_V1.md"
CONFIG_PATH = PAPER / "SARQSAR_METRIC_COUPLING_CONFIG_V1.json"
OUT = PAPER / "results" / "sarqsar_metric_coupling_v1"
CACHE = OUT / "candidate_cache"
BY_SEED = OUT / "by_seed"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
MANIFEST = OUT / "RUN_MANIFEST.json"

EFFECT_COLUMNS = {
    "classification": [
        "effect_roc_auc",
        "effect_average_precision",
        "effect_brier",
        "effect_log_loss",
        "effect_prevalence_gap",
        "effect_brier_variance",
        "effect_brier_gap_sq",
    ],
    "regression": [
        "effect_rmse",
        "effect_mse",
        "effect_mae",
        "effect_r2",
        "effect_test_variance",
        "effect_squared_mean_gap",
    ],
}

CHECKLIST = [
    ("Molecular identity", "Canonicalization, stereochemistry, duplicate aggregation, and conflicting-label policy", "The benchmark dataset changes silently across implementations."),
    ("Disconnected components", "Full-record, salt-removal, or dominant-fragment rule", "Fingerprints, scaffolds, duplicate mappings, and labels can change."),
    ("Scaffold semantics", "Scaffold algorithm, chirality, and acyclic-molecule handling", "The meaning of an unseen scaffold is ambiguous."),
    ("Endpoint use", "Whether endpoint values enter candidate generation, filtering, or final selection", "Response-aware optimization can be mistaken for target-blind validation."),
    ("Test cardinality", "Requested and realized test size for every compared split", "Performance changes can be confounded by sample size."),
    ("Candidate search", "Generation algorithm, requested draw budget, unique candidates, and stopping rule", "Search effort becomes a hidden benchmark hyperparameter."),
    ("Response-only control", "Training-mean or training-prevalence predictor matched to the evaluation metric", "Metric coupling can be misattributed to molecular learning."),
    ("Collateral diagnostics", "Variance, tails, prevalence, scaffold concentration, and acyclic fraction", "A multivariate test-population change is described as a one-variable intervention."),
    ("Partition identity", "Molecule-level manifests, hashes, and number of unique partitions", "Duplicate splits can create pseudo-replication."),
    ("Inferential unit", "Partition pair, dataset, model seed, and multiplicity family", "Model seeds or repeated fits can inflate inferential sample size."),
    ("Protocol sensitivity", "Alternative scaffold and molecular-record policies declared before outcomes", "A convenient implementation choice can determine the conclusion."),
    ("Provenance", "Code version, software environment, data hashes, and immutable release", "The reported benchmark cannot be reproduced or audited."),
]


@dataclass(frozen=True)
class Candidate:
    group_indices: tuple[int, ...]
    candidate_hash: str
    n_test: int
    size_deviation: float
    first_draw: int


@dataclass
class CandidateCache:
    matrix: csr_matrix
    hashes: np.ndarray
    n_test: np.ndarray
    size_deviation: np.ndarray
    first_draw: np.ndarray
    metadata: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(master_seed: int, dataset: str, permutation_id: int) -> int:
    payload = f"{master_seed}|{dataset}|{permutation_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()
    except Exception:
        return "unknown"


def load_config(permutations_override: int | None = None) -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if permutations_override is not None:
        if permutations_override < 2:
            raise ValueError("--permutations must be at least 2")
        config = json.loads(json.dumps(config))
        config["n_permutations"] = int(permutations_override)
    return config


def protocol_fingerprint(config: dict) -> dict:
    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "protocol_sha256": sha256_file(PROTOCOL),
        "config_sha256": hashlib.sha256(canonical_config).hexdigest(),
        "protocol_version": config["protocol_version"],
    }


def ensure_dirs() -> None:
    for path in [OUT, CACHE, BY_SEED, TABLES, FIGURES, REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def validate_clean_frame(dataset: str, task_type: str) -> tuple[pd.DataFrame, Path, str]:
    path = DATA_DIR / f"{dataset.lower()}_clean_v2.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen clean input: {path}\n"
            "The SAR/QSAR null simulation consumes the existing clean_v2 artifacts and does not download or rebuild them."
        )
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"canonical_smiles", "target"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{dataset}: missing clean-v2 columns {sorted(missing)}")
    if frame.empty:
        raise AssertionError(f"{dataset}: clean-v2 input is empty")
    frame = frame.reset_index(drop=True)
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(float)
    if not np.isfinite(frame["target"]).all():
        raise AssertionError(f"{dataset}: non-finite target values")
    if frame["canonical_smiles"].astype(str).str.len().eq(0).any():
        raise AssertionError(f"{dataset}: empty canonical SMILES")
    if task_type == "classification":
        labels = set(np.unique(frame["target"]).tolist())
        if not labels.issubset({0.0, 1.0}) or len(labels) != 2:
            raise AssertionError(f"{dataset}: classification target must be binary 0/1, found {sorted(labels)}")
    return frame, path, sha256_file(path)


def generate_permutations(y: np.ndarray, dataset: str, config: dict) -> tuple[np.ndarray, np.ndarray]:
    n_permutations = int(config["n_permutations"])
    seeds = np.array(
        [stable_seed(int(config["master_seed"]), dataset, index) for index in range(n_permutations)],
        dtype=np.uint32,
    )
    permutations = np.empty((len(y), n_permutations), dtype=np.float64)
    for index, seed in enumerate(seeds):
        permutations[:, index] = np.random.default_rng(int(seed)).permutation(y)
    return permutations, seeds


def build_candidate_cache(
    frame: pd.DataFrame,
    *,
    dataset: str,
    mode: str,
    partition_seed: int,
    max_budget: int,
    clean_sha256: str,
    fingerprint: dict,
    force_cache: bool,
) -> CandidateCache:
    stem = f"{dataset}_{mode}_seed{partition_seed}_draws{max_budget}"
    matrix_path = CACHE / f"{stem}.npz"
    meta_path = CACHE / f"{stem}.json"
    candidates_path = CACHE / f"{stem}_candidates.csv"
    expected_meta = {
        "dataset": dataset,
        "scaffold_mode": mode,
        "partition_seed": int(partition_seed),
        "max_requested_draws": int(max_budget),
        "clean_sha256": clean_sha256,
        **fingerprint,
    }

    if not force_cache and matrix_path.exists() and meta_path.exists() and candidates_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected_meta.items()):
            table = pd.read_csv(candidates_path, keep_default_na=False)
            matrix = sparse.load_npz(matrix_path).tocsr()
            if matrix.shape != (len(table), len(frame)):
                raise AssertionError(f"Candidate cache shape mismatch for {stem}")
            return CandidateCache(
                matrix=matrix,
                hashes=table["candidate_hash"].astype(str).to_numpy(),
                n_test=table["n_test"].to_numpy(dtype=int),
                size_deviation=table["size_deviation"].to_numpy(dtype=float),
                first_draw=table["first_draw"].to_numpy(dtype=int),
                metadata=metadata,
            )

    scaffold_frame = prepare_scaffold_frame(frame, acyclic_mode=mode)
    groups = build_scaffold_groups(scaffold_frame, target_col="target")
    test_fraction = float(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["test_fraction"])
    target_n = int(round(len(frame) * test_fraction))
    rng = np.random.default_rng(int(partition_seed))
    unique: dict[str, Candidate] = {}

    for draw in range(1, max_budget + 1):
        selected = _random_prefix_candidate(groups, target_n=target_n, rng=rng)
        scaffold_key = tuple(sorted(groups[index].scaffold for index in selected))
        candidate_hash = _hash_scaffolds(scaffold_key)
        if candidate_hash in unique:
            continue
        n_test = int(sum(groups[index].n for index in selected))
        unique[candidate_hash] = Candidate(
            group_indices=selected,
            candidate_hash=candidate_hash,
            n_test=n_test,
            size_deviation=float(abs(n_test - target_n) / max(target_n, 1)),
            first_draw=draw,
        )

    candidates = list(unique.values())
    if not candidates:
        raise RuntimeError(f"{dataset}/{mode}/seed{partition_seed}: candidate pool is empty")

    nnz = int(sum(candidate.n_test for candidate in candidates))
    indptr = np.zeros(len(candidates) + 1, dtype=np.int64)
    indices = np.empty(nnz, dtype=np.int32)
    offset = 0
    for row, candidate in enumerate(candidates):
        blocks = [groups[index].indices.astype(np.int32, copy=False) for index in candidate.group_indices]
        members = np.concatenate(blocks)
        members.sort()
        next_offset = offset + len(members)
        indices[offset:next_offset] = members
        offset = next_offset
        indptr[row + 1] = offset
    data = np.ones(nnz, dtype=np.float64)
    matrix = csr_matrix((data, indices, indptr), shape=(len(candidates), len(frame)))

    table = pd.DataFrame(
        {
            "candidate_hash": [candidate.candidate_hash for candidate in candidates],
            "n_test": [candidate.n_test for candidate in candidates],
            "size_deviation": [candidate.size_deviation for candidate in candidates],
            "first_draw": [candidate.first_draw for candidate in candidates],
        }
    )
    metadata = {
        **expected_meta,
        "n_molecules": int(len(frame)),
        "target_test_n": target_n,
        "n_scaffold_groups": int(len(groups)),
        "unique_candidates": int(len(candidates)),
        "duplicate_draw_fraction": float(1.0 - len(candidates) / max_budget),
        "matrix_nnz": nnz,
        "candidate_generation": "random_scaffold_prefix_closest_crossing",
        "budget_semantics": "requested_draw_prefix_with_first-seen_unique_candidates",
    }
    sparse.save_npz(matrix_path, matrix, compressed=True)
    table.to_csv(candidates_path, index=False)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return CandidateCache(
        matrix=matrix,
        hashes=table["candidate_hash"].astype(str).to_numpy(),
        n_test=table["n_test"].to_numpy(dtype=int),
        size_deviation=table["size_deviation"].to_numpy(dtype=float),
        first_draw=table["first_draw"].to_numpy(dtype=int),
        metadata=metadata,
    )


def choose_baseline(cache: CandidateCache, budget: int, partition_seed: int) -> tuple[int, np.ndarray]:
    available = np.flatnonzero(cache.first_draw <= int(budget))
    if available.size == 0:
        raise AssertionError(f"No candidate available at requested budget {budget}")
    minimum = float(np.min(cache.size_deviation[available]))
    ties = available[np.isclose(cache.size_deviation[available], minimum, rtol=0.0, atol=1e-15)]
    baseline = min(ties.tolist(), key=lambda index: _seeded_rank(partition_seed, cache.hashes[index]))
    matched = available[cache.n_test[available] == cache.n_test[baseline]]
    if matched.size == 0 or baseline not in set(matched.tolist()):
        raise AssertionError("Exact-size candidate subset lost its baseline")
    return int(baseline), matched.astype(int)


def choose_balanced_indices(
    gap_matrix: np.ndarray,
    matched_global: np.ndarray,
    hashes: np.ndarray,
    partition_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.array(
        sorted(
            range(len(matched_global)),
            key=lambda position: _seeded_rank(partition_seed + 10_000_019, hashes[matched_global[position]]),
        ),
        dtype=int,
    )
    ordered_gaps = gap_matrix[order, :]
    minimum = np.min(ordered_gaps, axis=0)
    is_minimum = np.isclose(ordered_gaps, minimum[None, :], rtol=0.0, atol=1e-15)
    first_position = np.argmax(is_minimum, axis=0)
    selected = matched_global[order[first_position]]
    return selected.astype(int), minimum.astype(float)


def regression_metrics(
    *,
    test_sum: np.ndarray,
    test_sumsq: np.ndarray,
    n_test: int,
    total_sum: float,
    n_total: int,
    mae: np.ndarray,
) -> dict[str, np.ndarray]:
    n_train = n_total - n_test
    test_mean = test_sum / n_test
    train_mean = (total_sum - test_sum) / n_train
    variance = np.maximum(test_sumsq / n_test - test_mean**2, 0.0)
    gap_sq = (test_mean - train_mean) ** 2
    mse = variance + gap_sq
    rmse = np.sqrt(np.maximum(mse, 0.0))
    sse = test_sumsq - 2.0 * train_mean * test_sum + n_test * train_mean**2
    sst = test_sumsq - test_sum**2 / n_test
    r2 = np.full_like(mse, np.nan)
    valid = sst > 1e-15
    r2[valid] = 1.0 - sse[valid] / sst[valid]
    return {
        "variance": variance,
        "gap_sq": gap_sq,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def classification_metrics(
    *,
    positives_test: np.ndarray,
    n_test: int,
    positives_total: float,
    n_total: int,
) -> dict[str, np.ndarray]:
    n_train = n_total - n_test
    q = positives_test / n_test
    p = (positives_total - positives_test) / n_train
    p_clip = np.clip(p, 1e-12, 1.0 - 1e-12)
    brier_variance = q * (1.0 - q)
    gap_sq = (q - p) ** 2
    brier = brier_variance + gap_sq
    log_loss = -(q * np.log(p_clip) + (1.0 - q) * np.log(1.0 - p_clip))
    auc = np.where((q > 0.0) & (q < 1.0), 0.5, np.nan)
    return {
        "test_prevalence": q,
        "train_prevalence": p,
        "prevalence_gap": np.abs(q - p),
        "brier_variance": brier_variance,
        "gap_sq": gap_sq,
        "brier": brier,
        "log_loss": log_loss,
        "roc_auc": auc,
        "average_precision": q.copy(),
    }


def selected_mae(
    permutations: np.ndarray,
    matrix: csr_matrix,
    candidate_indices: np.ndarray,
    train_means: np.ndarray,
) -> np.ndarray:
    result = np.empty(permutations.shape[1], dtype=float)
    for candidate in np.unique(candidate_indices):
        columns = np.flatnonzero(candidate_indices == candidate)
        members = matrix.getrow(int(candidate)).indices
        values = permutations[np.ix_(members, columns)]
        result[columns] = np.mean(np.abs(values - train_means[columns][None, :]), axis=0)
    return result


def expected_seed_rows(n_permutations: int, budgets: list[int]) -> int:
    return int(n_permutations * len(budgets))


def validate_checkpoint(
    path: Path,
    *,
    dataset: str,
    mode: str,
    partition_seed: int,
    clean_sha256: str,
    fingerprint: dict,
    n_permutations: int,
    budgets: list[int],
) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except Exception:
        return False
    return bool(
        len(frame) == expected_seed_rows(n_permutations, budgets)
        and set(frame["budget"].astype(int)) == set(budgets)
        and frame["permutation_id"].nunique() == n_permutations
        and frame["dataset"].eq(dataset).all()
        and frame["scaffold_mode"].eq(mode).all()
        and frame["partition_seed"].astype(int).eq(partition_seed).all()
        and frame["clean_sha256"].eq(clean_sha256).all()
        and frame["protocol_sha256"].eq(fingerprint["protocol_sha256"]).all()
        and frame["config_sha256"].eq(fingerprint["config_sha256"]).all()
    )


def run_seed(
    *,
    frame: pd.DataFrame,
    permutations: np.ndarray,
    permutation_seeds: np.ndarray,
    dataset: str,
    task_type: str,
    mode: str,
    partition_seed: int,
    budgets: list[int],
    clean_sha256: str,
    fingerprint: dict,
    force: bool,
    force_cache: bool,
) -> Path:
    output = BY_SEED / f"{dataset}_{mode}_seed{partition_seed}.csv"
    if not force and validate_checkpoint(
        output,
        dataset=dataset,
        mode=mode,
        partition_seed=partition_seed,
        clean_sha256=clean_sha256,
        fingerprint=fingerprint,
        n_permutations=permutations.shape[1],
        budgets=budgets,
    ):
        print(f"  SKIP validated checkpoint {output.name}", flush=True)
        return output

    cache = build_candidate_cache(
        frame,
        dataset=dataset,
        mode=mode,
        partition_seed=partition_seed,
        max_budget=max(budgets),
        clean_sha256=clean_sha256,
        fingerprint=fingerprint,
        force_cache=force_cache,
    )
    matrix = cache.matrix

    selection_plan: dict[int, tuple[int, np.ndarray]] = {}
    needed_global: set[int] = set()
    for budget in budgets:
        baseline_index, matched = choose_baseline(cache, budget, partition_seed)
        selection_plan[int(budget)] = (baseline_index, matched)
        needed_global.update(int(value) for value in matched)
    needed = np.array(sorted(needed_global), dtype=int)
    global_to_local = np.full(len(cache.hashes), -1, dtype=int)
    global_to_local[needed] = np.arange(len(needed), dtype=int)
    relevant_matrix = matrix[needed, :]
    sums = np.asarray(relevant_matrix @ permutations, dtype=float)
    sumsq = np.asarray(relevant_matrix @ (permutations**2), dtype=float) if task_type == "regression" else None
    original_target = frame["target"].to_numpy(dtype=float)
    total_sum = float(np.sum(original_target))
    n_total = len(frame)
    permutation_ids = np.arange(permutations.shape[1], dtype=int)
    rows: list[pd.DataFrame] = []

    for budget in budgets:
        baseline_index, matched = selection_plan[int(budget)]
        baseline_local = int(global_to_local[baseline_index])
        matched_local = global_to_local[matched]
        if baseline_local < 0 or np.any(matched_local < 0):
            raise AssertionError("Relevant-candidate index map is incomplete")
        n_test = int(cache.n_test[baseline_index])
        n_train = n_total - n_test
        matched_sums = sums[matched_local, :]
        gap_matrix = np.abs(matched_sums / n_test - (total_sum - matched_sums) / n_train)
        balanced_indices, balanced_gap = choose_balanced_indices(gap_matrix, matched, cache.hashes, partition_seed)

        baseline_sum = sums[baseline_local, :]
        baseline_gap = np.abs(baseline_sum / n_test - (total_sum - baseline_sum) / n_train)
        if np.any(balanced_gap > baseline_gap + 1e-12):
            raise AssertionError(f"{dataset}/{mode}/seed{partition_seed}/K{budget}: response-aware gap exceeded baseline")

        balanced_local = global_to_local[balanced_indices]
        if np.any(balanced_local < 0):
            raise AssertionError("Balanced candidate was not included in the relevant-row union")
        balanced_sum = sums[balanced_local, np.arange(permutations.shape[1])]
        result = pd.DataFrame(
            {
                "dataset": dataset,
                "task_type": task_type,
                "scaffold_mode": mode,
                "partition_seed": int(partition_seed),
                "budget": int(budget),
                "permutation_id": permutation_ids,
                "permutation_seed": permutation_seeds.astype(np.uint64),
                "n_total": n_total,
                "n_test": n_test,
                "available_unique_candidates": int(np.sum(cache.first_draw <= budget)),
                "same_size_candidates": int(len(matched)),
                "baseline_candidate_hash": cache.hashes[baseline_index],
                "balanced_candidate_hash": cache.hashes[balanced_indices],
                "same_partition": balanced_indices == baseline_index,
                "baseline_abs_target_gap": baseline_gap,
                "balanced_abs_target_gap": balanced_gap,
                "target_gap_reduction": baseline_gap - balanced_gap,
                "clean_sha256": clean_sha256,
                "protocol_sha256": fingerprint["protocol_sha256"],
                "config_sha256": fingerprint["config_sha256"],
            }
        )

        if task_type == "regression":
            assert sumsq is not None
            baseline_sumsq = sumsq[baseline_local, :]
            balanced_sumsq = sumsq[balanced_local, np.arange(permutations.shape[1])]
            baseline_train_mean = (total_sum - baseline_sum) / n_train
            balanced_train_mean = (total_sum - balanced_sum) / n_train
            baseline_mae = selected_mae(
                permutations,
                matrix,
                np.full(permutations.shape[1], baseline_index, dtype=int),
                baseline_train_mean,
            )
            balanced_mae = selected_mae(permutations, matrix, balanced_indices, balanced_train_mean)
            size_metrics = regression_metrics(
                test_sum=baseline_sum,
                test_sumsq=baseline_sumsq,
                n_test=n_test,
                total_sum=total_sum,
                n_total=n_total,
                mae=baseline_mae,
            )
            balanced_metrics = regression_metrics(
                test_sum=balanced_sum,
                test_sumsq=balanced_sumsq,
                n_test=n_test,
                total_sum=total_sum,
                n_total=n_total,
                mae=balanced_mae,
            )
            for key in ["rmse", "mse", "mae", "r2", "variance", "gap_sq"]:
                result[f"size_{key}"] = size_metrics[key]
                result[f"balanced_{key}"] = balanced_metrics[key]
            result["effect_rmse"] = size_metrics["rmse"] - balanced_metrics["rmse"]
            result["effect_mse"] = size_metrics["mse"] - balanced_metrics["mse"]
            result["effect_mae"] = size_metrics["mae"] - balanced_metrics["mae"]
            result["effect_r2"] = balanced_metrics["r2"] - size_metrics["r2"]
            result["effect_test_variance"] = size_metrics["variance"] - balanced_metrics["variance"]
            result["effect_squared_mean_gap"] = size_metrics["gap_sq"] - balanced_metrics["gap_sq"]
            residual = (
                result["effect_mse"]
                - result["effect_test_variance"]
                - result["effect_squared_mean_gap"]
            ).abs().max()
            if float(residual) > 1e-9:
                raise AssertionError(
                    f"{dataset}/{mode}/seed{partition_seed}/K{budget}: MSE decomposition residual {residual}"
                )
        else:
            size_metrics = classification_metrics(
                positives_test=baseline_sum,
                n_test=n_test,
                positives_total=total_sum,
                n_total=n_total,
            )
            balanced_metrics = classification_metrics(
                positives_test=balanced_sum,
                n_test=n_test,
                positives_total=total_sum,
                n_total=n_total,
            )
            for key in [
                "roc_auc",
                "average_precision",
                "brier",
                "log_loss",
                "test_prevalence",
                "train_prevalence",
                "prevalence_gap",
                "brier_variance",
                "gap_sq",
            ]:
                result[f"size_{key}"] = size_metrics[key]
                result[f"balanced_{key}"] = balanced_metrics[key]
            result["effect_roc_auc"] = balanced_metrics["roc_auc"] - size_metrics["roc_auc"]
            result["effect_average_precision"] = balanced_metrics["average_precision"] - size_metrics["average_precision"]
            result["effect_brier"] = size_metrics["brier"] - balanced_metrics["brier"]
            result["effect_log_loss"] = size_metrics["log_loss"] - balanced_metrics["log_loss"]
            result["effect_prevalence_gap"] = size_metrics["prevalence_gap"] - balanced_metrics["prevalence_gap"]
            result["effect_brier_variance"] = size_metrics["brier_variance"] - balanced_metrics["brier_variance"]
            result["effect_brier_gap_sq"] = size_metrics["gap_sq"] - balanced_metrics["gap_sq"]
            finite_auc = np.isfinite(result["effect_roc_auc"].to_numpy(dtype=float))
            if finite_auc.any():
                maximum_auc_effect = np.max(np.abs(result.loc[finite_auc, "effect_roc_auc"]))
                if float(maximum_auc_effect) > 1e-12:
                    raise AssertionError(
                        f"{dataset}/{mode}/seed{partition_seed}/K{budget}: constant-score AUC effect {maximum_auc_effect}"
                    )
            result["auc_undefined_size"] = ~np.isfinite(size_metrics["roc_auc"])
            result["auc_undefined_balanced"] = ~np.isfinite(balanced_metrics["roc_auc"])

        rows.append(result)

    output_frame = pd.concat(rows, ignore_index=True)
    if len(output_frame) != expected_seed_rows(permutations.shape[1], budgets):
        raise AssertionError(f"Unexpected output row count for {output.name}")
    temporary = output.with_suffix(".tmp")
    output_frame.to_csv(temporary, index=False)
    temporary.replace(output)
    print(
        f"  DONE {dataset}/{mode}/seed{partition_seed}: "
        f"{len(cache.hashes):,} unique candidates at {max(budgets):,} draws; "
        f"{len(output_frame):,} rows",
        flush=True,
    )
    return output


def run_all(config: dict, *, force: bool, force_cache: bool, datasets_filter: set[str] | None) -> list[Path]:
    fingerprint = protocol_fingerprint(config)
    outputs: list[Path] = []
    started = time.time()
    conditions = [
        (dataset, mode)
        for dataset, specification in config["datasets"].items()
        if datasets_filter is None or dataset in datasets_filter
        for mode in specification["scaffold_modes"]
    ]
    total_seeds = len(conditions) * len(config["partition_seeds"])
    completed = 0

    for dataset, mode in conditions:
        specification = config["datasets"][dataset]
        task_type = specification["task_type"]
        budgets = [int(value) for value in specification["budgets"]]
        frame, clean_path, clean_sha256 = validate_clean_frame(dataset, task_type)
        permutations, permutation_seeds = generate_permutations(frame["target"].to_numpy(dtype=float), dataset, config)
        print(
            f"\n========== {dataset} | {task_type} | {mode} | "
            f"N={len(frame):,} | permutations={config['n_permutations']} ==========",
            flush=True,
        )
        print(f"input={clean_path} sha256={clean_sha256}", flush=True)
        for partition_seed in config["partition_seeds"]:
            outputs.append(
                run_seed(
                    frame=frame,
                    permutations=permutations,
                    permutation_seeds=permutation_seeds,
                    dataset=dataset,
                    task_type=task_type,
                    mode=mode,
                    partition_seed=int(partition_seed),
                    budgets=budgets,
                    clean_sha256=clean_sha256,
                    fingerprint=fingerprint,
                    force=force,
                    force_cache=force_cache,
                )
            )
            completed += 1
            elapsed = time.time() - started
            rate = elapsed / completed
            remaining = rate * (total_seeds - completed)
            print(
                f"  PROGRESS {completed}/{total_seeds} seed-conditions | "
                f"elapsed={elapsed/3600:.2f} h | ETA={remaining/3600:.2f} h",
                flush=True,
            )
    return outputs


def aggregate_results(config: dict, outputs: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(path, keep_default_na=False, na_values=["", "nan", "NaN"]) for path in outputs]
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(TABLES / "null_simulation_partition_seed_effects.csv", index=False)

    identity = ["dataset", "task_type", "scaffold_mode", "budget", "permutation_id", "permutation_seed"]
    numeric = raw.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"partition_seed", "budget", "permutation_id", "permutation_seed", "n_total", "n_test"}
    aggregate_columns = [
        column
        for column in numeric
        if column not in excluded and column not in {"auc_undefined_size", "auc_undefined_balanced"}
    ]
    seed_aggregate = raw.groupby(identity, as_index=False)[aggregate_columns].mean()
    for column in ["auc_undefined_size", "auc_undefined_balanced"]:
        if column in raw:
            undefined = (
                raw.assign(**{column: raw[column].astype(float)})
                .groupby(identity, as_index=False)[column]
                .mean()
            )
            seed_aggregate = seed_aggregate.merge(undefined, on=identity, how="left")
    seed_aggregate.to_csv(TABLES / "null_simulation_permutation_level_effects.csv", index=False)

    long_rows: list[dict] = []
    for task_type, effect_columns in EFFECT_COLUMNS.items():
        subset = seed_aggregate[seed_aggregate["task_type"].eq(task_type)]
        for keys, group in subset.groupby(["dataset", "scaffold_mode", "budget"], sort=True):
            dataset, mode, budget = keys
            for metric in effect_columns:
                if metric not in group:
                    continue
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    statistics = {
                        "mean": np.nan,
                        "sd": np.nan,
                        "q025": np.nan,
                        "median": np.nan,
                        "q975": np.nan,
                        "fraction_positive": np.nan,
                    }
                else:
                    statistics = {
                        "mean": float(np.mean(values)),
                        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                        "q025": float(np.quantile(values, 0.025)),
                        "median": float(np.quantile(values, 0.5)),
                        "q975": float(np.quantile(values, 0.975)),
                        "fraction_positive": float(np.mean(values > 0)),
                    }
                long_rows.append(
                    {
                        "dataset": dataset,
                        "task_type": task_type,
                        "scaffold_mode": mode,
                        "budget": int(budget),
                        "metric": metric,
                        "n_permutations_valid": int(values.size),
                        **statistics,
                    }
                )
    summary = pd.DataFrame(long_rows)
    summary.to_csv(TABLES / "null_metric_effect_summary.csv", index=False)

    condition_budget_cells = raw[["dataset", "scaffold_mode", "budget"]].drop_duplicates()
    expected_cells = int(len(condition_budget_cells) * int(config["n_permutations"]))
    if len(seed_aggregate) != expected_cells:
        raise AssertionError(
            f"Permutation-level cell count {len(seed_aggregate)} != expected {expected_cells} "
            "for the dataset conditions present in this run"
        )

    regression = seed_aggregate[seed_aggregate["task_type"].eq("regression")].copy()
    residual = (
        regression["effect_mse"]
        - regression["effect_test_variance"]
        - regression["effect_squared_mean_gap"]
    ).abs()
    if not residual.empty and float(residual.max()) > 1e-9:
        raise AssertionError(f"Aggregate MSE decomposition residual {residual.max()}")
    pd.DataFrame(
        {
            "max_abs_mse_decomposition_residual": [float(residual.max()) if not residual.empty else 0.0],
            "permutation_level_rows": [len(seed_aggregate)],
            "raw_partition_seed_rows": [len(raw)],
        }
    ).to_csv(TABLES / "null_simulation_quality_gate_summary.csv", index=False)
    return seed_aggregate, summary


def summary_slice(summary: pd.DataFrame, dataset: str, mode: str, metric: str) -> pd.DataFrame:
    return summary[
        summary["dataset"].eq(dataset)
        & summary["scaffold_mode"].eq(mode)
        & summary["metric"].eq(metric)
    ].sort_values("budget")


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(
        FIGURES / f"{stem}.tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        bbox_inches="tight",
    )
    plt.close(figure)


def build_figures(summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    for axis, dataset in zip(axes, ["ESOL", "FreeSolv"]):
        for mode, marker in [("single_group", "o"), ("singleton", "s")]:
            data = summary_slice(summary, dataset, mode, "effect_rmse")
            axis.plot(data["budget"], data["mean"], marker=marker, linewidth=1.4, label=mode.replace("_", " "))
            axis.fill_between(data["budget"], data["q025"], data["q975"], alpha=0.18)
        axis.axhline(0, color="0.35", linestyle="--", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_title(dataset, loc="left", fontweight="bold")
        axis.set_xlabel("Requested candidate draws")
        axis.set_ylabel("Null RMSE effect\n(size − response-aware)")
        axis.grid(axis="both", color="0.90", linewidth=0.6)
        axis.legend(frameon=False)
    figure.suptitle("Endpoint-permutation null simulation: regression metric coupling", fontweight="bold")
    figure.tight_layout()
    save_figure(figure, "figure_mc1_regression_null_coupling")

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    metrics = [
        ("effect_brier", "Brier"),
        ("effect_log_loss", "Log loss"),
        ("effect_average_precision", "Average precision"),
        ("effect_roc_auc", "ROC–AUC"),
    ]
    for axis, dataset in zip(axes.ravel(), ["BACE", "BBBP", "ClinTox", "HIV"]):
        for metric, label in metrics:
            data = summary_slice(summary, dataset, "single_group", metric)
            axis.plot(data["budget"], data["mean"], marker="o", linewidth=1.1, label=label)
        axis.axhline(0, color="0.35", linestyle="--", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_title(dataset, loc="left", fontweight="bold")
        axis.set_xlabel("Requested candidate draws")
        axis.set_ylabel("Null metric effect")
        axis.grid(axis="both", color="0.90", linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=4, loc="lower center")
    figure.suptitle("Endpoint-permutation null simulation: classification metric coupling", fontweight="bold")
    figure.tight_layout(rect=[0, 0.07, 1, 0.96])
    save_figure(figure, "figure_mc2_classification_null_coupling")

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    for axis, dataset in zip(axes, ["ESOL", "FreeSolv"]):
        for metric, label, marker in [
            ("effect_mse", "Total MSE effect", "o"),
            ("effect_squared_mean_gap", "Squared mean-gap contribution", "s"),
            ("effect_test_variance", "Test-variance contribution", "^"),
        ]:
            data = summary_slice(summary, dataset, "single_group", metric)
            axis.plot(data["budget"], data["mean"], marker=marker, linewidth=1.25, label=label)
        axis.axhline(0, color="0.35", linestyle="--", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_title(dataset, loc="left", fontweight="bold")
        axis.set_xlabel("Requested candidate draws")
        axis.set_ylabel("Null MSE component effect")
        axis.grid(axis="both", color="0.90", linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
    figure.suptitle("Decomposition of response-aware null MSE effects", fontweight="bold")
    figure.tight_layout(rect=[0, 0.09, 1, 0.96])
    save_figure(figure, "figure_mc3_mse_decomposition")


def build_checklist() -> None:
    frame = pd.DataFrame(CHECKLIST, columns=["audit_item", "minimum_report", "risk_if_omitted"])
    frame.to_csv(TABLES / "qsar_benchmark_minimum_reporting_checklist.csv", index=False)
    lines = [
        "# Minimum reporting checklist for response-aware QSAR benchmarks",
        "",
        "| Audit item | Minimum report | Risk if omitted |",
        "|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(f"| {row.audit_item} | {row.minimum_report} | {row.risk_if_omitted} |")
    (REPORTS / "QSAR_BENCHMARK_REPORTING_CHECKLIST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Minimum reporting checklist for response-aware QSAR benchmark construction.}",
        r"\label{tab:qsar-checklist}",
        r"\small",
        r"\begin{tabular}{p{0.18\textwidth}p{0.36\textwidth}p{0.36\textwidth}}",
        r"\hline",
        r"Audit item & Minimum report & Risk if omitted \",
        r"\hline",
    ]
    escape = lambda value: str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")
    for row in frame.itertuples(index=False):
        tex.append(f"{escape(row.audit_item)} & {escape(row.minimum_report)} & {escape(row.risk_if_omitted)} \\")
    tex.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    (REPORTS / "qsar_benchmark_checklist.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def format_interval(row: pd.Series, digits: int = 4) -> str:
    return f"{row['mean']:.{digits}f} [{row['q025']:.{digits}f}, {row['q975']:.{digits}f}]"


def build_report(config: dict, summary: pd.DataFrame, seed_aggregate: pd.DataFrame, fingerprint: dict) -> None:
    lines = [
        "# SAR/QSAR metric-coupling null simulation report",
        "",
        f"- Protocol version: {config['protocol_version']}",
        f"- Endpoint permutations per dataset: {config['n_permutations']}",
        f"- Partition seeds: {len(config['partition_seeds'])}",
        f"- Protocol SHA-256: `{fingerprint['protocol_sha256']}`",
        f"- Config SHA-256: `{fingerprint['config_sha256']}`",
        f"- Git commit: `{git_value('rev-parse', 'HEAD')}`",
        "",
        "## Maximum-budget findings",
        "",
    ]
    for dataset, specification in config["datasets"].items():
        for mode in specification["scaffold_modes"]:
            maximum_budget = max(specification["budgets"])
            lines.append(f"### {dataset} — {mode} — {maximum_budget:,} requested draws")
            metrics = (
                ["effect_rmse", "effect_mse", "effect_squared_mean_gap", "effect_test_variance", "effect_mae", "effect_r2"]
                if specification["task_type"] == "regression"
                else ["effect_brier", "effect_log_loss", "effect_average_precision", "effect_roc_auc"]
            )
            for metric in metrics:
                row = summary[
                    summary["dataset"].eq(dataset)
                    & summary["scaffold_mode"].eq(mode)
                    & summary["budget"].eq(maximum_budget)
                    & summary["metric"].eq(metric)
                ]
                if row.empty:
                    continue
                item = row.iloc[0]
                lines.append(
                    f"- {metric}: {format_interval(item)}; "
                    f"fraction positive={item['fraction_positive']:.3f}; "
                    f"valid permutations={int(item['n_permutations_valid'])}"
                )
            lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "These simulations preserve molecular and scaffold geometry but destroy structure–endpoint association. They identify metric coupling under a molecular null; they do not estimate prospective QSAR utility, and they do not replace the empirical paired audit.",
            "",
            "## Generated artifacts",
            "",
            "- `tables/null_simulation_partition_seed_effects.csv`",
            "- `tables/null_simulation_permutation_level_effects.csv`",
            "- `tables/null_metric_effect_summary.csv`",
            "- `figures/figure_mc1_regression_null_coupling.*`",
            "- `figures/figure_mc2_classification_null_coupling.*`",
            "- `figures/figure_mc3_mse_decomposition.*`",
            "- `reports/QSAR_BENCHMARK_REPORTING_CHECKLIST.md`",
        ]
    )
    (REPORTS / "SARQSAR_NULL_SIMULATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(config: dict, fingerprint: dict, outputs: list[Path], status: str) -> None:
    payload = {
        "status": status,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        **fingerprint,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "python": sys.version,
        "platform": platform.platform(),
        "n_permutations": config["n_permutations"],
        "partition_seeds": config["partition_seeds"],
        "seed_checkpoint_files": [str(path.relative_to(ROOT)) for path in outputs],
        "output_root": str(OUT.relative_to(ROOT)),
        "completed_at_unix": time.time() if status == "complete" else None,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen SAR/QSAR molecular null simulation.")
    parser.add_argument("--permutations", type=int, default=None, help="Override 200 only for smoke testing; output hashes change.")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset subset or 'all'.")
    parser.add_argument("--force", action="store_true", help="Recompute validated seed checkpoints.")
    parser.add_argument("--force-cache", action="store_true", help="Rebuild candidate caches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    config = load_config(args.permutations)
    fingerprint = protocol_fingerprint(config)
    datasets_filter = None if args.datasets == "all" else {item.strip() for item in args.datasets.split(",") if item.strip()}
    unknown = set() if datasets_filter is None else datasets_filter.difference(config["datasets"])
    if unknown:
        raise KeyError(f"Unknown dataset filter: {sorted(unknown)}")

    if MANIFEST.exists() and not args.force:
        previous = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for key in ["protocol_sha256", "config_sha256"]:
            if previous.get(key) not in {None, fingerprint[key]}:
                raise AssertionError(
                    f"Existing output protocol mismatch for {key}. Use a new protocol version/output directory; do not overwrite v1 results."
                )

    print("=" * 88)
    print("SAR/QSAR METRIC-COUPLING NULL SIMULATION v1.0")
    print("Molecular geometry preserved; endpoints permuted; seed-level resume enabled")
    print("=" * 88)
    print("protocol_sha256:", fingerprint["protocol_sha256"])
    print("config_sha256  :", fingerprint["config_sha256"])
    print("git branch     :", git_value("branch", "--show-current"))
    print("git commit     :", git_value("rev-parse", "HEAD"))
    print("permutations   :", config["n_permutations"])
    print("partition seeds:", len(config["partition_seeds"]))
    started = time.time()
    outputs: list[Path] = []
    write_manifest(config, fingerprint, outputs, "running")
    try:
        outputs = run_all(
            config,
            force=args.force,
            force_cache=args.force_cache,
            datasets_filter=datasets_filter,
        )
        seed_aggregate, summary = aggregate_results(config, outputs)
        build_figures(summary)
        build_checklist()
        build_report(config, summary, seed_aggregate, fingerprint)
        write_manifest(config, fingerprint, outputs, "complete")
    except Exception:
        write_manifest(config, fingerprint, outputs, "failed")
        raise
    elapsed = time.time() - started
    print("\n" + "=" * 88)
    print("SAR/QSAR METRIC-COUPLING NULL SIMULATION: PASS")
    print(f"Elapsed: {elapsed/3600:.2f} h")
    print("Output :", OUT)
    print("Report :", REPORTS / "SARQSAR_NULL_SIMULATION_REPORT.md")
    print("Summary:", TABLES / "null_metric_effect_summary.csv")
    print("=" * 88)


if __name__ == "__main__":
    main()
