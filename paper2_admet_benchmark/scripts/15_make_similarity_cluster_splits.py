from __future__ import annotations

"""Create repeated label-blind similarity-cluster confirmatory splits.

Molecules are clustered with a reproducible leader algorithm on binary ECFP4
fingerprints. For each randomized seed, a molecule joins its most similar existing
leader when Tanimoto similarity is at least the pre-specified threshold; otherwise it
starts a new cluster. Clusters are then assigned intact to train/calibration/test using
cluster size only. Target labels and target values are never used.

Default confirmatory columns:
- split_confirm_cluster_seed101 ... seed105

This is an approximate, order-randomized leader clustering analysis, not exact Butina
clustering. The distinction must be stated in the manuscript.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper2_admet_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
MANIFEST_DIR = PAPER_DIR / "data" / "manifests"
TABLE_DIR = PAPER_DIR / "results" / "tables"

DEFAULT_SEEDS = list(range(101, 106))
DEFAULT_SIMILARITY_THRESHOLD = 0.60
DEFAULT_MAX_ENDPOINT_N = 10_000
TRAIN_FRAC = 0.60
CALIBRATION_FRAC = 0.20
TEST_FRAC = 0.20
ROLE_FRACTIONS = {
    "train": TRAIN_FRAC,
    "calibration": CALIBRATION_FRAC,
    "test": TEST_FRAC,
}
ROLE_ORDER = ["train", "calibration", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create confirmatory ECFP similarity-cluster splits.")
    parser.add_argument("--endpoints", default=None, help="Optional comma-separated endpoints.")
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--max-endpoint-n", type=int, default=DEFAULT_MAX_ENDPOINT_N)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique.")
    return seeds


def available_endpoints() -> list[str]:
    return sorted(path.name.replace("_features.npz", "") for path in PROCESSED_DIR.glob("*_features.npz"))


def numpy_rows_to_bitvectors(X: np.ndarray) -> list[DataStructs.ExplicitBitVect]:
    bitvectors: list[DataStructs.ExplicitBitVect] = []
    n_bits = int(X.shape[1])
    for row in X:
        fp = DataStructs.ExplicitBitVect(n_bits)
        fp.SetBitsFromList(np.flatnonzero(row).astype(int).tolist())
        bitvectors.append(fp)
    return bitvectors


def leader_cluster_ids(
    bitvectors: list[DataStructs.ExplicitBitVect],
    seed: int,
    similarity_threshold: float,
) -> np.ndarray:
    """Return one cluster id per molecule using randomized leader clustering."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bitvectors))

    leader_fps: list[DataStructs.ExplicitBitVect] = []
    cluster_ids = np.full(len(bitvectors), -1, dtype=int)

    for processed_n, molecule_idx in enumerate(order, start=1):
        fp = bitvectors[int(molecule_idx)]
        if not leader_fps:
            leader_fps.append(fp)
            cluster_ids[int(molecule_idx)] = 0
            continue

        similarities = DataStructs.BulkTanimotoSimilarity(fp, leader_fps)
        best_cluster = int(np.argmax(similarities))
        best_similarity = float(similarities[best_cluster])
        if best_similarity >= similarity_threshold:
            cluster_ids[int(molecule_idx)] = best_cluster
        else:
            cluster_ids[int(molecule_idx)] = len(leader_fps)
            leader_fps.append(fp)

        if processed_n % 1000 == 0:
            print(
                f"  clustered {processed_n}/{len(bitvectors)} molecules; "
                f"leaders={len(leader_fps)}"
            )

    if (cluster_ids < 0).any():
        raise RuntimeError("Leader clustering left unassigned molecules.")
    return cluster_ids


def assign_groups_by_size(group_ids: np.ndarray, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed + 20_000)
    groups = [
        (int(group_id), np.flatnonzero(group_ids == group_id))
        for group_id in np.unique(group_ids)
    ]
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    n_total = len(group_ids)
    target_sizes = {role: max(1.0, fraction * n_total) for role, fraction in ROLE_FRACTIONS.items()}
    role_counts = {role: 0 for role in ROLE_ORDER}
    role_indices: dict[str, list[int]] = {role: [] for role in ROLE_ORDER}

    tie_order = list(ROLE_ORDER)
    rng.shuffle(tie_order)
    tie_rank = {role: idx for idx, role in enumerate(tie_order)}

    for _group_id, indices in groups:
        group_n = len(indices)
        chosen = min(
            ROLE_ORDER,
            key=lambda role: (
                (role_counts[role] + group_n) / target_sizes[role],
                tie_rank[role],
            ),
        )
        role_counts[chosen] += group_n
        role_indices[chosen].extend(indices.astype(int).tolist())

    split = pd.Series("unused", index=np.arange(n_total), dtype="object")
    for role, indices in role_indices.items():
        split.iloc[indices] = role
    return split


def summarize(
    df: pd.DataFrame,
    endpoint: str,
    seed: int,
    split_col: str,
    cluster_ids: np.ndarray,
    threshold: float,
    elapsed_seconds: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    task_type = str(df["task_type"].iloc[0])
    for role in ROLE_ORDER:
        part = df[df[split_col] == role]
        part_cluster_ids = cluster_ids[part.index.to_numpy()]
        row: dict[str, object] = {
            "endpoint": endpoint,
            "task_type": task_type,
            "split_col": split_col,
            "split_type": "confirm_similarity_leader_cluster",
            "seed": seed,
            "similarity_threshold": threshold,
            "role": role,
            "n": int(len(part)),
            "fraction": float(len(part) / len(df)),
            "n_clusters": int(len(np.unique(part_cluster_ids))),
            "total_clusters": int(len(np.unique(cluster_ids))),
            "elapsed_seconds": elapsed_seconds,
            "target_mean": float(part["target"].mean()) if len(part) else np.nan,
        }
        if task_type == "classification" and len(part):
            positives = int(part["target"].sum())
            row.update(
                {
                    "positive_count": positives,
                    "negative_count": int(len(part) - positives),
                    "positive_ratio": float(positives / len(part)),
                }
            )
        else:
            row.update(
                {
                    "positive_count": np.nan,
                    "negative_count": np.nan,
                    "positive_ratio": np.nan,
                }
            )
        rows.append(row)
    return rows


def validate_cluster_exclusion(split: pd.Series, cluster_ids: np.ndarray) -> int:
    frame = pd.DataFrame({"split": split.to_numpy(), "cluster_id": cluster_ids})
    role_counts = frame.groupby("cluster_id")["split"].nunique()
    return int((role_counts > 1).sum())


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if not 0 < args.similarity_threshold <= 1:
        raise ValueError("similarity-threshold must be in (0, 1].")
    if args.max_endpoint_n < 1:
        raise ValueError("max-endpoint-n must be positive.")

    endpoints = parse_csv_list(args.endpoints) or available_endpoints()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for endpoint in endpoints:
        feature_path = PROCESSED_DIR / f"{endpoint}_features.npz"
        split_path = PROCESSED_DIR / f"{endpoint}_splits.csv"
        if not feature_path.exists() or not split_path.exists():
            raise FileNotFoundError(f"Missing feature or split file for endpoint={endpoint}")

        feature_data = np.load(feature_path, allow_pickle=True)
        X = feature_data["X"]
        df = pd.read_csv(split_path)
        if len(df) != len(X):
            raise ValueError(f"{endpoint}: split rows {len(df)} != fingerprint rows {len(X)}")

        if len(df) > args.max_endpoint_n:
            reason = (
                f"n={len(df)} exceeds pre-specified max_endpoint_n={args.max_endpoint_n} "
                "for exact repeated leader clustering"
            )
            print(f"[SKIP] {endpoint}: {reason}")
            skipped_rows.append({"endpoint": endpoint, "reason": reason})
            continue

        print(f"\n========== {endpoint} n={len(df)} ==========")
        backup_path = PROCESSED_DIR / f"{endpoint}_splits.development_backup.csv"
        if not backup_path.exists():
            shutil.copy2(split_path, backup_path)
            print("saved development backup:", backup_path)

        bitvectors = numpy_rows_to_bitvectors(X)
        cluster_frame = pd.DataFrame({"row_index": np.arange(len(df), dtype=int)})

        for seed in seeds:
            split_col = f"split_confirm_cluster_seed{seed}"
            cluster_col = f"cluster_seed{seed}"
            existing_cluster_path = PROCESSED_DIR / f"{endpoint}_confirmatory_clusters.csv"

            if split_col in df.columns and not args.overwrite and existing_cluster_path.exists():
                existing_clusters = pd.read_csv(existing_cluster_path)
                if cluster_col in existing_clusters.columns:
                    cluster_ids = existing_clusters[cluster_col].to_numpy(dtype=int)
                    cluster_frame[cluster_col] = cluster_ids
                    print(f"retaining existing {split_col}")
                else:
                    raise KeyError(
                        f"{split_col} exists but {cluster_col} is missing from {existing_cluster_path}; "
                        "rerun with --overwrite."
                    )
                elapsed = 0.0
            else:
                t0 = time.time()
                cluster_ids = leader_cluster_ids(
                    bitvectors,
                    seed=seed,
                    similarity_threshold=args.similarity_threshold,
                )
                df[split_col] = assign_groups_by_size(cluster_ids, seed=seed).to_numpy()
                cluster_frame[cluster_col] = cluster_ids
                elapsed = time.time() - t0

            leakage_n = validate_cluster_exclusion(df[split_col], cluster_ids)
            roles = set(df[split_col].unique())
            passed = roles == set(ROLE_ORDER) and leakage_n == 0
            validation_rows.append(
                {
                    "endpoint": endpoint,
                    "split_col": split_col,
                    "seed": seed,
                    "validation_passed": passed,
                    "cluster_leakage_count": leakage_n,
                    "roles": ",".join(sorted(roles)),
                }
            )
            summary_rows.extend(
                summarize(
                    df=df,
                    endpoint=endpoint,
                    seed=seed,
                    split_col=split_col,
                    cluster_ids=cluster_ids,
                    threshold=args.similarity_threshold,
                    elapsed_seconds=elapsed,
                )
            )
            manifest_rows.append(
                {
                    "endpoint": endpoint,
                    "split_col": split_col,
                    "split_type": "confirm_similarity_leader_cluster",
                    "seed": seed,
                    "similarity_threshold": args.similarity_threshold,
                    "algorithm": "randomized_leader_clustering",
                    "label_blind_assignment": True,
                    "split_file": str(split_path.relative_to(ROOT)),
                    "cluster_file": str(
                        (PROCESSED_DIR / f"{endpoint}_confirmatory_clusters.csv").relative_to(ROOT)
                    ),
                }
            )

        df.to_csv(split_path, index=False)
        cluster_path = PROCESSED_DIR / f"{endpoint}_confirmatory_clusters.csv"
        cluster_frame.to_csv(cluster_path, index=False)
        print("updated", split_path)
        print("saved", cluster_path)

    manifest = pd.DataFrame(manifest_rows)
    summary = pd.DataFrame(summary_rows)
    validation = pd.DataFrame(validation_rows)
    skipped = pd.DataFrame(skipped_rows)

    manifest_path = MANIFEST_DIR / "confirmatory_cluster_manifest.csv"
    summary_path = TABLE_DIR / "confirmatory_cluster_split_summary.csv"
    validation_path = TABLE_DIR / "confirmatory_cluster_split_validation.csv"
    skipped_path = TABLE_DIR / "confirmatory_cluster_skipped_endpoints.csv"
    manifest.to_csv(manifest_path, index=False)
    summary.to_csv(summary_path, index=False)
    validation.to_csv(validation_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    failures = int((~validation["validation_passed"].astype(bool)).sum()) if not validation.empty else 0
    print("\nsaved", manifest_path)
    print("saved", summary_path)
    print("saved", validation_path)
    print("saved", skipped_path)
    print(
        "Similarity-cluster confirmatory split generation complete. "
        f"Validation failures: {failures}; skipped endpoints: {len(skipped)}."
    )
    if failures:
        raise SystemExit("One or more cluster splits failed structural validation.")


if __name__ == "__main__":
    main()
