from __future__ import annotations

"""Compute chemical applicability-domain diagnostics for Paper 2.

For every endpoint and split, this script compares calibration/test molecules with
that split's training molecules using ECFP4 Tanimoto similarity. It reports:
- maximum Tanimoto similarity to the training set;
- mean top-k Tanimoto similarity to the training set;
- whether the Bemis-Murcko scaffold was unseen in training;
- high/medium/low chemical-domain bins.

Default bins:
- high_domain: max similarity >= 0.70
- medium_domain: 0.40 <= max similarity < 0.70
- low_domain: max similarity < 0.40

Inputs:
- data/processed/<endpoint>_features.npz
- data/processed/<endpoint>_splits.csv

Outputs:
- results/applicability/applicability_details_<mode>.csv  (local, potentially large)
- results/applicability/applicability_summary_<mode>.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper2_admet_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
OUTPUT_DIR = PAPER_DIR / "results" / "applicability"

MVP_ENDPOINTS = ["bbbp", "clintox", "esol", "lipophilicity"]
ALL_RANDOM_SPLITS = [f"split_random_seed{i}" for i in range(5)]
MVP_SPLITS = ["split_random_seed0", "split_scaffold"]
ROLE_ORDER = ["calibration", "test"]
DOMAIN_ORDER = ["high_domain", "medium_domain", "low_domain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Paper 2 applicability-domain diagnostics.")
    parser.add_argument("--mode", choices=["mvp", "full"], default="mvp")
    parser.add_argument(
        "--endpoints",
        default=None,
        help="Optional comma-separated endpoints. Overrides mode defaults.",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Optional comma-separated split columns. Overrides mode defaults.",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--high-threshold", type=float, default=0.70)
    parser.add_argument("--medium-threshold", type=float, default=0.40)
    return parser.parse_args()


def available_endpoints() -> list[str]:
    return sorted(path.name.replace("_features.npz", "") for path in PROCESSED_DIR.glob("*_features.npz"))


def resolve_endpoints(mode: str, value: str | None) -> list[str]:
    if value:
        return [item.strip().lower() for item in value.split(",") if item.strip()]
    return MVP_ENDPOINTS if mode == "mvp" else available_endpoints()


def resolve_splits(mode: str, value: str | None) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return MVP_SPLITS if mode == "mvp" else ALL_RANDOM_SPLITS + ["split_scaffold"]


def pairwise_topk_tanimoto(
    X_eval: np.ndarray,
    X_train: np.ndarray,
    chunk_size: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return maximum and mean top-k Tanimoto similarities for each evaluation row.

    The computation is chunked over evaluation rows to keep memory bounded. The
    fingerprint matrices are binary ECFP arrays, so intersection is a matrix dot
    product and union is derived from row bit counts.
    """
    if len(X_train) == 0:
        raise ValueError("Training fingerprint matrix is empty.")
    if len(X_eval) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    X_train_f = X_train.astype(np.float32, copy=False)
    train_bits = X_train_f.sum(axis=1)
    k = min(max(int(top_k), 1), len(X_train_f))

    max_values: list[np.ndarray] = []
    topk_means: list[np.ndarray] = []

    for start in range(0, len(X_eval), chunk_size):
        stop = min(start + chunk_size, len(X_eval))
        chunk = X_eval[start:stop].astype(np.float32, copy=False)
        intersections = chunk @ X_train_f.T
        unions = chunk.sum(axis=1, keepdims=True) + train_bits[None, :] - intersections
        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=np.float32),
            where=unions > 0,
        )

        max_values.append(similarities.max(axis=1))
        if k == len(X_train_f):
            top_values = similarities
        else:
            top_values = np.partition(similarities, similarities.shape[1] - k, axis=1)[:, -k:]
        topk_means.append(top_values.mean(axis=1))

    return np.concatenate(max_values), np.concatenate(topk_means)


def assign_domain(max_similarity: np.ndarray, high: float, medium: float) -> np.ndarray:
    return np.where(
        max_similarity >= high,
        "high_domain",
        np.where(max_similarity >= medium, "medium_domain", "low_domain"),
    )


def analyze_endpoint(
    endpoint: str,
    split_cols: list[str],
    chunk_size: int,
    top_k: int,
    high_threshold: float,
    medium_threshold: float,
) -> list[pd.DataFrame]:
    feature_path = PROCESSED_DIR / f"{endpoint}_features.npz"
    split_path = PROCESSED_DIR / f"{endpoint}_splits.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    feature_data = np.load(feature_path, allow_pickle=True)
    X = feature_data["X"]
    split_df = pd.read_csv(split_path)
    if len(split_df) != len(X):
        raise ValueError(f"{endpoint}: split rows {len(split_df)} != feature rows {len(X)}")

    frames: list[pd.DataFrame] = []
    for split_col in split_cols:
        if split_col not in split_df.columns:
            print(f"[SKIP] {endpoint}: missing split column {split_col}")
            continue

        train_idx = split_df.index[split_df[split_col] == "train"].to_numpy()
        train_scaffolds = set(split_df.iloc[train_idx]["scaffold"].astype(str))
        X_train = X[train_idx]

        for role in ROLE_ORDER:
            eval_idx = split_df.index[split_df[split_col] == role].to_numpy()
            print(
                f"applicability endpoint={endpoint} split={split_col} role={role} "
                f"n_train={len(train_idx)} n_eval={len(eval_idx)}"
            )
            max_sim, topk_mean = pairwise_topk_tanimoto(
                X_eval=X[eval_idx],
                X_train=X_train,
                chunk_size=chunk_size,
                top_k=top_k,
            )
            eval_rows = split_df.iloc[eval_idx]
            scaffold_values = eval_rows["scaffold"].astype(str).to_numpy()
            unseen_scaffold = np.array([value not in train_scaffolds for value in scaffold_values], dtype=bool)
            domain_bin = assign_domain(max_sim, high=high_threshold, medium=medium_threshold)

            frames.append(
                pd.DataFrame(
                    {
                        "endpoint": endpoint,
                        "task_type": str(split_df["task_type"].iloc[0]),
                        "split_col": split_col,
                        "role": role,
                        "row_index": eval_idx,
                        "canonical_smiles": eval_rows["canonical_smiles"].to_numpy(),
                        "target": eval_rows["target"].to_numpy(),
                        "scaffold": scaffold_values,
                        "max_tanimoto_to_train": max_sim,
                        f"mean_top{top_k}_tanimoto_to_train": topk_mean,
                        "unseen_scaffold": unseen_scaffold,
                        "domain_bin": domain_bin,
                        "high_threshold": high_threshold,
                        "medium_threshold": medium_threshold,
                    }
                )
            )
    return frames


def summarize_details(details: pd.DataFrame, top_k: int) -> pd.DataFrame:
    topk_col = f"mean_top{top_k}_tanimoto_to_train"
    rows = []
    for keys, group in details.groupby(["endpoint", "task_type", "split_col", "role", "domain_bin"], sort=True):
        endpoint, task_type, split_col, role, domain_bin = keys
        rows.append(
            {
                "endpoint": endpoint,
                "task_type": task_type,
                "split_col": split_col,
                "role": role,
                "domain_bin": domain_bin,
                "n": int(len(group)),
                "fraction_within_role": float(len(group) / len(details[(details["endpoint"] == endpoint) & (details["split_col"] == split_col) & (details["role"] == role)])),
                "mean_max_tanimoto": float(group["max_tanimoto_to_train"].mean()),
                "median_max_tanimoto": float(group["max_tanimoto_to_train"].median()),
                f"mean_{topk_col}": float(group[topk_col].mean()),
                "unseen_scaffold_rate": float(group["unseen_scaffold"].mean()),
                "target_mean": float(group["target"].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["domain_bin"] = pd.Categorical(summary["domain_bin"], categories=DOMAIN_ORDER, ordered=True)
        summary = summary.sort_values(["endpoint", "split_col", "role", "domain_bin"]).reset_index(drop=True)
    return summary


def main() -> None:
    args = parse_args()
    if not 0 <= args.medium_threshold < args.high_threshold <= 1:
        raise ValueError("Require 0 <= medium_threshold < high_threshold <= 1.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    endpoints = resolve_endpoints(args.mode, args.endpoints)
    split_cols = resolve_splits(args.mode, args.splits)

    all_frames: list[pd.DataFrame] = []
    for endpoint in endpoints:
        print("\n==========", endpoint, "==========")
        all_frames.extend(
            analyze_endpoint(
                endpoint=endpoint,
                split_cols=split_cols,
                chunk_size=args.chunk_size,
                top_k=args.top_k,
                high_threshold=args.high_threshold,
                medium_threshold=args.medium_threshold,
            )
        )

    details = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    summary = summarize_details(details, top_k=args.top_k) if not details.empty else pd.DataFrame()

    details_path = OUTPUT_DIR / f"applicability_details_{args.mode}.csv"
    summary_path = OUTPUT_DIR / f"applicability_summary_{args.mode}.csv"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nsaved", details_path)
    print("saved", summary_path)
    print("Applicability-domain analysis complete.")


if __name__ == "__main__":
    main()
