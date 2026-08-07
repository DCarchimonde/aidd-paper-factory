from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.modeling_v3 import (
    build_model,
    classification_metrics,
    load_or_build_morgan_matrix,
    production_model_seed,
)
from shared_utils.scaffold_identity import prepare_scaffold_frame
from shared_utils.split_candidate_pool_v3 import (
    generate_candidate_pool,
    materialize_candidate_split,
    select_paired_candidates,
)
from shared_utils.split_manifest_v2 import split_manifest_rows, summarize_partition

RDLogger.DisableLog("rdApp.warning")

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
SOURCE_DATA_DIR = PAPER_DIR / "data" / "processed_v2"
PROTOCOL_FILE = PAPER_DIR / "PARENT_FRAGMENT_SENSITIVITY_PROTOCOL_V3.md"
OUT_DIR = PAPER_DIR / "results" / "parent_fragment_sensitivity_v3"
PARENT_DATA_DIR = OUT_DIR / "processed_parent"
SPLIT_DIR = OUT_DIR / "splits"
FROZEN_DIR = OUT_DIR / "frozen"
CACHE_DIR = OUT_DIR / "fingerprint_cache"
JOB_DIR = OUT_DIR / "jobs"
TABLE_DIR = OUT_DIR / "tables"
for directory in (OUT_DIR, PARENT_DATA_DIR, SPLIT_DIR, FROZEN_DIR, CACHE_DIR, JOB_DIR, TABLE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DATASETS = ("BBBP", "ClinTox", "HIV")
MODELS = ("LR", "RF", "XGB")
PROTOCOLS = ("size_matched_scaffold", "target_balanced_scaffold")
SEEDS = (
    42, 123, 2024, 2026, 3407,
    7, 19, 71, 101, 211,
    307, 401, 503, 601, 701,
    809, 907, 1009, 1201, 1429,
)
N_CANDIDATES = 300
BOOTSTRAP_REPS = 10000
BOOTSTRAP_BASE_SEED = 20260808
EXPECTED_JOBS = len(DATASETS) * len(PROTOCOLS) * len(SEEDS) * len(MODELS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def fragment_key(mol: Chem.Mol) -> tuple[int, int, str]:
    heavy = int(mol.GetNumHeavyAtoms())
    carbon = int(sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()))
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return heavy, carbon, smiles


def choose_parent_smiles(smiles: str) -> tuple[str, int, str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise AssertionError(f"Unexpected parse failure in clean_v2: {smiles}")
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    ranked = sorted(fragments, key=fragment_key, reverse=True)
    if not ranked:
        raise AssertionError(f"No RDKit fragments returned: {smiles}")
    parent = Chem.MolToSmiles(ranked[0], canonical=True, isomericSmiles=True)
    extras = [
        Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
        for fragment in ranked[1:]
    ]
    return parent, len(ranked), "|".join(extras)


def build_parent_dataset(dataset: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    source_path = SOURCE_DATA_DIR / f"{dataset.lower()}_clean_v2.csv"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source = pd.read_csv(source_path, keep_default_na=False, low_memory=False)
    source["target"] = pd.to_numeric(source["target"], errors="raise")
    if source["canonical_smiles"].duplicated().any():
        raise AssertionError(f"Source clean_v2 unexpectedly has duplicate molecules: {dataset}")

    mapped_rows: list[dict] = []
    for source_index, row in source.iterrows():
        source_smiles = str(row["canonical_smiles"])
        parent_smiles, n_fragments, extras = choose_parent_smiles(source_smiles)
        mapped_rows.append(
            {
                "source_index": int(source_index),
                "source_canonical_smiles": source_smiles,
                "parent_smiles": parent_smiles,
                "target": int(float(row["target"])),
                "n_fragments": int(n_fragments),
                "extra_fragments": extras,
            }
        )
    mapped = pd.DataFrame(mapped_rows)

    clean_rows: list[dict] = []
    decision_rows: list[dict] = []
    for parent_smiles, group in mapped.groupby("parent_smiles", sort=True):
        labels = sorted(set(group["target"].astype(int).tolist()))
        conflict = len(labels) > 1
        decision = (
            "exclude_conflicting_parent_labels"
            if conflict
            else "retain_or_collapse_consistent_parent"
        )
        decision_rows.append(
            {
                "dataset": dataset,
                "parent_smiles": parent_smiles,
                "n_source_records": int(len(group)),
                "target_values": "|".join(str(value) for value in labels),
                "decision": decision,
                "source_indices": "|".join(str(int(v)) for v in group["source_index"]),
                "source_smiles": "||".join(group["source_canonical_smiles"].astype(str)),
            }
        )
        if conflict:
            continue
        clean_rows.append(
            {
                "canonical_smiles": str(parent_smiles),
                "target": int(labels[0]),
                "n_source_records": int(len(group)),
                "source_indices": "|".join(str(int(v)) for v in group["source_index"]),
                "source_smiles": "||".join(group["source_canonical_smiles"].astype(str)),
                "n_multifragment_source_records": int(group["n_fragments"].gt(1).sum()),
            }
        )

    clean = pd.DataFrame(clean_rows).sort_values("canonical_smiles").reset_index(drop=True)
    decisions = pd.DataFrame(decision_rows)
    if clean.empty:
        raise AssertionError(f"Parent sensitivity produced no rows: {dataset}")
    if clean["canonical_smiles"].duplicated().any():
        raise AssertionError(f"Parent sensitivity still has duplicate parents: {dataset}")
    if set(clean["target"].astype(int).unique()) != {0, 1}:
        raise AssertionError(f"Parent sensitivity dataset lost a class: {dataset}")

    conflict_groups = int(decisions["decision"].eq("exclude_conflicting_parent_labels").sum())
    duplicate_groups = int(decisions["n_source_records"].gt(1).sum())
    summary = {
        "dataset": dataset,
        "source_clean_v2_rows": int(len(source)),
        "source_multifragment_rows": int(mapped["n_fragments"].gt(1).sum()),
        "unique_algorithmic_parents_before_conflict_exclusion": int(mapped["parent_smiles"].nunique()),
        "parent_duplicate_groups": duplicate_groups,
        "parent_conflicting_label_groups_excluded": conflict_groups,
        "parent_sensitivity_rows": int(len(clean)),
        "n_removed_relative_to_source": int(len(source) - len(clean)),
        "positive_fraction": float(clean["target"].mean()),
    }
    clean_path = PARENT_DATA_DIR / f"{dataset.lower()}_parent_sensitivity_v3.csv"
    decision_path = PARENT_DATA_DIR / f"{dataset.lower()}_parent_decisions_v3.csv"
    clean.to_csv(clean_path, index=False)
    decisions.to_csv(decision_path, index=False)
    summary["source_clean_v2_sha256"] = sha256_file(source_path)
    summary["parent_dataset_sha256"] = sha256_file(clean_path)
    return clean, summary, decisions


def class_gate(frame: pd.DataFrame, split_col: str, dataset: str, seed: int, protocol: str) -> None:
    for assignment in ("train", "test"):
        values = set(
            frame.loc[frame[split_col].eq(assignment), "target"].astype(int).unique().tolist()
        )
        if values != {0, 1}:
            raise AssertionError(
                f"Classification class gate failed: {dataset}/{protocol}/seed={seed}/{assignment}: {values}"
            )


def build_splits(parent_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifests: list[dict] = []
    audits: list[dict] = []
    pair_rows: list[dict] = []
    pool_frames: list[pd.DataFrame] = []

    for dataset in DATASETS:
        print(f"\n========== BUILD PARENT SPLITS: {dataset} ==========")
        base = prepare_scaffold_frame(parent_frames[dataset], acyclic_mode="single_group")
        for seed in SEEDS:
            candidates, groups, generation_meta = generate_candidate_pool(
                base, seed=seed, n_candidates=N_CANDIDATES
            )
            size_candidate, balanced_candidate, pool_table, pair_meta = select_paired_candidates(
                candidates,
                groups,
                seed=seed,
                total_n=len(base),
                total_target_sum=float(base["target"].sum()),
            )
            if not bool(pair_meta["exact_size_match"]):
                raise AssertionError(f"Exact-size pair failed: {dataset}/seed={seed}")
            pool_table.insert(0, "dataset", dataset)
            pool_table.insert(1, "partition_seed", seed)
            pool_frames.append(pool_table)

            for protocol, candidate, split_col in (
                (
                    "size_matched_scaffold",
                    size_candidate,
                    "split_size_matched_parent_v3",
                ),
                (
                    "target_balanced_scaffold",
                    balanced_candidate,
                    "split_target_balanced_parent_v3",
                ),
            ):
                split_df, meta = materialize_candidate_split(
                    base, groups, candidate, split_col=split_col
                )
                meta.update({**generation_meta, **pair_meta, "protocol": protocol})
                class_gate(split_df, split_col, dataset, seed, protocol)
                manifests.extend(
                    split_manifest_rows(
                        split_df,
                        dataset=dataset,
                        protocol=protocol,
                        partition_seed=seed,
                        split_col=split_col,
                        partition_hash_value=meta["partition_hash"],
                    )
                )
                audits.append(
                    summarize_partition(
                        split_df,
                        dataset=dataset,
                        task_type="classification",
                        protocol=protocol,
                        partition_seed=seed,
                        split_col=split_col,
                        meta=meta,
                    )
                )
                if protocol == "size_matched_scaffold":
                    size_hash = meta["partition_hash"]
                    size_n = int(split_df[split_col].eq("test").sum())
                else:
                    balanced_hash = meta["partition_hash"]
                    balanced_n = int(split_df[split_col].eq("test").sum())

            if size_n != balanced_n:
                raise AssertionError(f"Paired test size mismatch: {dataset}/seed={seed}")
            pair_rows.append(
                {
                    "dataset": dataset,
                    "partition_seed": int(seed),
                    "n_test": int(size_n),
                    "size_partition_hash": size_hash,
                    "balanced_partition_hash": balanced_hash,
                    **generation_meta,
                    **pair_meta,
                }
            )

    manifest = pd.DataFrame(manifests)
    audit = pd.DataFrame(audits)
    pairs = pd.DataFrame(pair_rows)
    pools = pd.concat(pool_frames, ignore_index=True)

    for dataset in DATASETS:
        for protocol in PROTOCOLS:
            hashes = audit.loc[
                audit["dataset"].eq(dataset) & audit["protocol"].eq(protocol),
                "partition_hash",
            ]
            if len(hashes) != 20 or hashes.nunique() != 20:
                raise AssertionError(
                    f"Expected 20/20 unique partitions: {dataset}/{protocol}; rows={len(hashes)}, unique={hashes.nunique()}"
                )

    manifest_path = SPLIT_DIR / "parent_fragment_split_manifest_v3.csv"
    audit_path = SPLIT_DIR / "parent_fragment_split_audit_v3.csv"
    pair_path = SPLIT_DIR / "parent_fragment_split_pairs_v3.csv"
    pool_path = SPLIT_DIR / "parent_fragment_candidate_pool_v3.csv"
    manifest.to_csv(manifest_path, index=False)
    audit.to_csv(audit_path, index=False)
    pairs.to_csv(pair_path, index=False)
    pools.to_csv(pool_path, index=False)
    return manifest, audit, pairs, pools


def freeze_artifacts(cleaning_summary: pd.DataFrame) -> dict:
    manifest_path = SPLIT_DIR / "parent_fragment_split_manifest_v3.csv"
    audit_path = SPLIT_DIR / "parent_fragment_split_audit_v3.csv"
    pair_path = SPLIT_DIR / "parent_fragment_split_pairs_v3.csv"
    pool_path = SPLIT_DIR / "parent_fragment_candidate_pool_v3.csv"
    registry_rows: list[dict] = []
    for path in (manifest_path, audit_path, pair_path, pool_path, PROTOCOL_FILE):
        registry_rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    for dataset in DATASETS:
        path = PARENT_DATA_DIR / f"{dataset.lower()}_parent_sensitivity_v3.csv"
        registry_rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    registry = pd.DataFrame(registry_rows)
    registry_path = FROZEN_DIR / "parent_fragment_artifact_registry_v3.csv"
    registry.to_csv(registry_path, index=False)
    metadata = {
        "analysis_role": "classification_parent_fragment_sensitivity",
        "git_branch": git_text("branch", "--show-current"),
        "git_commit": git_text("rev-parse", "HEAD"),
        "candidate_budget": N_CANDIDATES,
        "partition_seeds": list(SEEDS),
        "datasets": list(DATASETS),
        "protocols": list(PROTOCOLS),
        "models": list(MODELS),
        "expected_model_jobs": EXPECTED_JOBS,
        "protocol_sha256": sha256_file(PROTOCOL_FILE),
        "manifest_sha256": sha256_file(manifest_path),
        "pairs_sha256": sha256_file(pair_path),
        "artifact_registry_sha256": sha256_file(registry_path),
        "cleaning_summary": cleaning_summary.to_dict(orient="records"),
    }
    metadata_path = FROZEN_DIR / "parent_fragment_frozen_metadata_v3.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print("\nPARENT-FRAGMENT SENSITIVITY ARTIFACTS FROZEN")
    print(metadata_path)
    return metadata


def job_path(dataset: str, protocol: str, seed: int, partition_hash: str, model: str, model_seed: int) -> Path:
    return (
        JOB_DIR
        / dataset
        / protocol
        / f"p{seed}_{partition_hash[:12]}"
        / model
        / f"m{model_seed}.json"
    )


def fit_models(parent_frames: dict[str, pd.DataFrame], manifest: pd.DataFrame, frozen_meta: dict) -> None:
    manifest = manifest.copy()
    manifest["partition_seed"] = pd.to_numeric(manifest["partition_seed"], errors="raise").astype(int)
    protocol_sha = frozen_meta["protocol_sha256"]
    manifest_sha = frozen_meta["manifest_sha256"]
    completed = 0
    skipped = 0

    for dataset in DATASETS:
        print(f"\n========== FIT PARENT SENSITIVITY: {dataset} ==========")
        clean = parent_frames[dataset].reset_index(drop=True)
        smiles = clean["canonical_smiles"].astype(str).tolist()
        targets = clean["target"].to_numpy(dtype=float)
        smiles_to_index = {value: idx for idx, value in enumerate(smiles)}
        X, cache_meta = load_or_build_morgan_matrix(
            smiles, cache_dir=CACHE_DIR, dataset=f"parent_{dataset}"
        )
        print(f"fingerprints={X.shape}, nnz={X.nnz}, cache={cache_meta['cache_status']}")
        dataset_manifest = manifest.loc[manifest["dataset"].eq(dataset)].copy()

        for protocol in PROTOCOLS:
            protocol_manifest = dataset_manifest.loc[dataset_manifest["protocol"].eq(protocol)]
            for (seed, partition_hash), group in protocol_manifest.groupby(
                ["partition_seed", "partition_hash"], sort=False
            ):
                seed = int(seed)
                train_smiles = group.loc[group["assignment"].eq("train"), "canonical_smiles"].astype(str)
                test_smiles = group.loc[group["assignment"].eq("test"), "canonical_smiles"].astype(str)
                train_idx = np.asarray([smiles_to_index[value] for value in train_smiles], dtype=int)
                test_idx = np.asarray([smiles_to_index[value] for value in test_smiles], dtype=int)
                if len(train_idx) + len(test_idx) != len(clean):
                    raise AssertionError(f"Incomplete coverage: {dataset}/{protocol}/{seed}")
                if np.intersect1d(train_idx, test_idx).size:
                    raise AssertionError(f"Train/test overlap: {dataset}/{protocol}/{seed}")
                y_train = targets[train_idx]
                y_test = targets[test_idx]
                if set(np.unique(y_train).tolist()) != {0.0, 1.0} or set(np.unique(y_test).tolist()) != {0.0, 1.0}:
                    raise AssertionError(f"Lost classification class: {dataset}/{protocol}/{seed}")

                for model_name in MODELS:
                    model_seed = production_model_seed(model_name)
                    path = job_path(dataset, protocol, seed, str(partition_hash), model_name, model_seed)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        existing = json.loads(path.read_text(encoding="utf-8"))
                        if (
                            existing.get("partition_hash") == str(partition_hash)
                            and existing.get("protocol_sha256") == protocol_sha
                            and existing.get("manifest_sha256") == manifest_sha
                            and int(existing.get("model_seed")) == int(model_seed)
                        ):
                            skipped += 1
                            print(f"SKIP {dataset} {protocol} p={seed} {model_name} m={model_seed}")
                            continue
                        raise AssertionError(f"Stale parent sensitivity job found: {path}")

                    model = build_model(
                        task_type="classification",
                        model_name=model_name,
                        model_seed=model_seed,
                        y_train=y_train,
                    )
                    start = time.perf_counter()
                    model.fit(X[train_idx], y_train)
                    fit_seconds = time.perf_counter() - start
                    probabilities = np.asarray(model.predict_proba(X[test_idx])[:, 1], dtype=float)
                    metrics = classification_metrics(y_test, probabilities)
                    payload = {
                        "analysis_role": "classification_parent_fragment_sensitivity",
                        "dataset": dataset,
                        "protocol": protocol,
                        "partition_seed": seed,
                        "partition_hash": str(partition_hash),
                        "model": model_name,
                        "model_seed": int(model_seed),
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "fit_seconds": float(fit_seconds),
                        "protocol_sha256": protocol_sha,
                        "manifest_sha256": manifest_sha,
                        **metrics,
                    }
                    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                    completed += 1
                    print(
                        f"DONE {dataset} {protocol} p={seed} {model_name} m={model_seed} "
                        f"ROC-AUC={payload['roc_auc']:.6f}"
                    )
    print(f"\nParent sensitivity model run: completed={completed}, skipped={skipped}, expected={EXPECTED_JOBS}")


def collect_jobs(frozen_meta: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for path in JOB_DIR.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("analysis_role") != "classification_parent_fragment_sensitivity":
            continue
        if payload.get("protocol_sha256") != frozen_meta["protocol_sha256"]:
            continue
        if payload.get("manifest_sha256") != frozen_meta["manifest_sha256"]:
            continue
        rows.append(payload)
    jobs = pd.DataFrame(rows)
    if len(jobs) != EXPECTED_JOBS:
        raise AssertionError(f"Expected {EXPECTED_JOBS} current sensitivity jobs; found {len(jobs)}")
    key = ["dataset", "protocol", "partition_seed", "model"]
    if jobs.duplicated(key).any():
        raise AssertionError("Duplicate current parent sensitivity jobs")
    if not np.isfinite(pd.to_numeric(jobs["roc_auc"], errors="raise").to_numpy(float)).all():
        raise AssertionError("Non-finite ROC-AUC in parent sensitivity jobs")
    jobs_path = OUT_DIR / "parent_fragment_model_job_index_v3.csv"
    jobs.to_csv(jobs_path, index=False)
    print(f"\nPARENT SENSITIVITY MODEL COMPLETENESS PASSED: {len(jobs)}/{EXPECTED_JOBS}")
    return jobs


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(BOOTSTRAP_REPS, len(x)))
    means = x[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def signed_rank(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if np.allclose(x, 0.0, rtol=0.0, atol=1e-15):
        return 0.0, 1.0
    result = wilcoxon(
        x,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy(dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        adjusted[idx] = min(running, 1.0)
    return pd.Series(adjusted, index=pvalues.index)


def inference_label(mean_effect: float, ci_low: float, ci_high: float, p_holm: float) -> str:
    if p_holm < 0.05 and ci_low > 0:
        return "target_balanced_better"
    if p_holm < 0.05 and ci_high < 0:
        return "target_balanced_worse"
    return "inconclusive"


def analyze(jobs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    partition_rows: list[dict] = []
    summary_rows: list[dict] = []
    cell_index = 0
    jobs = jobs.copy()
    jobs["partition_seed"] = pd.to_numeric(jobs["partition_seed"], errors="raise").astype(int)

    for dataset in DATASETS:
        for model in MODELS:
            group = jobs.loc[jobs["dataset"].eq(dataset) & jobs["model"].eq(model)].copy()
            size = group.loc[group["protocol"].eq("size_matched_scaffold")].copy()
            balanced = group.loc[group["protocol"].eq("target_balanced_scaffold")].copy()
            pair = size.merge(
                balanced,
                on=["dataset", "model", "partition_seed"],
                suffixes=("_size", "_balanced"),
                validate="one_to_one",
            )
            if len(pair) != 20:
                raise AssertionError(f"Expected 20 parent sensitivity pairs: {dataset}/{model}; found {len(pair)}")
            s = pd.to_numeric(pair["roc_auc_size"], errors="raise").to_numpy(float)
            b = pd.to_numeric(pair["roc_auc_balanced"], errors="raise").to_numpy(float)
            effects = b - s
            for row, sv, bv, effect in zip(pair.itertuples(index=False), s, b, effects):
                partition_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "partition_seed": int(row.partition_seed),
                        "size_partition_hash": row.partition_hash_size,
                        "balanced_partition_hash": row.partition_hash_balanced,
                        "size_roc_auc": float(sv),
                        "balanced_roc_auc": float(bv),
                        "effect_positive_is_balanced_better": float(effect),
                    }
                )
            ci_low, ci_high = bootstrap_ci(effects, BOOTSTRAP_BASE_SEED + cell_index)
            statistic, p_raw = signed_rank(effects)
            summary_rows.append(
                {
                    "analysis_role": "parent_fragment_classification_sensitivity",
                    "dataset": dataset,
                    "model": model,
                    "n_unique_partition_pairs": 20,
                    "mean_size_roc_auc": float(np.mean(s)),
                    "mean_balanced_roc_auc": float(np.mean(b)),
                    "mean_effect": float(np.mean(effects)),
                    "median_effect": float(np.median(effects)),
                    "sd_effect": float(np.std(effects, ddof=1)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "n_balanced_better": int(np.sum(effects > 0)),
                    "n_equal": int(np.sum(np.isclose(effects, 0.0, atol=1e-15, rtol=0.0))),
                    "n_balanced_worse": int(np.sum(effects < 0)),
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_raw,
                }
            )
            cell_index += 1

    partition = pd.DataFrame(partition_rows)
    summary = pd.DataFrame(summary_rows)
    if len(summary) != 9:
        raise AssertionError(f"Expected 9 parent sensitivity cells; found {len(summary)}")
    summary["p_holm_sensitivity_family"] = holm_adjust(summary["p_raw"])
    summary["inference_label"] = [
        inference_label(m, lo, hi, p)
        for m, lo, hi, p in zip(
            summary["mean_effect"],
            summary["bootstrap_ci_low"],
            summary["bootstrap_ci_high"],
            summary["p_holm_sensitivity_family"],
        )
    ]

    comparison = summary.copy()
    main_path = PAPER_DIR / "results" / "tables" / "primary_inference_summary_v3.csv"
    if main_path.exists():
        main = pd.read_csv(main_path, keep_default_na=False)
        main = main.loc[
            main["dataset"].isin(DATASETS) & main["model"].isin(MODELS),
            ["dataset", "model", "mean_effect", "bootstrap_ci_low", "bootstrap_ci_high", "p_holm", "inference_label"],
        ].copy()
        main = main.rename(
            columns={
                "mean_effect": "main_mean_effect",
                "bootstrap_ci_low": "main_ci_low",
                "bootstrap_ci_high": "main_ci_high",
                "p_holm": "main_p_holm_18cell",
                "inference_label": "main_inference_label",
            }
        )
        comparison = main.merge(
            summary.rename(
                columns={
                    "mean_effect": "parent_mean_effect",
                    "bootstrap_ci_low": "parent_ci_low",
                    "bootstrap_ci_high": "parent_ci_high",
                    "p_holm_sensitivity_family": "parent_p_holm_9cell",
                    "inference_label": "parent_inference_label",
                }
            ),
            on=["dataset", "model"],
            how="right",
            validate="one_to_one",
        )
        comparison["effect_sign_agrees"] = np.sign(comparison["main_mean_effect"].astype(float)) == np.sign(comparison["parent_mean_effect"].astype(float))
        comparison["inference_label_agrees"] = comparison["main_inference_label"].astype(str) == comparison["parent_inference_label"].astype(str)

    partition.to_csv(TABLE_DIR / "parent_fragment_partition_effects_v3.csv", index=False)
    summary.to_csv(TABLE_DIR / "parent_fragment_inference_summary_v3.csv", index=False)
    comparison.to_csv(TABLE_DIR / "parent_fragment_vs_main_comparison_v3.csv", index=False)
    return partition, summary, comparison


def main() -> None:
    if not PROTOCOL_FILE.exists():
        raise FileNotFoundError(PROTOCOL_FILE)
    print("PARENT-FRAGMENT CLASSIFICATION SENSITIVITY V3")
    print(f"Protocol SHA256: {sha256_file(PROTOCOL_FILE)}")

    parent_frames: dict[str, pd.DataFrame] = {}
    cleaning_summaries: list[dict] = []
    for dataset in DATASETS:
        clean, summary, _ = build_parent_dataset(dataset)
        parent_frames[dataset] = clean
        cleaning_summaries.append(summary)
    cleaning_summary = pd.DataFrame(cleaning_summaries)
    cleaning_summary.to_csv(OUT_DIR / "parent_fragment_cleaning_summary_v3.csv", index=False)
    print("\nParent-fragment cleaning summary:")
    print(cleaning_summary.to_string(index=False))

    manifest, audit, pairs, _ = build_splits(parent_frames)
    print("\nParent split pair summary:")
    print(
        pairs.groupby("dataset", as_index=False).agg(
            n_pairs=("partition_seed", "size"),
            min_n_test=("n_test", "min"),
            max_n_test=("n_test", "max"),
            min_same_size_candidates=("n_same_size_candidates", "min"),
            mean_size_target_gap=("size_matched_target_gap", "mean"),
            mean_balanced_target_gap=("target_balanced_target_gap", "mean"),
        ).to_string(index=False)
    )
    print("\nParent split uniqueness:")
    print(
        audit.groupby(["dataset", "protocol"], as_index=False).agg(
            n_requested=("partition_hash", "size"),
            n_unique=("partition_hash", "nunique"),
            min_n_test=("n_test", "min"),
            max_n_test=("n_test", "max"),
        ).to_string(index=False)
    )

    frozen_meta = freeze_artifacts(cleaning_summary)
    fit_models(parent_frames, manifest, frozen_meta)
    jobs = collect_jobs(frozen_meta)
    _, summary, comparison = analyze(jobs)

    print("\nParent-fragment 9-cell inference summary:")
    print(
        summary[
            [
                "dataset", "model", "mean_size_roc_auc", "mean_balanced_roc_auc",
                "mean_effect", "bootstrap_ci_low", "bootstrap_ci_high",
                "p_raw", "p_holm_sensitivity_family", "inference_label",
            ]
        ].to_string(index=False)
    )
    print("\nInference counts:")
    print(summary["inference_label"].value_counts().to_string())
    if "main_inference_label" in comparison.columns:
        print("\nParent-fragment vs main-analysis comparison:")
        print(
            comparison[
                [
                    "dataset", "model", "main_mean_effect", "parent_mean_effect",
                    "main_inference_label", "parent_inference_label",
                    "effect_sign_agrees", "inference_label_agrees",
                ]
            ].to_string(index=False)
        )

    print("\nSaved under:")
    print(OUT_DIR)
    print("\nPARENT-FRAGMENT CLASSIFICATION SENSITIVITY V3 COMPLETED")


if __name__ == "__main__":
    main()
