from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
DATA_DIR = PAPER_DIR / "data" / "processed_v2"
SOURCE_DIR = PAPER_DIR / "results" / "split_rebuild_v3"
FREEZE_DIR = PAPER_DIR / "results" / "frozen_v3"
FREEZE_DIR.mkdir(parents=True, exist_ok=True)

N_PARTITION_SEEDS = 20


@dataclass(frozen=True)
class FreezeSpec:
    label: str
    datasets: tuple[str, ...]
    acyclic_mode: str
    candidate_budget: int
    suffix: str


SPECS = (
    FreezeSpec(
        label="main_classification",
        datasets=("BACE", "BBBP", "ClinTox", "HIV"),
        acyclic_mode="single_group",
        candidate_budget=300,
        suffix="BACE-BBBP-ClinTox-HIV_single_group_20s_300c",
    ),
    FreezeSpec(
        label="main_regression",
        datasets=("ESOL", "FreeSolv"),
        acyclic_mode="single_group",
        candidate_budget=20000,
        suffix="ESOL-FreeSolv_single_group_20s_20000c",
    ),
    FreezeSpec(
        label="acyclic_singleton_sensitivity",
        datasets=("ESOL", "FreeSolv"),
        acyclic_mode="singleton",
        candidate_budget=5000,
        suffix="ESOL-FreeSolv_singleton_20s_5000c",
    ),
)

FILE_PREFIXES = {
    "manifest": "split_manifest_v3",
    "audit": "split_audit_v3",
    "candidate_pool": "candidate_pool_v3",
    "pairs": "split_pairs_v3",
    "pair_summary": "split_pair_summary_v3",
    "uniqueness": "split_uniqueness_v3",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def normalize_seed(value: object) -> str:
    """Return a stable seed key across CSV integer/float/string inference."""
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


def normalize_hash(value: object) -> str:
    return str(value).strip().lower()


def manifest_partition_hash(group: pd.DataFrame) -> str:
    """Recompute the production partition hash from manifest test rows."""
    test = group.loc[
        group["assignment"].astype(str).str.strip().eq("test"),
        ["row_index", "canonical_smiles"],
    ].copy()
    test["row_index_numeric"] = pd.to_numeric(test["row_index"], errors="raise").astype(int)
    test = test.sort_values("row_index_numeric", kind="mergesort")
    payload = "\n".join(
        f"{int(row.row_index_numeric)}\t{str(row.canonical_smiles)}"
        for row in test.itertuples(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_paths(spec: FreezeSpec) -> dict[str, Path]:
    return {
        key: SOURCE_DIR / f"{prefix}_{spec.suffix}.csv"
        for key, prefix in FILE_PREFIXES.items()
    }


def read_csv_as_text(path: Path) -> pd.DataFrame:
    """Read audit artifacts without lossy mixed-type inference."""
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def verify_spec(spec: FreezeSpec, paths: dict[str, Path]) -> dict:
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing production artifact for {spec.label}: {path}"
            )

    manifest = read_csv_as_text(paths["manifest"])
    pairs = read_csv_as_text(paths["pairs"])
    uniqueness = read_csv_as_text(paths["uniqueness"])

    required_manifest = {
        "dataset",
        "protocol",
        "partition_seed",
        "partition_hash",
        "row_index",
        "canonical_smiles",
        "assignment",
    }
    missing_manifest = required_manifest.difference(manifest.columns)
    if missing_manifest:
        raise KeyError(
            f"{paths['manifest']} missing columns: {sorted(missing_manifest)}"
        )

    required_pairs = {
        "dataset",
        "partition_seed",
        "requested_candidates",
        "exact_size_match",
        "size_partition_hash",
        "balanced_partition_hash",
    }
    missing_pairs = required_pairs.difference(pairs.columns)
    if missing_pairs:
        raise KeyError(f"{paths['pairs']} missing columns: {sorted(missing_pairs)}")

    manifest["_seed_key"] = manifest["partition_seed"].map(normalize_seed)
    manifest["_hash_key"] = manifest["partition_hash"].map(normalize_hash)
    pairs["_seed_key"] = pairs["partition_seed"].map(normalize_seed)
    pairs["_size_hash_key"] = pairs["size_partition_hash"].map(normalize_hash)
    pairs["_balanced_hash_key"] = pairs["balanced_partition_hash"].map(normalize_hash)

    if set(manifest["dataset"]) != set(spec.datasets):
        raise AssertionError(
            f"Manifest dataset mismatch for {spec.label}: "
            f"{sorted(set(manifest['dataset']))}"
        )
    if set(pairs["dataset"]) != set(spec.datasets):
        raise AssertionError(f"Pair dataset mismatch for {spec.label}")
    if len(pairs) != len(spec.datasets) * N_PARTITION_SEEDS:
        raise AssertionError(
            f"Unexpected pair count for {spec.label}: {len(pairs)}"
        )
    if pairs[["dataset", "_seed_key"]].duplicated().any():
        duplicates = pairs.loc[
            pairs[["dataset", "_seed_key"]].duplicated(keep=False),
            ["dataset", "partition_seed"],
        ]
        raise AssertionError(
            "Duplicate dataset/partition_seed rows in pairs table:\n"
            + duplicates.to_string(index=False)
        )
    if not as_bool(pairs["exact_size_match"]).all():
        raise AssertionError(f"Non-exact paired test sizes remain in {spec.label}")
    if not pd.to_numeric(pairs["requested_candidates"], errors="raise").astype(int).eq(
        spec.candidate_budget
    ).all():
        raise AssertionError(f"Candidate-budget mismatch for {spec.label}")

    expected_protocols = {
        "legacy_scaffold",
        "random_observation",
        "size_matched_scaffold",
        "target_balanced_scaffold",
    }
    if set(manifest["protocol"]) != expected_protocols:
        raise AssertionError(
            f"Protocol mismatch for {spec.label}: {sorted(set(manifest['protocol']))}"
        )
    if not set(manifest["assignment"]).issubset({"train", "test"}):
        raise AssertionError(f"Unexpected assignment value in {spec.label}")

    unique_partition_rows: list[dict] = []
    for dataset in spec.datasets:
        clean_path = DATA_DIR / f"{dataset.lower()}_clean_v2.csv"
        if not clean_path.exists():
            raise FileNotFoundError(f"Missing clean_v2 file: {clean_path}")
        clean = pd.read_csv(
            clean_path,
            dtype={"canonical_smiles": str},
            keep_default_na=False,
            low_memory=False,
        )
        clean_smiles = set(clean["canonical_smiles"].astype(str))
        n_total = len(clean)
        dataset_manifest = manifest.loc[manifest["dataset"].eq(dataset)].copy()

        protocol_expected_counts = {
            "legacy_scaffold": 1,
            "random_observation": N_PARTITION_SEEDS,
            "size_matched_scaffold": N_PARTITION_SEEDS,
            "target_balanced_scaffold": N_PARTITION_SEEDS,
        }
        for protocol, expected_count in protocol_expected_counts.items():
            subset = dataset_manifest.loc[dataset_manifest["protocol"].eq(protocol)]
            actual_count = subset[["_seed_key", "_hash_key"]].drop_duplicates().shape[0]
            if actual_count != expected_count:
                raise AssertionError(
                    f"{dataset}/{protocol} has {actual_count} partitions; "
                    f"expected {expected_count}"
                )

        grouped = dataset_manifest.groupby(
            ["protocol", "_seed_key", "_hash_key"],
            dropna=False,
            sort=False,
        )
        for (protocol, seed, partition_hash), group in grouped:
            if len(group) != n_total:
                raise AssertionError(
                    f"{dataset}/{protocol}/{seed} has {len(group)} rows; "
                    f"expected {n_total}"
                )
            row_indices = pd.to_numeric(group["row_index"], errors="raise").astype(int)
            if row_indices.nunique() != n_total:
                raise AssertionError(
                    f"Duplicate or missing row_index in {dataset}/{protocol}/{seed}"
                )
            if set(group["canonical_smiles"].astype(str)) != clean_smiles:
                raise AssertionError(
                    f"Molecule universe mismatch in {dataset}/{protocol}/{seed}"
                )
            if not {"train", "test"}.issubset(set(group["assignment"])):
                raise AssertionError(
                    f"Empty train or test assignment in {dataset}/{protocol}/{seed}"
                )

            recomputed_hash = manifest_partition_hash(group)
            if recomputed_hash != partition_hash:
                raise AssertionError(
                    f"Manifest hash recomputation mismatch in "
                    f"{dataset}/{protocol}/seed={seed}: "
                    f"stored={partition_hash}, recomputed={recomputed_hash}"
                )

            unique_partition_rows.append(
                {
                    "freeze_label": spec.label,
                    "dataset": dataset,
                    "protocol": protocol,
                    "partition_seed": seed,
                    "partition_hash": partition_hash,
                    "n_total": n_total,
                    "n_train": int(group["assignment"].eq("train").sum()),
                    "n_test": int(group["assignment"].eq("test").sum()),
                }
            )

        for _, pair in pairs.loc[pairs["dataset"].eq(dataset)].iterrows():
            seed = pair["_seed_key"]
            size_hashes = dataset_manifest.loc[
                dataset_manifest["protocol"].eq("size_matched_scaffold")
                & dataset_manifest["_seed_key"].eq(seed),
                "_hash_key",
            ].unique()
            balanced_hashes = dataset_manifest.loc[
                dataset_manifest["protocol"].eq("target_balanced_scaffold")
                & dataset_manifest["_seed_key"].eq(seed),
                "_hash_key",
            ].unique()
            if len(size_hashes) != 1 or size_hashes[0] != pair["_size_hash_key"]:
                raise AssertionError(
                    f"Size hash mismatch in {dataset}, seed={seed}: "
                    f"manifest={size_hashes.tolist()}, pair={pair['_size_hash_key']}"
                )
            if (
                len(balanced_hashes) != 1
                or balanced_hashes[0] != pair["_balanced_hash_key"]
            ):
                raise AssertionError(
                    f"Balanced hash mismatch in {dataset}, seed={seed}: "
                    f"manifest={balanced_hashes.tolist()}, "
                    f"pair={pair['_balanced_hash_key']}"
                )

    if set(uniqueness["dataset"]) != set(spec.datasets):
        raise AssertionError(f"Uniqueness table dataset mismatch for {spec.label}")

    partition_table = pd.DataFrame(unique_partition_rows)
    return {
        "partition_table": partition_table,
        "n_manifest_rows": int(len(manifest)),
        "n_pairs": int(len(pairs)),
        "n_unique_size_partitions": int(pairs["_size_hash_key"].nunique()),
        "n_unique_balanced_partitions": int(pairs["_balanced_hash_key"].nunique()),
    }


def main() -> None:
    registry_rows: list[dict] = []
    partition_tables: list[pd.DataFrame] = []

    for dataset in sorted({dataset for spec in SPECS for dataset in spec.datasets}):
        clean_path = DATA_DIR / f"{dataset.lower()}_clean_v2.csv"
        if not clean_path.exists():
            raise FileNotFoundError(f"Missing clean_v2 file: {clean_path}")
        registry_rows.append(
            {
                "freeze_label": "clean_v2",
                "artifact_type": "clean_dataset",
                "path": str(clean_path.relative_to(ROOT)),
                "sha256": sha256_file(clean_path),
                "bytes": clean_path.stat().st_size,
                "frozen_copy": "",
            }
        )

    verification: dict[str, dict] = {}
    for spec in SPECS:
        paths = source_paths(spec)
        result = verify_spec(spec, paths)
        partition_tables.append(result.pop("partition_table"))
        verification[spec.label] = result

        destination_dir = FREEZE_DIR / spec.label
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        for artifact_type, source_path in paths.items():
            frozen_copy = ""
            if artifact_type != "candidate_pool":
                destination = destination_dir / source_path.name
                shutil.copy2(source_path, destination)
                frozen_copy = str(destination.relative_to(ROOT))
            registry_rows.append(
                {
                    "freeze_label": spec.label,
                    "artifact_type": artifact_type,
                    "path": str(source_path.relative_to(ROOT)),
                    "sha256": sha256_file(source_path),
                    "bytes": source_path.stat().st_size,
                    "frozen_copy": frozen_copy,
                }
            )

    protocol_path = PAPER_DIR / "REBUILD_PROTOCOL_V3.md"
    registry_rows.append(
        {
            "freeze_label": "protocol",
            "artifact_type": "rebuild_protocol",
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": sha256_file(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "frozen_copy": "",
        }
    )

    registry = pd.DataFrame(registry_rows)
    partitions = pd.concat(partition_tables, ignore_index=True)
    registry_path = FREEZE_DIR / "frozen_artifact_registry_v3.csv"
    partition_path = FREEZE_DIR / "frozen_partition_registry_v3.csv"
    metadata_path = FREEZE_DIR / "frozen_metadata_v3.json"
    registry.to_csv(registry_path, index=False)
    partitions.to_csv(partition_path, index=False)

    metadata = {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "protocol": "paper1_leakage_benchmark/REBUILD_PROTOCOL_V3.md",
        "n_partition_seeds": N_PARTITION_SEEDS,
        "production_specs": [
            {
                "label": spec.label,
                "datasets": list(spec.datasets),
                "acyclic_mode": spec.acyclic_mode,
                "candidate_budget": spec.candidate_budget,
                "suffix": spec.suffix,
            }
            for spec in SPECS
        ],
        "verification": verification,
        "registry_sha256": sha256_file(registry_path),
        "partition_registry_sha256": sha256_file(partition_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\nFrozen artifact registry:")
    print(registry.to_string(index=False))
    print("\nFrozen partition summary:")
    print(
        partitions.groupby(["freeze_label", "dataset", "protocol"], as_index=False)
        .agg(
            n_requested_partitions=("partition_hash", "size"),
            n_unique_partitions=("partition_hash", "nunique"),
            min_n_test=("n_test", "min"),
            max_n_test=("n_test", "max"),
        )
        .to_string(index=False)
    )
    print("\nSaved:")
    print(registry_path)
    print(partition_path)
    print(metadata_path)
    print("\nPRODUCTION MANIFEST FREEZE V3 PASSED")


if __name__ == "__main__":
    main()
