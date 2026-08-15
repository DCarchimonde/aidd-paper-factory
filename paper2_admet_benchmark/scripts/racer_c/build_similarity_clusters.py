from __future__ import annotations

"""Build one deterministic, label-blind ECFP leader partition per endpoint."""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from rdkit import DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
DEFAULT_PROCESSED = P2 / "data" / "processed" / "racer_c"
DEFAULT_MANIFESTS = P2 / "data" / "manifests" / "racer_c"
SIMILARITY_THRESHOLD = 0.60
ORDER_SEED = 1701
MAX_ENDPOINT_N = 15_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_order_key(structure_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{ORDER_SEED}|{structure_id}".encode("utf-8")).hexdigest()
    return digest, structure_id


def build_cluster_map(rows: list[dict[str, str]]) -> tuple[dict[str, str], Counter[str]]:
    if not rows:
        raise ValueError("clean endpoint is empty")
    if len(rows) > MAX_ENDPOINT_N:
        raise ValueError(f"n={len(rows)} exceeds frozen max_endpoint_n={MAX_ENDPOINT_N}")
    ordered = sorted(rows, key=lambda row: stable_order_key(row["structure_id"]))
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    leader_fps = []
    leader_ids: list[str] = []
    cluster_map: dict[str, str] = {}
    cluster_sizes: Counter[str] = Counter()
    for index, row in enumerate(ordered, start=1):
        mol = Chem.MolFromSmiles(row["standardized_smiles"], sanitize=True)
        if mol is None:
            raise ValueError(f"clean standardized SMILES no longer parses: {row['structure_id']}")
        fp = generator.GetFingerprint(mol)
        if not leader_fps:
            leader_fps.append(fp)
            leader_ids.append(row["structure_id"])
            cluster_id = row["structure_id"]
        else:
            similarities = DataStructs.BulkTanimotoSimilarity(fp, leader_fps)
            best_index = max(range(len(similarities)), key=lambda i: similarities[i])
            if similarities[best_index] >= SIMILARITY_THRESHOLD:
                cluster_id = leader_ids[best_index]
            else:
                leader_fps.append(fp)
                leader_ids.append(row["structure_id"])
                cluster_id = row["structure_id"]
        cluster_map[row["structure_id"]] = f"LEADER:{cluster_id}"
        cluster_sizes[f"LEADER:{cluster_id}"] += 1
        if index % 1000 == 0:
            print(f"clustered {index}/{len(ordered)}; leaders={len(leader_ids)}", flush=True)
    return cluster_map, cluster_sizes


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", required=True, help="comma-separated endpoint names")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFESTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if rdBase.rdkitVersion != "2026.03.4":
        raise RuntimeError(f"clustering requires RDKit 2026.03.4, got {rdBase.rdkitVersion}")
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    for endpoint in [value.strip() for value in args.endpoints.split(",") if value.strip()]:
        clean_path = args.processed_dir / f"{endpoint}_clean.csv"
        role_path = args.processed_dir / "role_inputs" / f"{endpoint}_role_input.csv"
        cleaning_manifest_path = args.manifest_dir / f"{endpoint}_cleaning.json"
        clean_rows = read_csv(clean_path)
        role_rows = read_csv(role_path)
        t0 = time.monotonic()
        cluster_map, cluster_sizes = build_cluster_map(clean_rows)
        for row in role_rows:
            row["similarity_cluster_id"] = cluster_map[row["structure_id"]]
        write_csv(role_path, role_rows)
        sizes = sorted(cluster_sizes.values())
        manifest = {
            "endpoint": endpoint,
            "algorithm": "deterministic_ordered_leader_clustering",
            "assignment_uses_labels": False,
            "fingerprint": "Morgan_radius2_2048bits_no_chirality",
            "similarity": "Tanimoto",
            "join_threshold_inclusive": SIMILARITY_THRESHOLD,
            "order_seed": ORDER_SEED,
            "order_key": "sha256(order_seed|structure_id)",
            "tie_rule": "earliest_leader_in_deterministic_order",
            "max_endpoint_n": MAX_ENDPOINT_N,
            "n_structures": len(role_rows),
            "n_clusters": len(cluster_sizes),
            "singleton_clusters": sum(size == 1 for size in sizes),
            "max_cluster_size": max(sizes),
            "median_cluster_size": sizes[len(sizes) // 2],
            "elapsed_seconds_development_audit": round(time.monotonic() - t0, 6),
            "rdkit_version": rdBase.rdkitVersion,
            "cleaned_byte_sha256": sha256_file(clean_path),
            "role_input_with_clusters_byte_sha256": sha256_file(role_path),
            "clustering_script_byte_sha256": sha256_file(Path(__file__)),
        }
        (args.manifest_dir / f"{endpoint}_similarity_clusters.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        cleaning = json.loads(cleaning_manifest_path.read_text(encoding="utf-8"))
        cleaning["role_input_byte_sha256"] = sha256_file(role_path)
        cleaning["similarity_cluster_status"] = "complete"
        cleaning["similarity_cluster_manifest"] = str(
            (args.manifest_dir / f"{endpoint}_similarity_clusters.json").relative_to(ROOT)
        )
        cleaning_manifest_path.write_text(
            json.dumps(cleaning, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
