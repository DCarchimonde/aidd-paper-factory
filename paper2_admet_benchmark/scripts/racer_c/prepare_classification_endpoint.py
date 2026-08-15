from __future__ import annotations

"""Create audited RACER-C classification cleaning and role-input artefacts.

This pre-model utility is intentionally strict.  It accepts only endpoints whose
provenance manifest explicitly permits analysis, verifies the raw byte hash, logs
every removed source row, and never resolves conflicting labels by majority vote.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
PROVENANCE_PATH = P2 / "protocols" / "data_provenance_license_manifest.csv"
CONTRACT_PATH = P2 / "protocols" / "chemical_standardization_contract.yaml"
DEFAULT_PROCESSED = P2 / "data" / "processed" / "racer_c"
DEFAULT_MANIFESTS = P2 / "data" / "manifests" / "racer_c"
STANDARDIZATION_ID = "racer_c_rdkit_2026_03_4_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_provenance(endpoint: str) -> dict[str, str]:
    with PROVENANCE_PATH.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["endpoint"] == endpoint]
    if len(rows) != 1:
        raise ValueError(f"expected one provenance row for {endpoint}, got {len(rows)}")
    row = rows[0]
    if row["analysis_use_status"] != "allowed_with_attribution":
        raise PermissionError(
            f"{endpoint} is not approved for extension analysis: {row['analysis_use_status']}"
        )
    if row["raw_sha256_expected"] == "pending":
        raise ValueError(f"populate and review {endpoint} raw_sha256_expected before cleaning")
    return row


def standardize_smiles(raw_smiles: str) -> tuple[str, str, str]:
    mol = Chem.MolFromSmiles(raw_smiles, sanitize=True)
    if mol is None:
        raise ValueError("parse_failure")
    params = rdMolStandardize.CleanupParameters()
    params.preferOrganic = True
    try:
        mol = rdMolStandardize.Cleanup(mol, params)
        mol = rdMolStandardize.FragmentParent(mol, params, skipStandardize=True)
        mol = rdMolStandardize.Uncharger(canonicalOrder=True).uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise ValueError("standardization_failure") from exc
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if not smiles:
        raise ValueError("standardization_failure")
    structure_id = hashlib.sha256(f"{STANDARDIZATION_ID}|{smiles}".encode("utf-8")).hexdigest()
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        # The primary scaffold is explicitly achiral.  Clearing stereochemistry
        # before canonicalization also avoids stale double-bond stereo flags in
        # structures such as TDC Drug_ID 44601848.0 under RDKit 2026.03.4.
        Chem.RemoveStereochemistry(scaffold_mol)
        Chem.SanitizeMol(scaffold_mol)
        scaffold = Chem.MolToSmiles(
            scaffold_mol, canonical=True, isomericSmiles=False
        )
    except Exception as exc:
        raise ValueError("scaffold_failure") from exc
    scaffold_id = scaffold if scaffold else f"ACYCLIC:{structure_id}"
    return smiles, structure_id, scaffold_id


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def clean_classification(
    endpoint: str,
    raw_path: Path,
    source_id_col: str,
    smiles_col: str,
    target_col: str,
) -> tuple[list[dict], list[dict], dict]:
    with raw_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle, delimiter="\t"))
    if not raw_rows:
        raise ValueError("raw dataset is empty")
    required = {source_id_col, smiles_col, target_col}
    missing = required - set(raw_rows[0])
    if missing:
        raise ValueError(f"raw file missing columns: {sorted(missing)}")

    staged: list[dict] = []
    rejections: list[dict] = []
    for raw_row_number, row in enumerate(raw_rows, start=2):
        source_id = str(row.get(source_id_col, "")).strip()
        raw_smiles = str(row.get(smiles_col, "")).strip()
        target_raw = str(row.get(target_col, "")).strip()
        base_rejection = {
            "endpoint": endpoint,
            "source_record_id": source_id,
            "raw_row_number": raw_row_number,
            "raw_smiles": raw_smiles,
            "stage": "source_validation",
            "reason_code": "",
            "detail": "",
            "structure_id_if_available": "",
        }
        if not raw_smiles:
            rejections.append({**base_rejection, "reason_code": "missing_smiles"})
            continue
        if target_raw not in {"0", "1", "0.0", "1.0"}:
            reason = "missing_target" if not target_raw else "invalid_target"
            rejections.append({**base_rejection, "reason_code": reason, "detail": target_raw})
            continue
        try:
            standardized, structure_id, scaffold_id = standardize_smiles(raw_smiles)
        except ValueError as exc:
            rejections.append(
                {
                    **base_rejection,
                    "stage": "standardization",
                    "reason_code": str(exc),
                }
            )
            continue
        staged.append(
            {
                "endpoint": endpoint,
                "source_record_id": source_id,
                "raw_row_number": raw_row_number,
                "raw_smiles": raw_smiles,
                "standardized_smiles": standardized,
                "structure_id": structure_id,
                "target": int(float(target_raw)),
                "murcko_scaffold_id": scaffold_id,
                "standardization_status": "pass",
            }
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in staged:
        grouped[row["structure_id"]].append(row)
    clean: list[dict] = []
    conflicting_groups = 0
    duplicate_rows = 0
    for structure_id, group in grouped.items():
        labels = {row["target"] for row in group}
        if len(labels) > 1:
            conflicting_groups += 1
            for row in group:
                rejections.append(
                    {
                        "endpoint": endpoint,
                        "source_record_id": row["source_record_id"],
                        "raw_row_number": row["raw_row_number"],
                        "raw_smiles": row["raw_smiles"],
                        "stage": "duplicate_resolution",
                        "reason_code": "conflicting_label_excluded",
                        "detail": "all standardized-structure records excluded",
                        "structure_id_if_available": structure_id,
                    }
                )
            continue
        ordered = sorted(group, key=lambda row: (row["source_record_id"], row["raw_row_number"]))
        keep = ordered[0]
        clean.append(keep)
        for row in ordered[1:]:
            duplicate_rows += 1
            rejections.append(
                {
                    "endpoint": endpoint,
                    "source_record_id": row["source_record_id"],
                    "raw_row_number": row["raw_row_number"],
                    "raw_smiles": row["raw_smiles"],
                    "stage": "duplicate_resolution",
                    "reason_code": "duplicate_aggregated",
                    "detail": f"retained_source_record_id={keep['source_record_id']}",
                    "structure_id_if_available": structure_id,
                }
            )

    clean.sort(key=lambda row: (row["structure_id"], row["source_record_id"]))
    rejections.sort(key=lambda row: (int(row["raw_row_number"]), row["reason_code"]))
    if len(clean) + len(rejections) != len(raw_rows):
        raise AssertionError(
            f"source-row reconciliation failed: clean={len(clean)} rejected={len(rejections)} raw={len(raw_rows)}"
        )
    counts = Counter(row["target"] for row in clean)
    summary = {
        "endpoint": endpoint,
        "source_rows": len(raw_rows),
        "standardized_candidate_rows": len(staged),
        "clean_unique_structures": len(clean),
        "clean_class_0_n": counts[0],
        "clean_class_1_n": counts[1],
        "rejection_rows": len(rejections),
        "duplicate_rows_aggregated": duplicate_rows,
        "conflicting_structure_groups": conflicting_groups,
        "murcko_scaffold_count_including_acyclic_singletons": len(
            {row["murcko_scaffold_id"] for row in clean}
        ),
    }
    return clean, rejections, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--source-id-col", default="Drug_ID")
    parser.add_argument("--smiles-col", default="Drug")
    parser.add_argument("--target-col", default="Y")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument("--source-code-commit", required=True)
    parser.add_argument("--source-file-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provenance = load_provenance(args.endpoint)
    raw_hash = sha256_file(args.raw)
    if raw_hash != provenance["raw_sha256_expected"]:
        raise ValueError(
            f"raw hash mismatch for {args.endpoint}: expected {provenance['raw_sha256_expected']} got {raw_hash}"
        )
    if rdBase.rdkitVersion != "2026.03.4":
        raise RuntimeError(f"development cleaning requires RDKit 2026.03.4, got {rdBase.rdkitVersion}")

    clean, rejections, summary = clean_classification(
        args.endpoint, args.raw, args.source_id_col, args.smiles_col, args.target_col
    )
    clean_path = args.processed_dir / f"{args.endpoint}_clean.csv"
    rejection_path = args.processed_dir / f"{args.endpoint}_rejections.csv"
    role_path = args.processed_dir / "role_inputs" / f"{args.endpoint}_role_input.csv"
    clean_fields = [
        "endpoint", "source_record_id", "raw_row_number", "raw_smiles",
        "standardized_smiles", "structure_id", "target", "murcko_scaffold_id",
        "standardization_status",
    ]
    rejection_fields = [
        "endpoint", "source_record_id", "raw_row_number", "raw_smiles", "stage",
        "reason_code", "detail", "structure_id_if_available",
    ]
    role_rows = [
        {
            "endpoint": row["endpoint"],
            "structure_id": row["structure_id"],
            "target": row["target"],
            "murcko_scaffold_id": row["murcko_scaffold_id"],
            "similarity_cluster_id": "",
        }
        for row in clean
    ]
    write_csv(clean_path, clean, clean_fields)
    write_csv(rejection_path, rejections, rejection_fields)
    write_csv(
        role_path,
        role_rows,
        ["endpoint", "structure_id", "target", "murcko_scaffold_id", "similarity_cluster_id"],
    )

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    acquisition = {
        "endpoint": args.endpoint,
        "retrieved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "access_layer": provenance["access_layer"],
        "loader_identifier": provenance["loader_identifier"],
        "distribution_identifier": provenance["distribution_identifier"],
        "source_file_id": str(args.source_file_id),
        "source_code_commit": args.source_code_commit,
        "source_url": f"https://dataverse.harvard.edu/api/access/datafile/{args.source_file_id}",
        "source_bytes": args.raw.stat().st_size,
        "raw_byte_sha256": raw_hash,
        "source_license_statement": provenance["source_license_statement"],
        "license_evidence_url": provenance["license_evidence_url"],
    }
    cleaning = {
        **summary,
        "standardization_id": STANDARDIZATION_ID,
        "rdkit_version": rdBase.rdkitVersion,
        "cleaning_code_commit": "containing_phase1_git_commit",
        "cleaning_script_byte_sha256": sha256_file(Path(__file__)),
        "standardization_contract_byte_sha256": sha256_file(CONTRACT_PATH),
        "cleaned_byte_sha256": sha256_file(clean_path),
        "rejection_log_byte_sha256": sha256_file(rejection_path),
        "role_input_byte_sha256": sha256_file(role_path),
        "classification_conflict_rule": "exclude_entire_standardized_structure_group",
        "same_label_duplicate_rule": "retain_lexicographically_first_source_record",
        "acyclic_scaffold_rule": "ACYCLIC:<structure_id>",
        "similarity_cluster_status": "pending_label_blind_algorithm_freeze",
    }
    (args.manifest_dir / f"{args.endpoint}_acquisition.json").write_text(
        json.dumps(acquisition, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.manifest_dir / f"{args.endpoint}_cleaning.json").write_text(
        json.dumps(cleaning, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"clean: {clean_path}")
    print(f"rejections: {rejection_path}")
    print(f"role input: {role_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
