from __future__ import annotations

"""Prepare the twelve official 2014 NCATS Tox21 challenge endpoints.

The source archive is one multi-assay SDF.  Each endpoint is reconciled against
every source record: absent assay calls are logged as ``label_unavailable``;
invalid structures and duplicate/conflicting standardized structures are also
logged explicitly.  Raw challenge bytes are never written to the repository.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, rdBase

from prepare_classification_endpoint import (
    PROVENANCE_PATH,
    STANDARDIZATION_ID,
    sha256_file,
    standardize_smiles,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
DEFAULT_PROCESSED = P2 / "data" / "processed" / "racer_c"
DEFAULT_MANIFESTS = P2 / "data" / "manifests" / "racer_c"
SOURCE_ACCESS_LAYER = "NCATS_Tox21_2014"

ENDPOINT_PROPERTIES = {
    "Tox21_NR_AR": "NR-AR",
    "Tox21_NR_AhR": "NR-AhR",
    "Tox21_NR_AR_LBD": "NR-AR-LBD",
    "Tox21_NR_ER": "NR-ER",
    "Tox21_NR_ER_LBD": "NR-ER-LBD",
    "Tox21_NR_Aromatase": "NR-Aromatase",
    "Tox21_NR_PPAR_gamma": "NR-PPAR-gamma",
    "Tox21_SR_ARE": "SR-ARE",
    "Tox21_SR_ATAD5": "SR-ATAD5",
    "Tox21_SR_HSE": "SR-HSE",
    "Tox21_SR_MMP": "SR-MMP",
    "Tox21_SR_p53": "SR-p53",
}


def read_provenance() -> dict[str, dict[str, str]]:
    with PROVENANCE_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        row["endpoint"]: row
        for row in rows
        if row.get("access_layer") == SOURCE_ACCESS_LAYER
    }
    if set(selected) != set(ENDPOINT_PROPERTIES):
        raise ValueError(
            "Tox21 provenance rows do not match frozen endpoint list: "
            f"missing={sorted(set(ENDPOINT_PROPERTIES) - set(selected))}; "
            f"extra={sorted(set(selected) - set(ENDPOINT_PROPERTIES))}"
        )
    for endpoint, row in selected.items():
        if row["analysis_use_status"] != "allowed_analysis_no_redistribution":
            raise PermissionError(f"{endpoint} is not approved for analysis")
        if row["raw_sha256_expected"] == "pending":
            raise ValueError(f"{endpoint} archive hash is pending")
    return selected


def molecule_to_source_smiles(mol: Chem.Mol) -> str:
    candidate = Chem.Mol(mol)
    Chem.SanitizeMol(candidate)
    smiles = Chem.MolToSmiles(candidate, canonical=False, isomericSmiles=True)
    if not smiles:
        raise ValueError("source_structure_to_smiles_failure")
    return smiles


def read_source_rows(sdf_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    supplier = Chem.ForwardSDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    for source_row_number, mol in enumerate(supplier, start=1):
        row: dict[str, object] = {
            "source_row_number": source_row_number,
            "mol": mol,
            "source_record_id": "",
            "raw_smiles": "",
            "structure_error": "",
            "standardized_smiles": "",
            "structure_id": "",
            "murcko_scaffold_id": "",
            "labels": {},
        }
        if mol is None:
            row["structure_error"] = "sdf_parse_failure"
            rows.append(row)
            continue
        row["source_record_id"] = (
            mol.GetProp("DSSTox_CID").strip() if mol.HasProp("DSSTox_CID") else ""
        )
        try:
            row["raw_smiles"] = molecule_to_source_smiles(mol)
            (
                row["standardized_smiles"],
                row["structure_id"],
                row["murcko_scaffold_id"],
            ) = standardize_smiles(str(row["raw_smiles"]))
        except Exception as exc:
            row["structure_error"] = "source_structure_to_smiles_failure"
            if isinstance(exc, ValueError) and str(exc):
                row["structure_error"] = str(exc)
        row["labels"] = {
            prop: (mol.GetProp(prop).strip() if mol.HasProp(prop) else "")
            for prop in ENDPOINT_PROPERTIES.values()
        }
        rows.append(row)
    if not rows:
        raise ValueError("Tox21 source SDF is empty")
    return rows


def prepare_endpoint(
    endpoint: str,
    property_name: str,
    source_rows: list[dict[str, object]],
) -> tuple[list[dict], list[dict], dict]:
    staged: list[dict] = []
    rejections: list[dict] = []
    observed_labels = 0
    for row in source_rows:
        source_record_id = str(row["source_record_id"])
        source_row_number = int(row["source_row_number"])
        raw_smiles = str(row["raw_smiles"])
        labels = dict(row["labels"])
        target_raw = str(labels.get(property_name, ""))
        base = {
            "endpoint": endpoint,
            "source_record_id": source_record_id,
            "raw_row_number": source_row_number,
            "raw_smiles": raw_smiles,
            "stage": "source_validation",
            "reason_code": "",
            "detail": "",
            "structure_id_if_available": "",
        }
        if target_raw == "":
            rejections.append({**base, "reason_code": "label_unavailable"})
            continue
        observed_labels += 1
        if target_raw not in {"0", "1"}:
            rejections.append(
                {**base, "reason_code": "invalid_target", "detail": target_raw}
            )
            continue
        if row["mol"] is None or row["structure_error"]:
            rejections.append(
                {
                    **base,
                    "stage": "standardization",
                    "reason_code": str(row["structure_error"] or "sdf_parse_failure"),
                }
            )
            continue
        staged.append(
            {
                "endpoint": endpoint,
                "source_record_id": source_record_id,
                "raw_row_number": source_row_number,
                "raw_smiles": raw_smiles,
                "standardized_smiles": str(row["standardized_smiles"]),
                "structure_id": str(row["structure_id"]),
                "target": int(target_raw),
                "murcko_scaffold_id": str(row["murcko_scaffold_id"]),
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
        ordered = sorted(
            group,
            key=lambda item: (str(item["source_record_id"]), int(item["raw_row_number"])),
        )
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

    clean.sort(key=lambda row: (row["structure_id"], str(row["source_record_id"])))
    rejections.sort(key=lambda row: (int(row["raw_row_number"]), row["reason_code"]))
    if len(clean) + len(rejections) != len(source_rows):
        raise AssertionError(
            f"{endpoint} reconciliation failed: clean={len(clean)} "
            f"rejected={len(rejections)} source={len(source_rows)}"
        )
    counts = Counter(int(row["target"]) for row in clean)
    summary = {
        "endpoint": endpoint,
        "source_archive_rows": len(source_rows),
        "endpoint_observed_label_rows": observed_labels,
        "standardized_candidate_rows": len(staged),
        "clean_unique_structures": len(clean),
        "clean_class_0_n": counts[0],
        "clean_class_1_n": counts[1],
        "duplicate_rows_aggregated": duplicate_rows,
        "conflicting_structure_groups_excluded": conflicting_groups,
        "missing_label_rows": sum(
            row["reason_code"] == "label_unavailable" for row in rejections
        ),
        "murcko_scaffold_count": len({row["murcko_scaffold_id"] for row in clean}),
    }
    return clean, rejections, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFESTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if rdBase.rdkitVersion != "2026.03.4":
        raise RuntimeError(f"cleaning requires RDKit 2026.03.4, got {rdBase.rdkitVersion}")
    provenance = read_provenance()
    archive_hash = sha256_file(args.archive)
    expected_hashes = {row["raw_sha256_expected"] for row in provenance.values()}
    if expected_hashes != {archive_hash}:
        raise ValueError(
            f"archive hash mismatch: observed={archive_hash}; expected={sorted(expected_hashes)}"
        )
    source_rows = read_source_rows(args.sdf)
    sdf_hash = sha256_file(args.sdf)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    (args.processed_dir / "role_inputs").mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    script_hash = sha256_file(Path(__file__))
    for endpoint, property_name in ENDPOINT_PROPERTIES.items():
        clean, rejections, summary = prepare_endpoint(
            endpoint, property_name, source_rows
        )
        clean_path = args.processed_dir / f"{endpoint}_clean.csv"
        rejection_path = args.processed_dir / f"{endpoint}_rejections.csv"
        role_path = args.processed_dir / "role_inputs" / f"{endpoint}_role_input.csv"
        write_csv(clean_path, clean, list(clean[0]))
        write_csv(
            rejection_path,
            rejections,
            [
                "endpoint",
                "source_record_id",
                "raw_row_number",
                "raw_smiles",
                "stage",
                "reason_code",
                "detail",
                "structure_id_if_available",
            ],
        )
        role_rows = [
            {
                "endpoint": endpoint,
                "structure_id": row["structure_id"],
                "target": row["target"],
                "murcko_scaffold_id": row["murcko_scaffold_id"],
                "similarity_cluster_id": "",
            }
            for row in clean
        ]
        write_csv(role_path, role_rows, list(role_rows[0]))
        now = datetime.now(timezone.utc).isoformat()
        acquisition = {
            "endpoint": endpoint,
            "access_layer": SOURCE_ACCESS_LAYER,
            "source_url": provenance[endpoint]["raw_url_or_identifier"],
            "source_archive_sha256": archive_hash,
            "source_archive_bytes": args.archive.stat().st_size,
            "source_member": args.sdf.name,
            "source_member_sha256": sdf_hash,
            "source_member_bytes": args.sdf.stat().st_size,
            "source_rows": len(source_rows),
            "assay_property": property_name,
            "label_semantics": "1=active; 0=inactive; absent property=unavailable",
            "analysis_use_status": provenance[endpoint]["analysis_use_status"],
            "redistribution_status": provenance[endpoint]["redistribution_status"],
            "retrieved_at_utc": now,
        }
        cleaning = {
            **summary,
            "rdkit_version": rdBase.rdkitVersion,
            "standardization_id": STANDARDIZATION_ID,
            "source_archive_sha256": archive_hash,
            "source_member_sha256": sdf_hash,
            "cleaned_byte_sha256": sha256_file(clean_path),
            "rejections_byte_sha256": sha256_file(rejection_path),
            "role_input_byte_sha256": sha256_file(role_path),
            "cleaning_script_byte_sha256": script_hash,
            "similarity_cluster_status": "pending",
        }
        (args.manifest_dir / f"{endpoint}_acquisition.json").write_text(
            json.dumps(acquisition, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.manifest_dir / f"{endpoint}_cleaning.json").write_text(
            json.dumps(cleaning, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(cleaning, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
