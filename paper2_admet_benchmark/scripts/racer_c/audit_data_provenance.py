from __future__ import annotations

"""Validate the pre-freeze endpoint provenance and license contract.

The default command reports blockers without treating unresolved licenses as a
software failure.  ``--require-freeze-ready`` is the explicit fail-closed gate for
Freeze 1 and exits non-zero while any candidate endpoint remains unresolved.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
DEFAULT_PROVENANCE = P2 / "protocols" / "data_provenance_license_manifest.csv"
DEFAULT_CANDIDATES = P2 / "protocols" / "endpoint_candidate_manifest.csv"
DEFAULT_OUTPUT = P2 / "results" / "racer_c_preflight" / "data_provenance_audit.csv"

ALLOWED_ANALYSIS_STATUSES = {
    "allowed_with_attribution",
    "allowed_analysis_no_redistribution",
    "legacy_secondary_only",
    "pending_original_terms",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_csv_sha256(path: Path) -> str:
    """Hash parsed CSV values with deterministic LF line endings.

    This is supplementary to, not a replacement for, the raw byte hash.
    """

    rows = read_csv(path)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    fieldnames = list(rows[0])
    chunks = [",".join(json.dumps(v, ensure_ascii=False) for v in fieldnames)]
    for row in sorted(rows, key=lambda item: tuple(item.get(k, "") for k in fieldnames)):
        chunks.append(",".join(json.dumps(row.get(k, ""), ensure_ascii=False) for k in fieldnames))
    return hashlib.sha256(("\n".join(chunks) + "\n").encode("utf-8")).hexdigest()


def audit_rows(
    provenance_rows: Iterable[dict[str, str]],
    candidate_rows: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    provenance = list(provenance_rows)
    candidates = list(candidate_rows)
    candidate_names = {row["endpoint"] for row in candidates}
    candidate_by_name = {row["endpoint"]: row for row in candidates}
    observed_names = [row.get("endpoint", "") for row in provenance]
    if len(observed_names) != len(set(observed_names)):
        raise ValueError("duplicate endpoints in provenance manifest")
    if set(observed_names) != candidate_names:
        missing = sorted(candidate_names - set(observed_names))
        extra = sorted(set(observed_names) - candidate_names)
        raise ValueError(f"provenance/candidate mismatch; missing={missing}, extra={extra}")

    audit: list[dict[str, str]] = []
    for row in provenance:
        endpoint = row["endpoint"]
        issues: list[str] = []
        status = row.get("analysis_use_status", "")
        license_statement = row.get("source_license_statement", "")
        raw_hash = row.get("raw_sha256_expected", "")
        if status not in ALLOWED_ANALYSIS_STATUSES:
            issues.append("invalid_analysis_use_status")
        if not row.get("official_dataset_page", "").startswith("https://"):
            issues.append("missing_official_https_page")
        if not row.get("license_evidence_url", "").startswith("https://"):
            issues.append("missing_license_evidence_url")
        if raw_hash != "pending" and not HEX64.fullmatch(raw_hash):
            issues.append("invalid_expected_raw_sha256")
        if status == "allowed_with_attribution" and license_statement == "Not Specified":
            issues.append("allowed_without_explicit_license")
        if status == "allowed_with_attribution" and row.get("redistribution_status") != "yes_with_attribution":
            issues.append("license_redistribution_mismatch")
        if status == "allowed_analysis_no_redistribution":
            if "analysis" not in license_statement.lower() and "model" not in license_statement.lower():
                issues.append("analysis_permission_not_documented")
            if row.get("redistribution_status") not in {"no", "no_raw_redistribution_claimed"}:
                issues.append("analysis_only_redistribution_mismatch")

        license_ready = status in {
            "allowed_with_attribution",
            "allowed_analysis_no_redistribution",
            "legacy_secondary_only",
        }
        acquisition_ready = raw_hash != "pending"
        extension_candidate = status in {
            "allowed_with_attribution",
            "allowed_analysis_no_redistribution",
        }
        freeze_ready = extension_candidate and acquisition_ready and not issues
        blockers: list[str] = []
        if not license_ready:
            blockers.append("original_terms_unresolved")
        if not acquisition_ready:
            blockers.append("raw_hash_pending")
        if status == "legacy_secondary_only":
            blockers.append("legacy_secondary_not_extension_candidate")
        blockers.extend(issues)
        audit.append(
            {
                "endpoint": endpoint,
                "task_type": row.get("task_type", ""),
                "candidate_status": candidate_by_name[endpoint].get("candidate_status", ""),
                "eligibility_status": candidate_by_name[endpoint].get("eligibility_status", ""),
                "analysis_use_status": status,
                "license_ready": str(license_ready).lower(),
                "raw_hash_ready": str(acquisition_ready).lower(),
                "freeze1_ready": str(freeze_ready).lower(),
                "blockers": "|".join(blockers),
            }
        )
    return sorted(audit, key=lambda row: row["endpoint"])


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty provenance audit")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-freeze-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = audit_rows(read_csv(args.provenance), read_csv(args.candidates))
    write_csv(args.output, rows)
    ready = sum(row["freeze1_ready"] == "true" for row in rows)
    unresolved = sum("original_terms_unresolved" in row["blockers"] for row in rows)
    print(f"provenance rows: {len(rows)}")
    print(f"Freeze-1 ready before cleaning: {ready}/{len(rows)}")
    print(f"original-license blockers: {unresolved}")
    print(f"manifest byte sha256: {sha256_file(args.provenance)}")
    print(f"manifest canonical sha256: {canonical_csv_sha256(args.provenance)}")
    print(f"wrote: {args.output}")
    if args.require_freeze_ready:
        required = [
            row
            for row in rows
            if row["candidate_status"] == "freeze1_primary_candidate"
        ]
        if not required or any(row["freeze1_ready"] != "true" for row in required):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
