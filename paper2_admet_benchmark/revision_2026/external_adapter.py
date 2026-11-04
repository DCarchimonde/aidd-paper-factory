"""Validate external comparator predictions against a label-free target manifest.

This imports outputs; it does not implement or certify CoDrug/KMM-CP. Run from
the repository root with ``python -m paper2_admet_benchmark.revision_2026.external_adapter --help``.
No test labels are loaded. Excluded molecules remain identifiable in the output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


PREDICTION_COLUMNS = {"sample_id", "alpha", "include_0", "include_1", "selected", "probability_1"}
LABEL_ROLES = {"fit", "router", "conformal"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise ValueError(f"Duplicate column names in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    if any(None in row for row in rows):
        raise ValueError(f"Malformed extra CSV fields in {path}")
    return fields, rows


def bit(value: str, name: str) -> int:
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be exactly 0 or 1")
    return int(value)


def finite_probability(value: str, name: str, strict: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or not (0 < number < 1 if strict else 0 <= number <= 1):
        raise ValueError(f"Invalid {name}: {value!r}")
    return number


def validate_metadata(meta: dict, expected_protocol_sha256: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_protocol_sha256):
        raise ValueError("Expected protocol SHA256 must be 64 lowercase hexadecimal characters")
    if meta.get("protocol_sha256") != expected_protocol_sha256:
        raise ValueError("Protocol SHA256 mismatch")
    for key in ("method", "evaluation_model", "source_url", "source_revision", "adapter_version", "implementation_kind"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError(f"Missing metadata field: {key}")
    if meta["implementation_kind"] not in {"official_code", "paper_reimplementation", "component_only"}:
        raise ValueError("implementation_kind must identify official code, paper reimplementation, or component only")
    if not meta["source_url"].startswith("https://"):
        raise ValueError("source_url must be an HTTPS source link")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", meta["method"]) or not re.fullmatch(r"[A-Za-z0-9_.-]+", meta["evaluation_model"]):
        raise ValueError("Method and evaluation_model require simple stable identifiers")
    if meta["evaluation_model"] in {"logistic", "stack", "equal_logit"}:
        raise ValueError("External models require a distinct evaluation_model name")
    if meta.get("target_labels_used") is not False:
        raise ValueError("Explicit target_labels_used=false declaration required")
    if not isinstance(meta.get("target_features_used"), bool):
        raise ValueError("Explicit target_features_used boolean required")
    roles = meta.get("fitted_label_roles")
    if not isinstance(roles, list) or not all(isinstance(r, str) and r in LABEL_ROLES for r in roles):
        raise ValueError("Only fit/router/conformal label roles are allowed; declare roles actually used")
    if not isinstance(meta.get("dependencies"), dict) or not meta["dependencies"]:
        raise ValueError("Provide dependency package versions")


def import_predictions(target_manifest: Path, predictions: Path, metadata: Path,
                       expected_protocol_sha256: str, expected_alphas: list[float],
                       output_dir: Path) -> dict:
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    validate_metadata(meta, expected_protocol_sha256)
    if not expected_alphas or len(set(expected_alphas)) != len(expected_alphas):
        raise ValueError("Expected alphas must be a nonempty list without duplicates")
    for alpha in expected_alphas:
        finite_probability(str(alpha), "expected alpha", strict=True)
    mf, manifest = read_csv(target_manifest)
    if any(c.lower() in {"y", "label", "target", "y_true", "true_label"} for c in mf):
        raise ValueError("Target manifest must contain no outcome label column")
    required_manifest = {"sample_id", "structure_id", "phase", "endpoint", "seed", "model"}
    if not required_manifest.issubset(mf):
        raise ValueError(f"Target manifest missing {sorted(required_manifest - set(mf))}")
    ids = [r["sample_id"] for r in manifest]
    if any(not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("Target sample IDs must be nonempty and unique in a cell")
    cells = {(r["phase"], r["endpoint"], r["seed"], r["model"]) for r in manifest}
    if len(cells) != 1:
        raise ValueError("Use one phase/endpoint/seed/reference-model cell per import")
    if meta.get("target_manifest_sha256") != sha256(target_manifest):
        raise ValueError("Metadata target_manifest_sha256 does not match the supplied manifest")
    pf, rows = read_csv(predictions)
    if not {"sample_id", "alpha", "include_0", "include_1"}.issubset(pf):
        raise ValueError("Predictions require sample_id, alpha, include_0, include_1")
    if set(pf) - PREDICTION_COLUMNS:
        raise ValueError(f"Unexpected columns (labels are prohibited): {sorted(set(pf) - PREDICTION_COLUMNS)}")
    lookup: dict[tuple[float, str], dict] = {}
    allowed_ids, allowed_alphas = set(ids), set(expected_alphas)
    for row in rows:
        alpha = finite_probability(row["alpha"], "alpha", strict=True)
        sid = row["sample_id"]
        key = (alpha, sid)
        if sid not in allowed_ids or alpha not in allowed_alphas:
            raise ValueError(f"Unexpected sample ID or alpha: {key}")
        if key in lookup:
            raise ValueError(f"Duplicate prediction: {key}")
        include = [bit(row[f"include_{i}"], f"include_{i}") for i in (0, 1)]
        selected = bit(row.get("selected", "1"), "selected")
        prob = row.get("probability_1", "")
        if prob:
            finite_probability(prob, "probability_1")
        lookup[key] = {"alpha": alpha, "include_0": include[0] if selected else 1,
                       "include_1": include[1] if selected else 1, "external_selected": selected,
                       "external_original_include_0": include[0],
                       "external_original_include_1": include[1],
                       "probability_1": prob}
    expected = {(alpha, sid) for alpha in expected_alphas for sid in ids}
    if set(lookup) != expected:
        raise ValueError(f"Incomplete target universe: missing {len(expected - set(lookup))} predictions")
    output_rows = []
    for alpha in expected_alphas:
        for record in manifest:
            row = dict(record)
            row["reference_model"] = row["model"]
            row["model"] = meta["evaluation_model"]
            row.update(lookup[(alpha, record["sample_id"])])
            row["method"] = meta["method"]
            output_rows.append(row)
    if output_dir.exists():
        raise ValueError("Output directory already exists; use a new directory to preserve earlier imports")
    output_dir.mkdir(parents=True)
    fields = list(dict.fromkeys(mf + ["reference_model", "alpha", "include_0", "include_1", "external_selected", "external_original_include_0", "external_original_include_1", "probability_1", "method"]))
    result_path = output_dir / "predictions.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    report = {"status": "IMPORTED_NOT_REPRODUCTION_CERTIFIED", "metadata": meta,
              "input_sha256": {"target_manifest": sha256(target_manifest), "predictions": sha256(predictions), "metadata": sha256(metadata)},
              "output_sha256": sha256(result_path), "rows": len(output_rows),
              "unique_molecules": len(ids), "alphas": expected_alphas,
              "unselected_policy": "full_label_set_for_full_population_metrics; report selection rate separately",
              "checks": ["exact_target_universe_per_alpha", "no_label_columns", "no_duplicate_ids", "explicit_provenance"],
              "limitations": "Provenance declarations do not prove that upstream code avoided test-label use. Review upstream code and logs before claiming faithful reproduction."}
    (output_dir / "import_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = import_predictions(args.target_manifest, args.predictions, args.metadata,
                                args.protocol_sha256, args.alphas, args.output_dir)
    print(json.dumps({"status": report["status"], "rows": report["rows"]}))


if __name__ == "__main__":
    main()
