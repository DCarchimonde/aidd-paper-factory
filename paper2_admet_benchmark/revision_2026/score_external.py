"""Score already-frozen external predictions after an explicit outcome join.

Prediction construction and import precede this command. External unselected
rows defer in full-cohort accounting; retained-cohort metrics are separate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .core import decision_metrics
from .external_adapter import bit, finite_probability, read_csv, sha256


def outcome(value: str) -> int:
    if value.strip().lower() in {"", "nan", "none"}:
        return -1
    number = float(value)
    if not math.isfinite(number) or number not in {-1, 0, 1}:
        raise ValueError("Reference outcomes must be 0/1 or explicit missing (-1/blank/NaN)")
    return int(number)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_external(import_dir: Path, reference_predictions: Path, output_dir: Path) -> dict:
    # Verify the label-free artifact before opening any outcome-bearing file.
    pred_path = import_dir / "predictions.csv"
    report_path = import_dir / "import_report.json"
    imported_report = json.loads(report_path.read_text(encoding="utf-8"))
    if imported_report.get("status") != "IMPORTED_NOT_REPRODUCTION_CERTIFIED":
        raise ValueError("Expected a successful external adapter import report")
    if imported_report.get("output_sha256") != sha256(pred_path):
        raise ValueError("Imported prediction hash mismatch; predictions changed after import")
    _, external = read_csv(pred_path)
    required = {"sample_id", "structure_id", "phase", "endpoint", "seed", "model", "reference_model",
                "method", "alpha", "include_0", "include_1", "external_selected",
                "external_original_include_0", "external_original_include_1"}
    if not required.issubset(external[0]) or "y" in external[0]:
        raise ValueError("Expected the label-free external adapter output schema")
    cell_set = {(r["phase"], r["endpoint"], r["seed"], r["reference_model"]) for r in external}
    if len(cell_set) != 1 or len({(r["model"], r["method"]) for r in external}) != 1:
        raise ValueError("One external method and reference cell per evaluation")
    cell = next(iter(cell_set))
    ext_keys = set()
    for row in external:
        key = (finite_probability(row["alpha"], "alpha", strict=True), row["sample_id"])
        if key in ext_keys:
            raise ValueError("Duplicate external prediction")
        ext_keys.add(key)
        selected = bit(row["external_selected"], "external_selected")
        for c in (0, 1):
            actual = bit(row[f"include_{c}"], f"include_{c}")
            original = bit(row[f"external_original_include_{c}"], f"external_original_include_{c}")
            if actual != (original if selected else 1):
                raise ValueError("External full-population flags violate the imported deferral policy")
    # Only now load outcomes. The reference file is never used to revise sets.
    rf, reference = read_csv(reference_predictions)
    if not {"phase", "endpoint", "seed", "model", "method", "alpha", "sample_id", "structure_id", "y", "eligible"}.issubset(rf):
        raise ValueError("Reference predictions lack identity/outcome/eligibility columns")
    if {(r["phase"], r["endpoint"], r["seed"], r["model"]) for r in reference} != {cell}:
        raise ValueError("Reference phase/endpoint/seed/model cell differs from imported reference identity")
    wanted_alphas = {key[0] for key in ext_keys}
    lookup = {}
    physical = set()
    across_alphas = {}
    for row in reference:
        alpha = finite_probability(row["alpha"], "reference alpha", strict=True)
        if alpha not in wanted_alphas:
            continue
        sid = row["sample_id"]
        physical_key = (alpha, row["method"], sid)
        if physical_key in physical:
            raise ValueError("Duplicate reference method/sample/alpha identity")
        physical.add(physical_key)
        y, eligible = outcome(row["y"]), bit(row["eligible"], "eligible")
        facts = (row["structure_id"], y, eligible)
        key = (alpha, sid)
        if key in lookup and lookup[key] != facts:
            raise ValueError("Reference methods disagree on structure/outcome/eligibility")
        if sid in across_alphas and across_alphas[sid] != facts:
            raise ValueError("Reference alphas disagree on structure/outcome/eligibility")
        lookup[key] = facts
        across_alphas[sid] = facts
    if set(lookup) != ext_keys:
        raise ValueError("External/reference target universe differs at one or more alphas")
    joined = []
    for row in external:
        facts = lookup[(float(row["alpha"]), row["sample_id"])]
        if row["structure_id"] != facts[0]:
            raise ValueError("External/reference structure identity mismatch")
        if "eligible" in row and bit(row["eligible"], "eligible") != facts[2]:
            raise ValueError("External/reference eligibility mismatch")
        joined.append({**row, "y": facts[1], "eligible": facts[2]})
    metrics_full, metrics_selected = [], []
    for alpha in sorted(wanted_alphas):
        all_rows = [r for r in joined if float(r["alpha"]) == alpha]
        valid = [r for r in all_rows if r["eligible"] == 1 and r["y"] in (0, 1)]
        chosen = [r for r in valid if r["external_selected"] == "1"]
        ratio = lambda a, b: a / b if b else math.nan
        common = {"phase": cell[0], "endpoint": cell[1], "seed": cell[2],
                  "model": all_rows[0]["model"], "reference_model": cell[3],
                  "method": all_rows[0]["method"], "alpha": alpha,
                  "n_target_total": len(all_rows), "n_eligible_labeled": len(valid),
                  "n_selected_eligible_labeled": len(chosen),
                  "selection_rate_all_target": ratio(sum(r["external_selected"] == "1" for r in all_rows), len(all_rows)),
                  "selection_rate_eligible_labeled": ratio(len(chosen), len(valid))}
        for c in (0, 1):
            denom = sum(r["y"] == c for r in valid)
            numer = sum(r["y"] == c for r in chosen)
            common.update({f"class_{c}_n_full": denom, f"class_{c}_n_selected": numer,
                           f"class_{c}_retention": ratio(numer, denom)})
        for rows, target, prefix, population in ((valid, metrics_full, "include", "full_eligible_labeled"),
                                                (chosen, metrics_selected, "external_original_include", "selected_eligible_labeled")):
            y = np.asarray([r["y"] for r in rows], dtype=int)
            sets = np.asarray([[int(r[f"{prefix}_0"]), int(r[f"{prefix}_1"])] for r in rows], dtype=bool).reshape(-1, 2)
            target.append({**common, "population": population, **decision_metrics(y, sets)})
    if output_dir.exists():
        raise ValueError("Output directory already exists; use a new evaluation directory")
    output_dir.mkdir(parents=True)
    write_csv(output_dir / "predictions.csv", joined)
    write_csv(output_dir / "metrics_full.csv", metrics_full)
    write_csv(output_dir / "metrics_selected.csv", metrics_selected)
    report = {"status": "EXTERNAL_OUTPUTS_SCORED_NOT_REPRODUCTION_CERTIFIED",
              "input_sha256": {"import_predictions": sha256(pred_path), "import_report": sha256(report_path),
                               "reference_predictions": sha256(reference_predictions)},
              "output_sha256": {p: sha256(output_dir / p) for p in ("predictions.csv", "metrics_full.csv", "metrics_selected.csv")},
              "rows": len(joined), "alphas": sorted(wanted_alphas),
              "protocol_sha256": imported_report["metadata"]["protocol_sha256"],
              "population_note": "Full means the common eligible labeled target cohort. Unselected molecules defer; selected-set metrics use original flags and a separately reported denominator.",
              "limitation": "Input hashing and joins do not certify upstream algorithm fidelity or lack of earlier label access."}
    (output_dir / "scoring_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-dir", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = score_external(args.import_dir, args.reference_predictions, args.output_dir)
    print(json.dumps({"status": result["status"], "rows": result["rows"]}))


if __name__ == "__main__":
    main()
