"""Frozen AMES, standard hERG, DILI and CYP2D6 data; no outcome-driven selection.

Public labels are read to exclude missing/conflicting annotations. Consequently
these are new endpoint-family benchmarks under a locked analysis, not blind
prospective validation. Outer target selection itself never reads labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES_PATH = HERE / "independent_sources.json"
ENDPOINTS = ("AMES", "hERG", "DILI", "CYP2D6_Veith")
SPLITS = ("random", "scaffold")
PHASES = tuple(f"independent_{split}" for split in SPLITS)
SEEDS = (311, 312, 313, 314, 315)
OUTER_SEEDS = {"random": 43021, "scaffold": 43022}
SCHEMA_VERSION = "independent_endpoint_families_v1"
ROW_FIELDS = ("sample_id", "structure_id", "standardized_smiles", "scaffold_id",
              "structure_status", "source_record_id", "source_row", "raw_smiles", "target")
EXCLUSION_FIELDS = ("source_row", "source_record_id", "raw_smiles", "raw_label",
                    "structure_id", "reason", "retained_source_row")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_manifest() -> dict:
    manifest = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    if tuple(manifest["sources"]) != ENDPOINTS:
        raise RuntimeError("independent endpoint set/order changed")
    contract = manifest["split_contract"]
    if (tuple(contract["fit_seeds"]) != SEEDS or contract["target_fraction"] != .2
            or any(contract[f"{split}_outer_seed"] != OUTER_SEEDS[split] for split in SPLITS)):
        raise RuntimeError("independent split constants and source manifest disagree")
    return manifest


def case_specs() -> list[dict]:
    """The complete required 4 endpoint x 2 regime x 5 training-seed matrix."""
    return [dict(phase=f"independent_{split}", endpoint=endpoint, split=split, seed=seed)
            for split in SPLITS for endpoint in ENDPOINTS for seed in SEEDS]


get_case_specs = case_specs


def _chemistry_contract(manifest: dict) -> dict:
    from rdkit import rdBase
    spec = manifest["chemistry"]
    if rdBase.rdkitVersion != spec["rdkit_version"]:
        raise RuntimeError(f"independent data require RDKit {spec['rdkit_version']}; "
                           f"found {rdBase.rdkitVersion}")
    return {"rdkit_version": rdBase.rdkitVersion,
            "standardization_id": spec["standardization_id"],
            "standardization_code_sha256": sha256_file(ROOT / spec["implementation"]),
            "standardization_contract_sha256": sha256_file(ROOT / spec["contract"])}


def _download_locked(source: dict, destination: Path) -> None:
    if destination.exists():
        if sha256_file(destination) != source["sha256"]:
            raise RuntimeError(f"raw SHA256 mismatch: {destination}; do not reuse this cache")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        raise RuntimeError(f"interrupted download exists: {temporary}; inspect/remove it before retry")
    request = urllib.request.Request(source["url"], headers={"User-Agent": "RACER-C4/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    if sha256_file(temporary) != source["sha256"]:
        raise RuntimeError(f"download SHA256 mismatch: {temporary}; source may have changed")
    if temporary.stat().st_size != source["bytes"]:
        raise RuntimeError(f"download byte-count mismatch: {temporary}")
    os.replace(temporary, destination)


def clean_rows(raw_rows: Sequence[dict], endpoint: str) -> tuple[list[dict], list[dict], dict]:
    """Use the existing chemistry contract; never majority-vote or flip labels.

    Duplicate selection follows the existing lexicographic source-record policy.
    Source row is an audit location only, never the molecular identity.
    """
    from paper2_admet_benchmark.scripts.racer_c.prepare_classification_endpoint import standardize_smiles
    groups: dict[str, list[dict]] = defaultdict(list)
    exclusions: list[dict] = []
    raw_counts: Counter = Counter()

    def exclusion(row: dict, reason: str, retained: str = "") -> dict:
        return dict(source_row=row["source_row"], source_record_id=row["source_record_id"],
                    raw_smiles=row["raw_smiles"], raw_label=row["raw_label"],
                    structure_id=row.get("structure_id", ""), reason=reason,
                    retained_source_row=retained)

    for ordinal, raw in enumerate(raw_rows, 2):
        if not {"Drug_ID", "Drug", "Y"}.issubset(raw):
            raise ValueError("TDC TSV requires Drug_ID, Drug and Y columns")
        row = dict(source_row=str(ordinal), source_record_id=str(raw["Drug_ID"]),
                   raw_smiles=str(raw["Drug"] or ""), raw_label=str(raw["Y"] or ""))
        try:
            value = float(row["raw_label"])
            if not math.isfinite(value) or value not in (0., 1.):
                raise ValueError("non-binary annotation")
        except ValueError:
            exclusions.append(exclusion(row, "missing_or_nonbinary_label"))
            continue
        row["target"] = int(value)
        raw_counts[str(int(value))] += 1
        try:
            smiles, identity, scaffold = standardize_smiles(row["raw_smiles"])
        except ValueError as error:
            exclusions.append(exclusion(row, str(error)))
            continue
        row.update(standardized_smiles=smiles, structure_id=identity, scaffold_id=scaffold)
        groups[identity].append(row)
    clean = []
    conflict_groups = 0
    for identity in sorted(groups):
        group = sorted(groups[identity], key=lambda row: (row["source_record_id"],
                                                        row["raw_smiles"], int(row["source_row"])))
        if len({row["target"] for row in group}) > 1:
            conflict_groups += 1
            exclusions.extend(exclusion(row, "conflicting_structure_labels") for row in group)
            continue
        keep = group[0]
        clean.append(dict(sample_id=f"{endpoint}:{identity}", structure_id=identity,
                          standardized_smiles=keep["standardized_smiles"], scaffold_id=keep["scaffold_id"],
                          structure_status="pass", source_record_id=keep["source_record_id"],
                          source_row=keep["source_row"], raw_smiles=keep["raw_smiles"], target=keep["target"]))
        exclusions.extend(exclusion(row, "same_label_structure_duplicate", keep["source_row"])
                          for row in group[1:])
    exclusions.sort(key=lambda row: int(row["source_row"]))
    counts = {"raw_rows": len(raw_rows), "clean_structures": len(clean),
              "excluded_rows": len(exclusions), "conflicting_structure_groups": conflict_groups,
              "raw_label_counts": dict(sorted(raw_counts.items())),
              "clean_label_counts": dict(sorted(Counter(str(row["target"]) for row in clean).items())),
              "exclusion_reasons": dict(sorted(Counter(row["reason"] for row in exclusions).items()))}
    if counts["raw_rows"] != counts["clean_structures"] + counts["excluded_rows"]:
        raise AssertionError("cleaning accounting does not balance")
    return clean, exclusions, counts


def outer_split(rows: Sequence[dict], split: str) -> tuple[list[str], list[str]]:
    """Return source/target structure IDs using structural columns ONLY.

    The immutable outer seeds are deliberately independent of training seeds.
    All IDs are sorted on output. Molecular/scaffold membership cannot depend
    on file order, labels, model scores or gate success.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown independent split: {split}")
    identities = sorted(str(row["structure_id"]) for row in rows)
    if len(identities) < 2 or len(set(identities)) != len(identities):
        raise ValueError("outer split requires at least two distinct cleaned structures")
    target_n = math.ceil(.2 * len(identities))
    seed = OUTER_SEEDS[split]
    rank = lambda identity: hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest()
    if split == "random":
        target = set(sorted(identities, key=rank)[:target_n])
    else:
        groups: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            groups[str(row["scaffold_id"])].append(str(row["structure_id"]))
        target: set[str] = set()
        for scaffold in sorted(groups, key=rank):
            group = groups[scaffold]
            if abs(len(target) + len(group) - target_n) < abs(len(target) - target_n):
                target.update(group)
    source = set(identities) - target
    if not source or not target:
        raise RuntimeError("label-blind split produced an empty role; retain failure, do not retry seeds")
    return sorted(source), sorted(target)


def prepare_endpoint(cache: Path, endpoint: str) -> dict:
    """Download, clean and freeze two outer splits, or verify a completed cache."""
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown independent endpoint: {endpoint}")
    manifest = source_manifest()
    chemistry = _chemistry_contract(manifest)
    source = manifest["sources"][endpoint]
    folder = Path(cache) / "independent" / endpoint
    contract = dict(schema_version=SCHEMA_VERSION, endpoint=endpoint,
                    sources_manifest_sha256=sha256_file(SOURCES_PATH),
                    preparation_code_sha256=sha256_file(Path(__file__)), **chemistry)
    _download_locked(source, folder / "raw.tab")
    manifest_path = folder / "preparation.json"
    if manifest_path.exists():
        prepared = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prepared["contract"] != contract:
            raise RuntimeError(f"independent preparation contract changed: {folder}; use a new cache")
        for filename, expected in prepared["artifact_hashes"].items():
            path = folder / filename
            if not path.exists() or sha256_file(path) != expected:
                raise RuntimeError(f"independent preparation artifact missing/changed: {path}")
        return prepared
    with (folder / "raw.tab").open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(raw_rows) != source["raw_rows"]:
        raise RuntimeError(f"raw row count changed for {endpoint}")
    clean, exclusions, counts = clean_rows(raw_rows, endpoint)
    _atomic_csv(folder / "cleaned.csv", clean, ROW_FIELDS)
    _atomic_csv(folder / "exclusions.csv", exclusions, EXCLUSION_FIELDS)
    split_metadata = {}
    for split in SPLITS:
        source_ids, target_ids = outer_split(clean, split)
        source_set, target_set = set(source_ids), set(target_ids)
        source_scaffolds = {row["scaffold_id"] for row in clean if row["structure_id"] in source_set}
        target_scaffolds = {row["scaffold_id"] for row in clean if row["structure_id"] in target_set}
        n_calibration = math.ceil(.5 * math.ceil(.4 * len(source_ids)))
        record = dict(split=split, outer_seed=OUTER_SEEDS[split],
                      target_selection_uses_labels=False, fixed_across_fit_seeds=list(SEEDS),
                      source_structure_ids=source_ids, target_structure_ids=target_ids,
                      n_source=len(source_ids), n_target=len(target_ids),
                      target_fraction=len(target_ids) / len(clean),
                      requested_target_fraction=.2,
                      structure_overlap=0, scaffold_overlap=len(source_scaffolds & target_scaffolds),
                      target_id_sha256=stable_sha256(target_ids),
                      target_n_below_historical_100_gate=len(target_ids) < 100,
                      n_source_calibration_expected=n_calibration,
                      source_calibration_n_below_historical_100_gate=n_calibration < 100,
                      source_label_counts=dict(sorted(Counter(str(row["target"]) for row in clean
                                                              if row["structure_id"] in source_set).items())),
                      target_label_counts=dict(sorted(Counter(str(row["target"]) for row in clean
                                                              if row["structure_id"] in target_set).items())))
        if split == "scaffold" and record["scaffold_overlap"]:
            raise AssertionError("scaffold crossed outer roles")
        _atomic_json(folder / f"split_{split}.json", record)
        split_metadata[split] = {key: value for key, value in record.items()
                                 if key not in ("source_structure_ids", "target_structure_ids")}
    artifact_names = ["raw.tab", "cleaned.csv", "exclusions.csv", "split_random.json", "split_scaffold.json"]
    prepared = dict(contract=contract, counts=counts, splits=split_metadata,
                    study_status=manifest["study_status"], documented_drugs=source["documented_drugs"],
                    artifact_hashes={name: sha256_file(folder / name) for name in artifact_names})
    _atomic_json(manifest_path, prepared)
    return prepared


def load_case(cache: Path, endpoint: str, split: str, seed: int) -> dict:
    """Runner interface; source_y is untouched binary y, target labels are separate.

    Source internal fit/router/cal splitting is the runner's existing
    split_internal_roles(source_y, seed), not an operation on this target.
    """
    if split.startswith("independent_"):
        split = split.removeprefix("independent_")
    if split not in SPLITS or seed not in SEEDS:
        raise ValueError("case is outside the locked independent split/seed matrix")
    prepared = prepare_endpoint(cache, endpoint)
    folder = Path(cache) / "independent" / endpoint
    record = json.loads((folder / f"split_{split}.json").read_text(encoding="utf-8"))
    rows = _read_csv(folder / "cleaned.csv")
    source_ids, target_ids = set(record["source_structure_ids"]), set(record["target_structure_ids"])
    source_rows = [row for row in rows if row["structure_id"] in source_ids]
    target_labeled = [row for row in rows if row["structure_id"] in target_ids]
    labels = {row["sample_id"]: {endpoint: float(row["target"])} for row in target_labeled}
    # Preserve structural fields and provenance, but keep outcomes out of the
    # target table handed to feature extraction/domain classification.
    target_rows = [{key: value for key, value in row.items() if key != "target"}
                   for row in target_labeled]
    source_y = np.asarray([int(row["target"]) for row in source_rows], dtype=np.int8)
    return dict(source_rows=source_rows, source_y=source_y, target_rows=target_rows,
                labels=labels, eligible=np.ones(len(target_rows), dtype=bool),
                phase=f"independent_{split}", endpoint=endpoint, split=split, seed=int(seed),
                input_hashes={**prepared["artifact_hashes"],
                              "sources_manifest_sha256": sha256_file(SOURCES_PATH),
                              "preparation_sha256": sha256_file(folder / "preparation.json"),
                              "independent_data_code_sha256": sha256_file(Path(__file__)),
                              **prepared["contract"]},
                split_metadata={key: value for key, value in record.items()
                                if key not in ("source_structure_ids", "target_structure_ids")},
                preparation=prepared)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--endpoint", choices=ENDPOINTS, action="append")
    args = parser.parse_args(argv)
    for endpoint in args.endpoint or ENDPOINTS:
        prepared = prepare_endpoint(args.cache, endpoint)
        print(json.dumps({"endpoint": endpoint, "counts": prepared["counts"],
                          "splits": prepared["splits"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
