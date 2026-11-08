"""Read NCATS paired reporter/viability curves without using activity calls as labels.

CONC0..15 are molar concentrations; DATA0..15 are normalized percent responses.
Official fitted AC50 is in micromolar and is retained only as descriptive metadata.
A pair uses the documented replicate index of the two channels within one sample
and protocol. Public files expose no physical plate/well/batch identifiers.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping

HERE = Path(__file__).resolve().parent
SOURCE_FILE = HERE / "sources.json"
STRUCTURE_POLICY = "ncats_window_rdkit_parent_v1"
MISSING = {"", "na", "nan", "null", "none"}
REQUIRED = {"SAMPLE_ID", "SAMPLE_DATA_ID", "PROTOCOL_NAME", "SAMPLE_DATA_TYPE",
            "SMILES", "PURITY_RATING", *[f"DATA{i}" for i in range(16)],
            *[f"CONC{i}" for i in range(16)]}


def source_manifest() -> dict:
    return json.loads(SOURCE_FILE.read_text(encoding="utf-8"))


def _spec(protocol: str) -> dict:
    sources = source_manifest()
    if protocol not in sources["protocols"]:
        raise ValueError(f"undeclared protocol: {protocol}")
    return sources["protocols"][protocol]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def download_protocol(protocol: str, cache_root: Path) -> tuple[Path, dict]:
    """Download only the selected archive; verify content on cache reuse."""
    spec = _spec(protocol)
    cache = Path(cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{protocol}.zip"
    receipt_path = cache / f"{protocol}.download.json"
    url = source_manifest()["download_base"] + f"{protocol}.zip"
    if archive.exists():
        if not receipt_path.exists():
            raise RuntimeError(f"archive has no download receipt: {archive}; use parse_protocol_zip for an explicitly supplied local archive")
        receipt = json.loads(receipt_path.read_text())
        actual_hash = _sha(archive)
        if receipt.get("url") != url or receipt.get("sha256") != actual_hash:
            raise RuntimeError(f"cached archive differs from its receipt: {archive}")
        if spec.get("sha256") and actual_hash != spec["sha256"]:
            raise RuntimeError("cached archive differs from the verified source release")
        if archive.stat().st_size != spec["expected_bytes"]:
            raise RuntimeError("cached source archive size differs from the declared release")
        return archive, receipt
    partial = archive.with_suffix(".zip.partial")
    request = urllib.request.Request(url, headers={"User-Agent": "CHEMOLAB-assay-window/1.0", "If-Match": spec["etag"]})
    try:
        with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as f:
            headers = dict(response.headers.items())
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                f.write(chunk)
        if partial.stat().st_size != spec["expected_bytes"]:
            raise RuntimeError("download size differs from the declared NCATS release; inspect updated source before replacing the manifest")
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:
                raise RuntimeError("downloaded ZIP failed CRC validation")
            if f"{protocol}.txt" not in z.namelist():
                raise RuntimeError("NCATS archive lacks the expected raw concentration-response member")
        actual_hash = _sha(partial)
        if spec.get("sha256") and actual_hash != spec["sha256"]:
            raise RuntimeError("download hash differs from the verified NCATS source release")
        receipt = {"protocol": protocol, "url": url, "bytes": partial.stat().st_size,
                   "sha256": actual_hash, "expected_etag": spec["etag"],
                   "response_headers": {k: v for k, v in headers.items()
                                        if k.lower() in {"etag", "last-modified", "content-length", "content-type"}},
                   "source_manifest_sha256": _sha(SOURCE_FILE)}
        os.replace(partial, archive)
        _json(receipt_path, receipt)
        return archive, receipt
    except Exception:
        partial.unlink(missing_ok=True)
        raise


@lru_cache(maxsize=50000)
def standardize_structure(raw_smiles: str) -> dict:
    """Keep full chemistry for provenance and a common parent for grouped splits.

    A parent ID intentionally groups different salts/protonation variants. The
    exact canonical full structure is retained separately; endpoint records are
    never merged because their parent structures match.
    """
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as error:
        raise ImportError("RDKit is required for molecular preparation; install revision_2026/requirements.txt") from error
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("invalid_structure")
    full = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    params = rdMolStandardize.CleanupParameters()
    params.preferOrganic = True
    mol = rdMolStandardize.Cleanup(mol, params)
    mol = rdMolStandardize.FragmentParent(mol, params, skipStandardize=True)
    mol = rdMolStandardize.Uncharger(canonicalOrder=True).uncharge(mol)
    Chem.SanitizeMol(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if not smiles:
        raise ValueError("empty_parent_structure")
    group = hashlib.sha256(f"{STRUCTURE_POLICY}|{smiles}".encode()).hexdigest()
    exact = hashlib.sha256(f"canonical_full_isomeric|{full}".encode()).hexdigest()
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    Chem.RemoveStereochemistry(scaffold)
    Chem.SanitizeMol(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
    return {"smiles": smiles, "standardized_smiles": smiles, "canonical_smiles": smiles,
            "canonical_full_smiles": full, "structure_id": exact,
            "compound_id": group, "group_id": group,
            "scaffold_id": scaffold_smiles or f"ACYCLIC:{group}",
            "structure_policy": STRUCTURE_POLICY, "rdkit_version": rdBase.rdkitVersion}


def _number(raw, field):
    value = "" if raw is None else str(raw).strip()
    if value.lower() in MISSING:
        return None
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"invalid_numeric_{field}") from error
    if not math.isfinite(number):
        return None
    return number


def _curve(row) -> tuple[list[float], list[float | None]]:
    points = []
    for i in range(16):
        c, y = _number(row[f"CONC{i}"], "concentration"), _number(row[f"DATA{i}"], "response")
        if c is None:
            if y is not None:
                raise ValueError("response_without_concentration")
            continue
        if c <= 0:
            raise ValueError("nonpositive_concentration")
        points.append((c, y))
    points.sort(key=lambda item: item[0])
    if len(points) < 2:
        raise ValueError("fewer_than_two_concentrations")
    if any(math.isclose(a[0], b[0], rel_tol=1e-12, abs_tol=0)
           for a, b in zip(points, points[1:])):
        raise ValueError("duplicate_concentration")
    return [x[0] for x in points], [x[1] for x in points]


def _same_doses(a, b):
    return len(a) == len(b) and all(math.isclose(x, y, rel_tol=1e-8, abs_tol=0)
                                  for x, y in zip(a, b))


def _rows(handle):
    reader = csv.DictReader(handle, delimiter="\t")
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)) or not REQUIRED.issubset(fields):
        raise ValueError(f"unexpected NCATS schema; missing {sorted(REQUIRED - set(fields))}")
    for row in reader:
        # The published TSV has one trailing empty field in data rows.
        extras = row.pop(None, None)
        if extras and any(str(v).strip() for v in extras):
            raise ValueError("nonempty overflow fields in NCATS TSV")
        if any(v is None for v in row.values()):
            raise ValueError("truncated NCATS TSV record")
        yield row


def parse_protocol_zip(path: Path, protocol: str, purity=("A", "B"),
                       standardizer: Callable[[str], Mapping] | None = None) -> tuple[list[dict], dict]:
    """Parse one verified protocol ZIP into paired records and exclusion counts.

    ``purity=None`` retains all purity ratings for a declared sensitivity run.
    ``standardizer`` is an injection point for parser tests, not an alternate
    default chemistry pipeline. No assay outcomes are used to retain a pair.
    """
    spec = _spec(protocol)
    standardizer = standardize_structure if standardizer is None else standardizer
    accepted = None if purity is None else {str(p).upper() for p in purity}
    if accepted is not None and not accepted:
        raise ValueError("purity must be None or a nonempty collection")
    pairs = defaultdict(lambda: {"target": [], "viability": []})
    types, row_reasons = Counter(), Counter()
    curve_keys = defaultdict(set)
    row_count = 0
    with zipfile.ZipFile(path) as z:
        member = f"{protocol}.txt"
        if member not in z.namelist():
            raise ValueError(f"raw member absent: {member}")
        with z.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as f:
            for row in _rows(f):
                row_count += 1
                kind = row["SAMPLE_DATA_TYPE"].strip()
                types[kind] += 1
                if row["PROTOCOL_NAME"].strip() != protocol:
                    row_reasons["other_protocol"] += 1
                    continue
                match = re.fullmatch(spec["target_type_regex"], kind)
                channel = "target"
                if match is None:
                    match = re.fullmatch(spec["viability_type_regex"], kind)
                    channel = "viability"
                if match is None:
                    row_reasons["nonprimary_channel"] += 1
                    continue
                sample, curve = row["SAMPLE_ID"].strip(), row["SAMPLE_DATA_ID"].strip()
                if not sample or not curve:
                    row_reasons["missing_sample_or_curve_id"] += 1
                    continue
                key = (sample, int(match.group(1)))
                pairs[key][channel].append(row)
                curve_keys[curve].add((key, channel))
    reused = {cid for cid, keys in curve_keys.items() if len(keys) > 1}
    records, excluded, reasons = [], [], Counter()
    structures = {}
    for (sample, replicate), pair in sorted(pairs.items()):
        why = []
        for kind in ("target", "viability"):
            if len(pair[kind]) == 0:
                why.append(f"missing_{kind}_channel")
            elif len(pair[kind]) != 1:
                why.append(f"duplicate_{kind}_channel")
        if not why:
            target, viability = pair["target"][0], pair["viability"][0]
            if target["SAMPLE_DATA_ID"] in reused or viability["SAMPLE_DATA_ID"] in reused:
                why.append("curve_id_reused_across_pairs")
            ratings = {k: pair[k][0]["PURITY_RATING"].strip().upper() for k in pair}
            if accepted is not None and any(v not in accepted for v in ratings.values()):
                why.append("purity_not_in_declared_set")
            if target["SMILES"].strip() != viability["SMILES"].strip():
                why.append("paired_raw_structure_mismatch")
            try:
                tx, ty = _curve(target)
                vx, vy = _curve(viability)
                if not _same_doses(tx, vx):
                    why.append("paired_concentration_mismatch")
            except ValueError as error:
                why.append(str(error))
            if not why:
                raw_smiles = target["SMILES"].strip()
                try:
                    if raw_smiles not in structures:
                        structures[raw_smiles] = dict(standardizer(raw_smiles))
                    structure = structures[raw_smiles]
                    if not {"compound_id", "group_id", "structure_id", "scaffold_id", "smiles"}.issubset(structure):
                        raise ValueError("standardizer_missing_identity_fields")
                except ImportError:
                    raise
                except Exception as error:
                    why.append("invalid_structure:" + type(error).__name__)
        if why:
            reasons.update(why)
            excluded.append({"sample_id": sample, "replicate": replicate, "reasons": why})
            continue
        official = {k: {name: row.get(name, "") for name in
                        ("ASSAY_OUTCOME", "CHANNEL_OUTCOME", "CURVE_CLASS2", "AC50", "EFFICACY", "R2")}
                    for k, row in (("target", target), ("viability", viability))}
        records.append({**structure, "sample_id": sample, "protocol": protocol, "replicate": replicate,
                        "target_curve_id": target["SAMPLE_DATA_ID"],
                        "viability_curve_id": viability["SAMPLE_DATA_ID"],
                        "raw_smiles": raw_smiles, "tox21_id": target.get("TOX21_ID", ""),
                        "concentrations_m": tx, "target_responses_pct": ty, "viability_responses_pct": vy,
                        "target_direction": spec["target_direction"], "viability_direction": spec["viability_direction"],
                        "purity": ratings["target"] if ratings["target"] == ratings["viability"] else "mixed",
                        "purity_by_channel": ratings,
                        "purity_4m_by_channel": {"target": target.get("PURITY_RATING_4M", ""),
                                                 "viability": viability.get("PURITY_RATING_4M", "")},
                        "official_outcomes": official,
                        "missing_target_responses": sum(y is None for y in ty),
                        "missing_viability_responses": sum(y is None for y in vy)})
    counts = Counter(r["group_id"] for r in records)
    diagnostics = {"protocol": protocol, "raw_rows": row_count, "channel_counts": dict(types),
                   "row_exclusions": dict(row_reasons), "candidate_pairs": len(pairs),
                   "retained_pairs": len(records), "retained_samples": len({r["sample_id"] for r in records}),
                   "retained_compound_groups": len(counts), "excluded_pairs": len(excluded),
                   "pair_exclusion_reasons": dict(reasons), "exclusions": excluded,
                   "retained_replicate_counts": dict(Counter(str(r["replicate"]) for r in records)),
                   "pairs_per_compound_distribution": dict(Counter(str(n) for n in counts.values())),
                   "purity_filter": sorted(accepted) if accepted is not None else None,
                   "pairing_rule": "protocol + sample ID + documented channel replicate index; concentration grids checked",
                   "concentration_unit": "M", "response_unit": "percent",
                   "structure_policy": STRUCTURE_POLICY,
                   "outcome_labels_used_for_selection": False,
                   "raw_archive_sha256": _sha(Path(path))}
    return records, diagnostics


def prepare_protocol(protocol: str, cache_root: Path, purity=("A", "B")) -> tuple[list[dict], dict]:
    archive, receipt = download_protocol(protocol, Path(cache_root))
    records, diagnostics = parse_protocol_zip(archive, protocol, purity=purity)
    diagnostics["download"] = receipt
    return records, diagnostics
