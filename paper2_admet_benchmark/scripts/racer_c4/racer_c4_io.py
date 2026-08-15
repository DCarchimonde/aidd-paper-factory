from __future__ import annotations

"""Locked data acquisition, parsing, and final-label firewall for RACER-C4."""

import csv
import hashlib
import io
import json
import os
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(payload)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows do not share one stable field order")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def download_locked(url: str, path: Path, expected_sha256: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise RuntimeError(
                f"locked download mismatch for {path}: "
                f"expected={expected_sha256} observed={observed}"
            )
        return path
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"incomplete prior download requires inspection: {partial}")
    request = urllib.request.Request(url, headers={"User-Agent": "RACER-C4/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open(
        "wb"
    ) as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    observed = sha256_file(partial)
    if observed != expected_sha256:
        raise RuntimeError(
            f"download SHA256 mismatch for {url}: "
            f"expected={expected_sha256} observed={observed}"
        )
    os.replace(partial, path)
    return path


def extract_member_locked(
    archive: Path,
    member: str,
    destination: Path,
    expected_sha256: str,
) -> Path:
    if destination.is_file():
        observed = sha256_file(destination)
        if observed != expected_sha256:
            raise RuntimeError(
                f"locked archive member mismatch for {destination}: "
                f"expected={expected_sha256} observed={observed}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        matches = [name for name in names if name == member or Path(name).name == member]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one archive member {member!r}; observed {matches!r}"
            )
        with handle.open(matches[0]) as source, temporary.open("wb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    observed = sha256_file(temporary)
    if observed != expected_sha256:
        raise RuntimeError(
            f"archive member SHA256 mismatch for {member}: "
            f"expected={expected_sha256} observed={observed}"
        )
    os.replace(temporary, destination)
    return destination


def acquire_unlabeled_sources(
    lock: Mapping[str, object], source_root: Path
) -> dict[str, Path]:
    """Acquire every allowed source except the final label file."""

    sources = lock["data_sources"]
    training = sources["training_archive"]
    development = sources["leaderboard_development_labels"]
    final_structures = sources["final_epa_structures"]
    training_archive = download_locked(
        str(training["url"]),
        source_root / str(training["filename"]),
        str(training["sha256"]),
    )
    training_sdf = extract_member_locked(
        training_archive,
        str(training["member"]),
        source_root / "extracted" / str(training["member"]),
        str(training["member_sha256"]),
    )
    development_archive = download_locked(
        str(development["url"]),
        source_root / str(development["filename"]),
        str(development["sha256"]),
    )
    development_sdf = extract_member_locked(
        development_archive,
        str(development["member"]),
        source_root / "extracted" / str(development["member"]),
        str(development["member_sha256"]),
    )
    final_structure_path = download_locked(
        str(final_structures["url"]),
        source_root / str(final_structures["filename"]),
        str(final_structures["sha256"]),
    )
    return {
        "training_archive": training_archive,
        "training_sdf": training_sdf,
        "development_sdf": development_sdf,
        "final_structures": final_structure_path,
    }


def acquire_final_label_bytes(
    lock: Mapping[str, object], source_root: Path
) -> Path:
    source = lock["data_sources"]["final_epa_labels"]
    return download_locked(
        str(source["url"]),
        source_root / str(source["filename"]),
        str(source["sha256"]),
    )


def read_clean_endpoint(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"structure_id", "standardized_smiles", "target"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"invalid cleaned endpoint table: {path}")
    target = np.asarray([int(row["target"]) for row in rows], dtype=np.int8)
    if np.any((target != 0) & (target != 1)):
        raise RuntimeError(f"non-binary cleaned target in {path}")
    return rows, target


def _standardize(raw_smiles: str) -> tuple[str, str, str]:
    from prepare_classification_endpoint import standardize_smiles

    return standardize_smiles(raw_smiles)


def read_structure_table(path: Path) -> list[dict[str, str]]:
    """Read a public SMILES/Sample-ID table without consulting any labels."""

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value or value.lower().startswith("#smiles"):
                continue
            fields = value.split("\t")
            if len(fields) != 2:
                fields = value.rsplit(None, 1)
            if len(fields) != 2:
                raise RuntimeError(f"malformed structure row {line_number} in {path}")
            raw_smiles, sample_id = (field.strip() for field in fields)
            if not raw_smiles or not sample_id or sample_id in seen_ids:
                raise RuntimeError(f"blank or duplicate sample ID at row {line_number}")
            try:
                standardized_smiles, structure_id, scaffold_id = _standardize(raw_smiles)
                structure_status = "pass"
            except Exception as exc:
                # Preserve the public sample identity and fail closed.  Do not
                # invent a relaxed standardization path after external X is seen.
                standardized_smiles = ""
                structure_id = "invalid-" + sha256_bytes(raw_smiles.encode("utf-8"))
                scaffold_id = ""
                reason = str(exc).strip() or type(exc).__name__
                structure_status = "excluded_" + re.sub(
                    r"[^a-z0-9]+", "_", reason.lower()
                ).strip("_")
            rows.append(
                {
                    "sample_id": sample_id,
                    "raw_smiles": raw_smiles,
                    "standardized_smiles": standardized_smiles,
                    "structure_id": structure_id,
                    "murcko_scaffold_id": scaffold_id,
                    "structure_status": structure_status,
                }
            )
            seen_ids.add(sample_id)
    if not rows:
        raise RuntimeError(f"structure table is empty: {path}")
    return rows


def read_development_sdf(path: Path) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    """Read the explicitly development-only public leaderboard batch."""

    from rdkit import Chem

    rows: list[dict[str, str]] = []
    labels: dict[str, list[float]] = {endpoint: [] for endpoint in ENDPOINT_PROPERTIES}
    supplier = Chem.ForwardSDMolSupplier(str(path), sanitize=False, removeHs=False)
    for position, molecule in enumerate(supplier, start=1):
        if molecule is None:
            raise RuntimeError(f"development SDF parse failure at record {position}")
        candidate = Chem.Mol(molecule)
        Chem.SanitizeMol(candidate)
        raw_smiles = Chem.MolToSmiles(
            candidate, canonical=False, isomericSmiles=True
        )
        standardized_smiles, structure_id, scaffold_id = _standardize(raw_smiles)
        sample_id = (
            molecule.GetProp("Compound ID").strip()
            if molecule.HasProp("Compound ID")
            else f"development-{position}"
        )
        rows.append(
            {
                "sample_id": sample_id,
                "raw_smiles": raw_smiles,
                "standardized_smiles": standardized_smiles,
                "structure_id": structure_id,
                "murcko_scaffold_id": scaffold_id,
                "structure_status": "pass",
            }
        )
        for endpoint, property_name in ENDPOINT_PROPERTIES.items():
            raw = (
                molecule.GetProp(property_name).strip()
                if molecule.HasProp(property_name)
                else ""
            )
            labels[endpoint].append(float(raw) if raw in {"0", "1"} else np.nan)
    if not rows:
        raise RuntimeError("development SDF is empty")
    return rows, {key: np.asarray(value, dtype=float) for key, value in labels.items()}


def fingerprints(rows: Sequence[Mapping[str, str]], n_bits: int = 2048) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=int(n_bits))
    output = np.empty((len(rows), int(n_bits)), dtype=np.uint8)
    for index, row in enumerate(rows):
        molecule = Chem.MolFromSmiles(str(row["standardized_smiles"]))
        if molecule is None:
            raise RuntimeError(f"standardized structure failed at row {index}")
        output[index] = generator.GetFingerprintAsNumPy(molecule)
    return output


def physchem_features(rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    """Nine deterministic, interpretable RDKit transport descriptors."""

    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    output = np.empty((len(rows), 9), dtype=float)
    for index, row in enumerate(rows):
        molecule = Chem.MolFromSmiles(str(row["standardized_smiles"]))
        if molecule is None:
            raise RuntimeError(f"standardized structure failed at row {index}")
        output[index] = [
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.RingCount(molecule),
            rdMolDescriptors.CalcFractionCSP3(molecule),
            Lipinski.HeavyAtomCount(molecule),
        ]
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite physicochemical descriptor")
    return output


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_final_label_text(
    text: str,
    endpoint_order: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Parse the official final result table after the promotion firewall opens."""

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("final label file is empty")
    delimiter = "\t" if "\t" in lines[0] else "," if "," in lines[0] else None
    if delimiter is None:
        parsed = [re.split(r"\s+", line.strip()) for line in lines]
    else:
        parsed = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter))
    aliases = {
        _normalize_header(endpoint): endpoint for endpoint in endpoint_order
    }
    aliases.update(
        {
            _normalize_header(ENDPOINT_PROPERTIES[endpoint]): endpoint
            for endpoint in endpoint_order
        }
    )
    first = [_normalize_header(value) for value in parsed[0]]
    has_header = any(value in aliases for value in first) or any(
        "sample" in value or value in {"id", "compoundid"} for value in first
    )
    if has_header:
        header = parsed.pop(0)
        normalized = [_normalize_header(value) for value in header]
        id_candidates = [
            index
            for index, value in enumerate(normalized)
            if "sample" in value or value in {"id", "compoundid"}
        ]
        if len(id_candidates) != 1:
            raise RuntimeError("final label header has no unique sample-ID column")
        id_index = id_candidates[0]
        endpoint_indices: dict[str, int] = {}
        for index, value in enumerate(normalized):
            if value in aliases:
                endpoint_indices[aliases[value]] = index
        missing = set(endpoint_order) - set(endpoint_indices)
        if missing:
            raise RuntimeError(f"final label header misses endpoints: {sorted(missing)}")
    else:
        if len(parsed[0]) != len(endpoint_order) + 1:
            raise RuntimeError("headerless final label table has an unexpected width")
        id_index = 0
        endpoint_indices = {
            endpoint: index + 1 for index, endpoint in enumerate(endpoint_order)
        }
    output: dict[str, dict[str, float]] = {}
    missing_tokens = {"", "na", "nan", "null", "x", "-"}
    required_width = max([id_index, *endpoint_indices.values()]) + 1
    for row_number, fields in enumerate(parsed, start=2 if has_header else 1):
        if len(fields) < required_width:
            raise RuntimeError(f"short final label row {row_number}")
        sample_id = fields[id_index].strip()
        if not sample_id or sample_id in output:
            raise RuntimeError(f"blank or duplicate final sample ID at row {row_number}")
        values: dict[str, float] = {}
        for endpoint, index in endpoint_indices.items():
            raw = fields[index].strip().lower()
            if raw in missing_tokens:
                values[endpoint] = float("nan")
            elif raw in {"0", "0.0"}:
                values[endpoint] = 0.0
            elif raw in {"1", "1.0"}:
                values[endpoint] = 1.0
            else:
                raise RuntimeError(
                    f"invalid final label {raw!r} for {endpoint} at row {row_number}"
                )
        output[sample_id] = values
    if not output:
        raise RuntimeError("final label table contains no data rows")
    return output


def open_final_labels_after_promotion(
    label_path: Path,
    promotion_record_path: Path,
    expected_label_sha256: str,
    endpoint_order: Sequence[str],
) -> dict[str, dict[str, float]]:
    """The only authorized function that interprets final EPA labels."""

    if not promotion_record_path.is_file():
        raise PermissionError("final labels are sealed until a promotion record exists")
    promotion = json.loads(promotion_record_path.read_text(encoding="utf-8"))
    required = {
        "status": "predictions_sealed_before_final_labels",
        "development_gate_passed": True,
        "final_labels_opened": False,
        "expected_final_label_sha256": expected_label_sha256,
    }
    failures = [key for key, value in required.items() if promotion.get(key) != value]
    prediction_hash = str(promotion.get("sealed_predictions_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", prediction_hash):
        failures.append("sealed_predictions_sha256")
    if failures:
        raise PermissionError("promotion record contract failed: " + ", ".join(failures))
    observed = sha256_file(label_path)
    if observed != expected_label_sha256:
        raise RuntimeError(
            f"final label SHA256 mismatch: expected={expected_label_sha256} "
            f"observed={observed}"
        )
    # Do not move this read above the promotion and hash checks.
    text = label_path.read_text(encoding="utf-8-sig")
    return parse_final_label_text(text, endpoint_order)
