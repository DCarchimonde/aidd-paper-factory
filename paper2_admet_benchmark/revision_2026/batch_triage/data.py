"""Source-only score construction and a query-only finite-pool label oracle.

Public benchmark labels are already available to the researchers.  Keeping them
in the simulator is an information-flow contract, not a claim of prospective
blinding.  A decision algorithm receives ``public_view`` and ``oracle.query``;
the orchestrator alone retains the case and final-evaluation capability.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
ROOT = REVISION.parents[1]
SCHEMA_VERSION = "finite_pool_scores_v1"
PREDICTORS = ("logistic", "equal_logit", "stack")
ENDPOINTS = ("AMES", "hERG", "DILI", "CYP2D6_Veith")
SPLITS = ("random", "scaffold")
SOURCE_ROLES = ("fit", "router", "conformal")
ROW_FIELDS = ("sample_id", "structure_id", "standardized_smiles")


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _runtime_versions():
    result = {}
    for name in ("numpy", "scipy", "scikit-learn", "rdkit", "PyYAML"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not_installed"
    return result


def _training_dependencies():
    # Audit algorithms and report formatting are deliberately absent: changing
    # them creates a new decision run, without invalidating source-only scores.
    paths = [Path(__file__), REVISION / "runner.py",
             ROOT / "paper2_admet_benchmark/scripts/racer_c4/run_prospective_racer_c4.py",
             ROOT / "paper2_admet_benchmark/scripts/racer_c4/racer_c4_io.py",
             ROOT / "paper2_admet_benchmark/scripts/racer_c4/racer_c4_core.py",
             ROOT / "paper2_admet_benchmark/configs/racer_c4/prospective_lock_v1.yaml"]
    return {str(p.relative_to(ROOT)): _file_hash(p) for p in paths}


def case_specs(endpoints=None, splits=None, fit_seed=311, include_epa=False):
    """Eight public target pools by default; no audit-seed training multiplier."""
    endpoints = list(ENDPOINTS if endpoints is None else endpoints)
    splits = list(SPLITS if splits is None else splits)
    if not endpoints or len(set(endpoints)) != len(endpoints) or set(endpoints) - set(ENDPOINTS):
        raise ValueError("choose unique declared independent endpoints")
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(SPLITS):
        raise ValueError("choose unique random/scaffold split regimes")
    if int(fit_seed) != fit_seed or int(fit_seed) not in range(311, 316):
        raise ValueError("independent source loader permits fit seeds 311 through 315")
    result = [dict(phase=f"independent_{split}", endpoint=endpoint, split=split,
                   fit_seed=int(fit_seed)) for split in splits for endpoint in endpoints]
    if include_epa:
        from ..runner import ENDPOINT_PROPERTIES
        result += [dict(phase="retrospective_epa", endpoint=endpoint, split="historical",
                        fit_seed=int(fit_seed)) for endpoint in ENDPOINT_PROPERTIES]
    return result


def _load_raw_case(spec, cache_root):
    from .. import independent_data, runner
    if spec["phase"] in ("independent_random", "independent_scaffold"):
        return independent_data.load_case(Path(cache_root), spec["endpoint"], spec["split"], spec["fit_seed"])
    if spec["phase"] != "retrospective_epa":
        raise ValueError("unsupported pool phase")
    import yaml
    lock = yaml.safe_load(runner.LEGACY_LOCK.read_text())
    sources, processed = runner.prepare(Path(cache_root) / "tox21", lock)
    source_rows, source_y = runner.read_clean_endpoint(processed / f"{spec['endpoint']}_clean.csv")
    targets, labels, label_path = runner.load_target("retrospective_epa", sources, lock, Path(cache_root) / "tox21")
    return {"source_rows": source_rows, "source_y": source_y, "target_rows": targets,
            "labels": labels, "input_hashes": {"target_labels": _file_hash(label_path),
             "training": _file_hash(processed / f"{spec['endpoint']}_clean.csv")},
            "split_metadata": {"status": "previously_observed_EPA_retrospective_reanalysis"}}


def _source_roles(y, seed):
    from ..runner import split_internal_roles
    return split_internal_roles(np.asarray(y, dtype=int), seed)


def _build_score_arrays(source_rows, source_y, target_rows, spec, predictors, jobs):
    """This function has no target-outcome argument."""
    import yaml
    from .. import runner
    lock = yaml.safe_load(runner.LEGACY_LOCK.read_text())
    x, qx = runner.fingerprints(source_rows), runner.fingerprints(target_rows)
    roles, fitted = runner.fit_predictors(x, np.asarray(source_y, dtype=int), qx,
        lock, spec["fit_seed"], jobs, predictors)
    support = runner.max_tanimoto(x[roles[0]], qx)
    scores = {name: fitted[name][1] for name in predictors}
    return roles, scores, support


def _public_rows(rows):
    return [{key: str(row[key]) for key in ROW_FIELDS} for row in rows]


def _eligible_pool(raw, endpoint):
    source_ids = {str(row["structure_id"]) for row in raw["source_rows"]}
    targets, labels, exclusions = [], [], []
    ids = set()
    for row in raw["target_rows"]:
        sample_id = str(row["sample_id"])
        if sample_id in ids:
            raise ValueError("duplicate target sample_id")
        ids.add(sample_id)
        if sample_id not in raw["labels"] or endpoint not in raw["labels"][sample_id]:
            raise ValueError("target labels must contain every target ID, using null/NaN for missing outcomes")
        value = raw["labels"][sample_id][endpoint]
        missing = value is None or not math.isfinite(float(value))
        if not missing and float(value) not in (0., 1.):
            raise ValueError("target outcomes must be binary or explicitly missing")
        reasons = []
        if row.get("structure_status", "pass") != "pass":
            reasons.append("invalid_structure")
        if str(row["structure_id"]) in source_ids:
            reasons.append("source_structure_overlap")
        if missing:
            reasons.append("unobserved_endpoint_outcome")
        if reasons:
            exclusions.append({"sample_id": sample_id, "reasons": reasons})
        else:
            targets.append(row)
            labels.append(int(value))
    if not targets:
        raise ValueError("no eligible fully observed target records")
    # Ordering is covariate/identity-only; no retry or class-balance selection.
    order = sorted(range(len(targets)), key=lambda i: str(targets[i]["sample_id"]))
    return [targets[i] for i in order], np.asarray([labels[i] for i in order], dtype=np.int8), exclusions


@dataclass(frozen=True)
class ManifestImport:
    """Optional known-source transfer, bound to the exact computed case identity.

    ``identity_sha256`` must come from an audited score export, not an inferred
    match based on endpoint names or metrics. Older review bundles contain no
    target manifests and cannot satisfy this input contract.
    """
    paths: Mapping[str, Path]
    sha256: Mapping[str, str]
    identity_sha256: str


def _import_scores(bundle, identity_sha256, target_rows, spec, predictors):
    if bundle.identity_sha256 != identity_sha256:
        raise ValueError("imported scores belong to a different data/roles/code/environment identity")
    if set(bundle.paths) != set(predictors) or set(bundle.sha256) != set(predictors):
        raise ValueError("import must contain exactly the requested predictor set")
    expected = {row["sample_id"]: row for row in _public_rows(target_rows)}
    scores, reference_support = {}, None
    for model in predictors:
        path = Path(bundle.paths[model])
        if _file_hash(path) != bundle.sha256[model]:
            raise ValueError("imported target manifest checksum differs")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if {"y", "target", "label"}.intersection(reader.fieldnames or []):
                raise ValueError("score manifests must not contain target labels")
            rows = list(reader)
        by_id = {row["sample_id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != set(expected):
            raise ValueError("imported target IDs are duplicated, missing or extra")
        p, support = [], []
        for sample_id, wanted in expected.items():
            row = by_id[sample_id]
            if any(row[k] != wanted[k] for k in ROW_FIELDS):
                raise ValueError("imported molecular identity differs")
            if row["model"] != model or int(row["seed"]) != spec["fit_seed"] or row["endpoint"] != spec["endpoint"] or row["phase"] != spec["phase"]:
                raise ValueError("imported predictor/task identity differs")
            if int(row["eligible"]) != 1:
                raise ValueError("imported scores must describe the exact eligible pool")
            p.append(float(row["probability_1"]))
            support.append(float(row["max_train_tanimoto"]))
        scores[model] = np.asarray(p)
        support = np.asarray(support)
        if reference_support is not None and not np.array_equal(reference_support, support):
            raise ValueError("predictors disagree about their common fit-only support reference")
        reference_support = support
    return scores, reference_support


class FrozenLabelOracle:
    """One-label access with a fresh ledger for each simulated audit run."""
    def __init__(self, sample_ids, labels):
        self._ids = tuple(str(s) for s in sample_ids)
        values = np.asarray(labels)
        if not self._ids or len(set(self._ids)) != len(self._ids) or values.shape != (len(self._ids),) or not np.isin(values, [0, 1]).all():
            raise ValueError("oracle requires unique IDs and aligned observed binary labels")
        self._labels = tuple(int(x) for x in values)
        self._indices = {name: i for i, name in enumerate(self._ids)}
        self._events = []
        self._queried = set()

    @property
    def query_count(self):
        return len(self._events)

    @property
    def ledger(self):
        return [dict(event) for event in self._events]

    def query(self, sample):
        if isinstance(sample, (int, np.integer)) and not isinstance(sample, (bool, np.bool_)):
            index = int(sample)
            if index < 0 or index >= len(self._ids):
                raise KeyError("target index outside the frozen pool")
        else:
            if str(sample) not in self._indices:
                raise KeyError("unknown target sample_id")
            index = self._indices[str(sample)]
        if index in self._queried:
            raise ValueError("without-replacement oracle forbids a second query of the same record")
        label = self._labels[index]
        self._queried.add(index)
        self._events.append({"query_number": len(self._events) + 1, "index": index,
            "sample_id": self._ids[index], "y": label, "label_cost": 1,
            "label_role": "audit", "queried_at_ns": time.time_ns()})
        return label

    def evaluate(self, actions):
        """Trusted final evaluation only; audit queries must remain human-assayed.

        Codes are 0=clear, 1=discard, 2=assay; named equivalents are accepted.
        Unqueried assay decisions count as subsequent laboratory work. The
        finite-batch class denominators include every frozen pool member.
        """
        names = {"clear": 0, "discard": 1, "assay": 2, "defer": 2}
        values = np.asarray([names.get(x, x) if isinstance(x, str) else x for x in actions])
        if values.shape != (len(self._ids),) or not np.isin(values, [0, 1, 2]).all():
            raise ValueError("final actions must align with the pool and be clear/discard/assay")
        if any(values[i] != 2 for i in self._queried):
            raise ValueError("audit-queried records cannot also be counted as assay-saving releases")
        y = np.asarray(self._labels)
        active, inactive = int((y == 1).sum()), int((y == 0).sum())
        fn, fp = int(((values == 0) & (y == 1)).sum()), int(((values == 1) & (y == 0)).sum())
        human = int((values == 2).sum())
        return {"n": len(y), "n_active": active, "n_inactive": inactive,
            "n_audit_queries": self.query_count, "n_remaining_assays": human - self.query_count,
            "n_total_assays": human, "n_auto_clear": int((values == 0).sum()),
            "n_auto_discard": int((values == 1).sum()), "assay_savings": 1 - human / len(y),
            "false_clearance_n": fn, "false_discard_n": fp,
            "false_clearance": fn / active if active else 0.,
            "false_discard": fp / inactive if inactive else 0.,
            "active_risk_vacuous": active == 0, "inactive_risk_vacuous": inactive == 0,
            "risk_scope": "fixed_fully_observed_target_batch_including_human_audited_records"}


@dataclass(frozen=True)
class PreparedCase:
    case_id: str
    rows: tuple
    scores: Mapping[str, np.ndarray]
    support: np.ndarray
    metadata: Mapping
    cache_path: Path
    _oracle_labels: tuple

    def public_view(self, predictor="stack"):
        if predictor not in self.scores:
            raise KeyError("predictor was not built in this score bundle")
        score, support = self.scores[predictor].copy(), self.support.copy()
        score.setflags(write=False)
        support.setflags(write=False)
        return {"case_id": self.case_id, "predictor": predictor,
                "rows": [dict(row) for row in self.rows], "score": score,
                "max_source_tanimoto": support,
                "metadata": json.loads(json.dumps(self.metadata))}

    def new_oracle(self):
        return FrozenLabelOracle([row["sample_id"] for row in self.rows], self._oracle_labels)

    def manifest_import(self):
        """Small transfer descriptor; outcomes are reloaded by the simulator."""
        receipt = json.loads((self.cache_path / "complete.json").read_text())
        return ManifestImport({name: self.cache_path / f"target_manifest_{name}.csv" for name in self.scores},
            {name: receipt["files"][f"target_manifest_{name}.csv"] for name in self.scores},
            receipt["identity_sha256"])


def _load_completed(folder, identity_sha256):
    receipt = json.loads((folder / "complete.json").read_text())
    if receipt["identity_sha256"] != identity_sha256:
        raise RuntimeError("score cache identity differs")
    for name, digest in receipt["files"].items():
        if not (folder / name).is_file() or _file_hash(folder / name) != digest:
            raise RuntimeError("score cache artifact is missing or changed: " + name)
    metadata = json.loads((folder / "metadata.json").read_text())
    rows = tuple(json.loads((folder / "target_rows.json").read_text()))
    private = json.loads((folder / "oracle_labels.json").read_text())
    if private["sample_ids"] != [row["sample_id"] for row in rows]:
        raise RuntimeError("oracle rows differ from score rows")
    labels = tuple(private["y"])
    with np.load(folder / "scores.npz", allow_pickle=False) as saved:
        scores = {name: saved["score_" + name].copy() for name in metadata["predictors"]}
        support = saved["support"].copy()
    return PreparedCase(metadata["case_id"], rows, scores, support, metadata, folder, labels)


def prepare_case(spec, cache_root, score_cache_root, *, predictors=PREDICTORS, jobs=1,
                 score_import: ManifestImport | None = None):
    """Load public data, train once or verify reusable scores, and freeze an oracle.

    Audit RNG seeds, risk caps, strata choices and budgets are not score-cache
    keys and cannot trigger refitting. This function performs source training;
    tests mock that boundary and do not run large estimators.
    """
    spec = {key: spec[key] for key in ("phase", "endpoint", "split", "fit_seed")}
    if spec["fit_seed"] != int(spec["fit_seed"]) or int(spec["fit_seed"]) not in range(311, 316):
        raise ValueError("source fit seed must be one of 311 through 315")
    spec["fit_seed"] = int(spec["fit_seed"])
    if spec["phase"].startswith("independent_"):
        if spec["endpoint"] not in ENDPOINTS or spec["split"] not in SPLITS or spec["phase"] != f"independent_{spec['split']}":
            raise ValueError("independent case phase/endpoint/split disagree")
    elif spec["phase"] != "retrospective_epa" or spec["split"] != "historical":
        raise ValueError("unsupported case specification")
    requested = tuple(predictors)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(PREDICTORS):
        raise ValueError("predictors must be unique supported model names")
    predictors = tuple(name for name in PREDICTORS if name in requested)
    if jobs < 1 or int(jobs) != jobs:
        raise ValueError("request at least one supported predictor and a positive CPU limit")
    raw = _load_raw_case(spec, Path(cache_root))
    source_rows, y = raw["source_rows"], np.asarray(raw["source_y"], dtype=int)
    source_ids = [str(row["structure_id"]) for row in source_rows]
    if len(set(source_ids)) != len(source_ids) or y.shape != (len(source_rows),) or not np.isin(y, [0, 1]).all():
        raise ValueError("source data require unique structures and aligned binary outcomes")
    targets, oracle_y, exclusions = _eligible_pool(raw, spec["endpoint"])
    public_rows = _public_rows(targets)
    roles = tuple(np.asarray(x, dtype=int) for x in _source_roles(y, spec["fit_seed"]))
    if len(roles) != 3 or sorted(np.concatenate(roles).tolist()) != list(range(len(y))):
        raise ValueError("source fit/router/conformal roles must partition source rows")
    ledger = [{"structure_id": source_ids[i], "role": role} for role, indices in zip(SOURCE_ROLES, roles) for i in indices]
    case_id = f"{spec['phase']}__{spec['endpoint']}__fit{spec['fit_seed']}"
    identity = {"schema": SCHEMA_VERSION, "case": spec, "predictors": list(predictors),
        "source_rows_sha256": _hash(_public_source_rows(source_rows)),
        "source_y_sha256": _hash(y.tolist()), "source_roles_sha256": _hash(ledger),
        "target_rows_sha256": _hash(public_rows), "oracle_labels_sha256": _hash(oracle_y.tolist()),
        "input_hashes": raw.get("input_hashes", {}), "code": _training_dependencies(),
        "versions": _runtime_versions(), "jobs": int(jobs)}
    digest = _hash(identity)
    folder = Path(score_cache_root).resolve() / case_id / digest
    if (folder / "complete.json").exists():
        return _load_completed(folder, digest)
    if folder.exists() and any(folder.iterdir()):
        folder.rename(folder.with_name(digest + f"_incomplete_{time.time_ns()}"))
    folder.mkdir(parents=True, exist_ok=True)
    if score_import is None:
        returned_roles, scores, support = _build_score_arrays(source_rows, y, public_rows, spec, predictors, int(jobs))
        if len(returned_roles) != 3 or any(not np.array_equal(a, b) for a, b in zip(roles, returned_roles)):
            raise RuntimeError("predictor training changed the frozen source role allocation")
        origin = "fresh_source_only_fit"
    else:
        scores, support = _import_scores(score_import, digest, targets, spec, predictors)
        origin = "known_source_manifest_import"
    support = np.asarray(support, dtype=float)
    scores = {name: np.asarray(scores[name], dtype=float) for name in predictors}
    for values in [support, *scores.values()]:
        if values.shape != (len(targets),) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("scores and fit-only Tanimoto support must be finite aligned values in [0,1]")
    public_split = {key: raw.get("split_metadata", {})[key] for key in
        ("split", "outer_seed", "target_selection_uses_labels", "structure_overlap", "scaffold_overlap")
        if key in raw.get("split_metadata", {})}
    metadata = {"case_id": case_id, **spec, "schema": SCHEMA_VERSION,
        "predictors": list(predictors), "n_source": len(y), "n_pool": len(targets),
        "n_target_original": len(raw["target_rows"]), "n_excluded": len(exclusions),
        "source_role_sizes": {name: len(v) for name, v in zip(SOURCE_ROLES, roles)},
        "source_roles_sha256": _hash(ledger), "target_rows_sha256": _hash(public_rows),
        "split": spec["split"], "outer_split_metadata": public_split,
        "support_reference": "source_fit_only", "risk_population": "eligible_fully_observed_endpoint_records",
        "observed_outcome_eligibility_uses_label_value": False,
        "target_labels_used_for_scores": False, "score_origin": origin,
        "study_status": "retrospective_public_benchmark_simulation_not_new_blinded_validation"}
    _json(folder / "identity.json", identity)
    _json(folder / "metadata.json", metadata)
    _json(folder / "target_rows.json", public_rows)
    _json(folder / "oracle_labels.json", {"sample_ids": [r["sample_id"] for r in public_rows], "y": oracle_y.tolist()})
    _json(folder / "source_roles.json", ledger)
    _json(folder / "exclusions.json", exclusions)
    with (folder / "scores.npz").open("wb") as handle:
        np.savez_compressed(handle, support=support, **{"score_" + name: p for name, p in scores.items()})
    for name in predictors:
        rows = [{**row, "phase": spec["phase"], "endpoint": spec["endpoint"],
                 "seed": spec["fit_seed"], "model": name, "eligible": 1,
                 "probability_1": float(scores[name][i]), "max_train_tanimoto": float(support[i])}
                for i, row in enumerate(public_rows)]
        _csv(folder / f"target_manifest_{name}.csv", rows, list(rows[0]))
    files = {p.name: _file_hash(p) for p in sorted(folder.iterdir()) if p.is_file()}
    _json(folder / "complete.json", {"status": "complete", "identity_sha256": digest, "files": files})
    return _load_completed(folder, digest)


def _public_source_rows(rows):
    return [{"structure_id": str(row["structure_id"]),
             "standardized_smiles": str(row["standardized_smiles"])} for row in rows]
