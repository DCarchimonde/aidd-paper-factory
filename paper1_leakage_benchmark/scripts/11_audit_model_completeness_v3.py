from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.modeling_v3 import model_names, production_model_seed

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
FROZEN_DIR = PAPER_DIR / "results" / "frozen_v3"
OUT_DIR = PAPER_DIR / "results" / "model_rerun_v3"
JOB_DIR = OUT_DIR / "jobs" / "production"
PROTOCOL_PATH = PAPER_DIR / "MODEL_PROTOCOL_V3.md"
PARTITION_REGISTRY = FROZEN_DIR / "frozen_partition_registry_v3.csv"

SPECS = {
    "main_classification": {
        "task_type": "classification",
        "datasets": ("BACE", "BBBP", "ClinTox", "HIV"),
        "protocols": (
            "legacy_scaffold",
            "random_observation",
            "size_matched_scaffold",
            "target_balanced_scaffold",
        ),
        "suffix": "BACE-BBBP-ClinTox-HIV_single_group_20s_300c",
    },
    "main_regression": {
        "task_type": "regression",
        "datasets": ("ESOL", "FreeSolv"),
        "protocols": (
            "legacy_scaffold",
            "random_observation",
            "size_matched_scaffold",
            "target_balanced_scaffold",
        ),
        "suffix": "ESOL-FreeSolv_single_group_20s_20000c",
    },
    "acyclic_singleton_sensitivity": {
        "task_type": "regression",
        "datasets": ("ESOL", "FreeSolv"),
        "protocols": (
            "size_matched_scaffold",
            "target_balanced_scaffold",
        ),
        "suffix": "ESOL-FreeSolv_singleton_20s_5000c",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_seed(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def collect_jobs() -> pd.DataFrame:
    rows: list[dict] = []
    if not JOB_DIR.exists():
        return pd.DataFrame()
    for path in JOB_DIR.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["job_file"] = str(path.relative_to(ROOT))
        rows.append(payload)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["partition_seed_key"] = frame["partition_seed"].map(normalize_seed)
        frame["model_seed_key"] = frame["model_seed"].astype(int)
    return frame


def primary_metric(task_type: str) -> str:
    return "roc_auc" if task_type == "classification" else "rmse"


def main() -> None:
    if not PARTITION_REGISTRY.exists():
        raise FileNotFoundError(PARTITION_REGISTRY)
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)

    registry = pd.read_csv(
        PARTITION_REGISTRY,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    registry["partition_seed_key"] = registry["partition_seed"].map(normalize_seed)
    jobs = collect_jobs()
    if jobs.empty:
        raise RuntimeError("No production model jobs found")

    current_protocol_sha = sha256_file(PROTOCOL_PATH)
    detail_rows: list[dict] = []

    for freeze_label, spec in SPECS.items():
        task_type = str(spec["task_type"])
        manifest_path = (
            FROZEN_DIR
            / freeze_label
            / f"split_manifest_v3_{spec['suffix']}.csv"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest_sha = sha256_file(manifest_path)

        selected_registry = registry.loc[
            registry["freeze_label"].eq(freeze_label)
            & registry["dataset"].isin(spec["datasets"])
            & registry["protocol"].isin(spec["protocols"])
        ].copy()

        for part in selected_registry.itertuples(index=False):
            for model in model_names(task_type):
                expected_seed = int(production_model_seed(model))
                match = jobs.loc[
                    jobs["freeze_label"].eq(freeze_label)
                    & jobs["dataset"].eq(str(part.dataset))
                    & jobs["protocol"].eq(str(part.protocol))
                    & jobs["partition_seed_key"].eq(str(part.partition_seed_key))
                    & jobs["partition_hash"].astype(str).eq(str(part.partition_hash))
                    & jobs["model"].eq(model)
                    & jobs["model_seed_key"].eq(expected_seed)
                ].copy()
                status = "complete"
                reason = ""
                if len(match) == 0:
                    status = "missing"
                    reason = "job_not_found"
                elif len(match) > 1:
                    status = "invalid"
                    reason = "duplicate_job_key"
                else:
                    row = match.iloc[0]
                    if str(row.get("model_protocol_sha256", "")) != current_protocol_sha:
                        status = "stale"
                        reason = "model_protocol_sha256_mismatch"
                    elif str(row.get("frozen_manifest_sha256", "")) != manifest_sha:
                        status = "stale"
                        reason = "frozen_manifest_sha256_mismatch"
                    elif int(row.get("n_train", -1)) != int(part.n_train):
                        status = "invalid"
                        reason = "n_train_mismatch"
                    elif int(row.get("n_test", -1)) != int(part.n_test):
                        status = "invalid"
                        reason = "n_test_mismatch"
                    else:
                        metric = primary_metric(task_type)
                        value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
                        if not np.isfinite(value):
                            status = "invalid"
                            reason = f"nonfinite_{metric}"

                detail_rows.append(
                    {
                        "freeze_label": freeze_label,
                        "dataset": str(part.dataset),
                        "task_type": task_type,
                        "protocol": str(part.protocol),
                        "partition_seed": str(part.partition_seed_key),
                        "partition_hash": str(part.partition_hash),
                        "model": model,
                        "expected_model_seed": expected_seed,
                        "status": status,
                        "reason": reason,
                        "n_matching_jobs": int(len(match)),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    expected_total = 1338
    if len(detail) != expected_total:
        raise AssertionError(
            f"Internal expected-job plan changed: {len(detail)} rows, expected {expected_total}"
        )

    summary = (
        detail.groupby(
            ["freeze_label", "dataset", "task_type", "protocol", "model"],
            as_index=False,
        )
        .agg(
            expected_jobs=("status", "size"),
            complete_jobs=("status", lambda s: int((s == "complete").sum())),
            missing_jobs=("status", lambda s: int((s == "missing").sum())),
            stale_jobs=("status", lambda s: int((s == "stale").sum())),
            invalid_jobs=("status", lambda s: int((s == "invalid").sum())),
            unique_partition_hashes=("partition_hash", "nunique"),
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUT_DIR / "model_completeness_detail_v3.csv"
    summary_path = OUT_DIR / "model_completeness_summary_v3.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nProduction model completeness summary:")
    print(summary.to_string(index=False))
    totals = detail["status"].value_counts().to_dict()
    print("\nOverall expected jobs:", expected_total)
    print("Status counts:", totals)
    print("\nSaved:")
    print(detail_path)
    print(summary_path)

    bad = detail.loc[detail["status"].ne("complete")]
    if not bad.empty:
        print("\nIncomplete/stale jobs (first 50):")
        print(bad.head(50).to_string(index=False))
        raise AssertionError(
            f"Production model completeness failed: {len(bad)} of {expected_total} jobs are not complete"
        )

    print("\nMODEL PRODUCTION COMPLETENESS V3 PASSED")


if __name__ == "__main__":
    main()
