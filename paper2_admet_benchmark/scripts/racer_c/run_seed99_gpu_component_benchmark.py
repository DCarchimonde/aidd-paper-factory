from __future__ import annotations

"""Measure frozen MoLFormer extraction and one Chemprop fit on seed 99.

This is a technical timing benchmark.  It is hard-coded to the development role,
does not create policy/conformal/test predictions, and computes no model metric.
"""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from lineage_contract import FitNode, PredictionNode, validate_prediction_lineage
from molformer_token_contract import (
    filter_model_eligible_rows,
    verify_runtime_tokenizer,
)
from prepare_seed99_gpu_benchmark import (
    FRACTIONS,
    GROUP_COLUMN,
    META_FOLDS,
    SEED,
    assert_primary,
    label_blind_group_folds,
    read_clean,
    sha256_file,
)
from role_feasibility import allocate_groups, read_csv, validate_role_input


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
DEFAULT_CONFIG = P2 / "configs" / "racer_c" / "gpu_environment_lock.yaml"
DEFAULT_ROLE_INPUT = (
    P2 / "data" / "processed" / "racer_c" / "role_inputs" / "Tox21_NR_ER_role_input.csv"
)
DEFAULT_CLEAN = P2 / "data" / "processed" / "racer_c" / "Tox21_NR_ER_clean.csv"
DEFAULT_DECISIONS = (
    P2 / "results" / "racer_c_phase2_preflight" / "endpoint_eligibility_decision.csv"
)
DEFAULT_ENV_AUDIT = (
    P2
    / "results"
    / "racer_c_phase3_preflight"
    / "environment"
    / "environment_audit.json"
)
DEFAULT_WORK = P2 / "data" / "processed" / "racer_c" / "gpu_benchmark_seed99"
DEFAULT_OUTPUT = (
    P2
    / "results"
    / "racer_c_phase3_preflight"
    / "seed99_gpu_component_benchmark.json"
)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def require_environment(path: Path, config: Mapping[str, object]) -> dict[str, object]:
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("status") != "pass":
        raise RuntimeError("GPU environment audit did not pass")
    if row.get("molformer_revision") != config["molformer"]["revision"]:
        raise RuntimeError("environment audit uses a different MoLFormer revision")
    return row


def require_input_hashes(
    role_input: Path, clean: Path, config: Mapping[str, object]
) -> dict[str, str]:
    observed = {
        "role_input_sha256": sha256_file(role_input),
        "clean_sha256": sha256_file(clean),
    }
    expected = config["inputs"]
    failures = [
        f"{key}: expected {expected[key]}, observed {value}"
        for key, value in observed.items()
        if value != str(expected[key])
    ]
    if failures:
        raise RuntimeError("benchmark input hash mismatch: " + "; ".join(failures))
    return observed


def development_rows(
    role_rows: list[dict[str, str]], clean_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    allocation = allocate_groups(
        role_rows,
        GROUP_COLUMN,
        FRACTIONS,
        SEED,
        use_labels_for_assignment=False,
    )
    dev_ids = {
        row["structure_id"]
        for row in role_rows
        if allocation.assignment[row[GROUP_COLUMN]] == "dev"
    }
    rows = [row for row in clean_rows if row["structure_id"] in dev_ids]
    if len(rows) != len(dev_ids):
        raise ValueError("development rows are incomplete")
    return sorted(rows, key=lambda row: row["structure_id"])


def split_anchor(
    rows: list[dict[str, str]], validation_fraction: float
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    outer = label_blind_group_folds(rows, GROUP_COLUMN, META_FOLDS, SEED)
    outer_train = [row for row in rows if outer[row[GROUP_COLUMN]] != 0]
    outer_valid = [row for row in rows if outer[row[GROUP_COLUMN]] == 0]
    denominator = round(1.0 / validation_fraction)
    if denominator < 2 or not math.isclose(1.0 / denominator, validation_fraction):
        raise ValueError("internal validation fraction must be a reciprocal integer")
    internal = label_blind_group_folds(
        outer_train, GROUP_COLUMN, denominator, SEED * 100
    )
    fit = [row for row in outer_train if internal[row[GROUP_COLUMN]] != 0]
    validation = [row for row in outer_train if internal[row[GROUP_COLUMN]] == 0]
    for name, subset in (("fit", fit), ("validation", validation), ("predict", outer_valid)):
        if {int(row["target"]) for row in subset} != {0, 1}:
            raise ValueError(f"{name} partition is not binary")
    return fit, validation, outer_valid


def extract_molformer(
    rows: list[dict[str, str]],
    config: Mapping[str, object],
    tokenizer: object,
    output: Path,
) -> dict[str, object]:
    import numpy as np
    import torch
    from transformers import AutoModel

    model_config = config["molformer"]
    model_id = model_config["model_id"]
    revision = model_config["revision"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=bool(model_config["trust_remote_code"]),
        deterministic_eval=bool(model_config["deterministic_eval"]),
    ).to("cuda")
    model.eval()
    embeddings: list[object] = []
    max_observed = 0
    started = time.perf_counter()
    batch_size = int(model_config["inference_batch_size"])
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["standardized_smiles"] for row in batch],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        lengths = encoded["attention_mask"].sum(dim=1)
        batch_max = int(lengths.max().item())
        max_observed = max(max_observed, batch_max)
        if batch_max > int(model_config["max_tokens_including_special_tokens"]):
            raise ValueError(
                "MoLFormer token length exceeds the frozen limit; no truncation permitted"
            )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            pooled = model(**encoded).pooler_output
        embeddings.append(pooled.detach().to(dtype=torch.float32).cpu().numpy())
    values = np.concatenate(embeddings, axis=0)
    if values.shape[0] != len(rows) or not np.isfinite(values).all():
        raise AssertionError("MoLFormer embeddings are incomplete or non-finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, values, allow_pickle=False)
    return {
        "n": len(rows),
        "dimension": int(values.shape[1]),
        "max_tokens_observed": max_observed,
        "seconds": time.perf_counter() - started,
        "batch_size": batch_size,
        "peak_torch_allocated_gib": round(
            torch.cuda.max_memory_allocated() / (1024**3), 3
        ),
        "peak_torch_reserved_gib": round(
            torch.cuda.max_memory_reserved() / (1024**3), 3
        ),
        "output_sha256": sha256_file(output),
    }


def gpu_memory_used_mib() -> int | None:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        return max(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def run_timed_gpu_command(command: list[str]) -> tuple[float, int | None]:
    started = time.perf_counter()
    process = subprocess.Popen(command)
    peak_mib = gpu_memory_used_mib()
    while process.poll() is None:
        observed = gpu_memory_used_mib()
        if observed is not None:
            peak_mib = observed if peak_mib is None else max(peak_mib, observed)
        time.sleep(0.5)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return time.perf_counter() - started, peak_mib


def read_chemprop_probabilities(
    path: Path, expected_rows: list[dict[str, str]]
) -> tuple[list[float], str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        table = list(reader)
        fields = reader.fieldnames or []
    if len(table) != len(expected_rows):
        raise AssertionError("Chemprop prediction row count mismatch")
    if "structure_id" in fields:
        observed_ids = [row["structure_id"] for row in table]
        expected_ids = [row["structure_id"] for row in expected_rows]
        if observed_ids != expected_ids:
            raise AssertionError("Chemprop prediction row order/identity mismatch")
    prediction_fields = [
        field for field in fields if field not in {"smiles", "structure_id"}
    ]
    if len(prediction_fields) != 1:
        raise AssertionError(
            "expected exactly one Chemprop prediction column after preserved "
            f"inputs, observed {prediction_fields}"
        )
    prediction_field = prediction_fields[0]
    values = [float(row[prediction_field]) for row in table]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise AssertionError("Chemprop predictions are non-finite or outside [0,1]")
    return values, prediction_field


def chemprop_commands(
    train_path: Path,
    predict_path: Path,
    model_dir: Path,
    prediction_path: Path,
    config: Mapping[str, object],
) -> tuple[list[str], list[str]]:
    cp = config["chemprop"]
    devices = str(cp["devices"])
    if not devices.isdecimal() or int(devices) < 1:
        raise ValueError(
            "Chemprop devices must be a positive device count; use 1 for the "
            "single visible RTX-4060 GPU"
        )
    train = [
        "chemprop",
        "train",
        "--data-path",
        str(train_path),
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "target",
        "--splits-column",
        "split",
        "--task-type",
        str(cp["task_type"]),
        "--loss-function",
        str(cp["loss_function"]),
        "--message-hidden-dim",
        str(cp["message_hidden_dim"]),
        "--depth",
        str(cp["depth"]),
        "--dropout",
        str(cp["dropout"]),
        "--aggregation",
        str(cp["aggregation"]),
        "--ffn-hidden-dim",
        str(cp["ffn_hidden_dim"]),
        "--ffn-num-layers",
        str(cp["ffn_num_layers"]),
        "--batch-size",
        str(cp["batch_size"]),
        "--epochs",
        str(cp["epochs"]),
        "--patience",
        str(cp["patience"]),
        "--init-lr",
        str(cp["init_lr"]),
        "--max-lr",
        str(cp["max_lr"]),
        "--final-lr",
        str(cp["final_lr"]),
        "--pytorch-seed",
        str(SEED),
        "--accelerator",
        str(cp["accelerator"]),
        "--devices",
        devices,
        "--output-dir",
        str(model_dir),
    ]
    predict = [
        "chemprop",
        "predict",
        "--test-path",
        str(predict_path),
        "--smiles-columns",
        "smiles",
        "--model-paths",
        str(model_dir),
        "--preds-path",
        str(prediction_path),
        "--accelerator",
        str(cp["accelerator"]),
        "--devices",
        devices,
    ]
    return train, predict


def run_chemprop(
    fit: list[dict[str, str]],
    validation: list[dict[str, str]],
    predict_rows: list[dict[str, str]],
    config: Mapping[str, object],
    work: Path,
    dry_run: bool,
) -> dict[str, object]:
    train_path = work / "chemprop_train.csv"
    predict_path = work / "chemprop_predict.csv"
    model_dir = work / "chemprop_model"
    prediction_path = work / "chemprop_predictions.csv"
    training_rows = [
        {
            "smiles": row["standardized_smiles"],
            "target": row["target"],
            "split": split,
            "structure_id": row["structure_id"],
        }
        for split, subset in (("train", fit), ("val", validation))
        for row in subset
    ]
    write_csv(train_path, training_rows, ["smiles", "target", "split", "structure_id"])
    write_csv(
        predict_path,
        [
            {"smiles": row["standardized_smiles"], "structure_id": row["structure_id"]}
            for row in predict_rows
        ],
        ["smiles", "structure_id"],
    )
    train_command, predict_command = chemprop_commands(
        train_path, predict_path, model_dir, prediction_path, config
    )
    if dry_run:
        return {
            "status": "dry_run",
            "fit_n": len(fit),
            "validation_n": len(validation),
            "predict_n": len(predict_rows),
            "train_command": train_command,
            "predict_command": predict_command,
        }
    if model_dir.exists() or prediction_path.exists():
        raise FileExistsError(
            "benchmark model/prediction output already exists; use a fresh work directory"
        )
    train_seconds, train_peak_gpu_memory_mib = run_timed_gpu_command(train_command)
    predict_seconds, predict_peak_gpu_memory_mib = run_timed_gpu_command(
        predict_command
    )
    values, prediction_column = read_chemprop_probabilities(
        prediction_path, predict_rows
    )
    return {
        "status": "pass_component_timing",
        "fit_n": len(fit),
        "validation_n": len(validation),
        "predict_n": len(predict_rows),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "train_peak_gpu_memory_mib": train_peak_gpu_memory_mib,
        "predict_peak_gpu_memory_mib": predict_peak_gpu_memory_mib,
        "prediction_column": prediction_column,
        "finite_probability_count": len(values),
        "prediction_file_sha256": sha256_file(prediction_path),
        "probabilities_retained_only_in_ignored_workdir": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--role-input", type=Path, default=DEFAULT_ROLE_INPUT)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--environment-audit", type=Path, default=DEFAULT_ENV_AUDIT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    input_hashes = require_input_hashes(args.role_input, args.clean, config)
    role_rows = validate_role_input(read_csv(args.role_input))
    endpoint = role_rows[0]["endpoint"]
    primary_count = assert_primary(endpoint, args.decisions)
    clean_rows = read_clean(args.clean, {row["structure_id"] for row in role_rows})
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["molformer"]["model_id"],
        revision=config["molformer"]["revision"],
        trust_remote_code=bool(config["molformer"]["trust_remote_code"]),
    )
    verify_runtime_tokenizer(
        clean_rows,
        tokenizer,
        str(config["molformer"]["input_column"]),
    )
    role_rows, clean_rows, model_eligibility = filter_model_eligible_rows(
        role_rows, clean_rows, config
    )
    dev_rows = development_rows(role_rows, clean_rows)
    fit, validation, predict_rows = split_anchor(
        dev_rows, float(config["chemprop"]["internal_validation_fraction"])
    )
    if not args.dry_run:
        environment = require_environment(args.environment_audit, config)
        molformer = extract_molformer(
            dev_rows,
            config,
            tokenizer,
            args.work_dir / "molformer_embeddings.npy",
        )
    else:
        environment = {"status": "not_read_in_dry_run"}
        molformer = {
            "status": "dry_run",
            "n": len(dev_rows),
            "revision": config["molformer"]["revision"],
        }
    chemprop = run_chemprop(
        fit, validation, predict_rows, config, args.work_dir, args.dry_run
    )
    roles = {row["structure_id"]: "dev" for row in dev_rows}
    fit_node = FitNode(
        "seed99_component_dmpnn_fit",
        "dmpnn_fit_and_internal_validation",
        frozenset(row["structure_id"] for row in fit + validation),
    )
    predictions = [
        PredictionNode(
            f"seed99_component_{row['structure_id']}",
            row["structure_id"],
            "dev",
            (fit_node.node_id,),
        )
        for row in predict_rows
    ]
    validate_prediction_lineage([fit_node], predictions, roles)
    result = {
        "status": "dry_run" if args.dry_run else "pass_gpu_component_benchmark",
        "scientific_interpretation": "technical timing and lineage only; not a scientific result",
        "endpoint": endpoint,
        "primary_endpoint_count": primary_count,
        "seed": SEED,
        "track": "strict_scaffold",
        "trainer_label_roles": ["dev"],
        "policy_conformal_test_predictions_generated": False,
        "performance_metrics_computed": False,
        "dev_n": len(dev_rows),
        "model_eligibility": model_eligibility,
        "lineage_prediction_count": len(predictions),
        "environment": environment,
        "molformer": molformer,
        "chemprop": chemprop,
        "role_input_sha256": input_hashes["role_input_sha256"],
        "clean_input_sha256": input_hashes["clean_sha256"],
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
    }
    if not args.dry_run:
        train_seconds = float(chemprop["train_seconds"])
        cells = primary_count * 3 * 5
        result["planning_projection"] = {
            "primary_endpoint_track_seed_cells": cells,
            "dmpnn_final_fit_equivalents_per_cell": 6.0,
            "projected_primary_dmpnn_gpu_hours": train_seconds * 6.0 * cells / 3600.0,
            "projected_with_20pct_rerun_reserve_gpu_hours": train_seconds
            * 6.0
            * cells
            * 1.2
            / 3600.0,
            "projection_is_engineering_not_scientific_result": True,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
