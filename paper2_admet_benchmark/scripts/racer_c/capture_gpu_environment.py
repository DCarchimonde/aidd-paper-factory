from __future__ import annotations

"""Fail-closed runtime audit for the RACER-C seed-99 GPU benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import yaml


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
DEFAULT_CONFIG = P2 / "configs" / "racer_c" / "gpu_environment_lock.yaml"
DEFAULT_OUTPUT = P2 / "results" / "racer_c_phase3_preflight" / "environment"


PACKAGE_DISTRIBUTIONS = {
    "torch": "torch",
    "chemprop": "chemprop",
    "transformers": "transformers",
    "rdkit": "rdkit",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "xgboost": "xgboost",
    "pyyaml": "PyYAML",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def base_version(value: str) -> str:
    return value.split("+", 1)[0]


def compare_versions(
    expected: Mapping[str, object], observed: Mapping[str, str]
) -> list[str]:
    failures: list[str] = []
    for key, wanted in expected.items():
        got = observed.get(key)
        if got is None:
            failures.append(f"missing package: {key}")
        elif base_version(got) != str(wanted):
            failures.append(f"{key}: expected {wanted}, observed {got}")
    return failures


def platform_key() -> str:
    return f"{platform.system().lower()}_{platform.machine().lower()}"


def numeric_version(value: str) -> tuple[int, ...]:
    """Return a comparable numeric prefix for NVIDIA-style driver versions."""

    fields = value.strip().split(".")
    if not fields or any(not field.isdigit() for field in fields):
        raise ValueError(f"invalid numeric version: {value!r}")
    return tuple(int(field) for field in fields)


def query_nvidia_smi() -> tuple[str, dict[str, object], list[str]]:
    failures: list[str] = []
    details: dict[str, object] = {}
    try:
        full_output = subprocess.run(
            ["nvidia-smi"], check=True, capture_output=True, text=True
        ).stdout
        query_output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        rows = []
        for line in query_output.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                raise ValueError(f"unexpected nvidia-smi query row: {line!r}")
            rows.append(
                {
                    "driver_version": fields[0],
                    "device_name": fields[1],
                    "memory_total_mib": int(fields[2]),
                }
            )
        if not rows:
            raise ValueError("nvidia-smi returned no GPU rows")
        details = {"gpus": rows}
        return full_output, details, failures
    except Exception as exc:
        failures.append(f"nvidia-smi failed: {type(exc).__name__}: {exc}")
        return f"FAILED: {type(exc).__name__}: {exc}\n", details, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate the lock file without importing GPU packages",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_packages = config["packages"]
    if args.plan_only:
        print(
            json.dumps(
                {
                    "status": "candidate_lock_parsed",
                    "python": config["python"],
                    "packages": expected_packages,
                    "model_revision": config["molformer"]["revision"],
                    "gpu_required": config["gpu"]["required"],
                },
                sort_keys=True,
            )
        )
        return 0

    observed: dict[str, str] = {}
    failures: list[str] = []
    for key, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            observed[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"missing package: {key}")
    failures.extend(compare_versions(expected_packages, observed))

    observed_python = platform.python_version()
    if observed_python != str(config["python"]):
        failures.append(
            f"python: expected {config['python']}, observed {observed_python}"
        )

    observed_platform = platform_key()
    if observed_platform != str(config["platform"]).lower():
        failures.append(
            f"platform: expected {config['platform']}, observed {observed_platform}"
        )

    torch_details: dict[str, object] = {}
    try:
        import torch

        torch_details = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_build": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "device_total_memory_gib": [
                round(
                    torch.cuda.get_device_properties(index).total_memory
                    / (1024**3),
                    3,
                )
                for index in range(torch.cuda.device_count())
            ],
        }
        gpu_config = config["gpu"]
        if not torch_details["cuda_available"]:
            failures.append("torch.cuda.is_available() is false")
        if torch_details["device_count"] != int(gpu_config["count"]):
            failures.append(
                f"GPU count: expected {gpu_config['count']}, "
                f"observed {torch_details['device_count']}"
            )
        expected_name = str(gpu_config["device_name_contains"])
        if not any(
            expected_name in name for name in torch_details["device_names"]
        ):
            failures.append(f"no CUDA device name contains {expected_name!r}")
        if str(torch_details["cuda_build"]) != str(gpu_config["torch_cuda_build"]):
            failures.append(
                f"torch CUDA build: expected {gpu_config['torch_cuda_build']}, "
                f"observed {torch_details['cuda_build']}"
            )
        minimum_vram = gpu_config.get("minimum_vram_gib")
        if minimum_vram is not None and (
            not torch_details["device_total_memory_gib"]
            or min(torch_details["device_total_memory_gib"]) < float(minimum_vram)
        ):
            failures.append(
                f"GPU VRAM: expected at least {minimum_vram} GiB, "
                f"observed {torch_details['device_total_memory_gib']}"
            )
    except Exception as exc:
        failures.append(f"torch runtime audit failed: {type(exc).__name__}: {exc}")

    nvidia_smi, nvidia_details, nvidia_failures = query_nvidia_smi()
    failures.extend(nvidia_failures)
    gpu_config = config["gpu"]
    if nvidia_details:
        gpu_rows = nvidia_details["gpus"]
        minimum_driver = str(gpu_config["minimum_driver_version"])
        for row in gpu_rows:
            observed_driver = str(row["driver_version"])
            try:
                if numeric_version(observed_driver) < numeric_version(minimum_driver):
                    failures.append(
                        "NVIDIA driver: CUDA "
                        f"{gpu_config['torch_cuda_build']} requires at least "
                        f"{minimum_driver}, observed {observed_driver}"
                    )
            except ValueError as exc:
                failures.append(str(exc))

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not freeze.endswith("\n"):
        freeze += "\n"
    freeze_bytes = freeze.encode("utf-8")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pip_freeze.txt").write_bytes(freeze_bytes)
    (args.output_dir / "pip_freeze.sha256").write_text(
        sha256_bytes(freeze_bytes) + "\n", encoding="utf-8"
    )
    (args.output_dir / "nvidia_smi.txt").write_text(
        nvidia_smi, encoding="utf-8"
    )
    result = {
        "status": "pass" if not failures else "fail_closed",
        "lock_status": config["status"],
        "python": observed_python,
        "platform": observed_platform,
        "packages": observed,
        "torch": torch_details,
        "nvidia_smi": nvidia_details,
        "molformer_model_id": config["molformer"]["model_id"],
        "molformer_revision": config["molformer"]["revision"],
        "pip_freeze_sha256": sha256_bytes(freeze_bytes),
        "failures": sorted(set(failures)),
    }
    (args.output_dir / "environment_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
