from __future__ import annotations

import importlib.util
import json
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
SCRIPTS = PAPER / "scripts"
BASE_RUNNER = SCRIPTS / "40_run_paper1_sarqsar_overnight_v1.py"
MATERIALIZER = SCRIPTS / "33_materialize_metric_coupling_null_v1.py"
GENERATED_SIM = SCRIPTS / "_generated" / "34_run_metric_coupling_null_v1_materialized.py"
DRAFT_SCRIPT = SCRIPTS / "35_build_sarqsar_draft_v1.py"
WINDOWS_WRAPPER = SCRIPTS / "43_run_paper1_sarqsar_overnight_windows_hardened_v1.py"
PROTOCOL = PAPER / "SARQSAR_METRIC_COUPLING_PROTOCOL_V1.md"
CONFIG = PAPER / "SARQSAR_METRIC_COUPLING_CONFIG_V1.json"
EXPECTED_BRANCH = "paper1-sarqsar-metric-coupling-2026"


def load_base_runner():
    spec = importlib.util.spec_from_file_location("paper1_sarqsar_base_runner_v1", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(path: Path) -> Path:
    if not path.exists() or (path.is_file() and path.stat().st_size == 0):
        raise FileNotFoundError(path)
    return path


def run_materializer() -> None:
    require(MATERIALIZER)
    subprocess.run([sys.executable, "-u", str(MATERIALIZER)], cwd=str(ROOT), check=True)
    require(GENERATED_SIM)


def hardened_preflight(base) -> None:
    branch = base.git_value("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise AssertionError(f"Wrong branch: {branch!r}. Expected {EXPECTED_BRANCH!r}.")

    run_materializer()

    required_files = [
        BASE_RUNNER,
        MATERIALIZER,
        GENERATED_SIM,
        DRAFT_SCRIPT,
        PROTOCOL,
        CONFIG,
        Path(__file__),
        WINDOWS_WRAPPER,
    ]
    for path in required_files:
        require(path)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if int(config.get("n_permutations", 0)) != 200:
        raise AssertionError("Authoritative configuration must contain 200 endpoint permutations.")
    if len(config.get("partition_seeds", [])) != 20:
        raise AssertionError("Authoritative configuration must contain 20 partition seeds.")

    compile_targets = [
        MATERIALIZER,
        GENERATED_SIM,
        DRAFT_SCRIPT,
        BASE_RUNNER,
        Path(__file__),
        WINDOWS_WRAPPER,
    ]
    for path in compile_targets:
        py_compile.compile(str(path), doraise=True)
        print("  SYNTAX OK", path.relative_to(ROOT))

    dependency_gate = (
        "import matplotlib, numpy, pandas, scipy; "
        "from rdkit import Chem; "
        "print('  DEPENDENCIES OK')"
    )
    subprocess.run([sys.executable, "-c", dependency_gate], cwd=str(ROOT), check=True)

    subprocess.run(
        [sys.executable, str(GENERATED_SIM), "--help"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=True,
    )
    print("  IMPORT/CLI OK", GENERATED_SIM.relative_to(ROOT))
    subprocess.run(
        [sys.executable, str(DRAFT_SCRIPT), "--help"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=True,
    )
    print("  IMPORT/CLI OK", DRAFT_SCRIPT.relative_to(ROOT))

    for dataset in config["datasets"]:
        clean = PAPER / "data" / "processed_v2" / f"{dataset.lower()}_clean_v2.csv"
        require(clean)
        print("  INPUT OK ", clean.relative_to(ROOT))

    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if free_gb < 3.0:
        raise OSError(f"At least 3 GB free disk space is required; found {free_gb:.2f} GB")
    print(f"  DISK OK  {free_gb:.1f} GB free")
    print("HARDENED OVERNIGHT PREFLIGHT: PASS")


def main() -> None:
    run_materializer()
    base = load_base_runner()
    base.SIM_SCRIPT = GENERATED_SIM
    base.preflight = lambda: hardened_preflight(base)
    base.main()


if __name__ == "__main__":
    main()
