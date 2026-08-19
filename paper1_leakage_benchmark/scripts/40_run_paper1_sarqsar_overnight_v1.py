from __future__ import annotations

import argparse
import json
import py_compile
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
SCRIPTS = PAPER / "scripts"
SIM_SCRIPT = SCRIPTS / "34_run_metric_coupling_null_v1.py"
DRAFT_SCRIPT = SCRIPTS / "35_build_sarqsar_draft_v1.py"
PROTOCOL = PAPER / "SARQSAR_METRIC_COUPLING_PROTOCOL_V1.md"
CONFIG = PAPER / "SARQSAR_METRIC_COUPLING_CONFIG_V1.json"
SIM_OUT = PAPER / "results" / "sarqsar_metric_coupling_v1"
LATEX = ROOT / "paper1_sarqsar_latex"
BUNDLE = ROOT / "paper1_sarqsar_overnight_bundle_v1"
LOG = ROOT / "paper1_sarqsar_overnight_run_v1.log"
EXPECTED_BRANCH = "paper1-sarqsar-metric-coupling-2026"


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()


def require(path: Path) -> Path:
    if not path.exists() or (path.is_file() and path.stat().st_size == 0):
        raise FileNotFoundError(path)
    return path


def preflight() -> None:
    branch = git_value("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise AssertionError(f"Wrong branch: {branch!r}. Expected {EXPECTED_BRANCH!r}.")
    for path in [SIM_SCRIPT, DRAFT_SCRIPT, PROTOCOL, CONFIG]:
        require(path)
    for path in [SIM_SCRIPT, DRAFT_SCRIPT, Path(__file__)]:
        py_compile.compile(str(path), doraise=True)
        print("  SYNTAX OK", path.relative_to(ROOT))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for dataset in config["datasets"]:
        clean = PAPER / "data" / "processed_v2" / f"{dataset.lower()}_clean_v2.csv"
        require(clean)
        print("  INPUT OK ", clean.relative_to(ROOT))
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if free_gb < 3.0:
        raise OSError(f"At least 3 GB free disk space is required; found {free_gb:.2f} GB")
    print(f"  DISK OK  {free_gb:.1f} GB free")
    print("OVERNIGHT PREFLIGHT: PASS")


def stream_command(command: list[str], log_handle) -> None:
    print("\n>>>", " ".join(command), flush=True)
    log_handle.write("\n>>> " + " ".join(command) + "\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
        log_handle.flush()
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def zip_tree(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())


def package() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)
    for path in [PROTOCOL, CONFIG, SIM_OUT / "RUN_MANIFEST.json"]:
        shutil.copy2(require(path), BUNDLE / path.name)
    for name in [
        "null_metric_effect_summary.csv",
        "null_simulation_permutation_level_effects.csv",
        "null_simulation_quality_gate_summary.csv",
        "qsar_benchmark_minimum_reporting_checklist.csv",
    ]:
        source = require(SIM_OUT / "tables" / name)
        shutil.copy2(source, BUNDLE / name)
    for name in [
        "SARQSAR_NULL_SIMULATION_REPORT.md",
        "QSAR_BENCHMARK_REPORTING_CHECKLIST.md",
    ]:
        source = require(SIM_OUT / "reports" / name)
        shutil.copy2(source, BUNDLE / name)
    figure_out = BUNDLE / "figures"
    figure_out.mkdir()
    for path in sorted((SIM_OUT / "figures").glob("*")):
        if path.is_file():
            shutil.copy2(path, figure_out / path.name)
    manuscript_out = BUNDLE / "manuscript"
    manuscript_out.mkdir()
    for name in [
        "main.tex",
        "title_page.tex",
        "supplementary.tex",
        "WORKING_DRAFT_NOTES.md",
        "references.tex",
    ]:
        shutil.copy2(require(LATEX / name), manuscript_out / name)
    generated = manuscript_out / "generated"
    shutil.copytree(require(LATEX / "generated"), generated)
    build = LATEX / "build"
    if build.exists():
        for name in ["main.pdf", "title_page.pdf", "supplementary.pdf"]:
            if (build / name).exists():
                shutil.copy2(build / name, manuscript_out / name)
    completion = {
        "status": "complete",
        "branch": git_value("branch", "--show-current"),
        "commit": git_value("rev-parse", "HEAD"),
        "completed_at_unix": time.time(),
        "simulation_output": str(SIM_OUT.relative_to(ROOT)),
        "manuscript_output": str(LATEX.relative_to(ROOT)),
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "log": str(LOG.relative_to(ROOT)),
    }
    (BUNDLE / "OVERNIGHT_COMPLETION.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    zip_tree(BUNDLE, ROOT / "paper1_sarqsar_overnight_bundle_v1.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command overnight runner for the SAR/QSAR Paper 1 enhancement.")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 92)
    print("PAPER 1 SAR/QSAR OVERNIGHT BUILD")
    print("Frozen molecular null simulation + checklist + scientific working draft")
    print("=" * 92)
    preflight()
    started = time.time()
    with LOG.open("a", encoding="utf-8") as log_handle:
        log_handle.write("\n" + "=" * 92 + "\n")
        log_handle.write(f"RUN START {time.ctime()} commit={git_value('rev-parse', 'HEAD')}\n")
        sim_command = [
            sys.executable,
            "-u",
            str(SIM_SCRIPT),
            "--permutations",
            str(args.permutations),
            "--datasets",
            args.datasets,
        ]
        if args.force:
            sim_command.append("--force")
        if args.force_cache:
            sim_command.append("--force-cache")
        stream_command(sim_command, log_handle)
        draft_command = [sys.executable, "-u", str(DRAFT_SCRIPT)]
        if args.no_compile:
            draft_command.append("--no-compile")
        stream_command(draft_command, log_handle)
        package()
        elapsed = time.time() - started
        log_handle.write(f"\nRUN PASS elapsed_hours={elapsed/3600:.3f}\n")
    print("\n" + "=" * 92)
    print("PAPER 1 SAR/QSAR OVERNIGHT BUILD: PASS")
    print(f"Elapsed: {elapsed/3600:.2f} h")
    print("Simulation report:", SIM_OUT / "reports" / "SARQSAR_NULL_SIMULATION_REPORT.md")
    print("Working manuscript:", LATEX / "build" / "main.pdf")
    print("Bundle folder:", BUNDLE)
    print("Bundle ZIP:", ROOT / "paper1_sarqsar_overnight_bundle_v1.zip")
    print("Run log:", LOG)
    print("=" * 92)


if __name__ == "__main__":
    main()
