from __future__ import annotations

import importlib.util
import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and some import-time utilities expect the module to be
    # registered while it is executed.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_base_runner():
    return load_module(BASE_RUNNER, "paper1_sarqsar_base_runner_v1")


def require(path: Path) -> Path:
    if not path.exists() or (path.is_file() and path.stat().st_size == 0):
        raise FileNotFoundError(path)
    return path


def run_materializer() -> None:
    require(MATERIALIZER)
    subprocess.run([sys.executable, "-u", str(MATERIALIZER)], cwd=str(ROOT), check=True)
    require(GENERATED_SIM)


def run_cli_check(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        print(f"\nIMPORT/CLI CHECK FAILED: {path.relative_to(ROOT)}", flush=True)
        print(result.stdout or "<no subprocess output>", flush=True)
        raise RuntimeError(
            f"Import/CLI check returned {result.returncode} for {path.relative_to(ROOT)}"
        )
    print("  IMPORT/CLI OK", path.relative_to(ROOT))


def build_synthetic_summary(sim):
    rows: list[dict] = []

    def add(dataset: str, task: str, mode: str, budget: int, metric: str, value: float) -> None:
        rows.append(
            {
                "dataset": dataset,
                "task_type": task,
                "scaffold_mode": mode,
                "budget": budget,
                "metric": metric,
                "n_permutations_valid": 2,
                "mean": value,
                "sd": 0.01,
                "q025": value - 0.01,
                "median": value,
                "q975": value + 0.01,
                "fraction_positive": float(value > 0),
            }
        )

    for dataset_index, dataset in enumerate(["ESOL", "FreeSolv"], start=1):
        for mode_index, mode in enumerate(["single_group", "singleton"], start=1):
            for budget_index, budget in enumerate([100, 300], start=1):
                base = 0.01 * (dataset_index + mode_index + budget_index)
                add(dataset, "regression", mode, budget, "effect_rmse", base)
                add(dataset, "regression", mode, budget, "effect_mse", base * 2.0)
                add(dataset, "regression", mode, budget, "effect_squared_mean_gap", base * 1.5)
                add(dataset, "regression", mode, budget, "effect_test_variance", base * 0.5)

    for dataset_index, dataset in enumerate(["BACE", "BBBP", "ClinTox", "HIV"], start=1):
        for budget_index, budget in enumerate([10, 30], start=1):
            base = 0.001 * (dataset_index + budget_index)
            add(dataset, "classification", "single_group", budget, "effect_brier", base)
            add(dataset, "classification", "single_group", budget, "effect_log_loss", base * 1.2)
            add(dataset, "classification", "single_group", budget, "effect_average_precision", base * 0.5)
            add(dataset, "classification", "single_group", budget, "effect_roc_auc", 0.0)

    return sim.pd.DataFrame(rows)


def functional_smoke(config: dict) -> None:
    sim = load_module(GENERATED_SIM, "paper1_metric_coupling_materialized_smoke")
    if Path(sim.ROOT).resolve() != ROOT.resolve():
        raise AssertionError(
            f"Generated simulation resolved repository root to {sim.ROOT}, expected {ROOT}"
        )

    np = sim.np
    regression = sim.regression_metrics(
        test_sum=np.array([3.0, 5.0]),
        test_sumsq=np.array([5.0, 13.0]),
        n_test=2,
        total_sum=10.0,
        n_total=5,
        mae=np.array([0.5, 0.75]),
    )
    residual = regression["mse"] - regression["variance"] - regression["gap_sq"]
    if float(np.max(np.abs(residual))) > 1e-12:
        raise AssertionError(f"Regression metric decomposition smoke failed: {residual}")

    classification = sim.classification_metrics(
        positives_test=np.array([1.0, 2.0]),
        n_test=4,
        positives_total=4.0,
        n_total=8,
    )
    finite_auc = classification["roc_auc"][np.isfinite(classification["roc_auc"])]
    if finite_auc.size and not np.allclose(finite_auc, 0.5, rtol=0.0, atol=1e-15):
        raise AssertionError(f"Constant-score AUC smoke failed: {finite_auc}")

    with tempfile.TemporaryDirectory(prefix=".sarqsar_preflight_", dir=str(ROOT)) as temp_name:
        temp = Path(temp_name)
        sim.OUT = temp / "results"
        sim.CACHE = sim.OUT / "candidate_cache"
        sim.BY_SEED = sim.OUT / "by_seed"
        sim.TABLES = sim.OUT / "tables"
        sim.FIGURES = sim.OUT / "figures"
        sim.REPORTS = sim.OUT / "reports"
        sim.MANIFEST = sim.OUT / "RUN_MANIFEST.json"
        sim.ensure_dirs()

        smoke_config = json.loads(json.dumps(config))
        smoke_config["n_permutations"] = 2
        frame, _, clean_sha256 = sim.validate_clean_frame("BACE", "classification")
        permutations, permutation_seeds = sim.generate_permutations(
            frame["target"].to_numpy(dtype=float), "BACE", smoke_config
        )
        fingerprint = sim.protocol_fingerprint(smoke_config)
        checkpoint = sim.run_seed(
            frame=frame,
            permutations=permutations,
            permutation_seeds=permutation_seeds,
            dataset="BACE",
            task_type="classification",
            mode="single_group",
            partition_seed=42,
            budgets=[10],
            clean_sha256=clean_sha256,
            fingerprint=fingerprint,
            force=True,
            force_cache=True,
        )
        checkpoint_frame = sim.pd.read_csv(checkpoint)
        if len(checkpoint_frame) != 2:
            raise AssertionError(f"Real-data smoke checkpoint row count is {len(checkpoint_frame)}, expected 2")
        _, smoke_summary = sim.aggregate_results(smoke_config, [checkpoint])
        if smoke_summary.empty or "effect_roc_auc" not in set(smoke_summary["metric"]):
            raise AssertionError("Real-data smoke aggregation did not produce classification effects")

        sim.build_checklist()
        require(sim.TABLES / "qsar_benchmark_minimum_reporting_checklist.csv")
        require(sim.REPORTS / "QSAR_BENCHMARK_REPORTING_CHECKLIST.md")
        require(sim.REPORTS / "qsar_benchmark_checklist.tex")

        sim.plt.switch_backend("Agg")
        synthetic = build_synthetic_summary(sim)
        sim.build_figures(synthetic)
        for stem in [
            "figure_mc1_regression_null_coupling",
            "figure_mc2_classification_null_coupling",
            "figure_mc3_mse_decomposition",
        ]:
            for extension in ["pdf", "png", "tiff"]:
                require(sim.FIGURES / f"{stem}.{extension}")

    print("  REAL-DATA/FUNCTIONAL SMOKE OK (BACE, 2 permutations, 1 seed, 10 draws)")
    print("  FIGURE/CHECKLIST SMOKE OK")


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

    run_cli_check(GENERATED_SIM)
    run_cli_check(DRAFT_SCRIPT)

    for dataset in config["datasets"]:
        clean = PAPER / "data" / "processed_v2" / f"{dataset.lower()}_clean_v2.csv"
        require(clean)
        print("  INPUT OK ", clean.relative_to(ROOT))

    functional_smoke(config)

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
