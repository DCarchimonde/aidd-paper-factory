"""Run the complete declared revision experiment and produce its report bundle.

Only hardware and storage choices are operator inputs. The scientific matrix,
training configurations and analyses are read from versioned protocol files.
Every declared job must finish and pass its receipt before a completed bundle
is emitted. A failed training run or unavailable data source remains a failure.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from paper2_admet_benchmark.revision_2026 import core, runner


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()


@contextlib.contextmanager
def execution_log(output):
    with (output / "execution.log").open("a", encoding="utf-8", buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            print(f"execution started UTC={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
            try:
                yield
            except Exception as error:
                trace = traceback.format_exc()
                print(trace, file=sys.stderr, flush=True)
                write_json(output / "failures" / f"failure_{time.time_ns()}.json",
                    {"status": "execution_failed_not_complete", "type": type(error).__name__,
                     "message": str(error), "traceback": trace})
                raise


def hardware_metadata(device):
    import torch
    result = {"device": device, "platform": platform.platform(), "cpu": platform.processor(),
              "logical_cpu_count": os.cpu_count(), "torch": torch.__version__,
              "torch_cuda_runtime": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
              "torch_compiled_cuda_architectures": torch.cuda.get_arch_list(),
              "cuda_tiny_forward_backward_preflight_passed": device == "cuda"}
    if device == "cuda":
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        result.update({"cuda_device_name": prop.name, "cuda_total_memory_bytes": prop.total_memory,
                       "cuda_compute_capability": [prop.major, prop.minor]})
    return result


def peak_memory_mb():
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / (1024 ** 2 if sys.platform == "darwin" else 1024)
    except ImportError:
        return None


def storage_metadata(output, formal_case_count):
    """Planning estimate, not a disk-space guarantee or training-time claim."""
    return {"available_bytes_at_check": shutil.disk_usage(output).free,
            "recommended_formal_free_bytes": 300 * 1024 ** 3,
            "formal_neural_fits": formal_case_count * 3,
            "checkpoint_estimate_basis": "diagnostic default-architecture checkpoints: CoDrug ~121 MB and KMM ~149 MB per fit",
            "estimated_formal_checkpoint_bytes": formal_case_count * (2 * 121 + 149) * 1000 ** 2,
            "planning_note": "Allow at least 300 GiB for weights, row-level tables, ZIP and duplicate transport parts. Actual size depends on cohorts and artifacts; raw outputs and original ZIP are retained."}


def completed_stage(run_root, name, builder):
    """Resume a completed stage or preserve an incomplete attempt and rebuild."""
    directory = run_root / name
    receipt = run_root / "stage_receipts" / f"{name}.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        runner.verify_hashes(directory, saved["files"])
        return saved["result"]
    if directory.exists():
        previous = run_root / "incomplete_stage_history" / f"{name}_{time.time_ns()}"
        previous.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(previous)
    result = builder(directory)
    files = {str(p.relative_to(directory)): runner.sha256_file(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    if not files:
        raise RuntimeError(f"completed stage produced no files: {name}")
    write_json(receipt, {"status": "complete", "files": files, "result": result})
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    runner.atomic_json(path, json_safe(value))


def get_modules():
    from paper2_admet_benchmark.revision_2026.full_baselines import codrug, kmm_cp
    from paper2_admet_benchmark.revision_2026 import independent_data
    return codrug, kmm_cp, independent_data


def resolve_device(requested):
    import torch
    resolved = "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested
    if resolved == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but the installed PyTorch cannot use a CUDA device")
    if resolved == "cuda":
        try:
            probe = torch.randn((4, 4), device="cuda", requires_grad=True)
            (probe @ probe).square().mean().backward()
            torch.cuda.synchronize()
        except Exception as error:
            raise RuntimeError("CUDA tensor forward/backward preflight failed before data acquisition. "
                "The installed PyTorch/CUDA build or driver may not support this GPU. "
                "Keep the error log and ask the authors for a validated environment; "
                "the program will not silently change dependencies or switch to CPU.") from error
    return resolved


def case_specs(protocol, independent_specs):
    result = []
    for phase in ("development", "retrospective_epa"):
        result += [{"kind": "historical", "phase": phase, "endpoint": ep, "seed": seed}
                   for ep in runner.ENDPOINT_PROPERTIES for seed in protocol[f"{phase}_seeds"]]
    result += [{"kind": "independent", **spec} for spec in independent_specs]
    return result


def build_required_jobs(protocol, specs, codrug_methods, kmm_methods):
    """Enumerate the whole matrix without loading any molecular outcomes."""
    execution = protocol["complete_execution"]
    shared_methods = runner.expected_method_names(protocol, execution["audit_sensitivities"],
                                                   execution["density_components"])
    jobs = []
    for spec in specs:
        phase, endpoint, seed = spec["phase"], spec["endpoint"], spec["seed"]
        for model in protocol["models"]:
            relative = Path("shared") / phase / "cells" / f"{phase}__{endpoint}__{seed}__{model}"
            jobs.append({"job_id": f"shared:{phase}:{endpoint}:{seed}:{model}", "family": "shared",
                "implementation": "shared_predictor_reliability_methods", "phase": phase,
                "endpoint": endpoint, "seed": seed, "model": model, "alphas": protocol["alpha_grid"],
                "methods": shared_methods, "predictions": str(relative / "predictions.csv"),
                "completion": str(relative / "complete.json")})
        for model, methods in (("codrug", codrug_methods), ("codrug_margin_swap", codrug_methods), ("kmm_cp", kmm_methods)):
            relative = Path("full") / phase / "cells" / f"{phase}__{endpoint}__{seed}__{model}"
            jobs.append({"job_id": f"full:{phase}:{endpoint}:{seed}:{model}", "family": "full_reference",
                "implementation": model, "implementation_status": "paper_based_independent_reproduction",
                "phase": phase, "endpoint": endpoint, "seed": seed, "model": model,
                "alphas": protocol["alpha_grid"], "methods": list(methods),
                "predictions": str(relative / "predictions.csv"), "completion": str(relative / "complete.json")})
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate scheduled jobs")
    return jobs


def build_phase_specs(specs):
    return {phase: {"endpoints": sorted({s["endpoint"] for s in specs if s["phase"] == phase}),
                    "seeds": sorted({s["seed"] for s in specs if s["phase"] == phase}),
                    "target_cohort_identical_across_fit_seeds": True,
                    "scope": "historical_reanalysis" if phase in ("development", "retrospective_epa") else "independent_source_validation",
                    "outer_split_seed": 43021 if phase == "independent_random" else 43022 if phase == "independent_scaffold" else None}
            for phase in sorted({s["phase"] for s in specs})}


def bind_case_inputs(output, case):
    """Bind chemical identities/outcomes to the run before any fit or reuse."""
    phase, endpoint, seed = case["phase"], case["endpoint"], case["seed"]
    value = {"phase": phase, "endpoint": endpoint, "seed": seed,
        "input_hashes": case.get("input_hashes", {}), "split_metadata": case.get("split_metadata", {}),
        "source_rows_sha256": runner.stable_sha256(case["source_rows"]),
        "source_outcomes_sha256": runner.stable_sha256(np.asarray(case["source_y"]).tolist()),
        "target_rows_sha256": runner.stable_sha256(case["target_rows"]),
        "target_outcomes_sha256": runner.stable_sha256(case["labels"])}
    path = output / "case_inputs" / f"{phase}__{endpoint}__{seed}.json"
    clean = json_safe(value)
    if path.exists():
        if json.loads(path.read_text()) != clean:
            raise RuntimeError(f"case inputs changed: {phase}/{endpoint}/{seed}")
    else:
        write_json(path, clean)


def role_records(rows, y, roles):
    return [{"structure_id": rows[i]["structure_id"], "standardized_smiles": rows[i]["standardized_smiles"],
             "y": int(y[i]), "role": role}
            for role, indices in zip(("fit", "router", "conformal"), roles) for i in indices]


def save_numeric_state(cell, **arrays):
    path = cell / "numeric_state.npz"
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{k: np.asarray(v) for k, v in arrays.items()})
    os.replace(temporary, path)


def finish_full_receipt(cell, implementation, metadata):
    write_json(cell / "implementation.json", {"implementation": implementation,
        "implementation_status": "paper_based_independent_reproduction_not_official_repository_code",
        **metadata})
    files = [p for p in cell.rglob("*") if p.is_file() and p.name != "complete.json" and is_deliverable(p, cell)]
    write_json(cell / "complete.json", {"status": "complete", "implementation": implementation,
        "files": {str(p.relative_to(cell)): runner.sha256_file(p) for p in sorted(files)}})


def is_deliverable(path, root):
    relative = path.relative_to(root)
    return not any(part.startswith(".") for part in relative.parts) and not path.name.endswith((".partial", ".tmp", ".download"))


def _default_config(module):
    value = module.default_config() if callable(getattr(module, "default_config", None)) else getattr(module, "DEFAULT_CONFIG", {})
    if not isinstance(value, dict):
        raise TypeError("baseline default configuration must be a dictionary")
    return json.loads(json.dumps(value))


def execute_full_case(case, protocol, output, cache, device, jobs, codrug, kmm_cp, auxiliary):
    source, y = case["source_rows"], np.asarray(case["source_y"], dtype=int)
    targets, labels = case["target_rows"], case["labels"]
    phase, endpoint, seed = case["phase"], case["endpoint"], int(case["seed"])
    alphas = protocol["alpha_grid"]
    source_ids = {r["structure_id"] for r in source}
    overlap = np.array([r["structure_id"] in source_ids for r in targets])
    valid = np.array([r.get("structure_status", "pass") == "pass" for r in targets])
    eligible = valid & ~overlap
    if eligible.sum() < 2:
        raise RuntimeError("full reference baseline requires at least two eligible target compounds")
    targets = [{**row, "excluded_structure_overlap": int(overlap[i])} for i, row in enumerate(targets)]
    roles = runner.split_internal_roles(y, seed)
    fit, tuning, cal = roles
    smiles = np.asarray([r["standardized_smiles"] for r in source], dtype=object)
    query_smiles = [r["standardized_smiles"] for r, keep in zip(targets, eligible) if keep]
    records = role_records(source, y, roles)
    sim_all = np.full(len(targets), np.nan)
    fit_x = runner.fingerprints([source[i] for i in fit])
    query_x = runner.fingerprints([r for r, keep in zip(targets, eligible) if keep])
    sim_all[eligible] = runner.max_tanimoto(fit_x, query_x)
    for model in ("codrug", "codrug_margin_swap", "kmm_cp"):
        cell = output / "full" / phase / "cells" / f"{phase}__{endpoint}__{seed}__{model}"
        if runner.validate_cell_receipt(cell):
            print(f"verified full baseline {phase} {endpoint} seed={seed} model={model}", flush=True)
            continue
        cell.mkdir(parents=True, exist_ok=True)
        config = _default_config(codrug)
        if model == "kmm_cp":
            config.update(kmm_cp.backbone_config() if callable(getattr(kmm_cp, "backbone_config", None)) else {})
            config["energy_weight"] = 0.0
        elif model == "codrug_margin_swap":
            config.update({"margin_in": -35.0, "margin_out": -5.0})
        config.update(protocol["complete_execution"].get("backbone_overrides", {}).get(model, {}))
        config["verbose"] = True
        print(f"start full backbone {phase} {endpoint} seed={seed} model={model} "
              f"fit={len(fit)} router={len(tuning)} calibration={len(cal)} target={int(eligible.sum())}", flush=True)
        result = codrug.fit_predict(smiles[fit].tolist(), y[fit], smiles[tuning].tolist(), y[tuning],
            smiles[cal].tolist(), y[cal], query_smiles, alphas=alphas, seed=seed,
            device=device, jobs=jobs, model_dir=cell / "backbone", config=config,
            auxiliary_smiles=auxiliary if model.startswith("codrug") else None)
        cp, qp = np.asarray(result["calibration_probability"]), np.asarray(result["probability_1"])
        p_all = np.full(len(targets), np.nan)
        p_all[eligible] = qp
        selection, raw = None, None
        extra_numeric = {}
        metadata = {"backbone": result["diagnostics"], "config": config}
        if model.startswith("codrug"):
            methods = result["sets_by_alpha"]
        else:
            kconfig = _default_config(kmm_cp)
            kconfig.update(protocol["complete_execution"].get("kmm_overrides", {}))
            print(f"start KMM bandwidth/ordinary/joint-selection/refit {phase} {endpoint} seed={seed}", flush=True)
            weighted = kmm_cp.fit_predict(result["calibration_embedding"], y[cal], cp,
                result["target_embedding"], qp, alphas=alphas, seed=seed, config=kconfig)
            print(f"complete KMM optimization selected={int(np.asarray(weighted['selected']).sum())}/{len(qp)}", flush=True)
            methods = weighted["predictions"]
            selection = weighted["method_selected"]
            raw = weighted["raw_predictions"]
            metadata.update({"calibration_method": weighted["diagnostics"], "calibration_config": kconfig})
            extra_numeric = {"selection_weights": weighted["selection_weights"], "selected": weighted["selected"],
                             **{f"source_weights_{name}": value for name, value in weighted["source_weights"].items()}}
        expected_names = set(codrug.METHOD_NAMES if model.startswith("codrug") else kmm_cp.METHOD_NAMES)
        if set(map(float, methods)) != set(map(float, alphas)) or any(set(v) != expected_names for v in methods.values()):
            raise RuntimeError(f"{model} returned an incomplete or unexpected method/alpha matrix")
        runner.write_cell(cell, phase, endpoint, seed, model, targets, labels, eligible, p_all,
            [source[i] for i in cal], y[cal], cp, records, methods, {}, json_safe(metadata),
            selection_by_method=selection, raw_methods_by_alpha=raw, max_train_tanimoto=sim_all)
        # Remove the provisional receipt until model/state artifacts are also
        # validated and included. A crash before the final receipt recomputes.
        (cell / "complete.json").unlink()
        save_numeric_state(cell, calibration_probability=cp, target_probability=qp,
            calibration_embedding=result["calibration_embedding"], target_embedding=result["target_embedding"],
            calibration_labels=y[cal], eligible_target_indices=np.flatnonzero(eligible), **extra_numeric)
        finish_full_receipt(cell, model, json_safe(metadata))
        print(f"complete full baseline {phase} {endpoint} seed={seed} model={model}", flush=True)


def verify_required_jobs(output, jobs):
    """Completion means every declared cell exists and all saved bytes match."""
    completed = []
    for job in jobs:
        cell = (output / job["completion"]).parent
        if not runner.validate_cell_receipt(cell):
            raise RuntimeError(f"required job is incomplete: {job['job_id']}")
        if not (output / job["predictions"]).is_file():
            raise RuntimeError(f"required predictions missing: {job['job_id']}")
        completed.append(job["job_id"])
    return completed


def reduce_pilot_case(case, source_per_class=80, target_count=32):
    """Fixed diagnostic reduction. It can never produce formal study claims."""
    source, y = case["source_rows"], np.asarray(case["source_y"], dtype=int)
    chosen = []
    for label in (0, 1):
        candidates = [i for i in range(len(y)) if y[i] == label]
        candidates.sort(key=lambda i: source[i]["structure_id"])
        chosen.extend(candidates[:source_per_class])
    chosen.sort(key=lambda i: source[i]["structure_id"])
    targets = sorted(case["target_rows"], key=lambda r: r["sample_id"])[:target_count]
    result = {**case, "source_rows": [source[i] for i in chosen], "source_y": y[chosen],
              "target_rows": targets, "labels": {r["sample_id"]: case["labels"][r["sample_id"]] for r in targets}}
    result["split_metadata"] = {**case.get("split_metadata", {}),
        "pilot_only": True, "formal_source_n": len(source), "formal_target_n": len(case["target_rows"]),
        "pilot_source_n": len(chosen), "pilot_target_n": len(targets),
        "source_selection": "up_to_80_per_true_source_class_sorted_structure_id",
        "target_selection": "first_32_sample_ids_without_outcome_selection"}
    return result


def package_results(output, package_path, resume=False, review=False):
    """Package produced evidence; raw downloaded sources stay in the cache."""
    if package_path.exists():
        if not resume:
            raise FileExistsError(f"result bundle already exists: {package_path}")
        # A complete execution receipt is handled before this function. An
        # archive without that receipt is an interrupted packaging attempt;
        # retain its bytes and rebuild from the verified experiment/report.
        previous = output.parent / (output.name + "_bundle_history") / f"{time.time_ns()}_{package_path.name}"
        previous.parent.mkdir(parents=True, exist_ok=True)
        previous_hash = runner.sha256_file(package_path)
        package_path.rename(previous)
        write_json(output / "bundle_recovery.json", {"status": "preserved_unreceipted_archive_before_rebuild",
            "previous_path": str(previous), "previous_sha256": previous_hash})
    temporary = package_path.with_suffix(".zip.partial")
    files = [p for p in output.rglob("*") if p.is_file() and p.name != "execution.log" and is_deliverable(p, output)]
    if review:
        files = [p for p in files if p.relative_to(output).parts[0] == "report"
                 or p.suffix in {".json", ".log"} or p.name.endswith("metrics.csv")]
    manifest = {str(p.relative_to(output)): runner.sha256_file(p) for p in sorted(files)}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        archive.writestr("BUNDLE_CONTENT_SHA256.json", json.dumps(manifest, indent=2))
        archive.writestr("BUNDLE_SCOPE.json", json.dumps({"kind": "review_bundle" if review else "complete_run_bundle",
            "contains_model_weights": not review, "contains_row_level_predictions": not review,
            "review_receipts_reference_full_run_artifacts_not_in_this_lightweight_bundle": review}, indent=2))
        for path in sorted(files):
            archive.write(path, str(path.relative_to(output)))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("result archive verification failed")
    os.replace(temporary, package_path)
    return {"path": str(package_path), "sha256": runner.sha256_file(package_path), "files": len(files)}


REASSEMBLE_SCRIPT = '''"""Reassemble a verified result ZIP from its numbered parts."""
import hashlib, json, pathlib, sys
folder = pathlib.Path(__file__).resolve().parent
manifest = json.loads((folder / "parts_manifest.json").read_text())
destination = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else folder.parent / ("reassembled_" + manifest["archive_name"])
if destination.exists():
    raise FileExistsError("Destination exists; choose a new output path")
temporary = destination.with_suffix(destination.suffix + ".partial")
whole = hashlib.sha256()
with temporary.open("wb") as target:
    for item in manifest["parts"]:
        digest = hashlib.sha256()
        with (folder / item["name"]).open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk); whole.update(chunk); target.write(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError("Part checksum failed: " + item["name"])
if whole.hexdigest() != manifest["archive_sha256"]:
    raise ValueError("Reassembled archive checksum failed")
temporary.replace(destination)
print(destination)
'''


def split_bundle(package, part_size=1024 ** 3, resume=False):
    """Split large archives into verified 1 GiB transport parts, automatically."""
    path = Path(package["path"])
    if path.stat().st_size <= part_size:
        return {"split": False, "archive": package, "reason": "archive_at_most_one_GiB"}
    directory = path.with_name(path.name + "_parts")
    manifest_path = directory / "parts_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["archive_sha256"] == package["sha256"]:
            runner.verify_hashes(directory, {row["name"]: row["sha256"] for row in existing["parts"]})
            if (directory / "reassemble.py").read_text() != REASSEMBLE_SCRIPT:
                raise RuntimeError("existing reassembly script differs")
            return {"split": True, "directory": str(directory), "manifest": str(manifest_path), "parts": len(existing["parts"])}
        if not resume:
            raise RuntimeError("existing parts belong to a different archive")
        directory.rename(directory.with_name(directory.name + f"_previous_{time.time_ns()}"))
    if directory.exists() and any(directory.iterdir()):
        history = directory.with_name(directory.name + f"_incomplete_{time.time_ns()}")
        directory.rename(history)
    directory.mkdir(parents=True, exist_ok=True)
    parts, whole = [], hashlib.sha256()
    with path.open("rb") as source:
        index = 0
        while source.tell() < path.stat().st_size:
            index += 1
            part = directory / f"{path.name}.part{index:03d}"
            digest, size = hashlib.sha256(), 0
            with part.open("wb") as target:
                while size < part_size:
                    chunk = source.read(min(1024 ** 2, part_size - size))
                    if not chunk:
                        break
                    target.write(chunk); digest.update(chunk); whole.update(chunk)
                    size += len(chunk)
            parts.append({"name": part.name, "sha256": digest.hexdigest(), "bytes": size})
    if whole.hexdigest() != package["sha256"]:
        raise RuntimeError("archive changed during splitting")
    (directory / "reassemble.py").write_text(REASSEMBLE_SCRIPT, encoding="utf-8")
    write_json(manifest_path, {"archive_name": path.name, "archive_sha256": package["sha256"],
                              "part_size_bytes": part_size, "parts": parts})
    return {"split": True, "directory": str(directory), "manifest": str(manifest_path), "parts": len(parts)}


def verify_delivery_packages(receipt):
    for key in ("bundle", "review_bundle"):
        package = receipt[key]
        path = Path(package["path"])
        if not path.is_file() or runner.sha256_file(path) != package["sha256"]:
            raise RuntimeError(f"completed {key} is missing or changed")
    transport = receipt["transport_parts"]
    if transport["split"]:
        manifest_path = Path(transport["manifest"])
        if runner.sha256_file(manifest_path) != transport["manifest_sha256"]:
            raise RuntimeError("transport manifest changed")
        manifest = json.loads(manifest_path.read_text())
        if manifest["archive_sha256"] != receipt["bundle"]["sha256"]:
            raise RuntimeError("transport parts do not belong to the completed archive")
        runner.verify_hashes(Path(transport["directory"]), {p["name"]: p["sha256"] for p in manifest["parts"]})
        if runner.sha256_file(Path(transport["directory"]) / "reassemble.py") != transport["reassemble_sha256"]:
            raise RuntimeError("transport reassembly script changed")


def verify_completed_stages(output):
    for stage in (output / "stage_receipts").glob("*.json"):
        runner.verify_hashes(output / stage.stem, json.loads(stage.read_text())["files"])


def write_delivery_packages(args, *, pilot=False):
    sys.stdout.flush(); sys.stderr.flush()
    shutil.copyfile(args.output / "execution.log", args.output / "execution_snapshot.log")
    prefix = args.output.name + ("_pilot" if pilot else "")
    review = package_results(args.output, args.output.with_name(prefix + "_review.zip"), args.resume, review=True)
    write_json(args.output / "review_bundle.json", review)
    full = package_results(args.output, args.output.with_name(prefix + "_results.zip"), args.resume)
    parts = split_bundle(full, resume=args.resume)
    if parts["split"]:
        parts.update(manifest_sha256=runner.sha256_file(Path(parts["manifest"])),
                     reassemble_sha256=runner.sha256_file(Path(parts["directory"]) / "reassemble.py"))
    return {"bundle": full, "review_bundle": review, "transport_parts": parts}


def execute(args):
    codrug, kmm_cp, independent_data = get_modules()
    protocol = json.loads(args.protocol.read_text())
    lock = yaml.safe_load(runner.LEGACY_LOCK.read_text())
    device = resolve_device(args.device)
    runner.check_processing_environment()
    versions = runner.package_versions()
    for package in ("torch", "matplotlib", "osqp"):
        versions[package] = importlib.metadata.version(package)
    specs = case_specs(protocol, independent_data.case_specs())
    formal_case_count = len(specs)
    if args.pilot:
        specs = [s for s in specs if (s["phase"], s["endpoint"], s["seed"]) in
                 {("development", "Tox21_NR_AhR", 101), ("independent_scaffold", "hERG", 311)}]
        if len(specs) != 2:
            raise RuntimeError("the fixed two-case diagnostic pilot could not be constructed")
        protocol = json.loads(json.dumps(protocol))
        protocol["study_status"] = "diagnostic_pilot_not_formal_results"
        protocol["complete_execution"]["backbone_overrides"] = {
            name: {"max_epochs": 2, "patience": 2, "aux_max_molecules": 512}
            for name in ("codrug", "codrug_margin_swap", "kmm_cp")}
        protocol["complete_execution"]["kmm_overrides"] = {"permutations": 20}
    jobs = build_required_jobs(protocol, specs, codrug.METHOD_NAMES, kmm_cp.METHOD_NAMES)
    config_paths = [args.protocol, HERE / "independent_sources.json"]
    identity = {"protocol_sha256": runner.sha256_file(args.protocol), "code": runner.code_hashes(),
        "configs": {str(p.relative_to(ROOT)): runner.sha256_file(p) for p in config_paths},
        "device": device, "jobs": args.jobs, "python": platform.python_version(), "packages": versions,
        "hardware": hardware_metadata(device),
        "execution_kind": "diagnostic_pilot" if args.pilot else "formal_declared_matrix",
        "scheduled_jobs_sha256": runner.stable_sha256(jobs)}
    runner.ensure_run(args.output, identity, args.resume)
    with execution_log(args.output):
        return execute_owned(args, codrug, kmm_cp, independent_data, protocol, lock, device, specs, jobs, formal_case_count)


def execute_owned(args, codrug, kmm_cp, independent_data, protocol, lock, device, specs, jobs, formal_case_count):
    write_json(args.output / "hardware.json", hardware_metadata(device))
    write_json(args.output / "protocol_snapshot.json", protocol)
    phase_specs = build_phase_specs(specs)
    write_json(args.output / "required_jobs.json", {"protocol_sha256": runner.sha256_file(args.output / "protocol_snapshot.json"),
        "source_protocol_sha256": runner.sha256_file(args.protocol),
        "execution_kind": "diagnostic_pilot" if args.pilot else "formal_declared_matrix", "phase_specs": phase_specs, "jobs": jobs})
    if args.pilot and (args.output / "pilot_complete.json").exists():
        receipt = json.loads((args.output / "pilot_complete.json").read_text())
        verify_required_jobs(args.output, jobs)
        verify_delivery_packages(receipt)
        verify_completed_stages(args.output)
        print(json.dumps({"status": "verified_completed_diagnostic_pilot", "bundle": receipt["bundle"]["path"]}))
        return receipt
    if (args.output / "execution_complete.json").exists():
        receipt = json.loads((args.output / "execution_complete.json").read_text())
        verify_required_jobs(args.output, jobs)
        verify_delivery_packages(receipt)
        verify_completed_stages(args.output)
        print(json.dumps({"status": "verified_completed_execution", "bundle": receipt["bundle"]["path"]}))
        return receipt
    started = time.monotonic()
    write_json(args.output / "storage_estimate.json", storage_metadata(args.output, formal_case_count))
    print(f"scheduled {len(specs)} cases and {len(jobs)} predictor cells on {device}", flush=True)
    from paper2_admet_benchmark.revision_2026 import synthetic_mechanisms
    synthetic_config = protocol["synthetic_mechanisms"].copy()
    if args.pilot:
        synthetic_config.update(replicates=2, n_calibration=100, n_target=100)
    completed_stage(args.output, "synthetic_mechanisms",
                    lambda destination: synthetic_mechanisms.run(destination, **synthetic_config))
    sources, processed = runner.prepare(args.cache / "tox21", lock)
    auxiliary, auxiliary_metadata = codrug.acquire_auxiliary_smiles(args.cache / "codrug_auxiliary", _default_config(codrug))
    write_json(args.output / "auxiliary_source.json", auxiliary_metadata)
    targets = {}
    for phase in ("development", "retrospective_epa"):
        target_rows, labels, label_path = runner.load_target(phase, sources, lock, args.cache / "tox21")
        targets[phase] = (target_rows, labels, label_path)
    for spec in specs:
        phase, endpoint, seed = spec["phase"], spec["endpoint"], int(spec["seed"])
        if spec["kind"] == "historical":
            source_rows, source_y = runner.read_clean_endpoint(processed / f"{endpoint}_clean.csv")
            target_rows, labels, label_path = targets[phase]
            case = {**spec, "source_rows": source_rows, "source_y": source_y, "target_rows": target_rows,
                "labels": labels, "input_hashes": {"preparation": runner.sha256_file(args.cache / "tox21/preparation.json"),
                "labels": runner.sha256_file(label_path)}, "split_metadata": {"status": "historical_cohort_reanalysis"}}
        else:
            case = independent_data.load_case(args.cache / "independent", endpoint, spec["split"], seed)
        if args.pilot:
            case = reduce_pilot_case(case)
        bind_case_inputs(args.output, case)
        runner.execute_shared_case(case["source_rows"], np.asarray(case["source_y"]), case["target_rows"], case["labels"],
            phase=phase, endpoint=endpoint, seed=seed, models=protocol["models"], alphas=protocol["alpha_grid"],
            protocol=protocol, lock=lock, output=args.output / "shared" / phase, jobs=args.jobs,
            sensitivities=protocol["complete_execution"]["audit_sensitivities"],
            selected_density=protocol["complete_execution"]["density_components"])
        execute_full_case(case, protocol, args.output, args.cache, device, args.jobs, codrug, kmm_cp, auxiliary)
    complete = verify_required_jobs(args.output, jobs)
    from paper2_admet_benchmark.revision_2026 import reporting
    if args.pilot:
        facts = []
        for job in jobs:
            frame, evidence = reporting.verify_job(args.output, job)
            reporting.verify_matched_budgets(frame)
            facts.append({"job_id": job["job_id"], "rows": len(frame), "receipt_verified": True})
        receipt = {"status": "diagnostic_pilot_complete_not_formal_results", "molecular_pilot_run": True,
            "formal_matrix_completed": False, "jobs": facts, "elapsed_seconds": time.monotonic() - started,
            "resource_note": "Same network architectures, 2 epochs, <=160 source and 32 target compounds; only measured pilot time is evidence. Formal training time depends on molecular graph sizes, early stopping and device, and cannot be inferred as a fixed duration.",
            "formal_planned_cases": formal_case_count, "formal_planned_predictor_cells": formal_case_count * 6,
            "formal_planned_neural_network_fits": formal_case_count * 3, "peak_resident_memory_mb": peak_memory_mb()}
        write_json(args.output / "pilot_results.json", receipt)
        sys.stdout.flush(); sys.stderr.flush()
        shutil.copyfile(args.output / "execution.log", args.output / "execution_snapshot.log")
        receipt.update(write_delivery_packages(args, pilot=True))
        write_json(args.output / "pilot_complete.json", receipt)
        print(json.dumps(json_safe(receipt), indent=2), flush=True)
        return receipt
    report = completed_stage(args.output, "report", lambda destination: reporting.build_report(args.output, destination, protocol=protocol))
    write_json(args.output / "experiment_complete.json", {"status": "all_declared_jobs_and_reporting_completed",
        "completed_jobs": complete, "completed_job_count": len(complete), "report": report,
        "elapsed_seconds": time.monotonic() - started, "epa_is_new_blinded_validation": False})
    packages = write_delivery_packages(args)
    receipt = {"status": "complete_execution_and_verified_bundle", **packages,
        "declared_jobs": len(jobs), "report": report, "elapsed_seconds": time.monotonic() - started}
    write_json(args.output / "execution_complete.json", receipt)
    print(json.dumps(json_safe(receipt), indent=2), flush=True)
    return receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=ROOT / ".local/chemolab_complete_cache")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot", action="store_true", help="fixed two-case diagnostic only; cannot produce the formal manuscript")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.output is None:
        args.output = ROOT / (".local/chemolab_complete_pilot" if args.pilot else ".local/chemolab_complete_results")
    args.cache, args.output = args.cache.resolve(), args.output.resolve()
    args.protocol = runner.DEFAULT_PROTOCOL
    if args.output == args.cache or args.output in args.cache.parents or args.cache in args.output.parents:
        parser.error("cache and output must be separate non-nested directories")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Log creation occurs only after output ownership/provenance validation
    # inside execute. Outer failures are printed and propagated unchanged.
    with threadpool_limits(limits=args.jobs):
        return execute(args)


if __name__ == "__main__":
    main()
