"""Fixed finite-batch triage experiments; expensive score fitting is cached once."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from .core import CountCS, InconsistentConfidenceSet, OptimizationFailure
from .policies import choose_stratum, make_strata, optimize_policy

HERE = Path(__file__).resolve().parent
METHODS = ("joint_lookahead", "joint_uniform", "joint_entropy", "joint_neyman",
           "threshold_lookahead", "loose_lookahead", "score_only_lookahead",
           "uniform_cs_lookahead")


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=serial,
                                    separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=serial, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def code_identity():
    # Include the owned data and score builders, not just the new policy file.
    paths = list(HERE.rglob("*.py")) + list(HERE.parent.glob("*.py"))
    paths += list((HERE.parents[1] / "scripts" / "racer_c4").glob("*.py"))
    paths += list((HERE.parents[1] / "scripts" / "racer_c").glob("*.py"))
    root = HERE.parents[2]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if p.is_file()}


def run_policy(public, query, *, method, seed, config):
    """Yield decisions before the evaluator sees their outcomes.

    The only outcome interface available here is query(sample_id). Sampling
    orders are frozen independent uniform permutations within each stratum.
    Different allocation policies share these permutations for paired trials.
    """
    if method not in METHODS:
        raise ValueError(f"unknown policy {method}")
    score = np.asarray(public["score"], float)
    support = np.asarray(public["max_source_tanimoto"], float)
    ids = np.asarray([str(r["sample_id"]) for r in public["rows"]])
    if len(score) != len(ids) or len(set(ids)) != len(ids) or len(ids) == 0:
        raise ValueError("nonempty unique target identities and aligned scores required")
    strata = make_strata(score, support, ids,
                         mode="score_only" if method == "score_only_lookahead" else "score_support",
                         score_bins=config["score_bins"], support_bins=config["support_bins"])
    h_count = int(strata.max()) + 1
    groups = [np.flatnonzero(strata == h) for h in range(h_count)]
    prior = np.asarray([score[g].mean() for g in groups])
    order = np.argsort(prior, kind="stable")
    prior_kind = "uniform" if method == "uniform_cs_lookahead" else config["count_prior"]
    states = [CountCS(len(g), config["delta"] / h_count, prior=prior_kind) for g in groups]
    # Distinct streams prevent a policy's random choices from changing audit orders.
    permutations = [np.random.default_rng(np.random.SeedSequence([int(seed), 91, h])).permutation(g)
                    for h, g in enumerate(groups)]
    choice_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 93]))
    optimizer = "thresholds" if method == "threshold_lookahead" else "loose" if method == "loose_lookahead" else "joint"
    acquisition = {"joint_uniform": "uniform", "joint_entropy": "entropy",
                   "joint_neyman": "neyman"}.get(method, "lookahead")
    checkpoints = sorted(set(int(math.floor(f * len(ids))) for f in config["audit_fractions"]))
    seen = np.zeros(len(ids), bool)
    history = []
    valid = True
    failure = None
    for budget in checkpoints:
        while int(seen.sum()) < budget and valid:
            block = min(int(config["query_block"]), budget - int(seen.sum()))
            h = choose_stratum(states, prior, choice_rng, policy=acquisition,
                               optimizer_mode=optimizer, order=order, block_size=block,
                               eps_fn=config["eps_fn"], eps_fp=config["eps_fp"])
            actual = min(block, states[h].N - states[h].n)
            if actual <= 0:
                raise RuntimeError("allocation selected an exhausted stratum")
            for _ in range(actual):
                i = int(permutations[h][states[h].n])
                if seen[i]:
                    raise RuntimeError("duplicate assay request")
                label = query(str(ids[i]))
                if label not in (0, 1):
                    raise ValueError("oracle must return a binary assay outcome")
                seen[i] = True
                history.append({"sample_id": str(ids[i]), "stratum": h, "label": int(label)})
                try:
                    states[h].observe(label)
                except InconsistentConfidenceSet as error:
                    valid = False
                    failure = str(error)
                    break
        actions = np.full(len(ids), 2, dtype=np.int8)
        if valid:
            try:
                solution = optimize_policy(states, config["eps_fn"], config["eps_fp"],
                                           mode=optimizer, order=order)
                cell_actions = np.where(solution["clear_mask"], 0,
                                        np.where(solution["discard_mask"], 1, 2))
                actions[~seen] = cell_actions[strata[~seen]]
                certificate = solution["certificate"]
            except OptimizationFailure:
                # A numerical failure is not a scientific negative result.
                raise
        else:
            certificate = {"certified": True, "fn_upper": 0., "fp_upper": 0.,
                           "fallback_reason": "empty_confidence_set_all_remaining_assayed"}
            solution = {"saved_items": 0, "status": "empty_cs_fallback"}
        saved = int(np.sum(actions != 2))
        if saved != int(solution["saved_items"]):
            raise RuntimeError("assay savings double-counted observed items")
        yield {"audit_budget": budget, "queried": int(seen.sum()), "n_target": len(ids),
               "n_strata": h_count,
               "saved_items": saved, "saved_fraction": saved / len(ids),
               "total_assays_required": len(ids) - saved,
               "remaining_assays_required": len(ids) - saved - int(seen.sum()),
               "certificate": certificate, "actions": actions,
               "stratum": strata, "query_history": list(history),
               "state": [s.snapshot() if s.valid else {"N": s.N, "n": s.n, "s": s.s,
                                                        "valid": False} for s in states],
               "status": solution["status"], "failure": failure}


class SyntheticCase:
    """Small contract check only, never passed off as molecular evidence."""
    def __init__(self):
        rng = np.random.default_rng(814003)
        self.score = np.repeat([.01, .05, .95, .99], 40)
        self.labels = (rng.random(len(self.score)) < self.score).astype(int)
        self.support = rng.uniform(.1, .9, len(self.score))
        self.ids = [f"synthetic_{i:04d}" for i in range(len(self.score))]
        self.case_id = "synthetic_contract_not_molecular"

    def public_view(self, predictor):
        return {"case_id": self.case_id, "rows": [{"sample_id": s} for s in self.ids],
                "score": self.score, "max_source_tanimoto": self.support,
                "metadata": {"kind": "synthetic_software_check", "predictor": predictor}}

    def new_oracle(self):
        return SyntheticOracle(self.ids, self.labels)


class SyntheticOracle:
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels.copy()
        self.index = {s: i for i, s in enumerate(ids)}
        self.seen = set()

    def query(self, sample_id):
        i = self.index[sample_id]
        if i in self.seen:
            raise ValueError("duplicate query")
        self.seen.add(i)
        return int(self.labels[i])

    def evaluate(self, actions):
        return evaluate_labels(self.labels, actions, self.seen)


def evaluate_labels(labels, actions, queried):
    """Evaluator-only function: never called by query allocation or optimizer."""
    y, a = np.asarray(labels), np.asarray(actions)
    if np.any(a[list(queried)] != 2):
        raise ValueError("audited compounds cannot count as automated decisions")
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    fn, fp = int(((y == 1) & (a == 0)).sum()), int(((y == 0) & (a == 1)).sum())
    return {"false_negative_count": fn, "false_positive_count": fp,
            "class_1_n": n1, "class_0_n": n0,
            "fn_rate": fn / n1 if n1 else None, "fp_rate": fp / n0 if n0 else None}


def normalized_evaluation(oracle, point):
    """Map the trusted data oracle to one explicit report schema."""
    raw = oracle.evaluate(point["actions"])
    if "false_clearance" in raw:
        result = {**raw, "fn_rate": None if raw["active_risk_vacuous"] else raw["false_clearance"],
                  "fp_rate": None if raw["inactive_risk_vacuous"] else raw["false_discard"],
                  "class_1_n": raw["n_active"], "class_0_n": raw["n_inactive"],
                  "false_negative_count": raw["false_clearance_n"],
                  "false_positive_count": raw["false_discard_n"]}
        if (raw["n_total_assays"] != point["total_assays_required"] or
                raw["n_audit_queries"] != point["queried"]):
            raise RuntimeError("oracle and decision algorithm disagree about assay costs")
    else:
        result = raw
    if not {"fn_rate", "fp_rate", "class_0_n", "class_1_n"}.issubset(result):
        raise ValueError("missing actual class-risk metrics")
    return result


def summarize(output, required, identity):
    rows = []
    for job in required:
        path = output / "runs" / (job["job_id"] + ".json")
        receipt = json.loads(path.read_text())
        if receipt["job"] != job or receipt["identity"] != identity:
            raise RuntimeError(f"incompatible completed experiment: {path}")
        for point in receipt["checkpoints"]:
            m = point["metrics"]
            row = {**job, "audit_budget": point["audit_budget"], "queried": point["queried"],
                   "n_target": point["n_target"], "saved_fraction": point["saved_fraction"],
                   "n_strata": point["n_strata"],
                   "saved_items": point["saved_items"], "total_assays_required": point["total_assays_required"],
                   "fn_upper": point["certificate"]["fn_upper"],
                   "fp_upper": point["certificate"]["fp_upper"], **m}
            fn, fp = m.get("fn_rate"), m.get("fp_rate")
            row["violated_realized_class_risk"] = bool((fn is not None and fn > job["eps_fn"] + 1e-12)
                                                      or (fp is not None and fp > job["eps_fp"] + 1e-12))
            for fraction in (.2, .4, .6, .8):
                row[f"fits_total_assay_budget_{fraction:g}"] = point["total_assays_required"] <= math.floor(fraction * point["n_target"])
            rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "all_checkpoint_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    # Only contrasts within the exact same finite pool/fit/audit seed/cap/budget.
    by_key = {}
    for row in rows:
        key = (row["case_id"], row["predictor"], row["audit_seed"], row["eps_fn"], row["eps_fp"], row["audit_budget"])
        by_key.setdefault(key, {})[row["method"]] = row
    paired = []
    for key, methods in by_key.items():
        if "joint_lookahead" not in methods:
            continue
        for name, row in methods.items():
            if name == "joint_lookahead":
                continue
            pair = {k: row[k] for k in ("case_id", "predictor", "audit_seed", "eps_fn", "eps_fp", "audit_budget")}
            pair.update(reference=name, delta_saved_fraction=methods["joint_lookahead"]["saved_fraction"] - row["saved_fraction"])
            paired.append(pair)
    atomic_json(output / "paired_savings.json", paired)
    from .report import build_report
    artifacts = build_report(rows, paired, output, identity["mode"])
    atomic_json(output / "experiment_complete.json", {"status": "complete_declared_matrix",
                "kind": "synthetic_software_check" if identity["mode"] == "pilot" else "retrospective_finite_batch_experiment",
                "jobs": len(required), "checkpoint_rows": len(rows), "identity": identity,
                "report_artifacts": artifacts,
                "interpretation": "Audit-seed repeats quantify random audit behavior on fixed batches; they are not independent chemical samples. Total assays count observed queries plus all remaining deferred compounds."})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "screen", "formal"), default="pilot")
    parser.add_argument("--cache", type=Path, default=Path(".local/chemolab_batch_cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sensitivity", action="store_true", help="predeclared extra predictors and asymmetric risk caps")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.sensitivity and args.mode != "formal":
        parser.error("--sensitivity requires --mode formal and a separate output directory")
    config = json.loads((HERE / "protocol.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    source = code_identity()
    identity = {"mode": args.mode, "sensitivity": args.sensitivity, "protocol": digest(config),
                "code": source, "python": platform.python_version(), "numpy": np.__version__}
    import scipy
    identity["scipy"] = scipy.__version__
    identity["source_fit_jobs"] = args.jobs
    if args.mode == "pilot":
        specs = [{"case_id": "synthetic_contract_not_molecular"}]
        seeds, predictors, caps = [81001], ["stack"], [(.1, .1)]
    else:
        from .data import case_specs
        endpoints = ["AMES", "CYP2D6_Veith"] if args.mode == "screen" else None
        specs = case_specs(endpoints=endpoints, fit_seed=311)
        specs = [{**s, "case_id": f"{s['phase']}__{s['endpoint']}__fit{s['fit_seed']}"} for s in specs]
        seeds = list(range(81001, 81004 if args.mode == "screen" else 81021))
        predictors = ["stack", "logistic", "equal_logit"] if args.sensitivity else ["stack"]
        caps = [(.1, .1), (.05, .1), (.1, .05)] if args.sensitivity else [(.1, .1)]
    required = []
    for spec in specs:
        for predictor in predictors:
            for seed in seeds:
                for fn, fp in caps:
                    for method in METHODS:
                        job = dict(case_id=spec["case_id"], predictor=predictor, audit_seed=seed,
                                   eps_fn=fn, eps_fp=fp, method=method)
                        job["job_id"] = digest(job)[:24]
                        required.append(job)
    manifest = {"identity": identity, "required_jobs": required}
    manifest_path = args.output / "required_jobs.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("existing run differs or --resume missing; use a new output directory")
    else:
        if any(args.output.iterdir()):
            raise RuntimeError("unrecognized nonempty output directory")
        atomic_json(manifest_path, manifest)
    print(f"{args.mode}: {len(specs)} finite pools; {len(required)} audit-policy runs", flush=True)
    started = time.monotonic()
    for spec in specs:
        pending = [j for j in required if j["case_id"] == spec["case_id"] and
                   not (args.output / "runs" / (j["job_id"] + ".json")).exists()]
        if not pending:
            continue
        if args.mode == "pilot":
            case = SyntheticCase()
        else:
            from .data import prepare_case
            case = prepare_case(spec, args.cache / "data", args.cache / "scores",
                                predictors=tuple(predictors), jobs=args.jobs)
        for job in pending:
            public = case.public_view(job["predictor"])
            oracle = case.new_oracle()
            experiment_config = {**config, "eps_fn": job["eps_fn"], "eps_fp": job["eps_fp"]}
            points = []
            for point in run_policy(public, oracle.query, method=job["method"], seed=job["audit_seed"], config=experiment_config):
                point["metrics"] = normalized_evaluation(oracle, point)
                points.append(point)
            atomic_json(args.output / "runs" / (job["job_id"] + ".json"),
                        {"identity": identity, "job": job, "public_metadata": public["metadata"],
                         "public_score_identity": digest({"ids": [r["sample_id"] for r in public["rows"]],
                                                          "scores": public["score"], "support": public["max_source_tanimoto"]}),
                         "checkpoints": points})
            print(f"completed {job['case_id']} {job['method']} audit-seed {job['audit_seed']} "
                  f"cap=({job['eps_fn']},{job['eps_fp']})", flush=True)
    summarize(args.output, required, identity)
    archive = args.output.parent / (args.output.name + "_review.zip")
    temporary = archive.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.output.rglob("*")):
            if path.is_file() and not path.name.endswith(".partial"):
                z.write(path, str(path.relative_to(args.output)))
    temporary.replace(archive)
    print(f"complete; elapsed_this_invocation={time.monotonic()-started:.1f}s; return {archive}", flush=True)


if __name__ == "__main__":
    main()
