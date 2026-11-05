"""Explain equal-budget expansion through discordant molecular decisions.

This is an outcome-based *evaluation*, not a rule-selection routine. It checks
budgets on the full eligible target cohort before excluding missing outcomes.
Counts across seeds are descriptive repeated-fit summaries, not new molecules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_METHOD = "tame_label1_active"
DEFAULT_REFERENCE = "matched_label1_active_omitted_label_probability"
DEFAULT_BASELINE = "mondrian_lac_empty_to_full"
SIMILARITY_COLUMNS = ("max_train_tanimoto", "max_tanimoto", "training_similarity")
CELL = ["phase", "endpoint", "seed", "predictor", "alpha"]
EXTRA_CELL = ("outer_split", "split_id", "cohort")


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _bins(values: np.ndarray) -> np.ndarray:
    # Fixed 0.1-wide bins; 1.0 belongs to the final closed-right bin.
    index = np.minimum(np.floor(np.asarray(values) * 10).astype(int), 9)
    return np.asarray([f"[{i / 10:.1f},{(i + 1) / 10:.1f}{']' if i == 9 else ')'}" for i in index])


def _validate(frame: pd.DataFrame, method: str, reference: str, baseline: str,
              predictor: str, alpha: float, phase: str | None,
              endpoints: list[str] | None) -> tuple[pd.DataFrame, list[str]]:
    if len({method, reference, baseline}) != 3:
        raise ValueError("method, reference, and baseline identifiers must differ")
    if not 0 < alpha < 1 or not math.isfinite(alpha):
        raise ValueError("alpha must lie in (0,1)")
    out = frame.rename(columns={"model": "predictor"}) if "predictor" not in frame else frame.copy()
    required = {*CELL, "method", "sample_id", "structure_id", "y", "eligible", "include_0", "include_1", "probability_1"}
    if not required.issubset(out):
        raise ValueError(f"missing prediction columns: {sorted(required - set(out))}")
    out = out[out.method.isin([method, reference, baseline]) & (out.predictor == predictor)].copy()
    out["alpha"] = pd.to_numeric(out.alpha, errors="raise")
    out = out[out.alpha == alpha]
    if phase is not None:
        out = out[out.phase == phase]
    if endpoints is not None:
        if not endpoints or len(set(endpoints)) != len(endpoints):
            raise ValueError("endpoint list must be nonempty and unique")
        out = out[out.endpoint.isin(endpoints)]
        if set(out.endpoint) != set(endpoints):
            raise ValueError("requested endpoint panel is incomplete")
    if out.empty:
        raise ValueError("no matching predictions")
    cell = CELL + [c for c in EXTRA_CELL if c in out]
    for column in cell + ["sample_id", "method"]:
        if out[column].isna().any() or out[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"empty identity column: {column}")
    for column in ("eligible", "include_0", "include_1"):
        out[column] = pd.to_numeric(out[column], errors="raise")
        if not out[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be exactly 0/1")
    # Invalid structures still belong to the complete target manifest. Their
    # molecule identity may be blank; eligible identities must be present.
    out["structure_id"] = out.structure_id.astype(object).fillna("").astype(str)
    if out.loc[out.eligible == 1, "structure_id"].str.strip().eq("").any():
        raise ValueError("empty eligible structure identity")
    out["sample_id"] = out.sample_id.astype(str)
    out["seed"] = pd.to_numeric(out.seed, errors="raise")
    if not (np.isfinite(out.seed) & (out.seed == np.floor(out.seed))).all():
        raise ValueError("seed must be a finite integer")
    out["seed"] = out.seed.astype(int)
    out["y"] = pd.to_numeric(out.y, errors="raise").replace(-1, np.nan)
    if not (out.y.isna() | out.y.isin([0, 1])).all():
        raise ValueError("outcomes must be 0/1 or explicit missing")
    out["probability_1"] = pd.to_numeric(out.probability_1, errors="raise")
    if out.duplicated(cell + ["method", "sample_id"]).any():
        raise ValueError("duplicate prediction identity")
    return out, cell


def analyze_predictions(frame: pd.DataFrame, method: str = DEFAULT_METHOD,
                        reference: str = DEFAULT_REFERENCE, baseline: str = DEFAULT_BASELINE,
                        predictor: str = "stack", alpha: float = .1,
                        phase: str | None = None, endpoints: list[str] | None = None) -> dict:
    """Return JSON-serializable report/cell/endpoint/member/stratum records.

    Only same-predictor nested set expansions are supported. A global budget
    match does not imply a budget match in the subset with observed outcomes;
    the efficiency identity therefore includes its observed-budget correction.
    """
    frame, cells = _validate(frame, method, reference, baseline, predictor, alpha, phase, endpoints)
    counts, members, strata = [], [], []
    similarity_available = False
    for key, group in frame.groupby(cells, sort=True, dropna=False):
        metadata = dict(zip(cells, key))
        by_method = {name: group[group.method == name].sort_values("sample_id").reset_index(drop=True)
                     for name in (baseline, method, reference)}
        b, t, p = (by_method[name] for name in (baseline, method, reference))
        if b.empty or any(not f.sample_id.equals(b.sample_id) for f in (t, p)):
            raise ValueError("baseline and compared methods require identical full target IDs")
        for f in (t, p):
            if not f[["structure_id", "eligible", "y"]].equals(b[["structure_id", "eligible", "y"]]):
                raise ValueError("paired structures, eligibility, or outcomes differ")
        eligible = b.eligible.to_numpy(bool)
        b, t, p = (f[eligible].reset_index(drop=True) for f in (b, t, p))
        if b.empty:
            raise ValueError("cell contains no eligible target rows")
        probability = b.probability_1.to_numpy(float)
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("eligible baseline probabilities must be finite in [0,1]")
        if any(not np.allclose(f.probability_1.to_numpy(float), probability, rtol=0, atol=1e-12) for f in (t, p)):
            raise ValueError("mechanism comparison requires identical underlying probabilities")
        sets = [f[["include_0", "include_1"]].to_numpy(bool) for f in (b, t, p)]
        bs, ts, ps = sets
        if np.any(bs.sum(1) == 0):
            raise ValueError("use the common empty-repaired baseline")
        if np.any(bs & ~ts) or np.any(bs & ~ps):
            raise ValueError("compared methods must preserve every baseline label")
        size = bs.sum(1)
        dt, dp = (size == 1) & (ts.sum(1) == 2), (size == 1) & (ps.sum(1) == 2)
        y = b.y.to_numpy(float)
        observed = np.isfinite(y)
        probability_bin = _bins(probability)
        similarity = None
        similarity_bin = None
        for column in SIMILARITY_COLUMNS:
            if column in b:
                candidate = pd.to_numeric(b[column], errors="raise").to_numpy(float)
                if np.isfinite(candidate).all():
                    if np.any((candidate < 0) | (candidate > 1)):
                        raise ValueError("similarity must lie in [0,1]")
                    similarity = candidate
                    similarity_bin = _bins(similarity)
                    similarity_available = True
                    break
        for c in (0, 1):
            pool = (size == 1) & bs[:, c]
            mt, mp = dt & pool, dp & pool
            if mt.sum() != mp.sum():
                raise ValueError(f"class-specific full-cohort budgets differ in {metadata}, class {c}")
            wrong_class, correct_class = y == 1 - c, y == c
            nwrong, ncorrect = int(wrong_class.sum()), int(correct_class.sum())
            et, ep = int(np.sum(mt & wrong_class)), int(np.sum(mp & wrong_class))
            zt, zp = int(np.sum(mt & correct_class)), int(np.sum(mp & correct_class))
            ot, op = int(np.sum(mt & observed)), int(np.sum(mp & observed))
            t_only, p_only = mt & ~mp, mp & ~mt
            delta_false = _fraction(int(np.sum(pool & ~mt & wrong_class)) - int(np.sum(pool & ~mp & wrong_class)), nwrong)
            delta_csy = _fraction(int(np.sum(pool & ~mt & correct_class)) - int(np.sum(pool & ~mp & correct_class)), ncorrect)
            expected_false = _fraction(ep - et, nwrong)
            expected_csy = _fraction(et - ep + op - ot, ncorrect)
            if nwrong and not math.isclose(delta_false, expected_false, abs_tol=1e-12):
                raise AssertionError("false-decision rescue identity failed")
            if ncorrect and not math.isclose(delta_csy, expected_csy, abs_tol=1e-12):
                raise AssertionError("correct-singleton rescue identity failed")
            record = {
                **metadata, "baseline_class": c, "protected_class": 1 - c,
                "n_eligible": len(b), "n_labeled": int(observed.sum()),
                "n_true_active": int(np.sum(y == 1)), "n_true_inactive": int(np.sum(y == 0)),
                "n_baseline_singletons": int(pool.sum()), "defer_budget_method": int(mt.sum()),
                "defer_budget_reference": int(mp.sum()), "both_deferred": int(np.sum(mt & mp)),
                "method_only_deferred": int(t_only.sum()), "reference_only_deferred": int(p_only.sum()),
                "neither_deferred": int(np.sum(pool & ~mt & ~mp)),
                "method_only_true_active": int(np.sum(t_only & (y == 1))),
                "reference_only_true_active": int(np.sum(p_only & (y == 1))),
                "method_only_error_rescues": int(np.sum(t_only & wrong_class)),
                "reference_only_error_rescues": int(np.sum(p_only & wrong_class)),
                "method_error_rescues": et, "reference_error_rescues": ep,
                "method_correct_singletons_lost": zt, "reference_correct_singletons_lost": zp,
                "method_observed_defer_budget": ot, "reference_observed_defer_budget": op,
                "method_missing_outcome_deferred": int(mt.sum()) - ot,
                "reference_missing_outcome_deferred": int(mp.sum()) - op,
                "observed_budget_correction_reference_minus_method": op - ot,
                "error_rescue_increment_method_minus_reference": et - ep,
                "false_decision_exposure_delta": delta_false,
                "false_decision_identity_value": expected_false,
                "correct_singleton_yield_delta": delta_csy,
                "correct_singleton_identity_value": expected_csy,
                "simplified_efficiency_identity_applicable": ot == op,
                "identity_status": "verified" if nwrong and ncorrect else "verified_available_denominators_missing_class_reported",
            }
            counts.append(record)
            for i in np.flatnonzero(pool):
                state = "both" if mt[i] and mp[i] else "method_only" if mt[i] else "reference_only" if mp[i] else "neither"
                members.append({**metadata, "sample_id": b.sample_id.iloc[i], "structure_id": b.structure_id.iloc[i],
                    "baseline_class": c, "deferred_by": state,
                    "y": int(y[i]) if observed[i] else None, "outcome_observed": bool(observed[i]),
                    "probability_1": probability[i], "probability_1_bin": probability_bin[i],
                    "max_train_tanimoto": similarity[i] if similarity is not None else None,
                    "similarity_bin": similarity_bin[i] if similarity_bin is not None else None,
                    "error_rescue": bool(observed[i] and y[i] != c and (mt[i] or mp[i])),
                    "correct_singleton_sacrificed": bool(observed[i] and y[i] == c and (mt[i] or mp[i]))})
            features = [("probability_1", probability_bin)]
            if similarity_bin is not None:
                features.append(("max_train_tanimoto", similarity_bin))
            for feature, bins in features:
                for name in sorted(set(bins[pool])):
                    mask = pool & (bins == name)
                    only_t, only_p = mask & t_only, mask & p_only
                    strata.append({**metadata, "baseline_class": c, "feature": feature, "bin": name,
                        "n_baseline_singletons": int(mask.sum()), "n_labeled": int(np.sum(mask & observed)),
                        "n_true_active": int(np.sum(mask & (y == 1))),
                        "method_only_deferred": int(only_t.sum()), "reference_only_deferred": int(only_p.sum()),
                        "method_only_true_active": int(np.sum(only_t & (y == 1))),
                        "reference_only_true_active": int(np.sum(only_p & (y == 1))),
                        "method_only_error_rescues": int(np.sum(only_t & wrong_class)),
                        "reference_only_error_rescues": int(np.sum(only_p & wrong_class)),
                        "note": "descriptive stratum; budgets need not match within a probability/similarity bin"})
    count_frame = pd.DataFrame(counts)
    summary_keys = [c for c in cells if c != "seed"] + ["baseline_class", "protected_class"]
    summary = []
    mean_columns = ["defer_budget_method", "defer_budget_reference", "method_only_deferred", "reference_only_deferred",
                    "method_only_true_active", "reference_only_true_active", "error_rescue_increment_method_minus_reference",
                    "false_decision_exposure_delta", "correct_singleton_yield_delta", "observed_budget_correction_reference_minus_method"]
    for key, part in count_frame.groupby(summary_keys, sort=True):
        row = {**dict(zip(summary_keys, key)), "n_seed_cells": len(part), "aggregation": "equal_seed_mean_not_independent_counts"}
        for column in mean_columns:
            values = part[column].to_numpy(float)
            row["mean_" + column] = float(np.mean(values)) if np.isfinite(values).all() else None
        summary.append(row)
    return finite_json({
        "report": {"status": "descriptive_mechanism_diagnostics_complete", "method": method, "reference": reference,
            "baseline": baseline, "predictor": predictor, "alpha": alpha,
            "phases": sorted(frame.phase.unique().tolist()), "endpoints": sorted(frame.endpoint.unique().tolist()),
            "n_prediction_cells": len(counts) // 2, "full_eligible_class_budgets_verified": True,
            "similarity_stratification_available": similarity_available,
            "interpretation": "Rescues mean erroneous baseline singleton calls converted to deferral, not corrected positive diagnoses. Missing outcomes remain in budget denominators. Strata are descriptive; repeated seeds are not independent molecules.",
            "identities": {"false_decision_delta": "(reference_error_rescues-method_error_rescues)/n_true_opposite_class",
                "correct_singleton_delta": "(method_error_rescues-reference_error_rescues+reference_observed_budget-method_observed_budget)/n_true_baseline_class"}},
        "cells": counts, "endpoints": summary, "members": members, "strata": strata,
    })


def load_prediction_files(source: Path, methods: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    paths = [source] if source.is_file() else sorted(source.rglob("predictions.csv"))
    if not paths:
        raise FileNotFoundError("no prediction files found")
    frames = []
    for path in paths:
        for chunk in pd.read_csv(path, dtype={"sample_id": str, "structure_id": str}, chunksize=100000):
            if "method" not in chunk:
                raise ValueError(f"missing method column in {path}")
            selected = chunk[chunk.method.isin(methods)]
            if not selected.empty:
                frames.append(selected)
    if not frames:
        raise ValueError("requested diagnostic methods absent")
    return pd.concat(frames, ignore_index=True), paths


def write_diagnostics(result: dict, output: Path, input_paths: list[Path] | None = None) -> dict:
    """Persist diagnostics from analyze_predictions; refuse prior nonempty output."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose a new mechanism output directory")
    output.mkdir(parents=True, exist_ok=True)
    filenames = {"cells": "cell_counts.csv", "endpoints": "endpoint_seed_means.csv",
                 "members": "discordant_members.csv", "strata": "strata.csv"}
    for key, name in filenames.items():
        pd.DataFrame(result[key]).to_csv(output / name, index=False)
    report = {**result["report"], "files": filenames,
              "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (input_paths or [])}}
    (output / "report.json").write_text(json.dumps(finite_json(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--predictor", default="stack")
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--phase")
    parser.add_argument("--endpoints", nargs="+")
    args = parser.parse_args()
    frame, paths = load_prediction_files(args.predictions, [args.method, args.reference, args.baseline])
    result = analyze_predictions(frame, args.method, args.reference, args.baseline,
                                 args.predictor, args.alpha, args.phase, args.endpoints)
    report = write_diagnostics(result, args.output, paths)
    print(json.dumps({"status": report["status"], "output": str(args.output), "cells": report["n_prediction_cells"]}))


if __name__ == "__main__":
    main()
