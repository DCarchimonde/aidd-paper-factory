"""Prepare paired NCATS curves and compare four CPU onset-separation models."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
import zipfile

import numpy as np

from .model import CensoredPairModel
from .targets import (DEFAULT_EDGES, DEFAULT_MARGIN_LOG10, onset_interval,
                      pair_allowed_mask, window_cell_masks, window_reference_bounds)

HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = "tox21-ar-mda-kb2-luc-antagonist-p1"
METHODS = ("joint_censored", "independent_censored", "point_imputed", "binary_window")
SPLITS = ("random", "scaffold")


def _serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    default=_serial, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_serial,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_gzip_json(path, value):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, default=_serial, allow_nan=False).encode()
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(gzip.compress(data, mtime=0))
    temporary.replace(path)


def code_identity():
    files = sorted(HERE.glob("*.py")) + sorted(HERE.glob("sources.json")) + sorted(HERE.glob("protocol.json"))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def runtime_identity():
    result = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "scikit-learn", "rdkit"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not_installed"
    return result


def molecule_weights(records):
    counts = Counter(str(r["group_id"]) for r in records)
    return np.asarray([1.0 / counts[str(r["group_id"])] for r in records])


def resolved_molecule_weights(records):
    counts = Counter(str(r["group_id"]) for r in records
                     if r["window_reference"]["status"] != "unresolved")
    return np.asarray([0.0 if r["window_reference"]["status"] == "unresolved"
                       else 1.0 / counts[str(r["group_id"])] for r in records])


def status_molecule_weights(records):
    keys = [(str(r["group_id"]), r["window_reference"]["status"]) for r in records]
    counts = Counter(keys)
    return np.asarray([1.0 / counts[key] for key in keys])


def derive_records(records, target_threshold=30.0, viability_threshold=20.0,
                   margin=DEFAULT_MARGIN_LOG10, edges=DEFAULT_EDGES):
    output, masks = [], []
    for row in records:
        target = onset_interval(row["concentrations_m"], row["target_responses_pct"],
                                target_threshold, direction=-1)
        viability = onset_interval(row["concentrations_m"], row["viability_responses_pct"],
                                   viability_threshold, direction=-1)
        reference = window_reference_bounds(target, viability, margin_log10=margin)
        output.append({**row, "target_onset": target, "viability_onset": viability,
                       "window_reference": reference})
        masks.append(pair_allowed_mask(target, viability, edges))
    if not output:
        raise ValueError("no paired records are available")
    return output, np.asarray(masks, dtype=bool)


def describe_records(records):
    statuses = Counter(r["window_reference"]["status"] for r in records)
    by_group = defaultdict(list)
    for r in records:
        by_group[(str(r["group_id"]), r["protocol"])].append(r["window_reference"]["status"])
    repeats = [values for values in by_group.values() if len(values) >= 2]
    resolved_repeats = [v for v in repeats if "unresolved" not in v]
    sample_profiles = []
    for (sample, protocol), values in _sample_protocol_groups(records).items():
        values = sorted(values, key=lambda r: int(r["replicate"]))
        statuses_here = [r["window_reference"]["status"] for r in values]
        sample_profiles.append({
            "sample_id": sample, "protocol": protocol,
            "replicates": [int(r["replicate"]) for r in values],
            "reference_statuses": statuses_here,
            "complete_three_repeats": len(values) == 3 and {int(r["replicate"]) for r in values} == {1, 2, 3},
            "resolved_status_disagreement": {"definite_yes", "definite_no"}.issubset(statuses_here),
        })
    complete_profiles = [p for p in sample_profiles if p["complete_three_repeats"]]
    return {
        "paired_rows": len(records), "molecular_groups": len({r["group_id"] for r in records}),
        "reference_status_counts": dict(statuses),
        "target_censoring": dict(Counter(r["target_onset"]["censoring"] for r in records)),
        "viability_censoring": dict(Counter(r["viability_onset"]["censoring"] for r in records)),
        "curves_with_target_recrossing": sum(r["target_onset"]["recrossings"] > 0 for r in records),
        "curves_with_viability_recrossing": sum(r["viability_onset"]["recrossings"] > 0 for r in records),
        "group_protocol_units_with_repeats": len(repeats),
        "fully_resolved_repeat_units": len(resolved_repeats),
        "fully_resolved_repeat_units_with_disagreement": sum(len(set(v)) > 1 for v in resolved_repeats),
        "repeat_unit_definition": "same parent structure and protocol; original repeats are not averaged",
        "complete_three_repeat_sample_protocol_units": len(complete_profiles),
        "three_repeats_all_definite_yes": sum(set(p["reference_statuses"]) == {"definite_yes"} for p in complete_profiles),
        "three_repeats_all_definite_no": sum(set(p["reference_statuses"]) == {"definite_no"} for p in complete_profiles),
        "three_repeats_with_unresolved": sum("unresolved" in p["reference_statuses"] for p in complete_profiles),
        "three_repeats_with_yes_no_conflict": sum(p["resolved_status_disagreement"] for p in complete_profiles),
        "sample_protocol_repeat_profiles": sample_profiles,
    }


def _sample_protocol_groups(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["sample_id"], row["protocol"])].append(row)
    return grouped


def split_indices(records, split, seed=9021, test_fraction=.25):
    """Hash-order groups without using outcome, concentration, or response data."""
    if split not in SPLITS or not 0 < test_fraction < 1:
        raise ValueError("invalid split or test fraction")
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[str(r["group_id"])].append(i)
    if len(groups) < 4:
        raise ValueError("at least four molecular groups are required")
    buckets = defaultdict(list)
    for group, indices in groups.items():
        if split == "random":
            key = group
        else:
            values = {str(records[i]["scaffold_id"]) for i in indices}
            if len(values) != 1:
                raise ValueError("one parent molecular group has inconsistent scaffolds")
            # All empty scaffolds remain one bucket, preserving group disjointness.
            key = next(iter(values)) or "EMPTY_ACYCLIC_SCAFFOLD"
        buckets[key].append(group)
    if len(buckets) < 2:
        raise ValueError("fewer than two independent split buckets")
    ordering = sorted(buckets, key=lambda x: digest(["window_split", seed, split, x]))
    wanted = max(1, math.ceil(test_fraction * len(groups)))
    test_groups = set()
    for key in ordering[:-1]:
        if len(test_groups) >= wanted:
            break
        test_groups.update(buckets[key])
    test = np.asarray([i for i, r in enumerate(records) if str(r["group_id"]) in test_groups], dtype=int)
    train = np.asarray([i for i, r in enumerate(records) if str(r["group_id"]) not in test_groups], dtype=int)
    if not len(test) or not len(train):
        raise ValueError("empty training or test split")
    return train, test


def _design_features(records):
    """Only declared concentration design, never response-dependent counts."""
    rows = []
    for r in records:
        conc = np.asarray([float(x) for x in r["concentrations_m"]
                           if x is not None and math.isfinite(float(x)) and float(x) > 0])
        if not len(conc):
            raise ValueError("concentration design unavailable")
        rows.append([float(np.log10(conc.min())), float(np.log10(conc.max()))])
    return np.asarray(rows)


def build_features(records, train_indices, synthetic=False):
    """Freeze chemical reduction and protocol vocabulary using training only."""
    design = _design_features(records)
    if synthetic:
        chemical = np.asarray([r["synthetic_features"] for r in records], dtype=float)
        feature_info = {"kind": "synthetic_features_for_software_check"}
    else:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        from scipy.sparse import csr_matrix
        from sklearn.decomposition import TruncatedSVD
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        smiles_values = sorted({r["canonical_smiles"] for r in records})
        index = {s: i for i, s in enumerate(smiles_values)}
        sparse_rows, columns = [], []
        for i, smiles in enumerate(smiles_values):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError("prepared canonical SMILES cannot be parsed")
            bits = generator.GetFingerprint(molecule).GetOnBits()
            sparse_rows.extend([i] * len(bits))
            columns.extend(bits)
        matrix = csr_matrix((np.ones(len(columns)), (sparse_rows, columns)), shape=(len(smiles_values), 2048))
        row_lookup = np.asarray([index[r["canonical_smiles"]] for r in records])
        training_structures = np.unique(row_lookup[train_indices])
        components = max(1, min(32, len(training_structures) - 1, matrix.shape[1] - 1))
        reduction = TruncatedSVD(n_components=components, random_state=9021)
        reduction.fit(matrix[training_structures])
        chemical = reduction.transform(matrix)[row_lookup]
        feature_info = {"kind": "Morgan_radius2_2048_training_only_SVD", "components": components,
                        "fit_unique_structures": len(training_structures),
                        "svd_components_sha256": hashlib.sha256(reduction.components_.tobytes()).hexdigest()}
    vocabulary = sorted({records[int(i)]["protocol"] for i in train_indices})
    context = np.asarray([[float(r["protocol"] == p) for p in vocabulary] for r in records])
    return np.column_stack((chemical, design, context)), {
        **feature_info, "protocol_vocabulary": vocabulary,
        "design_inputs": ["minimum_tested_log10_M", "maximum_tested_log10_M"],
        "curve_response_and_fitted_AC50_are_predictors": False,
    }


def point_impute_masks(masks):
    """Declared point-imputation comparator; censored intervals use grid mid-bin."""
    hard = np.zeros_like(masks)
    for i, mask in enumerate(masks):
        if mask.all():
            hard[i] = True
            continue
        target = np.flatnonzero(mask.any(axis=1))
        viability = np.flatnonzero(mask.any(axis=0))
        hard[i, target[len(target) // 2], viability[len(viability) // 2]] = True
    return hard


def _weighted_mean(values, weights):
    return float(np.average(np.asarray(values), weights=weights)) if len(values) else None


def reference_counts(records):
    return {status: {
        "rows": sum(r["window_reference"]["status"] == status for r in records),
        "molecular_groups": len({r["group_id"] for r in records if r["window_reference"]["status"] == status})}
        for status in ("definite_yes", "definite_no", "unresolved")}


def probability_metrics(records, lower, upper, subset=None):
    from sklearn.metrics import average_precision_score, balanced_accuracy_score
    selection = np.arange(len(records)) if subset is None else np.asarray(subset, dtype=int)
    selected = [records[int(i)] for i in selection]
    if not selected:
        return {"rows": 0, "resolved_rows": 0, "brier": None, "average_precision": None,
                "balanced_accuracy_at_0_5": None}
    weights = molecule_weights(selected)
    statuses = np.asarray([r["window_reference"]["status"] for r in selected])
    resolved = statuses != "unresolved"
    lo, hi = np.asarray(lower)[selection], np.asarray(upper)[selection]
    result = {"rows": len(selected), "molecular_groups": len({r["group_id"] for r in selected}),
              "resolved_rows": int(resolved.sum()), "resolved_fraction_molecule_weighted": _weighted_mean(resolved, weights),
              "mean_probability_resolution_width": _weighted_mean(hi - lo, weights),
              "score_definition": "midpoint of model-implied concentration-grid resolution interval"}
    midpoint = (lo + hi) / 2
    y_known = (statuses == "definite_yes").astype(int)
    known_loss = (midpoint - y_known) ** 2
    lower_loss = np.where(resolved, known_loss, np.minimum(midpoint ** 2, (1-midpoint) ** 2))
    upper_loss = np.where(resolved, known_loss, np.maximum(midpoint ** 2, (1-midpoint) ** 2))
    result["all_reference_brier_lower"] = _weighted_mean(lower_loss, weights)
    result["all_reference_brier_upper"] = _weighted_mean(upper_loss, weights)
    if resolved.any():
        y = (statuses[resolved] == "definite_yes").astype(int)
        probability = ((lo + hi) / 2)[resolved]
        w = resolved_molecule_weights(selected)[resolved]
        result.update(reference_positive_rows=int(y.sum()), reference_negative_rows=int((1-y).sum()),
                      brier=_weighted_mean((probability - y) ** 2, w),
                      average_precision=float(average_precision_score(y, probability, sample_weight=w)) if y.sum() else None,
                      balanced_accuracy_at_0_5=float(balanced_accuracy_score(y, probability >= .5, sample_weight=w)) if len(set(y)) == 2 else None)
    else:
        result.update(reference_positive_rows=0, reference_negative_rows=0,
                      brier=None, average_precision=None, balanced_accuracy_at_0_5=None)
    return result


def fit_evaluate(method, X, records, masks, train, test, *, edges, margin, max_iter):
    if method not in METHODS:
        raise ValueError("unknown comparison method")
    train_records, test_records = [records[int(i)] for i in train], [records[int(i)] for i in test]
    weights = molecule_weights(train_records)
    test_weights = molecule_weights(test_records)
    nll = informative_nll = None
    observation_losses = np.full(len(test), np.nan)
    model = None
    start = time.monotonic()
    if method == "binary_window":
        from sklearn.linear_model import LogisticRegression
        labels = np.asarray([1 if r["window_reference"]["status"] == "definite_yes" else
                             0 if r["window_reference"]["status"] == "definite_no" else -1 for r in train_records])
        observed = labels >= 0
        if not observed.any():
            return unavailable_result(test_records, "no_resolved_binary_training_reference", train_records)
        fit_x = X[train][observed]
        fit_weight = weights[observed] / weights[observed].sum()
        mean = fit_weight @ fit_x
        scale = np.sqrt(fit_weight @ np.square(fit_x - mean))
        scale[scale <= np.finfo(float).eps] = 1.0
        if len(set(labels[observed])) < 2:
            q = np.full(len(test), float(labels[observed][0]))
            convergence = {"converged": True, "status": "constant_single_training_class", "iterations": 0}
        else:
            estimator = LogisticRegression(C=1.0, max_iter=max_iter, random_state=9021)
            estimator.fit((fit_x-mean)/scale, labels[observed], sample_weight=weights[observed])
            q = estimator.predict_proba((X[test]-mean)/scale)[:, 1]
            convergence = {"converged": bool(int(estimator.n_iter_[0]) < max_iter),
                           "status": "logistic_fit", "iterations": int(estimator.n_iter_[0])}
        lower, upper = q, q
    else:
        training_masks = point_impute_masks(masks[train]) if method == "point_imputed" else masks[train]
        if masks[train].all(axis=(1, 2)).all():
            return unavailable_result(test_records, "no_informative_training_observations", train_records)
        model = CensoredPairModel(len(edges) + 1, association=method == "joint_censored", max_iter=max_iter)
        model.fit(X[train], training_masks, sample_weight=weights)
        joint = model.predict_joint(X[test])
        observation_losses = model.observation_losses(X[test], masks[test])
        nll = _weighted_mean(observation_losses, test_weights)
        informative = ~masks[test].all(axis=(1, 2))
        if informative.any():
            informative_nll = _weighted_mean(observation_losses[informative], test_weights[informative])
        lower, upper = [], []
        for i, r in enumerate(test_records):
            ht = r["target_onset"]["concentration_max_log_m"]
            hv = r["viability_onset"]["concentration_max_log_m"]
            if ht is None or hv is None:
                lower.append(0.0); upper.append(1.0)
                continue
            guaranteed, possible = window_cell_masks(edges, ht, hv, margin)
            lower.append(float(joint[i][guaranteed].sum()))
            upper.append(float(joint[i][possible].sum()))
        lower, upper = np.asarray(lower), np.asarray(upper)
        convergence = {"converged": model.converged_, "status": model.optimization_message_,
                       "iterations": model.n_iter_, "informative_training_rows": model.n_informative_}
    metrics = {"masked_nll_molecule_weighted": nll,
               "informative_masked_nll_molecule_weighted": informative_nll,
               "training_reference_counts": reference_counts(train_records),
               "all_repeats": probability_metrics(test_records, lower, upper),
               "repeat3_reference": probability_metrics(test_records, lower, upper,
                       [i for i, r in enumerate(test_records) if int(r["replicate"]) == 3]),
               "elapsed_seconds": time.monotonic() - start, "convergence": convergence}
    for key in ("all_repeats", "repeat3_reference"):
        metrics[key]["probability_bound_kind"] = "binary_point_prediction" if method == "binary_window" else "concentration_grid_resolution"
        if method == "binary_window":
            metrics[key]["mean_probability_resolution_width"] = None
            metrics[key]["score_definition"] = "direct binary probability; concentration-grid interval is not applicable"
    metrics["by_reference_status"] = {}
    for status in ("definite_yes", "definite_no", "unresolved"):
        take = np.asarray([r["window_reference"]["status"] == status for r in test_records])
        subset_records = [r for r, keep in zip(test_records, take) if keep]
        metrics["by_reference_status"][status] = {
            "rows": int(take.sum()), "molecular_groups": len({r["group_id"] for r in subset_records}),
            "masked_nll_molecule_weighted": _weighted_mean(observation_losses[take], molecule_weights(subset_records))
                if model is not None and take.any() else None}
    predictions = []
    resolved_weights = resolved_molecule_weights(test_records)
    status_weights = status_molecule_weights(test_records)
    for i, (row, lo, hi) in enumerate(zip(test_records, lower, upper)):
        predictions.append({key: row[key] for key in ("group_id", "compound_id", "sample_id", "protocol", "replicate", "target_curve_id", "viability_curve_id")})
        predictions[-1].update(window_reference=row["window_reference"]["status"],
                               q_lower=float(lo), q_upper=float(hi), q_midpoint=float((lo+hi)/2),
                               masked_nll=float(observation_losses[i]) if model is not None else None,
                               informative_observation=bool(not masks[int(test[i])].all()),
                               test_molecule_weight=float(test_weights[i]),
                               resolved_molecule_weight=float(resolved_weights[i]),
                               status_molecule_weight=float(status_weights[i]))
    return metrics, predictions


def unavailable_result(test_records, reason, train_records=()):
    """Retain an unavailable comparison without inventing a probability model."""
    metrics = {"masked_nll_molecule_weighted": None, "informative_masked_nll_molecule_weighted": None,
               "training_reference_counts": reference_counts(train_records),
               "all_repeats": {"rows": len(test_records), "brier": None, "average_precision": None},
               "repeat3_reference": {"rows": sum(int(r["replicate"]) == 3 for r in test_records),
                                     "brier": None, "average_precision": None},
               "by_reference_status": {status: {**counts, "masked_nll_molecule_weighted": None}
                                       for status, counts in reference_counts(test_records).items()},
               "elapsed_seconds": 0.0,
               "convergence": {"converged": False, "status": "not_evaluable", "reason": reason, "iterations": 0}}
    predictions = []
    for row, weight, resolved_weight, status_weight in zip(
            test_records, molecule_weights(test_records), resolved_molecule_weights(test_records),
            status_molecule_weights(test_records)):
        p = {key: row[key] for key in ("group_id", "compound_id", "sample_id", "protocol", "replicate", "target_curve_id", "viability_curve_id")}
        p.update(window_reference=row["window_reference"]["status"], q_lower=None, q_upper=None, q_midpoint=None,
                 masked_nll=None, informative_observation=None, test_molecule_weight=float(weight),
                 resolved_molecule_weight=float(resolved_weight), status_molecule_weight=float(status_weight))
        predictions.append(p)
    return metrics, predictions


def synthetic_records():
    """Legal paired curves for software integration, not chemical evidence."""
    rng = np.random.default_rng(99117)
    concentration = np.logspace(-9, -4, 12)
    rows = []
    for compound in range(48):
        feature = rng.normal(size=3)
        for repeat in (1, 2, 3):
            t = int(np.clip(5 + 2 * feature[0] + rng.normal(scale=.4), 0, 12))
            v = int(np.clip(t + 3 * feature[1] + rng.normal(scale=.5), 0, 12))
            target = [-60.0 if j >= t else 0.0 for j in range(12)]
            viability = [-60.0 if j >= v else 0.0 for j in range(12)]
            group = f"synthetic_{compound:04d}"
            rows.append({"compound_id": group, "group_id": group, "sample_id": group,
                         "protocol": "synthetic_paired_curve_check", "replicate": repeat,
                         "target_curve_id": f"{group}_t{repeat}", "viability_curve_id": f"{group}_v{repeat}",
                         "canonical_smiles": "synthetic_not_a_molecule", "smiles": "synthetic_not_a_molecule",
                         "scaffold_id": f"synthetic_scaffold_{compound//3:02d}",
                         "concentrations_m": concentration.tolist(), "target_responses_pct": target,
                         "viability_responses_pct": viability, "purity": "synthetic",
                         "official_outcomes": {}, "synthetic_features": feature.tolist()})
    return rows


def write_summary(output, jobs, identity, mode):
    summary, all_predictions = [], []
    for job in jobs:
        receipt = json.loads((output / "runs" / f"{job['job_id']}.json").read_text())
        if receipt["identity"] != identity or receipt["job"] != job:
            raise ValueError("completed run does not match this experiment")
        summary.append({**job, **receipt["metrics"]})
        all_predictions.extend({**job, **row} for row in receipt["predictions"])
    write_json(output / "metrics.json", summary)
    if all_predictions:
        with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_predictions[0]))
            writer.writeheader(); writer.writerows(all_predictions)
    nonconverged = [r["job_id"] for r in summary if not r["convergence"]["converged"]]
    write_json(output / "experiment_complete.json", {
        "mode": mode, "kind": "synthetic_software_check" if mode == "pilot" else "retrospective_paired_assay_experiment",
        "status": "complete_with_nonconverged_fits" if nonconverged else "complete_declared_matrix",
        "completed_jobs": len(summary), "required_jobs": len(jobs), "nonconverged_jobs": nonconverged,
        "identity": identity})
    lines = ["# 配对起始响应间隔实验读表", "",
             "合成流程检查，不能作为分子结果。" if mode == "pilot" else "结果来自原始配对浓度读数的回顾性分子划分。",
             "", "主比较为 joint_censored 对 independent_censored 的相同观测mask似然。每个分子组在各集合中的重复权重合计为1。",
             "目标为报告响应首次跨阈值的起始浓度早于viability损失起始浓度，间隔满足预设倍数，且测试范围覆盖该间隔。首次跨越后的回落单独记录；起始浓度间隔不代表整段浓度持续有效。",
             "q_lower/q_upper描述模型概率在浓度网格分辨率下的界；其中点仅用于明确标记的分类指标，不是分布无关风险证书。",
             "重复3单独作为预设参考展示；所有重复仍随整个分子组划分，训练没有读取测试分子的其他重复。",
             "二元baseline仅在训练参考明确的行拟合，并以相同测试参考评价；其joint-mask NLL不适用。",
             "", "| 划分 | 方法 | Mask NLL | 重复3 Brier | 重复3 AUPRC | 收敛 |",
             "|---|---|---:|---:|---:|---|"]
    def display(value):
        return "NA" if value is None else f"{value:.5f}"
    for row in summary:
        ref = row["repeat3_reference"]
        lines.append(f"| {row['split']} | {row['method']} | {display(row['masked_nll_molecule_weighted'])} | {display(ref.get('brier'))} | {display(ref.get('average_precision'))} | {row['convergence']['converged']} |")
    lines += ["", "同时阅读 metrics.json 中 definite_yes、definite_no、unresolved 三组的 NLL，及 data_profile.json 的三次复测一致性；整体 NLL 改善本身不能证明阳性起始响应间隔预测改善。",
              "未收敛的拟合保留输出并单列，不能作为有效方法比较完成。原始配对曲线、参考状态、划分清单和逐行预测均随包保存。", ""]
    (output / "READOUT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def archive_output(output):
    archive = output.with_name(output.name + "_review.zip")
    temp = archive.with_name(archive.name + ".partial")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and not path.name.endswith(".partial"):
                handle.write(path, str(path.relative_to(output)))
    temp.replace(archive)
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "prepare", "screen"), default="pilot")
    parser.add_argument("--protocol", action="append", help="repeat for additional verified protocols")
    parser.add_argument("--cache", type=Path, default=Path(".local/assay_window_cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4, help="numerical-library thread limit")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.max_iter < 1:
        parser.error("jobs and max-iter must be positive")
    protocols = args.protocol or [DEFAULT_PROTOCOL]
    if len(set(protocols)) != len(protocols):
        parser.error("protocols must be unique")
    protocol_spec = json.loads((HERE / "protocol.json").read_text())
    if (protocol_spec["default_protocol"] != DEFAULT_PROTOCOL or
            protocol_spec["primary_target_threshold_pct"] != 30.0 or
            protocol_spec["primary_viability_loss_threshold_pct"] != 20.0 or
            protocol_spec["response_direction"] != -1 or
            protocol_spec["separation_factor"] != 3.0 or
            not np.array_equal(protocol_spec["grid_log10_molar_edges"], DEFAULT_EDGES)):
        raise ValueError("protocol.json and implemented primary targets differ")
    config = {"mode": args.mode, "protocols": protocols, "target_threshold_pct": 30.0,
              "viability_threshold_pct": 20.0, "margin_log10": DEFAULT_MARGIN_LOG10,
              "edges_log10_M": DEFAULT_EDGES.tolist(), "seed": 9021,
              "test_fraction": .25, "max_iter": args.max_iter, "threads": args.jobs,
              "purity": ["A", "B"], "methods": METHODS, "code": code_identity(),
              "runtime": runtime_identity(), "protocol_spec": protocol_spec}
    config = json.loads(json.dumps(config))
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "configuration.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise ValueError("existing output differs or --resume missing; use a separate output directory")
    elif any(args.output.iterdir()):
        raise ValueError("unrecognized nonempty output directory")
    else:
        write_json(config_path, config)
    paired_path = args.output / "paired_records.json.gz"
    if paired_path.exists() and (args.output / "source_diagnostics.json").exists():
        records = json.loads(gzip.decompress(paired_path.read_bytes()))
        diagnostics = json.loads((args.output / "source_diagnostics.json").read_text())
    else:
        if args.mode == "pilot":
            records, diagnostics = synthetic_records(), {"kind": "synthetic_software_check"}
        else:
            from .data import prepare_protocol
            records, diagnostics = [], {}
            for protocol in protocols:
                block, detail = prepare_protocol(protocol, args.cache)
                records.extend(block); diagnostics[protocol] = detail
        write_gzip_json(paired_path, records)
        write_json(args.output / "source_diagnostics.json", diagnostics)
    derived, masks = derive_records(records)
    identity = {"configuration_sha256": digest(config), "paired_records_sha256": digest(records)}
    write_gzip_json(args.output / "derived_records.json.gz", derived)
    write_json(args.output / "data_profile.json", describe_records(derived))
    if args.mode == "prepare":
        write_json(args.output / "experiment_complete.json", {"status": "data_preparation_complete",
                   "identity": identity, "paired_rows": len(derived), "model_fits": 0})
        profile = describe_records(derived)
        readout = ["# 配对曲线准备结果", "", f"配对记录：{len(derived)}；分子组：{profile['molecular_groups']}；模型拟合：0。",
                   f"参考状态：{profile['reference_status_counts']}。",
                   f"完整三次复测样本：{profile['complete_three_repeat_sample_protocol_units']}；三次均明确阳性：{profile['three_repeats_all_definite_yes']}；存在明确阳性/阴性冲突：{profile['three_repeats_with_yes_no_conflict']}。",
                   "", "参考阳性行数包含重复测量。请结合 sample_protocol_repeat_profiles 评估复测一致性。",
                   "本包含原始配对浓度读数、起始响应区间、参考状态、来源诊断和纯度筛选结果。", ""]
        (args.output / "READOUT_CN.md").write_text("\n".join(readout), encoding="utf-8")
        print(f"data preparation complete; return {archive_output(args.output)}", flush=True)
        return
    jobs = []
    for split in SPLITS:
        for method in METHODS:
            job = {"split": split, "seed": 9021, "method": method}
            job["job_id"] = digest(job)[:20]
            jobs.append(job)
    required = {"identity": identity, "jobs": jobs}
    required_path = args.output / "required_jobs.json"
    if required_path.exists() and json.loads(required_path.read_text()) != required:
        raise ValueError("prepared records or declared matrix changed since this run")
    write_json(required_path, required)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.jobs):
        for split in SPLITS:
            pending = [j for j in jobs if j["split"] == split and not (args.output / "runs" / f"{j['job_id']}.json").exists()]
            if not pending:
                continue
            train, test = split_indices(derived, split)
            X, feature_info = build_features(derived, train, synthetic=args.mode == "pilot")
            manifest = {"split": split, "seed": 9021, "train_group_ids": sorted({derived[int(i)]["group_id"] for i in train}),
                        "test_group_ids": sorted({derived[int(i)]["group_id"] for i in test}),
                        "train_rows": len(train), "test_rows": len(test), "feature_info": feature_info}
            if set(manifest["train_group_ids"]) & set(manifest["test_group_ids"]):
                raise RuntimeError("molecular group leakage")
            write_json(args.output / f"split_{split}.json", manifest)
            for job in pending:
                metrics, predictions = fit_evaluate(job["method"], X, derived, masks, train, test,
                    edges=DEFAULT_EDGES, margin=DEFAULT_MARGIN_LOG10, max_iter=args.max_iter)
                write_json(args.output / "runs" / f"{job['job_id']}.json",
                           {"identity": identity, "job": job, "metrics": metrics, "predictions": predictions})
                print(f"completed {split} {job['method']}; converged={metrics['convergence']['converged']}", flush=True)
    write_summary(args.output, jobs, identity, args.mode)
    completion = json.loads((args.output / "experiment_complete.json").read_text())
    print(f"{completion['status']}; return {archive_output(args.output)}", flush=True)


if __name__ == "__main__":
    main()
