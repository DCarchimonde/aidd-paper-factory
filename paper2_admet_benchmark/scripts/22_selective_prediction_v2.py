from __future__ import annotations

"""Selective-prediction analysis with AURC and random-rejection baselines.

Primary rankings are model confidence and chemical-domain similarity. The script
reports standard error/RMSE risk, class-balanced error where defined, class retention,
and a repeated random-rejection baseline at matched coverage.
"""

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
PREDICTION_DIR = PAPER_DIR / "results" / "predictions"
APPLICABILITY_DIR = PAPER_DIR / "results" / "applicability"
SELECTIVE_DIR = PAPER_DIR / "results" / "selective"

COVERAGES = np.round(np.arange(0.10, 1.0001, 0.05), 2).tolist()
RANDOM_REPEATS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selective prediction with AURC and random baselines.")
    parser.add_argument("--mode", default="confirmatory_smoke_seed99")
    parser.add_argument("--pattern", default="*confirm*seed99*_test_predictions.csv")
    parser.add_argument("--applicability-details", default=None)
    parser.add_argument("--endpoints", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--random-repeats", type=int, default=RANDOM_REPEATS)
    return parser.parse_args()


def parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def load_predictions(pattern: str) -> pd.DataFrame:
    files = sorted(PREDICTION_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No test prediction files matched {pattern}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def filter_scope(df: pd.DataFrame, endpoints: list[str] | None, splits: list[str] | None) -> pd.DataFrame:
    out = df.copy()
    if endpoints is not None:
        out = out[out["endpoint"].str.lower().isin({x.lower() for x in endpoints})].copy()
    if splits is not None:
        out = out[out["split_col"].isin(splits)].copy()
    return out


def classification_metrics(group: pd.DataFrame) -> dict[str, float | int | bool]:
    y_true = group["y_true"].astype(int).to_numpy()
    score = np.clip(group["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
    pred = (score >= 0.5).astype(int)
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    error = float(np.mean(pred != y_true))
    if positives > 0 and negatives > 0:
        balanced_error = float(1 - balanced_accuracy_score(y_true, pred))
        fnr = float(np.mean(pred[y_true == 1] == 0))
        fpr = float(np.mean(pred[y_true == 0] == 1))
    else:
        balanced_error = math.nan
        fnr = math.nan if positives == 0 else float(np.mean(pred[y_true == 1] == 0))
        fpr = math.nan if negatives == 0 else float(np.mean(pred[y_true == 0] == 1))
    return {
        "risk": error,
        "error_rate": error,
        "balanced_error_rate": balanced_error,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "positive_count": positives,
        "negative_count": negatives,
        "positive_ratio": float(positives / len(y_true)),
        "single_class_subset": positives == 0 or negatives == 0,
    }


def regression_metrics(group: pd.DataFrame) -> dict[str, float]:
    y_true = group["y_true"].astype(float).to_numpy()
    pred = group["y_output"].astype(float).to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    return {
        "risk": rmse,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, pred)),
    }


def add_uncertainty(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    out["one_minus_max_tanimoto_to_train"] = 1 - out["max_tanimoto_to_train"].astype(float)
    if str(out["task_type"].iloc[0]) == "classification":
        p1 = np.clip(out["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
        out["one_minus_max_probability"] = 1 - np.maximum(p1, 1 - p1)
    return out


def uncertainty_columns(task_type: str) -> list[str]:
    if task_type == "classification":
        return ["one_minus_max_probability", "one_minus_max_tanimoto_to_train"]
    return ["one_minus_max_tanimoto_to_train"]


def evaluate_metrics(group: pd.DataFrame, task_type: str) -> dict[str, float | int | bool]:
    return classification_metrics(group) if task_type == "classification" else regression_metrics(group)


def random_baseline(
    group: pd.DataFrame,
    n_retain: int,
    task_type: str,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    risks: list[float] = []
    balanced_risks: list[float] = []
    for _ in range(repeats):
        chosen = rng.choice(len(group), size=n_retain, replace=False)
        metrics = evaluate_metrics(group.iloc[chosen], task_type)
        risks.append(float(metrics["risk"]))
        if task_type == "classification" and np.isfinite(float(metrics["balanced_error_rate"])):
            balanced_risks.append(float(metrics["balanced_error_rate"]))
    out = {
        "random_risk_mean": float(np.mean(risks)),
        "random_risk_ci_low": float(np.quantile(risks, 0.025)),
        "random_risk_ci_high": float(np.quantile(risks, 0.975)),
    }
    if task_type == "classification" and balanced_risks:
        out.update(
            {
                "random_balanced_error_mean": float(np.mean(balanced_risks)),
                "random_balanced_error_ci_low": float(np.quantile(balanced_risks, 0.025)),
                "random_balanced_error_ci_high": float(np.quantile(balanced_risks, 0.975)),
            }
        )
    return out


def evaluate_curve(
    group: pd.DataFrame,
    uncertainty_col: str,
    repeats: int,
) -> list[dict[str, object]]:
    ranked = group.sort_values(uncertainty_col, ascending=True, kind="mergesort").reset_index(drop=True)
    task_type = str(group["task_type"].iloc[0])
    full_metrics = evaluate_metrics(ranked, task_type)
    full_positive = int(full_metrics.get("positive_count", 0))
    full_negative = int(full_metrics.get("negative_count", 0))
    rows: list[dict[str, object]] = []

    for coverage in COVERAGES:
        n_retain = min(len(ranked), max(1, int(math.ceil(len(ranked) * coverage))))
        retained = ranked.iloc[:n_retain]
        metrics = evaluate_metrics(retained, task_type)
        baseline = random_baseline(
            ranked,
            n_retain,
            task_type,
            repeats,
            stable_seed(
                str(group["endpoint"].iloc[0]),
                str(group["split_col"].iloc[0]),
                str(group["model"].iloc[0]),
                uncertainty_col,
                coverage,
            ),
        )
        row: dict[str, object] = {
            "endpoint": str(group["endpoint"].iloc[0]),
            "task_type": task_type,
            "split_col": str(group["split_col"].iloc[0]),
            "model": str(group["model"].iloc[0]),
            "uncertainty_measure": uncertainty_col,
            "target_coverage": coverage,
            "actual_coverage": float(n_retain / len(ranked)),
            "n_total": int(len(ranked)),
            "n_retained": int(n_retain),
            "mean_uncertainty_retained": float(retained[uncertainty_col].mean()),
            "max_uncertainty_retained": float(retained[uncertainty_col].max()),
            "risk_improvement_vs_random_mean": float(baseline["random_risk_mean"] - float(metrics["risk"])),
            **metrics,
            **baseline,
        }
        if task_type == "classification":
            retained_positive = int(metrics["positive_count"])
            retained_negative = int(metrics["negative_count"])
            row.update(
                {
                    "full_positive_count": full_positive,
                    "full_negative_count": full_negative,
                    "positive_retention_rate": retained_positive / full_positive if full_positive else math.nan,
                    "negative_retention_rate": retained_negative / full_negative if full_negative else math.nan,
                    "class_balance_shift": float(metrics["positive_ratio"]) - float(full_metrics["positive_ratio"]),
                }
            )
        rows.append(row)
    return rows


def integrate_aurc(curve: pd.DataFrame, value_col: str) -> float:
    valid = curve[["actual_coverage", value_col]].dropna().sort_values("actual_coverage")
    if len(valid) < 2:
        return math.nan
    return float(
        np.trapezoid(
            valid[value_col].to_numpy(float),
            valid["actual_coverage"].to_numpy(float),
        )
    )


def summarize_aurc(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["endpoint", "task_type", "split_col", "model", "uncertainty_measure"]
    for keys, group in curves.groupby(group_cols, sort=True):
        endpoint, task_type, split_col, model, uncertainty = keys
        row: dict[str, object] = {
            "endpoint": endpoint,
            "task_type": task_type,
            "split_col": split_col,
            "model": model,
            "uncertainty_measure": uncertainty,
            "aurc_primary_risk": integrate_aurc(group, "risk"),
            "aurc_random_mean": integrate_aurc(group, "random_risk_mean"),
        }
        row["aurc_improvement_vs_random"] = (
            float(row["aurc_random_mean"]) - float(row["aurc_primary_risk"])
        )
        if task_type == "classification":
            row["aurc_balanced_error"] = integrate_aurc(group, "balanced_error_rate")
            row["aurc_random_balanced_error_mean"] = integrate_aurc(
                group, "random_balanced_error_mean"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.random_repeats < 1:
        raise ValueError("random-repeats must be positive")
    predictions = filter_scope(
        load_predictions(args.pattern),
        parse_list(args.endpoints),
        parse_list(args.splits),
    )
    details_path = (
        Path(args.applicability_details)
        if args.applicability_details
        else APPLICABILITY_DIR / f"applicability_details_{args.mode}.csv"
    )
    if not details_path.exists():
        raise FileNotFoundError(details_path)
    applicability = pd.read_csv(details_path)
    applicability = filter_scope(
        applicability[applicability["role"] == "test"].copy(),
        parse_list(args.endpoints),
        parse_list(args.splits),
    )

    merge_cols = ["endpoint", "task_type", "split_col", "row_index"]
    ad_cols = merge_cols + ["max_tanimoto_to_train", "domain_bin", "unseen_scaffold"]
    merged = predictions.merge(applicability[ad_cols], on=merge_cols, how="inner", validate="many_to_one")
    if merged.empty:
        raise RuntimeError("Prediction/applicability merge produced zero rows.")

    rows: list[dict[str, object]] = []
    for _, group in merged.groupby(["endpoint", "task_type", "split_col", "model"], sort=True):
        enriched = add_uncertainty(group)
        task_type = str(group["task_type"].iloc[0])
        for uncertainty in uncertainty_columns(task_type):
            print(
                f"selective-v2 endpoint={group['endpoint'].iloc[0]} split={group['split_col'].iloc[0]} "
                f"model={group['model'].iloc[0]} uncertainty={uncertainty}"
            )
            rows.extend(evaluate_curve(enriched, uncertainty, args.random_repeats))

    curves = pd.DataFrame(rows)
    aurc = summarize_aurc(curves)
    if curves.empty or aurc.empty:
        raise RuntimeError("Selective v2 produced empty outputs.")
    SELECTIVE_DIR.mkdir(parents=True, exist_ok=True)
    curves_path = SELECTIVE_DIR / f"selective_prediction_v2_curves_{args.mode}.csv"
    aurc_path = SELECTIVE_DIR / f"selective_prediction_v2_aurc_{args.mode}.csv"
    curves.to_csv(curves_path, index=False)
    aurc.to_csv(aurc_path, index=False)
    print("saved", curves_path)
    print("saved", aurc_path)
    print(f"Selective prediction v2 complete. Curve rows: {len(curves)}; AURC rows: {len(aurc)}")


if __name__ == "__main__":
    main()
