from __future__ import annotations

"""Count-only exact precision audit for RACER-C pre-freeze decisions."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
DEFAULT_ROLES = P2 / "results" / "racer_c_phase2_preflight" / "role_count_feasibility.csv"
DEFAULT_OUTPUT = P2 / "results" / "racer_c_phase2_preflight"
SELECTED_ALLOCATION = "50_20_15_15"
ASSUMED_COVERAGES = (0.80, 0.90, 0.95)
RETENTION_SCENARIOS = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
ERROR_SCENARIOS = (0.00, 0.02, 0.05, 0.10)
DISCORDANT_RATES = (0.05, 0.10, 0.20)
FAMILYWISE_ALPHA = 0.05
POLICY_TESTS = 36 * 3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def two_sided_cp(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lower, upper


def one_sided_lower(successes: int, n: int, alpha: float) -> float:
    return 0.0 if successes <= 0 or n <= 0 else float(beta.ppf(alpha, successes, n - successes + 1))


def one_sided_upper(successes: int, n: int, alpha: float) -> float:
    if n <= 0 or successes >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, successes + 1, n - successes))


def audit(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    chosen = [row for row in rows if row["allocation"] == SELECTED_ALLOCATION]
    if not chosen:
        raise ValueError(f"allocation {SELECTED_ALLOCATION} absent")
    precision_inputs: list[dict] = []
    intervals: list[dict] = []
    policy: list[dict] = []
    paired: list[dict] = []
    simultaneous_alpha = FAMILYWISE_ALPHA / POLICY_TESTS
    for row in chosen:
        base = {
            "endpoint": row["endpoint"],
            "track": row["track"],
            "seed": row["seed"],
            "allocation": row["allocation"],
        }
        precision_inputs.append(
            {
                **base,
                **{
                    key: row[key]
                    for key in row
                    if key.endswith("_class_0_n") or key.endswith("_class_1_n")
                },
                "primary_count_gate": row["primary_count_gate"],
                "failure_reasons": row["failure_reasons"],
            }
        )
        for population, prefix in (("full_conformal", "conformal"), ("test", "test")):
            for label in (0, 1):
                n = int(row[f"{prefix}_class_{label}_n"])
                for expected in ASSUMED_COVERAGES:
                    successes = min(n, max(0, round(n * expected)))
                    lower, upper = two_sided_cp(successes, n)
                    intervals.append(
                        {
                            **base,
                            "population": population,
                            "true_class": label,
                            "n": n,
                            "assumed_rate": expected,
                            "successes_rounded": successes,
                            "cp95_lower": lower,
                            "cp95_upper": upper,
                            "cp95_width": upper - lower,
                        }
                    )
        for label in (0, 1):
            n = int(row[f"policy_class_{label}_n"])
            for rate in RETENTION_SCENARIOS:
                successes = min(n, max(0, round(n * rate)))
                policy.append(
                    {
                        **base,
                        "constraint": "class_retention_lower",
                        "true_class": label,
                        "n": n,
                        "assumed_retention": "",
                        "assumed_rate": rate,
                        "events_rounded": successes,
                        "per_test_alpha": simultaneous_alpha,
                        "simultaneous_bound": one_sided_lower(
                            successes, n, simultaneous_alpha
                        ),
                        "certifies_primary_limit": str(
                            one_sided_lower(successes, n, simultaneous_alpha) >= 0.50
                        ).lower(),
                    }
                )
            if label == 1:
                for retention in RETENTION_SCENARIOS:
                    selected_n = max(1, math.floor(retention * n))
                    for rate in ERROR_SCENARIOS:
                        errors = min(selected_n, max(0, round(selected_n * rate)))
                        bound = one_sided_upper(errors, selected_n, simultaneous_alpha)
                        policy.append(
                            {
                                **base,
                                "constraint": "critical_selected_error_upper",
                                "true_class": label,
                                "n": selected_n,
                                "assumed_retention": retention,
                                "assumed_rate": rate,
                                "events_rounded": errors,
                                "per_test_alpha": simultaneous_alpha,
                                "simultaneous_bound": bound,
                                "certifies_primary_limit": str(bound <= 0.10).lower(),
                            }
                        )
        for label in (0, 1):
            n = int(row[f"test_class_{label}_n"])
            for discordant_rate in DISCORDANT_RATES:
                discordant_n = max(1, round(n * discordant_rate))
                lower, upper = two_sided_cp(round(discordant_n * 0.5), discordant_n)
                paired.append(
                    {
                        **base,
                        "true_class": label,
                        "test_n": n,
                        "discordant_rate": discordant_rate,
                        "discordant_n_rounded": discordant_n,
                        "paired_difference_ci_width_under_null": (upper - lower)
                        * discordant_n
                        / n,
                    }
                )
    return precision_inputs, intervals, policy, paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv(args.roles)
    inputs, intervals, policy, paired = audit(rows)
    write_csv(args.output_dir / "precision_power_inputs.csv", inputs)
    write_csv(args.output_dir / "exact_interval_precision.csv", intervals)
    write_csv(args.output_dir / "policy_grid_error_control.csv", policy)
    write_csv(args.output_dir / "paired_effect_simulation.csv", paired)
    decision = {
        "status": "pre_freeze_precision_complete",
        "selected_allocation": SELECTED_ALLOCATION,
        "selection_uses_model_outputs": False,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "policy_candidate_pairs": 36,
        "simultaneous_constraints_per_pair": 3,
        "bonferroni_tests": POLICY_TESTS,
        "retention_floor": 0.50,
        "critical_selected_error_ceiling": 0.10,
        "primary_count_rule": "all tracks and seeds pass under selected allocation",
        "role_count_input_sha256": sha256_file(args.roles),
        "precision_power_inputs_sha256": sha256_file(args.output_dir / "precision_power_inputs.csv"),
        "exact_interval_precision_sha256": sha256_file(args.output_dir / "exact_interval_precision.csv"),
        "policy_grid_error_control_sha256": sha256_file(args.output_dir / "policy_grid_error_control.csv"),
        "paired_effect_simulation_sha256": sha256_file(args.output_dir / "paired_effect_simulation.csv"),
    }
    (args.output_dir / "precision_policy_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
