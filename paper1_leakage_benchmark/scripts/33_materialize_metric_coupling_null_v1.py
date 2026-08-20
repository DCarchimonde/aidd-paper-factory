from __future__ import annotations

import ast
import hashlib
import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "paper1_leakage_benchmark" / "scripts"
SOURCE = SCRIPTS / "_templates" / "34_run_metric_coupling_null_v1.py.in"
GENERATED_DIR = SCRIPTS / "_generated"
OUTPUT = GENERATED_DIR / "34_run_metric_coupling_null_v1_materialized.py"
MANIFEST = GENERATED_DIR / "34_run_metric_coupling_null_v1_materialized.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_exact(text: str, old: str, new: str, *, label: str, counts: dict[str, int]) -> str:
    occurrences = text.count(old)
    if occurrences != 1:
        raise AssertionError(
            f"The source template no longer matches the audited repair contract for {label}: "
            f"expected exactly 1 occurrence, found {occurrences}. Refuse to guess."
        )
    counts[label] = 1
    return text.replace(old, new, 1)


def repair_source(source: str) -> tuple[str, dict[str, int]]:
    repaired: list[str] = []
    counts = {
        "root_depth": 0,
        "table_header": 0,
        "table_row": 0,
        "subset_aggregation_guard": 0,
        "residual_summary": 0,
    }

    for line in source.splitlines():
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        if stripped == 'ROOT = Path(__file__).resolve().parents[2]':
            repaired.append(indent + 'ROOT = Path(__file__).resolve().parents[3]')
            counts["root_depth"] += 1
            continue

        if stripped.startswith('r"Audit item & Minimum report & Risk if omitted'):
            repaired.append(
                indent + '"Audit item & Minimum report & Risk if omitted " + r"\\\\",'
            )
            counts["table_header"] += 1
            continue

        if stripped.startswith('tex.append(f"{escape(row.audit_item)}'):
            repaired.append(
                indent
                + 'tex.append(f"{escape(row.audit_item)} & {escape(row.minimum_report)} & '
                + '{escape(row.risk_if_omitted)} " + r"\\\\")'
            )
            counts["table_row"] += 1
            continue

        repaired.append(line)

    expected_line_repairs = {"root_depth": 1, "table_header": 1, "table_row": 1}
    actual_line_repairs = {key: counts[key] for key in expected_line_repairs}
    if actual_line_repairs != expected_line_repairs:
        raise AssertionError(
            "The source template no longer matches the audited line-repair contract: "
            f"{actual_line_repairs}. Refuse to guess."
        )

    text = "\n".join(repaired) + "\n"

    old_subset_block = '''    regression = seed_aggregate[seed_aggregate["task_type"].eq("regression")].copy()
    residual = (
        regression["effect_mse"]
        - regression["effect_test_variance"]
        - regression["effect_squared_mean_gap"]
    ).abs()
    if not residual.empty and float(residual.max()) > 1e-9:
        raise AssertionError(f"Aggregate MSE decomposition residual {residual.max()}")
'''
    new_subset_block = '''    regression = seed_aggregate[seed_aggregate["task_type"].eq("regression")].copy()
    required_regression_columns = {
        "effect_mse",
        "effect_test_variance",
        "effect_squared_mean_gap",
    }
    if regression.empty:
        max_residual = 0.0
    else:
        missing_regression_columns = required_regression_columns.difference(regression.columns)
        if missing_regression_columns:
            raise KeyError(
                "Regression rows are present but required decomposition columns are missing: "
                f"{sorted(missing_regression_columns)}"
            )
        residual = (
            regression["effect_mse"]
            - regression["effect_test_variance"]
            - regression["effect_squared_mean_gap"]
        ).abs()
        max_residual = float(residual.max()) if not residual.empty else 0.0
        if max_residual > 1e-9:
            raise AssertionError(f"Aggregate MSE decomposition residual {max_residual}")
'''
    text = replace_exact(
        text,
        old_subset_block,
        new_subset_block,
        label="subset_aggregation_guard",
        counts=counts,
    )

    old_summary = (
        '"max_abs_mse_decomposition_residual": '
        '[float(residual.max()) if not residual.empty else 0.0],'
    )
    new_summary = '"max_abs_mse_decomposition_residual": [max_residual],'
    text = replace_exact(
        text,
        old_summary,
        new_summary,
        label="residual_summary",
        counts=counts,
    )

    if "ROOT = Path(__file__).resolve().parents[3]" not in text:
        raise AssertionError("Generated simulation did not receive the required repository-root repair")
    ast.parse(text, filename=str(OUTPUT))
    return text, counts


def main() -> None:
    if not SOURCE.exists() or SOURCE.stat().st_size == 0:
        raise FileNotFoundError(SOURCE)

    raw = SOURCE.read_bytes()
    source = raw.decode("utf-8")
    materialized, counts = repair_source(source)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(materialized, encoding="utf-8")
    temporary.replace(OUTPUT)
    py_compile.compile(str(OUTPUT), doraise=True)

    payload = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_bytes(raw),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256_bytes(OUTPUT.read_bytes()),
        "repairs": counts,
        "generated_repository_root_depth": 3,
        "status": "syntax_pass",
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("METRIC-COUPLING SOURCE MATERIALIZATION: PASS")
    print("  source:", SOURCE)
    print("  output:", OUTPUT)
    print("  repairs:", counts)
    print("  output_sha256:", payload["output_sha256"])


if __name__ == "__main__":
    main()
