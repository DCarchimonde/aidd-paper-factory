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


def repair_source(source: str) -> tuple[str, dict[str, int]]:
    repaired: list[str] = []
    counts = {"root_depth": 0, "table_header": 0, "table_row": 0}

    for line in source.splitlines():
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        # The audited template was originally located directly in scripts/, where
        # parents[2] is the repository root. The materialized executable lives one
        # directory deeper in scripts/_generated/, so its repository root is
        # parents[3]. Failing to adjust this makes imports and every output path
        # resolve under paper1_leakage_benchmark/paper1_leakage_benchmark.
        if stripped == "ROOT = Path(__file__).resolve().parents[2]":
            repaired.append(indent + "ROOT = Path(__file__).resolve().parents[3]")
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

    expected = {"root_depth": 1, "table_header": 1, "table_row": 1}
    if counts != expected:
        raise AssertionError(
            "The source template no longer matches the audited repair contract: "
            f"found={counts}, expected={expected}. Refuse to guess."
        )

    text = "\n".join(repaired) + "\n"
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
