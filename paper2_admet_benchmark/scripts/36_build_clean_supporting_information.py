from __future__ import annotations

"""Generate and sanitize the Paper 2 Supporting Information LaTeX.

The underlying SI builder contains LaTeX commands in ordinary Python string
literals. On Windows/Python, sequences such as ``\alpha`` and ``\rho`` can be
interpreted as ASCII control characters before they are written to the generated
TeX file. This wrapper runs the frozen-asset SI builder, repairs those escaped
control sequences at the byte level, and fails if any disallowed control byte
remains. It performs no model fitting or statistical analysis.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "paper2_admet_benchmark" / "scripts" / "35_build_supporting_information.py"
OUTPUT = ROOT / "paper2_latex" / "generated_supplementary_tables.tex"


def sanitize_generated_tex(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Generated SI table file was not created: {path}")

    data = path.read_bytes()
    replacements = {
        bytes([0x07]) + b"lpha": b"\\alpha",
        bytes([0x0D]) + b"ho": b"\\rho",
    }
    replacement_counts: dict[str, int] = {}
    for bad, good in replacements.items():
        count = data.count(bad)
        if count:
            data = data.replace(bad, good)
        replacement_counts[bad.hex()] = count

    disallowed = sorted(
        {
            byte
            for byte in data
            if byte < 32 and byte not in {9, 10, 13}
        }
    )
    if disallowed:
        codes = ", ".join(f"0x{byte:02X}" for byte in disallowed)
        raise RuntimeError(
            "Generated SI TeX still contains disallowed control bytes after "
            f"sanitization: {codes}"
        )

    path.write_bytes(data)
    print("sanitized", path)
    print("control-sequence replacements", replacement_counts)


def main() -> None:
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    sanitize_generated_tex(OUTPUT)
    print("Clean Supporting Information LaTeX generation complete.")


if __name__ == "__main__":
    main()
