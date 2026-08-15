from __future__ import annotations

"""Generate, sanitize, and typeset-polish the Paper 2 SI table source.

The underlying SI builder reads only the frozen, integrity-checked manuscript
asset CSVs. This wrapper performs presentation-only normalization after table
generation: it repairs accidental control bytes, prevents section-only portrait
pages before landscape tables, expands raw machine identifiers into readable
labels, and shortens printed SHA-256 values while retaining the full hashes in
the machine-readable manifest. It performs no model fitting, result selection,
or statistical re-analysis.
"""

from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "paper2_admet_benchmark" / "scripts" / "35_build_supporting_information.py"
OUTPUT = ROOT / "paper2_latex" / "generated_supplementary_tables.tex"


DISPLAY_REPLACEMENTS = {
    r"density\_ratio\_weighted\_lac": "Density-ratio weighted LAC",
    r"marginal\_lac": "Marginal LAC",
    r"mondrian\_lac": "Mondrian LAC",
    r"mondrian\_minus\_marginal": "Mondrian - marginal",
    r"shift\_weighted\_minus\_marginal": "Shift-weighted - marginal",
    r"density\_ratio\_weighted\_absolute\_residual": "Density-ratio weighted residual",
    r"marginal\_absolute\_residual": "Marginal absolute residual",
    r"split\_calibration\_adaptive\_normalized": "Adaptive normalized",
    r"adaptive\_minus\_marginal": "Adaptive - marginal",
    r"one\_minus\_max\_probability": "1 - max predicted probability",
    r"one\_minus\_max\_tanimoto\_to\_train": "1 - max Tanimoto to training set",
}


def sanitize_control_bytes(data: bytes) -> tuple[bytes, dict[str, int]]:
    replacements = {
        bytes([0x07]) + b"lpha": b"\\alpha",
        bytes([0x0D]) + b"ho": b"\\rho",
    }
    counts: dict[str, int] = {}
    for bad, good in replacements.items():
        count = data.count(bad)
        if count:
            data = data.replace(bad, good)
        counts[bad.hex()] = count
    return data, counts


def polish_generated_tex(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Generated SI table file was not created: {path}")

    data, control_counts = sanitize_control_bytes(path.read_bytes())
    text = data.decode("utf-8")

    # The pdflscape environment begins with a page clear. A section command placed
    # immediately before it is therefore stranded on a portrait page by itself.
    # Move each such heading inside the landscape environment so the heading and
    # its first table share the same page.
    text, moved_sections = re.subn(
        r"(\\section\{[^{}]+\})\s*(\\begin\{landscape\})",
        lambda match: f"{match.group(2)}\n{match.group(1)}\n",
        text,
    )

    # Prevent the first portrait table section from being orphaned at the bottom
    # of the introductory page.
    first_section = r"\section{Predictive performance and calibration}"
    if first_section in text and r"\clearpage" + "\n" + first_section not in text:
        text = text.replace(first_section, r"\clearpage" + "\n" + first_section, 1)

    for raw_label, display_label in DISPLAY_REPLACEMENTS.items():
        text = text.replace(raw_label, display_label)

    # The two densest tables were previously set in \tiny, making them difficult
    # to inspect at normal zoom. The revised layouts support \scriptsize.
    text = text.replace(r"\tiny", r"\scriptsize")

    # Constrain the selective-prediction table to the landscape text width. The
    # previous natural-width numeric columns extended beyond the right margin and
    # clipped the class-balance-shift column in the rendered PDF.
    old_selective_layout = r"lllp{4.5cm}cccccc"
    new_selective_layout = (
        r"p{1.7cm}p{1.7cm}p{1.4cm}p{4.0cm}"
        r"*{6}{>{\centering\arraybackslash}p{1.55cm}}"
    )
    selective_layout_updates = text.count(old_selective_layout)
    text = text.replace(old_selective_layout, new_selective_layout)

    # Full 64-character hashes remain in final_results_integrity_manifest.csv.
    # The printed SI shows stable prefixes to avoid an unreadably compressed table.
    def shorten_hash(match: re.Match[str]) -> str:
        return r"\texttt{" + match.group(1)[:12] + r"\ldots}"

    text, shortened_hashes = re.subn(
        r"\\path\{([0-9a-f]{64})\}",
        shorten_hash,
        text,
    )
    text = text.replace(
        r"p{8.0cm}cccp{10.0cm}",
        r"p{11.0cm}cccp{3.2cm}",
    )
    text = text.replace(
        "Integrity manifest for the manuscript-ready frozen result tables.",
        "Integrity manifest for the manuscript-ready frozen result tables. "
        "SHA-256 prefixes are printed for readability; full hashes are retained "
        "in the machine-readable integrity manifest.",
    )

    encoded = text.encode("utf-8")
    disallowed = sorted(
        {
            byte
            for byte in encoded
            if byte < 32 and byte not in {9, 10, 13}
        }
    )
    if disallowed:
        codes = ", ".join(f"0x{byte:02X}" for byte in disallowed)
        raise RuntimeError(
            "Generated SI TeX still contains disallowed control bytes after "
            f"sanitization: {codes}"
        )

    stranded = re.findall(
        r"\\section\{[^{}]+\}\s*\\begin\{landscape\}",
        text,
    )
    if stranded:
        raise RuntimeError(
            "SI layout normalization left section headings outside landscape "
            f"tables: {stranded}"
        )
    if old_selective_layout in text or selective_layout_updates != 1:
        raise RuntimeError(
            "Selective-prediction table layout was not replaced exactly once; "
            f"replacement count={selective_layout_updates}"
        )

    path.write_text(text, encoding="utf-8", newline="\n")
    print("sanitized and polished", path)
    print("control-sequence replacements", control_counts)
    print("landscape section headings moved", moved_sections)
    print("selective table layouts constrained", selective_layout_updates)
    print("integrity hashes shortened for print", shortened_hashes)


def main() -> None:
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    polish_generated_tex(OUTPUT)
    print("Clean Supporting Information LaTeX generation complete.")


if __name__ == "__main__":
    main()
