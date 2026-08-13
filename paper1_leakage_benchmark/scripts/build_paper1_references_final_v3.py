from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "paper1_latex"
SOURCE = LATEX / "references_joc.tex"
OUTPUT = LATEX / "references_joc_submission.tex"

ORDER = [
    "esbensen2010proper", "tropsha2003earnest", "gramatica2007principles",
    "lasfar2024robustness", "kiraly2025leakage", "camacho2026validation",
    "wu2018moleculenet", "bemis1996properties", "yang2019learned",
    "deng2023systematic", "li2026partition", "joeres2025datasail",
    "landrum2023simpd", "nael2026dataset", "rdkit", "rogers2010extended",
    "pedregosa2011scikit", "breiman2001random", "chen2016xgboost",
]


def blocks(text: str) -> dict[str, str]:
    body = text.split("\\begin{thebibliography}{99}", 1)[1].rsplit("\\end{thebibliography}", 1)[0]
    hits = list(re.finditer(r"(?m)^\\bibitem\{([^}]+)\}\s*$", body))
    out = {}
    for i, hit in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(body)
        out[hit.group(1)] = body[hit.start():end].strip()
    return out


def main() -> None:
    found = blocks(SOURCE.read_text(encoding="utf-8"))
    missing = [key for key in ORDER if key not in found]
    if missing:
        raise AssertionError("Missing bibliography keys: " + ", ".join(missing))
    output = "\\begin{thebibliography}{99}\n\n"
    output += "\n\n".join(found[key] for key in ORDER)
    output += "\n\n\\end{thebibliography}\n"
    OUTPUT.write_text(output, encoding="utf-8")
    print(f"FINAL REFERENCES: PASS ({len(ORDER)})")


if __name__ == "__main__":
    main()
