from __future__ import annotations

"""Audit Paper 2 LaTeX citations and bibliography metadata.

This script does not claim to replace publisher-level DOI verification.  It checks
that the locally verified bibliography remains internally consistent after edits:

- every citation key in the manuscript exists in references.bib;
- bibliography keys are unique;
- cited recent references (2021--2026) meet the manuscript minimum;
- cited journal articles contain a DOI or a complete verified journal record;
- cited conference entries contain either a DOI, URL, or official venue record;
- unused entries are listed for manual review.
"""

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIB_PATH = ROOT / "references.bib"
TEX_PATHS = [ROOT / "main.tex", *sorted((ROOT / "sections").glob("*.tex"))]
RECENT_START = 2021
RECENT_END = 2026
MIN_RECENT_CITED = 30


def parse_entries(text: str) -> dict[str, dict[str, str]]:
    starts = list(re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text))
    entries: dict[str, dict[str, str]] = {}
    for idx, match in enumerate(starts):
        entry_type, key = match.group(1).lower(), match.group(2)
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
        block = text[match.start():end]
        fields = {
            name.lower(): value.strip()
            for name, value in re.findall(
                r"(?ms)^\s*(\w+)\s*=\s*\{(.*?)\}\s*,?\s*$", block
            )
        }
        entries[key] = {"entry_type": entry_type, **fields}
    return entries


def citation_keys() -> list[str]:
    keys: list[str] = []
    pattern = re.compile(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)?\{([^}]+)\}")
    for path in TEX_PATHS:
        text = path.read_text(encoding="utf-8")
        for group in pattern.findall(text):
            keys.extend(key.strip() for key in group.split(",") if key.strip())
    return keys


def complete_doi_less_article(entry: dict[str, str]) -> bool:
    """Allow journals that legitimately publish without Crossref DOI metadata.

    A DOI-less entry is accepted only when journal, volume, and pages are all
    present.  Such entries remain listed in literature_audit.md with their
    official publisher record (for example, JMLR).
    """

    return all(entry.get(field) for field in ("journal", "volume", "pages"))


def main() -> None:
    bib_text = BIB_PATH.read_text(encoding="utf-8")
    raw_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text)
    duplicates = sorted(key for key, count in Counter(raw_keys).items() if count > 1)
    entries = parse_entries(bib_text)
    cited = citation_keys()
    cited_unique = sorted(set(cited))
    missing = sorted(set(cited_unique) - set(entries))
    unused = sorted(set(entries) - set(cited_unique))

    recent_cited: list[str] = []
    metadata_issues: list[str] = []
    for key in cited_unique:
        entry = entries.get(key)
        if not entry:
            continue
        year_text = re.sub(r"[^0-9]", "", entry.get("year", ""))
        year = int(year_text[:4]) if len(year_text) >= 4 else None
        if year is not None and RECENT_START <= year <= RECENT_END:
            recent_cited.append(key)
        entry_type = entry.get("entry_type")
        if (
            entry_type == "article"
            and not entry.get("doi")
            and not complete_doi_less_article(entry)
        ):
            metadata_issues.append(
                f"{key}: cited article has neither DOI nor complete journal/volume/pages"
            )
        if entry_type == "inproceedings" and not (
            entry.get("doi") or entry.get("url") or entry.get("booktitle")
        ):
            metadata_issues.append(
                f"{key}: cited proceedings entry lacks DOI/URL/booktitle"
            )
        for required in ("author", "title", "year"):
            if not entry.get(required):
                metadata_issues.append(f"{key}: missing {required}")

    print(f"bibliography_entries={len(entries)}")
    print(f"citations_total={len(cited)}")
    print(f"citations_unique={len(cited_unique)}")
    print(f"recent_cited_{RECENT_START}_{RECENT_END}={len(recent_cited)}")
    print(f"minimum_recent_required={MIN_RECENT_CITED}")
    print(f"duplicate_keys={len(duplicates)}")
    print(f"missing_citation_keys={len(missing)}")
    print(f"metadata_issues={len(metadata_issues)}")
    print(f"unused_entries={len(unused)}")

    if duplicates:
        print("\nDUPLICATE KEYS")
        print("\n".join(duplicates))
    if missing:
        print("\nMISSING CITATION KEYS")
        print("\n".join(missing))
    if metadata_issues:
        print("\nMETADATA ISSUES")
        print("\n".join(metadata_issues))
    if unused:
        print("\nUNUSED ENTRIES (manual review, not an automatic failure)")
        print("\n".join(unused))

    failures = []
    if duplicates:
        failures.append("duplicate bibliography keys")
    if missing:
        failures.append("missing citation keys")
    if metadata_issues:
        failures.append("cited-entry metadata issues")
    if len(recent_cited) < MIN_RECENT_CITED:
        failures.append(
            f"only {len(recent_cited)} cited references are from "
            f"{RECENT_START}--{RECENT_END}"
        )
    if failures:
        raise SystemExit("Bibliography audit failed: " + "; ".join(failures))

    print("\nBIBLIOGRAPHY AUDIT: PASS")


if __name__ == "__main__":
    main()
