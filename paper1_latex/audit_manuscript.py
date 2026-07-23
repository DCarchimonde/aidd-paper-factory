from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main.tex"
SUPPLEMENTARY = ROOT / "supplementary.tex"
BIB_FILES = [ROOT / "references.bib", ROOT / "references_extra.bib", ROOT / "references_recent.bib"]
RECENT_START = 2021
RECENT_END = 2026
MIN_RECENT_CITED = 30


class AuditFailure(RuntimeError):
    pass


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def resolve_input_target(raw_target: str, including_file: Path) -> Path:
    """Resolve a LaTeX input using project-root semantics with a local fallback.

    LaTeX in this project is compiled from ``paper1_latex``. Therefore an input such
    as ``sections/results_chemometrics_target`` remains root-relative even when it
    appears inside another file in ``sections``. A current-file-relative fallback is
    retained for genuinely local input paths.
    """
    target = Path(raw_target.strip())
    if target.suffix.lower() != ".tex":
        target = target.with_suffix(".tex")

    if target.is_absolute():
        candidates = [target]
    else:
        candidates = [ROOT / target, including_file.parent / target]

    attempted: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in attempted:
            continue
        attempted.append(resolved)
        if resolved.exists():
            return resolved

    attempted_text = ", ".join(str(path) for path in attempted)
    raise AuditFailure(
        f"Missing LaTeX input '{raw_target}' referenced by "
        f"{including_file.relative_to(ROOT)}; attempted: {attempted_text}"
    )


def resolve_inputs(path: Path, visited: set[Path] | None = None) -> list[Path]:
    if visited is None:
        visited = set()
    path = path.resolve()
    if path in visited:
        return []
    if not path.exists():
        raise AuditFailure(f"Missing LaTeX input: {path}")
    visited.add(path)
    files = [path]
    text = read_text(path)
    for match in re.finditer(r"\\input\{([^}]+)\}", text):
        child = resolve_input_target(match.group(1), path)
        files.extend(resolve_inputs(child, visited))
    return files


def parse_bib_entries() -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    entry_pattern = re.compile(r"@(\w+)\s*\{\s*([^,]+),(.+?)(?=\n@|\Z)", re.S)
    field_pattern = re.compile(r"(\w+)\s*=\s*[\{\"](.+?)[\}\"]\s*,?\s*(?=\n|\Z)", re.S)
    for bib_path in BIB_FILES:
        if not bib_path.exists():
            raise AuditFailure(f"Missing bibliography file: {bib_path}")
        text = read_text(bib_path)
        for entry_match in entry_pattern.finditer(text):
            entry_type, key, body = entry_match.groups()
            key = key.strip()
            if key in entries:
                raise AuditFailure(f"Duplicate BibTeX key: {key}")
            fields = {name.lower(): value.strip() for name, value in field_pattern.findall(body)}
            fields["entry_type"] = entry_type.lower()
            fields["source_file"] = bib_path.name
            entries[key] = fields
    return entries


def collect_citations(tex_files: list[Path]) -> set[str]:
    cited: set[str] = set()
    citation_pattern = re.compile(r"\\cite\w*\{([^}]+)\}")
    for path in tex_files:
        text = read_text(path)
        for group in citation_pattern.findall(text):
            cited.update(key.strip() for key in group.split(",") if key.strip())
    return cited


def strip_label(equation_body: str) -> str:
    return re.sub(r"\\label\{[^}]+\}", "", equation_body).strip()


def check_equations(path: Path, errors: list[str]) -> None:
    lines = read_text(path).splitlines()
    i = 0
    while i < len(lines):
        if "\\begin{equation}" not in lines[i]:
            i += 1
            continue
        begin = i
        if begin == 0 or not lines[begin - 1].strip():
            errors.append(f"{path.relative_to(ROOT)}:{begin + 1}: blank line immediately before equation")
        j = begin + 1
        body_lines: list[str] = []
        while j < len(lines) and "\\end{equation}" not in lines[j]:
            body_lines.append(lines[j])
            j += 1
        if j >= len(lines):
            errors.append(f"{path.relative_to(ROOT)}:{begin + 1}: unclosed equation environment")
            return
        if j + 1 >= len(lines) or not lines[j + 1].strip():
            errors.append(f"{path.relative_to(ROOT)}:{j + 1}: blank line immediately after equation")
        equation = strip_label(" ".join(body_lines))
        if equation and equation[-1] not in ".,;:":
            errors.append(
                f"{path.relative_to(ROOT)}:{j + 1}: displayed equation lacks sentence punctuation"
            )
        i = j + 1


def check_captions(path: Path, errors: list[str]) -> None:
    text = read_text(path)
    for match in re.finditer(r"\\caption\{(.+?)\}", text, flags=re.S):
        caption = re.sub(r"\s+", " ", match.group(1)).strip()
        if caption and caption[-1] not in ".!?":
            line = text[: match.start()].count("\n") + 1
            errors.append(f"{path.relative_to(ROOT)}:{line}: caption lacks final punctuation")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    main_files = resolve_inputs(MAIN)
    supplementary_files = resolve_inputs(SUPPLEMENTARY)
    all_tex_files = list(dict.fromkeys(main_files + supplementary_files))

    entries = parse_bib_entries()
    cited = collect_citations(main_files)

    missing = sorted(cited - entries.keys())
    if missing:
        errors.append("Missing BibTeX entries for cited keys: " + ", ".join(missing))

    recent_cited: list[str] = []
    for key in sorted(cited & entries.keys()):
        raw_year = entries[key].get("year", "")
        match = re.search(r"\d{4}", raw_year)
        if match and RECENT_START <= int(match.group()) <= RECENT_END:
            recent_cited.append(key)
    if len(recent_cited) < MIN_RECENT_CITED:
        errors.append(
            f"Only {len(recent_cited)} cited references are from {RECENT_START}–{RECENT_END}; "
            f"minimum is {MIN_RECENT_CITED}."
        )

    doi_to_keys: dict[str, list[str]] = {}
    for key, fields in entries.items():
        doi = fields.get("doi", "").lower().strip()
        if doi:
            doi_to_keys.setdefault(doi, []).append(key)
        author = fields.get("author", "")
        if "and others" in author.lower():
            warnings.append(f"Abbreviated author list remains in BibTeX entry: {key}")
        year_match = re.search(r"\d{4}", fields.get("year", ""))
        if year_match and RECENT_START <= int(year_match.group()) <= RECENT_END:
            if not any(fields.get(name) for name in ("doi", "url", "eprint", "note")):
                errors.append(f"Recent entry lacks DOI or official identifier: {key}")
    duplicate_dois = {doi: keys for doi, keys in doi_to_keys.items() if len(keys) > 1}
    if duplicate_dois:
        for doi, keys in sorted(duplicate_dois.items()):
            errors.append(f"Duplicate DOI {doi}: {', '.join(keys)}")

    for path in all_tex_files:
        text = read_text(path)
        if "??" in text:
            errors.append(f"{path.relative_to(ROOT)}: source contains literal '??'")
        check_equations(path, errors)
        check_captions(path, errors)

    main_combined = "\n".join(read_text(path) for path in main_files)
    if re.search(r"\bAppendix\b", main_combined):
        errors.append("Main manuscript still contains an 'Appendix' reference after SI separation.")

    cited_counter = Counter(cited)
    if any(count > 1 for count in cited_counter.values()):
        warnings.append("Citation-key counter contained duplicates unexpectedly.")

    unused = sorted(entries.keys() - cited)
    if unused:
        warnings.append(f"Unused BibTeX entries: {len(unused)} (not printed by BibTeX).")

    print(f"Main LaTeX files audited: {len(main_files)}")
    print(f"Supporting Information files audited: {len(supplementary_files)}")
    print(f"Unique cited references: {len(cited)}")
    print(f"Cited references from {RECENT_START}–{RECENT_END}: {len(recent_cited)}")
    print("Recent cited keys: " + ", ".join(recent_cited))

    if warnings:
        print("\nWARNINGS")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nAUDIT FAILURES")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nSOURCE AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
