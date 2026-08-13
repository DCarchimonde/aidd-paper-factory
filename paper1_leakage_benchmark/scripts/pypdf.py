from __future__ import annotations

"""Tiny local compatibility shim used only for page-count gates.

The submission pipeline does not need PDF editing. It only needs ``len(PdfReader(...).pages)``.
Prefer the sibling LaTeX log, then fall back to counting PDF page dictionaries.
"""

import re
from pathlib import Path


class PdfReader:
    def __init__(self, filename):
        path = Path(filename)
        log = path.with_suffix(".log")
        count = None
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="ignore")
            hits = re.findall(r"Output written on .*?\((\d+) pages?[,)]", text)
            if hits:
                count = int(hits[-1])
        if count is None:
            data = path.read_bytes()
            count = len(re.findall(rb"/Type\s*/Page(?!s)\b", data))
        if count <= 0:
            raise ValueError(f"Could not determine page count for {path}")
        self.pages = [None] * count
