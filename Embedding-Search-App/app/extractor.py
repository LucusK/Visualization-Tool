import json
import csv
import io
from pathlib import Path

import pdfplumber

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md", ".rst", ".csv", ".json", ".xml", ".html"}


def extract_text(filepath: str | Path) -> str:
    """
    Extract plain text from a file, dispatching by extension.
    Raises ValueError for unsupported types, IOError if the file can't be read.
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext!r}")

    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".csv":
        return _extract_csv(path)
    if ext == ".json":
        return _extract_json(path)
    # .txt, .md, .rst, .xml, .html — read as plain text
    return path.read_text(encoding="utf-8", errors="replace")


# ── Per-format helpers ────────────────────────────────────────────────────────

def _extract_pdf(path: Path) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
    if not pages:
        raise IOError(f"No extractable text found in {path.name}")
    return "\n\n".join(pages)


def _extract_csv(path: Path) -> str:
    """Convert CSV rows to readable lines: 'col1: val1 | col2: val2 ...'"""
    raw = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    lines = []
    for row in reader:
        lines.append(" | ".join(f"{k}: {v}" for k, v in row.items() if v.strip()))
    return "\n".join(lines)


def _extract_json(path: Path) -> str:
    """Flatten JSON to a readable string via json.dumps with indentation."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        # malformed JSON — return raw text so it still gets indexed
        return raw
