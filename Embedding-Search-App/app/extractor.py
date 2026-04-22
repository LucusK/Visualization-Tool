import json
import csv
import io
import urllib.parse
from pathlib import Path

import requests
import pdfplumber

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

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


# ── URL extraction ────────────────────────────────────────────────────────────

def search_wikipedia(term: str, limit: int = 3) -> list[tuple[str, str]]:
    """
    Search Wikipedia for `term` and return the top `limit` articles as
    (title, text) tuples. Uses OpenSearch for discovery, then fetches
    intro-section plain text for all matches in one batch request.
    """
    # Step 1: discover matching page titles via OpenSearch
    search_resp = requests.get(WIKIPEDIA_API, params={
        "action":    "opensearch",
        "search":    term,
        "limit":     limit,
        "namespace": "0",        # main articles only
        "format":    "json",
    }, timeout=10)
    search_resp.raise_for_status()

    titles = search_resp.json()[1]   # [query, [titles], [descs], [urls]]
    if not titles:
        raise ValueError(f"No Wikipedia results found for: {term!r}")

    # Step 2: fetch intro plain text for all titles in one batch request
    extract_resp = requests.get(WIKIPEDIA_API, params={
        "action":      "query",
        "prop":        "extracts",
        "titles":      "|".join(titles),
        "format":      "json",
        "explaintext": "1",
        "exintro":     "1",
    }, timeout=10)
    extract_resp.raise_for_status()

    pages = extract_resp.json()["query"]["pages"]
    results = []
    for page in pages.values():
        text = page.get("extract", "").strip()
        if text and "missing" not in page:
            results.append((page.get("title", ""), text))

    if not results:
        raise ValueError(f"Wikipedia returned no extractable text for: {term!r}")

    return results


def extract_url(url: str) -> tuple[str, str]:
    """
    Fetch plain text from a URL. Returns (text, title).
    Wikipedia URLs use the MediaWiki API for clean, citation-free text.
    Other URLs fall back to stripping HTML tags with stdlib.
    """
    parsed = urllib.parse.urlparse(url)
    if "wikipedia.org" in parsed.netloc:
        return _extract_wikipedia(url, parsed)
    return _extract_generic(url)


def _extract_wikipedia(url: str, parsed: urllib.parse.ParseResult) -> tuple[str, str]:
    # /wiki/Linear_regression  →  "Linear_regression"
    path_parts = parsed.path.split("/wiki/")
    if len(path_parts) < 2 or not path_parts[1]:
        raise ValueError(f"Could not parse Wikipedia page title from URL: {url}")
    title = urllib.parse.unquote(path_parts[1])

    resp = requests.get(WIKIPEDIA_API, params={
        "action":  "query",
        "prop":    "extracts",
        "titles":  title,
        "format":  "json",
        "explaintext": "1",   # plain text, no HTML or wiki markup
        "exintro": "1",       # intro section only — focused and readable
    }, timeout=10)
    resp.raise_for_status()

    pages = resp.json()["query"]["pages"]
    page  = next(iter(pages.values()))

    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {title!r}")

    text = page.get("extract", "").strip()
    if not text:
        raise ValueError(f"No extractable text for Wikipedia page: {title!r}")

    return text, page.get("title", title.replace("_", " "))


def _extract_generic(url: str) -> tuple[str, str]:
    """Strip HTML tags from any webpage using stdlib html.parser."""
    from html.parser import HTMLParser

    resp = requests.get(url, timeout=10, headers={"User-Agent": "EmbeddingVisualizationTool/1.0"})
    resp.raise_for_status()

    class _StripHTML(HTMLParser):
        def __init__(self):
            super().__init__()
            self._skip = False
            self.parts = []

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "nav", "header", "footer"):
                self._skip = True

        def handle_endtag(self, tag):
            if tag in ("script", "style", "nav", "header", "footer"):
                self._skip = False

        def handle_data(self, data):
            if not self._skip and data.strip():
                self.parts.append(data.strip())

    parser = _StripHTML()
    parser.feed(resp.text)
    text  = " ".join(parser.parts)
    title = urllib.parse.urlparse(url).netloc + urllib.parse.urlparse(url).path
    return text, title
