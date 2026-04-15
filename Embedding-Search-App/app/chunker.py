import re


def chunk_text(text: str, size: int = 200, overlap: int = 20) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text:    Raw extracted text from a document.
        size:    Target chunk size in words.
        overlap: Number of words carried over from the previous chunk.

    Returns:
        List of non-empty chunk strings.
    """
    words = _tokenize_words(text)
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap   # step back by overlap so context carries over

    return chunks


# ── Helper ────────────────────────────────────────────────────────────────────

def _tokenize_words(text: str) -> list[str]:
    """
    Normalize whitespace and split into words.
    Collapses newlines/tabs into single spaces so chunk boundaries
    don't land inside a paragraph break.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized.split(" ") if normalized else []
