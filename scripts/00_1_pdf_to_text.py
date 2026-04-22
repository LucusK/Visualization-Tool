from pathlib import Path
import re
import sys

def clean_text(s: str) -> str:
    # de-hyphenate across line breaks: "exam-\nple" -> "example"
    s = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', s)
    # normalize newlines: multiple blank lines -> single blank line
    s = re.sub(r'\n{3,}', '\n\n', s)
    # collapse weird spacing
    s = re.sub(r'[ \t]+', ' ', s)
    # strip trailing spaces on lines
    s = re.sub(r'[ \t]+\n', '\n', s)
    return s.strip()

def pdf_to_txt(pdf_path: Path, out_path: Path):
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not installed")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append((text or "").strip() + "\n\n")
    raw = "".join(pages)
    cleaned = clean_text(raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cleaned, encoding="utf-8")
    print(f"Wrote text to: {out_path} ({len(cleaned):,} chars from {len(doc)} pages)")

def main():
    repo = Path(__file__).resolve().parents[1]
    raw_dir = repo / "data" / "raw"

    if len(sys.argv) == 1:
        pdf_path = raw_dir / "samplepdf.pdf"
        out_path = raw_dir / "samplepdf.txt"
    elif len(sys.argv) == 2:
        pdf_path = Path(sys.argv[1]).resolve()
        out_path = pdf_path.with_suffix(".txt")
    elif len(sys.argv) == 3:
        pdf_path = Path(sys.argv[1]).resolve()
        out_path = Path(sys.argv[2]).resolve()
    else:
        print("Usage:")
        print("  python scripts/00_pdf_to_text.py")
        print("  python scripts/00_pdf_to_text.py input.pdf")
        print("  python scripts/00_pdf_to_text.py input.pdf output.txt")
        sys.exit(2)

    if not pdf_path.exists():
        print(f"Missing input PDF: {pdf_path}")
        sys.exit(1)

    pdf_to_txt(pdf_path, out_path)

if __name__ == "__main__":
    main()
