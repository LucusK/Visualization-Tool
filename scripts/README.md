# Scripts — Older Pipeline (ColBERT → MUVERA → FAISS)

> **Note:** This is the older research pipeline. The active pipeline is in [`/weaviate`](../weaviate/).

These scripts implement an end-to-end pipeline for encoding documents with **ColBERT** and compressing them into **FDE (Fixed-Dimensional Encoding)** vectors via the MUVERA algorithm, then indexing them with FAISS for fast retrieval.

## Pipeline

```
PDF / Text
    │
    ▼ 00_1_pdf_to_text.py
Raw text
    │
    ▼ 01_chunk_txt.py
Passages (~800 chars each)
    │
    ▼ 02_colbert_encode.py
ColBERT token embeddings (768-dim, per token)
    │
    ▼ 03_run_muvera.py
FDE vectors (single fixed-size vector per document)
    │
    ▼ 04_weaviate.py  OR  05_build_msmarco_faiss.py
Weaviate / FAISS index
```

## Scripts

### `00_1_pdf_to_text.py`
Converts PDF files to plain text using PyMuPDF (`fitz`). Cleans up hyphenation artifacts and normalizes whitespace.
- **Input:** PDF files
- **Output:** `data/raw/*.txt`

### `01_chunk_txt.py`
Splits raw text into passages of ~800 characters (~150–200 tokens) by grouping sentences.
- **Input:** `data/raw/*.txt`
- **Output:** `data/passages/*.passages.txt` (one passage per line)

### `02_colbert_encode.py`
Encodes passages using the pre-trained `colbert-ir/colbertv2.0` model. Generates per-token embeddings (128-dim), filtering out `[CLS]`/`[SEP]` tokens.
- **Input:** `data/passages/*.passages.txt`
- **Output:**
  - `colbert_out/bow/doc_embeddings.npy` — concatenated token embeddings
  - `colbert_out/bow/token_counts.json` — token count per document

### `03_run_muvera.py`
Runs the MUVERA FDE algorithm over ColBERT embeddings to produce a single fixed-size vector per document.
- **Config:** 5 repetitions, 7 SimHash projections (128 partitions), identity projection
- **Input:** `colbert_out/bow/`
- **Output:** `muvera_out/fde/sample_FDE.npy`

### `04_weaviate.py`
Loads FDE vectors into a Weaviate collection (`MuveraDoc`) using bring-your-own-vectors mode.
- **Input:** `muvera_out/fde/sample_FDE.npy`
- **Output:** Weaviate collection

### `05_build_msmarco_faiss.py`
Large-scale pipeline for the full MS MARCO dataset (~8.8M passages). Encodes with ColBERT, projects down to 8D, generates FDE vectors, and builds a FAISS HNSW index.
- **Input:** `data/MSMARCO/collection.tsv`
- **Output:** `ann_out/msmarco/` (FAISS index + passage ID mapping + metadata)

### `docker-compose.yml`
Docker Compose file for running Weaviate (used by `04_weaviate.py`).

## Why MUVERA/FDE?

ColBERT stores 100s of token vectors per document. Searching over millions of documents requires comparing many vectors — too slow at scale. FDE compresses all token vectors into one fixed-size vector while approximating the Chamfer similarity, enabling fast MIPS (Maximum Inner Product Search) with standard ANN libraries.

See [`/muvera-py`](../muvera-py/) for the algorithm implementation and benchmarks.
