# Commit History

## Pending
<!-- Changes staged here until ready to commit -->

---

## Log
<!-- Past commits recorded here after push -->

## [2026-04-14] — Add Wikipedia text examples, Dockerfile, chunker, encoder, search, API, and visualization pipeline

**Branch:** `main`

### Commits (newest first)

- **`1acc917`** — `added text docs examples from wikipedia` — Added sample Wikipedia text documents for use as search corpus examples.
- **`40beb16`** — `added dockerfile, requirements, and a db.py that creates the database to be searching from. Heatmap is the same as visualization` — Added Dockerfile and requirements for containerized setup; `db.py` initializes the searchable database; heatmap visualization retained from prior work.
- **`26ef6cf`** — `Added chunker to turn text into passages, encoder to turn those passages into embeddings, and search` — Text chunking pipeline splits raw docs into passages; encoder converts passages to embeddings; search module queries the embedded corpus.
- **`4ed1d46`** — `Added api.py which ingests and searches, extractor takes text out depending on extension` — `api.py` provides ingest and search endpoints; extractor handles multiple file extensions (.pdf, .txt, etc.).
- *(earlier)* — `Visualization simulator for in line command query + doc` — CLI-based visualizer for querying a document inline and displaying similarity output.
- *(earlier)* — `initial commit, added planning stages` — Project scaffolding and initial planning documents.

---

## [2026-04-14 20:07] — Add ColBERT heatmap visualizer and project scaffolding

**Commit:** `2b709f1c6fc542cc15cd680a695cd1676a048245`
**Branch:** `main`

### Changes
- `visualize.py` — New core script: loads ColBERT (`colbert-ir/colbertv2.0`), encodes a query and document into per-token embeddings, L2-normalizes them, computes a cosine similarity matrix, and renders a seaborn heatmap with MaxSim winners marked. Supports single-shot CLI and interactive mode (model loads once).
- `requirements.txt` — New file pinning dependencies (`transformers`, `torch`, `seaborn`, `matplotlib>=3.9`, `numpy<2`) for reproducible installs.
- `PLANNING.md` — Updated with full implementation plan: pipeline diagram, step-by-step encoding/normalization/heatmap logic, and future app extension notes.
- `scripts/` — Added existing ColBERT pipeline scripts (PDF→text, chunking, encoding, MUVERA FDE generation, Weaviate ingest, FAISS index build) for reference.
- `.claude/` — Added Claude Code skill configuration (`my-command` commit/push skill and local settings).
- `muvera-py` — Registered as embedded git reference (files tracked in its own repo; added here as a pointer).

---

