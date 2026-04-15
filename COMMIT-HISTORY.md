# Commit History

## Pending
<!-- Changes staged here until ready to commit -->

---

## Log
<!-- Past commits recorded here after push -->

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

