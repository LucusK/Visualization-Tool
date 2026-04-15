# Embedding Visualization Tool — Plan

## Goal
Given a query and a retrieved document, render a **token-level similarity heatmap**
(query tokens × document tokens) so it's visually clear which query tokens drive retrieval.

---

## What we already have
- `scripts/02_colbert_encode.py` — loads `colbert-ir/colbertv2.0` via HuggingFace
  `transformers`, tokenizes passages, runs the model, strips `[CLS]`/`[SEP]` tokens,
  and produces per-token embeddings (`shape: [num_tokens, 768]`).
- `scripts/03_run_muvera.py` — L2-normalizes the ColBERT embeddings before computing
  FDE vectors. The same normalization applies for cosine similarity in the heatmap.
- `muvera-py/fde_generator.py` — compresses multi-vectors into a single FDE for ANN
  search (not needed for the heatmap itself).

---

## Pipeline for the visualization

```
query string  ──►  ColBERT tokenize + encode  ──►  Q × 128  ─┐
                                                               ├──►  Q × D similarity matrix  ──►  heatmap
doc string    ──►  ColBERT tokenize + encode  ──►  D × 128  ─┘
```

1. **Encode** — reuse the encoding approach from `scripts/02_colbert_encode.py`:
   load `colbert-ir/colbertv2.0` via `AutoTokenizer` / `AutoModel`, tokenize the
   query and document, run the model, and strip `[CLS]`/`[SEP]` tokens.
   Embeddings are `768`-dimensional (not 128 as in neural-cherche).
2. **Tokenize labels** — decode the kept token IDs back to strings for axis labels
   (e.g. "what", "##ever"). `[CLS]`/`[SEP]` are stripped, matching the pipeline.
3. **L2-normalize** — normalize per-token vectors as in `scripts/03_run_muvera.py`
   so dot products give cosine similarity scores in `[-1, 1]`.
4. **Similarity matrix** — `sim[i, j] = dot(Q[i], D[j])` → shape `(Q, D)`.
5. **Heatmap** — `seaborn.heatmap` with query tokens on Y-axis, doc tokens on X-axis.
   Highlight the max-sim doc token per query token (the ColBERT "winner") with a marker.

---

## Files to create

| File | Purpose |
|------|---------|
| `Embedding-Visualization-Tool/visualize.py` | Core script: encode → similarity matrix → heatmap |
| `Embedding-Visualization-Tool/requirements.txt` | `transformers`, `torch`, `seaborn`, `matplotlib`, `numpy` |

---

## Implementation steps

1. **`visualize.py` — encode function**
   - Load `colbert-ir/colbertv2.0` with `AutoTokenizer` and `AutoModel` (same as
     `scripts/02_colbert_encode.py`).
   - Tokenize query and document separately; strip `[CLS]`/`[SEP]` token IDs (101, 102)
     and decode remaining IDs for axis labels.
   - Run model, extract `last_hidden_state`, keep only non-special token rows.

2. **Similarity matrix**
   - `sim = Q_emb @ D_emb.T` (both already L2-normalised by neural-cherche).

3. **Heatmap render**
   - `seaborn.heatmap(sim, xticklabels=doc_tokens, yticklabels=query_tokens, cmap="RdYlGn")`.
   - Mark `argmax` per row with `*` to show the MaxSim winner.
   - Save to `output/heatmap_<timestamp>.png`.

4. **CLI entry point**
   - `py -3.11 visualize.py "your query" "document text here"`.

---

## Future extension (app)
- Web UI (Flask or FastAPI) accepting file uploads.
- Ingest files → produce embeddings → BM25 or cosine top-10 search.
- For each result, generate and display the heatmap inline.
- Stack: Python backend + simple HTML/JS frontend (no framework needed initially).
