# Embedding Search App — Plan

## Goal
A web app where a user uploads text-based files, searches over them semantically,
gets the top-10 matching passages, and sees a ColBERT token-level heatmap for each
result explaining why it was retrieved.

---

## Tech Stack & Roles

| Technology | Role |
|---|---|
| **Python** | Text extraction, ColBERT encoding, MaxSim search, heatmap generation |
| **Flask** | Serves the HTML/JS frontend and the `/ingest`, `/search`, `/heatmaps` API endpoints |
| **HTML/JS** | Single-page UI — sample query dropdown, free-text input, inline heatmap results |
| **SQL (SQLite)** | Stores document metadata and passage chunks |
| **Agile** | Development methodology — work in short iterations, one feature at a time |

> **Note on PHP:** PHP was originally planned for the frontend but was dropped — Flask can serve the HTML page directly, keeping the stack to a single process with no extra server to manage.

---

## Architecture

```
Browser
  │
  ▼
Flask (api.py)  ─── GET /             → serves web/index.html
  │              ─── GET /heatmaps/*  → serves generated PNGs
  │              ─── GET /search?q=   → returns top-10 + heatmap URLs
  │              ─── POST /ingest     → file upload (optional, for future use)
  │
  ├── Auto-ingest on startup: txt-docs/*.txt → "Docs" collection (if DB empty)
  │
  ├── SQLite DB  (db/app.db)
  │     ├── documents  (id, filename, upload_time)
  │     └── passages   (id, doc_id, chunk_index, chunk_text, emb_path)
  │
  └── Disk  (embeddings/<doc_id>/<chunk_index>.npy)
            (output/heatmaps/<query_hash>_<passage_id>.png)
```

---

## Accepted File Types

| Extension | Extraction method |
|---|---|
| `.txt` | Read directly |
| `.pdf` | `pdfplumber` (preserves layout better than PyPDF2) |
| `.md`, `.rst`, `.csv`, `.json`, `.xml`, `.html` | Read as plain text |

Files are rejected at upload if their extension is not in the allowed list.

---

## Database Schema (SQLite)

```sql
CREATE TABLE documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    upload_time TEXT NOT NULL
);

CREATE TABLE passages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    emb_path    TEXT NOT NULL   -- relative path to .npy file on disk
);
```

Embeddings are stored as `.npy` files (one per passage) rather than BLOBs — SQL stays
lightweight and numpy files load fast with `np.load()`.

---

## Pipeline

### Ingest (on file upload)
```
uploaded file
  │
  ├── extract_text()     → raw text string
  ├── chunk_text()       → list of ~200-word passages (with 20-word overlap)
  ├── encode_passage()   → ColBERT per-token embeddings, L2-normalised
  ├── np.save()          → embeddings/<doc_id>/<chunk_index>.npy
  └── INSERT INTO passages
```

### Search (on query submit)
```
query string
  │
  ├── encode_query()     → Q × 768 embeddings, L2-normalised
  ├── load all passage embeddings from disk
  ├── maxsim_score()     → for each passage: sum of max cosine sim per query token
  ├── argsort descending → top-10 passage ids
  ├── render_heatmap()   → one heatmap PNG per result (reuse visualize.py logic)
  └── return JSON:  [{passage_text, doc_name, score, heatmap_url}, ...]
```

MaxSim score (the actual ColBERT scoring function):
```
score(Q, D) = Σ_i  max_j  cos_sim(Q[i], D[j])
```

---

## Files to Create / Modify

### Python API (already built — Iterations 1–3 complete)
| File | Status | Purpose |
|---|---|---|
| `app/api.py` | ✅ done | Flask app — `/ingest` and `/search` endpoints |
| `app/extractor.py` | ✅ done | `extract_text(filepath)` — dispatches by extension |
| `app/chunker.py` | ✅ done | `chunk_text(text, size=200, overlap=20)` → list of strings |
| `app/encoder.py` | ✅ done | Loads ColBERT once; `encode(text)` → (emb, tokens) |
| `app/search.py` | ✅ done | `maxsim_score()`, `top_k()` |
| `app/db.py` | ✅ done | SQLite helpers — init schema, insert/query documents & passages |
| `app/heatmap.py` | ✅ done | `render_heatmap()` — saves PNG to caller-specified path |

### Frontend & wiring (Iteration 4)
| File | Status | Purpose |
|---|---|---|
| `web/index.html` | 🔲 todo | Single-page UI — sample query dropdown, free-text input, inline results |
| `app/api.py` | 🔲 modify | Add `GET /` (serve index.html), `GET /heatmaps/<file>` (serve PNGs), auto-ingest on startup |

### Shared
| File | Purpose |
|---|---|
| `db/app.db` | SQLite database (auto-created on first run) |
| `embeddings/` | Per-passage `.npy` embedding files |
| `output/heatmaps/` | Generated heatmap PNGs served to browser |
| `txt-docs/` | Sample documents pre-loaded on startup (apple, banana, mango, pineapple, watermelon) |

---

## Agile Iterations

### Iteration 1 — Ingest pipeline (Python only)
- `extractor.py`, `chunker.py`, `encoder.py`, `db.py`
- Test: upload a `.txt` and a `.pdf`, verify passages stored in DB with embeddings on disk

### Iteration 2 — Search (Python only)
- `search.py`, `/search` endpoint in `api.py`
- Test: query returns correct top-10 passages with scores

### Iteration 3 — Heatmaps
- Wire `render_heatmap()` from `visualize.py` into the search results
- Test: heatmap PNG generated for each of the 10 results

### Iteration 4 — Flask-served HTML/JS frontend *(next)*
- Modify `api.py`: auto-ingest `txt-docs/*.txt` on startup if DB is empty; add `GET /` to serve `web/index.html`; add `GET /heatmaps/<filename>` to serve generated PNGs
- Create `web/index.html`: sample query dropdown (pre-defined queries about the fruit docs), free-text input, JS `fetch` to `/search`, render top results + inline heatmap `<img>` tags
- Sample queries to include: "what fruit grows in tropical climates?", "which fruits are native to Asia?", "what are the health benefits of fruit?", "how are fruits grown commercially?"
- Test: open browser, pick a sample query, see top passages + heatmaps; type a custom query, same result

### Iteration 5 — Polish
- Loading spinner while search runs (JS fetch is async)
- Error messages for empty queries or no documents ingested
- Score displayed next to each result

---

## Key Decisions & Rationale

- **SQLite over PostgreSQL** — no server to manage; fine for a single-user research tool
- **Embeddings on disk, not in DB** — a 200-token passage embedding is 200×768×4 = 600 KB; storing thousands as BLOBs would make the DB unwieldy
- **Flask serves the frontend directly** — eliminates PHP as a separate process; `send_from_directory` for HTML and heatmap PNGs keeps everything in one server
- **Auto-ingest sample docs on startup** — users see results immediately without a manual upload step; DB emptiness check prevents re-ingesting on restart
- **Flask over FastAPI** — simpler setup, no async needed for this scale
- **~200-word chunks with overlap** — balances passage length against ColBERT's 300-token limit; overlap prevents boundary cuts from losing context
- **MaxSim scoring** — exact same function ColBERT uses internally, consistent with `visualize.py`
