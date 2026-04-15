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
| **PHP** | Web frontend — file upload UI, search form, results page |
| **SQL (SQLite)** | Stores document metadata and passage chunks |
| **JDBC** | ⚠️ JDBC is Java-specific and does not naturally fit a PHP/Python stack. It is noted here as a stated requirement. If a Java component is needed, a small Java service could act as a database access layer — but for this project, Python (`sqlite3`) and PHP (`PDO`) connect to SQLite directly without JDBC. |
| **Agile** | Development methodology — work in short iterations, one feature at a time |

---

## Architecture

```
Browser
  │
  ▼
PHP Frontend  (upload.php / search.php / results.php)
  │  HTTP calls (form POST / fetch)
  ▼
Python API  (Flask — api.py)
  ├── /ingest   ← receives file, extracts text, chunks, embeds, stores
  └── /search   ← receives query, returns top-10 passages + heatmap paths
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

## Files to Create
 
### Python API
| File | Purpose |
|---|---|
| `app/api.py` | Flask app — `/ingest` and `/search` endpoints |
| `app/extractor.py` | `extract_text(filepath)` — dispatches by extension |
| `app/chunker.py` | `chunk_text(text, size=200, overlap=20)` → list of strings |
| `app/encoder.py` | Loads ColBERT once; `encode(text)` → (emb, tokens) |
| `app/search.py` | `maxsim_score()`, `top_k()` |
| `app/db.py` | SQLite helpers — init schema, insert/query documents & passages |

### PHP Frontend
| File | Purpose |
|---|---|
| `web/index.php` | Landing page — upload form + search bar |
| `web/upload.php` | Handles POST, calls `/ingest`, shows confirmation |
| `web/search.php` | Handles GET `?q=`, calls `/search`, renders results + heatmaps |
| `web/style.css` | Minimal styling |

### Shared
| File | Purpose |
|---|---|
| `db/app.db` | SQLite database (auto-created on first run) |
| `embeddings/` | Per-passage `.npy` embedding files |
| `output/heatmaps/` | Generated heatmap PNGs served to browser |

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

### Iteration 4 — PHP frontend
- `index.php`, `upload.php`, `search.php`, `style.css`
- Test: full end-to-end in browser — upload file, search, see heatmaps

### Iteration 5 — Polish
- Loading spinner during ingest/search (JS fetch + polling)
- Error messages for bad file types, empty queries
- Heatmap displayed inline (as `<img>`) next to each result

---

## Key Decisions & Rationale

- **SQLite over PostgreSQL** — no server to manage; fine for a single-user research tool
- **Embeddings on disk, not in DB** — a 200-token passage embedding is 200×768×4 = 600 KB; storing thousands as BLOBs would make the DB unwieldy
- **PHP for frontend** — satisfies stated stack requirement; simple `curl`/`file_get_contents` calls to the Python API are sufficient
- **Flask over FastAPI** — simpler setup, no async needed for this scale
- **~200-word chunks with overlap** — balances passage length against ColBERT's 300-token limit; overlap prevents boundary cuts from losing context
- **MaxSim scoring** — exact same function ColBERT uses internally, consistent with `visualize.py`
