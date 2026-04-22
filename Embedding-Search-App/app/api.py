import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory

from db import (init_db, insert_document, insert_passage,
                get_all_passages, get_document_name,
                get_all_documents, delete_all)
from extractor import extract_text, extract_url, search_wikipedia
from chunker import chunk_text
from encoder import encode
from search import top_k
from heatmap import render_heatmap

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
HEATMAPS_DIR   = BASE_DIR / "output" / "heatmaps"
UPLOADS_DIR    = BASE_DIR / "output" / "uploads"
WEB_DIR        = BASE_DIR.parent / "web"

for d in (EMBEDDINGS_DIR, HEATMAPS_DIR, UPLOADS_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md", ".rst", ".csv", ".json", ".xml", ".html"}

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB upload limit

init_db()


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _ingest_text(text: str, doc_id: int) -> int:
    """Chunk, encode, and store pre-extracted text. Returns number of passages created."""
    doc_emb_dir = EMBEDDINGS_DIR / str(doc_id)
    doc_emb_dir.mkdir(parents=True, exist_ok=True)

    chunks = chunk_text(text)
    for idx, chunk in enumerate(chunks):
        emb, _tokens = encode(chunk)
        emb_path = doc_emb_dir / f"{idx}.npy"
        np.save(emb_path, emb)
        rel_path = str(emb_path.relative_to(BASE_DIR.parent))
        insert_passage(doc_id, idx, chunk, rel_path)

    return len(chunks)


def _ingest_file(filepath: Path, doc_id: int) -> int:
    """Extract text from a file then ingest it."""
    return _ingest_text(extract_text(filepath), doc_id)


# ── GET / ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


# ── GET /heatmaps/<filename> ──────────────────────────────────────────────────
@app.route("/heatmaps/<path:filename>")
def serve_heatmap(filename):
    return send_from_directory(HEATMAPS_DIR, filename)


# ── GET /documents ────────────────────────────────────────────────────────────
@app.route("/documents", methods=["GET"])
def documents():
    """Return all ingested documents. Used by the UI corpus bar."""
    return jsonify({"documents": get_all_documents()}), 200


# ── POST /reset ───────────────────────────────────────────────────────────────
@app.route("/reset", methods=["POST"])
def reset():
    """Delete all documents, passages, and embedding files. Schema is preserved."""
    import shutil
    delete_all()
    if EMBEDDINGS_DIR.exists():
        shutil.rmtree(EMBEDDINGS_DIR)
    EMBEDDINGS_DIR.mkdir()
    return jsonify({"status": "reset complete"}), 200


# ── POST /ingest ──────────────────────────────────────────────────────────────
@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Accepts a file upload, extracts text, chunks it, encodes each chunk
    with ColBERT, saves embeddings to disk, and stores metadata in SQLite.

    Request:  multipart/form-data  { file: <binary> }
    Response: { doc_id, filename, num_passages }  or  { error }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed(f.filename):
        return jsonify({
            "error": f"File type not allowed. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 415

    safe_name = Path(f.filename).name
    upload_path = UPLOADS_DIR / safe_name
    f.save(upload_path)

    doc_id = insert_document(safe_name, datetime.utcnow().isoformat())

    try:
        num_passages = _ingest_file(upload_path, doc_id)
    except Exception as e:
        return jsonify({"error": f"Ingest failed: {e}"}), 500

    return jsonify({
        "doc_id": doc_id,
        "filename": safe_name,
        "num_passages": num_passages,
    }), 201


# ── POST /ingest_search ───────────────────────────────────────────────────────
@app.route("/ingest_search", methods=["POST"])
def ingest_search():
    """
    Search Wikipedia for a term, ingest the top results automatically.

    Request:  { "term": "decision trees", "limit": 3 }
    Response: { term, added: [{title, doc_id, num_passages}, ...] }  or  { error }
    """
    body  = request.get_json(force=True, silent=True) or {}
    term  = (body.get("term") or "").strip()
    limit = min(int(body.get("limit", 3)), 5)

    if not term:
        return jsonify({"error": "Missing 'term' in request body"}), 400

    try:
        articles = search_wikipedia(term, limit=limit)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    added = []
    for title, text in articles:
        doc_id       = insert_document(title, datetime.utcnow().isoformat())
        num_passages = _ingest_text(text, doc_id)
        added.append({"title": title, "doc_id": doc_id, "num_passages": num_passages})

    return jsonify({"term": term, "added": added}), 201


# ── GET /search ───────────────────────────────────────────────────────────────
@app.route("/search", methods=["GET"])
def search():
    """
    Encodes the query, scores every stored passage with MaxSim, returns
    the top-k results and a heatmap PNG URL for each.

    Request:  ?q=<query string>  [&k=10]
    Response: { query, results: [{passage_id, doc_name, chunk_text, score, heatmap_url}] }
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing query parameter ?q="}), 400

    k = min(int(request.args.get("k", 10)), 50)

    passages = get_all_passages()
    if not passages:
        return jsonify({"error": "No documents ingested yet"}), 404

    q_emb, q_tokens = encode(query)

    passage_embs = [np.load(BASE_DIR.parent / p["emb_path"]) for p in passages]

    results = top_k(q_emb, passage_embs, passages, k=k)

    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    output = []
    for r in results:
        d_emb = passage_embs[r["list_index"]]
        _, d_tokens = encode(r["chunk_text"])

        sim = q_emb @ d_emb.T
        heatmap_filename = f"{query_hash}_p{r['passage_id']}.png"
        render_heatmap(
            sim=sim,
            query_tokens=q_tokens,
            doc_tokens=d_tokens,
            query_text=query,
            doc_text=r["chunk_text"],
            out_path=HEATMAPS_DIR / heatmap_filename,
        )

        output.append({
            "passage_id":  r["passage_id"],
            "doc_name":    get_document_name(r["doc_id"]),
            "chunk_text":  r["chunk_text"],
            "score":       round(float(r["score"]), 4),
            "heatmap_url": f"/heatmaps/{heatmap_filename}",
            "sim_min":     round(float(sim.min()), 3),
            "sim_max":     round(float(sim.max()), 3),
            "query_tokens": list(q_tokens),
        })

    return jsonify({"query": query, "results": output}), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
