import os
import hashlib
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify

from db import init_db, insert_document, insert_passage, get_all_passages, get_document_name
from extractor import extract_text
from chunker import chunk_text
from encoder import encode
from search import top_k
from heatmap import render_heatmap

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
HEATMAPS_DIR   = BASE_DIR / "output" / "heatmaps"
UPLOADS_DIR    = BASE_DIR / "output" / "uploads"   # temp storage for incoming files

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

    # save uploaded file to temp location
    safe_name = Path(f.filename).name          # strip any path components
    upload_path = UPLOADS_DIR / safe_name
    f.save(upload_path)

    # insert document record and get its id
    doc_id = insert_document(safe_name, datetime.utcnow().isoformat())

    # per-document embedding directory
    doc_emb_dir = EMBEDDINGS_DIR / str(doc_id)
    doc_emb_dir.mkdir(parents=True, exist_ok=True)

    try:
        text = extract_text(upload_path)
    except Exception as e:
        return jsonify({"error": f"Text extraction failed: {e}"}), 500

    chunks = chunk_text(text)

    for idx, chunk in enumerate(chunks):
        emb, _tokens = encode(chunk)

        emb_path = doc_emb_dir / f"{idx}.npy"
        import numpy as np
        np.save(emb_path, emb)

        # store path relative to BASE_DIR.parent so it stays portable
        rel_path = str(emb_path.relative_to(BASE_DIR.parent))
        insert_passage(doc_id, idx, chunk, rel_path)

    return jsonify({
        "doc_id": doc_id,
        "filename": safe_name,
        "num_passages": len(chunks),
    }), 201


# ── GET /search ───────────────────────────────────────────────────────────────
@app.route("/search", methods=["GET"])
def search():
    """
    Encodes the query, scores every stored passage with MaxSim, returns
    the top-10 results and a heatmap PNG URL for each.

    Request:  ?q=<query string>  [&k=10]
    Response: { query, results: [{passage_id, doc_name, chunk_text, score, heatmap_url}] }
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing query parameter ?q="}), 400

    k = min(int(request.args.get("k", 10)), 50)   # cap at 50

    passages = get_all_passages()   # [{id, doc_id, chunk_text, emb_path}, ...]
    if not passages:
        return jsonify({"error": "No documents ingested yet"}), 404

    import numpy as np
    q_emb, q_tokens = encode(query)

    # load all passage embeddings
    passage_embs = []
    for p in passages:
        emb_abs = BASE_DIR.parent / p["emb_path"]
        passage_embs.append(np.load(emb_abs))

    results = top_k(q_emb, passage_embs, passages, k=k)

    # generate a heatmap for each result
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    output = []
    for r in results:
        d_emb = passage_embs[r["list_index"]]
        d_tokens_decoded = None   # render_heatmap re-encodes to get token labels

        heatmap_filename = f"{query_hash}_p{r['passage_id']}.png"
        heatmap_path = HEATMAPS_DIR / heatmap_filename

        # re-encode passage to get token labels (embeddings already computed above)
        _, d_tokens = encode(r["chunk_text"])
        _, q_tokens_labels = encode(query)

        import numpy as np
        sim = q_emb @ d_emb.T
        render_heatmap(
            sim=sim,
            query_tokens=q_tokens_labels,
            doc_tokens=d_tokens,
            query_text=query,
            doc_text=r["chunk_text"],
            out_path=heatmap_path,
        )

        output.append({
            "passage_id":   r["passage_id"],
            "doc_name":     get_document_name(r["doc_id"]),
            "chunk_text":   r["chunk_text"],
            "score":        round(float(r["score"]), 4),
            "heatmap_url":  f"/heatmaps/{heatmap_filename}",
        })

    return jsonify({"query": query, "results": output}), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
