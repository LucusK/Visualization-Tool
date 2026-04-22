import io
import os
import sys
import json
import time
import tarfile
from pathlib import Path

import numpy as np
import torch
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel

PROJ_IN_DIM = 768
PROJ_OUT_DIM = 8

_rng = np.random.default_rng(0)  # fixed seed for reproducibility
PROJ_MAT = _rng.standard_normal((PROJ_IN_DIM, PROJ_OUT_DIM), dtype=np.float32)


# --------- paths / project roots ---------
BASE_DIR = Path(__file__).resolve().parents[1]
MSMARCO_TAR = BASE_DIR / "data" / "MSMARCO" / "collection.tar.gz"

OUT_DIR = BASE_DIR / "ann_out" / "msmarco"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = OUT_DIR / "faiss_hnsw.index"
PASSAGE_IDS_PATH = OUT_DIR / "passage_ids.npy"
META_PATH = OUT_DIR / "build_meta.json"

# --------- MuVERA import (same idea as your 03_run_muvera.py) ---------
MUVERA_DIR = BASE_DIR / "muvera-py"
sys.path.append(str(MUVERA_DIR))

from fde_generator import (
    FixedDimensionalEncodingConfig,
    ProjectionType,
    EncodingType,
    generate_document_fde_batch
)

# --------- config knobs ---------
DEVICE = "cpu"  # keep consistent with your existing scripts
BATCH_SIZE = 64  # increase if you have RAM/CPU headroom
MAX_LENGTH = 300

# FAISS HNSW params
HNSW_M = 32

# checkpointing
CHECKPOINT_EVERY_BATCHES = 200  # write index + ids every N batches

# ColBERT filtering ids (matching your script)
CLS_ID = 101
SEP_ID = 102

# --------- utils ---------
def l2_normalize_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32, order="C")
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    return X

def l2_normalize_tokens(toks: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # toks: (T, D)
    toks = np.asarray(toks, dtype=np.float32, order="C")
    toks /= (np.linalg.norm(toks, axis=1, keepdims=True) + eps)
    return toks

def find_collection_tsv_member(tf: tarfile.TarFile) -> tarfile.TarInfo:
    # common names: collection.tsv, collection.tsv.gz, msmarco/collection.tsv, etc.
    members = tf.getmembers()
    # prefer uncompressed tsv if present
    for m in members:
        name = m.name.lower()
        if name.endswith("collection.tsv"):
            return m
    # otherwise allow gz inside tar
    for m in members:
        name = m.name.lower()
        if name.endswith("collection.tsv.gz"):
            return m
    raise FileNotFoundError("Could not find collection.tsv or collection.tsv.gz inside collection.tar.gz")

def iter_collection_lines_from_tar(tar_path: Path):
    """
    Yields (passage_id:int, passage_text:str) from the MSMARCO collection.
    Expects each line formatted as: <passage_id>\t<passage_text>\n
    """
    with tarfile.open(tar_path, "r:gz") as tf:
        member = find_collection_tsv_member(tf)
        f = tf.extractfile(member)
        if f is None:
            raise RuntimeError(f"Failed to extract member: {member.name}")

        # handle optional gzip inside the tar
        if member.name.lower().endswith(".gz"):
            import gzip
            stream = gzip.GzipFile(fileobj=f)
            text_stream = io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
        else:
            text_stream = io.TextIOWrapper(f, encoding="utf-8", errors="replace")

        for line in text_stream:
            line = line.rstrip("\n")
            if not line:
                continue
            # MSMARCO passage collection format: pid \t passage
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            pid_str, passage = parts
            try:
                pid = int(pid_str)
            except:
                continue
            yield pid, passage

def encode_colbert_token_embeddings(tokenizer, model, texts):
    """
    Returns list of np arrays, one per text:
      doc_emb_i: (Ti, 768) float32, with CLS/SEP removed and mask applied.
    """
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            token_type_ids=tokens.get("token_type_ids", None),
        )
        emb = outputs.last_hidden_state  # (B, L, 768)

    docs = []
    for i in range(len(texts)):
        input_ids = tokens["input_ids"][i]
        attention_mask = tokens["attention_mask"][i]
        keep = (attention_mask == 1) & (input_ids != CLS_ID) & (input_ids != SEP_ID)

        doc_emb = emb[i][keep].detach().cpu().numpy().astype(np.float32)  # (Ti, 768)
        docs.append(doc_emb)

    return docs

def build_fde_vectors(docs_token_embs, cfg):
    """
    docs_token_embs: list of (Ti, 768) arrays
    returns: (B, fde_dim) float32
    """
    docs_normed = []
    for toks in docs_token_embs:
    toks = toks @ PROJ_MAT            # (Ti, 768) -> (Ti, 8)
    toks = l2_normalize_tokens(toks)  # normalize in 8-D
    docs_normed.append(toks)

    fdes = generate_document_fde_batch(docs_normed, cfg)  # (B, fde_dim)
    return np.asarray(fdes, dtype=np.float32, order="C")

def save_checkpoint(index, passage_ids, meta, faiss_path, ids_path, meta_path):
    import faiss
    faiss.write_index(index, str(faiss_path))
    np.save(ids_path, np.asarray(passage_ids, dtype=np.int64))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    # ---- load ColBERT ----
    tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    model = AutoModel.from_pretrained("colbert-ir/colbertv2.0").to(DEVICE)
    model.eval()

    # ---- MuVERA cfg (same spirit as your 03_run_muvera.py) ----
    cfg = FixedDimensionalEncodingConfig(
        dimension=8,
        num_repetitions=20,
        num_simhash_projections=3,
        projection_type=ProjectionType.DEFAULT_IDENTITY,
        projection_dimension=8,
        encoding_type=EncodingType.AVERAGE,
        fill_empty_partitions=True,
    )
    fde_dim = cfg.num_repetitions * (2 ** cfg.num_simhash_projections) * cfg.dimension

    # ---- init FAISS HNSW ----
    import faiss
    index = faiss.IndexHNSWFlat(fde_dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)

    # store ids in insertion order (position i corresponds to vector i in the index)
    passage_ids = []

    # ---- build loop ----
    t0 = time.time()
    batch_texts = []
    batch_ids = []
    batches_done = 0
    passages_done = 0

    meta = {
        "source": str(MSMARCO_TAR),
        "colbert_model": "colbert-ir/colbertv2.0",
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "muvera_cfg": {
            "dimension": int(cfg.dimension),
            "num_repetitions": int(cfg.num_repetitions),
            "num_simhash_projections": int(cfg.num_simhash_projections),
            "projection_type": str(cfg.projection_type),
            "projection_dimension": int(cfg.projection_dimension),
            "encoding_type": str(cfg.encoding_type),
            "fill_empty_partitions": bool(cfg.fill_empty_partitions),
        },
        "faiss": {
            "type": "IndexHNSWFlat",
            "metric": "INNER_PRODUCT",
            "hnsw_m": HNSW_M,
            "vector_dim": int(fde_dim),
        },
        "progress": {}
    }

    for pid, text in iter_collection_lines_from_tar(MSMARCO_TAR):
        batch_ids.append(pid)
        batch_texts.append(text)

        if len(batch_texts) < BATCH_SIZE:
            continue

        # ColBERT -> token embeddings
        docs = encode_colbert_token_embeddings(tokenizer, model, batch_texts)

        # MuVERA -> FDE
        fdes = build_fde_vectors(docs, cfg)

        # normalize FDEs so IP behaves like cosine similarity
        fdes = l2_normalize_rows(fdes)

        # add to index
        index.add(fdes)
        passage_ids.extend(batch_ids)

        passages_done += len(batch_ids)
        batches_done += 1

        if batches_done % 25 == 0:
            elapsed = time.time() - t0
            rate = passages_done / max(elapsed, 1e-6)
            print(f"[{batches_done} batches] passages={passages_done} rate={rate:.1f} passages/sec")

        if batches_done % CHECKPOINT_EVERY_BATCHES == 0:
            meta["progress"] = {
                "batches_done": batches_done,
                "passages_done": passages_done,
                "elapsed_sec": round(time.time() - t0, 2),
            }
            save_checkpoint(index, passage_ids, meta, FAISS_INDEX_PATH, PASSAGE_IDS_PATH, META_PATH)
            print(f"✅ checkpoint saved -> {FAISS_INDEX_PATH}")

        batch_ids = []
        batch_texts = []

    # flush last partial batch
    if batch_texts:
        docs = encode_colbert_token_embeddings(tokenizer, model, batch_texts)
        fdes = build_fde_vectors(docs, cfg)
        fdes = l2_normalize_rows(fdes)
        index.add(fdes)
        passage_ids.extend(batch_ids)
        passages_done += len(batch_ids)
        batches_done += 1

    meta["progress"] = {
        "batches_done": batches_done,
        "passages_done": passages_done,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    save_checkpoint(index, passage_ids, meta, FAISS_INDEX_PATH, PASSAGE_IDS_PATH, META_PATH)

    print("\nDone.")
    print(f"FAISS index: {FAISS_INDEX_PATH}")
    print(f"Passage IDs: {PASSAGE_IDS_PATH}")
    print(f"Meta:       {META_PATH}")
    print(f"Total passages indexed: {passages_done}")

if __name__ == "__main__":
    main()
