import sys
import time
import numpy as np
import json
from numpy.linalg import norm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MUVERA_DIR = BASE_DIR/"muvera-py"
sys.path.append(str(MUVERA_DIR))

# importing FDE generator from muvera-py
from fde_generator import (
    FixedDimensionalEncodingConfig,
    ProjectionType,
    EncodingType,
    generate_document_fde_batch
)

# setting paths for ColBERT output from 02_colbert_encode.py
EMB_PATH = BASE_DIR / "colbert_out" / "bow" / "doc_embeddings.npy"
LENS_PATH = BASE_DIR / "colbert_out" / "bow" / "token_counts.json"

# loading embeddings and document token counts
emb = np.load(EMB_PATH, mmap_mode="r")  
with open(LENS_PATH, "r", encoding="utf-8") as f:
    doc_lens = json.load(f)
doc_lens = np.asarray(doc_lens, dtype=np.int32)

# computing document offsets for slicing embeddings
doc_offsets = np.zeros(len(doc_lens) + 1, dtype=np.int64)
np.cumsum(doc_lens, out=doc_offsets[1:])

# printing status updates to console
print(f"Loaded embeddings shape: {emb.shape}")
print(f"Doc token counts: {doc_lens.tolist()}")

# normalizing function
def l2_normalize(X):
    X /= (norm(X, axis=1, keepdims=True) + 1e-8)

# treating all passages as one document for benchmarking
s, e = doc_offsets[0], doc_offsets[-1]
toks = np.array(emb[s:e], dtype=np.float32)
l2_normalize(toks)
docs = [toks]  

# configuring FDE parameters
cfg = FixedDimensionalEncodingConfig(
    dimension=768,               
    num_repetitions=5,              
    num_simhash_projections=7,     
    projection_type=ProjectionType.DEFAULT_IDENTITY,
    projection_dimension=32,
    encoding_type=EncodingType.AVERAGE,
    fill_empty_partitions=True,
)

print("Starting MUVERA FDE generation...")
start = time.perf_counter()

# generating FDEs
fdes = generate_document_fde_batch(docs, cfg)
elapsed = time.perf_counter() - start

# saving generated FDEs
OUT_DIR = BASE_DIR / "muvera_out" / "fde"
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUT_DIR / "sample_FDE.npy", fdes[0])

fde_dim = cfg.num_repetitions * (2 ** cfg.num_simhash_projections) * cfg.dimension
print(f"\n✅ Generated 1 document FDE")
print(f"Expected FDE dimension  : {fde_dim}")
print(f"FDE shape               : {fdes.shape}")
print(f"FDE dtype               : {fdes.dtype}")
print(f"Per-vector size         : {fdes[0].nbytes / 1024.0:.2f} KB")
print(f"Generation time         : {elapsed:.4f} sec total ({elapsed:.4f} sec/doc)")