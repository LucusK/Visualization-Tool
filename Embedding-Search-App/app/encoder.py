import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "colbert-ir/colbertv2.0"
CLS_ID = 101
SEP_ID = 102

# ── Model loading (once per process) ─────────────────────────────────────────
# Loaded at import time so both /ingest and /search share the same instance
# without reloading from disk on every request.
print(f"Loading {MODEL_NAME} ...")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModel.from_pretrained(MODEL_NAME).to("cpu").eval()
print("Encoder ready.\n")


# ── Public API ────────────────────────────────────────────────────────────────

def encode(text: str) -> tuple[np.ndarray, list[str]]:
    """
    Tokenize text, run ColBERT, strip [CLS]/[SEP], L2-normalise.

    Returns:
        emb    — float32 array, shape (num_tokens, 768)
        tokens — decoded token strings for heatmap axis labels
    """
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=300,
    )

    input_ids     = inputs["input_ids"][0]       # (seq_len,)
    attention_mask = inputs["attention_mask"][0]

    keep = (
        (attention_mask == 1) &
        (input_ids != CLS_ID) &
        (input_ids != SEP_ID)
    )

    with torch.no_grad():
        hidden = _model(**inputs).last_hidden_state[0]  # (seq_len, 768)

    emb = hidden[keep].cpu().numpy().astype(np.float32)

    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb = emb / norms

    tokens = _tokenizer.convert_ids_to_tokens(input_ids[keep].tolist())

    return emb, tokens
