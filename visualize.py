import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "colbert-ir/colbertv2.0"
CLS_ID = 101
SEP_ID = 102
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# ── Model loading (done once) ─────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to("cpu").eval()
print("Model ready.\n")


# ── Step 1: Encode ────────────────────────────────────────────────────────────
def encode(text: str) -> tuple[np.ndarray, list[str]]:
    """
    Tokenize `text`, run ColBERT, strip [CLS]/[SEP], L2-normalise.
    Returns:
        emb    — float32 array of shape (num_tokens, 768)
        labels — decoded token strings for axis labels
    """
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=300,
    )

    input_ids = tokens["input_ids"][0]           # shape: (seq_len,)
    attention_mask = tokens["attention_mask"][0]

    # which positions to keep (real tokens, not [CLS] or [SEP])
    keep_mask = (
        (attention_mask == 1) &
        (input_ids != CLS_ID) &
        (input_ids != SEP_ID)
    )

    with torch.no_grad():
        outputs = model(**tokens)

    # last_hidden_state: (1, seq_len, 768) → strip batch dim
    hidden = outputs.last_hidden_state[0]        # (seq_len, 768)
    emb = hidden[keep_mask].cpu().numpy().astype(np.float32)

    # decode kept token IDs for axis labels
    kept_ids = input_ids[keep_mask].tolist()
    labels = tokenizer.convert_ids_to_tokens(kept_ids)

    # Step 2: L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb = emb / norms

    return emb, labels


# ── Step 3: Similarity matrix ─────────────────────────────────────────────────
def similarity_matrix(q_emb: np.ndarray, d_emb: np.ndarray) -> np.ndarray:
    """
    sim[i, j] = cosine_similarity(query_token_i, doc_token_j)
    Shape: (num_query_tokens, num_doc_tokens)
    """
    return q_emb @ d_emb.T


# ── Step 4: Heatmap ───────────────────────────────────────────────────────────
def render_heatmap(
    sim: np.ndarray,
    query_tokens: list[str],
    doc_tokens: list[str],
    query_text: str,
    doc_text: str,
) -> Path:
    """
    Draw a seaborn heatmap of the similarity matrix.
    Marks the argmax doc token per query token (ColBERT MaxSim winner) with *.
    Saves to output/heatmap_<timestamp>.png and returns the path.
    """
    Q, D = sim.shape

    # dynamic figure size — grows with token count
    fig_w = max(10, D * 0.45)
    fig_h = max(6,  Q * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        sim,
        ax=ax,
        xticklabels=doc_tokens,
        yticklabels=query_tokens,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        linecolor="grey",
        annot=(Q * D <= 400),   # show numbers only for small matrices
        fmt=".2f",
        cbar_kws={"label": "cosine similarity"},
    )

    # mark the MaxSim winner per query token with *
    winners = np.argmax(sim, axis=1)   # shape: (Q,)
    for row, col in enumerate(winners):
        ax.text(
            col + 0.5, row + 0.5, "*",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color="black",
        )

    ax.set_title(
        f'Query: "{query_text}"\nDoc: "{doc_text[:80]}{"…" if len(doc_text) > 80 else ""}"',
        fontsize=10, pad=12,
    )
    ax.set_xlabel("Document tokens", fontsize=10)
    ax.set_ylabel("Query tokens", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"heatmap_{timestamp}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ── Step 5: CLI entry point ───────────────────────────────────────────────────
def run_once(query: str, document: str):
    print(f"Encoding query:    {query!r}")
    q_emb, q_tokens = encode(query)
    print(f"  → {len(q_tokens)} query tokens: {q_tokens}\n")

    print(f"Encoding document: {document!r}")
    d_emb, d_tokens = encode(document)
    print(f"  → {len(d_tokens)} document tokens: {d_tokens}\n")

    sim = similarity_matrix(q_emb, d_emb)
    print(f"Similarity matrix shape: {sim.shape}  (query tokens × doc tokens)")
    print(f"  min={sim.min():.3f}  max={sim.max():.3f}  mean={sim.mean():.3f}\n")

    out = render_heatmap(sim, q_tokens, d_tokens, query, document)
    print(f"Heatmap saved → {out}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ColBERT token-level similarity heatmap"
    )
    parser.add_argument("query",    nargs="?", help="Query string (omit for interactive mode)")
    parser.add_argument("document", nargs="?", help="Document string (omit for interactive mode)")
    args = parser.parse_args()

    if args.query and args.document:
        # single-shot mode
        run_once(args.query, args.document)
    else:
        # interactive mode — model already loaded, no reload between runs
        print("Interactive mode. Enter a blank line to quit.\n")
        while True:
            query = input("Query    > ").strip()
            if not query:
                break
            document = input("Document > ").strip()
            if not document:
                break
            print()
            run_once(query, document)


if __name__ == "__main__":
    main()
