import numpy as np


def maxsim_score(q_emb: np.ndarray, d_emb: np.ndarray) -> float:
    """
    ColBERT MaxSim scoring function.

    For each query token, find its highest cosine similarity against any
    document token, then sum those maxima across all query tokens.

    score(Q, D) = Σ_i  max_j  cos_sim(Q[i], D[j])

    Both embeddings must already be L2-normalised (encoder.py does this),
    so cos_sim == dot product.

    Args:
        q_emb: float32 array, shape (Q, 768)
        d_emb: float32 array, shape (D, 768)

    Returns:
        Scalar score — higher means more relevant.
    """
    sim = q_emb @ d_emb.T          # (Q, D)
    return float(sim.max(axis=1).sum())


def top_k(
    q_emb: np.ndarray,
    passage_embs: list[np.ndarray],
    passages: list[dict],
    k: int = 10,
) -> list[dict]:
    """
    Score all passages with MaxSim and return the top-k.

    Args:
        q_emb:        Encoded query, shape (Q, 768).
        passage_embs: List of encoded passage arrays, one per passage.
        passages:     Parallel list of passage dicts from db.get_all_passages()
                      — each has keys: id, doc_id, chunk_text, emb_path.
        k:            Number of results to return.

    Returns:
        List of up to k dicts, sorted by score descending, each containing:
            passage_id, doc_id, chunk_text, emb_path, score, list_index
        list_index is the position in passage_embs/passages so api.py can
        retrieve the embedding for heatmap rendering without reloading it.
    """
    scores = np.array([
        maxsim_score(q_emb, d_emb)
        for d_emb in passage_embs
    ])

    # argsort descending — take top k
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        p = passages[idx]
        results.append({
            "passage_id": p["id"],
            "doc_id":     p["doc_id"],
            "chunk_text": p["chunk_text"],
            "emb_path":   p["emb_path"],
            "score":      scores[idx],
            "list_index": int(idx),   # position in passage_embs list
        })

    return results
