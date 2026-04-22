from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed in Docker
import matplotlib.pyplot as plt
import seaborn as sns


def render_heatmap(
    sim: np.ndarray,
    query_tokens: list[str],
    doc_tokens: list[str],
    query_text: str,
    doc_text: str,
    out_path: str | Path,
) -> Path:
    """
    Draw a ColBERT token-level similarity heatmap and save it to out_path.

    Differences from visualize.py's version:
    - Caller controls the output path (api.py uses a query-hash + passage-id
      naming scheme) rather than auto-generating a timestamped filename.
    - Parent directory is created if it doesn't exist.

    Args:
        sim:          Similarity matrix, shape (Q, D), values in [-1, 1].
        query_tokens: Decoded query token strings — Y-axis labels.
        doc_tokens:   Decoded document token strings — X-axis labels.
        query_text:   Raw query string for the title.
        doc_text:     Raw document text for the title (truncated to 80 chars).
        out_path:     Destination path for the PNG file.

    Returns:
        Resolved Path of the saved file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Q, D = sim.shape

    fig_w = max(10, D * 0.45)
    fig_h = max(6,  Q * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        sim,
        ax=ax,
        xticklabels=doc_tokens,
        yticklabels=False,
        cmap="RdYlGn",
        vmin=float(sim.min()),
        vmax=float(sim.max()),
        linewidths=0.3,
        linecolor="grey",
        annot=(Q * D <= 400),   # numbers only for small matrices — avoids clutter
        fmt=".2f",
        cbar_kws={"label": "cosine similarity"},
    )

    # mark the MaxSim winner per query token with *
    winners = np.argmax(sim, axis=1)
    for row, col in enumerate(winners):
        ax.text(
            col + 0.5, row + 0.5, "*",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color="black",
        )

    ax.set_title(
        f'Query: "{query_text}"\n'
        f'Doc: "{doc_text[:80]}{"…" if len(doc_text) > 80 else ""}"',
        fontsize=10, pad=12,
    )
    ax.set_xlabel("Document tokens", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path.resolve()
