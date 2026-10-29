"""Task 6 – Cross-encoder reranker (ms-marco-MiniLM-L-6-v2).
rerank(query, chunk_dicts) -> sorted list with reranker_score added.

Features:
  - Cross-encoder scoring: (query, passage) joint attention.
  - Deterministic tie-breaking: sorted by (-score, chunk_id).
  - Passage windowing for long passages (400-word windows, stride 200, score = max).
  - Configurable batch size for CPU efficiency.

Usage:
  python retrieve/reranker.py "Which schools are on Diamond Harbour Road?"
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
    return _reranker


def _window_text(text: str, window_words: int = 400, stride_words: int = 200) -> list[str]:
    """Split long text into overlapping windows for cross-encoder scoring."""
    words = text.split()
    if len(words) <= window_words:
        return [text]
    windows = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + window_words])
        windows.append(chunk)
        if i + window_words >= len(words):
            break
        i += stride_words
    return windows


def rerank(query: str, chunks: list[dict], text_field: str = "text",
           id_field: str = "chunk_id", top_n: int | None = None,
           batch_size: int = 16, window_long_passages: bool = True) -> list[dict]:
    """Score (query, passage) pairs with cross-encoder logit. Returns chunks sorted
    best-first with 'reranker_score' added.
    - Tie-breaking: sorted deterministically by (-score, chunk_id).
    - Long passage windowing: if True, scores overlapping windows and keeps max logit.
    - batch_size: batch size passed to cross-encoder.predict().
    """
    if not chunks:
        return []
    model = get_reranker()

    # Prepare pairs, applying windowing to long passages if requested
    eval_pairs = []
    chunk_index_map = []  # maps pair index to chunk index

    for c_idx, c in enumerate(chunks):
        txt = c.get(text_field, "")
        if window_long_passages and len(txt.split()) > 400:
            windows = _window_text(txt, window_words=400, stride_words=200)
            for w in windows:
                eval_pairs.append((query, w))
                chunk_index_map.append(c_idx)
        else:
            eval_pairs.append((query, txt))
            chunk_index_map.append(c_idx)

    # Cross-encoder batch inference
    scores = model.predict(eval_pairs, batch_size=batch_size).tolist()

    # Aggregate max score per chunk
    chunk_max_scores = {}
    for c_idx, sc in zip(chunk_index_map, scores):
        if c_idx not in chunk_max_scores or sc > chunk_max_scores[c_idx]:
            chunk_max_scores[c_idx] = sc

    for c_idx, c in enumerate(chunks):
        c["reranker_score"] = round(chunk_max_scores.get(c_idx, -999.0), 4)

    # Deterministic tie-break by (-score, chunk_id)
    ranked = sorted(chunks, key=lambda x: (-x["reranker_score"], str(x.get(id_field, ""))))
    return ranked[:top_n] if top_n else ranked


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    from retrieve import hybrid_retrieve, fetch_chunks
    from graph import KnowledgeGraph, load as graph_load

    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="Which schools are on Diamond Harbour Road?")
    ap.add_argument("--source", default="es", choices=["es", "files"])
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=5)
    a = ap.parse_args()

    G = KnowledgeGraph(*graph_load(a.source))
    candidates = hybrid_retrieve(a.query, G, k=a.candidates)
    chunk_texts = fetch_chunks([r["chunk_id"] for r in candidates])
    for r in candidates:
        r["text"] = chunk_texts.get(r["chunk_id"], {}).get("text", "")

    ranked = rerank(a.query, candidates, top_n=a.top_n)

    print(f"\nQuery: {a.query!r}")
    print(f"\nTop-{a.top_n} after reranking (from {len(candidates)} candidates):\n")
    for i, r in enumerate(ranked, 1):
        snippet = (r["text"][:110] + "…") if len(r.get("text", "")) > 110 else r.get("text", "")
        print(f"  {i:2d}. [reranker={r['reranker_score']:+.2f}  rrf={r['rrf_score']:.4f}]  "
              f"{r['chunk_id']:28s}  | {snippet}")
