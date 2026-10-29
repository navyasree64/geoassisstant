"""The two embedding models. bge needs an instruction prefix on QUERIES only; MiniLM needs none.
Vectors are L2-normalised, so cosine similarity = dot product.

Usage:
    from embedder import embed_passages, embed_queries
    p_vecs = embed_passages("bge", ["text 1", "text 2"])
"""
import os, numpy as np
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
MODELS = {
    "bge": dict(name="BAAI/bge-small-en-v1.5", query_prefix="Represent this sentence for searching relevant passages: "),
    "minilm": dict(name="sentence-transformers/all-MiniLM-L6-v2", query_prefix=""),
}
_cache = {}
def get_model(key):
    if key not in _cache:
        from sentence_transformers import SentenceTransformer
        _cache[key] = SentenceTransformer(MODELS[key]["name"], device="cpu")
    return _cache[key]
def embed_passages(key, texts, batch=32):
    return get_model(key).encode(list(texts), batch_size=batch, normalize_embeddings=True, show_progress_bar=False).astype("float32")
def embed_queries(key, queries, batch=32):
    pre = MODELS[key]["query_prefix"]
    return get_model(key).encode([pre + q for q in queries], batch_size=batch, normalize_embeddings=True, show_progress_bar=False).astype("float32")
