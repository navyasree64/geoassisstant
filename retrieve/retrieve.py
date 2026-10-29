"""Task 5 – Hybrid retrieval: BM25 + KNN (dense) + graph expansion, fused by Reciprocal Rank Fusion (RRF).

Exposes four primary retrieval functions returning ranked [(chunk_id, score)]:
  - keyword(): BM25 via Elasticsearch match query
  - vector(): kNN dense vector search (bge-small-en-v1.5)
  - hybrid(): RRF fusion of BM25 + kNN (default constant k=60)
  - graph_hybrid(): RRF fusion of BM25 + kNN + Knowledge Graph traversal

All functions support optional metadata_filter and geo_filter (implemented with Elasticsearch geo_distance).

Usage (standalone):
    python retrieve/retrieve.py "Which schools apply to WBBSE?"
    python retrieve/retrieve.py --source files "..."
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from common import es, load_judgments
from embedder import embed_queries
from graph import KnowledgeGraph, load as graph_load

# ---------- RRF Constant -------------------------------------------------------
RRF_K = 60          # Cormack et al. (2009) recommended constant


def _build_filter(metadata_filter: dict | None = None, geo_filter: dict | None = None) -> list[dict]:
    """Construct Elasticsearch filter clauses for metadata and geo-distance."""
    clauses = []
    if metadata_filter:
        if "boards" in metadata_filter:
            b = metadata_filter["boards"]
            if isinstance(b, list):
                clauses.append({"terms": {"boards": b}})
            else:
                clauses.append({"term": {"boards": b}})
        if "doc_type" in metadata_filter:
            clauses.append({"term": {"doc_type": metadata_filter["doc_type"]}})
        if "issue_date" in metadata_filter:
            clauses.append({"range": {"issue_date": metadata_filter["issue_date"]}})
    if geo_filter:
        lat = geo_filter.get("lat")
        lon = geo_filter.get("lon")
        dist = geo_filter.get("distance", "5km")
        if lat is not None and lon is not None:
            clauses.append({
                "geo_distance": {
                    "distance": dist,
                    "school_locations": {"lat": lat, "lon": lon}
                }
            })
    return clauses


def rrf(rankings: dict[str, list[str]], rrf_k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.  rankings = {source_name: [chunk_id, ...] best-first}."""
    scores: dict[str, float] = {}
    for ranked in rankings.values():
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


# ---------- 1. keyword (BM25) --------------------------------------------------
def keyword(query: str, k: int = 20, metadata_filter: dict | None = None,
            geo_filter: dict | None = None) -> list[tuple[str, float]]:
    """BM25 keyword search via multi_match with optional metadata and geo filters."""
    filter_clauses = _build_filter(metadata_filter, geo_filter)
    query_clause = {
        "multi_match": {
            "query": query,
            "fields": ["text", "circular_number^2"],
            "type": "best_fields",
            "analyzer": "english"
        }
    }
    if filter_clauses:
        es_query = {"bool": {"must": query_clause, "filter": filter_clauses}}
    else:
        es_query = query_clause

    body = {"size": k, "_source": ["chunk_id"], "query": es_query}
    hits = es("POST", "/east_india_chunks/_search", json=body).json()["hits"]["hits"]
    return [(h["_source"]["chunk_id"], float(h["_score"])) for h in hits]


# ---------- 2. vector (kNN) ----------------------------------------------------
def vector(query: str, k: int = 20, num_candidates: int = 50,
           metadata_filter: dict | None = None,
           geo_filter: dict | None = None) -> list[tuple[str, float]]:
    """kNN dense semantic search with optional metadata and geo_distance filters."""
    qv = embed_queries("bge", [query])[0].tolist()
    filter_clauses = _build_filter(metadata_filter, geo_filter)
    eff_candidates = max(num_candidates, k)
    knn_clause = {
        "field": "embedding",
        "query_vector": qv,
        "k": k,
        "num_candidates": eff_candidates
    }
    if filter_clauses:
        knn_clause["filter"] = filter_clauses if len(filter_clauses) > 1 else filter_clauses[0]

    body = {"size": k, "_source": ["chunk_id"], "knn": knn_clause}
    hits = es("POST", "/east_india_chunks/_search", json=body).json()["hits"]["hits"]
    return [(h["_source"]["chunk_id"], float(h["_score"])) for h in hits]


# ---------- 3. hybrid (RRF BM25 + kNN) -----------------------------------------
def hybrid(query: str, k: int = 20, rrf_k: int = RRF_K,
           metadata_filter: dict | None = None,
           geo_filter: dict | None = None,
           num_candidates: int = 50) -> list[tuple[str, float]]:
    """RRF fusion of BM25 and kNN. Default RRF constant is k=60."""
    eff_k = k * 2
    eff_cand = max(num_candidates, eff_k)
    kw_hits = keyword(query, k=eff_k, metadata_filter=metadata_filter, geo_filter=geo_filter)
    vec_hits = vector(query, k=eff_k, num_candidates=eff_cand, metadata_filter=metadata_filter, geo_filter=geo_filter)
    rankings = {
        "bm25": [cid for cid, _ in kw_hits],
        "knn": [cid for cid, _ in vec_hits]
    }
    return rrf(rankings, rrf_k=rrf_k)[:k]


# ---------- 4. graph_hybrid (hybrid + graph expansion) -------------------------
def graph_hybrid(query: str, G: KnowledgeGraph, k: int = 20, rrf_k: int = RRF_K,
                 metadata_filter: dict | None = None,
                 geo_filter: dict | None = None,
                 graph_top: int = 20, num_candidates: int = 50) -> list[tuple[str, float]]:
    """Hybrid RRF fusion combining BM25, kNN, and graph traversal candidates."""
    eff_k = k * 2
    eff_cand = max(num_candidates, eff_k)
    kw_hits = keyword(query, k=eff_k, metadata_filter=metadata_filter, geo_filter=geo_filter)
    vec_hits = vector(query, k=eff_k, num_candidates=eff_cand, metadata_filter=metadata_filter, geo_filter=geo_filter)
    g_ids = graph_expand(query, G, top_n=graph_top)

    rankings = {
        "bm25": [cid for cid, _ in kw_hits],
        "knn": [cid for cid, _ in vec_hits]
    }
    if g_ids:
        rankings["graph"] = g_ids
    return rrf(rankings, rrf_k=rrf_k)[:k]


# ---------- Graph expansion helper ---------------------------------------------
def graph_expand(query: str, G: KnowledgeGraph, top_n: int = 20) -> list[str]:
    detected = G.detect_entities(query)
    if not detected:
        return []
    seed_ids = [eid for eid, _, _ in detected]
    expanded = G.expand(seed_ids)
    return [x["chunk_id"] for x in expanded[:top_n]]


# ---------- Backward compatibility wrappers ------------------------------------
def bm25(query: str, k: int = 20) -> list[str]:
    return [cid for cid, _ in keyword(query, k=k)]


def knn(query: str, k: int = 20, num_candidates: int = 50) -> list[str]:
    return [cid for cid, _ in vector(query, k=k, num_candidates=num_candidates)]


def hybrid_retrieve(query: str, G: KnowledgeGraph, k: int = 10,
                    bm25_k: int = 20, knn_k: int = 20, graph_top: int = 20,
                    use_graph: bool = True) -> list[dict]:
    """Legacy wrapper returning list of dicts with sources."""
    rankings: dict[str, list[str]] = {}
    rankings["bm25"] = bm25(query, bm25_k)
    rankings["knn"] = knn(query, knn_k)
    if use_graph:
        g_ids = graph_expand(query, G, graph_top)
        if g_ids:
            rankings["graph"] = g_ids

    fused = rrf(rankings)[:k]
    source_map: dict[str, list[str]] = {}
    for name, ranked in rankings.items():
        for cid in ranked:
            source_map.setdefault(cid, []).append(name)
    return [{"chunk_id": cid, "rrf_score": round(sc, 6),
             "sources": source_map.get(cid, [])} for cid, sc in fused]


def fetch_chunks(chunk_ids: list[str]) -> dict[str, dict]:
    """Return {chunk_id: {text, doc_id, ...}} for a list of ids (parallel mget)."""
    if not chunk_ids:
        return {}
    body = {"ids": chunk_ids}
    docs = es("POST", "/east_india_chunks/_mget", json=body).json()["docs"]
    return {d["_id"]: d.get("_source", {}) for d in docs if d.get("found")}


# ---------- CLI demo -----------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="Which schools apply to WBBSE in the new circular?")
    ap.add_argument("--source", default="es", choices=["es", "files"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-graph", action="store_true")
    a = ap.parse_args()

    G = KnowledgeGraph(*graph_load(a.source))
    results = hybrid_retrieve(a.query, G, k=a.k, use_graph=not a.no_graph)
    print(f"\nQuery: {a.query!r}")
    print(f"Top-{a.k} fused results (BM25 + KNN + graph RRF):\n")
    chunks = fetch_chunks([r["chunk_id"] for r in results])
    for i, r in enumerate(results, 1):
        c = chunks.get(r["chunk_id"], {})
        snippet = (c.get("text", "")[:100] + "…") if c.get("text") else "—"
        print(f"  {i:2d}. [{r['rrf_score']:.4f}] {r['chunk_id']:25s}  via {'+'.join(r['sources'])}  |  {snippet}")
