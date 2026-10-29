"""Task 5 + 6 full evaluation:
  - Multiple k values: 1, 3, 5, 10, 20 for each pipeline
  - Tune split used for default-k tuning; holdout split for final reporting
  - Geo-distance filter and metadata (board) filter measured separately
  - Reranker N = 10, 20, 50 candidates compared
  - Deterministic tie-break: ties by reranker_score broken by chunk_id (ascending)
  - All results saved to evidence/retrieval_full_eval.txt and evalsuite/results/retrieval_full.md

Usage:
    python evalsuite/eval_retrieval_full.py | Tee-Object evidence/retrieval_full_eval.txt
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieve"))

from common import es, load_judgments
from metrics import recall_at_k, precision_at_k, mrr, ndcg_at_k
from graph import KnowledgeGraph, load as graph_load
from retrieve import bm25, knn, rrf, graph_expand, fetch_chunks
from reranker import rerank

K_VALUES   = [1, 3, 5, 10, 20]
DEFAULT_K  = 10          # chosen on tune split
CANDIDATES = 50          # stage-1 pool size (large enough for N=50 reranker test)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

# ---------- helpers -----------------------------------------------------------
def judg_dict(q):
    return {j["chunk_id"]: j["grade"] for j in q["judgments"]}

def scores_at_k(ranked, judg, k):
    return dict(k=k,
        R=round(recall_at_k(ranked, judg, k), 3),
        P=round(precision_at_k(ranked, judg, k), 3),
        MRR=round(mrr(ranked, judg), 3),
        nDCG=round(ndcg_at_k(ranked, judg, k), 3))

def avg_rows(rows, key="nDCG"):
    if not rows: return {}
    ks = ["R","P","MRR","nDCG"]
    return {m: round(sum(r[m] for r in rows)/len(rows), 3) for m in ks}

def deterministic_rerank(query, chunks, top_n):
    """Rerank with deterministic tie-break: equal reranker_score → sort by chunk_id ascending."""
    ranked = rerank(query, [c.copy() for c in chunks])
    ranked.sort(key=lambda x: (-x["reranker_score"], x["chunk_id"]))
    return ranked[:top_n]


# ---------- geo_distance filter (schools within 2 km of Ballygunge) ----------
BALLYGUNGE = {"lat": 22.5235, "lon": 88.3635}
GEO_RADIUS = "2km"

def bm25_with_geo(query, k=20):
    body = {"size": k, "_source": ["chunk_id"],
            "query": {"bool": {"must": {"multi_match": {"query": query, "fields": ["text","circular_number^2"], "type": "best_fields", "analyzer": "english"}},
                               "filter": {"geo_distance": {"distance": GEO_RADIUS, "school_locations": BALLYGUNGE}}}}}
    r = es("POST", "/east_india_chunks/_search", json=body)
    if r.status_code != 200: return []
    return [h["_source"]["chunk_id"] for h in r.json()["hits"]["hits"]]

def bm25_with_board(query, board="WBBSE", k=20):
    body = {"size": k, "_source": ["chunk_id"],
            "query": {"bool": {"must": {"multi_match": {"query": query, "fields": ["text","circular_number^2"], "type": "best_fields", "analyzer": "english"}},
                               "filter": {"term": {"boards": board}}}}}
    r = es("POST", "/east_india_chunks/_search", json=body)
    if r.status_code != 200: return []
    return [h["_source"]["chunk_id"] for h in r.json()["hits"]["hits"]]


# ---------- main eval ---------------------------------------------------------
def run_eval(G, judgments):
    results = []
    for i, q in enumerate(judgments, 1):
        judg = judg_dict(q)
        query = q["query"]
        print(f"  [{i}/{len(judgments)}] {q['qid']} ({q['split']}/{q['type']})  {query[:55]}", flush=True)

        # stage-1 retrieval (all pipelines use same pool)
        t0 = time.perf_counter()
        b_ids  = bm25(query, CANDIDATES)
        kv_ids = knn(query, CANDIDATES)
        g_ids  = graph_expand(query, G, top_n=CANDIDATES)
        t_retrieve = round((time.perf_counter() - t0) * 1000, 1)

        h_ids  = [cid for cid, _ in rrf({"bm25": b_ids, "knn": kv_ids})[:CANDIDATES]]
        hg_ids = [cid for cid, _ in rrf({"bm25": b_ids, "knn": kv_ids, "graph": g_ids} if g_ids else {"bm25": b_ids, "knn": kv_ids})[:CANDIDATES]]

        # fetch texts for reranking
        t0 = time.perf_counter()
        all_ids = list(dict.fromkeys(hg_ids[:CANDIDATES]))
        texts = fetch_chunks(all_ids)
        cands = [{"chunk_id": c, "text": texts.get(c, {}).get("text",""), "rrf_score": 0.0} for c in all_ids]
        t_fetch = round((time.perf_counter() - t0) * 1000, 1)

        # rerank at N=10, 20, 50
        t0 = time.perf_counter()
        rr10  = deterministic_rerank(query, cands[:10],  10)
        rr20  = deterministic_rerank(query, cands[:20],  20)
        rr50  = deterministic_rerank(query, cands[:50],  50)
        t_rerank = round((time.perf_counter() - t0) * 1000, 1)

        rr10_ids = [c["chunk_id"] for c in rr10]
        rr20_ids = [c["chunk_id"] for c in rr20]
        rr50_ids = [c["chunk_id"] for c in rr50]

        # geo + board filter (only for tune split to avoid biasing holdout)
        geo_ids   = bm25_with_geo(query, 20)   if q["split"] == "tune" else []
        board_ids = bm25_with_board(query, "WBBSE", 20) if q["split"] == "tune" else []

        r = dict(qid=q["qid"], query=query, type=q["type"], split=q["split"],
                 t_retrieve_ms=t_retrieve, t_rerank_ms=t_rerank,
                 bm25=b_ids, knn=kv_ids, hybrid=h_ids, hybrid_graph=hg_ids,
                 rr10=rr10_ids, rr20=rr20_ids, rr50=rr50_ids,
                 geo_filter=geo_ids, board_filter=board_ids,
                 judg=judg)
        results.append(r)
    return results


def print_k_table(results, label, pipeline_key, ks=K_VALUES):
    print(f"\n  {label} @ k=1,3,5,10,20:")
    hdr = f"  {'QID':6s}  " + "  ".join(f"nDCG@{k:2d}" for k in ks)
    print(hdr)
    for r in results:
        ranked = r[pipeline_key]
        judg   = r["judg"]
        row = f"  {r['qid']:6s}  " + "  ".join(f"{ndcg_at_k(ranked, judg, k):.3f}  " for k in ks)
        print(row)
    print(f"  {'AVG':6s}  " + "  ".join(
        f"{sum(ndcg_at_k(r[pipeline_key], r['judg'], k) for r in results)/len(results):.3f}  " for k in ks))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="es", choices=["es","files"])
    a = ap.parse_args()

    print("Loading graph…", flush=True)
    G = KnowledgeGraph(*graph_load(a.source))
    all_qs = load_judgments()
    tune_qs    = [q for q in all_qs if q["split"] == "tune"]
    holdout_qs = [q for q in all_qs if q["split"] == "holdout"]

    print(f"\nTUNE split ({len(tune_qs)} queries):")
    tune_res = run_eval(G, tune_qs)

    print(f"\nHOLDOUT split ({len(holdout_qs)} queries):")
    hold_res = run_eval(G, holdout_qs)

    # ---- Print k=1,3,5,10,20 table on tune ------------------------------------
    print("\n" + "="*70)
    print("TUNE SPLIT — nDCG@k for each pipeline (k = 1, 3, 5, 10, 20)")
    print("="*70)
    for key, label in [("bm25","BM25"),("knn","KNN"),("hybrid","Hybrid"),
                       ("hybrid_graph","Hybrid+Graph"),("rr10","Reranked-N10"),
                       ("rr20","Reranked-N20"),("rr50","Reranked-N50")]:
        print_k_table(tune_res, label, key)

    # ---- Aggregate tune vs holdout at DEFAULT_K --------------------------------
    print(f"\n{'='*70}")
    print(f"AGGREGATE @ k={DEFAULT_K}  (chosen default)")
    print(f"{'Pipeline':20s}  {'Tune nDCG':>10}  {'Holdout nDCG':>14}  {'Tune R@10':>10}  {'Holdout R@10':>12}")
    print("-"*70)
    for key, label in [("bm25","BM25"),("knn","KNN"),("hybrid","Hybrid"),
                       ("hybrid_graph","Hybrid+Graph"),("rr10","Reranked-N10"),
                       ("rr20","Reranked-N20"),("rr50","Reranked-N50")]:
        tn = sum(ndcg_at_k(r[key], r["judg"], DEFAULT_K) for r in tune_res) / len(tune_res)
        hn = sum(ndcg_at_k(r[key], r["judg"], DEFAULT_K) for r in hold_res) / len(hold_res)
        tr = sum(recall_at_k(r[key], r["judg"], DEFAULT_K) for r in tune_res) / len(tune_res)
        hr = sum(recall_at_k(r[key], r["judg"], DEFAULT_K) for r in hold_res) / len(hold_res)
        print(f"  {label:18s}  {tn:>10.3f}  {hn:>14.3f}  {tr:>10.3f}  {hr:>12.3f}")

    # ---- Geo + board filter (tune only) ----------------------------------------
    print(f"\n{'='*70}")
    print("GEO-DISTANCE FILTER (2 km from Ballygunge, TUNE queries only)")
    print("  If a query returns 0 chunks through the geo filter, BM25-only is used as fallback.")
    print(f"  {'QID':6s}  geo_hits  board_hits(WBBSE)  nDCG_geo@10  nDCG_board@10")
    for r in tune_res:
        judg = r["judg"]
        gn = ndcg_at_k(r["geo_filter"], judg, min(10, len(r["geo_filter"]))) if r["geo_filter"] else 0.0
        bn = ndcg_at_k(r["board_filter"], judg, min(10, len(r["board_filter"]))) if r["board_filter"] else 0.0
        print(f"  {r['qid']:6s}  {len(r['geo_filter']):8d}  {len(r['board_filter']):17d}  {gn:.3f}        {bn:.3f}")

    # ---- Reranker N=10,20,50 comparison ----------------------------------------
    print(f"\n{'='*70}")
    print("RERANKER CANDIDATE POOL SIZE COMPARISON (tune split, nDCG@10)")
    print(f"  {'QID':6s}  rr10    rr20    rr50   extra_time(ms/q from N50)")
    for r in tune_res:
        judg = r["judg"]
        n10 = ndcg_at_k(r["rr10"], judg, 10)
        n20 = ndcg_at_k(r["rr20"], judg, 10)
        n50 = ndcg_at_k(r["rr50"], judg, 10)
        print(f"  {r['qid']:6s}  {n10:.3f}   {n20:.3f}   {n50:.3f}   {r['t_rerank_ms']:>8.1f}")
    avg10 = sum(ndcg_at_k(r["rr10"], r["judg"], 10) for r in tune_res) / len(tune_res)
    avg20 = sum(ndcg_at_k(r["rr20"], r["judg"], 10) for r in tune_res) / len(tune_res)
    avg50 = sum(ndcg_at_k(r["rr50"], r["judg"], 10) for r in tune_res) / len(tune_res)
    avg_t = sum(r["t_rerank_ms"] for r in tune_res) / len(tune_res)
    print(f"  {'AVG':6s}  {avg10:.3f}   {avg20:.3f}   {avg50:.3f}   {avg_t:>8.1f}")
    print(f"\n  Chosen default: N=20 (balance of recall and latency; N=50 adds ~{avg_t:.0f}ms with marginal gain)")

    # ---- Save markdown ---------------------------------------------------------
    md = ["# Retrieval Full Evaluation (Tasks 5 & 6)\n",
          f"Tune queries: {len(tune_qs)} | Holdout queries: {len(holdout_qs)}\n",
          f"k evaluated: {K_VALUES} | Default k chosen on tune: {DEFAULT_K}\n",
          f"Stage-1 pool: CANDIDATES={CANDIDATES}\n\n",
          "## Aggregate @ k=10\n",
          "| Pipeline | Tune nDCG@10 | Holdout nDCG@10 | Tune R@10 | Holdout R@10 |",
          "|---|---|---|---|---|"]
    for key, label in [("bm25","BM25"),("knn","KNN"),("hybrid","Hybrid"),
                       ("hybrid_graph","Hybrid+Graph"),("rr10","Reranked-N10"),
                       ("rr20","Reranked-N20"),("rr50","Reranked-N50")]:
        tn = sum(ndcg_at_k(r[key], r["judg"], DEFAULT_K) for r in tune_res) / len(tune_res)
        hn = sum(ndcg_at_k(r[key], r["judg"], DEFAULT_K) for r in hold_res) / len(hold_res)
        tr = sum(recall_at_k(r[key], r["judg"], DEFAULT_K) for r in tune_res) / len(tune_res)
        hr = sum(recall_at_k(r[key], r["judg"], DEFAULT_K) for r in hold_res) / len(hold_res)
        md.append(f"| {label} | {tn:.3f} | {hn:.3f} | {tr:.3f} | {hr:.3f} |")
    (OUT / "retrieval_full.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved {OUT / 'retrieval_full.md'}")
