"""Task 4 demonstration: graph measures, traversal (cycles, fan-out, paths), one benefit example, one noise example.
Usage: python retrieve/graph_demo.py            (reads the graph from Elasticsearch)
       python retrieve/graph_demo.py --source files    (reads ingest/out/*.json instead; no plain-search comparison)"""
import sys, argparse
from graph import *

ap = argparse.ArgumentParser(); ap.add_argument("--source", default="es"); a = ap.parse_args()
G = KnowledgeGraph(*load(a.source))
J = {q["query"]: q for q in load_judgments()}

print("=== 1. Graph measures ===")
m = G.measures()
print(f"nodes {m['nodes']} | edges (distinct node pairs) {m['edges']} | typed edges {m['typed_edges']} | connected components {m['components']} (largest {m['largest_component']} nodes)")
print("top-5 degree:", m["top_degree"]); print("top-5 PageRank:", m["top_pagerank"])

hare = next(k for k, e in G.nodes.items() if e["canonical_name"] == "Hare School")
print("\n=== 2. Traversal from Hare School ===")
for h in (1, 2):
    r = G.neighbours(hare, h)
    print(f"\n{h} hop(s): {len(r)} entities reached")
    for k, v in sorted(r.items(), key=lambda x: (x[1]["hops"], -x[1]["score"], x[0]))[:8]:
        print(f"  hop {v['hops']} score {v['score']:.3f}  {G.name(k)}   via  {G.path_str(v['path'])}")
wb = "board:wbbse"
print(f"\nFan-out limit: WBBSE has {G.degree(wb)} neighbours; neighbours(WBBSE, hops=1, fanout=3) returns", len(G.neighbours(wb, 1, fanout=3)))
f, cy = G.neighbours(hare, 2, with_cycles=True)
print(f"Cycle handling: a 2-hop walk from Hare School met {len(cy)} edges that lead back to nodes already reached; they are not re-expanded (visited set), so the walk always terminates.")

def evaluate(ids, rel):
    good = [c for c in ids if rel.get(c, 0) >= 1]
    return len(ids), len(good), sorted(c for c, g in rel.items() if g >= 1 and c not in ids)

def plain_search(q, k=5):
    from embedder import embed_queries
    bm = [h["_source"]["chunk_id"] for h in es("POST", "/east_india_chunks/_search", json={"size": k, "_source": ["chunk_id"], "query": {"match": {"text": q}}}).json()["hits"]["hits"]]
    qv = embed_queries("bge", [q])[0].tolist()
    vec = [h["_source"]["chunk_id"] for h in es("POST", "/east_india_chunks/_search", json={"size": k, "_source": ["chunk_id"], "knn": {"field": "embedding", "query_vector": qv, "k": k, "num_candidates": 50}}).json()["hits"]["hits"]]
    return bm, vec

print("\n=== 3. Benefit: evidence that plain search misses ===")
q = "Which circulars apply to the board of Hare School?"; rel = J[q]["rel"]
det = G.detect_entities(q); print("question:", q); print("entities detected:", det)
ex = G.expand([d[0] for d in det]); ids = [x["chunk_id"] for x in ex]
print(f"judged relevant chunks: {sorted(c for c, g in rel.items() if g >= 1)}")
if a.source == "es":
    bm, vec = plain_search(q); plain = list(dict.fromkeys(bm + vec))
    for label, got in (("BM25 top-5", bm), ("vector top-5", vec), ("plain union", plain)):
        n, good, miss = evaluate(got, rel); print(f"  {label:14s}: {n} chunks, {good} relevant, missing {len(miss)}")
    n, good, miss = evaluate(ids, rel); print(f"  {'graph expand':14s}: {n} chunks, {good} relevant, missing {len(miss)}")
    only = [c for c in ids if rel.get(c, 0) >= 1 and c not in plain]
    print("  relevant chunks found ONLY through the graph:", only)
else:
    n, good, miss = evaluate(ids, rel); print(f"  graph expand: {n} chunks, {good} relevant, missing {miss}")
for x in [x for x in ex if rel.get(x["chunk_id"], 0) >= 1][:3]: print("   example path:", x["chunk_id"], "<-", x["path"])

print("\n=== 4. Noise: expansion adds irrelevant chunks ===")
q = "How many marks are needed to enter Hare School?"; rel = J[q]["rel"]; det = G.detect_entities(q); seeds = [d[0] for d in det]
print("question:", q, "| entities:", det, "| judged relevant:", sorted(c for c, g in rel.items() if g >= 1))
settings = {"loose (2 hops, no limits)": dict(fanout=50, min_weight=0.0, hub_degree=10 ** 6, max_frontier=500, no_sibling=False, hop_decay=1.0),
            "default limits (fanout 20, weight>=0.5, hub skip, sibling block, decay 0.7)": {},
            "tight (1 hop, weight>=0.8)": dict(hops=1, fanout=5, min_weight=0.8)}
for label, kw in settings.items():
    ids = [x["chunk_id"] for x in G.expand(seeds, **kw)]; n, good, _ = evaluate(ids, rel)
    print(f"  {label:78s}: {n:3d} chunks, {good} relevant, precision {good / n if n else 0:.2f}")
top10 = [x["chunk_id"] for x in G.expand(seeds)][:10]
print("  default limits, only the 10 best-scored graph chunks:", evaluate(top10, rel)[:2])
print("  Limits used: min_weight, fan-out cap per node, hub entities contribute no mention chunks, no sibling hops, hop decay, and a top-N cut;\n  in Task 5 graph chunks are only extra candidates - the reranker and the confidence score still decide what is answered.")
