"""Approximate (HNSW) vs exact (NumPy brute force) kNN, float vs int8-quantised, at several num_candidates.
Uses TEMPORARY benchmark indices created by the elastic admin user (an experiment, not the application) and deletes them.
The real chunk vectors are padded with SCALE noisy copies so that HNSW has something to approximate (disclose this!).
Usage: python evalsuite/ann_benchmark.py --model bge --scale 20000"""
import sys, time, json, argparse, statistics, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from common import *
from embedder import embed_queries

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="bge"); ap.add_argument("--scale", type=int, default=20000)
ap.add_argument("--m", type=int, default=16); ap.add_argument("--efc", type=int, default=100); a = ap.parse_args()
P = np.load(OUT / f"emb_{a.model}.npy"); d = P.shape[1]; rng = np.random.default_rng(42)
src = rng.integers(0, len(P), a.scale); X = P[src] + rng.normal(0, 0.4 / np.sqrt(d), (a.scale, d)).astype("float32")
X /= np.linalg.norm(X, axis=1, keepdims=True); V = np.vstack([P, X]).astype("float32"); N = len(V)
qs = [q["query"] for q in load_judgments()[:20]]; Q = embed_queries(a.model, qs)
exact = np.argsort(-(Q @ V.T), axis=1, kind="stable")[:, :10]                    # my own brute-force top-10
rows = []; sizes = {}
for kind in ("hnsw", "int8_hnsw"):
    idx = f"bench_{kind}"
    es("DELETE", f"/{idx}", admin=True, expect=None)
    es("PUT", f"/{idx}", admin=True, json={"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {
        "embedding": {"type": "dense_vector", "dims": d, "index": True, "similarity": "cosine",
                      "index_options": {"type": kind, "m": a.m, "ef_construction": a.efc}}}}})
    t0 = time.perf_counter()
    for s in range(0, N, 2000):
        lines = []
        for i in range(s, min(N, s + 2000)):
            lines += [json.dumps({"index": {"_index": idx, "_id": str(i)}}), json.dumps({"embedding": [round(float(x), 6) for x in V[i]]})]
        es("POST", "/_bulk", admin=True, data="\n".join(lines) + "\n", headers={"Content-Type": "application/x-ndjson"})
    es("POST", f"/{idx}/_refresh", admin=True); es("POST", f"/{idx}/_forcemerge?max_num_segments=1", admin=True)
    build = time.perf_counter() - t0
    sizes[kind] = es("GET", f"/{idx}/_stats/store", admin=True).json()["_all"]["primaries"]["store"]["size_in_bytes"]
    for nc in (20, 50, 100):
        rec, lat = [], []
        for qi in range(len(Q)):
            body = {"size": 10, "_source": False, "knn": {"field": "embedding", "query_vector": Q[qi].tolist(), "k": 10, "num_candidates": nc}}
            ts = []
            for rep in range(4):
                r = es("POST", f"/{idx}/_search", admin=True, json=body).json(); ts.append(r["took"])
            got = {int(h["_id"]) for h in r["hits"]["hits"]}
            rec.append(len(got & set(exact[qi].tolist())) / 10); lat.append(statistics.median(ts[1:]))
        rows.append((kind, nc, np.mean(rec), statistics.mean(lat)))
    print(f"{kind}: build+merge {build:.0f}s, size {sizes[kind] / 1e6:.1f} MB")
    es("DELETE", f"/{idx}", admin=True, expect=None)
md = [f"Vectors: {N} ({len(P)} real chunk vectors + {a.scale} noisy synthetic copies), dims {d}, m={a.m}, ef_construction={a.efc}, 20 queries, k=10", "",
      "| Index type | num_candidates | recall@10 vs exact | median latency (ms, server `took`) |", "|---|---|---|---|"]
md += [f"| {k} | {nc} | {r:.3f} | {l:.1f} |" for k, nc, r, l in rows]
md += ["", "| Index type | size on disk (MB, 1 segment) | float32 data alone (MB) |", "|---|---|---|"]
md += [f"| {k} | {sizes[k] / 1e6:.1f} | {N * d * 4 / 1e6:.1f} |" for k in sizes]
txt = "\n".join(md); print("\n" + txt); (ROOT / "evalsuite" / "results").mkdir(exist_ok=True)
(ROOT / "evalsuite" / "results" / "ann_benchmark.md").write_text(txt, encoding="utf-8")
