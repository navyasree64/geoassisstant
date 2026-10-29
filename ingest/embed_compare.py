"""Compare the two embedding models: dimension, speed (chunks/s) and retrieval quality on the first 10 queries.
Also saves the chunk embeddings so later steps do not recompute them.   Usage: python ingest/embed_compare.py"""
import sys, time, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evalsuite"))
from common import *
from chunker import build_chunks
from embedder import *
from metrics import recall_at_k, mrr, ndcg_at_k, precision_at_k

chunks = build_chunks(); ids = [c["chunk_id"] for c in chunks]; texts = [c["text"] for c in chunks]
base = lambda cid: cid.split("#")[0]
qs = load_judgments()[:10]
rows = []
for key in MODELS:
    embed_passages(key, texts[:8])                                  # warm-up (model load, first-batch overhead)
    t = time.perf_counter(); P = embed_passages(key, texts); dt = time.perf_counter() - t
    Q = embed_queries(key, [q["query"] for q in qs])
    S = Q @ P.T
    r5, p5, rr, nd = [], [], [], []
    for qi, q in enumerate(qs):
        order = np.argsort(-S[qi], kind="stable")
        ranked = []
        for i in order:                                              # collapse window-chunks to their paragraph id
            b = base(ids[i])
            if b not in ranked: ranked.append(b)
        r5.append(recall_at_k(ranked, q["rel"], 5)); p5.append(precision_at_k(ranked, q["rel"], 5))
        rr.append(mrr(ranked, q["rel"])); nd.append(ndcg_at_k(ranked, q["rel"], 5))
    rows.append((key, MODELS[key]["name"], P.shape[1], len(texts) / dt, np.mean(r5), np.mean(p5), np.mean(rr), np.mean(nd)))
    np.save(OUT / f"emb_{key}.npy", P)
json.dump(ids, open(OUT / "chunk_ids.json", "w"))
md = ["| Model | Dims | Chunks/s | Query prefix | Recall@5 | Precision@5 | MRR | nDCG@5 |", "|---|---|---|---|---|---|---|---|"]
for key, name, d, cps, r5, p5, rr, nd in rows:
    md.append(f"| {name} | {d} | {cps:.1f} | {'yes' if MODELS[key]['query_prefix'] else 'no'} | {r5:.3f} | {p5:.3f} | {rr:.3f} | {nd:.3f} |")
txt = f"Chunks embedded: {len(texts)}; queries: first 10 of retrieval_judgments.jsonl (relevant = grade>=1)\n\n" + "\n".join(md)
print(txt); (ROOT / "evalsuite" / "results").mkdir(exist_ok=True)
(ROOT / "evalsuite" / "results" / "embedding_comparison.md").write_text(txt, encoding="utf-8")
