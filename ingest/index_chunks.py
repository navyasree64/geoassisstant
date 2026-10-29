"""Create east_india_chunks (explicit mapping, HNSW) and index all chunks with deterministic ids (idempotent).
Usage: python ingest/index_chunks.py --model bge --index-type hnsw          (or int8_hnsw)
The app role rag_app_user creates and fills the index. Deleting/recreating needs the elastic user (see steps)."""
import argparse, json, numpy as np
from common import *
from chunker import build_chunks
from embedder import *

def mapping(dims, index_type, m, efc):
    return {"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"dynamic": "strict", "properties": {
        "chunk_id": {"type": "keyword"}, "doc_id": {"type": "keyword"}, "doc_type": {"type": "keyword"}, "file": {"type": "keyword"},
        "page": {"type": "integer"}, "para_index": {"type": "integer"}, "start": {"type": "integer"}, "end": {"type": "integer"},
        "char_count": {"type": "integer"}, "issue_date": {"type": "date", "format": "yyyy-MM-dd"},
        "circular_number": {"type": "keyword"}, "boards": {"type": "keyword"}, "school_ids": {"type": "keyword"},
        "school_locations": {"type": "geo_point"},
        "text": {"type": "text", "analyzer": "english"},
        "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine",
                      "index_options": {"type": index_type, "m": m, "ef_construction": efc}}}}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="bge"); ap.add_argument("--index-type", default="hnsw")
    ap.add_argument("--m", type=int, default=16); ap.add_argument("--ef-construction", type=int, default=100)
    a = ap.parse_args()
    chunks = build_chunks(); ids = [c["chunk_id"] for c in chunks]
    f = OUT / f"emb_{a.model}.npy"
    P = np.load(f) if f.exists() and json.load(open(OUT / "chunk_ids.json")) == ids else embed_passages(a.model, [c["text"] for c in chunks])
    mp = mapping(P.shape[1], a.index_type, a.m, a.ef_construction)
    (ROOT / "es").mkdir(exist_ok=True); json.dump(mp, open(ROOT / "es" / "chunks_mapping.json", "w"), indent=2)
    r = es("HEAD", "/east_india_chunks", expect=None)
    if r.status_code == 404: es("PUT", "/east_india_chunks", json=mp); print("created index east_india_chunks:", a.index_type, f"m={a.m} ef_construction={a.ef_construction} dims={P.shape[1]}")
    elif r.status_code == 200:
        cur = es("GET", "/east_india_chunks/_mapping").json()["east_india_chunks"]["mappings"].get("properties", {})
        if "embedding" not in cur: raise SystemExit("index exists with a default mapping: delete it as elastic first (see steps)")
    else: raise SystemExit(f"cannot access index: HTTP {r.status_code}")
    lines = []
    for c, v in zip(chunks, P):
        lines.append(json.dumps({"index": {"_index": "east_india_chunks", "_id": c["chunk_id"]}}))     # deterministic _id
        lines.append(json.dumps({**c, "embedding": [round(float(x), 6) for x in v]}))
    res = es("POST", "/_bulk?refresh=true", data="\n".join(lines) + "\n", headers={"Content-Type": "application/x-ndjson"}).json()
    bad = [list(i.values())[0] for i in res["items"] if "error" in list(i.values())[0]]
    print("bulk items:", len(res["items"]), "| failed:", len(bad)); bad and print(bad[:2])
    print("documents in east_india_chunks:", es("GET", "/east_india_chunks/_count").json()["count"])
