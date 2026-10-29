"""Task 1 – Verification of local embedding models and cross-encoder on CPU.
Usage:
    python ingest/check_models.py
"""
import time
from sentence_transformers import SentenceTransformer, CrossEncoder

texts = [
    "Admission to Class VI requires a minimum age of 10 years.",
    "CBSE schools must submit seat details before 30 June.",
    "The school is located near the main road in Midnapore.",
]
query = "What is the minimum age for admission?"

embedders = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
}

for name, prefix in embedders.items():
    m = SentenceTransformer(name, device="cpu")
    t = time.time()
    vecs = m.encode(texts, normalize_embeddings=True)
    dt = time.time() - t
    qv = m.encode(prefix + query, normalize_embeddings=True)
    print(f"{name}: dims={vecs.shape[1]}, max_seq_len={m.max_seq_length}, "
          f"{len(texts)/dt:.1f} chunks/s, query_prefix={'yes' if prefix else 'no'}")
    print("   cosine(query, texts):", [round(float(qv @ v), 3) for v in vecs])

rr = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
scores = rr.predict([(query, t) for t in texts])
print("reranker: score type = raw logit (one float per pair, higher = more relevant)")
print("   scores:", [round(float(s), 3) for s in scores])