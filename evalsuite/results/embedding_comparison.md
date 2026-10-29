Chunks embedded: 67; queries: first 10 of retrieval_judgments.jsonl (relevant = grade>=1)

| Model | Dims | Chunks/s | Query prefix | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 384 | 57.3 | yes | 0.667 | 0.240 | 0.737 | 0.646 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 111.2 | no | 0.650 | 0.220 | 0.762 | 0.628 |