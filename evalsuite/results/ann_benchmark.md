Vectors: 20067 (67 real chunk vectors + 20000 noisy synthetic copies), dims 384, m=16, ef_construction=100, 20 queries, k=10

| Index type | num_candidates | recall@10 vs exact | median latency (ms, server `took`) |
|---|---|---|---|
| hnsw | 20 | 0.360 | 5.1 |
| hnsw | 50 | 0.430 | 5.7 |
| hnsw | 100 | 0.470 | 4.6 |
| int8_hnsw | 20 | 0.425 | 4.2 |
| int8_hnsw | 50 | 0.470 | 3.8 |
| int8_hnsw | 100 | 0.500 | 4.4 |

| Index type | size on disk (MB, 1 segment) | float32 data alone (MB) |
|---|---|---|
| hnsw | 81.2 | 30.8 |
| int8_hnsw | 89.0 | 30.8 |