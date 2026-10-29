# Measured Pipeline Latency: Cold-Start vs Warm Query

Evaluation Query: `What is the cut-off mark for Hare School?`
Environment: Local Intel CPU, Windows 11, Ollama `qwen2.5:3b`, Elasticsearch 8.15.0, GeoServer 2.26.1

## Per-Stage Measured Latency Breakdown

| Pipeline Stage | Cold-Start (First Query Execution) | Warm Query (Cached Models in RAM) | Speedup Factor |
|---|---|---|---|
| **Python Imports & Initial Setup** | 423.7 ms | 0.0 ms (resident) | ∞ |
| **Knowledge Graph Loading** | 48.9 ms | 0.0 ms (cached) | ∞ |
| **Safety Gate** | 81.7 ms | 5.3 ms | 15.4x |
| **GeoServer WFS Check** | 0.1 ms | 0.6 ms | 1.0x |
| **Hybrid Graph Retrieval (BM25 + kNN)** | 22888.0 ms | 153.5 ms | 149.1x |
| **Cross-Encoder Rerank (ms-marco-MiniLM)** | 6557.7 ms | 1380.8 ms | 4.7x |
| **Ollama LLM Generation** | 5760.0 ms | 5390.0 ms | 1.1x |
| **Total Query Latency** | **35328.6 ms (35.33 s)** | **6976.6 ms (6.98 s)** | **5.1x** |

## Query Outputs Verification
- **Cold Answer**: `Hare School's cut-off is 78 marks.` (Confidence: `0.9678`, Verification: `Verified`, Citations: `['CIR-006-table']`)
- **Warm Answer**: `Hare School's cut-off is 78 marks.` (Confidence: `0.9678`, Verification: `Verified`, Citations: `['CIR-006-table']`)
