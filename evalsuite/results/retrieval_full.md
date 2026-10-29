# Retrieval Full Evaluation (Tasks 5 & 6)

Tune queries: 20 | Holdout queries: 10

k evaluated: [1, 3, 5, 10, 20] | Default k chosen on tune: 10

Stage-1 pool: CANDIDATES=50


## Aggregate @ k=10

| Pipeline | Tune nDCG@10 | Holdout nDCG@10 | Tune R@10 | Holdout R@10 |
|---|---|---|---|---|
| BM25 | 0.738 | 0.700 | 0.817 | 0.794 |
| KNN | 0.736 | 0.552 | 0.775 | 0.733 |
| Hybrid | 0.735 | 0.708 | 0.800 | 0.844 |
| Hybrid+Graph | 0.797 | 0.746 | 0.883 | 0.867 |
| Reranked-N10 | 0.811 | 0.673 | 0.883 | 0.867 |
| Reranked-N20 | 0.753 | 0.661 | 0.775 | 0.817 |
| Reranked-N50 | 0.753 | 0.638 | 0.775 | 0.767 |