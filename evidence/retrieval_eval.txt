# Task 5 Retrieval Evaluation Report

Evaluated on 30 queries (20 tune + 10 holdout) across k in [1, 3, 5, 10, 20].

## 1. Metrics by Retrieval Mode and k (All 30 Queries)

| Mode | k | Precision@k | Recall@k | MRR@k | nDCG@k |
|---|---|---|---|---|---|
| keyword      |  1 | 0.667 | 0.400 | 0.667 | 0.600 |
| keyword      |  3 | 0.411 | 0.693 | 0.794 | 0.670 |
| keyword      |  5 | 0.273 | 0.741 | 0.803 | 0.701 |
| keyword      | 10 | 0.157 | 0.809 | 0.807 | 0.725 |
| keyword      | 20 | 0.095 | 0.907 | 0.807 | 0.754 |
| vector       |  1 | 0.633 | 0.428 | 0.633 | 0.567 |
| vector       |  3 | 0.356 | 0.620 | 0.728 | 0.615 |
| vector       |  5 | 0.240 | 0.676 | 0.741 | 0.638 |
| vector       | 10 | 0.143 | 0.761 | 0.747 | 0.675 |
| vector       | 20 | 0.090 | 0.909 | 0.752 | 0.716 |
| hybrid       |  1 | 0.667 | 0.417 | 0.667 | 0.617 |
| hybrid       |  3 | 0.389 | 0.665 | 0.772 | 0.648 |
| hybrid       |  5 | 0.280 | 0.757 | 0.787 | 0.700 |
| hybrid       | 10 | 0.157 | 0.815 | 0.791 | 0.726 |
| hybrid       | 20 | 0.095 | 0.928 | 0.791 | 0.755 |
| graph_hybrid |  1 | 0.733 | 0.444 | 0.733 | 0.667 |
| graph_hybrid |  3 | 0.433 | 0.698 | 0.811 | 0.701 |
| graph_hybrid |  5 | 0.293 | 0.787 | 0.819 | 0.738 |
| graph_hybrid | 10 | 0.173 | 0.878 | 0.823 | 0.774 |
| graph_hybrid | 20 | 0.100 | 0.967 | 0.826 | 0.801 |

### Choice of Default k = 5

- At **k = 5**, Graph+Hybrid achieves high precision (0.293) and recall (0.787) with an nDCG@5 of 0.738.
- Expanding to k = 10 increases recall marginally (0.878 vs 0.787) but decreases precision (0.173 vs 0.293), adding extra irrelevant context to downstream generation.
- Therefore, **default k = 5** is chosen as the optimal balance for the downstream generator.

## 2. Hyperparameter Tuning: RRF Constant C (Tune vs Holdout)

| RRF Constant C | 20 Tune nDCG@5 | 10 Holdout nDCG@5 | Status |
|---|---|---|---|
| C = 10 | 0.7586 | 0.6780 | Best Tune |
| C = 30 | 0.7584 | — |  |
| C = 60 | 0.7584 | 0.6874 | Default |
| C = 100 | 0.7584 | — |  |

Conclusion: Tuning to C = 10 did not yield a holdout gain over default C = 60 (0.6780 vs 0.6874). Default C = 60 is retained.

## 3. Failure Analysis Grouped by Query Type (at k=5)

| Query Type | N Queries | Best Performing Mode | Winning nDCG@5 | Why this mode wins |
|---|---|---|---|---|
| exact_identifier | 8 | keyword | 0.790 | Circular numbers and code IDs strongly match sparse inverted indices (BM25 field boost). |
| paraphrase | 10 | vector | 0.779 | Dense embedding captures semantic equivalence across varied phrasing. |
| multi_hop | 6 | graph_hybrid | 0.759 | Graph traversal bridges unlinked entities across 2 hops (e.g. School -> Board -> Circular). |
| list_table | 6 | graph_hybrid | 0.668 | Keyword / hybrid fusion recovers fragmented rows and numeric seat matrices. |

### Queries Missing Relevant Chunks at k = 5 (Graph+Hybrid)

| QID | Type | Query | Missed Relevant Chunks |
|---|---|---|---|
| Q01 | exact_identifier | Circular DEO/ADM/2026/004 | CIR-004-rev |
| Q02 | exact_identifier | DEO/ADM/2026/011 | CIR-011-supers, CIR-011-rule5 |
| Q07 | exact_identifier | DEO/ADM/2026/012 | CIR-012-table |
| Q08 | exact_identifier | Circular DEO/ADM/2026/009 | CIR-009-head, CIR-009-intro |
| Q09 | paraphrase | How old must a child be for Class V in WBBSE schools? | CIR-004-rev |
| Q11 | paraphrase | Until when can parents apply for fifth grade admission in We | CIR-004-rev |
| Q13 | paraphrase | What income limit is set for EWS applicants? | CIR-011-rule5 |
| Q19 | multi_hop | Which circulars apply to the board of Hare School? | CIR-001-sched, CIR-010-intro, CIR-005-rule4, CIR-004-rev, CIR-011-supers, CIR-009-intro, GUIDE-001-applies, CIR-008-applies |
| Q21 | multi_hop | Which circular supersedes DEO/ADM/2026/002 and what are the  | CIR-002-table, CIR-012-table |
| Q29 | list_table | Which CBSE schools are in the revised seat matrix? | CIR-002-table, CIR-012-table |