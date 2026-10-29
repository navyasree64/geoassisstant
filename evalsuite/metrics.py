"""My own ranking metrics. `ranked` = list of chunk ids (best first); `judg` = {chunk_id: grade 0/1/2}.
A chunk counts as relevant when grade >= min_grade (default 1). nDCG uses the grade itself as the gain
(linear gain; the 2^grade-1 variant is also common).

Usage:
    from metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k
    p5 = precision_at_k(["c1", "c2"], {"c1": 2, "c2": 0}, k=2)
"""
import math

def precision_at_k(ranked, judg, k, min_grade=1):
    return sum(1 for c in ranked[:k] if judg.get(c, 0) >= min_grade) / k

def recall_at_k(ranked, judg, k, min_grade=1):
    rel = {c for c, g in judg.items() if g >= min_grade}
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0

def mrr(ranked, judg, min_grade=1):
    for i, c in enumerate(ranked, 1):
        if judg.get(c, 0) >= min_grade: return 1.0 / i
    return 0.0

def dcg(grades, k):
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades[:k]))

def ndcg_at_k(ranked, judg, k):
    ideal = dcg(sorted(judg.values(), reverse=True), k)
    return dcg([judg.get(c, 0) for c in ranked], k) / ideal if ideal > 0 else 0.0
