"""Unit tests for hand-written ranking metrics (evalsuite/metrics.py)."""
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evalsuite"))
from metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k

def run_tests():
    print("=== METRIC UNIT TESTS ===")
    ranked = ["a", "b", "c"]
    judg = {"a": 2, "c": 1}

    # 1. Precision@2
    p2 = precision_at_k(ranked, judg, 2)
    print(f"precision_at_k(['a','b','c'], {{'a':2,'c':1}}, 2) = {p2:.4f} (Expected: 0.5000)")
    assert abs(p2 - 0.5) < 1e-9, f"Failed: {p2}"

    # 2. Recall@3
    r3 = recall_at_k(ranked, judg, 3)
    print(f"recall_at_k(['a','b','c'], {{'a':2,'c':1}}, 3) = {r3:.4f} (Expected: 1.0000)")
    assert abs(r3 - 1.0) < 1e-9, f"Failed: {r3}"

    # 3. MRR
    m = mrr(["b", "a"], {"a": 2})
    print(f"mrr(['b','a'], {{'a':2}}) = {m:.4f} (Expected: 0.5000)")
    assert abs(m - 0.5) < 1e-9, f"Failed: {m}"

    # 4. nDCG@3
    ndcg = ndcg_at_k(ranked, judg, 3)
    print(f"ndcg_at_k(['a','b','c'], {{'a':2,'c':1}}, 3) = {ndcg:.4f} (Expected: 0.9502)")
    assert abs(ndcg - 0.9502) < 1e-3, f"Failed: {ndcg}"

    print("\nALL METRIC UNIT TESTS PASSED.")

if __name__ == "__main__":
    run_tests()
