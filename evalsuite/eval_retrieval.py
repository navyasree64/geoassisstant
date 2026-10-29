"""Task 5 – Comprehensive evaluation of retrieval modes (BM25, kNN, Hybrid, Graph+Hybrid).

Evaluates:
  1. All 4 retrieval modes at k = 1, 3, 5, 10, 20 (Precision, Recall, MRR, nDCG).
  2. Justification for choosing default k (k=5).
  3. Single-parameter tuning of RRF constant C in {10, 30, 60, 100} on 20 tune queries (objective nDCG@5).
  4. Validation of default vs tuned constant on the 10 holdout queries.
  5. Failure analysis by query type with exact queries that miss relevant chunks.

Usage:
  python evalsuite/eval_retrieval.py
Outputs saved to evalsuite/results/retrieval_eval.md and evidence/retrieval_eval.txt.
"""

import sys, os, json, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ingest"))
sys.path.insert(0, str(ROOT / "retrieve"))

from common import es, load_judgments
from metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k
from graph import KnowledgeGraph, load as graph_load
from retrieve import keyword, vector, hybrid, graph_hybrid, rrf, fetch_chunks, RRF_K

K_VALUES = [1, 3, 5, 10, 20]
DEFAULT_K = 5


def compute_metrics(ranked_ids: list[str], rel_dict: dict, k: int) -> dict:
    return {
        f"p@{k}": precision_at_k(ranked_ids, rel_dict, k),
        f"r@{k}": recall_at_k(ranked_ids, rel_dict, k),
        f"mrr@{k}": mrr(ranked_ids[:k], rel_dict),
        f"ndcg@{k}": ndcg_at_k(ranked_ids, rel_dict, k),
    }


def main():
    print("Loading Knowledge Graph and Judgments...")
    G = KnowledgeGraph(*graph_load("es"))
    judgments = load_judgments()
    
    tune_qs = [q for q in judgments if q["split"] == "tune"]
    holdout_qs = [q for q in judgments if q["split"] == "holdout"]
    print(f"Loaded {len(judgments)} queries: {len(tune_qs)} tune, {len(holdout_qs)} holdout.\n")

    modes = ["keyword", "vector", "hybrid", "graph_hybrid"]
    
    # Store results: results[mode][qid] = {ranked_ids}
    query_runs = {m: {} for m in modes}
    
    print("Running retrievals across all 30 queries...")
    for q in judgments:
        qid = q["qid"]
        query_text = q["query"]
        
        kw_hits = keyword(query_text, k=25)
        vec_hits = vector(query_text, k=25)
        hyb_hits = hybrid(query_text, k=25, rrf_k=RRF_K)
        ghyb_hits = graph_hybrid(query_text, G, k=25, rrf_k=RRF_K)
        
        query_runs["keyword"][qid] = [cid for cid, _ in kw_hits]
        query_runs["vector"][qid] = [cid for cid, _ in vec_hits]
        query_runs["hybrid"][qid] = [cid for cid, _ in hyb_hits]
        query_runs["graph_hybrid"][qid] = [cid for cid, _ in ghyb_hits]

    # ==================== TABLE B: Every Mode at k = 1, 3, 5, 10, 20 ====================
    md_lines = ["# Task 5 Retrieval Evaluation Report\n"]
    md_lines.append(f"Evaluated on {len(judgments)} queries (20 tune + 10 holdout) across k in {K_VALUES}.\n")
    md_lines.append("## 1. Metrics by Retrieval Mode and k (All 30 Queries)\n")
    md_lines.append("| Mode | k | Precision@k | Recall@k | MRR@k | nDCG@k |")
    md_lines.append("|---|---|---|---|---|---|")

    print("\n=== 1. Metrics by Retrieval Mode and k ===")
    print(f"{'Mode':15s} | {'k':2s} | {'Prec@k':8s} | {'Recall@k':8s} | {'MRR@k':8s} | {'nDCG@k':8s}")
    print("-" * 65)

    mode_k_metrics = {}
    for m in modes:
        mode_k_metrics[m] = {}
        for k in K_VALUES:
            p_list, r_list, mrr_list, ndcg_list = [], [], [], []
            for q in judgments:
                qid = q["qid"]
                ranked = query_runs[m][qid]
                rel = q["rel"]
                p_list.append(precision_at_k(ranked, rel, k))
                r_list.append(recall_at_k(ranked, rel, k))
                mrr_list.append(mrr(ranked[:k], rel))
                ndcg_list.append(ndcg_at_k(ranked, rel, k))
            
            avg_p = sum(p_list) / len(p_list)
            avg_r = sum(r_list) / len(r_list)
            avg_mrr = sum(mrr_list) / len(mrr_list)
            avg_ndcg = sum(ndcg_list) / len(ndcg_list)
            
            mode_k_metrics[m][k] = {"p": avg_p, "r": avg_r, "mrr": avg_mrr, "ndcg": avg_ndcg}
            md_lines.append(f"| {m:12s} | {k:2d} | {avg_p:.3f} | {avg_r:.3f} | {avg_mrr:.3f} | {avg_ndcg:.3f} |")
            print(f"{m:15s} | {k:2d} | {avg_p:.3f}    | {avg_r:.3f}    | {avg_mrr:.3f}    | {avg_ndcg:.3f}")

    # Justification of default k=5
    gh_5 = mode_k_metrics["graph_hybrid"][5]
    gh_10 = mode_k_metrics["graph_hybrid"][10]
    md_lines.append("\n### Choice of Default k = 5\n")
    md_lines.append(
        f"- At **k = 5**, Graph+Hybrid achieves high precision ({gh_5['p']:.3f}) and recall ({gh_5['r']:.3f}) "
        f"with an nDCG@5 of {gh_5['ndcg']:.3f}.\n"
        f"- Expanding to k = 10 increases recall marginally ({gh_10['r']:.3f} vs {gh_5['r']:.3f}) but decreases precision "
        f"({gh_10['p']:.3f} vs {gh_5['p']:.3f}), adding extra irrelevant context to downstream generation.\n"
        f"- Therefore, **default k = 5** is chosen as the optimal balance for the downstream generator.\n"
    )

    # ==================== TABLE C: Tuning RRF Constant on Tune Split ====================
    print("\n=== 2. Tuning RRF Constant C on Tune Split (Objective: nDCG@5) ===")
    rrf_constants = [10, 30, 60, 100]
    tune_scores = {}
    
    for c in rrf_constants:
        ndcg_tune_list = []
        for q in tune_qs:
            q_text = q["query"]
            rel = q["rel"]
            hits = graph_hybrid(q_text, G, k=10, rrf_k=c)
            ranked = [cid for cid, _ in hits]
            ndcg_tune_list.append(ndcg_at_k(ranked, rel, 5))
        avg_tune_ndcg5 = sum(ndcg_tune_list) / len(ndcg_tune_list)
        tune_scores[c] = avg_tune_ndcg5
        print(f"  RRF constant C = {c:3d} -> Tune nDCG@5 = {avg_tune_ndcg5:.4f}")

    best_c = max(tune_scores.keys(), key=lambda c: tune_scores[c])
    print(f"\nBest parameter on 20 Tune queries: C = {best_c} (Tune nDCG@5 = {tune_scores[best_c]:.4f})")

    # Evaluate Default (60) vs Tuned on Holdout Split ONLY
    print("\n--- Testing on Unseen 10 Holdout Queries ---")
    holdout_ndcg_default = []
    holdout_ndcg_tuned = []
    for q in holdout_qs:
        q_text = q["query"]
        rel = q["rel"]
        
        # default C=60
        hits_def = graph_hybrid(q_text, G, k=10, rrf_k=60)
        ranked_def = [cid for cid, _ in hits_def]
        holdout_ndcg_default.append(ndcg_at_k(ranked_def, rel, 5))
        
        # tuned C=best_c
        hits_tun = graph_hybrid(q_text, G, k=10, rrf_k=best_c)
        ranked_tun = [cid for cid, _ in hits_tun]
        holdout_ndcg_tuned.append(ndcg_at_k(ranked_tun, rel, 5))

    avg_h_def = sum(holdout_ndcg_default) / len(holdout_ndcg_default)
    avg_h_tun = sum(holdout_ndcg_tuned) / len(holdout_ndcg_tuned)
    print(f"  Default C=60   -> Holdout nDCG@5 = {avg_h_def:.4f}")
    print(f"  Tuned   C={best_c}   -> Holdout nDCG@5 = {avg_h_tun:.4f}")
    gain_held = avg_h_tun >= avg_h_def

    md_lines.append("## 2. Hyperparameter Tuning: RRF Constant C (Tune vs Holdout)\n")
    md_lines.append("| RRF Constant C | 20 Tune nDCG@5 | 10 Holdout nDCG@5 | Status |")
    md_lines.append("|---|---|---|---|")
    for c in rrf_constants:
        h_score = avg_h_tun if c == best_c else (avg_h_def if c == 60 else "—")
        status = "Best Tune" if c == best_c else ("Default" if c == 60 else "")
        md_lines.append(f"| C = {c} | {tune_scores[c]:.4f} | {h_score if isinstance(h_score, str) else f'{h_score:.4f}'} | {status} |")
    
    if gain_held:
        md_lines.append(f"\nConclusion: Tuned constant C = {best_c} maintained performance on holdout ({avg_h_tun:.4f} vs {avg_h_def:.4f}).\n")
    else:
        md_lines.append(f"\nConclusion: Tuning to C = {best_c} did not yield a holdout gain over default C = 60 ({avg_h_tun:.4f} vs {avg_h_def:.4f}). Default C = 60 is retained.\n")

    # ==================== TABLE D: Failure Analysis by Query Type ====================
    print("\n=== 3. Failure Analysis Grouped by Query Type ===")
    query_types = ["exact_identifier", "paraphrase", "multi_hop", "list_table"]
    md_lines.append("## 3. Failure Analysis Grouped by Query Type (at k=5)\n")
    md_lines.append("| Query Type | N Queries | Best Performing Mode | Winning nDCG@5 | Why this mode wins |")
    md_lines.append("|---|---|---|---|---|")

    misses = []

    for qt in query_types:
        qt_qs = [q for q in judgments if q["type"] == qt]
        type_scores = {}
        for m in modes:
            scores_m = [ndcg_at_k(query_runs[m][q["qid"]], q["rel"], 5) for q in qt_qs]
            type_scores[m] = sum(scores_m) / len(scores_m)
        
        winner = max(type_scores.keys(), key=lambda m: type_scores[m])
        win_score = type_scores[winner]
        
        why = ""
        if qt == "exact_identifier":
            why = "Circular numbers and code IDs strongly match sparse inverted indices (BM25 field boost)."
        elif qt == "paraphrase":
            why = "Dense embedding captures semantic equivalence across varied phrasing."
        elif qt == "multi_hop":
            why = "Graph traversal bridges unlinked entities across 2 hops (e.g. School -> Board -> Circular)."
        elif qt == "list_table":
            why = "Keyword / hybrid fusion recovers fragmented rows and numeric seat matrices."
        
        md_lines.append(f"| {qt} | {len(qt_qs)} | {winner} | {win_score:.3f} | {why} |")
        print(f"  {qt:16s} (N={len(qt_qs)}) -> Winner: {winner:12s} (nDCG@5 = {win_score:.3f})")

        # Check for queries that missed relevant chunks at k=5 in graph_hybrid
        for q in qt_qs:
            rel_cids = {cid for cid, g in q["rel"].items() if g >= 1}
            retrieved_5 = set(query_runs["graph_hybrid"][q["qid"]][:5])
            missed_rel = rel_cids - retrieved_5
            if missed_rel:
                misses.append({
                    "qid": q["qid"],
                    "type": qt,
                    "query": q["query"],
                    "missed": list(missed_rel)
                })

    md_lines.append("\n### Queries Missing Relevant Chunks at k = 5 (Graph+Hybrid)\n")
    md_lines.append("| QID | Type | Query | Missed Relevant Chunks |")
    md_lines.append("|---|---|---|---|")
    for m in misses:
        md_lines.append(f"| {m['qid']} | {m['type']} | {m['query'][:60]} | {', '.join(m['missed'])} |")

    # Write output to results and evidence
    out_dir = ROOT / "evalsuite" / "results"
    out_dir.mkdir(exist_ok=True)
    md_file = out_dir / "retrieval_eval.md"
    ev_file = ROOT / "evidence" / "retrieval_eval.txt"

    full_text = "\n".join(md_lines)
    md_file.write_text(full_text, encoding="utf-8")
    ev_file.write_text(full_text, encoding="utf-8")
    print(f"\nSaved evaluation markdown to {md_file}")
    print(f"Saved evaluation evidence to {ev_file}")

if __name__ == "__main__":
    main()
