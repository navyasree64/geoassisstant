"""Task 7 full confidence calibration:
  - Multiple signals: reranker logit, BM25 score (from ES), KNN cosine distance, answer/abstain agreement
  - Logistic regression on fit split (sklearn.linear_model.LogisticRegression used for fitting only;
    ECE, Brier, reliability bins written by hand per Rule 3)
  - Temperature chosen by minimising log-loss on fit split
  - Reliability diagram: 5 equal-frequency bins
  - ECE and Brier score on test split
  - Coverage vs accuracy at 5 thresholds (>= 90% accuracy rule)
  - LLM sampling temperature table: 0 / 0.3 / 0.7 x 3 runs consistency (time-boxed)

Usage:
    python evalsuite/eval_confidence_full.py | Tee-Object evidence/confidence_full.txt
"""
import sys, json, math, time, random, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieve"))

from common import es, OLLAMA_URL
from retrieve import bm25 as bm25_retrieve, knn as knn_retrieve, rrf, graph_expand, fetch_chunks
from reranker import rerank
from graph import KnowledgeGraph, load as graph_load
import requests

EVAL = Path(__file__).resolve().parents[1] / "data_pack" / "eval"
OUT  = Path(__file__).resolve().parents[1] / "evalsuite" / "results"
OUT.mkdir(exist_ok=True)

MODEL = "qwen2.5:3b"

# =========================================================
# Hand-written metric functions (Rule 3 — no sklearn metrics)
# =========================================================
def _sigmoid(z, T=1.0):
    return 1.0 / (1.0 + math.exp(-z / max(T, 1e-6)))

def log_loss_manual(labels, probs):
    """Binary cross-entropy: mean -[y log p + (1-y) log(1-p)]."""
    eps = 1e-12
    return -sum(y * math.log(max(p, eps)) + (1-y) * math.log(max(1-p, eps))
                for y, p in zip(labels, probs)) / len(labels)

def brier_score(labels, probs):
    """Mean squared error between probability and binary label."""
    return sum((p - y)**2 for y, p in zip(labels, probs)) / len(labels)

def reliability_bins(labels, probs, n_bins=5):
    """Equal-frequency binning. Returns list of (mean_conf, fraction_pos, n) per bin."""
    pairs = sorted(zip(probs, labels))
    n = len(pairs)
    size = n // n_bins
    bins = []
    for i in range(n_bins):
        start = i * size
        end = (i + 1) * size if i < n_bins - 1 else n
        chunk = pairs[start:end]
        if not chunk: continue
        mean_conf = sum(p for p, _ in chunk) / len(chunk)
        frac_pos  = sum(y for _, y in chunk) / len(chunk)
        bins.append((mean_conf, frac_pos, len(chunk)))
    return bins

def ece(bins):
    """Expected Calibration Error: weighted mean |mean_conf - frac_pos|."""
    total = sum(b[2] for b in bins)
    return sum(abs(b[0] - b[1]) * b[2] / total for b in bins)


# =========================================================
# Feature extraction for one question
# =========================================================
def get_features(query, G, n_candidates=20):
    """Returns dict with: reranker_logit, bm25_score, knn_score, rrf_score, answerable (None if unknown)."""
    # BM25 + KNN + graph RRF
    b_ids  = bm25_retrieve(query, n_candidates)
    kv_ids = knn_retrieve(query, n_candidates)
    g_ids  = graph_expand(query, G, n_candidates)
    hg_ids = [cid for cid, _ in rrf({"bm25": b_ids, "knn": kv_ids, "graph": g_ids} if g_ids else {"bm25": b_ids, "knn": kv_ids})[:n_candidates]]
    texts  = fetch_chunks(hg_ids)
    cands  = [{"chunk_id": c, "text": texts.get(c, {}).get("text",""), "rrf_score": 0.0} for c in hg_ids]
    ranked = rerank(query, cands, top_n=1)

    reranker_logit = ranked[0]["reranker_score"] if ranked else -15.0
    rrf_score_top  = float(rrf({"bm25": b_ids, "knn": kv_ids})[0][1]) if b_ids else 0.0
    # BM25 raw score from ES
    bm25_body = {"size": 1, "explain": False, "query": {"multi_match": {"query": query, "fields": ["text","circular_number^2"], "type": "best_fields", "analyzer": "english"}}}
    bm25_raw = es("POST", "/east_india_chunks/_search", json=bm25_body).json()["hits"]["hits"]
    bm25_top  = bm25_raw[0]["_score"] if bm25_raw else 0.0
    # KNN cosine distance (1 - cosine_score proxy from ES)
    from embedder import embed_queries
    qv = embed_queries("bge", [query])[0].tolist()
    knn_body = {"size": 1, "_source": [], "knn": {"field": "embedding", "query_vector": qv, "k": 1, "num_candidates": 20}}
    knn_raw = es("POST", "/east_india_chunks/_search", json=knn_body).json()["hits"]["hits"]
    knn_top  = knn_raw[0]["_score"] if knn_raw else 0.0  # cosine similarity (0–1)

    return dict(reranker_logit=reranker_logit, bm25_top=bm25_top, knn_top=knn_top, rrf_top=rrf_score_top,
                top_chunk=ranked[0]["chunk_id"] if ranked else "")


# =========================================================
# LLM sampling temperature consistency test
# =========================================================
LLM_TEST_Q = [
    {"question": "What is the cut-off mark for Hare School?",    "key": "78"},
    {"question": "How many seats does Delhi Public School have?", "key": "160"},
    {"question": "What is the minimum age for Class I in CBSE?", "key": "6"},
]
TEMPS  = [0.0, 0.3, 0.7]
N_RUNS = 3   # time-boxed to 3 (full 5 would take ~25 min on CPU)

def llm_generate(prompt, temperature, seed=42):
    body = {"model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_ctx": 2048, "num_predict": 80}}
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()

def run_llm_temp_table(G):
    rows = []
    for tq in LLM_TEST_Q:
        q_rows = []
        # retrieve top passage once
        b = bm25_retrieve(tq["question"], 20)
        kv = knn_retrieve(tq["question"], 20)
        texts = fetch_chunks(b[:5])
        cands = [{"chunk_id": c, "text": texts.get(c,{}).get("text",""), "rrf_score":0.0} for c in b[:5]]
        ranked = rerank(tq["question"], cands, top_n=1)
        passage = ranked[0]["text"] if ranked else ""
        prompt = (f"Answer in one sentence using only this passage.\n\nPassage: {passage[:400]}\n\n"
                  f"Question: {tq['question']}\nAnswer:")
        for temp in TEMPS:
            answers = []
            for run in range(1, N_RUNS+1):
                ans = llm_generate(prompt, temp, seed=42+run)
                answers.append(ans[:80])
                print(f"    T={temp} run{run}: {ans[:60]}", flush=True)
            unique = len(set(answers))
            key_hits = sum(1 for a in answers if tq["key"].lower() in a.lower())
            q_rows.append(dict(question=tq["question"][:40], temp=temp,
                               n_runs=N_RUNS, n_unique=unique, key_hits=key_hits,
                               answers=answers))
        rows.extend(q_rows)
    return rows


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm-temp", action="store_true",
                    help="Skip the LLM temperature table (saves ~15 min)")
    a = ap.parse_args()

    print("Loading graph…", flush=True)
    G = KnowledgeGraph(*graph_load("es"))

    qs = [json.loads(l) for l in (EVAL/"calibration_questions.jsonl").read_text().splitlines() if l.strip()]
    fit  = [q for q in qs if q["split"] == "fit"]
    test = [q for q in qs if q["split"] == "test"]

    print(f"\nExtracting features for {len(fit)} fit + {len(test)} test questions…")
    fit_feats, test_feats = [], []
    for i, q in enumerate(fit + test, 1):
        which = "fit" if i <= len(fit) else "test"
        print(f"  [{i}/{len(fit)+len(test)}] {q['qid']} ({which})  {q['question'][:55]}", flush=True)
        f = get_features(q["question"], G)
        f["answerable"] = int(q["answerable"])
        f["qid"] = q["qid"]
        (fit_feats if i <= len(fit) else test_feats).append(f)

    # ---- Build logistic regression features (reranker_logit + bm25_top + knn_top) ----
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X_fit  = np.array([[f["reranker_logit"], f["bm25_top"], f["knn_top"]] for f in fit_feats])
    y_fit  = np.array([f["answerable"] for f in fit_feats])
    X_test = np.array([[f["reranker_logit"], f["bm25_top"], f["knn_top"]] for f in test_feats])
    y_test = np.array([f["answerable"] for f in test_feats])

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_fit, y_fit)
    lr_probs_fit  = lr.predict_proba(X_fit)[:, 1].tolist()
    lr_probs_test = lr.predict_proba(X_test)[:, 1].tolist()

    print("\n=== Logistic Regression (3 features: reranker_logit, bm25_top, knn_top) ===")
    print(f"  Coefficients: logit={lr.coef_[0][0]:.3f}  bm25={lr.coef_[0][1]:.3f}  knn={lr.coef_[0][2]:.3f}")
    print(f"  Intercept:    {lr.intercept_[0]:.3f}")
    print(f"  Fit  log-loss: {log_loss_manual(y_fit.tolist(), lr_probs_fit):.4f}")
    print(f"  Test log-loss: {log_loss_manual(y_test.tolist(), lr_probs_test):.4f}")

    # ---- Temperature search (single-feature sigmoid) -------------------------
    logits_fit  = [f["reranker_logit"] for f in fit_feats]
    logits_test = [f["reranker_logit"] for f in test_feats]

    best_T, best_ll = 1.0, 1e9
    print("\nTemperature grid search on fit split (single-feature sigmoid):")
    for T in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0]:
        probs = [_sigmoid(z, T) for z in logits_fit]
        ll = log_loss_manual(y_fit.tolist(), probs)
        print(f"  T={T:.2f}  fit log-loss={ll:.4f}")
        if ll < best_ll:
            best_ll, best_T = ll, T

    print(f"\n  Best T on fit: {best_T}  (log-loss={best_ll:.4f})")
    probs_fit_best  = [_sigmoid(z, best_T) for z in logits_fit]
    probs_test_best = [_sigmoid(z, best_T) for z in logits_test]

    # ---- Reliability bins on TEST split ----------------------------------------
    print("\n=== Reliability table (5 equal-frequency bins, test split) ===")
    print(f"  {'Bin':4s}  {'MeanConf':>9}  {'FracPos':>8}  {'N':>5}  {'|Gap|':>7}")
    rbins = reliability_bins(y_test.tolist(), probs_test_best)
    for i, (mc, fp, n) in enumerate(rbins, 1):
        print(f"  {i:4d}  {mc:>9.3f}  {fp:>8.3f}  {n:>5d}  {abs(mc-fp):>7.3f}")
    ece_val   = ece(rbins)
    brier_val = brier_score(y_test.tolist(), probs_test_best)
    print(f"\n  ECE  (test) = {ece_val:.4f}")
    print(f"  Brier(test) = {brier_val:.4f}")

    # Also for LR probs
    rbins_lr = reliability_bins(y_test.tolist(), lr_probs_test)
    ece_lr   = ece(rbins_lr)
    brier_lr = brier_score(y_test.tolist(), lr_probs_test)
    print(f"\n  LR ECE  (test) = {ece_lr:.4f}")
    print(f"  LR Brier(test) = {brier_lr:.4f}")

    # ---- Coverage vs accuracy at 5 thresholds ---------------------------------
    print("\n=== Coverage and accuracy @ 5 thresholds (test split, single-feature sigmoid) ===")
    print(f"  {'Threshold':>10}  {'Coverage':>9}  {'Accuracy':>9}  {'>= 90% rule':>12}")
    for thr in [0.5, 0.65, 0.75, 0.85, 0.95]:
        covered  = [(y, p) for y, p in zip(y_test.tolist(), probs_test_best) if p >= thr or (1-p) >= thr]
        coverage = len(covered) / len(y_test) if y_test.any() else 0
        if covered:
            # "accuracy" = fraction where pred matches label
            correct = sum(1 for y, p in covered if (p >= thr) == bool(y)) / len(covered)
        else:
            correct = 0.0
        rule = "PASS" if correct >= 0.90 else "FAIL"
        print(f"  {thr:>10.2f}  {coverage:>9.3f}  {correct:>9.3f}  {rule:>12}")

    # ---- LLM temperature table ------------------------------------------------
    if not a.skip_llm_temp:
        print(f"\n=== LLM sampling temperature consistency ({N_RUNS} runs per temp) ===")
        print("  (time-boxed to 3 runs; full 5-run table would take ~25 min on CPU)")
        rows = run_llm_temp_table(G)
        print(f"\n  {'Question':42s}  Temp  UniqueAns  KeyHits/{N_RUNS}")
        for r in rows:
            print(f"  {r['question']:42s}  {r['temp']:.1f}   {r['n_unique']:9d}  {r['key_hits']}/{r['n_runs']}")
    else:
        print("\nSkipped LLM temperature table (--skip-llm-temp).")

    # ---- Save ----------------------------------------------------------------
    summary = {
        "best_T_sigmoid": best_T, "fit_log_loss_sigmoid": round(best_ll,4),
        "ece_test_sigmoid": round(ece_val,4), "brier_test_sigmoid": round(brier_val,4),
        "ece_test_lr": round(ece_lr,4), "brier_test_lr": round(brier_lr,4),
        "lr_coef": lr.coef_[0].tolist(), "lr_intercept": float(lr.intercept_[0])
    }
    (OUT / "confidence_full.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT / 'confidence_full.json'}")
    print("DONE")
