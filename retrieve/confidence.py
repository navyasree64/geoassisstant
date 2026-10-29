"""Task 7 – Confidence scoring and abstention gate.

confidence_score(reranker_logit, temperature=1.0):
    Applies sigmoid to the raw cross-encoder logit to produce a probability in (0, 1).
    Temperature > 1 flattens the distribution (more uncertain);
    Temperature < 1 sharpens it (more decisive).
    sigmoid(z) = 1 / (1 + exp(-z/T))

should_answer(reranker_logit, threshold=0.5, temperature=1.0):
    Returns True if confidence > threshold. The system abstains otherwise.

calibrate(questions_jsonl, threshold_range, temperature_range):
    Grid-search the threshold and temperature on calibration_questions.jsonl (fit split)
    that maximises balanced accuracy (TP rate + TN rate) / 2 on answerable vs unanswerable.

Usage:
    python retrieve/confidence.py                         # runs calibration on fit split + demo
    python retrieve/confidence.py --demo "question here"  # single question demo
"""
import sys, json, math, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

EVAL = Path(__file__).resolve().parents[1] / "data_pack" / "eval"


# ---------- core math ---------------------------------------------------------
def sigmoid(z: float, temperature: float = 1.0) -> float:
    """Sigmoid with temperature scaling: σ(z/T) = 1 / (1 + e^{-z/T})."""
    return 1.0 / (1.0 + math.exp(-z / max(temperature, 1e-6)))


def confidence_score(reranker_logit: float, temperature: float = 1.0) -> float:
    """Convert a raw cross-encoder logit to a calibrated confidence probability."""
    return round(sigmoid(reranker_logit, temperature), 4)


def should_answer(reranker_logit: float, threshold: float = 0.5,
                  temperature: float = 1.0) -> bool:
    """Return True (answer) if confidence >= threshold, False (abstain) otherwise."""
    return confidence_score(reranker_logit, temperature) >= threshold


# ---------- calibration -------------------------------------------------------
def calibrate(threshold_range=None, temperature_range=None, verbose=False):
    """Grid-search threshold and temperature on the fit split of calibration_questions.jsonl.
    Objective: balanced accuracy = (TPR + TNR) / 2, where:
        TPR = correctly answered answerable questions / all answerable
        TNR = correctly abstained unanswerable questions / all unanswerable
    Returns best (threshold, temperature, balanced_acc).
    """
    from retrieve import bm25, knn, rrf, graph_expand, fetch_chunks
    from reranker import rerank
    from graph import KnowledgeGraph, load as graph_load

    if threshold_range is None:
        threshold_range = [round(x * 0.05, 2) for x in range(4, 18)]   # 0.20 … 0.85
    if temperature_range is None:
        temperature_range = [0.5, 1.0, 1.5, 2.0]

    qs = [json.loads(l) for l in (EVAL / "calibration_questions.jsonl").read_text().splitlines() if l.strip()]
    fit = [q for q in qs if q["split"] == "fit"]

    print(f"Calibrating on {len(fit)} fit questions (answerable={sum(q['answerable'] for q in fit)}, "
          f"unanswerable={sum(not q['answerable'] for q in fit)})…", flush=True)

    G = KnowledgeGraph(*graph_load("es"))

    # --- collect top-1 reranker score for every question -------------------------
    scores_for = {}
    for i, q in enumerate(fit, 1):
        if verbose:
            print(f"  [{i}/{len(fit)}] {q['qid']}  {q['question'][:60]}", flush=True)
        query = q["question"]
        b = bm25(query, 20); kv = knn(query, 20)
        g_ids = graph_expand(query, G, 20)
        hg = [cid for cid, _ in rrf({"bm25": b, "knn": kv, "graph": g_ids} if g_ids else {"bm25": b, "knn": kv})[:20]]
        texts = fetch_chunks(hg)
        cands = [{"chunk_id": cid, "text": texts.get(cid, {}).get("text", ""), "rrf_score": 0.0} for cid in hg]
        ranked = rerank(query, cands, top_n=1)
        scores_for[q["qid"]] = (ranked[0]["reranker_score"] if ranked else -15.0, q["answerable"])

    # --- grid search ---------------------------------------------------------------
    best_acc, best_t, best_temp = -1.0, 0.5, 1.0
    for temp in temperature_range:
        for thr in threshold_range:
            tp = tn = fp = fn = 0
            for _, (logit, answerable) in scores_for.items():
                pred = confidence_score(logit, temp) >= thr
                if answerable and pred:     tp += 1
                elif answerable and not pred: fn += 1
                elif not answerable and not pred: tn += 1
                else: fp += 1
            tpr = tp / (tp + fn) if (tp + fn) else 0
            tnr = tn / (tn + fp) if (tn + fp) else 0
            bal_acc = (tpr + tnr) / 2
            if bal_acc > best_acc:
                best_acc, best_t, best_temp = bal_acc, thr, temp

    print(f"\nBest: threshold={best_t}, temperature={best_temp}, "
          f"balanced_accuracy={best_acc:.3f}")
    return best_t, best_temp, best_acc, scores_for


# ---------- evaluation on test split ----------------------------------------
def evaluate_test(threshold, temperature, verbose=False):
    """Run the calibrated gate on the held-out test split."""
    from retrieve import bm25, knn, rrf, graph_expand, fetch_chunks
    from reranker import rerank
    from graph import KnowledgeGraph, load as graph_load

    qs = [json.loads(l) for l in (EVAL / "calibration_questions.jsonl").read_text().splitlines() if l.strip()]
    test = [q for q in qs if q["split"] == "test"]
    G = KnowledgeGraph(*graph_load("es"))

    tp = tn = fp = fn = 0
    rows = []
    for q in test:
        query = q["question"]
        b = bm25(query, 20); kv = knn(query, 20)
        g_ids = graph_expand(query, G, 20)
        hg = [cid for cid, _ in rrf({"bm25": b, "knn": kv, "graph": g_ids} if g_ids else {"bm25": b, "knn": kv})[:20]]
        texts = fetch_chunks(hg)
        cands = [{"chunk_id": cid, "text": texts.get(cid, {}).get("text", ""), "rrf_score": 0.0} for cid in hg]
        ranked = rerank(query, cands, top_n=1)
        logit = ranked[0]["reranker_score"] if ranked else -15.0
        conf = confidence_score(logit, temperature)
        pred = conf >= threshold
        answerable = q["answerable"]
        if answerable and pred:     tp += 1; label = "TP"
        elif answerable and not pred: fn += 1; label = "FN"
        elif not answerable and not pred: tn += 1; label = "TN"
        else: fp += 1; label = "FP"
        rows.append((q["qid"], answerable, logit, conf, pred, label))
        if verbose:
            print(f"  {q['qid']} {'ans' if answerable else 'UNANS':6s}  logit={logit:+.2f}  "
                  f"conf={conf:.3f}  pred={'answer' if pred else 'ABSTAIN':8s}  {label}")

    tpr = tp / (tp + fn) if (tp + fn) else 0
    tnr = tn / (tn + fp) if (tn + fp) else 0
    bal_acc = (tpr + tnr) / 2
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, tpr=round(tpr, 3), tnr=round(tnr, 3),
                balanced_accuracy=round(bal_acc, 3), rows=rows)


# ---------- CLI ---------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=str, default=None, help="Single question demo")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    if a.demo:
        from retrieve import bm25, knn, rrf, graph_expand, fetch_chunks
        from reranker import rerank
        from graph import KnowledgeGraph, load as graph_load
        G = KnowledgeGraph(*graph_load("es"))
        query = a.demo
        b = bm25(query, 20); kv = knn(query, 20)
        g_ids = graph_expand(query, G, 20)
        hg = [cid for cid, _ in rrf({"bm25": b, "knn": kv, "graph": g_ids} if g_ids else {"bm25": b, "knn": kv})[:20]]
        texts = fetch_chunks(hg)
        cands = [{"chunk_id": cid, "text": texts.get(cid, {}).get("text", ""), "rrf_score": 0.0} for cid in hg]
        ranked = rerank(query, cands, top_n=1)
        logit = ranked[0]["reranker_score"] if ranked else -15.0
        thr = a.threshold or 0.5; temp = a.temperature or 1.0
        conf = confidence_score(logit, temp)
        print(f"\nQuery:      {query!r}")
        print(f"Top-1 chunk: {ranked[0]['chunk_id'] if ranked else 'none'}")
        print(f"Logit:      {logit:+.4f}")
        print(f"Confidence: {conf:.4f}  (temp={temp}, threshold={thr})")
        print(f"Decision:   {'ANSWER' if conf >= thr else 'ABSTAIN'}")
    else:
        thr, temp, ba, scores = calibrate(verbose=a.verbose)
        if a.threshold: thr = a.threshold
        if a.temperature: temp = a.temperature
        print(f"\n--- Test split evaluation (threshold={thr}, temperature={temp}) ---")
        res = evaluate_test(thr, temp, verbose=True)
        print(f"\nTP={res['tp']}  TN={res['tn']}  FP={res['fp']}  FN={res['fn']}")
        print(f"TPR (sensitivity)={res['tpr']}  TNR (specificity)={res['tnr']}")
        print(f"Balanced accuracy={res['balanced_accuracy']}")
