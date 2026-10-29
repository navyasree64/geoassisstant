"""My own precision / recall / F1 for entity extraction (exact and relaxed span match).
Gold and predicted entities are dicts with type, start, end. Matching is one-to-one, per document and per type.

Usage:
    from ner_metrics import score
    results = score(gold_dict, pred_dict)
"""
def overlap(a, b): return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))

def match_counts(gold, pred, relaxed):
    """Return (tp, fp, fn) for ONE document and ONE entity type."""
    used = set(); tp = 0
    for g in sorted(gold, key=lambda e: (e["start"], e["end"])):
        best, best_ov = None, 0
        for j, p in enumerate(pred):
            if j in used: continue
            if relaxed:
                ov = overlap(g, p)
                if ov > best_ov: best, best_ov = j, ov
            elif p["start"] == g["start"] and p["end"] == g["end"]:
                best = j; break
        if best is not None:
            used.add(best); tp += 1
    return tp, len(pred) - tp, len(gold) - tp

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f, 4), tp=tp, fp=fp, fn=fn)

def score(gold_by_doc, pred_by_doc, types):
    """gold_by_doc / pred_by_doc: {doc_id: [entity,...]}. Only documents present in gold are scored."""
    res = {}
    for mode, relaxed in (("exact", False), ("relaxed", True)):
        tot = {t: [0, 0, 0] for t in types}
        for d, gl in gold_by_doc.items():
            pl = pred_by_doc.get(d, [])
            for t in types:
                tp, fp, fn = match_counts([e for e in gl if e["type"] == t], [e for e in pl if e["type"] == t], relaxed)
                tot[t][0] += tp; tot[t][1] += fp; tot[t][2] += fn
        res[mode] = {t: prf(*tot[t]) for t in types}
        res[mode]["MICRO"] = prf(*[sum(tot[t][i] for t in types) for i in range(3)])
        f1s = [res[mode][t]["f1"] for t in types]
        res[mode]["MACRO_F1"] = round(sum(f1s) / len(f1s), 4)
    return res
