"""Scores every extractor output found in ingest/out against gold_entities.jsonl and compares runs.
Usage: python evalsuite/eval_entities.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from common import *
from ner_metrics import score

RUNS = [("baseline", "baseline.json"), ("llm_t0_run1", "llm_t0_run1.json"), ("llm_t0_run2", "llm_t0_run2.json"), ("llm_t0.7_run1", "llm_t0.7_run1.json")]
gold = {g["doc_id"]: g["entities"] for g in load_jsonl(EVAL / "gold_entities.jsonl")}
loaded = {}
for name, f in RUNS:
    if (OUT / f).exists(): loaded[name] = json.load(open(OUT / f))
lines = []; allres = {}
for name, data in loaded.items():
    pred = {d: v for d, v in data["entities"].items() if d in gold}
    res = score(gold, pred, ENTITY_TYPES); allres[name] = res
    lines.append(f"\n### {name}\n\n| Type | Exact P | Exact R | Exact F1 | Relaxed P | Relaxed R | Relaxed F1 | gold |\n|---|---|---|---|---|---|---|---|")
    for t in ENTITY_TYPES + ["MICRO"]:
        e, r = res["exact"][t], res["relaxed"][t]
        lines.append(f"| {t} | {e['precision']:.3f} | {e['recall']:.3f} | {e['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {e['tp'] + e['fn']} |")
    lines.append(f"\nMacro-F1 exact {res['exact']['MACRO_F1']:.3f}, relaxed {res['relaxed']['MACRO_F1']:.3f}")

def keyset(data):
    return {(d, e["type"], e["start"], e["end"]) for d, v in data["entities"].items() if d in gold for e in v}
lines.append("\n### Determinism / run-to-run differences (gold documents)\n\n| Pair | entities A | entities B | only in A | only in B | identical |\n|---|---|---|---|---|---|")
for a, b in [("llm_t0_run1", "llm_t0_run2"), ("llm_t0_run1", "llm_t0.7_run1")]:
    if a in loaded and b in loaded:
        ka, kb = keyset(loaded[a]), keyset(loaded[b])
        lines.append(f"| {a} vs {b} | {len(ka)} | {len(kb)} | {len(ka - kb)} | {len(kb - ka)} | {ka == kb} |")
lines.append("\n### Speed\n\n| Method | avg sec/doc | tokens/s (LLM) | invalid or dropped items |\n|---|---|---|---|")
for name, data in loaded.items():
    sec = data["seconds"]; avg = sum(sec.values()) / len(sec)
    tps = data.get("tokens_per_second"); dropped = data.get("dropped_total", "-")
    lines.append(f"| {name} | {avg:.2f} | {tps if tps else '-'} | {dropped} |")
text = "\n".join(lines); print(text)
(ROOT / "evalsuite" / "results").mkdir(exist_ok=True)
(ROOT / "evalsuite" / "results" / "entity_scores.md").write_text(text, encoding="utf-8")
json.dump(allres, open(ROOT / "evalsuite" / "results" / "entity_scores.json", "w"), indent=1)
print("\nsaved evalsuite/results/entity_scores.md")
