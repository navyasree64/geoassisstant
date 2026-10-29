"""Non-LLM baseline extractor: regular expressions + a gazetteer built from the GeoServer layers.
Usage:  python ingest/extract_baseline.py          (all 14 documents -> ingest/out/baseline.json)"""
import re, json, time
from common import *

MON = "|".join(MONTHS)
RX = {
    "Date": re.compile(rf"\b\d{{1,2}} (?:{MON}) \d{{4}}\b"),
    "Circular": re.compile(r"\bDEO/ADM/\d{4}/\d{3}\b"),
    "Rule": re.compile(r"\bRule \d+(?:\.\d+)?(?:\(\d+\))?"),
}
QUNIT = [(re.compile(r"\b(\d+) years\b"), "age_years"), (re.compile(r"\b(\d+) seats\b"), "seats"),
         (re.compile(r"\b(\d+) marks\b"), "marks"), (re.compile(r"\b(\d+) percent\b"), "percent"),
         (re.compile(r"Rs\. (\d+)\b"), "rupees"), (re.compile(r"\b(\d+) days\b"), "days"), (re.compile(r"\b(\d+) parents\b"), "persons")]
SCHOOL_FALLBACK = re.compile(r"\b(?:[A-Z][\w'\-]*(?:\.[\w'\-]+)*[ ]){1,6}(?:School|Academy|Institution|Institute|Vidyalaya|Vidyasram)\b")
BOARD_RX = re.compile(r"\b(" + "|".join(sorted(map(re.escape, BOARD_ALIASES), key=len, reverse=True)) + r")\b", re.I)

def build_gazetteer():
    sch = fetch_layer("schools"); roads = fetch_layer("roads"); lu = fetch_layer("landuse")
    names = sorted({f["properties"]["name"] for f in sch if f["properties"].get("name") and len(f["properties"]["name"]) >= 4}, key=len, reverse=True)
    localities = sorted({f["properties"]["name"] for f in lu} | {f["properties"]["locality"] for f in sch if f["properties"].get("locality")}, key=len, reverse=True)
    rnames = sorted({f["properties"]["name"] for f in roads}, key=len, reverse=True)
    return names, localities, rnames

def find_all(name, text, flags=0):
    for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, flags):
        yield m.start(), m.end()

def extract(text, gaz):
    names, localities, rnames = gaz
    c = []
    def add(t, s, e, canon, kind=None): c.append(dict(type=t, start=s, end=e, text=text[s:e], canonical=canon, kind=kind))
    for n in names:
        for s, e in find_all(n, text, re.I): add("School", s, e, n)
    for m in SCHOOL_FALLBACK.finditer(text): add("School", m.start(), m.end(), m.group(0))
    for n in rnames:
        for s, e in find_all(n, text): add("Road", s, e, n)
    for n in localities:
        for s, e in find_all(n, text): add("Locality", s, e, n)
    for m in BOARD_RX.finditer(text): add("Board", m.start(), m.end(), board_canonical(m.group(0)))
    for t, rx in RX.items():
        for m in rx.finditer(text): add(t, m.start(), m.end(), to_iso(m.group(0)) if t == "Date" else m.group(0))
    for rx, kind in QUNIT:
        for m in rx.finditer(text): add("Quantity", m.start(), m.end(), m.group(1), kind)
    # table cells: first numeric column = seats, second = cut-off marks
    for line in re.finditer(r"^\|.*\|[ ]*$", text, re.M):
        for i, m in enumerate(re.finditer(r"\|\s*(\d+)\s*(?=\|)", line.group(0))):
            s = line.start() + m.start(1); e = line.start() + m.end(1)
            add("Quantity", s, e, m.group(1), "seats" if i == 0 else "marks")
    return resolve_overlaps(c)

if __name__ == "__main__":
    gaz = build_gazetteer(); docs = load_docs(); out = {}; timing = {}
    for did, d in docs.items():
        t = time.perf_counter(); out[did] = extract(d["text"], gaz); timing[did] = time.perf_counter() - t
    json.dump(dict(method="baseline", entities=out, seconds=timing), open(OUT / "baseline.json", "w"), indent=1)
    print("documents:", len(docs), "entities:", sum(map(len, out.values())), "avg sec/doc: %.4f" % (sum(timing.values()) / len(timing)))
