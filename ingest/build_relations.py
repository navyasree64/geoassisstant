"""Build graph relations from the resolved entities (Task 2) and the GeoServer geometries, then index them.
Usage: python ingest/build_relations.py [--dry-run]
Weights (see PROJECT DOCUMENT): each piece of evidence gets a reliability score by evidence type; several independent
pieces of evidence for the same edge are combined later by noisy-OR (1 - prod(1 - w))."""
import re, json, hashlib, argparse
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer
from common import *

TR = Transformer.from_crs(4326, 32646, always_xy=True)       # projected CRS in metres (EPSG:32646 as required by the task)
def proj(g): return transform(TR.transform, shape(g))
NEAR_M = 300.0
APPLIES = re.compile(r"\bappl(?:ies|y) to\b")
SUPERSEDES = re.compile(r"supersedes\s+Circular\s*(?:No\.\s*)?$")

def edge(t, s, tg, w, ev, ents):
    r = dict(type=t, source_id=s, target_id=tg, weight=round(w, 3), evidence=ev, source_type=ents[s]["type"], target_type=ents[tg]["type"])
    r["edge_id"] = hashlib.sha1("|".join([t, s, tg, ev]).encode()).hexdigest()[:20]
    return r

def build():
    ents = {e["entity_id"]: e for e in json.load(open(OUT / "entities.json"))}
    docs, chunks = load_docs(), load_chunks_ref()
    by_name = {(e["type"], e["canonical_name"]): k for k, e in ents.items()}
    own = {e["attributes"]["doc_id"]: k for k, e in ents.items() if e["type"] == "Circular" and e["attributes"].get("doc_id")}
    out = {}
    def add(t, s, tg, w, ev):
        if s in ents and tg in ents:
            r = edge(t, s, tg, w, ev, ents)
            if r["edge_id"] not in out or out[r["edge_id"]]["weight"] < r["weight"]: out[r["edge_id"]] = r
    ctext = {c["chunk_id"]: c["text"] for c in chunks}; cstart = {c["chunk_id"]: c["start"] for c in chunks}
    mentions = [(k, m) for k, e in ents.items() for m in e["mentions"]]

    # ---- applies_to: Circular -> Board (and Rule -> Board, inherited from its document)
    doc_applies = {}
    for k, m in mentions:
        if ents[k]["type"] != "Board": continue
        d, ch = m["doc_id"], m["chunk_id"]; ct = ctext[ch]
        local = m["start"] - cstart[ch]
        before = ct[max(0, local - 200):local]
        hit = [x for x in APPLIES.finditer(before) if ". " not in before[x.end():]]
        line_start = ct.rfind("\n", 0, local) + 1
        if hit: w = 0.9
        elif ct[line_start:line_start + 8] == "Subject:": w = 0.7
        else: continue
        if k not in doc_applies.setdefault(d, {}) or doc_applies[d][k][0] < w: doc_applies[d][k] = (w, ch)
    for d, bs in doc_applies.items():
        for b, (w, ch) in bs.items():
            if d in own: add("applies_to", own[d], b, w, ch)
    for k, m in mentions:
        if ents[k]["type"] == "Rule":
            for b, (w, _) in doc_applies.get(m["doc_id"], {}).items(): add("applies_to", k, b, 0.8 * w, m["chunk_id"])

    # ---- mentions: Circular -> School (one edge per chunk where the school appears)
    for k, m in mentions:
        e = ents[k]
        if e["type"] != "School" or m["doc_id"] not in own: continue
        w = 0.7 if not e["attributes"].get("linked") else (0.9 if m["text"] == e["canonical_name"] else 0.8)
        add("mentions", own[m["doc_id"]], k, w, m["chunk_id"])

    # ---- affiliated_with and located_in: GeoServer attributes + explicit wording in the text
    for k, e in ents.items():
        if e["type"] != "School": continue
        a = e["attributes"]
        if a.get("board") and ("Board", a["board"]) in by_name: add("affiliated_with", k, by_name[("Board", a["board"])], 0.95, "geoserver")
        if a.get("locality") and ("Locality", a["locality"]) in by_name: add("located_in", k, by_name[("Locality", a["locality"])], 0.9, "geoserver")
    for k, m in mentions:
        if ents[k]["type"] != "School": continue
        t = docs[m["doc_id"]]["text"]; after, before = t[m["end"]:m["end"] + 40], t[max(0, m["start"] - 25):m["start"]]
        mb = re.match(r" \((CBSE|ICSE|WBBSE)\)", after)                       # "Hare School (WBBSE)"
        if mb: add("affiliated_with", k, by_name[("Board", mb.group(1))], 0.9, m["chunk_id"])
        mb = re.search(r"\b(CBSE|ICSE|WBBSE) school $", before)               # "the new CBSE school Lakeview Public School"
        if mb: add("affiliated_with", k, by_name[("Board", mb.group(1))], 0.8, m["chunk_id"])
        for ln in sorted((n for (ty, n) in by_name if ty == "Locality"), key=len, reverse=True):   # "Sunrise International School in Salt Lake"
            if after.startswith(" in " + ln):
                add("located_in", k, by_name[("Locality", ln)], 0.8, m["chunk_id"]); break
    for c in chunks:                                                           # "In Howrah, the schools covered ... are A (X), B (Y)."
        ml = re.match(r"In ([A-Z][A-Za-z ]+?), the schools", c["text"])
        if ml and ("Locality", ml.group(1)) in by_name:
            for k, m in mentions:
                if ents[k]["type"] == "School" and m["chunk_id"] == c["chunk_id"]:
                    add("located_in", k, by_name[("Locality", ml.group(1))], 0.85, c["chunk_id"])

    # ---- near: school polygon within 300 m of a road line, computed in EPSG:32646
    sch = {f["properties"]["osm_id"]: proj(f["geometry"]) for f in fetch_layer("schools")}
    rds = {f["properties"]["name"]: proj(f["geometry"]) for f in fetch_layer("roads")}
    for k, e in ents.items():
        if e["type"] != "School" or not e["attributes"].get("osm_id"): continue
        g = sch[e["attributes"]["osm_id"]]
        for rn, rg in rds.items():
            d = g.distance(rg)
            if d <= NEAR_M and ("Road", rn) in by_name:
                add("near", k, by_name[("Road", rn)], 0.5 + 0.5 * (1 - d / NEAR_M), "geoserver")   # 1.0 at 0 m, 0.5 at 300 m

    # ---- supersedes: explicit wording "supersedes Circular X"; weight lowered if the dates contradict it
    for k, m in mentions:
        if ents[k]["type"] != "Circular" or m["doc_id"] not in own or own[m["doc_id"]] == k: continue
        t = docs[m["doc_id"]]["text"]
        if SUPERSEDES.search(t[max(0, m["start"] - 40):m["start"]]):
            s = own[m["doc_id"]]; ds, dt = ents[s]["attributes"].get("date"), ents[k]["attributes"].get("date")
            add("supersedes", s, k, 0.95 if (ds and dt and ds > dt) else 0.4, m["chunk_id"])
    return ents, sorted(out.values(), key=lambda r: (r["type"], r["source_id"], r["target_id"], r["evidence"]))

def reference_check(ents, rels):
    ref = json.load(open(EVAL / "relations_reference.json"))
    own_doc = {k: e["attributes"]["doc_id"] for k, e in ents.items() if e["type"] == "Circular" and e["attributes"].get("doc_id")}
    name = lambda k: own_doc.get(k) if ents[k]["type"] == "Circular" else ents[k]["canonical_name"]
    ours = {(r["type"], name(r["source_id"]), name(r["target_id"])) for r in rels
            if (r["type"] in ("applies_to", "mentions", "supersedes") and ents[r["source_id"]]["type"] == "Circular") or r["type"] in ("affiliated_with", "located_in", "near")}
    gold = {(r["type"], r["source"], r["target"]) for r in ref
            if (r["type"] in ("applies_to", "mentions", "supersedes") and r["source"].startswith("CIR")) or r["type"] in ("affiliated_with", "located_in", "near")}
    lines = ["| Relation | ours | reference | matched | precision | recall |", "|---|---|---|---|---|---|"]
    for t in ["affiliated_with", "applies_to", "mentions", "located_in", "near", "supersedes"]:
        o, g = {x for x in ours if x[0] == t}, {x for x in gold if x[0] == t}
        tp = len(o & g); lines.append(f"| {t} | {len(o)} | {len(g)} | {tp} | {tp / len(o) if o else 0:.3f} | {tp / len(g) if g else 0:.3f} |")
    return "\n".join(lines)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    ents, rels = build()
    json.dump(rels, open(OUT / "relations.json", "w"), indent=1)
    cnt = {}
    for r in rels: cnt[r["type"]] = cnt.get(r["type"], 0) + 1
    print("relations:", len(rels), cnt); print("\nCheck against the data-pack reference (documents CIR-* and GeoServer relations):\n"); print(reference_check(ents, rels))
    if a.dry_run: raise SystemExit("\ndry run: nothing sent to Elasticsearch")
    mp = json.load(open(ROOT / "es" / "relations_mapping.json"))
    r = es("HEAD", "/east_india_relations", expect=None)
    if r.status_code == 404: es("PUT", "/east_india_relations", json=mp); print("\ncreated index east_india_relations")
    elif r.status_code != 200: raise SystemExit(f"cannot access index: HTTP {r.status_code}")
    lines = []
    for r_ in rels:
        lines += [json.dumps({"index": {"_index": "east_india_relations", "_id": r_["edge_id"]}}), json.dumps(r_)]    # deterministic _id
    res = es("POST", "/_bulk?refresh=true", data="\n".join(lines) + "\n", headers={"Content-Type": "application/x-ndjson"}).json()
    bad = [list(i.values())[0] for i in res["items"] if "error" in list(i.values())[0]]
    print("bulk items:", len(res["items"]), "| failed:", len(bad)); bad and print(bad[:2])
    print("documents in east_india_relations:", es("GET", "/east_india_relations/_count").json()["count"])
