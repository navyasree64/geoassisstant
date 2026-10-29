"""Entity resolution (merge spelling variants), GeoServer linking, and idempotent indexing into east_india_entities.
Usage:
  python ingest/resolve_and_index.py --source baseline            [--dry-run]
  python ingest/resolve_and_index.py --source llm_t0_run1_all
Runs with the rag_app_user role (never the elastic superuser); GeoServer reads use rag_geo."""
import re, json, argparse, hashlib, requests
from difflib import SequenceMatcher
from shapely.geometry import shape
from common import *

THRESHOLD = 0.85

def norm(s):
    s = s.lower().replace("\u2019", "'").replace("'", "").replace(".", " ")
    s = re.sub(r"\bst\b", "saint", s); s = re.sub(r"^the ", "", s.strip())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()
# words that distinguish two DIFFERENT schools even when the rest of the name is identical
DISCRIMINATING = {"boys", "girls", "boy", "girl", "primary", "secondary", "junior", "senior", "main", "hindi", "english", "north", "south", "east", "west"}
def sim(a, b):
    na, nb = norm(a), norm(b)
    if (set(na.split()) ^ set(nb.split())) & DISCRIMINATING: return 0.0   # e.g. Boys vs Girls must never merge
    return SequenceMatcher(None, na, nb).ratio()
def slug(s): return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def build(pred, docs, chunks, sch, roads, lu):
    layer = [f for f in sch if f["properties"].get("name")]
    ents = {}   # entity_id -> entity
    def get(eid, etype, name):
        return ents.setdefault(eid, dict(entity_id=eid, type=etype, canonical_name=name, aliases=set(), attributes={}, mentions=[]))
    def mention(e, did, m):
        e["mentions"].append(dict(chunk_id=chunk_id_for(did, m["start"], chunks), doc_id=did, start=m["start"], end=m["end"], text=m["text"]))
        if m["text"] != e["canonical_name"]: e["aliases"].add(m["text"])
    # ---- School resolution: cluster surface strings, then link each cluster to a GeoServer feature
    surfaces = {}
    for did, lst in pred.items():
        for m in lst:
            if m["type"] == "School": surfaces.setdefault(m["canonical"], []).append((did, m))
    clusters = []   # each: dict(rep, members=set(of canonical strings))
    for s in sorted(surfaces, key=lambda x: (-len(surfaces[x]), x)):
        for c in clusters:
            if sim(s, c["rep"]) >= THRESHOLD: c["members"].add(s); break
        else: clusters.append(dict(rep=s, members={s}))
    unlinked, merged_log, link_log = [], [], []
    for c in clusters:
        best, score_ = None, 0.0
        for f in layer:
            sc = max(sim(f["properties"]["name"], x) for x in c["members"])
            if sc > score_: best, score_ = f, sc
        linked = best is not None and score_ >= THRESHOLD
        if len(c["members"]) > 1: merged_log.append(sorted(c["members"]))
        if linked:
            p = best["properties"]; cen = shape(best["geometry"]).centroid
            e = get(f"school:{p['osm_id']}", "School", p["name"])
            toint = lambda v: int(float(v)) if v not in (None, "") else None
            e["attributes"].update(osm_id=p["osm_id"], board=p.get("board"), seats=toint(p.get("seats")), cutoff_marks=toint(p.get("cutoff_marks")),
                                   locality=p.get("locality"), location=dict(lat=cen.y, lon=cen.x), linked=True)
            link_log.append((sorted(c["members"]), p["name"], round(score_, 3)))
        else:
            e = get(f"school:unlinked-{slug(c['rep'])}", "School", c["rep"]); e["attributes"]["linked"] = False
            unlinked.append((c["rep"], round(score_, 3)))
        for sname in c["members"]:
            for did, m in surfaces[sname]: mention(e, did, m)
    # ---- other types: one entity per canonical value
    lu_by = {f["properties"]["name"]: shape(f["geometry"]).centroid for f in lu}
    rd_by = {f["properties"]["name"]: f for f in roads}
    for did, lst in pred.items():
        for m in lst:
            t, c = m["type"], m["canonical"]
            if t == "School" or c is None: continue
            if t == "Quantity":
                e = get(f"quantity:{m.get('kind') or 'unknown'}-{c}", t, f"{c} {m.get('kind') or ''}".strip())
                e["attributes"].update(value=int(c), kind=m.get("kind"))
            elif t == "Date":
                e = get(f"date:{c}", t, c); e["attributes"]["date"] = c
            elif t == "Locality":
                e = get(f"locality:{slug(c)}", t, c)
                if c in lu_by: e["attributes"]["location"] = dict(lat=lu_by[c].y, lon=lu_by[c].x)
            elif t == "Road":
                e = get(f"road:{slug(c)}", t, c)
                if c in rd_by: e["attributes"]["osm_id"] = rd_by[c]["properties"]["osm_id"]
            else:
                e = get(f"{t.lower()}:{slug(c)}", t, c)
            mention(e, did, m)
    # circular attributes: its own date (first Date after the header number) and the document it is issued in
    for did, d in docs.items():
        if d["text"].startswith("# Circular No."):
            lst = sorted(pred.get(did, []), key=lambda m: m["start"])
            circ = next((m for m in lst if m["type"] == "Circular"), None); dt = next((m for m in lst if m["type"] == "Date"), None)
            if circ and f"circular:{slug(circ['canonical'])}" in ents:
                a = ents[f"circular:{slug(circ['canonical'])}"]["attributes"]; a["number"] = circ["canonical"]; a["doc_id"] = did
                if dt: a["date"] = dt["canonical"]
    for e in ents.values():
        e["aliases"] = sorted(e["aliases"])
        e["attributes"] = {k: v for k, v in e["attributes"].items() if v is not None}
    return ents, unlinked, merged_log, link_log

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pred = json.load(open(OUT / f"{a.source}.json"))["entities"]
    docs = load_docs(); chunks = load_chunks_ref()
    sch, roads, lu = fetch_layer("schools"), fetch_layer("roads"), fetch_layer("landuse")
    ents, unlinked, merged, linked = build(pred, docs, chunks, sch, roads, lu)
    json.dump(list(ents.values()), open(OUT / "entities.json", "w"), indent=1)
    by_type = {}
    for e in ents.values(): by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    report = ["Entities by type: " + json.dumps(by_type), f"Similarity threshold: {THRESHOLD}", "", "Variants merged into one School entity:"]
    report += [f"  {m}" for m in merged] or ["  (none)"]
    report += ["", "School clusters linked to GeoServer (surface forms -> layer name, score):"] + [f"  {x}" for x in linked]
    report += ["", "School entities NOT linked (name, best score):"] + [f"  {x}" for x in unlinked]
    txt = "\n".join(report); print(txt); (OUT / "resolution_report.txt").write_text(txt, encoding="utf-8")
    if a.dry_run: raise SystemExit("dry run: nothing sent to Elasticsearch")
    mapping = json.load(open(ROOT / "es" / "entities_mapping.json"))
    r = es("HEAD", "/east_india_entities", expect=None)
    if r.status_code == 404:
        es("PUT", "/east_india_entities", json=mapping); print("created index east_india_entities")
    elif r.status_code != 200:
        raise SystemExit(f"cannot access index: HTTP {r.status_code} (check RAG_APP_PASSWORD)")
    lines = []
    for e in ents.values():
        lines.append(json.dumps({"index": {"_index": "east_india_entities", "_id": e["entity_id"]}}))   # deterministic _id -> idempotent
        lines.append(json.dumps(e))
    res = es("POST", "/_bulk?refresh=true", data="\n".join(lines) + "\n", headers={"Content-Type": "application/x-ndjson"}).json()
    bad = [list(i.values())[0] for i in res["items"] if "error" in list(i.values())[0]]
    print("bulk items:", len(res["items"]), "| failed:", len(bad))
    if bad: print(bad[:3])
    print("documents in east_india_entities:", es("GET", "/east_india_entities/_count").json()["count"])
