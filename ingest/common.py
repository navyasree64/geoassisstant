"""Shared helpers. Secrets come from environment variables only:
RAG_APP_PASSWORD (Elasticsearch user rag_app_user) and GEOSERVER_RAG_PASSWORD (GeoServer user rag_geo)."""
import os, json, re, requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_pack" / "data"
EVAL = ROOT / "data_pack" / "eval"
OUT = ROOT / "ingest" / "out"
OUT.mkdir(parents=True, exist_ok=True)
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
GS_URL = os.environ.get("GS_URL", "http://localhost:8080/geoserver")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
ENTITY_TYPES = ["School", "Board", "Circular", "Rule", "Locality", "Road", "Date", "Quantity"]

def es_auth():  return ("rag_app_user", os.environ["RAG_APP_PASSWORD"])
def geo_auth(): return ("rag_geo", os.environ["GEOSERVER_RAG_PASSWORD"])

def fetch_layer(name):
    """Fetch a GeoServer layer as GeoJSON features through WFS (authenticated as rag_geo)."""
    p = dict(service="WFS", version="1.0.0", request="GetFeature", typeName=f"east_india:{name}", outputFormat="application/json")
    r = requests.get(f"{GS_URL}/east_india/ows", params=p, auth=geo_auth(), timeout=60)
    r.raise_for_status()
    return r.json()["features"]

def load_docs():
    docs = {}
    for f in sorted((DATA / "docs").iterdir()):
        with open(f, encoding="utf-8", newline="") as fh:
            docs[f.stem] = dict(file=f.name, text=fh.read())
    return docs

def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

def load_chunks_ref():
    return load_jsonl(EVAL / "chunks_reference.jsonl")

def chunk_id_for(doc_id, start, chunks):
    for c in chunks:
        if c["doc_id"] == doc_id and c["start"] <= start < c["end"]:
            return c["chunk_id"]
    return None

MONTHS = {m: i + 1 for i, m in enumerate(["January","February","March","April","May","June","July","August","September","October","November","December"])}
def to_iso(text):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
    if m and m.group(2).capitalize() in MONTHS:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(2).capitalize()]:02d}-{int(m.group(1)):02d}"
    return None

BOARD_ALIASES = {"west bengal board of secondary education": "WBBSE", "central board of secondary education": "CBSE",
                 "council for the indian school certificate examinations": "ICSE", "wbbse": "WBBSE", "cbse": "CBSE", "icse": "ICSE"}
def board_canonical(text): return BOARD_ALIASES.get(text.strip().lower(), text.strip())

def resolve_overlaps(cands):
    """Greedy: longest span first, drop anything that overlaps an accepted span."""
    cands = sorted(cands, key=lambda e: (-(e["end"] - e["start"]), e["start"]))
    kept = []
    for e in cands:
        if all(e["end"] <= k["start"] or e["start"] >= k["end"] for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e["start"])


def es(method, path, admin=False, expect=(200, 201), **kw):
    """Elasticsearch call. Default user is rag_app_user; admin=True uses the elastic superuser (ONLY for one-off
    experiments such as benchmark indices, never inside the application). Raises with the server message on failure."""
    auth = ("elastic", os.environ["ELASTIC_PASSWORD"]) if admin else es_auth()
    r = requests.request(method, f"{ES_URL}{path}", auth=auth, timeout=600, **kw)
    if expect and r.status_code not in expect:
        raise SystemExit(f"Elasticsearch error: {method} {path} -> HTTP {r.status_code}: {r.text[:400]}\n"
                         f"(401 = wrong password in the environment variable; 403 = role lacks permission)")
    return r

def load_judgments():
    """[{qid, type, split, query, rel: {chunk_id: grade}}]"""
    out = []
    for q in load_jsonl(EVAL / "retrieval_judgments.jsonl"):
        q["rel"] = {j["chunk_id"]: j["grade"] for j in q["judgments"]}
        out.append(q)
    return out
