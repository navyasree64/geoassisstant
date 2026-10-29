r"""Self-check against the assessment checklist: folder layout, secrets, live services, index mappings, metric functions,
and what PROJECT DOCUMENT.md covers. Run from the project root:   python evalsuite\selfcheck.py
Set RAG_APP_PASSWORD and GEOSERVER_RAG_PASSWORD first to include the live checks."""
import os, re, sys, json, importlib.util
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
ES = os.environ.get("ES_URL", "http://localhost:9200"); GS = os.environ.get("GS_URL", "http://localhost:8080/geoserver")
R = []
def check(name, ok, detail=""):
    R.append(bool(ok)); print(("PASS  " if ok else "FAIL  ") + name + (f"   -> {detail}" if detail else ""))
def skip(name, why): print(f"SKIP  {name}   -> {why}")
def section(t): print(f"\n=== {t} ===")

# ---------------------------------------------------------------- A. layout
section("A. Folder layout and required files")
for d in ["ingest", "retrieve", "evalsuite", "es", "geoserver", "app", "evidence", "diagram"]:
    check(f"folder {d}/", (ROOT / d).is_dir())
req = ROOT / "requirements.txt"
check("requirements.txt exists", req.exists())
if req.exists():
    lines = [l.strip() for l in req.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip() and not l.startswith("#")]
    unpinned = [l for l in lines if "==" not in l]
    check("requirements.txt is pinned (==)", lines and not unpinned, f"{len(lines)} packages, unpinned: {unpinned[:5]}")
check("PROJECT DOCUMENT.md exists", (ROOT / "PROJECT DOCUMENT.md").exists())
check("diagram/ has a PNG or PDF", any(p.suffix.lower() in (".png", ".pdf") for p in (ROOT / "diagram").glob("*")) if (ROOT / "diagram").is_dir() else False)
mp = [p.name for p in (ROOT / "es").glob("*.json")] if (ROOT / "es").is_dir() else []
check("es/ has mappings for chunks, entities, relations", all(any(k in n for n in mp) for k in ("chunk", "entit", "relation")), str(mp))
check("es/ has role/API-key setup notes or script", any(p.suffix in (".md", ".txt", ".py") for p in (ROOT / "es").glob("*")) if (ROOT / "es").is_dir() else False)
check("geoserver/ has notes on the rag_geo role", any(p.suffix in (".md", ".txt") for p in (ROOT / "geoserver").glob("*")) if (ROOT / "geoserver").is_dir() else False)
ev = list((ROOT / "evidence").glob("*")) if (ROOT / "evidence").is_dir() else []
check("evidence/ has screenshots and logs", len(ev) >= 10, f"{len(ev)} files, {sum(p.suffix.lower() in ('.png', '.jpg') for p in ev)} images")
for f in ["app/pipeline.py", "retrieve/graph.py", "retrieve/retrieve.py", "retrieve/reranker.py", "retrieve/confidence.py", "evalsuite/metrics.py",
          "ingest/chunker.py", "ingest/extract_baseline.py", "ingest/extract_llm.py", "ingest/resolve_and_index.py", "ingest/build_relations.py", "ingest/index_chunks.py"]:
    check(f"file {f}", (ROOT / f).exists())
check("evidence/ has at least 5 screenshots (png/jpg)", sum(p.suffix.lower() in (".png", ".jpg", ".jpeg") for p in ev) >= 5, f"{sum(p.suffix.lower() in ('.png', '.jpg', '.jpeg') for p in ev)} images")
for f in ["evalsuite/eval_retrieval.py", "evalsuite/eval_rerank.py", "evalsuite/eval_calibration.py"]:
    if not (ROOT / f).exists(): print(f"INFO  no {f} (expected for tune/holdout, reranking table and calibration results; names may differ)")
check("app/ has an interface (Streamlit/FastAPI/CLI) showing entities, graph path and settings", any("entities" in p.read_text(encoding="utf-8", errors="ignore") and "path" in p.read_text(encoding="utf-8", errors="ignore").lower()
      for p in (ROOT / "app").glob("*.py")) if (ROOT / "app").is_dir() else False, "looks for the words 'entities' and 'path' in app/*.py")

appsrc = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in (ROOT / "app").rglob("*.py")) if (ROOT / "app").is_dir() else ""
retsrc = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in (ROOT / "retrieve").rglob("*.py")) if (ROOT / "retrieve").is_dir() else ""
check("code: answers validated against a JSON schema (entities_used + jsonschema/pydantic)", "entities_used" in appsrc and re.search(r"jsonschema|pydantic|validate", appsrc, re.I) is not None)
check("code: exact abstention sentence 'I cannot answer this from the available sources.'", "I cannot answer this from the available sources." in appsrc)
check("code: cited chunks checked against retrieved chunks", re.search(r"not retrieved|unretrieved|hallucinated_cit|invalid_cit|cited.{0,40}retrieved", appsrc, re.I) is not None)
check("code: refusal paths (delete/modify, secrets/system prompt, off-topic)", all(re.search(p, appsrc, re.I) for p in (r"delete|modify|drop", r"system prompt", r"off.?topic|out.?of.?scope|unrelated")))
check("code: rank ties broken by chunk id in the reranker", re.search(r"chunk_id", retsrc) is not None and re.search(r"sort|sorted", retsrc) is not None, "look at reranker.py")
check("code: tune/holdout split used in an evaluation script", any(re.search(r"holdout", p.read_text(encoding='utf-8', errors='ignore')) for p in (ROOT / "evalsuite").glob("*.py") if p.name != "selfcheck.py"))
check("code: ECE and Brier written by you", any(re.search(r"def\s+(ece|expected_calibration_error)", p.read_text(encoding='utf-8', errors='ignore'), re.I) and re.search(r"def\s+brier", p.read_text(encoding='utf-8', errors='ignore'), re.I) for p in [q for q in (ROOT / "evalsuite").glob("*.py") if q.name != "selfcheck.py"] + list((ROOT / "retrieve").glob("*.py"))))

# ---------------------------------------------------------------- B. secrets / personal data
section("B. Secrets and personal data in files that would be submitted")
SKIP_DIRS = {"venv", ".git", "__pycache__", "node_modules", "gisdata"}
hits, personal = [], []
vals = [v for k in ("ELASTIC_PASSWORD", "RAG_APP_PASSWORD", "GEOSERVER_RAG_PASSWORD", "GEOSERVER_ADMIN_PASSWORD") if (v := os.environ.get(k)) and len(v) >= 5]
pat = re.compile(r"(?i)(password|passwd|api[_-]?key|secret|token)\s*[:=]\s*['\"]?([A-Za-z0-9!@#$%^&*._\-]{6,})")
for p in ROOT.rglob("*"):
    if not p.is_file() or any(part in SKIP_DIRS for part in p.parts) or p.name == ".env": continue
    if p.suffix.lower() not in (".py", ".md", ".txt", ".json", ".jsonl", ".yml", ".yaml", ".ps1", ".sh", ".log", ".cfg", ".ini", ".csv"): continue
    if p.stat().st_size > 5_000_000: continue
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        rel = p.relative_to(ROOT)
        if any(v in line for v in vals): hits.append(f"{rel}:{i} (contains an actual password from your environment)")
        m = pat.search(line)
        if m and not re.search(r"environ|getenv|\$env:|<|\*\*\*|your-|xxxx|\{", line, re.I) and m.group(2).lower() not in ("none", "true", "false"):
            hits.append(f"{rel}:{i} (looks like a hard-coded {m.group(1).lower()})")
        if re.search(r"[A-Za-z]:\\Users\\|file:///[A-Za-z]:|/Users/[a-z]+/", line): personal.append(f"{rel}:{i}")
check("no passwords/keys in submitted files", not hits, "; ".join(hits[:8]) + (f" ... (+{len(hits) - 8} more)" if len(hits) > 8 else ""))
check("no personal local paths (C:\\Users\\<name>) in files", not personal, "; ".join(personal[:8]))
check(".env is not inside any folder that will be zipped (delete or exclude it)", True, "make sure the zip script excludes .env (it is in the project root here)")

# ---------------------------------------------------------------- C. live services
section("C. Live checks (Elasticsearch, GeoServer)")
pw = os.environ.get("RAG_APP_PASSWORD")
if not pw: skip("Elasticsearch checks", "set $env:RAG_APP_PASSWORD")
else:
    A = ("rag_app_user", pw)
    def g(path, **kw): return requests.get(ES + path, auth=A, timeout=30, **kw)
    try:
        r = g("/_cluster/health"); check("rag_app_user is refused cluster-level access (403)", r.status_code == 403, f"HTTP {r.status_code}")
        for idx in ("east_india_chunks", "east_india_entities", "east_india_relations"):
            r = g(f"/{idx}/_count"); check(f"index {idx} readable by rag_app_user", r.status_code == 200, f"count={r.json().get('count') if r.ok else r.status_code}")
        r = requests.put(ES + "/some_other_index", auth=A, timeout=30); check("rag_app_user cannot create other indices", r.status_code in (401, 403), f"HTTP {r.status_code}")
        m = g("/east_india_chunks/_mapping").json()["east_india_chunks"]["mappings"]; pr = m.get("properties", {}); emb = pr.get("embedding", {})
        check("chunks: dense_vector with dims, similarity, HNSW m and ef_construction", emb.get("type") == "dense_vector" and emb.get("dims") and emb.get("similarity")
              and emb.get("index_options", {}).get("m") and emb.get("index_options", {}).get("ef_construction"), json.dumps({k: emb.get(k) for k in ("dims", "similarity", "index_options")}))
        check("chunks: English analyser on text", pr.get("text", {}).get("analyzer") == "english", str(pr.get("text", {}).get("analyzer")))
        check("chunks: keyword, date, integer metadata + geo_point locations", {"keyword", "date", "integer", "geo_point"} <= {v.get("type") for v in pr.values()}, str(sorted({v.get('type') for v in pr.values()})))
        check("chunks: mapping is explicit (dynamic strict/false)", m.get("dynamic") in ("strict", "false", False), str(m.get("dynamic")))
        e = g("/east_india_entities/_mapping").json()["east_india_entities"]["mappings"]; ep = json.dumps(e)
        check("entities: nested mentions, geo_point, date, integer, keyword, text in the mapping", all(t in ep for t in ('"nested"', '"geo_point"', '"date"', '"integer"', '"keyword"', '"text"')))
        rp = g("/east_india_relations/_mapping").json()["east_india_relations"]["mappings"].get("properties", {})
        need = {"source_id", "target_id", "type", "weight", "evidence"}
        check("relations: edge fields source_id, target_id, type, weight, evidence", need <= set(rp), f"fields present: {sorted(rp)}; missing: {sorted(need - set(rp))}")
        tf = "type" if "type" in rp else ("rel_type" if "rel_type" in rp else None)
        if tf:
            a = requests.post(ES + "/east_india_relations/_search", auth=A, json={"size": 0, "aggs": {"t": {"terms": {"field": tf, "size": 30}}, "w": {"stats": {"field": "weight"}}}}, timeout=30).json()["aggregations"]
            types = {b["key"] for b in a["t"]["buckets"]}; need_t = {"affiliated_with", "applies_to", "mentions", "located_in", "near", "supersedes"}
            check("relations: the six required relation types exist", need_t <= types, f"found {sorted(types)}")
            check("relations: weights between 0 and 1", a["w"]["min"] is not None and 0 <= a["w"]["min"] and a["w"]["max"] <= 1, f"min {a['w']['min']} max {a['w']['max']}")
    except Exception as ex: check("Elasticsearch checks ran", False, repr(ex)[:200])
gp = os.environ.get("GEOSERVER_RAG_PASSWORD")
q = dict(service="WFS", version="1.0.0", request="GetFeature", typeName="east_india:schools", outputFormat="application/json", maxFeatures=1)
try:
    r = requests.get(f"{GS}/east_india/ows", params=q, timeout=30); check("GeoServer: request without login is refused", r.status_code in (401, 403, 404), f"HTTP {r.status_code}")
    if gp:
        r = requests.get(f"{GS}/east_india/ows", params=q, auth=("rag_geo", gp), timeout=30); check("GeoServer: rag_geo can fetch schools as GeoJSON", r.status_code == 200 and "features" in r.text, f"HTTP {r.status_code}")
        r = requests.get(f"{GS}/rest/workspaces.json", auth=("rag_geo", gp), timeout=30); check("GeoServer: rag_geo cannot use the admin REST API", r.status_code in (401, 403), f"HTTP {r.status_code}")
    else: skip("GeoServer rag_geo checks", "set $env:GEOSERVER_RAG_PASSWORD")
except Exception as ex: check("GeoServer reachable", False, repr(ex)[:150])

# ---------------------------------------------------------------- D. metric functions
section("D. Your own metric functions (hand-calculated example)")
mf = ROOT / "evalsuite" / "metrics.py"
if mf.exists():
    try:
        spec = importlib.util.spec_from_file_location("m", mf); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        ranked, j = ["a", "b", "c"], {"a": 2, "c": 1}
        check("precision_at_k(['a','b','c'], {a:2,c:1}, 2) = 0.5", abs(m.precision_at_k(ranked, j, 2) - 0.5) < 1e-9)
        check("recall_at_k(..., 3) = 1.0", abs(m.recall_at_k(ranked, j, 3) - 1.0) < 1e-9)
        check("mrr(['b','a'], {a:2}) = 0.5", abs(m.mrr(["b", "a"], {"a": 2}) - 0.5) < 1e-9)
        v = m.ndcg_at_k(ranked, j, 3); check("ndcg_at_k = 0.9502 (linear gain) or 0.9639 (2^g-1 gain)", abs(v - 0.9502) < 1e-3 or abs(v - 0.9639) < 1e-3, f"got {v:.4f}")
    except Exception as ex: check("metrics.py works with signature (ranked, judg, k)", False, repr(ex)[:200])

# ---------------------------------------------------------------- E. documentation coverage
section("E. What PROJECT DOCUMENT.md covers (statement requirements)")
dp = ROOT / "PROJECT DOCUMENT.md"
if dp.exists():
    t = dp.read_text(encoding="utf-8", errors="ignore"); tl = t.lower()
    items = [
      ("Algorithm notes heading", r"^#+\s*.*algorithm notes", re.M),
      ("BM25 / HNSW / RRF / cross-encoder / softmax with temperature / greedy decoding / graph traversal explained", None, None),
      ("T5: table for k = 1, 3, 5, 10 AND 20 and a chosen default k", r"@\s*20\b.*@\s*3\b|@\s*3\b.*@\s*20\b|\|\s*k\s*\|.*\b20\b", None),
      ("T5: tuning on the 20 'tune' queries, result on the 10 'holdout' queries", r"tun(ed|ing)[^\n]{0,250}\b(rrf|num_candidates|weights?|constant)\b.{0,1500}holdout[^\n]{0,250}\d\.\d", None),
      ("T5: geo_distance filter and metadata filter", r"geo_distance", None),
      ("T6: rerank N = 10, 20 and 50 with extra time per query", r"\bN\s*=\s*50|top[- ]?50|rerank(ing)?\s*(top|n)\s*[:=]?\s*50", None),
      ("T6: batch size effect and deterministic tie-break", r"batch size.{0,500}tie|tie.{0,500}batch size|tie-?break", None),
      ("T7: several signals combined (margin, keyword/vector agreement, log-probs or sampled agreement)", r"margin", None),
      ("T7: logistic regression fitted on the fit split", r"logistic regression", None),
      ("T7: reliability table (5 bins), ECE and Brier score", r"reliability.{0,800}\bECE\b.{0,800}brier|brier.{0,800}\bECE\b", None),
      ("T7: temperature chosen by minimising log loss on fit, effect on test ECE", r"log[- ]?loss", None),
      ("T7: coverage and accuracy at five thresholds, >= 90% accuracy rule", r"coverage", None),
      ("T7: sampling temperature 0 / 0.3 / 0.7 x 5 runs consistency table, top-p/top-k", r"0\.3.{0,400}consisten|consisten.{0,400}0\.3", None),
      ("T8: answers as validated JSON (answer, citations, confidence, entities_used) + cited chunk check", r"entities_used", None),
      ("T8: refusals for delete/modify data, secrets/system prompt, off-topic", r"system prompt.{0,600}(delete|modify)|(delete|modify).{0,600}system prompt", None),
      ("T8: three failure analyses (stage, cause, fix) and one fix with before/after", r"before.{0,300}after|after.{0,300}before", None),
      ("T8: architecture diagram referenced", r"architecture diagram|diagram/", None),
      ("Disclosure of the coding assistant/IDE agent you used (e.g. Antigravity)", r"antigravity|cursor|copilot|gemini", None),
    ]
    for name, pattern, flags in items:
        if pattern is None:
            terms = ["bm25", "hnsw", "reciprocal rank fusion", "cross-encoder", "softmax", "greedy", "graph traversal"]
            miss = [x for x in terms if x not in tl]; check(name, not miss, f"missing: {miss}")
        else: check(name, re.search(pattern, t, (flags or 0) | re.I | re.S) is not None)
    check("no leftover instructions/placeholders (paste, TODO, [fill)", not re.search(r"paste the|TODO|\[fill|<your", t, re.I))
    check("no absolute local paths or file:/// links", not re.search(r"[A-Za-z]:\\Users|file:///", t))
    nums = re.findall(r"\b0?\.\d{3}\b", t); print(f"INFO  {len(nums)} three-decimal numbers in the document: every one must come from a file in evidence/ (you will be asked in the viva)")

print(f"\nSUMMARY: {sum(R)} passed, {len(R) - sum(R)} failed")