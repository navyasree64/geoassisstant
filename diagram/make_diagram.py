"""Architecture diagram for the East India Geo-Assistant (matplotlib only). Usage: python diagram/make_diagram.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, INK, MUTED, ARROW = "#F8FAFC", "#0F172A", "#475569", "#475569"
C = dict(src=("#DBEAFE", "#3B82F6"), proc=("#DCFCE7", "#16A34A"), store=("#FEF3C7", "#D97706"),
         ret=("#EDE9FE", "#7C3AED"), gate=("#FCE7F3", "#DB2777"), ok=("#D1FAE5", "#059669"), no=("#FEE2E2", "#DC2626"), mon=("#F1F5F9", "#64748B"))
fig, ax = plt.subplots(figsize=(18, 11), dpi=200)
ax.set_xlim(0, 160); ax.set_ylim(0, 100); ax.axis("off"); fig.patch.set_facecolor(BG)

def container(x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.5", fc="#FFFFFF", ec="#CBD5E1", lw=1.6))
    ax.text(x0 + 1.5, y1 - 2.2, title, fontsize=11, fontweight="bold", color=INK, va="center")

def box(cx, cy, w, h, title, sub, kind):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc=fc, ec=ec, lw=1.5))
    n = sub.count("\n") + 1
    ax.text(cx, cy + h / 2 - 2.0, title, fontsize=8, fontweight="bold", color=INK, ha="center", va="center")
    ax.text(cx, cy - 1.3 + (n - 2) * 0.0, sub, fontsize=6.6, color=MUTED, ha="center", va="center", linespacing=1.35)
    return dict(l=(cx - w / 2, cy), r=(cx + w / 2, cy), t=(cx, cy + h / 2), b=(cx, cy - h / 2))

def arrow(p, q, label=None, lpos=None, color=ARROW, ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=1.3, color=color, ls=ls, shrinkA=0, shrinkB=0))
    if label: ax.text(*(lpos or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 1.3)), label, fontsize=6.2, color=color, ha="center", va="center", style="italic")

def elbow(p, mid, q, label=None, lpos=None):
    ax.plot([p[0], mid[0]], [p[1], mid[1]], color=ARROW, lw=1.3)
    arrow(mid, q, label, lpos)

ax.text(80, 98.3, "East India Geo-Assistant: architecture and data flow", fontsize=15, fontweight="bold", color=INK, ha="center", va="center")

# ---------------------------------------------------------------- ingestion
container(2, 52, 158, 95.5, "INGESTION  (offline, batch build)")
W, H = 27, 10.5
docs = box(17, 85, W, H, "Documents pack", "14 synthetic circulars,\nguideline and notes", "src")
geo = box(17, 62.5, W, H, "GeoServer WFS", "schools, roads, landuse\nPort 8080 | role rag_geo", "src")
chk = box(48, 85, W, H, "Chunking", "paragraph chunks (67)\npage + character spans", "proc")
emb = box(81, 85, W, H, "Embedding", "bge-small-en-v1.5\n384-d bi-encoder", "proc")
ext = box(48, 73.5, W, H, "Entity extraction", "rules + gazetteer\n(LLM extractor: evaluated only)", "proc")
res = box(81, 73.5, W, H, "Resolution + linking", "similarity 0.85 + guard words\nlinked to osm_id", "proc")
gra = box(81, 62.5, W, H, "Graph relations", "6 relation types, noisy-OR\nnear <= 300 m (EPSG:32646)", "proc")
ix_c = box(138, 85, 36, H, "east_india_chunks", "HNSW m=16, ef_construction=100\nPort 9200", "store")
ix_e = box(138, 73.5, 36, H, "east_india_entities", "130 entities (geo_point, nested)\nPort 9200", "store")
ix_r = box(138, 62.5, 36, H, "east_india_relations", "270 weighted edges\nPort 9200", "store")
kib = box(138, 55.0, 36, 4.2, "Kibana  Port 5601", "", "mon")
ax.text(138, 90.9, "Elasticsearch 8.15.0 | written by role rag_app_user", fontsize=6.8, color="#B45309", ha="center", fontweight="bold")
arrow(docs["r"], chk["l"]); arrow(chk["r"], emb["l"]); arrow(emb["r"], ix_c["l"], "bulk, deterministic _id", (110, 86.6))
arrow(chk["b"], ext["t"]); arrow(ext["r"], res["l"]); arrow(res["r"], ix_e["l"], "bulk", (110, 75.0))
arrow(res["b"], gra["t"]); arrow(gra["r"], ix_r["l"], "bulk", (110, 64.0))
ax.plot([17, 17], [67.75, 73.5], color=ARROW, lw=1.3); arrow((17, 73.5), ext["l"], "feature names (gazetteer)", (27.5, 75.0))
arrow(geo["r"], gra["l"], "geometries (near)", (47.8, 64.0))

# ---------------------------------------------------------------- retrieval
container(2, 3, 158, 50, "RETRIEVAL AND ANSWER  (online, end to end)")
cy = 26.5; w, h = 15.2, 12.5
q = box(11.5, cy, w, h, "User query", "Streamlit UI\nor command line", "ret")
sg = box(30.5, cy, w, h, "Safety guard", "refuses: modify data,\nsecrets, off-topic", "gate")
ed = box(49.5, cy, w, h, "Entity detection", "gazetteer + graph\naliases", "ret")
s1 = box(70.5, 40.0, 19.5, 7.8, "BM25", "Port 9200 | rag_app_user", "ret")
s2 = box(70.5, 28.5, 19.5, 7.8, "kNN dense search", "bge-small | Port 9200", "ret")
s3 = box(70.5, 17.0, 19.5, 7.8, "Graph expansion", "1-2 hops, no sibling hops", "ret")
rrf = box(91.5, cy, w, h, "RRF fusion", "reciprocal rank\nfusion, k = 60", "ret")
rr = box(110.0, cy, 16.5, h, "Cross-encoder", "ms-marco-MiniLM-L-6\ntop-N to top-k", "ret")
cg = box(128.5, cy, 15.5, h, "Confidence gate", "calibrated sigmoid,\nthreshold from Task 7", "gate")
llm = box(149, 38.5, 16.0, 13, "Local LLM", "qwen2.5:3b, Port 11434\nvalidated JSON +\ncitations", "ok")
ab = box(149, 15.0, 16.0, 13, "Abstain", "below threshold:\n\"I cannot answer this\nfrom the available sources.\"", "no")
rf = box(30.5, 9.0, 18, 7.0, "Refusal", "explicit message + category", "no")
arrow(q["r"], sg["l"]); arrow(sg["r"], ed["l"])
arrow(sg["b"], rf["t"], "refuse", (33.5, 14.0), color="#DC2626")
for s in (s1, s2, s3): arrow((ed["r"][0], cy), s["l"])
for s in (s1, s2, s3): arrow(s["r"], (rrf["l"][0], cy))
arrow(rrf["r"], rr["l"]); arrow(rr["r"], cg["l"])
arrow(cg["r"], llm["l"], "yes: confidence >= threshold", (133.0, 38.6), color="#059669")
arrow(cg["r"], ab["l"], "no: < threshold", (135.0, 17.0), color="#DC2626")
ax.text(88, 6.2, "The three searches read the Elasticsearch indices (Port 9200, role rag_app_user).  GeoServer (Port 8080, role rag_geo) is also read for school-location questions.", fontsize=6.8, color=MUTED, ha="center")
fig.savefig("architecture.png", facecolor=BG, bbox_inches="tight"); fig.savefig("architecture.pdf", facecolor=BG, bbox_inches="tight")
print("saved architecture.png and architecture.pdf")
