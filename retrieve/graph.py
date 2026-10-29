"""In-memory knowledge graph built from east_india_entities + east_india_relations.
neighbours(): breadth-first traversal (1-2 hops) with cycle protection, fan-out limits and the path to every entity.
expand(): chunk ids linked to entities reached within two hops (used by graph-expanded hybrid search in Task 5).

Usage:
    from graph import KnowledgeGraph, load
    entities, relations = load("es")
    G = KnowledgeGraph(entities, relations)
"""
import sys, json, math
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from common import *

def load(source="es"):
    if source == "files":
        return json.load(open(OUT / "entities.json")), json.load(open(OUT / "relations.json"))
    e = es("POST", "/east_india_entities/_search", json={"size": 1000, "query": {"match_all": {}}}).json()["hits"]["hits"]
    r = es("POST", "/east_india_relations/_search", json={"size": 10000, "query": {"match_all": {}}}).json()["hits"]["hits"]
    return [x["_source"] for x in e], [x["_source"] for x in r]

class KnowledgeGraph:
    def __init__(self, entities, relations):
        self.nodes = {e["entity_id"]: e for e in entities}
        agg = {}
        for r in relations:                                    # parallel evidence -> one edge, weights combined by noisy-OR
            a = agg.setdefault((r["source_id"], r["target_id"], r["type"]), dict(c=1.0, ev=set()))
            a["c"] *= 1.0 - r["weight"]; a["ev"].add(r["evidence"])
        self.edges = {k: dict(weight=round(1 - a["c"], 4), evidence=sorted(a["ev"])) for k, a in agg.items()}
        self.adj = defaultdict(list)
        for (u, v, t), a in self.edges.items():
            self.adj[u].append(dict(to=v, type=t, dir="out", weight=a["weight"], evidence=a["evidence"]))
            self.adj[v].append(dict(to=u, type=t, dir="in", weight=a["weight"], evidence=a["evidence"]))
        for u in self.adj: self.adj[u].sort(key=lambda x: (-x["weight"], x["to"], x["type"]))   # deterministic order
        self.chunks_of = {k: sorted({m["chunk_id"] for m in e["mentions"] if m.get("chunk_id")}) for k, e in self.nodes.items()}
    def name(self, eid): return self.nodes[eid]["canonical_name"] if eid in self.nodes else eid
    def degree(self, eid): return len({x["to"] for x in self.adj.get(eid, [])})

    def neighbours(self, entity_id, hops=1, types=None, fanout=10, max_frontier=50, min_weight=0.0, no_sibling=False, with_cycles=False):
        """Breadth-first search. Returns {entity_id: {hops, score, path}} where path is the list of steps
        (from, relation, to, weight, evidence) used to reach it and score = product of edge weights on the path.
        - cycles: a node is never revisited (visited set), so loops such as school -> board -> school cannot repeat;
        - fan-out: at most `fanout` neighbours are taken from any node, and at most `max_frontier` nodes are expanded per hop;
        - no_sibling: do not bounce straight back along the same relation type (school -> board -> OTHER schools,
          circular -> OTHER boards): those "sibling" hops are the main source of noise;
        - the heaviest edges are taken first, ties broken by id, so the result is deterministic."""
        visited, found, cycles = {entity_id}, {}, []
        frontier = [(entity_id, 1.0, [])]
        for h in range(1, hops + 1):
            nxt = []
            for node, score, path in sorted(frontier, key=lambda x: (-x[1], x[0]))[:max_frontier]:
                taken = 0
                for ed in self.adj.get(node, []):
                    if types and ed["type"] not in types: continue
                    if ed["weight"] < min_weight: continue
                    if ed["to"] in visited:                       # cycle / second route to a known node: do not re-expand it,
                        if path and ed["to"] == path[-1]["frm"]: continue      # (ignore the edge we just came along)
                        if with_cycles:                           # but keep it as extra evidence
                            cycles.append(dict(path=path + [dict(frm=node, rel=ed["type"] + ("" if ed["dir"] == "out" else "^-1"), to=ed["to"], weight=ed["weight"], evidence=ed["evidence"])], to=ed["to"]))
                        continue
                    if no_sibling and path:
                        last = path[-1]; last_dir = "in" if last["rel"].endswith("^-1") else "out"
                        if ed["type"] == last["rel"].replace("^-1", "") and ed["dir"] != last_dir: continue
                    if taken >= fanout: break
                    taken += 1; visited.add(ed["to"])
                    step = dict(frm=node, rel=ed["type"] + ("" if ed["dir"] == "out" else "^-1"), to=ed["to"], weight=ed["weight"], evidence=ed["evidence"])
                    sc = score * ed["weight"]; found[ed["to"]] = dict(hops=h, score=round(sc, 4), path=path + [step])
                    nxt.append((ed["to"], sc, path + [step]))
            frontier = nxt
        return (found, cycles) if with_cycles else found

    def path_str(self, path):
        return " ".join(f"{self.name(s['frm'])} -[{s['rel']} {s['weight']:.2f}]->" for s in path) + (f" {self.name(path[-1]['to'])}" if path else "")

    def expand(self, query_entities, hops=2, types=None, fanout=20, max_frontier=30, min_weight=0.5, hub_degree=25, hop_decay=0.7, no_sibling=True, include_seed_chunks=True):
        """Chunk ids linked to entities reached within `hops` hops, best first: [{chunk_id, score, entity, hops, path}].
        Chunk sources: (a) the relation evidence chunk on every step of the path (the sentence that states the relation);
        (b) the mention chunks of each reached entity - but NOT for hub entities (degree > hub_degree, e.g. a Board mentioned
        in 56 places), because that would pull in half the corpus.
        Scores: product of edge weights x hop_decay^(hops-1), so evidence two hops away ranks below direct evidence."""
        best = {}
        def put(cid, score, ent, h, path):
            if not cid or cid == "geoserver": return
            if cid not in best or score > best[cid]["score"] or (score == best[cid]["score"] and ent < best[cid]["entity"]):
                best[cid] = dict(chunk_id=cid, score=round(score, 4), entity=ent, hops=h, path=self.path_str(path))
        for q in query_entities:
            if q not in self.nodes: continue
            if include_seed_chunks and self.degree(q) <= hub_degree:
                for c in self.chunks_of[q]: put(c, 1.0, q, 0, [])
            found, cycles = self.neighbours(q, hops, types, fanout, max_frontier, min_weight, no_sibling, with_cycles=True)
            for cy in cycles:                                     # evidence stated on edges that close a cycle
                run = 1.0
                for i, st in enumerate(cy["path"]):
                    run *= st["weight"]
                    if i == len(cy["path"]) - 1:
                        for ev in st["evidence"]: put(ev, run * hop_decay ** i, cy["to"], len(cy["path"]), cy["path"])
            for v, info in found.items():
                run = 1.0
                for i, st in enumerate(info["path"]):
                    run *= st["weight"]
                    for ev in st["evidence"]: put(ev, run * hop_decay ** i, v, info["hops"], info["path"])
                if self.degree(v) <= hub_degree:
                    for c in self.chunks_of[v]: put(c, info["score"] * hop_decay ** (info["hops"] - 1), v, info["hops"], info["path"])
        return sorted(best.values(), key=lambda x: (-x["score"], x["chunk_id"]))

    def measures(self):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        for (u, v, t), a in self.edges.items():
            w = a["weight"]
            if G.has_edge(u, v): w = 1 - (1 - G[u][v]["weight"]) * (1 - w)
            G.add_edge(u, v, weight=w)
        comps = list(nx.connected_components(G))
        deg = sorted(G.degree(), key=lambda x: (-x[1], x[0]))[:5]
        pr = sorted(nx.pagerank(G, weight="weight").items(), key=lambda x: (-x[1], x[0]))[:5]
        return dict(nodes=G.number_of_nodes(), edges=G.number_of_edges(), typed_edges=len(self.edges), components=len(comps),
                    largest_component=max(map(len, comps)), top_degree=[(self.name(n), d) for n, d in deg], top_pagerank=[(self.name(n), round(p, 4)) for n, p in pr])

    def detect_entities(self, text):
        """Entities mentioned in a question: reuse the baseline rules, with a gazetteer taken from the entity index itself."""
        import extract_baseline as eb
        from resolve_and_index import sim, slug
        sch = {}; loc = []; rd = []
        for k, e in self.nodes.items():
            if e["type"] == "School":
                for n in [e["canonical_name"]] + e.get("aliases", []): sch[n] = k
            elif e["type"] == "Locality": loc.append(e["canonical_name"])
            elif e["type"] == "Road": rd.append(e["canonical_name"])
        names = sorted(sch, key=len, reverse=True)
        out = []
        for m in eb.extract(text, (names, sorted(loc, key=len, reverse=True), sorted(rd, key=len, reverse=True))):
            t, c = m["type"], m["canonical"]
            if t == "School":
                best = max(sch, key=lambda n: sim(n, c)); eid = sch[best] if sim(best, c) >= 0.85 else None
            elif t in ("Board", "Circular", "Rule", "Locality", "Road"): eid = f"{t.lower()}:{slug(c)}"
            else: eid = None
            if eid in self.nodes and eid not in [x[0] for x in out]: out.append((eid, t, m["text"]))
        return out
