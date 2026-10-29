"""Chunking: one chunk per paragraph (documents are structured: heading, rule, table...), with a sliding window
(MAX_CHARS, OVERLAP) only for paragraphs that are too long. Every chunk keeps its page and character span for citations.
MAX_CHARS=1000 because ~4 chars/token gives ~250 tokens, which fits the smaller embedding model (256-token limit).

Usage:
    python ingest/chunker.py
"""
import re, json
from common import *

MAX_CHARS, OVERLAP = 1000, 100

def split_paragraphs(text):
    pos, out = 0, []
    for part in text.split("\n\n"):
        out.append((pos, pos + len(part), part)); pos += len(part) + 2
    return out

def windows(start, end, text):
    if end - start <= MAX_CHARS: return [(start, end, text)]
    out, s = [], 0
    while s < len(text):
        e = min(len(text), s + MAX_CHARS)
        if e < len(text):                                  # cut at a whitespace, not in the middle of a word
            cut = text.rfind(" ", s + MAX_CHARS - 150, e)
            e = cut if cut > s else e
        out.append((start + s, start + e, text[s:e]))
        if e >= len(text): break
        s = e - OVERLAP
    return out

def entity_metadata():
    """doc boards, issue date, circular number and school ids/locations (from the resolved entities of Task 2)."""
    f = OUT / "entities.json"
    meta = dict(doc_boards={}, doc_date={}, doc_number={}, school_mentions=[])
    if not f.exists(): return meta
    first_date = {}
    for e in json.load(open(f)):
        a = e.get("attributes", {})
        if e["type"] == "Board":
            for m in e["mentions"]: meta["doc_boards"].setdefault(m["doc_id"], set()).add(e["canonical_name"])
        if e["type"] == "Circular" and a.get("doc_id"):
            meta["doc_number"][a["doc_id"]] = a.get("number")
            if a.get("date"): meta["doc_date"][a["doc_id"]] = a["date"]
        if e["type"] == "Date":
            for m in e["mentions"]:
                if m["doc_id"] not in first_date or m["start"] < first_date[m["doc_id"]][0]:
                    first_date[m["doc_id"]] = (m["start"], e["canonical_name"])
        if e["type"] == "School":
            for m in e["mentions"]:
                meta["school_mentions"].append((m["doc_id"], m["start"], e["entity_id"], a.get("location")))
    for d, (_, iso) in first_date.items(): meta["doc_date"].setdefault(d, iso)
    return meta

def build_chunks():
    docs, ref, meta = load_docs(), load_chunks_ref(), entity_metadata()
    ref_by = {(c["doc_id"], c["start"]): c["chunk_id"] for c in ref}
    out = []
    for did, d in docs.items():
        kind = "circular" if did.startswith("CIR") else ("guideline" if did.startswith("GUIDE") else "note")
        for pi, (s, e, txt) in enumerate(split_paragraphs(d["text"])):
            base = ref_by[(did, s)]
            for wi, (ws, we, wt) in enumerate(windows(s, e, txt)):
                cid = base if wi == 0 and e - s <= MAX_CHARS else f"{base}#{wi + 1}"
                sids, locs = [], []
                for (md, ms, sid, loc) in meta["school_mentions"]:
                    if md == did and ws <= ms < we:
                        if sid not in sids: sids.append(sid)
                        if loc: locs.append(loc)
                c = dict(chunk_id=cid, doc_id=did, doc_type=kind, file=d["file"], page=1, para_index=pi, start=ws, end=we,
                         text=wt, char_count=len(wt), boards=sorted(meta["doc_boards"].get(did, [])), school_ids=sids, school_locations=locs)
                if meta["doc_date"].get(did): c["issue_date"] = meta["doc_date"][did]
                if meta["doc_number"].get(did): c["circular_number"] = meta["doc_number"][did]
                out.append(c)
    return out

if __name__ == "__main__":
    ch = build_chunks(); L = [c["char_count"] for c in ch]
    print(f"chunks: {len(ch)} | chars min/mean/max: {min(L)}/{sum(L) / len(L):.0f}/{max(L)} | split windows: {sum('#' in c['chunk_id'] for c in ch)}")
    print(f"with boards: {sum(bool(c['boards']) for c in ch)} | with school ids: {sum(bool(c['school_ids']) for c in ch)} | with date: {sum('issue_date' in c for c in ch)}")
    json.dump(ch, open(OUT / "chunks.json", "w"), indent=1); print("saved ingest/out/chunks.json")
