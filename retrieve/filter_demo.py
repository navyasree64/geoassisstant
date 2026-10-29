"""Task 5 demonstration: filtered and geo-filtered retrieval.
Demonstrates:
  1. Metadata filter: queries with a board filter (e.g. boards="WBBSE")
  2. Geo filter: queries with geo_distance filter (lat, lon, distance="3km")

Usage:
  python retrieve/filter_demo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieve"))
from retrieve import keyword, vector, hybrid, fetch_chunks

def main():
    print("=== TASK 5: METADATA AND GEO-FILTERING DEMO ===\n")

    # 1. Metadata Filter Demo
    query1 = "Which schools have revised admission guidelines?"
    meta = {"boards": "WBBSE"}
    print(f"--- 1. Metadata Filtered Search ---")
    print(f"Query: {query1!r}")
    print(f"Filter: boards = 'WBBSE'")
    results_meta = keyword(query1, k=5, metadata_filter=meta)
    chunks_meta = fetch_chunks([cid for cid, _ in results_meta])
    print(f"Top-{len(results_meta)} Results:")
    for i, (cid, score) in enumerate(results_meta, 1):
        info = chunks_meta.get(cid, {})
        boards = info.get("boards", [])
        print(f"  {i}. [{score:.3f}] {cid:25s} | Boards: {boards}")

    # 2. Geo Filter Demo (Ballygunge: lat 22.528, lon 88.365, distance 3km)
    query2 = "What are the school options near Ballygunge?"
    geo = {"lat": 22.528, "lon": 88.365, "distance": "3km"}
    print(f"\n--- 2. Geo-Distance Filtered Search ---")
    print(f"Query: {query2!r}")
    print(f"Geo Filter: school_locations within 3km of ({geo['lat']}, {geo['lon']})")
    results_geo = hybrid(query2, k=5, geo_filter=geo)
    chunks_geo = fetch_chunks([cid for cid, _ in results_geo])
    print(f"Top-{len(results_geo)} Results (RRF fused):")
    for i, (cid, score) in enumerate(results_geo, 1):
        info = chunks_geo.get(cid, {})
        locs = info.get("school_locations", [])
        print(f"  {i}. [{score:.4f}] {cid:25s} | Locations: {locs}")

if __name__ == "__main__":
    main()
