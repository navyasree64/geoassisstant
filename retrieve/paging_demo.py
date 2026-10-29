"""Paging beyond page 1 with a point-in-time (PIT) and search_after.   Usage: python retrieve/paging_demo.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from common import *

pit = es("POST", "/east_india_chunks/_pit?keep_alive=2m").json()["id"]
body = {"size": 10, "track_total_hits": True, "_source": ["chunk_id"], "pit": {"id": pit, "keep_alive": "2m"},
        "query": {"match": {"text": "school admission seats cut-off"}},
        "sort": [{"_score": "desc"}, {"_shard_doc": "asc"}]}            # _shard_doc = cheap, unique tie-breaker for PIT
page1 = es("POST", "/_search", json=body).json()
print("total matches:", page1["hits"]["total"]["value"])
print("PAGE 1:", [(h["_source"]["chunk_id"], round(h["_score"], 2)) for h in page1["hits"]["hits"]])
last = page1["hits"]["hits"][-1]["sort"]
body["search_after"] = last
page2 = es("POST", "/_search", json=body).json()
print("PAGE 2:", [(h["_source"]["chunk_id"], round(h["_score"], 2)) for h in page2["hits"]["hits"]])
es("DELETE", "/_pit", json={"id": pit}, expect=None)
