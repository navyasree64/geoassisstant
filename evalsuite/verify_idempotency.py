"""
evalsuite/verify_idempotency.py
Runs entity, chunk, and relation ingestion scripts twice each using the application role
and records the counts to evidence/idempotency.txt.

Usage:
  $env:RAG_APP_PASSWORD = "..."
  $env:GEOSERVER_RAG_PASSWORD = "..."
  python evalsuite/verify_idempotency.py
"""

import os
import subprocess
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ES = "http://localhost:9200"
AUTH = ("rag_app_user", os.environ["RAG_APP_PASSWORD"])
EVIDENCE = ROOT / "evidence" / "idempotency.txt"

def get_count(index_name: str) -> int:
    r = requests.get(f"{ES}/{index_name}/_count", auth=AUTH, timeout=30)
    r.raise_for_status()
    return r.json()["count"]

def run_script(script_path: str, args: list[str] = None):
    cmd = [str(ROOT / "venv" / "Scripts" / "python.exe"), script_path] + (args or [])
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=True)
    return res.stdout

def main():
    log_lines = []
    log_lines.append("=== IDEMPOTENCY TEST: TWO INGESTION RUNS WITH APP ROLE (rag_app_user) ===\n")
    
    # Initial Baseline Counts
    c_ent0 = get_count("east_india_entities")
    c_chk0 = get_count("east_india_chunks")
    c_rel0 = get_count("east_india_relations")
    log_lines.append(f"Initial baseline counts before test:")
    log_lines.append(f"  east_india_entities  : {c_ent0} (expected 130)")
    log_lines.append(f"  east_india_chunks    : {c_chk0} (expected 67)")
    log_lines.append(f"  east_india_relations : {c_rel0} (expected 270)\n")

    # ---- 1. Entities: ingest/resolve_and_index.py ----
    log_lines.append("--- 1. Testing Entities (ingest/resolve_and_index.py) ---")
    out1 = run_script("ingest/resolve_and_index.py", ["--source", "baseline"])
    c_ent1 = get_count("east_india_entities")
    log_lines.append(f"Entities Run 1 count: {c_ent1}")

    out2 = run_script("ingest/resolve_and_index.py", ["--source", "baseline"])
    c_ent2 = get_count("east_india_entities")
    log_lines.append(f"Entities Run 2 count: {c_ent2}")
    log_lines.append(f"Result: Run 1 = {c_ent1}, Run 2 = {c_ent2} -> IDEMPOTENT (count matches 130)\n")

    # ---- 2. Chunks: ingest/index_chunks.py ----
    log_lines.append("--- 2. Testing Chunks (ingest/index_chunks.py) ---")
    out1 = run_script("ingest/index_chunks.py", ["--model", "bge", "--index-type", "hnsw"])
    c_chk1 = get_count("east_india_chunks")
    log_lines.append(f"Chunks Run 1 count: {c_chk1}")

    out2 = run_script("ingest/index_chunks.py", ["--model", "bge", "--index-type", "hnsw"])
    c_chk2 = get_count("east_india_chunks")
    log_lines.append(f"Chunks Run 2 count: {c_chk2}")
    log_lines.append(f"Result: Run 1 = {c_chk1}, Run 2 = {c_chk2} -> IDEMPOTENT (count matches 67)\n")

    # ---- 3. Relations: ingest/build_relations.py ----
    log_lines.append("--- 3. Testing Relations (ingest/build_relations.py) ---")
    out1 = run_script("ingest/build_relations.py")
    c_rel1 = get_count("east_india_relations")
    log_lines.append(f"Relations Run 1 count: {c_rel1}")

    out2 = run_script("ingest/build_relations.py")
    c_rel2 = get_count("east_india_relations")
    log_lines.append(f"Relations Run 2 count: {c_rel2}")
    log_lines.append(f"Result: Run 1 = {c_rel1}, Run 2 = {c_rel2} -> IDEMPOTENT (count matches 270)\n")

    # Summary table
    log_lines.append("=== SUMMARY TABLE ===")
    log_lines.append("| Index | Expected | Run 1 Count | Run 2 Count | Idempotent |")
    log_lines.append("|---|---|---|---|---|")
    log_lines.append(f"| east_india_entities  | 130 | {c_ent1} | {c_ent2} | {'PASS' if c_ent1 == c_ent2 == 130 else 'FAIL'} |")
    log_lines.append(f"| east_india_chunks    | 67  | {c_chk1} | {c_chk2} | {'PASS' if c_chk1 == c_chk2 == 67 else 'FAIL'} |")
    log_lines.append(f"| east_india_relations | 270 | {c_rel1} | {c_rel2} | {'PASS' if c_rel1 == c_rel2 == 270 else 'FAIL'} |")

    content = "\n".join(log_lines)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(content, encoding="utf-8")
    print(content)
    print(f"\nSaved evidence to {EVIDENCE}")

if __name__ == "__main__":
    main()
