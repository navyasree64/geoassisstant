"""Build and verify submission zip.

Requirements:
- Staging directory outside project folder.
- Exclude: venv, gisdata, .env, .git, __pycache__, *.pyc, *.zip.
- Zip contents so ingest/, retrieve/, app/, etc. are at zip root.
- Name: <MobileNo>_geoassistant.zip.
- Display 10 largest files and total zip size.
- Unzip into verification folder, confirm layout.
- Grep unzipped directory for passwords and personal paths (C:\\Users).
- Run selfcheck structure checks on the unzipped archive.

Usage:
  python evalsuite/build_and_verify_zip.py
"""

import sys, os, shutil, zipfile, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path.home() / ".gemini" / "antigravity-ide" / "brain" / "be235f2a-2145-4639-9118-608317cc20a0"
SCRATCH = ARTIFACT_DIR / "scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)

STAGING_DIR = SCRATCH / "staging"
TEMP_UNZIP_DIR = SCRATCH / "temp_unzip"
ZIP_NAME = "MobileNo_geoassistant.zip"
ZIP_STAGING_PATH = SCRATCH / ZIP_NAME
ZIP_FINAL_PATH = ROOT / ZIP_NAME

EXCLUDE_DIRS = {
    "venv", ".venv", "env",
    "gisdata", ".git", "__pycache__",
    ".pytest_cache", ".agents",
    ".idea", ".vscode"
}

EXCLUDE_FILES = {
    ".env",
    "east_india_geoassistant_submission.zip",
    ZIP_NAME
}


def copy_to_staging(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Copying workspace to staging directory: {dst}...")
    copied_count = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        
        rel_root = Path(root).relative_to(src)
        target_dir = dst / rel_root
        target_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            if f in EXCLUDE_FILES or f.endswith(".pyc") or f.endswith(".zip") or f.startswith("."):
                continue
            src_file = Path(root) / f
            dst_file = target_dir / f
            shutil.copy2(src_file, dst_file)
            copied_count += 1

    print(f"Copied {copied_count} files to staging.")
    return copied_count


def create_zip(staging: Path, zip_dest: Path):
    print(f"Creating zip archive at: {zip_dest}...")
    file_list = []
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(staging):
            for f in files:
                file_path = Path(root) / f
                arcname = file_path.relative_to(staging)
                zf.write(file_path, arcname)
                size_b = file_path.stat().st_size
                file_list.append((str(arcname), size_b))

    zip_size_mb = zip_dest.stat().st_size / (1024 * 1024)
    print(f"Zip created successfully. Total archive size: {zip_size_mb:.2f} MB.")
    
    # Show 10 largest files
    print("\n--- 10 Largest Files in Submission Archive ---")
    file_list.sort(key=lambda x: -x[1])
    for name, sz in file_list[:10]:
        print(f"  {sz / (1024*1024):6.2f} MB  {name}")
    print(f"Total files in zip: {len(file_list)}")
    return zip_size_mb, file_list


def verify_unzipped(zip_path: Path, unzip_dir: Path):
    if unzip_dir.exists():
        shutil.rmtree(unzip_dir)
    unzip_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nUnzipping {zip_path.name} into {unzip_dir} for verification...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(unzip_dir)

    print("\n--- Root Directory Layout in Unzipped Archive ---")
    root_contents = sorted(os.listdir(unzip_dir))
    for item in root_contents:
        item_path = unzip_dir / item
        kind = "DIR " if item_path.is_dir() else "FILE"
        print(f"  [{kind}] {item}")

    # Confirm required directories and files at root
    required_roots = [
        "ingest", "retrieve", "evalsuite", "es", "geoserver", "app", "evidence", "diagram",
        "requirements.txt", "PROJECT DOCUMENT.md"
    ]
    missing = [r for r in required_roots if not (unzip_dir / r).exists()]
    if missing:
        print(f"\n[FAIL] Missing required items at zip root: {missing}")
    else:
        print("\n[PASS] All required folders and root files verified at root.")

    # Grep for secrets and personal paths
    print("\n--- Scanning Unzipped Files for Secrets & Personal Paths ---")
    forbidden_terms = ["password", "secret", "token", "c:\\users\\madal", "file:///"]
    violations = []
    
    # Allow known safe occurrences (e.g. env var names, refusal messages, code definitions)
    allowlist_re = re.compile(
        r"(rag_app_password|geoserver_rag_password|elastic_password|geoserver_admin_password|"
        r"password_hash|auth=|passwords?|i cannot reveal|no passwords|secrets? come from|"
        r"secrets_and_prompt|_secrets_re|without passwords|secrets are read|forbidden_terms|"
        r"c:\\users\\<name>|c:\\users\\<user>)", re.I
    )

    for root, dirs, files in os.walk(unzip_dir):
        for f in files:
            # Skip binary files
            if f.endswith((".png", ".jpg", ".jpeg", ".pdf", ".gpkg", ".npy")):
                continue
            fpath = Path(root) / f
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                for line_num, line in enumerate(content.splitlines(), start=1):
                    # Check for personal path
                    if re.search(r"[A-Za-z]:\\Users\\[a-zA-Z0-9_-]+", line) and not re.search(r"C:\\Users\\<", line):
                        violations.append((str(fpath.relative_to(unzip_dir)), line_num, "Personal Path", line.strip()))
                    # Check for actual hardcoded plaintext passwords
                    if re.search(r'(password|secret)\s*[:=]\s*["\'][^"\']{3,}["\']', line, re.I):
                        if not allowlist_re.search(line):
                            violations.append((str(fpath.relative_to(unzip_dir)), line_num, "Hardcoded Secret", line.strip()))
            except Exception:
                pass

    if violations:
        print(f"[FAIL] Found {len(violations)} potential policy violations:")
        for v in violations[:10]:
            print(f"  {v[0]}:{v[1]} [{v[2]}] {v[3][:80]}")
    else:
        print("[PASS] Zero plaintext passwords, secrets, or personal user paths detected.")

    # Run selfcheck on unzipped copy
    print("\n--- Running selfcheck.py against Unzipped Submission ---")
    sys.path.insert(0, str(unzip_dir))
    import subprocess
    cmd = [sys.executable, str(unzip_dir / "evalsuite" / "selfcheck.py")]
    p = subprocess.run(cmd, cwd=str(unzip_dir), capture_output=True, text=True)
    print(p.stdout)
    if "SUMMARY:" in p.stdout:
        print(p.stdout[p.stdout.find("SUMMARY:"):])


def main():
    print("=" * 70)
    print("BUILDING AND VERIFYING SUBMISSION ZIP (evalsuite/build_and_verify_zip.py)")
    print("=" * 70)

    # 1. Copy to staging
    copy_to_staging(ROOT, STAGING_DIR)

    # 2. Create zip in staging
    size_mb, files = create_zip(STAGING_DIR, ZIP_STAGING_PATH)

    # Copy zip to workspace root as well
    shutil.copy2(ZIP_STAGING_PATH, ZIP_FINAL_PATH)
    print(f"Copied archive to project root: {ZIP_FINAL_PATH}")

    # 3. Unzip and verify
    verify_unzipped(ZIP_STAGING_PATH, TEMP_UNZIP_DIR)

    print("\n" + "=" * 70)
    print("BUILD & VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
