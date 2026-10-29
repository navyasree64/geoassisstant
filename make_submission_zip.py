"""
make_submission_zip.py
Prepares a clean zip archive for submission, excluding:
- .env (secrets)
- venv/
- .git/
- __pycache__/
- temporary or scratch files
"""

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_ZIP = ROOT / "east_india_geoassistant_submission.zip"

EXCLUDE_DIRS = {
    "venv", ".venv", "env",
    ".git", "__pycache__",
    ".pytest_cache", ".agents",
    ".idea", ".vscode"
}

EXCLUDE_FILES = {
    ".env",
    "east_india_geoassistant_submission.zip",
    "make_submission_zip.py"
}

def create_submission_zip():
    print(f"Creating submission zip at: {OUTPUT_ZIP.name}")
    count = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ROOT):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            
            for file in files:
                if file in EXCLUDE_FILES or file.endswith(".pyc") or file.startswith("."):
                    continue
                file_path = Path(root) / file
                arcname = file_path.relative_to(ROOT)
                zf.write(file_path, arcname)
                count += 1

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Done! Added {count} files ({size_mb:.2f} MB).")
    print(f"Excluded: .env, venv/, __pycache__/, and hidden files.")

if __name__ == "__main__":
    create_submission_zip()
