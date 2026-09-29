"""Script to package SonicSentinel AI into a complete, clean, standalone ZIP distribution.
Excludes duplicate/intermediate zip caches, temporary files, and __pycache__.
Saves to user Desktop for instant access.
"""
import os
import sys
import zipfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = Path(r"c:\Users\akber ali\Desktop\SonicSentinel_AI_Complete.zip")

EXCLUDED_FILENAMES = {
    "aiml_project.zip",
    "ESC-50-master.zip",
    "create_project_zip.py",
}

EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".vscode",
    ".idea",
    "venv",
    ".venv",
}


def make_zip():
    print(f"[*] Starting SonicSentinel AI complete distribution archive...")
    print(f"[*] Source Directory: {PROJECT_ROOT}")
    print(f"[*] Destination Archive: {OUTPUT_ZIP}")
    t0 = time.time()

    # Collect all files
    all_files = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Filter out excluded directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for f in files:
            if f in EXCLUDED_FILENAMES or f.endswith(".pyc") or f.endswith(".zip"):
                continue
            all_files.append(Path(root) / f)

    total_files = len(all_files)
    print(f"[*] Total files to archive: {total_files}")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for idx, file_path in enumerate(all_files, 1):
            arcname = file_path.relative_to(PROJECT_ROOT)
            # Prefix with clean root folder 'SonicSentinel_AI'
            full_arcname = Path("SonicSentinel_AI") / arcname
            zf.write(file_path, str(full_arcname))
            if idx % 500 == 0 or idx == total_files:
                print(f"  [{idx}/{total_files}] Archiving ({idx/total_files*100:5.1f}%) -> {arcname}")

    t1 = time.time()
    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n[+] Distribution ZIP successfully created in {t1 - t0:.1f}s!")
    print(f"[+] Final Archive Size: {size_mb:.2f} MB ({size_mb/1024:.2f} GB)")
    print(f"[+] Location: {OUTPUT_ZIP}")


if __name__ == "__main__":
    make_zip()
