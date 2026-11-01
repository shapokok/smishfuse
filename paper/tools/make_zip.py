"""paper.zip for Overleaf: only the files needed to compile main.tex (template, text, generated numbers,
tables, bibliography, figures). Runs the generators and the checker first; refuses to pack on a failure.
Run: python paper/tools/make_zip.py   (writes paper.zip in the repository root, not committed)
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
TEMPLATE = ["NSP1.cls", "everyshi.sty", "psfig.sty", "logo7pt.eps"]
TEXT = ["main.tex", "numbers.tex", "bibliography.tex", "references.bib"]


def main() -> int:
    for script in ("tools/paper_numbers.py", "tools/make_tables.py", "tools/make_bibliography.py", "check_numbers.py"):
        r = subprocess.run([sys.executable, str(PAPER / script)], cwd=ROOT)
        if r.returncode:
            print(f"{script} failed; paper.zip not written")
            return 1
    files = [PAPER / f for f in TEMPLATE + TEXT]
    for folder, pattern in (("sections", "*.tex"), ("tables", "*.tex"), ("figures", "*.eps"), ("figures", "*.pdf")):
        files += sorted((PAPER / folder).glob(pattern))
    out = ROOT / "paper.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(PAPER).as_posix())
    print(f"wrote {out.name}: {len(files)} files")
    for f in files:
        print("  ", f.relative_to(PAPER).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
