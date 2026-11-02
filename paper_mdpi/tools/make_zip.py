"""paper_mdpi.zip for Overleaf: only the files needed to compile main.tex (template, text, generated numbers,
tables, bibliography, figures). Runs the generators and the checker first; refuses to pack on a failure.
Run: python paper_mdpi/tools/make_zip.py   (writes paper_mdpi.zip in the repository root, not committed)
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_mdpi"
# MDPI class files (MDPI_template_ACS.zip, unchanged); template.tex and template.pdf are not needed to compile
# EPS logos are left out: main.tex uses logo-mdpi.pdf (no epstopdf run); logo-updates.eps is not used with "submit"
TEMPLATE = sorted(p.relative_to(PAPER).as_posix() for p in (PAPER / "Definitions").iterdir()
                  if p.is_file() and p.suffix != ".eps")
TEXT = ["main.tex", "numbers.tex", "bibliography.tex", "references.bib"]


def main() -> int:
    for script in ("tools/paper_numbers.py", "tools/make_tables.py", "tools/make_bibliography.py", "check_numbers.py"):
        r = subprocess.run([sys.executable, str(PAPER / script)], cwd=ROOT)
        if r.returncode:
            print(f"{script} failed; paper_mdpi.zip not written")
            return 1
    files = [PAPER / f for f in TEMPLATE + TEXT]
    for folder, pattern in (("sections", "*.tex"), ("tables", "*.tex"), ("figures", "*.pdf")):
        files += sorted((PAPER / folder).glob(pattern))
    out = ROOT / "paper_mdpi.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(PAPER).as_posix())
    print(f"wrote {out.name}: {len(files)} files")
    for f in files:
        print("  ", f.relative_to(PAPER).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
