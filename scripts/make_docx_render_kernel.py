"""Kaggle CPU kernel that renders paper_mdpi.docx to PDF with LibreOffice (visual check of the Word version; no
LibreOffice is installed locally). The docx goes to the private dataset smishfuse-docx (the notebook would be too
large with it embedded); its SHA-256 is written into the notebook, which checks it before rendering, and
kaggle/build/smishfuse-docx-render.sha256 keeps it for the local check.
Fonts: the URW and TeX Gyre clones of Palatino (fontconfig maps Palatino Linotype to them).
Run: python scripts/make_docx_render_kernel.py (uploads the dataset), then scripts/kaggle_sync.py push / output
smishfuse-docx-render
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "smishfuse-docx-render"
OUT = ROOT / "kaggle" / "kernels" / NAME
DOCX = ROOT / "paper_mdpi.docx"

CODE = r'''import glob, hashlib, os, shutil, subprocess
SHA = {sha!r}
work = "/tmp/render"
os.makedirs(work, exist_ok=True)
raw = open(sorted(glob.glob("/kaggle/input/**/paper_mdpi.docx", recursive=True))[0], "rb").read()
digest = hashlib.sha256(raw).hexdigest()
print("paper_mdpi.docx", digest)
if digest != SHA:
    raise SystemExit("hash mismatch")
open(os.path.join(work, "paper_mdpi.docx"), "wb").write(raw)
subprocess.run("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends "
               "libreoffice-writer libreoffice-math fonts-urw-base35 fonts-texgyre fonts-dejavu "
               "> /tmp/apt.log 2>&1", shell=True, check=True)
r = subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", work, os.path.join(work, "paper_mdpi.docx")],
                   capture_output=True, text=True, timeout=1200)
print(r.stdout, r.stderr)
shutil.copy(os.path.join(work, "paper_mdpi.pdf"), "/kaggle/working/paper_mdpi_libreoffice.pdf")
print(subprocess.run(["soffice", "--version"], capture_output=True, text=True).stdout)
print(subprocess.run("fc-match 'Palatino Linotype'; fc-match 'Cambria Math'", shell=True, capture_output=True, text=True).stdout)
'''


def main():
    raw = DOCX.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    folder = ROOT / "kaggle" / "build" / "smishfuse-docx"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "paper_mdpi.docx").write_bytes(raw)
    sys.path.insert(0, str(ROOT / "scripts"))
    import kaggle_sync
    kaggle_sync.upload("smishfuse-docx", folder, f"paper_mdpi.docx {sha[:12]}")
    src = CODE.format(sha=sha)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                     "source": src.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (OUT / f"{NAME}.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta = {"id": f"shapok/{NAME}", "title": NAME, "code_file": f"{NAME}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": ["shapok/smishfuse-docx"], "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []}
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    stamp = ROOT / "kaggle" / "build" / f"{NAME}.sha256"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(f"{sha}  paper_mdpi.docx\n")
    print(f"wrote {OUT.relative_to(ROOT)}; paper_mdpi.docx {sha}")


if __name__ == "__main__":
    main()
