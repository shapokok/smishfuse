"""Kaggle CPU kernel that builds two static PDF assets of the MDPI version (LaTeX is not installed locally):

  cascade_diagram_h.pdf, cascade_diagram_c.pdf   the two variants of Figure 1 (paper_mdpi/assets/*.tex, standalone,
                                           TikZ, Palatino), with the numbers of paper_mdpi/numbers.tex
  logo-mdpi.pdf                            the MDPI logo converted once from logo-mdpi.eps with Ghostscript
                                           (-dEPSCrop), so that pdfLaTeX does not run epstopdf on every compilation

The input files are embedded in the notebook (base64) together with their SHA-256; the notebook checks the
hashes before it runs anything, and kaggle/build/smishfuse-mdpi-assets.sha256 keeps them for the local check.
Run: python scripts/make_mdpi_assets_kernel.py, then scripts/kaggle_sync.py push / output smishfuse-mdpi-assets
"""
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "smishfuse-mdpi-assets"
OUT = ROOT / "kaggle" / "kernels" / NAME
INPUTS = {"cascade_diagram_h.tex": ROOT / "paper_mdpi" / "assets" / "cascade_diagram_h.tex",
          "cascade_diagram_c.tex": ROOT / "paper_mdpi" / "assets" / "cascade_diagram_c.tex",
          "numbers.tex": ROOT / "paper_mdpi" / "numbers.tex",
          "logo-mdpi.eps": ROOT / "paper_mdpi" / "Definitions" / "logo-mdpi.eps"}

CODE = r'''import base64, hashlib, os, shutil, subprocess
FILES = {files!r}
SHA = {sha!r}
work = "/tmp/assets"
os.makedirs(work, exist_ok=True)
for name, data in FILES.items():
    raw = base64.b64decode(data)
    digest = hashlib.sha256(raw).hexdigest()
    print(name, digest)
    if digest != SHA[name]:
        raise SystemExit(f"hash mismatch for {{name}}")
    open(os.path.join(work, name), "wb").write(raw)
subprocess.run("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends "
               "texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-pictures "
               "texlive-fonts-recommended ghostscript > /tmp/apt.log 2>&1", shell=True, check=True)
for v in ("h", "c"):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"cascade_diagram_{{v}}.tex"], cwd=work, check=True)
subprocess.run(["gs", "-q", "-dSAFER", "-dNOPAUSE", "-dBATCH", "-dEPSCrop", "-sDEVICE=pdfwrite",
                "-sOutputFile=logo-mdpi.pdf", "logo-mdpi.eps"], cwd=work, check=True)
for name in ("cascade_diagram_h.pdf", "cascade_diagram_c.pdf", "cascade_diagram_h.log", "cascade_diagram_c.log",
             "logo-mdpi.pdf"):
    shutil.copy(os.path.join(work, name), "/kaggle/working/" + name)
    print("wrote", name, os.path.getsize(os.path.join(work, name)), "bytes")
print(subprocess.run(["pdflatex", "--version"], capture_output=True, text=True).stdout.splitlines()[0])
print(subprocess.run(["gs", "--version"], capture_output=True, text=True).stdout.strip())
'''


def main():
    files = {k: base64.b64encode(p.read_bytes()).decode() for k, p in INPUTS.items()}
    sha = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in INPUTS.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    src = CODE.format(files=files, sha=sha)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                     "source": src.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (OUT / f"{NAME}.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta = {"id": f"shapok/{NAME}", "title": NAME, "code_file": f"{NAME}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    stamp = ROOT / "kaggle" / "build" / f"{NAME}.sha256"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text("".join(f"{v}  {k}\n" for k, v in sha.items()))
    print(f"wrote {OUT.relative_to(ROOT)}")
    for k, v in sha.items():
        print(f"  {k} {v}")


if __name__ == "__main__":
    main()
