"""Figures for the submission: paper_mdpi/figures_submit/Figure_<n>.pdf (vector) and .tif (600 dpi, LZW), in the
numbering of the article (main text 1, 2, ...; appendix A1, A2, ...); Figure 1 also as SVG (text as outlines).
The second variant of Figure 1 goes to figures_submit/alt/. Made from paper_mdpi/figures/*.pdf with poppler
(pdftoppm, pdftocairo); no figure is drawn here. Run: python paper_mdpi/tools/make_submit_figures.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "figures_submit"
ALT = PAPER / "assets" / "cascade_diagram_alt.pdf"


def export(src: Path, dst_stem: Path, svg: bool = False) -> None:
    shutil.copy(src, dst_stem.with_suffix(".pdf"))
    subprocess.run(["pdftoppm", "-r", "600", "-tiff", "-tiffcompression", "lzw", "-singlefile", str(src),
                    str(dst_stem)], check=True)
    if svg:
        subprocess.run(["pdftocairo", "-svg", str(src), str(dst_stem.with_suffix(".svg"))], check=True)


def main() -> int:
    sys.path.insert(0, str(PAPER / "tools"))
    import make_docx

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "alt").mkdir(parents=True)
    body = make_docx.strip_comments((PAPER / "main.tex").read_text(encoding="utf-8"))
    body = make_docx.flatten(body.split("\\begin{document}", 1)[1])
    app = body.find("\\appendix")
    n_main = n_app = 0
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        if 0 <= app < m.start():
            n_app += 1
            label = f"A{n_app}"
        else:
            n_main += 1
            label = str(n_main)
        src = PAPER / (m.group(1) + ".pdf")
        export(src, OUT / f"Figure_{label}", svg=label == "1")
        print(f"Figure {label}: {src.name}")
    export(ALT, OUT / "alt" / "Figure_1_alternative", svg=True)
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(ALT), str(OUT / "alt" / "Figure_1_alternative")],
                   check=True)
    print(f"wrote {OUT.relative_to(PAPER.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
