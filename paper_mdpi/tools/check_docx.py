"""Checks paper_mdpi.docx against the LaTeX sources and numbers.tex (run after tools/make_docx.py).

1. Leftovers: no "??", no @@ markers, no LaTeX commands or unresolved labels in the text.
2. Numbers: every \\val{key} used in the sources appears in the docx text with its value from numbers.tex at least
   as often as it is used; every number in the running text (outside tables and the reference list) is a value of
   numbers.tex, a section, figure, table, equation or citation number, or one of the allowed literals of
   check_numbers.py.
3. Tables: the numbers in every table (cells only) are the same multiset as in the LaTeX table, table by table.
4. Structure: figure, table and equation numbers appear in order with their captions; every figure has an image;
   the reference list has one numbered entry per \\bibitem and every reference is cited.
5. Lost text: every paragraph of the prepared LaTeX (pandoc plain rendering) is found in the docx text.
Exit status 1 on any failure. Run: python paper_mdpi/tools/check_docx.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_mdpi"
BUILD = PAPER / "build_docx"
DOCX = ROOT / "paper_mdpi.docx"
NUM = re.compile(r"\d+(?:\.\d+)*")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SPACES = "\\s\u00a0\u2000-\u200b\u202f"


def norm(s: str) -> str:
    """Spacing and minus signs as Word shows them, removed for comparison."""
    return re.sub(f"[{SPACES}]+", "", s.replace("\u2212", "-").replace("\u2013", "-"))


def join_digits(s: str) -> str:
    """1\u2006000 -> 1000 (thin spaces between digit groups)."""
    return re.sub(f"(?<=\\d)[\u00a0\u2000-\u200b\u202f](?=\\d)", "", s)


def plain_value(v: str) -> str:
    v = v.replace("\\textvisiblespace{}", "\u2423")
    v = re.sub(r"\\texttt\{(.*)\}", r"\1", v)
    v = v.replace("--", "-").replace("\\,", "").replace("$-$", "-").replace("$+$", "+").replace("$", "")
    v = v.replace("\\times 10^", "×10").replace("{", "").replace("}", "")
    v = v.replace("\\#", "#").replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    return v


def math_text(xml: str) -> str:
    """Text of a fragment with the delimiters of Word equations (m:dPr begChr/endChr, default parentheses)."""
    def delim(m):
        beg = re.search(r'<m:begChr m:val="([^"]*)"', m.group(1))
        end = re.search(r'<m:endChr m:val="([^"]*)"', m.group(1))
        return (beg.group(1) if beg else "(") + m.group(2) + (end.group(1) if end else ")")
    prev = None
    while prev != xml:
        prev = xml
        xml = re.sub(r"<m:d>(?:<m:dPr>(.*?)</m:dPr>)?((?:(?!<m:d>).)*?)</m:d>",
                     lambda m: delim(m) if m.group(1) is not None else "(" + m.group(2) + ")", xml, flags=re.S)
    return xml


def text_of(xml: str) -> str:
    xml = math_text(xml)
    return "".join(re.findall(r"<(?:w|m):t(?: [^>]*)?>([^<]*)</(?:w|m):t>", xml)) \
        .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')


def prose_of(xml: str) -> str:
    """Text without equations (m:oMath / m:oMathPara removed), for the comparison of running text."""
    return text_of(re.sub(r"<m:oMathPara>.*?</m:oMathPara>|<m:oMath>.*?</m:oMath>", " ", xml, flags=re.S))


def docx_parts() -> tuple[list[tuple[str, str]], list[list[list[str]]], str]:
    xml = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf-8")
    body = xml[xml.index("<w:body>"):]
    paras, tables = [], []
    for block in re.findall(r"<w:tbl>.*?</w:tbl>|<w:p(?: [^>]*)?>.*?</w:p>", body, flags=re.S):
        if block.startswith("<w:tbl>"):
            if "MDPI39equation" in block:      # numbered equation: equation cell + number cell
                cells = re.findall(r"<w:tc>(.*?)</w:tc>", block, flags=re.S)
                paras.append(("EQUATION", text_of(cells[0]) + "\t" + text_of(cells[1])))
                continue
            rows = [[text_of(c) for c in re.findall(r"<w:tc>(.*?)</w:tc>", r, flags=re.S)]
                    for r in re.findall(r"<w:tr[ >].*?</w:tr>", block, flags=re.S)]
            tables.append(rows)
            paras.append(("TABLE", f"#{len(tables)}"))
            continue
        st = re.search(r'<w:pStyle w:val="([^"]+)"', block)
        paras.append((st.group(1) if st else "", text_of(block) + ("[IMAGE]" if "<w:drawing>" in block else "")))
    return paras, tables, body


def main() -> int:
    fails = []
    paras, tables, body = docx_parts()
    text = "\n".join(t for _, t in paras) + "\n" + "\n".join("\t".join(c) for tb in tables for r in tb for c in r)
    labels = json.loads((BUILD / "labels.json").read_text())
    flat = (BUILD / "paper_flat.tex").read_text(encoding="utf-8")

    # 1. leftovers
    for bad in ("??", "@@", "\\", "[sec:", "[tab:", "[fig:", "[eq:", "[app:"):
        if bad in text:
            fails.append(f"leftover {bad!r} in the docx text")

    # 2. numbers against numbers.tex
    nums = load("paper_numbers", PAPER / "tools" / "paper_numbers.py").build()
    vals = {k: plain_value(v) for k, (v, _) in nums.items()}
    mk = load("make_docx", PAPER / "tools" / "make_docx.py")
    src = mk.strip_comments((PAPER / "main.tex").read_text(encoding="utf-8"))
    src = mk.flatten(src.split("\\begin{document}")[0] + src.split("\\begin{document}")[1])
    used = Counter(re.findall(r"\\val\{([^}]+)\}", src))
    ntext = norm(text)
    need = Counter()
    for k, n in used.items():
        need[norm(vals[k])] += n
    for v, n in need.items():
        have = ntext.count(v)
        if have < n:
            fails.append(f"value {v!r} used {n} times in the LaTeX text, found {have} times in the docx")
    allowed = {"95", "2.5", "97.5", "50", "200", "4.0"}
    structural = set(labels.values()) | {str(i) for i in range(1, 100)}
    value_tokens = {t for v in vals.values() for t in NUM.findall(norm(v))}
    in_refs = False
    for st, t in paras:
        if st == "MDPI21heading1" and t.strip() == "References":
            in_refs = True
        if in_refs or st in ("TABLE", "MDPI81references"):
            continue
        for tok in NUM.findall(join_digits(re.sub(r"https?://\S+", " ", t))):
            if len(tok) > 1 and tok not in value_tokens | allowed | structural:
                fails.append(f"number {tok!r} in [{st}] {t[:70]!r} is not a value of numbers.tex")

    # 3. tables, cell numbers per table
    tex_tables = re.findall(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", flat, flags=re.S)
    if len(tex_tables) != len(tables):
        fails.append(f"{len(tex_tables)} tables in LaTeX, {len(tables)} in the docx")
    for i, (tt, dt) in enumerate(zip(tex_tables, tables), 1):
        tt = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", tt)
        tt = re.sub(r"\\(toprule|midrule|bottomrule|textbf|emph|texttt)", "", tt)
        a = Counter(NUM.findall(join_digits(tt)))
        b = Counter(NUM.findall(join_digits(" | ".join(c for r in dt for c in r))))
        if a != b:
            fails.append(f"table {i}: numbers differ (only LaTeX {dict(a - b)}, only docx {dict(b - a)})")

    # 4. structure: captions, images, equations, references, citations
    def seq(kind):
        return [v for k, v in labels.items() if k.startswith(kind)]
    figs = [re.match(r"Figure (A?\d+)\.", t).group(1) for st, t in paras if st == "MDPI51figurecaption"]
    tabs = [re.match(r"Table (A?\d+)\.", t).group(1) for st, t in paras if st == "MDPI41tablecaption"]
    eqs = [t.split("\t")[1].strip("()") for st, t in paras if st == "EQUATION"]
    imgs = sum(1 for st, t in paras if t.endswith("[IMAGE]"))
    for name, got, exp in (("figure", figs, seq("fig:")), ("table", tabs, seq("tab:")), ("equation", eqs, seq("eq:"))):
        if got != exp:
            fails.append(f"{name} numbers in the docx {got} differ from the LaTeX order {exp}")
    if imgs != len(figs):
        fails.append(f"{imgs} images for {len(figs)} figure captions")
    rest = re.sub(r"\$\$.*?\$\$", " ", flat, flags=re.S)
    n_disp, n_inline = len(re.findall(r"\$\$.*?\$\$", flat, flags=re.S)), len(re.findall(r"(?<!\\)\$.*?(?<!\\)\$", rest, flags=re.S))
    xml_para = re.findall(r"<m:oMathPara>.*?</m:oMathPara>", body, flags=re.S)
    inline_omml = len(re.findall(r"<m:oMath>", body)) - sum(len(re.findall(r"<m:oMath>", b)) for b in xml_para)
    if (n_disp, n_inline) != (len(xml_para), inline_omml):
        fails.append(f"equations: LaTeX {n_disp} display / {n_inline} inline, Word {len(xml_para)} / {inline_omml}")
    refs = [t for st, t in paras if st == "MDPI81references"]
    nbib = len(re.findall(r"\\bibitem\{", (PAPER / "bibliography.tex").read_text(encoding="utf-8")))
    if [int(t.split(".", 1)[0]) for t in refs] != list(range(1, nbib + 1)):
        fails.append(f"reference list is not 1..{nbib}")
    cited = set()
    for st, t in paras:
        if st != "MDPI81references":
            for grp in re.findall(r"\[([1-9]\d*(?:[,\u2013]\d+)*)\]", t):
                for part in grp.split(","):
                    a, _, b = part.partition("\u2013")
                    cited.update(range(int(a), int(b or a) + 1))
    for tb in tables:
        for grp in re.findall(r"\[([1-9]\d*(?:[,\u2013]\d+)*)\]", "\t".join(c for r in tb for c in r)):
            for part in grp.split(","):
                a, _, b = part.partition("\u2013")
                cited.update(range(int(a), int(b or a) + 1))
    if cited != set(range(1, nbib + 1)):
        fails.append(f"citations do not cover 1..{nbib}: missing {sorted(set(range(1, nbib + 1)) - cited)}, "
                     f"extra {sorted(cited - set(range(1, nbib + 1)))}")

    # 5. lost text: every prepared paragraph (pandoc plain) in the docx, compared on letters only and without
    #    equations (display equations are checked by number in step 4, inline ones are compared as gaps)
    clean = re.sub(r"@@(STYLE:\w+|EQ:\w+|AFTEREQ|WIDE)@@", "", flat)
    clean = re.sub(r"\$\$.*?\$\$", " ", clean, flags=re.S)
    clean = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " ", clean, flags=re.S)
    plain = subprocess.run(["pandoc", "-f", "latex", "-t", "plain", "--wrap=none"], input=clean, text=True,
                           capture_output=True, check=True).stdout
    letters = lambda s: re.sub(r"[^a-z]", "", s.lower())  # noqa: E731
    prose = "\n".join(prose_of(b) for b in re.findall(r"<w:p(?: [^>]*)?>.*?</w:p>", body, flags=re.S))
    dl = letters(prose)
    lost = 0
    for para in plain.split("\n\n"):
        lp = letters(para)
        if len(lp) < 20:
            continue
        head, tail = lp[:40], lp[-40:]
        if head not in dl or tail not in dl:
            lost += 1
            fails.append(f"paragraph not found in the docx: {para[:90]!r}")
    print(f"docx: {len(paras)} blocks, {len(tables)} tables, {len(figs)} figures, {len(eqs)} numbered equations, "
          f"{len(refs)} references; values checked {len(need)}; paragraphs compared, lost {lost}")
    for f in fails:
        print("FAIL", f)
    print("OK" if not fails else f"{len(fails)} problem(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
