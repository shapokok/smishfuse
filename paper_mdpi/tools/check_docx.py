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
import unicodedata
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
    # numbered equations: one m:oMath alone in the equation cell of their table; the others are inline
    eq_cells = [re.findall(r"<w:tc>.*?</w:tc>", t, flags=re.S)[0]
                for t in re.findall(r"<w:tbl>.*?</w:tbl>", body, flags=re.S) if "MDPI39equation" in t]
    display = sum(1 for cell in eq_cells if cell.count("<m:oMath>") == 1 and not re.search(r"<w:t[ >]", cell))
    inline_omml = len(re.findall(r"<m:oMath>", body)) - len(eq_cells)
    if (n_disp, n_inline) != (display, inline_omml) or "<m:oMathPara>" in body:
        fails.append(f"equations: LaTeX {n_disp} display / {n_inline} inline, Word {display} / {inline_omml}")
    refs = [t for st, t in paras if st == "MDPI81references"]
    bib = re.split(r"\\bibitem\{[^}]+\}", (PAPER / "bibliography.tex").read_text(encoding="utf-8"))[1:]
    nbib = len(bib)
    def lt(s):   # letters only, accents dropped (Unicode in Word, accent macros in LaTeX)
        s = re.sub(r"\\[`'^\"~=.]|\\[a-zA-Z]+", "", s)
        s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
        return re.sub(r"[^a-z]", "", s.lower())[:30]
    if len(refs) != nbib or any(lt(a) != lt(b) for a, b in zip(refs, bib)):
        fails.append(f"reference list differs from bibliography.tex ({len(refs)} entries, {nbib} expected)")
    if len(re.findall(r'MDPI81references"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="23"/>', body)) != nbib:
        fails.append("references are not numbered by the template numbering of MDPI_8.1_references")
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
    clean = re.sub(r"@@(STYLE:\w+|EQ:\w+|AFTEREQ|TAB:\w+:\d+)@@", "", flat)
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
    # 6. layout of the tables: fixed widths that hold the longest word of every cell, repeated header rows as in
    #    the LaTeX head, rows that do not break, caption kept with its table; plus-minus as text; running head
    heads = [int(x) for x in re.findall(r"@@TAB:\w+:(\d+)@@", flat)]
    data_tbls = [t for t in re.findall(r"<w:tbl>.*?</w:tbl>", body, flags=re.S) if "MDPI39equation" not in t]
    for i, t in enumerate(data_tbls, 1):
        if '<w:tblLayout w:type="fixed"/>' not in t:
            fails.append(f"table {i}: no fixed layout")
        rows = re.findall(r"<w:tr(?: [^>]*)?>.*?</w:tr>", t, flags=re.S)
        if len(rows) <= mk.KEEP_TOGETHER_ROWS and any("<w:keepNext/>" not in r for r in rows[:-1]):
            fails.append(f"table {i}: a short table is not kept on one page")
        grid = [int(x) for x in re.findall(r'<w:gridCol w:w="(\d+)"', t)]
        pt = 8.0 if '<w:sz w:val="16"/>' in t else 10.0
        for r in re.findall(r"<w:tr(?: [^>]*)?>.*?</w:tr>", t, flags=re.S):
            if "<w:cantSplit/>" not in r:
                fails.append(f"table {i}: a row may break across pages")
            col = 0
            for tc in re.findall(r"<w:tc>.*?</w:tc>", r, flags=re.S):
                sp = re.search(r'<w:gridSpan w:val="(\d+)"', tc)
                n = int(sp.group(1)) if sp else 1
                width = sum(grid[col:col + n])
                for word in mk.words(mk.cell_text(tc)):
                    if mk.text_tw(word, pt) + mk.CELL_PAD_TW > width + 1:
                        fails.append(f"table {i}: {word!r} wider than its column ({width} twips)")
                col += n
    caption_idx = [k for k, (st, _) in enumerate(paras) if st == "MDPI41tablecaption"]
    table_idx = [k for k, (st, _) in enumerate(paras) if st == "TABLE"]
    for k, n_head in zip(caption_idx, heads):
        nxt = [j for j in table_idx if j > k][0]
        t = data_tbls[table_idx.index(nxt)]
        if t.count("<w:tblHeader/>") != n_head:
            fails.append(f"table after caption {paras[k][1][:12]!r}: {t.count('<w:tblHeader/>')} header rows, LaTeX {n_head}")
    for cap in re.findall(r"<w:p(?: [^>]*)?>(?:(?!</w:p>).)*MDPI41tablecaption.*?</w:p>", body, flags=re.S):
        if "<w:keepNext/>" not in cap:
            fails.append("a table caption is not kept with its table")
    if re.search(r"<m:t>\u00b1</m:t>", body):
        fails.append("plus-minus written as an equation")
    zf = zipfile.ZipFile(DOCX)
    heads_xml = "".join(zf.read(n).decode("utf-8") for n in zf.namelist() if re.fullmatch(r"word/(header|footer)\d\.xml", n))
    head_text = re.sub(r"<[^>]+>", "", heads_xml)
    if "Information 2026, 17" not in head_text or "2025" in head_text:
        fails.append("running head is not 'Information 2026, 17, x'")

    # 7. left edge of every paragraph of the running text = the text column of the template (2608 twips): style
    #    chain (w:basedOn), numbering level and direct formatting resolved as Word does; lists have a hanging
    #    indent with the marker at the column. Exempt: article type, title, authors (the template places them
    #    differently), figures, paragraphs in tables, and the reference list (full width in the template).
    zf = zipfile.ZipFile(DOCX)
    styles = zf.read("word/styles.xml").decode("utf-8")
    numbering = zf.read("word/numbering.xml").decode("utf-8")
    sblocks = {m.group(1): m.group(0) for m in re.finditer(r'<w:style [^>]*w:styleId="([^"]+)".*?</w:style>', styles, re.S)}

    def attrs(xml):
        m = re.search(r"<w:ind ([^>]*?)/?>", xml or "")
        return {k: int(v) for k, v in re.findall(r'w:(left|hanging|firstLine)="(-?\d+)"', m.group(1))} if m else {}

    def style_ind(sid, seen=()):
        """Indent of a style. Stricter than Word: a style with its own w:ind does not inherit the rest of the
        indent (as in LibreOffice), so an incomplete w:ind in a derived style is caught."""
        ppr = re.search(r"<w:pPr>.*?</w:pPr>", sblocks.get(sid, ""), re.S)
        mine = attrs(ppr.group(0) if ppr else "")
        if mine:
            return mine
        b = re.search(r'<w:basedOn w:val="([^"]+)"', sblocks.get(sid, ""))
        return style_ind(b.group(1), seen + (sid,)) if b and b.group(1) not in seen else {}
    absn = {m.group(1): m.group(2) for m in re.finditer(r'<w:num w:numId="(\d+)"[^>]*>\s*<w:abstractNumId w:val="(\d+)"', numbering)}

    def num_ind(num, lvl="0"):
        a = re.search(r'<w:abstractNum w:abstractNumId="%s".*?</w:abstractNum>' % absn.get(num, "x"), numbering, re.S)
        lv = re.search(r'<w:lvl w:ilvl="%s".*?</w:lvl>' % lvl, a.group(0), re.S) if a else None
        return attrs(lv.group(0)) if lv else {}
    exempt = {"MDPI11articletype", "MDPI12title", "MDPI13authornames", "MDPI52figure", "MDPI81references"}
    outside = re.sub(r"<w:tbl>.*?</w:tbl>", "", body, flags=re.S)
    off = []
    for blk in re.findall(r"<w:p(?: [^>]*)?>.*?</w:p>", outside, flags=re.S):
        st = re.search(r'<w:pStyle w:val="([^"]+)"', blk)
        sid = st.group(1) if st else "Normal"
        if sid in exempt or not re.search(r"<w:t[ >]", blk):
            continue
        ind = style_ind(sid)
        num = re.search(r'<w:numId w:val="(\d+)"', blk) or re.search(r'<w:numId w:val="(\d+)"', sblocks.get(sid, ""))
        if num and num.group(1) != "0":
            ind = {**ind, **num_ind(num.group(1))}
        ppr = re.search(r"<w:pPr>.*?</w:pPr>", blk, re.S)
        ind = {**ind, **attrs(ppr.group(0) if ppr else "")}
        start = ind.get("left", 0) - ind.get("hanging", 0)
        if start != 2608:
            off.append(f"[{sid}] start {start} twips: {text_of(blk)[:60]!r}")
    for o in off:
        fails.append("paragraph outside the text column " + o)
    tbl_off = [t for t in re.findall(r"<w:tbl>.*?</w:tbl>", body, flags=re.S)
               if 'w:type="pct"' not in t and '<w:tblW w:w="10466"' not in t and '<w:tblInd w:w="2608"' not in t]
    if tbl_off:
        fails.append(f"{len(tbl_off)} text-width tables not in the text column")

    # 8. figures as SVG with a PNG of at least 600 dpi; cases as equation arrays; data statement left-aligned
    rels = zf.read("word/_rels/document.xml.rels").decode("utf-8")
    target = dict(re.findall(r'Id="(rId\w+)"[^>]*Target="([^"]+)"', rels))
    for k, blip in enumerate(re.findall(r"<a:blip .*?</a:blip>|<a:blip [^>]*/>", body, flags=re.S), 1):
        svg = re.search(r'svgBlip [^>]*r:embed="(rId\w+)"', blip)
        png = re.search(r'<a:blip r:embed="(rId\w+)"', blip)
        if not svg or not target.get(svg.group(1), "").endswith(".svg"):
            fails.append(f"picture {k}: no SVG version")
        data = zf.read("word/" + target[png.group(1)])
        i = data.find(b"pHYs")
        ppm = int.from_bytes(data[i + 4:i + 8], "big") if i > 0 else 0
        if ppm * 0.0254 < 599:
            fails.append(f"picture {k}: PNG at {ppm * 0.0254:.0f} dpi")
    for m in re.finditer(r'<m:d><m:dPr><m:begChr m:val="\{" ?/>.*?</m:dPr><m:e>(<m:\w+)', body, flags=re.S):
        if m.group(1) == "<m:m":
            fails.append("a case distinction is written as a matrix, not as an equation array")
    da = [b for b in re.findall(r"<w:p(?: [^>]*)?>.*?</w:p>", body, flags=re.S) if "Data Availability Statement" in b]
    if not da or '<w:jc w:val="left"/>' not in da[0]:
        fails.append("the data availability statement is not left-aligned")

    print(f"paragraphs outside the text column: {len(off)}")
    print(f"docx: {len(paras)} blocks, {len(tables)} tables, {len(figs)} figures, {len(eqs)} numbered equations, "
          f"{len(refs)} references; values checked {len(need)}; paragraphs compared, lost {lost}")
    for f in fails:
        print("FAIL", f)
    print("OK" if not fails else f"{len(fails)} problem(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
