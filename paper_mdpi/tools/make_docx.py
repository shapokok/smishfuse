"""paper_mdpi (LaTeX, the source) -> paper_mdpi.docx in the MDPI Word template of Information, with pandoc.

1. word/reference.docx is made from information-template.dot (an OOXML template: content type switched to a
   document, body emptied, styles, page set-up, headers and footers kept).
2. The LaTeX sources are flattened (\\input expanded) and prepared for pandoc: every \\val{key} is replaced by its
   value from numbers.tex; sections, figures, tables and equations are numbered as in the PDF (appendix: A, A1, ...)
   and every \\ref and \\cite is replaced by its number ([1], [2,3], [4-6] with an en dash, in citation order);
   MDPI front and back matter commands become plain paragraphs; figures point to PNG copies at 300 dpi.
   Paragraphs that need an MDPI style carry a marker @@STYLE:<styleId>@@ (removed again in step 4).
3. pandoc -f latex -t docx --reference-doc word/reference.docx (equations become Word equations, OMML).
4. The docx is post-processed: MDPI paragraph styles (headings, text, captions, table body, references, back matter),
   the three-line table style, numbered equations as the two-cell table of the template, table widths.
Check: tools/check_docx.py. Run: python paper_mdpi/tools/make_docx.py   (writes paper_mdpi.docx in the repository root)
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_mdpi"
BUILD = PAPER / "build_docx"
TEMPLATE = ROOT / "information-template.dot"
REF = PAPER / "word" / "reference.docx"
OUT = ROOT / "paper_mdpi.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# ---------------------------------------------------------------- 1. reference.docx from the MDPI template

def make_reference() -> None:
    REF.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TEMPLATE) as src, zipfile.ZipFile(REF, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(b"wordprocessingml.template.main+xml", b"wordprocessingml.document.main+xml")
            elif item.filename == "word/document.xml":
                text = data.decode("utf-8")
                body_start = text.index("<w:body>") + len("<w:body>")
                sect = re.findall(r"<w:sectPr[ >].*?</w:sectPr>", text, flags=re.S)[-1]
                text = text[:body_start] + "<w:p/>" + sect + "</w:body></w:document>"
                data = text.encode("utf-8")
            dst.writestr(item, data)


# ---------------------------------------------------------------- 2. LaTeX preparation

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def flatten(text: str) -> str:
    def repl(m):
        name = m.group(1)
        if name == "numbers":
            return ""
        return flatten(strip_comments(read(PAPER / (name if name.endswith(".tex") else name + ".tex"))))
    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def group(text: str, start: int) -> tuple[str, int]:
    """Content of the brace group that opens at text[start] == '{', and the index after it."""
    if text[start] != "{":
        raise ValueError(f"no brace group at {start}: {text[start:start + 40]!r}")
    depth, i = 0, start
    while True:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(c, 0)
        i += 1
        if depth == 0:
            return text[start + 1:i - 1], i


def command_arg(text: str, cmd: str) -> str:
    m = re.search(r"\\" + cmd + r"\s*\{", text)
    if not m:
        raise ValueError(f"\\{cmd} not found")
    return group(text, m.end() - 1)[0]


def replace_command(text: str, cmd: str, fn, nargs: int = 1) -> str:
    """Replace every \\cmd{a}{b}... by fn(a, b, ...), with brace matching."""
    out, i = [], 0
    rx = re.compile(r"\\" + cmd + r"(?![a-zA-Z])\s*")
    while True:
        m = rx.search(text, i)
        if not m:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:m.start()])
        j, args = m.end(), []
        for _ in range(nargs):
            a, j = group(text, j)
            args.append(a)
        out.append(fn(*args))
        i = j


def values() -> dict[str, str]:
    t = read(PAPER / "numbers.tex")
    out = {}
    for m in re.finditer(r"\\@namedef\{val@([^}]+)\}\{", t):
        out[m.group(1)] = group(t, m.end() - 1)[0]
    return out


def expand_vals(text: str, vals: dict[str, str]) -> str:
    def repl(m):
        if m.group(1) not in vals:
            raise KeyError(f"\\val{{{m.group(1)}}} undefined")
        return vals[m.group(1)]
    return re.sub(r"\\val\{([^}]+)\}", repl, text)


def style(style_id: str, text: str) -> str:
    return f"\n\n@@STYLE:{style_id}@@{text}\n\n"


def front_matter(main: str) -> str:
    title = command_arg(main, "Title")
    author = re.sub(r"\\orcid[A-Z]\{\}", "", command_arg(main, "Author"))
    author = re.sub(r"\$\^\{([^}]*)\}\$", r"\\textsuperscript{\1}", author)
    address = command_arg(main, "address")
    lines = [re.sub(r"\$\^\{([^}]*)\}\$\s*\\quad\s*", r"\\textsuperscript{\1} ", x).strip()
             for x in re.split(r"\\\\", address) if x.strip()]
    corres = command_arg(main, "corres")
    abstract = command_arg(main, "abstract")
    keywords = command_arg(main, "keyword")
    out = style("MDPI11articletype", "Article") + style("MDPI12title", title) + style("MDPI13authornames", author)
    out += "".join(style("MDPI16affiliation", x) for x in lines)
    out += style("MDPI16affiliation", "\\textbf{*} " + corres)
    out += style("MDPI17abstract", "\\textbf{Abstract:} " + abstract)
    out += style("MDPI18keywords", "\\textbf{Keywords:} " + keywords)
    return out


BACK = [("authorcontributions", "Author Contributions"), ("funding", "Funding"),
        ("institutionalreview", "Institutional Review Board Statement"),
        ("informedconsent", "Informed Consent Statement"), ("dataavailability", "Data Availability Statement"),
        ("acknowledgments", "Acknowledgments"), ("conflictsofinterest", "Conflicts of Interest")]


def back_matter(text: str) -> str:
    for cmd, label in BACK:
        text = replace_command(text, cmd, lambda a, label=label: style("MDPI62backmatter", f"\\textbf{{{label}:}} {a}"))
    text = replace_command(text, "abbreviations",
                           lambda title, body: style("MDPI62backmatter", f"\\textbf{{{title}:}}")
                           + body.replace("\\\\\n", "\n", 1), nargs=2)
    return text


def number_everything(body: str) -> tuple[str, dict[str, str], dict[str, str]]:
    """Section, figure, table and equation numbers as in the PDF; returns (text, label -> number, label -> kind)."""
    envs = [(m.start(), m.end(), m.group(1)) for m in
            re.finditer(r"\\begin\{(figure|table|equation)\}.*?\\end\{\1\}", body, flags=re.S)]
    heads = list(re.finditer(r"\\(section|subsection|subsubsection)(\*?)(\[[^\]]*\])?\{", body))
    app = body.find("\\appendix")
    labels, kinds = {}, {}
    sec, sub, subsub = 0, 0, 0
    counters = {"figure": 0, "table": 0, "equation": 0}
    heading_number = {}
    for h in heads:
        appendix = 0 <= app < h.start()
        if h.group(2):
            continue
        if h.group(1) == "section":
            if appendix and sec >= 0 and not heading_number.get("_app"):
                sec, heading_number["_app"] = 0, True
            sec, sub, subsub = sec + 1, 0, 0
            num = chr(ord("A") + sec - 1) if appendix else str(sec)
        elif h.group(1) == "subsection":
            sub, subsub = sub + 1, 0
            num = f"{chr(ord('A') + sec - 1) if appendix else sec}.{sub}"
        else:
            subsub += 1
            num = f"{chr(ord('A') + sec - 1) if appendix else sec}.{sub}.{subsub}"
        heading_number[h.start()] = (num, appendix, h.group(1))
    env_number = {}
    for s, e, kind in envs:
        appendix = 0 <= app < s
        key = kind + ("_app" if appendix else "")
        counters[key] = counters.get(key, 0) + 1
        env_number[s] = ("A" if appendix else "") + str(counters[key])
    for m in re.finditer(r"\\label\{([^}]+)\}", body):
        pos = m.start()
        env = [x for x in envs if x[0] <= pos < x[1]]
        if env:
            labels[m.group(1)], kinds[m.group(1)] = env_number[env[0][0]], env[0][2]
        else:
            prev = [k for k in heading_number if isinstance(k, int) and k < pos]
            if not prev:
                raise ValueError(f"label {m.group(1)} before any heading")
            num, appendix, _ = heading_number[max(prev)]
            labels[m.group(1)] = num
            kinds[m.group(1)] = "appendix" if appendix and "." not in num else "section"

    # headings with explicit numbers (MDPI: "1. Introduction", "2.1. Data", "Appendix A. Title")
    out, last = [], 0
    for h in heads:
        title, end = group(body, h.end() - 1)
        out.append(body[last:h.start()])
        if h.start() in heading_number:
            num, appendix, level = heading_number[h.start()]
            text = (f"Appendix {num}. {title}" if appendix and level == "section" else f"{num}. {title}")
        else:
            text = title
        out.append("\\" + {"section": "section", "subsection": "subsection", "subsubsection": "subsubsection"}[h.group(1)]
                   + "*{" + text + "}")
        last = end
    out.append(body[last:])
    body = "".join(out)
    return body, labels, kinds


TEXTWIDTH_CM = 21.0 - 5.87 - 1.27   # MDPI page: A4, left margin 5.87 cm, right margin 1.27 cm (mdpi.cls geometry)


def width(opt: str | None) -> str:
    """[width=0.85\\textwidth] -> [width=11.78cm]: pandoc would take the fraction of the whole page width."""
    if not opt:
        return ""
    m = re.fullmatch(r"\[width=([\d.]+)\\textwidth\]", opt)
    if not m:
        raise ValueError(f"unexpected graphics option {opt}")
    return f"[width={float(m.group(1)) * TEXTWIDTH_CM:.2f}cm]"


def figure_table_equation(body: str, labels: dict[str, str], png: dict[str, Path]) -> str:
    def caption_prefix(env_text: str, word: str) -> str:
        m = re.search(r"\\label\{([^}]+)\}", env_text)
        num = labels[m.group(1)]
        return env_text.replace("\\caption{", f"\\caption{{\\textbf{{{word} {num}.}} ", 1)

    def fig(m):
        t = caption_prefix(m.group(0), "Figure")
        t = re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}",
                   lambda g: f"\\includegraphics{width(g.group(1))}{{{png[g.group(2)].as_posix()}}}", t)
        return t.replace("\\centering", "")

    def tab(m):
        t = caption_prefix(m.group(0), "Table")
        wide = "\\extralength" in t
        note = re.search(r"\\noindent\{\\footnotesize\{(.*)\}\}\s*\n", t, flags=re.S)
        t = re.sub(r"\\noindent\{\\footnotesize\{.*\}\}\s*\n", "", t, flags=re.S)
        t = re.sub(r"\\(begin|end)\{adjustwidth\}(\{[^}]*\}\{[^}]*\})?", "", t)
        t = re.sub(r"^\\(small|footnotesize)\s*$", "", t, flags=re.M)
        t = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", t)
        t = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", t)

        def tabx(g):
            spec = g.group(2).replace(" ", "")
            spec = spec.replace("C", "c").replace("L", "l").replace("R", "r").replace("X", "l")
            return "\\begin{tabular}{" + spec + "}"
        t = re.sub(r"\\begin\{tabularx\}\{([^}]*)\}\{([^}]*)\}", tabx, t)
        t = t.replace("\\end{tabularx}", "\\end{tabular}")
        if wide:
            t = t.replace("\\caption{", "\\caption{@@WIDE@@", 1)
        after = style("MDPI43tablefooter", note.group(1)) if note else ""
        return t + after

    def eq(m):
        lab = re.search(r"\\label\{([^}]+)\}", m.group(0))
        inner = re.sub(r"\\label\{[^}]+\}", "", m.group(1)).strip()
        return f"\n\n@@EQ:{labels[lab.group(1)]}@@$${inner}$$\n\n@@AFTEREQ@@"

    # stacked mean / sd cells of the appendix tables -> one line
    body = re.sub(r"\\begin\{tabular\}\[t\]\{@\{\}c@\{\}\}(.*?)\\\\(.*?)\\end\{tabular\}", r"\1 \2", body)
    body = re.sub(r"\\begin\{figure\}(\[[^\]]*\])?.*?\\end\{figure\}", fig, body, flags=re.S)
    body = re.sub(r"\\begin\{table\}(\[[^\]]*\])?.*?\\end\{table\}", tab, body, flags=re.S)
    body = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", eq, body, flags=re.S)
    return body


def cite_numbers() -> dict[str, int]:
    keys = re.findall(r"\\bibitem\{([^}]+)\}", read(PAPER / "bibliography.tex"))
    return {k: i + 1 for i, k in enumerate(keys)}


def compress(nums: list[int]) -> str:
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}--{nums[j]}" if j - i >= 2 else ",".join(str(x) for x in nums[i:j + 1]))
        i = j + 1
    return "[" + ",".join(parts) + "]"


def references(body: str) -> str:
    def items(m):
        out = []
        for i, it in enumerate(re.split(r"\\bibitem\{[^}]+\}", m.group(1))[1:], 1):
            out.append(style("MDPI81references", f"{i}. " + " ".join(it.split())))
        return "".join(out)
    return re.sub(r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}", items, body, flags=re.S)


def prepare() -> tuple[str, dict[str, str]]:
    main = strip_comments(read(PAPER / "main.tex"))
    vals = values()
    body = main.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
    body = flatten(body)
    body = expand_vals(body, vals)
    body, labels, kinds = number_everything(body)
    # figures as PNG at 300 dpi
    png = {}
    for g in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body))):
        src = PAPER / (g if g.endswith(".pdf") else g + ".pdf")
        dst = BUILD / "figures" / (src.stem + ".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(src), str(dst.with_suffix(""))], check=True)
        png[g] = dst
    body = figure_table_equation(body, labels, png)
    nums = cite_numbers()
    body = re.sub(r"\\cite\{([^}]+)\}", lambda m: compress([nums[k.strip()] for k in m.group(1).split(",")]), body)
    body = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels[m.group(1)], body)
    body = re.sub(r"\\label\{[^}]+\}", "", body)
    body = back_matter(body)
    body = references(body)
    body = body.replace("\\reftitle{References}", "\\section*{References}")
    for cmd in ("appendixstart", "appendix", "PublishersNote{}", "vspace{6pt}"):
        body = body.replace("\\" + cmd, "")
    body = re.sub(r"\\(begin|end)\{adjustwidth\}(\{[^}]*\}\{[^}]*\})?", "", body)
    body = body.replace("\\textvisiblespace{}", "\u2423").replace("\\textvisiblespace", "\u2423")
    main_vals = expand_vals(main, vals)
    text = "\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n" + front_matter(main_vals) + body \
        + "\n\\end{document}\n"
    leftover = sorted(set(re.findall(r"\\(?:ref|cite|val|label)\{", text)))
    if leftover:
        raise ValueError(f"unresolved commands: {leftover}")
    return text, labels


# ---------------------------------------------------------------- 4. post-processing of the docx

PSTYLE = {"Heading1": "MDPI21heading1", "Heading2": "MDPI22heading2", "Heading3": "MDPI23heading3",
          "BodyText": "MDPI31text", "FirstParagraph": "MDPI31text", "Compact": "MDPI31text",
          "ImageCaption": "MDPI51figurecaption", "CaptionedFigure": "MDPI52figure", "Figure": "MDPI52figure",
          "TableCaption": "MDPI41tablecaption"}
EQ_TABLE = ('<w:tbl><w:tblPr><w:tblW w:w="7859" w:type="dxa"/><w:tblInd w:w="2608" w:type="dxa"/>'
            '<w:tblCellMar><w:left w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>'
            '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" '
            'w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid><w:gridCol w:w="7428"/><w:gridCol w:w="431"/>'
            '</w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:w="7428" w:type="dxa"/></w:tcPr>{eq}</w:tc><w:tc><w:tcPr>'
            '<w:tcW w:w="431" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr>'
            '<w:pStyle w:val="MDPI3aequationnumber"/></w:pPr><w:r><w:t>({num})</w:t></w:r></w:p></w:tc></w:tr></w:tbl>')


def set_style(p: str, sid: str) -> str:
    if "<w:pStyle " in p:
        return re.sub(r'<w:pStyle w:val="[^"]*" ?/>', f'<w:pStyle w:val="{sid}"/>', p, count=1)
    if "<w:pPr>" in p:
        return p.replace("<w:pPr>", f'<w:pPr><w:pStyle w:val="{sid}"/>', 1)
    return re.sub(r"(<w:p(?: [^>]*)?>)", rf'\1<w:pPr><w:pStyle w:val="{sid}"/></w:pPr>', p, count=1)


def drop_marker(p: str, marker: str) -> str:
    """Remove a marker that pandoc wrote at the start of a paragraph (it may be split over runs)."""
    if marker in p:
        return p.replace(marker, "", 1)
    raise ValueError(f"marker {marker} split over runs")


def list_formats(numbering: str) -> dict[str, str]:
    """numId -> number format of level 0 (decimal for enumerate, bullet for itemize)."""
    abstract = {m.group(1): re.search(r'<w:numFmt w:val="([^"]+)"', m.group(2)).group(1)
                for m in re.finditer(r'<w:abstractNum w:abstractNumId="(\d+)"[^>]*>(.*?)</w:abstractNum>', numbering, re.S)}
    return {m.group(1): abstract.get(m.group(2), "bullet")
            for m in re.finditer(r'<w:num w:numId="(\d+)"[^>]*>\s*<w:abstractNumId w:val="(\d+)"', numbering)}


def postprocess(xml: str, formats: dict[str, str]) -> str:
    body_start = xml.index("<w:body>") + len("<w:body>")
    head, body = xml[:body_start], xml[body_start:]
    # tables first (cell paragraphs keep the table body style)
    def table(m):
        t = m.group(0)
        t = re.sub(r'<w:tblStyle w:val="[^"]*" ?/>', '<w:tblStyle w:val="MDPI41threelinetable"/>', t)
        t = re.sub(r"<w:p(?: [^>]*)?>.*?</w:p>", lambda p: set_style(p.group(0), "MDPI42tablebody"), t, flags=re.S)
        return t
    body = re.sub(r"<w:tbl>.*?</w:tbl>", table, body, flags=re.S)
    parts = re.split(r"(<w:tbl>.*?</w:tbl>)", body, flags=re.S)
    out, after_eq, wide_next = [], False, False
    for part in parts:
        if part.startswith("<w:tbl>"):
            # wide LaTeX tables span the page between the margins; the others the text column of the template
            # (indent 2608 twips, width 7859 twips, as the sample tables of the template)
            if wide_next:
                part = re.sub(r"<w:tblW [^>]*/>", '<w:tblW w:w="5000" w:type="pct"/>', part, count=1)
                wide_next = False
            else:
                part = re.sub(r"<w:tblW [^>]*/>", '<w:tblW w:w="7859" w:type="dxa"/>', part, count=1)
                part = part.replace("<w:tblPr>", '<w:tblPr><w:tblInd w:w="2608" w:type="dxa"/>', 1)
            out.append(part)
            continue
        paras = re.split(r"(<w:p(?: [^>]*)?>.*?</w:p>|<w:p/>)", part, flags=re.S)
        for p in paras:
            if not p.startswith("<w:p"):
                out.append(p)
                continue
            m = re.search(r"@@STYLE:(\w+)@@", p)
            if m:
                p = set_style(drop_marker(p, m.group(0)), m.group(1))
                if m.group(1) == "MDPI81references":
                    p = p.replace(f'<w:pStyle w:val="{m.group(1)}"/>',
                                  f'<w:pStyle w:val="{m.group(1)}"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>', 1)
                out.append(p)
                continue
            m = re.search(r"@@EQ:(\w+)@@", p)
            if m:
                p = set_style(drop_marker(p, m.group(0)), "MDPI39equation")
                out.append(EQ_TABLE.format(eq=p, num=m.group(1)))
                continue
            if "@@AFTEREQ@@" in p:
                p = drop_marker(p, "@@AFTEREQ@@")
                # the sentence continues after the equation only if the marker shares its paragraph with text
                after_eq = bool(re.search(r"<w:t[ >][^<]*\S", p))
                if not after_eq:
                    continue
            if "@@WIDE@@" in p:
                p = drop_marker(p, "@@WIDE@@")
                wide_next = True
            sm = re.search(r'<w:pStyle w:val="([^"]*)" ?/>', p)
            sid = sm.group(1) if sm else "BodyText"
            if "<w:numPr>" in p and sid in ("Compact", "BodyText", "FirstParagraph"):
                num = re.search(r'<w:numId w:val="(\d+)"', p).group(1)
                p = set_style(p, "MDPI37itemize" if formats.get(num) == "decimal" else "MDPI38bullet")
            elif sid in PSTYLE:
                new = PSTYLE[sid]
                if after_eq and new == "MDPI31text":
                    new = "MDPI32textnoindent"
                p = set_style(p, new)
            after_eq = False
            out.append(p)
    return head + "".join(out)


def main() -> int:
    if shutil.which("pandoc") is None:
        print("pandoc not found")
        return 1
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    if TEMPLATE.exists():
        make_reference()
    elif not REF.exists():
        print(f"{TEMPLATE.name} and {REF.relative_to(ROOT)} not found")
        return 1
    text, labels = prepare()
    flat = BUILD / "paper_flat.tex"
    flat.write_text(text, encoding="utf-8")
    raw = BUILD / "raw.docx"
    r = subprocess.run(["pandoc", str(flat), "-f", "latex", "-t", "docx", "--reference-doc", str(REF),
                        "--resource-path", str(BUILD), "-o", str(raw)], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr)
        return 1
    if r.stderr.strip():
        print("pandoc:", r.stderr.strip())
    (BUILD / "labels.json").write_text(json.dumps(labels, indent=1) + "\n", encoding="utf-8")
    with zipfile.ZipFile(raw) as src, zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as dst:
        formats = list_formats(src.read("word/numbering.xml").decode("utf-8"))
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/document.xml":
                data = postprocess(data.decode("utf-8"), formats).encode("utf-8")
            dst.writestr(item, data)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB); {len(labels)} labels numbered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
