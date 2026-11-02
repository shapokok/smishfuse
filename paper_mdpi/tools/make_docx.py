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

IND_ATTRS = ("left", "start", "right", "end", "firstLine", "hanging")


def explicit_indents(styles: str) -> str:
    """Write the complete indent of every paragraph style that inherits part of it (w:basedOn): Word merges an
    incomplete w:ind with the base style, LibreOffice replaces it (MDPI_3.2_text_no_indent sets only firstLine=0
    and lost the left indent of MDPI_3.1_text there)."""
    blocks = {m.group(1): m.group(0) for m in re.finditer(r'<w:style [^>]*w:styleId="([^"]+)".*?</w:style>', styles, re.S)}

    def own(sid):
        m = re.search(r"<w:ind ([^>]*?)/?>", blocks.get(sid, ""))
        return dict(re.findall(r'w:(\w+)="([^"]*)"', m.group(1))) if m else {}

    def resolved(sid, seen=()):
        b = re.search(r'<w:basedOn w:val="([^"]+)"', blocks.get(sid, ""))
        base = resolved(b.group(1), seen + (sid,)) if b and b.group(1) not in seen else {}
        mine = own(sid)
        if "firstLine" in mine:
            base.pop("hanging", None)
        if "hanging" in mine:
            base.pop("firstLine", None)
        return {**base, **mine}

    for sid, block in blocks.items():
        if '<w:basedOn ' in block and re.search(r"<w:ind ", block) and 'w:type="paragraph"' in block:
            ind = resolved(sid)
            new = "<w:ind " + " ".join(f'w:{k}="{ind[k]}"' for k in IND_ATTRS if k in ind) + "/>"
            styles = styles.replace(block, re.sub(r"<w:ind [^>]*>", new, block, count=1), 1)
    return styles


def make_reference() -> None:
    REF.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TEMPLATE) as src, zipfile.ZipFile(REF, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(b"wordprocessingml.template.main+xml", b"wordprocessingml.document.main+xml")
            elif re.fullmatch(r"word/(header|footer)\d\.xml", item.filename):
                # running head of the template: "Information 2025, 16, x" -> the volume of 2026
                data = data.replace(b"<w:t>2025</w:t>", b"<w:t>2026</w:t>")
                data = data.replace(b'<w:t xml:space="preserve"> 16</w:t>', b'<w:t xml:space="preserve"> 17</w:t>')
                # "FOR PEER REVIEW <right> N of M": the absolute tab (w:ptab) of the template is not supported by
                # LibreOffice; the paragraph has a right tab stop at the margin, so an ordinary tab does the same
                data = re.sub(rb"<w:ptab [^>]*/>", b"<w:tab/>", data)
            elif item.filename == "word/styles.xml":
                data = explicit_indents(data.decode("utf-8")).encode("utf-8")
            elif item.filename == "word/document.xml":
                text = data.decode("utf-8")
                body_start = text.index("<w:body>") + len("<w:body>")
                sect = re.findall(r"<w:sectPr[ >].*?</w:sectPr>", text, flags=re.S)[-1]
                # <w:bidi/> marks the section right-to-left; Word ignores it for this text, LibreOffice mirrors the
                # cells of tables (the number of an equation went to the left of the equation)
                sect = re.sub(r"<w:bidi ?/>", "", sect)
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
        top = t.find("\\toprule")
        n_head = t[top:t.find("\\midrule", top)].count("\\\\") if top >= 0 else 0
        t = t.replace("\\caption{", f"\\caption{{@@TAB:{'wide' if wide else 'text'}:{n_head}@@", 1)
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


SYMBOL_TEXT = {"\\ast": "*", "\\dagger": "\u2020", "\\pm": "\u00b1", "+": "+", "-": "\u2212", "<": "<", ">": ">"}


def symbols_as_text(body: str) -> str:
    """Formulas that are only signs or marks ($+$, $-$, $<$, $\\ast\\ast$, $\\dagger$) become text."""
    def repl(m):
        toks = re.findall(r"\\[a-z]+|[+\-<>]", m.group(1))
        if toks and "".join(toks) == m.group(1).replace(" ", "") and all(t in SYMBOL_TEXT for t in toks):
            return "".join(SYMBOL_TEXT[t] for t in toks)
        return m.group(0)
    return re.sub(r"(?<![\\$])\$([^$]{1,40})\$(?!\$)", repl, body)


def pair_delimiters(math: str) -> str:
    """Write every delimiter pair as \\left ... \\right: Word shows the same, and LibreOffice, which reads bare
    [ ], \\{ \\} and | | as its own grouping signs, no longer marks the formula as an error. Pairs are matched within
    one brace group and one line (\\\\ and & end a line)."""
    math = re.sub(r"\\[Bb]ig{1,2}l(?![a-z])", r"\\left", math)
    math = re.sub(r"\\[Bb]ig{1,2}r(?![a-z])", r"\\right", math)
    out, i, depth = [], 0, 0
    stack = []                       # (depth, kind, index in out)
    bars = {}                        # depth -> index in out of an open |
    while i < len(math):
        if math.startswith("\\left", i) or math.startswith("\\right", i):
            m = re.match(r"\\(left|right)(\\[{}|]|\\[a-z]+|.)", math[i:])
            out.append(m.group(0))
            i += len(m.group(0))
            continue
        if math.startswith("\\\\", i) or math[i] == "&":
            stack = [x for x in stack if x[0] < depth]
            bars.pop(depth, None)
            tok = "\\\\" if math.startswith("\\\\", i) else "&"
            out.append(tok)
            i += len(tok)
            continue
        if math.startswith("\\{", i) or math[i] == "[":
            tok = "\\{" if math[i] == "\\" else "["
            stack.append((depth, tok, len(out)))
            out.append(tok)
            i += len(tok)
            continue
        if math.startswith("\\}", i) or math[i] == "]":
            tok = "\\}" if math[i] == "\\" else "]"
            want = "\\{" if tok == "\\}" else "["
            if stack and stack[-1][0] == depth and stack[-1][1] == want:
                _, _, j = stack.pop()
                out[j] = "\\left" + want
                out.append("\\right" + tok)
            else:
                out.append(tok)
            i += len(tok)
            continue
        if math.startswith("\\", i):
            m = re.match(r"\\([a-zA-Z]+|.)", math[i:])
            out.append(m.group(0))
            i += len(m.group(0))
            continue
        c = math[i]
        if c == "{":
            depth += 1
        elif c == "}":
            stack = [x for x in stack if x[0] < depth]
            bars.pop(depth, None)
            depth -= 1
        elif c == "|":
            if depth in bars:
                out[bars.pop(depth)] = "\\left|"
                out.append("\\right|")
                i += 1
                continue
            bars[depth] = len(out)
        out.append(c)
        i += 1
    return "".join(out)


def libreoffice_safe(math: str) -> str:
    # a star as superscript (tau*, Sigma*) as text: as an operator symbol LibreOffice reads it as an operator
    math = math.replace("^{*}", "^{\\text{*}}").replace("^{\\ast}", "^{\\text{*}}")
    math = math.replace("\\begin{aligned}", "\\begin{gathered}").replace("\\end{aligned}", "\\end{gathered}")
    if "\\begin{gathered}" in math:
        math = math.replace("&", "")
    return pair_delimiters(math)


def math_segments(body: str, fn) -> str:
    """Apply fn to the content of every $$...$$ and $...$ (not to \\$)."""
    body = re.sub(r"\$\$(.*?)\$\$", lambda m: "$$" + fn(m.group(1)) + "$$", body, flags=re.S)
    return re.sub(r"(?<![\\$])\$(?!\$)((?:\\\$|[^$])+?)(?<!\\)\$(?!\$)", lambda m: "$" + fn(m.group(1)) + "$", body)


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
            out.append(style("MDPI81references", " ".join(it.split())))   # numbered by the style (template numbering)
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
    body = re.sub(r"\s*\$\\pm\$\s*", " \u00b1\u00a0", body)   # "0.9276 ±\u00a00.0025": the sd stays with its sign
    body = re.sub(r"\$\^\{((?:\\ast|\\dagger|[a-z0-9])+)\}\$",
                  lambda m: "\\textsuperscript{" + m.group(1).replace("\\ast", "*").replace("\\dagger", "\u2020") + "}", body)
    body = figure_table_equation(body, labels, png)
    body = symbols_as_text(body)
    body = math_segments(body, libreoffice_safe)
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
            '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>'
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


def list_numbering(numbering: str, document: str) -> tuple[dict[str, str], str]:
    """Map the list numbering of pandoc to the list numbering of the template: bullets -> the num of MDPI_3.8_bullet,
    each numbered list -> a new num on the abstract numbering of MDPI_3.7_itemize that restarts at 1. Returns the map
    pandoc numId -> numId and the numbering part with the new nums."""
    styles_num = {"bullet": "20", "decimal": "22"}       # numIds of MDPI_3.8_bullet and MDPI_3.7_itemize in styles.xml
    abstract = {m.group(1): m.group(2) for m in
                re.finditer(r'<w:num w:numId="(\d+)"[^>]*>\s*<w:abstractNumId w:val="(\d+)"', numbering)}
    formats = list_formats(numbering)
    used = []
    for n in re.findall(r'<w:numId w:val="(\d+)" ?/>', document):
        if n not in used and n != "0":
            used.append(n)
    mapping, new, next_id = {}, [], max(int(x) for x in abstract) + 1
    for n in used:
        if formats.get(n) == "decimal":
            mapping[n] = str(next_id)
            new.append(f'<w:num w:numId="{next_id}"><w:abstractNumId w:val="{abstract[styles_num["decimal"]]}"/>'
                       f'<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride></w:num>')
            next_id += 1
        else:
            mapping[n] = styles_num["bullet"]
    return mapping, numbering.replace("</w:numbering>", "".join(new) + "</w:numbering>")


def list_formats(numbering: str) -> dict[str, str]:
    """numId -> number format of level 0 (decimal for enumerate, bullet for itemize)."""
    abstract = {m.group(1): re.search(r'<w:numFmt w:val="([^"]+)"', m.group(2)).group(1)
                for m in re.finditer(r'<w:abstractNum w:abstractNumId="(\d+)"[^>]*>(.*?)</w:abstractNum>', numbering, re.S)}
    return {m.group(1): abstract.get(m.group(2), "bullet")
            for m in re.finditer(r'<w:num w:numId="(\d+)"[^>]*>\s*<w:abstractNumId w:val="(\d+)"', numbering)}


TEXT_TW, FULL_TW = 7859, 10466        # text column and full width between the margins of the template, twips
CELL_PAD_TW = 260                      # left + right cell margins of a Word table (108 twips each) and a reserve


# advance widths in em of Palatino (the font of the template is Palatino Linotype), measured once with Pillow from
# the system font for the characters of the tables; a reserve covers metric differences and font substitution
PALATINO_EM = {
    ' ': 0.25, '!': 0.278, '"': 0.371, '#': 0.606, '$': 0.5, '%': 0.84, '&': 0.778, "'": 0.208, '(': 0.333,
    ')': 0.333, '*': 0.389, '+': 0.606, ',': 0.25, '-': 0.333, '.': 0.25, '/': 0.606, '0': 0.5, '1': 0.5, '2': 0.5,
    '3': 0.5, '4': 0.5, '5': 0.5, '6': 0.5, '7': 0.5, '8': 0.5, '9': 0.5, ':': 0.25, ';': 0.25, '<': 0.606,
    '=': 0.606, '>': 0.606, '?': 0.444, '@': 0.747, 'A': 0.778, 'B': 0.611, 'C': 0.709, 'D': 0.774, 'E': 0.611,
    'F': 0.556, 'G': 0.763, 'H': 0.832, 'I': 0.337, 'J': 0.333, 'K': 0.726, 'L': 0.611, 'M': 0.946, 'N': 0.831,
    'O': 0.786, 'P': 0.604, 'Q': 0.786, 'R': 0.668, 'S': 0.525, 'T': 0.613, 'U': 0.778, 'V': 0.722, 'W': 1.0,
    'X': 0.667, 'Y': 0.667, 'Z': 0.667, '[': 0.333, '\\': 0.606, ']': 0.333, '^': 0.606, '_': 0.5, '`': 0.333,
    'a': 0.5, 'b': 0.553, 'c': 0.444, 'd': 0.611, 'e': 0.479, 'f': 0.333, 'g': 0.556, 'h': 0.582, 'i': 0.291,
    'j': 0.234, 'k': 0.556, 'l': 0.291, 'm': 0.883, 'n': 0.582, 'o': 0.546, 'p': 0.601, 'q': 0.56, 'r': 0.395,
    's': 0.424, 't': 0.326, 'u': 0.603, 'v': 0.565, 'w': 0.834, 'x': 0.516, 'y': 0.556, 'z': 0.5, '{': 0.333,
    '|': 0.606, '}': 0.333, '~': 0.606, '±': 0.549, 'Δ': 0.689, '–': 0.5, '†': 0.5, '−': 0.606, '␣': 0.75,
}
WIDTH_RESERVE = 1.12


def char_em(c: str) -> float:
    return PALATINO_EM.get(c, 0.0 if c == "\u200b" else 0.75)


def text_tw(text: str, pt: float) -> float:
    return sum(char_em(c) for c in text) * pt * 20 * WIDTH_RESERVE


def cell_text(xml: str) -> str:
    return "".join(re.findall(r"<(?:w|m):t(?: [^>]*)?>([^<]*)</(?:w|m):t>", xml)).replace("&amp;", "&")


def words(text: str) -> list[str]:
    """Units that Word does not break: separated by ordinary spaces only (a no-break space joins)."""
    return [x for x in re.split(r"[ \t\n]+", text) if x]


def column_widths(cells: list[list[tuple[str, int]]], ncols: int, total: int, pt: float) -> list[int]:
    """Widths in twips: every column at least as wide as its longest word (no word is broken), then towards the width
    of its longest cell (the first column, with the names of models or corpora, up to 38% of the table)."""
    word = [0.0] * ncols
    full = [0.0] * ncols
    spans = []
    for row in cells:
        col = 0
        for text, span in row:
            w_word = max([text_tw(x, pt) for x in words(text)] or [0]) + CELL_PAD_TW
            w_full = text_tw(text, pt) + CELL_PAD_TW
            if span == 1:
                word[col] = max(word[col], w_word)
                full[col] = max(full[col], w_full)
            else:
                spans.append((col, span, w_word))
            col += span
    target = [max(word[0], min(full[0], 0.38 * total))] + [max(word[c], full[c]) for c in range(1, ncols)]
    if sum(word) > total:
        raise ValueError(f"table too narrow: the longest words need {sum(word):.0f} of {total} twips")
    extra = [t - w for t, w in zip(target, word)]
    room = total - sum(word)
    if sum(extra) <= room:
        widths = target[:]
        left = room - sum(extra)
        widths = [w + left * w / sum(widths) for w in widths]
    else:
        widths = [w + e * room / sum(extra) for w, e in zip(word, extra)]
    for col, span, need in spans:   # a header that spans columns must fit its longest word too
        have = sum(widths[col:col + span])
        if have < need:
            others = [c for c in range(ncols) if not col <= c < col + span]
            take = need - have
            spare = sum(widths[c] - word[c] for c in others)
            if spare < take:
                raise ValueError("spanning header does not fit")
            for c in others:
                widths[c] -= take * (widths[c] - word[c]) / spare
            for c in range(col, col + span):
                widths[c] += take / span
    out = [int(round(w)) for w in widths]
    out[0] += total - sum(out)
    return out


RPR_AFTER_SZ = ("<w:highlight", "<w:u ", "<w:u>", "<w:effect", "<w:bdr", "<w:shd", "<w:fitText", "<w:vertAlign",
                "<w:rtl", "<w:cs", "<w:em ", "<w:lang", "<w:eastAsianLayout", "<w:specVanish", "<w:oMath")


def with_size(rpr_inner: str, sz: str) -> str:
    """Insert w:sz / w:szCs at their place in the sequence of run properties."""
    rpr_inner = re.sub(r"<w:szCs? [^>]*/>", "", rpr_inner)
    cut = min([i for i in (rpr_inner.find(t) for t in RPR_AFTER_SZ) if i >= 0] or [len(rpr_inner)])
    return rpr_inner[:cut] + sz + rpr_inner[cut:]


def font_size(xml: str, half_points: int) -> str:
    sz = f'<w:sz w:val="{half_points}"/><w:szCs w:val="{half_points}"/>'
    xml = re.sub(r"<w:r>(?:<w:rPr>(.*?)</w:rPr>)?", lambda m: f"<w:r><w:rPr>{with_size(m.group(1) or '', sz)}</w:rPr>",
                 xml, flags=re.S)
    xml = re.sub(r"<m:r>(<m:rPr>.*?</m:rPr>)?", lambda m: f"<m:r>{m.group(1) or ''}<w:rPr>{sz}</w:rPr>", xml, flags=re.S)
    return re.sub(r"</w:pPr>", f"<w:rPr>{sz}</w:rPr></w:pPr>", xml)


KEEP_TOGETHER_ROWS = 25   # longer tables (appendix) break across pages with a repeated header


def format_table(t: str, wide: bool, n_head: int) -> str:
    """Fixed column widths, font size, repeated header rows, rows that do not break across pages."""
    total, pt = (FULL_TW, 8.0) if wide else (TEXT_TW, 10.0)
    rows = re.findall(r"<w:tr(?: [^>]*)?>.*?</w:tr>", t, flags=re.S)
    cells = []
    for r in rows:
        row = []
        for tc in re.findall(r"<w:tc>.*?</w:tc>", r, flags=re.S):
            span = re.search(r'<w:gridSpan w:val="(\d+)"', tc)
            row.append((cell_text(tc), int(span.group(1)) if span else 1))
        cells.append(row)
    ncols = len(re.findall(r"<w:gridCol ", t))
    widths = column_widths(cells, ncols, total, pt)
    t = re.sub(r"<w:tblPr>(.*?)</w:tblPr>", lambda m: table_properties(m.group(1), total, wide), t, count=1, flags=re.S)
    t = re.sub(r"<w:tblGrid>.*?</w:tblGrid>",
               "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>", t, count=1, flags=re.S)
    new_rows = []
    for i, r in enumerate(rows):
        col = 0

        def cell(m):
            nonlocal col
            tc = m.group(0)
            span = re.search(r'<w:gridSpan w:val="(\d+)"', tc)
            n = int(span.group(1)) if span else 1
            tcw = f'<w:tcW w:w="{sum(widths[col:col + n])}" w:type="dxa"/>'
            col += n
            tc = re.sub(r"<w:tcW [^>]*/>", "", tc)
            tc = tc.replace("<w:tcPr />", f"<w:tcPr>{tcw}</w:tcPr>", 1) if "<w:tcPr />" in tc else \
                tc.replace("<w:tcPr>", f"<w:tcPr>{tcw}", 1)
            return tc
        r = re.sub(r"<w:tc>.*?</w:tc>", cell, r, flags=re.S)
        r = re.sub(r"<w:trPr>.*?</w:trPr>|<w:trPr ?/>", "", r, flags=re.S)
        trpr = "<w:cantSplit/>" + ("<w:tblHeader/>" if i < n_head else "")
        r = re.sub(r"(<w:tr(?: [^>]*)?>)", rf"\1<w:trPr>{trpr}</w:trPr>", r, count=1)
        new_rows.append(r)
    if len(new_rows) <= KEEP_TOGETHER_ROWS:   # a short table stays on one page with its caption
        new_rows = [re.sub(r"<w:p(?: [^>]*)?>.*?</w:p>", lambda m: keep_next(m.group(0)), r, flags=re.S)
                    for r in new_rows[:-1]] + new_rows[-1:]
    for old, new in zip(rows, new_rows):
        t = t.replace(old, new, 1)
    return font_size(t, 16) if wide else t


def keep_next(p: str) -> str:
    if "<w:keepNext" in p:
        return p
    return re.sub(r'(<w:pStyle w:val="[^"]*"/>)', r"\1<w:keepNext/>", p, count=1)


TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize", "tblW",
               "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
               "tblCaption", "tblDescription"]


def table_properties(tblpr: str, total: int, wide: bool) -> str:
    """tblPr rebuilt in the order of the OOXML schema, with width, indent and fixed layout."""
    items = {m.group(1): m.group(0) for m in
             re.finditer(r"<w:(\w+)(?: [^>]*)?(?:/>|>.*?</w:\1>)", tblpr, flags=re.S)}
    items["tblW"] = f'<w:tblW w:w="{total}" w:type="dxa"/>'
    items.pop("tblInd", None)
    if not wide:
        items["tblInd"] = '<w:tblInd w:w="2608" w:type="dxa"/>'
    items["tblLayout"] = '<w:tblLayout w:type="fixed"/>'
    unknown = set(items) - set(TBLPR_ORDER)
    if unknown:
        raise ValueError(f"unexpected table properties {unknown}")
    return "<w:tblPr>" + "".join(items[k] for k in TBLPR_ORDER if k in items) + "</w:tblPr>"


def postprocess(xml: str, formats: dict[str, str], lists: dict[str, str]) -> str:
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
    out, after_eq, pending, eq_number = [], False, None, None
    for part in parts:
        if part.startswith("<w:tbl>"):
            # wide LaTeX tables span the page between the margins (8 pt); the others the text column of the
            # template (indent 2608 twips, width 7859 twips, as its sample tables); tables without a caption
            # (abbreviations) are text-column tables without a header row
            wide, n_head = pending or (False, 0)
            out.append(format_table(part, wide, n_head))
            pending = None
            continue
        paras = re.split(r"(<w:p(?: [^>]*)?>.*?</w:p>|<w:p/>)", part, flags=re.S)
        for p in paras:
            if not p.startswith("<w:p"):
                out.append(p)
                continue
            m = re.search(r"@@STYLE:(\w+)@@", p)
            if m:
                p = set_style(drop_marker(p, m.group(0)), m.group(1))
                if m.group(1) == "MDPI81references":   # the numbering of the style, also written on the paragraph
                    p = p.replace(f'<w:pStyle w:val="{m.group(1)}"/>',
                                  f'<w:pStyle w:val="{m.group(1)}"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="23"/></w:numPr>', 1)
                out.append(p)
                continue
            m = re.search(r"@@EQ:(\w+)@@", p)
            if m:
                # pandoc writes the marker and the display equation as two paragraphs: the number is kept and the
                # equation paragraph that follows goes into the equation cell
                if eq_number is not None or re.search(r"<m:oMath", p):
                    raise ValueError(f"unexpected layout around equation {m.group(1)}")
                eq_number = m.group(1)
                continue
            if eq_number is not None:
                if "<m:oMathPara>" not in p or re.search(r"<w:t[ >]", p):
                    raise ValueError(f"equation {eq_number}: the paragraph after the marker is not a display equation")
                p = set_style(p, "MDPI39equation")
                # the equation alone in its cell as m:oMath: Word shows an equation that is alone in its paragraph
                # in display mode, and LibreOffice keeps it on the line of its number
                p = re.sub(r"<m:oMathPara>(?:<m:oMathParaPr>.*?</m:oMathParaPr>)?(.*?)</m:oMathPara>", r"\1", p, flags=re.S)
                out.append(EQ_TABLE.format(eq=p, num=eq_number))
                eq_number = None
                continue
            if "@@AFTEREQ@@" in p:
                p = drop_marker(p, "@@AFTEREQ@@")
                # the sentence continues after the equation only if the marker shares its paragraph with text
                after_eq = bool(re.search(r"<w:t[ >][^<]*\S", p))
                if not after_eq:
                    continue
            tm = re.search(r"@@TAB:(wide|text):(\d+)@@", p)
            if tm:
                p = drop_marker(p, tm.group(0))
                pending = (tm.group(1) == "wide", int(tm.group(2)))
            sm = re.search(r'<w:pStyle w:val="([^"]*)" ?/>', p)
            sid = sm.group(1) if sm else "BodyText"
            if "<w:numPr>" in p and sid in ("Compact", "BodyText", "FirstParagraph"):
                num = re.search(r'<w:numId w:val="(\d+)"', p).group(1)
                p = set_style(p, "MDPI37itemize" if formats.get(num) == "decimal" else "MDPI38bullet")
                p = re.sub(r'<w:numId w:val="\d+" ?/>', f'<w:numId w:val="{lists[num]}"/>', p, count=1)
                p = re.sub(r"<w:ind [^>]*/>", "", p)
            elif sid in PSTYLE:
                new = PSTYLE[sid]
                if after_eq and new == "MDPI31text":
                    new = "MDPI32textnoindent"
                p = set_style(p, new)
                if new in ("MDPI41tablecaption", "MDPI52figure", "MDPI21heading1", "MDPI22heading2", "MDPI23heading3"):
                    # caption with its table, image with its caption, heading with its first paragraph
                    p = keep_next(p)
                if new == "MDPI52figure":   # figures of text width centred in the text column, as in the PDF
                    p = re.sub(r"(<w:keepNext/>)", r'\1<w:ind w:left="2608"/>', p, count=1)
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
        numbering = src.read("word/numbering.xml").decode("utf-8")
        formats = list_formats(numbering)
        lists, numbering = list_numbering(numbering, src.read("word/document.xml").decode("utf-8"))
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/document.xml":
                data = postprocess(data.decode("utf-8"), formats, lists).encode("utf-8")
            elif item.filename == "word/numbering.xml":
                data = numbering.encode("utf-8")
            dst.writestr(item, data)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB); {len(labels)} labels numbered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
