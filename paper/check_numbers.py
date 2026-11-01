"""Checks the paper sources against the result files.

1. paper/numbers.tex equals what paper/tools/paper_numbers.py computes now from results/, reports/, configs/.
2. Every \\val{key} used in the text is defined.
3. No number is typed by hand in the text: after removing \\val{...}, \\ref, \\cite, \\label, layout arguments and
   comments, any remaining digit sequence must be a single digit or be listed in ALLOWED with its reason.
4. Generated tables (paper/tables/*.tex) equal what paper/tools/make_tables.py produces now (when present).
5. No em dash anywhere in the sources; the abstract has at most 200 words.
Exit status 1 on any failure. Run: python paper/check_numbers.py
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER / "tools"))

# literal numbers allowed in the text, with the reason (anything else must come from \val{...})
ALLOWED = {
    "95": "level of the bootstrap interval (2.5 and 97.5 percentiles in src/eval/stats.py) and p95 latency",
    "2.5": "lower percentile of the 95% bootstrap interval; also the model name Qwen2.5",
    "97.5": "upper percentile of the 95% bootstrap interval",
    "50": "name of the median latency p50",
}


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, PAPER / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def text_files() -> list[Path]:
    return [PAPER / "main.tex"] + sorted((PAPER / "sections").glob("*.tex"))


def strip(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", text, flags=re.S)  # drawing layout only
    text = re.sub(r"\\val\{[^}]*\}", " ", text)
    text = re.sub(r"\\(ref|eqref|cite|label|input|includegraphics|url|href|hyperlink|begin|end|setlength"
                  r"|vspace|hspace|resizebox|multicolumn|cline|cmidrule|setcounter|def)\*?(\[[^\]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\{(tabular|table|figure)\*?\}\{[^}]*\}", " ", text)
    text = re.sub(r"\[[^\]]*(width|height|scale)[^\]]*\]", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return text


def main() -> int:
    fails = []
    nums = load("paper_numbers")
    expected = nums.render(nums.build())
    current = (PAPER / "numbers.tex").read_text(encoding="utf-8")
    if current != expected:
        fails.append("numbers.tex is out of date with the result files (run paper/tools/paper_numbers.py)")
    defined = set(re.findall(r"\\@namedef\{val@([^}]+)\}", expected))

    used = set()
    for p in text_files():
        raw = p.read_text(encoding="utf-8")
        for k in re.findall(r"\\val\{([^}]+)\}", re.sub(r"(?<!\\)%.*", "", raw)):
            used.add(k)
            if k not in defined:
                fails.append(f"{p.name}: \\val{{{k}}} is not defined")
        if p.name == "main.tex":
            raw = raw.split("\\begin{document}", 1)[1]
            raw = re.sub(r"\\def\\[a-z]+\{[^}]*\}", " ", raw)
        for line_no, line in enumerate(strip(raw).splitlines(), 1):
            for tok in re.findall(r"\d+(?:\.\d+)*(?:,\d{3})*", line):
                if len(tok) == 1 or tok in ALLOWED:
                    continue
                fails.append(f"{p.name}: literal number '{tok}' (must come from \\val or be in ALLOWED)")

    for p in [*text_files(), PAPER / "references.bib"] + sorted((PAPER / "tables").glob("*.tex")):
        t = p.read_text(encoding="utf-8")
        if "\u2014" in t or "---" in t or "\\textemdash" in t:
            fails.append(f"{p.name}: em dash")

    if (PAPER / "tools" / "make_tables.py").exists():
        tabs = load("make_tables").build()
        for name, content in tabs.items():
            p = PAPER / "tables" / name
            if not p.exists() or p.read_text(encoding="utf-8") != content:
                fails.append(f"tables/{name} is out of date (run paper/tools/make_tables.py)")

    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    m = re.search(r"\\abstracttext\{(.*?)\}\s*\n\s*\\keywords", main_tex, flags=re.S)
    if m:
        words = len(re.sub(r"\\[a-zA-Z]+|[{}$\\]", " ", m.group(1)).split())
        print(f"abstract: {words} words")
        if words > 200:
            fails.append(f"abstract has {words} words (> 200)")

    print(f"values defined {len(defined)}, used {len(used)}; unused: {', '.join(sorted(defined - used)) or 'none'}")
    for f in fails:
        print("FAIL", f)
    print("OK" if not fails else f"{len(fails)} problem(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
