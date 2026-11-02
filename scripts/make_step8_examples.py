"""Step 8 (local, no computation on models): collect the outputs of smishfuse-step8-explain, check them against
the frozen seed-1 predictions of Lite, and build the masked version of the three example messages.

Input:  results/kaggle_runs/smishfuse-step8-explain/results/step8/ (kernel output), data/processed/mendeley/test.csv
Output: results/step8/{model.json, top_ngrams.csv, test_predictions.csv, examples.json}  (copied from the kernel)
        results/step8/examples_masked.json  masked texts (phone numbers, links and codes replaced by [PHONE], [URL],
                                            [CODE]; whitespace runs collapsed as in the char analyzer)
        results/step8/example_contributions.csv  per example: n-gram (or mask label), contribution to s_smishing,
                                            to s_spam and their difference; all n-grams that overlap a masked span
                                            are summed into one row per mask label
        results/step8/example_words.csv     per example and word (runs of letters and digits; a masked span is one
                                            word): the summed contributions of the n-grams that overlap the word.
                                            The contribution of an n-gram is split equally over its occurrences and
                                            an occurrence that spans several words equally over them; the words of a
                                            message add up to its n-gram total exactly (checked)
The per-message n-gram list of the kernel (example_ngrams.csv) is not copied: together it spells the message text,
which is not redistributed (README, data rules).
Run: python scripts/make_step8_examples.py
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "kaggle_runs" / "smishfuse-step8-explain" / "results" / "step8"
OUT = ROOT / "results" / "step8"
COPY = ["model.json", "top_ngrams.csv", "top_ngrams_long.csv", "test_predictions.csv", "examples.json"]
# order matters: links first (they may contain digits), then phone numbers (5+ digits, X as a hidden digit), then codes
MASKS = [("URL", re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|co\.uk|ac|biz|info)(?:/\S*)?",
                            re.I)),
         ("PHONE", re.compile(r"\+?\d[\dX]{4,}")),
         ("CODE", re.compile(r"\b(?:code|pin|otp)[: ]+(\w+)", re.I)),  # the code itself is group 1
         # personal names, found by reading the three selected messages (only one contains a name)
         ("NAME", re.compile(r"\bNatalja\b"))]


def normalise(text: str) -> str:
    """The whitespace normalisation of sklearn's char analyzer (runs of whitespace -> one space)."""
    return re.sub(r"\s\s+", " ", text)


def mask(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    spans: list[tuple[int, int, str]] = []
    for label, rx in MASKS:
        for m in rx.finditer(text):
            g = 1 if rx.groups else 0
            if not any(a < m.end(g) and m.start(g) < b for a, b, _ in spans):
                spans.append((m.start(g), m.end(g), label))
    spans.sort()
    out, last = [], 0
    for a, b, label in spans:
        out += [text[last:a], f"[{label}]"]
        last = b
    out.append(text[last:])
    return "".join(out), spans


def owner(gram: str, lower: str, spans) -> str | None:
    """Mask label if every occurrence of the n-gram overlaps a masked span (of one label), else None."""
    starts = [m.start() for m in re.finditer(f"(?={re.escape(gram)})", lower)]
    if not starts:
        raise ValueError(f"n-gram {gram!r} not found in the normalised message")
    labels = set()
    for s in starts:
        hit = [lab for a, b, lab in spans if a < s + len(gram) and s < b]
        if not hit:
            return None
        labels.update(hit)
    return labels.pop() if len(labels) == 1 else "+".join(sorted(labels))


WORD = re.compile(r"[^\W_]+(?:[/'][^\W_]+)*")   # letters and digits, with an inner / or ' (25/F, don't)


def word_contributions(t: str, spans, grams: pd.DataFrame) -> pd.DataFrame:
    """Contributions of the words of the normalised message t (see the module docstring). Words: a masked span is one
    word; elsewhere runs of letters and digits, so that spaces, punctuation and brackets separate words. An n-gram
    is split equally over its occurrences, an occurrence equally over the words it overlaps; an occurrence made
    only of spaces and punctuation goes in equal parts to the words just before and after it."""
    lower = t.lower()
    tokens = [(a, b, f"[{lab}]") for a, b, lab in spans]
    for m in WORD.finditer(t):
        if not any(a < m.end() and m.start() < b for a, b, _ in spans):
            tokens.append((m.start(), m.end(), m.group(0)))
    tokens.sort()
    cols = ["contribution_ham", "contribution_smishing", "contribution_spam"]
    acc = {i: [0.0, 0.0, 0.0] for i in range(len(tokens))}
    for g in grams.itertuples():
        starts = [m.start() for m in re.finditer(f"(?={re.escape(g.ngram)})", lower)]
        if not starts:
            raise ValueError(f"n-gram {g.ngram!r} not in the message")
        for s0 in starts:
            s1 = s0 + len(g.ngram)
            hit = [i for i, (a, b, _) in enumerate(tokens) if a < s1 and s0 < b]
            if not hit:
                before = [i for i, (a, b, _) in enumerate(tokens) if b <= s0]
                after = [i for i, (a, b, _) in enumerate(tokens) if a >= s1]
                hit = ([before[-1]] if before else []) + ([after[0]] if after else [])
            for i in hit:
                for k, c in enumerate(cols):
                    acc[i][k] += getattr(g, c) / len(starts) / len(hit)
    rows = [{"word": tokens[i][2], **dict(zip(cols, v))} for i, v in acc.items()]
    df = pd.DataFrame(rows).groupby("word", sort=False)[cols].sum().reset_index()
    for c in cols:
        if abs(df[c].sum() - grams[c].sum()) > 1e-9:
            raise ValueError("word contributions do not add up to the n-gram contributions")
    df["diff_smishing_minus_spam"] = df.contribution_smishing - df.contribution_spam
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        shutil.copy2(SRC / name, OUT / name)
    a = pd.read_csv(OUT / "test_predictions.csv").set_index("id")
    b = pd.read_csv(ROOT / "results" / "final" / "preds" / "mendeley_3class" / "svm_char" / "clean_seed1.csv").set_index("id")
    cols = ["prob_ham", "prob_smishing", "prob_spam"]
    if not (a.index.equals(b.index) and (a.y_pred == b.y_pred).all() and np.allclose(a[cols], b[cols], atol=1e-9)):
        raise SystemExit("kernel predictions differ from the frozen seed-1 predictions of svm_char")
    print(f"predictions: {len(a)} messages, identical to results/final/preds (max diff "
          f"{np.abs(a[cols].values - b[cols].values).max():.1e})")

    ex = json.loads((OUT / "examples.json").read_text())
    texts = pd.read_csv(ROOT / "data" / "processed" / "mendeley" / "test.csv", dtype=str,
                        keep_default_na=False).set_index("id").text
    grams = pd.read_csv(SRC / "example_ngrams.csv", keep_default_na=False)
    masked, rows, words = {}, [], []
    for key, e in ex.items():
        t = normalise(texts[e["id"]])
        shown, spans = mask(t)
        masked[key] = {"id": e["id"], "text": shown, "masks": [lab for _, _, lab in spans]}
        g = grams[grams.example == key].copy()
        owners = [owner(x, t.lower(), spans) for x in g.ngram]
        g["label"] = [f"[{o}]" if o else x for x, o in zip(g.ngram, owners)]
        agg = g.groupby("label", sort=False)[["contribution_smishing", "contribution_spam", "contribution_ham"]].sum()
        agg["n_ngrams"] = g.groupby("label", sort=False).size()
        agg["diff_smishing_minus_spam"] = agg.contribution_smishing - agg.contribution_spam
        agg = agg.reset_index().assign(example=key, id=e["id"])
        rows.append(agg)
        wd = word_contributions(t, spans, g).assign(example=key, id=e["id"])
        words.append(wd)
        print(f"{key}: {e['id']}, masks {masked[key]['masks']}, {len(wd)} words")
    (OUT / "examples_masked.json").write_text(json.dumps(masked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cols = ["example", "id", "label", "n_ngrams", "contribution_ham", "contribution_smishing", "contribution_spam",
            "diff_smishing_minus_spam"]
    pd.concat(rows)[cols].to_csv(OUT / "example_contributions.csv", index=False)
    pd.concat(words)[["example", "id", "word", "contribution_ham", "contribution_smishing", "contribution_spam",
                      "diff_smishing_minus_spam"]].to_csv(OUT / "example_words.csv", index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {', '.join(sorted(p.name for p in OUT.iterdir()))}")


if __name__ == "__main__":
    main()
