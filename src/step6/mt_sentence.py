"""Step 6b, POST-HOC (README decision 21): sentence-wise machine translation of the case-study messages.

Added after the step-6 results showed that NLLB often translates only the first sentence of a Russian
message. The 42 cases and their variants are unchanged (data/kz_cases/cases.csv). For every RU / KK
message: split into sentences (after . ! ? followed by a space), keep links / domains, numbers and codes
as they are (replaced by placeholder tokens before translation and put back after; a placeholder lost by
the translator is appended to the end of its sentence and counted), translate each sentence with
facebook/nllb-200-distilled-600M (same settings as step 6), join with spaces. Then the frozen Lite, Full
and BERT of step 6 predict on the result (variant mt_sentence).
Output: results/step6b/mt_sentence/translations.csv, predictions.csv, latency.json
Run (Kaggle): python src/step6/mt_sentence.py --queue '["run"]' --worker cpu
"""
from __future__ import annotations

import os

for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[var] = "1"

import argparse  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "baselines"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fusion.envcheck import check_versions  # noqa: E402
from neural.common import RESULTS, write_json  # noqa: E402
from step6 import run as step6  # noqa: E402

OUT = RESULTS / "step6b" / "mt_sentence"
# kept as is: tokens with a dot inside (links, domains, Kaspi.kz), the parcel code, numbers (with spaces,
# dots, colons or commas inside: 500 000, 12.03, 14:25)
PROTECT = re.compile(r"\S+\.[^\s.]\S*|\b[A-Z]{2}\d{9}[A-Z]{2}\b|\d(?:[\d.,:]|\s(?=\d))*\d|\d")
SENT = re.compile(r"(?<=[.!?])\s+")


def protect(sentence: str):
    spans = []

    def sub(m):
        spans.append(m.group(0))
        return f"XQ{len(spans)}"

    return PROTECT.sub(sub, sentence), spans


def restore(text: str, spans: list[str]) -> tuple[str, int]:
    lost = 0
    for i, s in reversed(list(enumerate(spans, start=1))):  # XQ10 before XQ1
        if f"XQ{i}" in text:
            text = text.replace(f"XQ{i}", s, 1)
        else:
            text = f"{text} {s}"
            lost += 1
    return text, lost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--worker", required=True)
    args = ap.parse_args()
    check_versions(args.worker, OUT / "environment")
    torch.set_num_threads(1)
    cases = pd.read_csv(ROOT / "data" / "kz_cases" / "cases.csv", dtype=str, keep_default_na=False)
    clf, tau = step6.models()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(step6.NLLB)
    mt = AutoModelForSeq2SeqLM.from_pretrained(step6.NLLB).eval()
    eng = tok.convert_tokens_to_ids("eng_Latn")

    def translate_sentence(text: str, lang: str) -> str:
        tok.src_lang = step6.SRC[lang]
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = mt.generate(**enc, forced_bos_token_id=eng)
        return tok.batch_decode(out, skip_special_tokens=True)[0]

    def translate(text: str, lang: str):
        parts, lost_total, n = [], 0, 0
        for sent in [s for s in SENT.split(text.strip()) if s]:
            masked, spans = protect(sent)
            if re.search(r"[^\W\d_]", re.sub(r"XQ\d+", "", masked)):  # anything to translate
                out = translate_sentence(masked, lang)
            else:
                out = masked
            out, lost = restore(out, spans)
            parts.append(out)
            lost_total += lost
            n += 1
        return " ".join(parts), n, lost_total

    todo = cases[cases.language.isin(list(step6.SRC))]
    for r in todo.head(step6.WARMUP).itertuples():
        translate(r.text, r.language)
    trans, t_mt, rows_t = {}, {}, []
    for r in todo.itertuples():
        (text, n, lost), sec = step6.timed(lambda t, lang=r.language: translate(t, lang), r.text)
        trans[r.id], t_mt[r.id] = text, sec
        rows_t.append({"id": r.id, "language": r.language, "variant": r.variant, "n_sentences": n,
                       "protected_lost_and_appended": lost, "translation": text, "seconds": sec})
    pd.DataFrame(rows_t).to_csv(OUT / "translations.csv", index=False, lineterminator="\n")

    for fn in clf.values():
        for t in cases.text.head(step6.WARMUP):
            fn(t)
    rows, t_model = [], {k: {} for k in clf}
    for r in todo.itertuples():
        for name, fn in clf.items():
            (p, routed), sec = step6.timed(fn, trans[r.id])
            t_model[name][r.id] = sec
            rows.append({"id": r.id, "scenario": r.scenario, "pair": r.pair, "language": r.language,
                         "variant": r.variant, "label": r.label, "input": "mt_sentence", "model": name,
                         "pred": step6.CLASSES[int(p.argmax())], "confidence": float(p.max()),
                         **{f"prob_{c}": float(p[i]) for i, c in enumerate(step6.CLASSES)},
                         "routed_to_distilbert": bool(routed) if name == "full" else None, "model_seconds": sec})
    pd.DataFrame(rows).to_csv(OUT / "predictions.csv", index=False, lineterminator="\n")
    lat = {"translation_nllb_sentencewise": step6.pct(list(t_mt.values()))}
    for name in clf:
        lat[f"translation_plus_{name}"] = step6.pct([t_mt[i] + t_model[name][i] for i in t_mt])
    write_json(lat, OUT / "latency.json")
    step6.log(f"[step6b] mt_sentence: {len(rows)} predictions, tau {tau}")


if __name__ == "__main__":
    main()
