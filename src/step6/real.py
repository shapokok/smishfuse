"""Step 6c, POST-HOC (README decision 22): the 4 public Kazakhstan smishing SMS (data/kz_real/public_smishing.csv,
fixed by commit 1596963) through the frozen and adapted models. Qualitative illustration, no metric.

Models: Lite before (final svm_char, seed-1 weights), Lite adapted A and B (re-trained here exactly as in
src/step6/adapt.py with all adaptation messages: same code, data and seed 1; their adaptation weights were not
saved; Mendeley VAL macro F1 is logged to compare with step 6b), Full (cascade), BERT. Inputs: the message as
written and the sentence-wise NLLB translation of step 6b (mt_sentence).
Output: results/step6c/predictions.csv, translations.csv, check.json
Run (Kaggle): python src/step6/real.py --queue '["run"]' --worker cpu
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from fusion.envcheck import check_versions  # noqa: E402
from neural.common import RESULTS, write_json  # noqa: E402
from step6 import mt_sentence  # noqa: E402
from step6 import run as step6  # noqa: E402

OUT = RESULTS / "step6c"
CLASSES = step6.CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--worker", required=True)
    args = ap.parse_args()
    check_versions(args.worker, OUT / "environment")
    torch.set_num_threads(1)
    from final import run

    real = pd.read_csv(ROOT / "data" / "kz_real" / "public_smishing.csv", dtype=str, keep_default_na=False)
    clf, tau = step6.models()  # lite (before), full, bert

    # adapted Lite A / B, re-trained as in src/step6/adapt.py (all adaptation messages, seed 1)
    tr_ids, tr_texts, tr_y = run.data("mendeley_3class", "train")
    _, va_texts, va_y = run.data("mendeley_3class", "val")
    adapt = pd.read_csv(ROOT / "data" / "kz_adapt" / "adapt.csv", dtype=str, keep_default_na=False)
    adapt["y"] = adapt["label"].map(CLASSES.index)
    check = {}
    for cond, pool in (("A", adapt[adapt.same_category_as_test == "false"]), ("B", adapt)):
        m = run.Classic("svm_char", "mendeley_3class").fit(
            np.concatenate([tr_ids, pool["id"].to_numpy(dtype=object)]),
            np.concatenate([tr_texts, pool["text"].to_numpy(dtype=object)]),
            np.concatenate([tr_y, pool["y"].to_numpy()]), 1)
        check[f"lite_adapted_{cond}_val_macro_f1"] = float(f1_score(va_y, m.proba(va_texts).argmax(1), labels=[0, 1, 2],
                                                                    average="macro", zero_division=0))
        clf[f"lite_adapted_{cond}"] = (lambda t, m=m: (m.proba([t])[0], False))
    step6.log(f"[step6c] adapted: {check}")

    # sentence-wise translation (step 6b procedure)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(step6.NLLB)
    mt = AutoModelForSeq2SeqLM.from_pretrained(step6.NLLB).eval()
    eng = tok.convert_tokens_to_ids("eng_Latn")

    def translate(text: str, lang: str) -> str:
        parts = []
        for sent in [s for s in mt_sentence.SENT.split(text.strip()) if s]:
            masked, spans = mt_sentence.protect(sent)
            if re.search(r"[^\W\d_]", re.sub(r"XQ\d+", "", masked)):
                tok.src_lang = step6.SRC[lang]
                with torch.no_grad():
                    out = mt.generate(**tok(masked, return_tensors="pt"), forced_bos_token_id=eng)
                masked = tok.batch_decode(out, skip_special_tokens=True)[0]
            parts.append(mt_sentence.restore(masked, spans)[0])
        return " ".join(parts)

    trans = {r.id: translate(r.text, r.language) for r in real.itertuples()}
    pd.DataFrame([{"id": i, "translation": t} for i, t in trans.items()]).to_csv(OUT / "translations.csv", index=False,
                                                                                lineterminator="\n")
    rows = []
    for r in real.itertuples():
        for inp, text in (("as_written", r.text), ("mt_sentence", trans[r.id])):
            for name, fn in clf.items():
                p, routed = fn(text)
                p = step6.as_probs(p)
                rows.append({"id": r.id, "language": r.language, "verbatim": r.verbatim, "input": inp, "model": name,
                             "pred": CLASSES[int(p.argmax())], "confidence": float(p.max()),
                             **{f"prob_{c}": float(p[i]) for i, c in enumerate(CLASSES)},
                             "routed_to_distilbert": bool(routed) if name == "full" else None})
    pd.DataFrame(rows).to_csv(OUT / "predictions.csv", index=False, lineterminator="\n")
    write_json({**check, "tau": tau}, OUT / "check.json")
    step6.log(f"[step6c] {len(rows)} predictions")


if __name__ == "__main__":
    main()
