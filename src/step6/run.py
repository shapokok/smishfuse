"""Step 6: Kazakhstan case study with the FROZEN final models (no training, no change), Kaggle CPU.

Messages: data/kz_cases/cases.csv (fixed by commit 924ff27 before this run). Models, seed-1 weights of
smishfuse-final-3class (kernel source), mendeley_3class as they are:
  lite  = char LinearSVM without augmentation (svm_char);
  full  = cascade svm_char_aug -> distilbert_aug with the frozen tau (configs/final.yaml);
  bert  = BERT (best step-3 parameters) for comparison.
Inputs: a) the message as written (EN / RU / KK, all variants); b) RU and KK messages machine-translated
to English with facebook/nllb-200-distilled-600M (rus_Cyrl / kaz_Cyrl -> eng_Latn, CPU, the model's
default generation settings, forced eng_Latn start token). One thread, batch 1, as the step-5b timing.
Output: results/step6/translations.csv, predictions.csv, latency.json, run_config.json
Run (Kaggle): python src/step6/run.py --queue '["run"]' --worker cpu
"""
from __future__ import annotations

import os

for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[var] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "baselines"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fusion.envcheck import check_versions  # noqa: E402
from neural.common import RESULTS, load_yaml, write_json  # noqa: E402

OUT = RESULTS / "step6"
NLLB = "facebook/nllb-200-distilled-600M"
SRC = {"ru": "rus_Cyrl", "kk": "kaz_Cyrl"}
CLASSES = ["ham", "smishing", "spam"]
WARMUP = 3


def log(msg: str) -> None:
    print(msg, flush=True)


def as_probs(p) -> np.ndarray:
    p = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
    return np.asarray(p, dtype=float).reshape(-1)[: len(CLASSES)]


def models():
    from final import latency

    fns = {}
    for m in ["svm_char", "svm_char_aug", "distilbert_aug", "bert"]:
        path = latency.find("mendeley_3class", m)
        if path is None:
            raise SystemExit(f"weights of {m} not found (kernel source smishfuse-final-3class)")
        fns[m] = latency.load("mendeley_3class", m, path)
        log(f"[step6] {m}: {path}")
    tau = load_yaml("configs/final.yaml")["method"]["tau"]

    def full(t):
        p1 = as_probs(fns["svm_char_aug"](t))
        if p1.max() >= tau:
            return p1, False
        return as_probs(fns["distilbert_aug"](t)), True

    return {"lite": lambda t: (as_probs(fns["svm_char"](t)), False), "full": full,
            "bert": lambda t: (as_probs(fns["bert"](t)), False)}, tau


def timed(fn, t):
    start = time.perf_counter()
    out = fn(t)
    return out, time.perf_counter() - start


def pct(a) -> dict:
    a = np.asarray(a)
    return {"n": int(len(a)), "p50_ms": float(np.median(a) * 1000), "p95_ms": float(np.percentile(a, 95) * 1000),
            "mean_ms": float(a.mean() * 1000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--worker", required=True)
    args = ap.parse_args()
    check_versions(args.worker, OUT / "environment")
    torch.set_num_threads(1)
    cases = pd.read_csv(ROOT / "data" / "kz_cases" / "cases.csv", dtype=str, keep_default_na=False)
    clf, tau = models()

    # b) machine translation of every RU / KK message (all variants)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(NLLB)
    mt = AutoModelForSeq2SeqLM.from_pretrained(NLLB).eval()
    eng = tok.convert_tokens_to_ids("eng_Latn")

    def translate(text: str, lang: str) -> str:
        tok.src_lang = SRC[lang]
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = mt.generate(**enc, forced_bos_token_id=eng)
        return tok.batch_decode(out, skip_special_tokens=True)[0]

    todo = cases[cases.language.isin(list(SRC))]
    for r in todo.head(WARMUP).itertuples():
        translate(r.text, r.language)
    trans, t_mt = {}, {}
    for r in todo.itertuples():
        trans[r.id], t_mt[r.id] = timed(lambda t, lang=r.language: translate(t, lang), r.text)
    pd.DataFrame([{"id": i, "language": cases.set_index("id").loc[i, "language"], "translation": trans[i],
                   "seconds": t_mt[i]} for i in trans]).to_csv(OUT / "translations.csv", index=False, lineterminator="\n")
    log(f"[step6] translated {len(trans)} messages")

    # predictions: every message as written, and RU / KK also after translation
    for fn in clf.values():  # warm-up
        for t in cases.text.head(WARMUP):
            fn(t)
    rows, t_model = [], {k: {} for k in clf}
    for r in cases.itertuples():
        inputs = [("original", r.text)] + ([("translated_en", trans[r.id])] if r.id in trans else [])
        for inp, text in inputs:
            for name, fn in clf.items():
                (p, routed), sec = timed(fn, text)
                if inp == "translated_en":
                    t_model[name][r.id] = sec
                rows.append({"id": r.id, "scenario": r.scenario, "pair": r.pair, "language": r.language,
                             "variant": r.variant, "label": r.label, "input": inp, "model": name,
                             "pred": CLASSES[int(p.argmax())], "confidence": float(p.max()),
                             **{f"prob_{c}": float(p[i]) for i, c in enumerate(CLASSES)},
                             "routed_to_distilbert": bool(routed) if name == "full" else None,
                             "model_seconds": sec})
    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "predictions.csv", index=False, lineterminator="\n")

    lat = {"translation_nllb": pct(list(t_mt.values()))}
    for name in clf:
        lat[f"model_{name}_on_translation"] = pct(list(t_model[name].values()))
        lat[f"translation_plus_{name}"] = pct([t_mt[i] + t_model[name][i] for i in t_mt])
    write_json(lat, OUT / "latency.json")
    write_json({"nllb": NLLB, "src_lang": SRC, "tgt_lang": "eng_Latn", "generation": mt.generation_config.to_dict(),
                "threads": 1, "tau": tau, "models": {"lite": "svm_char", "full": "cascade svm_char_aug -> distilbert_aug",
                                                     "bert": "bert"}, "weights": "seed 1, smishfuse-final-3class"},
               OUT / "run_config.json")
    log(f"[step6] done: {len(pred)} predictions; NLLB p50 {lat['translation_nllb']['p50_ms']:.0f} ms")


if __name__ == "__main__":
    main()
