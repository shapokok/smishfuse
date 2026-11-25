"""Step 6d (README decision 24, POST HOC, no training): transliteration and RU/KK mixing in the Kazakhstan case study.

The operators, tables and metrics below are frozen before the pilot of smishfuse-app (the second paper): nothing here is
tuned on the results, and no model is trained on these texts.

make  data/kz_translit/cases.csv from the 28 RU / KK original rows of data/kz_cases/cases.csv (fraud and legit; the
      adapted Lite never saw them: they are its test cases) and the 4 rows of data/kz_real/public_smishing.csv:
  translit_ru_sms, translit_kk_sms   MAIN: everyday SMS transliteration, our tables (no standard exists); where a letter has
                                     several spellings, each occurrence draws one;
  translit_partial                   every word with Cyrillic letters is written in the SMS transliteration with
                                     probability 0.5 (scripts mixed inside one message);
  mix_ru_kk, mix_kk_ru               the parallel RU and KK versions of one case, sentences taken in turn (RU first /
                                     KK first); pairs with a different number of sentences or one sentence are skipped;
  translit_ru_icao                   ADDITIONAL: ICAO Doc 9303 Part 3, 8th edition 2021, Section 6 B (passports);
  translit_kk_latin2018              ADDITIONAL: the Latin alphabet of Decree No. 569 of 26.10.2017 as amended by Decree
                                     No. 637 of 19.02.2018 (the only version fixed by a decree; the 2021 and 2023
                                     drafts were not adopted). Я Ю Ё Ц Щ Э Ъ Ь are not in the decree table: OUR RULE
                                     (sounds of the table: я -> ıa, ю -> ıý, ё -> ıo, ц -> ts, щ -> sh, э -> e, ъ ь dropped).
      Only Cyrillic letters change: links, numbers, codes and Latin words stay as they are. Random stream:
      default_rng([SEED, 6, 4, sha256(id), salt]) with SEED of src/eval/obfuscation.py.
eval  Lite base and adapted from the exported JSON (models/lite_*.json of scripts/export_lite.py) through its reference
      implementation, i.e. exactly what the phone computes. Metrics: as in section 5.9 of the paper (fraud labelled
      smishing, fraud labelled ham, legit labelled smishing, correct 3-class labels) with Wilson 95% intervals; paired
      change against the source original (p(smishing), label flips, uncertain share below tau); the cost rule, post
      hoc: L_r = (r FN + FP) / n for r of configs/final.yaml under argmax and under "flag if p >= 1/(1+r)", in the view
      of the paper (malicious = smishing or spam) and of the app (alarm = smishing); share of the character n-grams
      of a message found in the vocabulary. Overlap: exact Jaccard of character 5-grams of every new text with
      data/kz_adapt/adapt.csv (>= 0.5 is reported). A demonstration with a few dozen messages, not a statistical test.
Outputs: data/kz_translit/cases.csv, results/step6d/{predictions.csv, metrics.json, overlap.csv},
         reports/step6d_kz_translit.md
Run: python scripts/make_kz_translit.py make && python scripts/make_kz_translit.py eval
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval.obfuscation import SEED  # noqa: E402

DATA = ROOT / "data" / "kz_translit" / "cases.csv"
RESULTS = ROOT / "results" / "step6d"
REPORT = ROOT / "reports" / "step6d_kz_translit.md"
CLASSES = ["ham", "smishing", "spam"]
PARTIAL_P = 0.5
SENTENCE = re.compile(r"(?<=[.!?])\s+")
CYR = re.compile(r"[Ѐ-ӿ]")

# ---------------------------------------------------------------- tables (lower case; upper case follows the source)

RU_ICAO = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "i",
           "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
           "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "ie", "ы": "y", "ь": "", "э": "e", "ю": "iu",
           "я": "ia"}

KK_DECREE = {"а": "a", "ә": "á", "б": "b", "в": "v", "г": "g", "ғ": "ǵ", "д": "d", "е": "e", "ж": "j", "з": "z", "и": "ı",
             "й": "ı", "к": "k", "қ": "q", "л": "l", "м": "m", "н": "n", "ң": "ń", "о": "o", "ө": "ó", "п": "p", "р": "r",
             "с": "s", "т": "t", "у": "ý", "ұ": "u", "ү": "ú", "ф": "f", "х": "h", "һ": "h", "і": "i", "ш": "sh",
             "ч": "ch", "ы": "y"}
KK_OUR_RULE = {"я": "ıa", "ю": "ıý", "ё": "ıo", "ц": "ts", "щ": "sh", "э": "e", "ъ": "", "ь": ""}
KK_LATIN2018 = {**KK_DECREE, **KK_OUR_RULE}

RU_SMS = {"а": ["a"], "б": ["b"], "в": ["v"], "г": ["g"], "д": ["d"], "е": ["e"], "ё": ["e", "yo"], "ж": ["zh", "j"],
          "з": ["z"], "и": ["i"], "й": ["i", "y", "j"], "к": ["k"], "л": ["l"], "м": ["m"], "н": ["n"], "о": ["o"],
          "п": ["p"], "р": ["r"], "с": ["s"], "т": ["t"], "у": ["u"], "ф": ["f"], "х": ["h", "kh", "x"],
          "ц": ["c", "ts"], "ч": ["ch", "4"], "ш": ["sh", "w"], "щ": ["sch", "sh", "shh"], "ъ": [""], "ы": ["y", "i"],
          "ь": ["", "'"], "э": ["e"], "ю": ["yu", "ju", "u"], "я": ["ya", "ja", "ia"]}
KK_SMS = {**RU_SMS, "ә": ["a", "ae"], "ғ": ["g", "gh"], "қ": ["k", "q"], "ң": ["n", "ng"], "ө": ["o", "oe"], "ұ": ["u"],
          "ү": ["u", "ue"], "һ": ["h"], "і": ["i"]}

SALT = {"translit_ru_sms": 1, "translit_kk_sms": 2, "translit_partial": 3, "translit_ru_icao": 4,
        "translit_kk_latin2018": 5}
MAIN = ("translit_ru_sms", "translit_kk_sms")


def rng_for(message_id: str, salt: int) -> np.random.Generator:
    return np.random.default_rng([SEED, 6, 4, int(hashlib.sha256(message_id.encode()).hexdigest()[:16], 16), salt])


def _cased(src: str, lat: str, word_upper: bool) -> str:
    if not lat or not src.isupper():
        return lat
    return lat.upper() if word_upper else lat[0].upper() + lat[1:]


def transliterate(text: str, table: dict, rng: np.random.Generator | None = None) -> str:
    """Cyrillic letters by the table (a list of spellings: one is drawn per occurrence), the rest unchanged."""
    out = []
    for word in re.split(r"(\s+)", text):
        letters = [c for c in word if c.isalpha()]
        word_upper = len(letters) > 1 and all(c.isupper() for c in letters)
        for c in word:
            lat = table.get(c.lower())
            if lat is None:
                out.append(c)
                continue
            if isinstance(lat, list):
                lat = lat[int(rng.integers(len(lat)))] if len(lat) > 1 else lat[0]
            out.append(_cased(c, lat, word_upper))
    return "".join(out)


def partial(text: str, table: dict, rng: np.random.Generator, p: float = PARTIAL_P) -> tuple[str, int, int]:
    """Each word with a Cyrillic letter goes to the SMS transliteration with probability p."""
    parts, done, total = [], 0, 0
    for word in re.split(r"(\s+)", text):
        if CYR.search(word):
            total += 1
            if rng.random() < p:
                word = transliterate(word, table, rng)
                done += 1
        parts.append(word)
    return "".join(parts), done, total


def sentences(text: str) -> list[str]:
    return SENTENCE.split(text.strip())


def mix(first: str, second: str) -> str | None:
    """Sentences of two parallel versions in turn, starting with `first`; None if they cannot be aligned."""
    a, b = sentences(first), sentences(second)
    if len(a) != len(b) or len(a) < 2:
        return None
    return " ".join(x if i % 2 == 0 else y for i, (x, y) in enumerate(zip(a, b)))


# ---------------------------------------------------------------- make

def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make() -> list[dict]:
    cases = [r for r in read_csv(ROOT / "data" / "kz_cases" / "cases.csv") if r["variant"] == "original"]
    real = read_csv(ROOT / "data" / "kz_real" / "public_smishing.csv")
    rows: list[dict] = []

    def add(group, source_id, case, pair, language, variant, label, text, note=""):
        rows.append({"id": f"{source_id}_{variant}" if variant != "original" else source_id, "group": group,
                     "source_id": source_id, "case": case, "pair": pair, "language": language, "variant": variant,
                     "label": label, "text": text, "note": note})

    sources = [("case", r["id"], r["id"].rsplit("_", 1)[0], r["pair"], r["language"], r["label"], r["text"])
               for r in cases if r["language"] in ("ru", "kk")]
    sources += [("real", r["id"], r["id"], "fraud", r["language"], "smishing", r["text"]) for r in real]
    for group, sid, case, pair, lang, label, text in sources:
        add(group, sid, case, pair, lang, "original", label, text)
        sms = RU_SMS if lang == "ru" else KK_SMS
        main = "translit_ru_sms" if lang == "ru" else "translit_kk_sms"
        add(group, sid, case, pair, lang, main, label, transliterate(text, sms, rng_for(sid, SALT[main])))
        t, done, total = partial(text, sms, rng_for(sid, SALT["translit_partial"]))
        add(group, sid, case, pair, lang, "translit_partial", label, t, f"{done} of {total} words")
        if lang == "ru":
            add(group, sid, case, pair, lang, "translit_ru_icao", label, transliterate(text, RU_ICAO))
        else:
            add(group, sid, case, pair, lang, "translit_kk_latin2018", label, transliterate(text, KK_LATIN2018))
    by_case: dict[str, dict] = {}
    for r in cases:
        by_case.setdefault(r["id"].rsplit("_", 1)[0], {})[r["language"]] = r
    skipped = []
    for case, v in sorted(by_case.items()):
        ru, kk = v["ru"], v["kk"]
        for variant, a, b, src in (("mix_ru_kk", ru, kk, ru), ("mix_kk_ru", kk, ru, kk)):
            text = mix(a["text"], b["text"])
            if text is None:
                skipped.append(case)
                continue
            add("case", src["id"], case, ru["pair"], "mix", variant, ru["label"], text,
                f"sentences {len(sentences(a['text']))}")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    left = [r["id"] for r in rows if r["variant"].startswith("translit_") and r["variant"] != "translit_partial"
            and CYR.search(r["text"])]
    if left:
        raise SystemExit(f"Cyrillic letters left after full transliteration: {left}")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["variant"]] = counts.get(r["variant"], 0) + 1
    print(f"wrote {DATA.relative_to(ROOT)}: {len(rows)} rows {counts}; mixing skipped for {sorted(set(skipped))}")
    return rows


# ---------------------------------------------------------------- eval

def load_models() -> dict:
    import export_lite  # noqa: PLC0415  (the reference implementation of the exported JSON)

    models = {}
    for name in ("base", "adapted"):
        path = ROOT / "models" / f"lite_{name}.json"
        if not path.exists():
            raise SystemExit(f"{path.relative_to(ROOT)} missing: run python scripts/export_lite.py first")
        m = json.loads(path.read_text(encoding="utf-8"))
        models[name] = (export_lite.prepare(m), export_lite.reference_proba,
                        hashlib.sha256(path.read_bytes()).hexdigest())
    return models


def coverage(model: dict, text: str) -> float:
    v = model["vectorizer"]
    t = re.sub(r"\s\s+", " ", text.lower() if v["lowercase"] else text)
    lo, hi = v["ngram_range"]
    grams = [t[i:i + n] for n in range(lo, min(hi + 1, len(t) + 1)) for i in range(len(t) - n + 1)]
    return sum(g in model["_index"] for g in grams) / len(grams) if grams else 0.0


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0.0, c - h)), 4), round(float(min(1.0, c + h)), 4)]


def five_grams(text: str) -> set[str]:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return {t[i:i + 5] for i in range(len(t) - 4)}


def overlap(rows: list[dict]) -> list[dict]:
    adapt = read_csv(ROOT / "data" / "kz_adapt" / "adapt.csv")
    grams = [(a["id"], five_grams(a["text"])) for a in adapt]
    out = []
    for r in rows:
        if r["variant"] == "original":
            continue
        g = five_grams(r["text"])
        best, best_id = 0.0, ""
        for aid, ag in grams:
            u = len(g | ag)
            j = len(g & ag) / u if u else 0.0
            if j > best:
                best, best_id = j, aid
        out.append({"id": r["id"], "variant": r["variant"], "max_jaccard": round(best, 4), "nearest_adapt_id": best_id})
    return out


def summarise(preds: list[dict], cost_ratios: list[float], tau: float) -> dict:
    """Counts of one cell (one model, one set of rows)."""
    fraud = [p for p in preds if p["pair"] == "fraud"]
    legit = [p for p in preds if p["pair"] == "legit"]
    n = len(preds)

    def k(rows, cond):
        return sum(1 for p in rows if cond(p))

    out = {"n": n, "n_fraud": len(fraud), "n_legit": len(legit),
           "fraud_smishing": k(fraud, lambda p: p["pred"] == "smishing"),
           "fraud_ham": k(fraud, lambda p: p["pred"] == "ham"),
           "legit_smishing": k(legit, lambda p: p["pred"] == "smishing"),
           "correct": k(preds, lambda p: p["pred"] == p["label"]),
           "uncertain": k(preds, lambda p: p["confidence"] < tau)}
    for key, (num, den) in {"fraud_smishing": (out["fraud_smishing"], len(fraud)),
                            "fraud_ham": (out["fraud_ham"], len(fraud)),
                            "legit_smishing": (out["legit_smishing"], len(legit)),
                            "correct": (out["correct"], n)}.items():
        out[key + "_ci95"] = wilson(num, den)
    cost = {}
    for view in ("paper", "app"):
        if view == "paper":  # malicious = smishing or spam (paper, section 4.5)
            y = [p["label"] != "ham" for p in preds]
            score = [1.0 - p["p_ham"] for p in preds]
            arg = [p["pred"] != "ham" for p in preds]
        else:  # the alarm of the app: smishing only
            y = [p["label"] == "smishing" for p in preds]
            score = [p["p_smishing"] for p in preds]
            arg = [p["pred"] == "smishing" for p in preds]
        for r in cost_ratios:
            thr = 1.0 / (1.0 + r)
            for rule, flag in (("argmax", arg), ("rule", [s >= thr for s in score])):
                fn = sum(1 for a, b in zip(y, flag) if a and not b)
                fp = sum(1 for a, b in zip(y, flag) if not a and b)
                cost[f"{view}_r{r:g}_{rule}"] = {"FN": fn, "FP": fp, "L": round((r * fn + fp) / n, 4) if n else None,
                                                 "threshold": round(thr, 4)}
    out["cost"] = cost
    return out


def evaluate() -> dict:
    rows = read_csv(DATA)
    models = load_models()
    import yaml  # noqa: PLC0415

    cost_ratios = list(yaml.safe_load((ROOT / "configs" / "final.yaml").read_text())["metrics"]["cost_ratios"])
    preds = []
    for name, (m, proba, _sha) in models.items():
        for r in rows:
            p, _ = proba(m, r["text"])
            preds.append({**{k: r[k] for k in ("id", "group", "source_id", "case", "pair", "language", "variant",
                                               "label")},
                          "model": name, "p_ham": float(p[0]), "p_smishing": float(p[1]), "p_spam": float(p[2]),
                          "pred": CLASSES[int(np.argmax(p))], "confidence": float(p.max()),
                          "coverage": round(coverage(m, r["text"]), 4)})
    tau = float(models["base"][0]["tau"])
    orig = {(p["model"], p["source_id"]): p for p in preds if p["variant"] == "original"}
    for p in preds:
        o = orig[(p["model"], p["source_id"])]
        p["uncertain"] = p["confidence"] < tau
        p["d_p_smishing"] = round(p["p_smishing"] - o["p_smishing"], 6)
        p["label_flip"] = p["pred"] != o["pred"]

    metrics: dict = {"tau": tau, "cost_ratios": cost_ratios,
                     "models": {n: {"file": f"models/lite_{n}.json", "sha256": s} for n, (_m, _f, s) in models.items()},
                     "cells": {}}
    variants = sorted({p["variant"] for p in preds}, key=lambda v: (v != "original", v not in MAIN, v))
    for group in ("case", "real"):
        for name in models:
            for variant in variants:
                for lang in ("ru", "kk", "mix", "all"):
                    cell = [p for p in preds if p["group"] == group and p["model"] == name and p["variant"] == variant
                            and (lang == "all" or p["language"] == lang)]
                    if not cell:
                        continue
                    s = summarise(cell, cost_ratios, tau)
                    if variant != "original":
                        d = [p["d_p_smishing"] for p in cell]
                        s["d_p_smishing_median"] = round(float(np.median(d)), 4)
                        s["d_p_smishing_min"] = round(float(np.min(d)), 4)
                        s["label_flips"] = sum(p["label_flip"] for p in cell)
                    s["coverage_median"] = round(float(np.median([p["coverage"] for p in cell])), 4)
                    metrics["cells"][f"{group}/{name}/{variant}/{lang}"] = s
    ov = overlap(rows)
    metrics["overlap"] = {"threshold": 0.5, "max_jaccard": max(o["max_jaccard"] for o in ov),
                          "n_at_or_above": sum(o["max_jaccard"] >= 0.5 for o in ov), "n": len(ov)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(preds)
    with open(RESULTS / "overlap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ov[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(ov)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    REPORT.write_text(report(metrics, preds, rows), encoding="utf-8")
    print(f"wrote {RESULTS.relative_to(ROOT)}/ and {REPORT.relative_to(ROOT)}; overlap with the adaptation set: "
          f"max Jaccard {metrics['overlap']['max_jaccard']}, {metrics['overlap']['n_at_or_above']} >= 0.5")
    return metrics


# ---------------------------------------------------------------- report

NAMES = {"original": "оригинал", "translit_ru_sms": "SMS-транслит RU", "translit_kk_sms": "SMS-транслит KK",
         "translit_partial": "частичный транслит (слово с p = 0.5)", "mix_ru_kk": "смесь по предложениям, RU первым",
         "mix_kk_ru": "смесь по предложениям, KK первым", "translit_ru_icao": "ICAO 9303 (RU)",
         "translit_kk_latin2018": "латиница указа № 637 (KK)"}


def frac(k: int, n: int) -> str:
    return f"{k}/{n}" if n else "—"


def row(metrics: dict, key: str, label: str) -> str | None:
    s = metrics["cells"].get(key)
    if not s:
        return None
    d = f"{s['d_p_smishing_median']:+.3f}" if "d_p_smishing_median" in s else "—"
    return (f"| {label} | {frac(s['fraud_smishing'], s['n_fraud'])} | {frac(s['fraud_ham'], s['n_fraud'])} | "
            f"{frac(s['legit_smishing'], s['n_legit'])} | {frac(s['correct'], s['n'])} | {frac(s['uncertain'], s['n'])} | "
            f"{d} | {s['coverage_median']:.2f} |")


HEAD = ("| вариант | мошеннические → smishing | мошеннические → ham | обычные → smishing | верная метка | "
        "неуверенно (< τ) | Δp(smishing), медиана | покрытие словаря |\n|---|---|---|---|---|---|---|---|")


def table(metrics: dict, model: str, cells: list[tuple[str, str, str]]) -> str:
    lines = [HEAD]
    for variant, lang, label in cells:
        r = row(metrics, f"case/{model}/{variant}/{lang}", label)
        if r:
            lines.append(r)
    return "\n".join(lines)


def report(metrics: dict, preds: list[dict], rows: list[dict]) -> str:
    n_case = sum(r["group"] == "case" and r["variant"] != "original" for r in rows)
    mixed = sorted({r["case"] for r in rows if r["variant"].startswith("mix_")})
    all_cases = sorted({r["case"] for r in rows if r["group"] == "case"})
    skipped = [c for c in all_cases if c not in mixed]
    out = [
        "# Шаг 6г: транслит и смесь RU/KK в кейс-стади Казахстана (решение 24, пост-хок)",
        "",
        "Пост-хок, без обучения: модели не видели ни этих текстов, ни их источников (28 оригиналов RU и KK — тестовые "
        "кейсы шага 6). Операторы, таблицы и метрики заморожены до пилота smishfuse-app. Это демонстрация на нескольких "
        "десятках сообщений, а не статистическая оценка. Доверительные интервалы (Уилсон, 95%) — в "
        "`results/step6d/metrics.json`. Смесь сравнивается с оригиналом языка первого предложения.",
        "",
        "**Главный вариант** — бытовой SMS-транслит (так пишут реальные люди; стандарта нет, таблицы наши: "
        "`scripts/make_kz_translit.py`). **Дополнительно** — официальные системы: ICAO Doc 9303 (паспорта) для RU и "
        "латиница Указа № 569 в редакции Указа № 637 от 19.02.2018 для KK. Восемь букв, которых нет в таблице указа "
        "(я, ю, ё, ц, щ, э, ъ, ь), переданы **по нашему правилу** (звуками из таблицы указа); его нужно проверить у "
        "носителя языка.",
        "",
        f"Новых текстов в кейсах: {n_case}. Смесь по предложениям — {len(mixed)} пар из {len(all_cases)}; "
        f"пропущены (одно предложение или разное число): {', '.join(skipped) or 'нет'}.",
        "",
        f"Модели: `models/lite_base.json` (sha256 `{metrics['models']['base']['sha256'][:12]}`), `models/lite_adapted.json` "
        f"(sha256 `{metrics['models']['adapted']['sha256'][:12]}`), эталонная реализация `scripts/export_lite.py` — то, "
        f"что считает телефон. τ = {metrics['tau']:.4f}.",
        "",
    ]
    for model in ("adapted", "base"):
        title = "adapted (по умолчанию в приложении)" if model == "adapted" else "base (модель статьи без изменений)"
        out += [f"## Lite {title}", "", "### Главное: SMS-транслит", "",
                table(metrics, model, [("original", "ru", "оригинал RU"), ("translit_ru_sms", "ru", "SMS-транслит RU"),
                                       ("original", "kk", "оригинал KK"), ("translit_kk_sms", "kk", "SMS-транслит KK")]),
                "", "### Смесь алфавитов и языков", "",
                table(metrics, model, [("translit_partial", "ru", "частичный транслит RU"),
                                       ("translit_partial", "kk", "частичный транслит KK"),
                                       ("mix_ru_kk", "mix", "смесь RU/KK, RU первым"),
                                       ("mix_kk_ru", "mix", "смесь RU/KK, KK первым")]),
                "", "### Дополнительно: официальные системы", "",
                table(metrics, model, [("translit_ru_icao", "ru", "ICAO 9303 (RU)"),
                                       ("translit_kk_latin2018", "kk", "латиница указа № 637 (KK)")]), ""]
    out += ["## Правило стоимости ошибок (пост-хок)", "",
            "L_r = (r·FN + FP)/n, как в разделе 4.5 статьи. Сравнивается argmax и правило «флаг, если p ≥ 1/(1+r)». "
            "Вид статьи: вредоносное = smishing или spam, p = 1 − p(ham). Вид приложения: тревога = smishing, "
            "p = p(smishing). Все тексты кейсов (оригиналы и варианты) вместе.", "",
            "| модель | вид | r | порог | FN / FP, argmax | L_r, argmax | FN / FP, правило | L_r, правило |",
            "|---|---|---|---|---|---|---|---|"]
    for model in ("adapted", "base"):
        cell = [p for p in preds if p["group"] == "case" and p["model"] == model]
        s = summarise(cell, metrics["cost_ratios"], metrics["tau"])
        for view in ("app", "paper"):
            for r in metrics["cost_ratios"]:
                a, b = s["cost"][f"{view}_r{r:g}_argmax"], s["cost"][f"{view}_r{r:g}_rule"]
                out.append(f"| {model} | {'приложение' if view == 'app' else 'статья'} | {r:g} | {a['threshold']:.3f} | "
                           f"{a['FN']} / {a['FP']} | {a['L']:.3f} | {b['FN']} / {b['FP']} | {b['L']:.3f} |")
    out += ["", f"n = {len([p for p in preds if p['group'] == 'case' and p['model'] == 'adapted'])} текстов на модель.",
            "", "## Реальные SMS из открытых источников (4, только иллюстрация)", "",
            "| сообщение | вариант | adapted: метка, p(smishing) | base: метка, p(smishing) |", "|---|---|---|---|"]
    real = {(p["model"], p["id"]): p for p in preds if p["group"] == "real"}
    for r in rows:
        if r["group"] != "real":
            continue
        a, b = real[("adapted", r["id"])], real[("base", r["id"])]
        out.append(f"| {r['source_id']} ({r['language']}) | {NAMES.get(r['variant'], r['variant'])} | "
                   f"{a['pred']}, {a['p_smishing']:.3f} | {b['pred']}, {b['p_smishing']:.3f} |")
    ov = metrics["overlap"]
    out += ["", "## Пересечение с набором адаптации", "",
            f"Точный коэффициент Жаккара символьных 5-грамм каждого нового текста с `data/kz_adapt/adapt.csv` (на нём "
            f"обучена `adapted`): максимум {ov['max_jaccard']:.3f}, текстов с Жаккаром ≥ {ov['threshold']}: "
            f"{ov['n_at_or_above']} из {ov['n']} (`results/step6d/overlap.csv`).", "",
            "## Ограничения", "",
            "- Несколько десятков синтетических сообщений, написанных авторами; реальных — 4. Числа — иллюстрация, не "
            "оценка качества.",
            "- Операторы транслита и смеси детерминированы: естественную речь со смешением языков (вставки слов, "
            "сокращения) они не моделируют. Естественный набор — позже, с носителем языка.",
            "- SMS-таблицы — наш выбор; официальные системы в SMS почти не используются.",
            "- Правило 1/(1+r) оценено пост-хок; в статье оно не оценивалось. r выбирается по данным пилота.",
            ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["make", "eval"])
    a = ap.parse_args()
    if a.what == "make":
        make()
    else:
        evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
