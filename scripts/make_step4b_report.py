"""Builds reports/step4b_report.md from results/step4b/ (Kaggle notebook smishfuse-step4b).

Mean ± std (ddof = 1) over seeds of val macro F1 per condition; differences of
SmishFuse against the text-only variant and against the better classical model
(LogReg / LinearSVM) of the same setting, in percentage points. "Gain larger than
the seed spread" = mean difference > max(std of the two models); descriptive only,
no significance test.
Run: python scripts/make_step4b_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.registry import SHORT_NAMES  # noqa: E402

RESULTS = ROOT / "results"
S4 = RESULTS / "step4b"
CFG = yaml.safe_load((ROOT / "configs" / "step4b.yaml").read_text(encoding="utf-8"))
NAMES = {**SHORT_NAMES, "smishfuse": "SmishFuse (гейт)", "distilbert_text_only": "DistilBERT без признаков (4а)",
         "smishfuse_concat": "Конкатенация без гейта"}
CLASSICAL = ["linear_svm", "logreg"]


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(S4.glob("*/*/*_seed*.json")):
        r = read_json(f)
        rows.append({k: v for k, v in r.items() if not isinstance(v, (dict, list))} |
                    {"gate": r.get("gate_mean_by_true_class")})
    return pd.DataFrame(rows)


def ms(v: pd.Series) -> str:
    if v.empty:
        return "—"
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" + ("" if len(v) == len(CFG["seeds"]) else f" (n={len(v)})")


def cond_table(df, cond, settings, models, labels) -> str:
    d = df[df["condition"] == cond]
    rows = [[NAMES[m]] + [ms(d[(d["model"] == m) & (d["setting"] == s)]["macro_f1"]) for s in settings]
            for m in models]
    return table(["Модель"] + labels, rows)


def diff_rows(df, cond, settings, labels):
    d = df[df["condition"] == cond]
    rows, flags = [], []
    for s, lab in zip(settings, labels):
        get = lambda m: d[(d["model"] == m) & (d["setting"] == s)].set_index("seed")["macro_f1"]  # noqa: E731
        sf, tx = get("smishfuse"), get("distilbert_text_only")
        cl = {m: get(m) for m in CLASSICAL}
        cl = {m: v for m, v in cl.items() if len(v)}
        if sf.empty or tx.empty or not cl:
            rows.append([lab, "нет данных", "", "", "", "", ""])
            continue
        best_c = max(cl, key=lambda m: cl[m].mean())
        common = sf.index.intersection(tx.index)
        paired = (sf.loc[common] - tx.loc[common]) * 100
        d_tx = 100 * (sf.mean() - tx.mean())
        d_cl = 100 * (sf.mean() - cl[best_c].mean())
        spread_tx = 100 * max(sf.std(ddof=1), tx.std(ddof=1))
        spread_cl = 100 * max(sf.std(ddof=1), cl[best_c].std(ddof=1))
        above_tx = abs(d_tx) > spread_tx
        above_cl = abs(d_cl) > spread_cl
        same_sign = "да" if (np.sign(paired) == np.sign(paired.iloc[0])).all() and paired.iloc[0] != 0 else "нет"
        rows.append([lab, f"{d_tx:+.2f}", ", ".join(f"{x:+.2f}" for x in paired), same_sign,
                     f"{spread_tx:.2f}", "да" if above_tx else "нет",
                     f"{NAMES[best_c]}: {d_cl:+.2f} (разброс {spread_cl:.2f}, {'больше' if above_cl else 'не больше'})"])
        flags.append((cond, lab, d_tx, above_tx, d_cl, above_cl))
    md = table(["Настройка", "SmishFuse − текст, п.п.", "По seed, п.п.", "Один знак", "Разброс (max std), п.п.",
                "Больше разброса", "SmishFuse − лучший классический, п.п."], rows)
    return md, flags


def gate_table(df, cond, settings, labels, classes) -> str:
    d = df[(df["condition"] == cond) & (df["model"] == "smishfuse")]
    rows = []
    for s, lab in zip(settings, labels):
        g = [x for x in d[d["setting"] == s]["gate"] if isinstance(x, dict)]
        if g:
            rows.append([lab] + [f"{np.mean([x[c] for x in g]):.3f}" for c in classes])
    return table(["Настройка"] + [f"g ({c})" for c in classes], rows) if rows else "Нет данных."


def main():
    md = ["# SmishFuse, шаг 4б: стресс-условия (только val)\n",
          "Сгенерировано `scripts/make_step4b_report.py` из `results/step4b/`. Предварительная оценка: "
          f"{len(CFG['seeds'])} seed, только val, **без проверки значимости**. Test не читался. Гиперпараметры "
          "моделей — шаг 4а, без подбора.\n"]
    if not S4.exists():
        md.append("Результатов нет.")
        (ROOT / "reports" / "step4b_report.md").write_text("\n".join(md), encoding="utf-8")
        return
    df = load()
    seeds = CFG["seeds"]

    # environment
    md.append("## 0. Окружение\n")
    envs = sorted((S4 / "environment").glob("*.json")) if (S4 / "environment").exists() else []
    for f in envs:
        e = read_json(f)
        md.append(f"- `{f.stem}`: расхождений с `requirements.txt`: "
                  f"{len(e['mismatch'])}" + (f" ({', '.join(e['mismatch'])})" if e["mismatch"] else "") + "; " +
                  ", ".join(f"{k} {v['installed']}" for k, v in e["packages"].items()
                            if k in ("numpy", "pandas", "scikit-learn", "xgboost", "torch", "transformers")) + ".")
    md.append("")

    # reference
    md.append("## 1. Ориентиры: baseline шагов 2 и 3 на чистом val (mendeley_3class)\n")
    md.append(f"Лучшие параметры шагов 2–3, seed {seeds[0]}–{seeds[-1]}, закреплённое окружение. Для Qwen seed — "
              "набор из 8 примеров. Дальше сравнение только с этими числами.\n")
    base = df[df["condition"] == "baseline"]
    rows = []
    for m in CFG["baselines"]:
        v = base[base["model"] == m]
        if v.empty:
            rows.append([NAMES[m], "нет результата", "", ""])
            continue
        inv = f"{100 * v['invalid_rate'].mean():.2f}%" if "invalid_rate" in v and v["invalid_rate"].notna().any() else ""
        rows.append([NAMES[m], ms(v["macro_f1"]), ms(v["f1_spam"]), inv])
    md.append(table(["Модель", "Macro F1", "F1 spam", "Невалидные ответы"], rows))
    md.append("")

    all_flags = []
    classes3 = ["ham", "smishing", "spam"]
    models = CFG["compared"]

    # a) obfuscation
    lv = CFG["obfuscation"]["levels"]
    st = ["clean"] + [f"obf{int(round(x * 100)):02d}" for x in lv]
    labs = ["чистый val"] + [f"{int(round(x * 100))}%" for x in lv]
    md.append("## 2а. Обфускация val\n")
    stats = [read_json(f) for f in sorted((S4 / "obfuscation" / "levels").glob("*.json"))] \
        if (S4 / "obfuscation" / "levels").exists() else []
    md.append("Обучение на чистом train; одна фиксированная обфускация val на уровень (`src/eval/obfuscation.py`): "
              "замены 0/o/O, 1/l/I, a/A/@, e/E/3, s/S/$, гомоглифы кириллицы (а е о р с), пробелы внутри слов. "
              "Фактическая доля изменённых символов: " +
              ", ".join(f"{int(round(s['level'] * 100))}% → {100 * s['realised_share']:.2f}%" for s in stats) + ".\n")
    md.append(cond_table(df, "obfuscation", st, models, labs))
    t, f = diff_rows(df, "obfuscation", st, labs)
    md += ["", t, ""]
    all_flags += f
    md.append("Средний вес гейта g (ветвь признаков) по истинному классу:\n")
    md += [gate_table(df, "obfuscation", st, labs, classes3), ""]

    # b) low data
    fr = CFG["low_data"]["fractions"]
    st = [f"frac{int(round(x * 100)):02d}" for x in fr]
    subs = {s: read_json(S4 / "low_data" / "subsets" / f"{s}.json") for s in st
            if (S4 / "low_data" / "subsets" / f"{s}.json").exists()}
    labs = [f"{int(round(x * 100))}% (n={subs[s]['n']})" if s in subs else f"{int(round(x * 100))}%"
            for x, s in zip(fr, st)]
    md.append("## 2б. Мало данных\n")
    md.append("Вложенные стратифицированные подвыборки train (одинаковые для всех моделей, seed подвыборки "
              f"{CFG['low_data']['subset_seed']}), не меньше {CFG['low_data']['min_per_class']} сообщений на класс "
              "(5-кратная калибровка LinearSVM); оценка на полном val, признаки стандартизуются по подвыборке. "
              "Состав: " + "; ".join(f"{s}: " + ", ".join(f"{c} {n}" for c, n in v["per_class"].items())
                                     for s, v in subs.items()) + ".\n")
    md.append(cond_table(df, "low_data", st, models, labs))
    t, f = diff_rows(df, "low_data", st, labs)
    md += ["", t, ""]
    all_flags += f
    md.append("Средний вес гейта g по истинному классу:\n")
    md += [gate_table(df, "low_data", st, labs, classes3), ""]

    # c) transfer
    tc = CFG["transfer"]
    ft = f"finetune{int(round(tc['finetune_fraction'] * 100)):02d}"
    sub = S4 / "transfer" / "subsets" / f"{ft}.json"
    ftn = read_json(sub) if sub.exists() else None
    labs = ["zero-shot", f"дообучение на {int(round(tc['finetune_fraction'] * 100))}% MOZ" +
            (f" (n={ftn['n']})" if ftn else "")]
    md.append("## 2в. Перенос языка: mendeley_binary (англ.) → moz_binary (порт.)\n")
    md.append(f"Текстовая ветвь `{tc['checkpoint']}`; бинарная задача — одна голова. Zero-shot: обучение на "
              "mendeley_binary (ранняя остановка по val mendeley_binary), оценка на val moz_binary. Дообучение: "
              "та же нейросеть дообучается на стратифицированной подвыборке train MOZ (ранняя остановка по val "
              "MOZ), признаки стандартизованы по train mendeley; LinearSVM и LogReg дообучать нельзя — они "
              "заново обучены на train mendeley_binary + подвыборка MOZ (параметры шага 2 для mendeley_binary).\n")
    md.append(cond_table(df, "transfer", ["zeroshot", ft], models, labs))
    t, f = diff_rows(df, "transfer", ["zeroshot", ft], labs)
    md += ["", t, ""]
    all_flags += f
    md.append("Средний вес гейта g по истинному классу:\n")
    md += [gate_table(df, "transfer", ["zeroshot", ft], labs, ["benign", "malicious"]), ""]

    # summary of flags
    md.append("## 3. Где признаки дают выигрыш больше разброса по seed\n")
    md.append("Критерий (описательный, не тест): |разница средних SmishFuse и текстовой модели| больше "
              "наибольшего из двух стандартных отклонений по seed.\n")
    pos = [f"{c} / {lab} ({d:+.2f} п.п.)" for c, lab, d, a, _, _ in all_flags if a and d > 0]
    neg = [f"{c} / {lab} ({d:+.2f} п.п.)" for c, lab, d, a, _, _ in all_flags if a and d < 0]
    none = [f"{c} / {lab} ({d:+.2f})" for c, lab, d, a, _, _ in all_flags if not a]
    md += [f"- Выигрыш больше разброса: {'; '.join(pos) if pos else 'нигде'}.",
           f"- Проигрыш больше разброса: {'; '.join(neg) if neg else 'нигде'}.",
           f"- Разница в пределах разброса: {'; '.join(none) if none else 'нигде'}.", ""]
    posc = [f"{c} / {lab} ({dc:+.2f} п.п.)" for c, lab, _, _, dc, ac in all_flags if ac and dc > 0]
    md.append("Против лучшего классического (LogReg / LinearSVM) выигрыш SmishFuse больше разброса: "
              + ("; ".join(posc) if posc else "нигде") + ".\n")

    md.append("## 3а. Наблюдения\n")
    md += [f"- {o}" for o in OBSERVATIONS]
    md.append("")

    # GPU hours
    md.append("## 4. GPU-часы\n")
    rt = RESULTS / "kaggle_runs" / "smishfuse-step4b" / "results" / "runtime" / "smishfuse-step4b.json"
    if rt.exists():
        sess = read_json(rt)["sessions"]
        md.append(table(["Сессия", "Часы", "Коды возврата"],
                        [[i + 1, f"{s['wall_seconds'] / 3600:.2f}", ", ".join(map(str, s["return_codes"]))]
                         for i, s in enumerate(sess)]))
        md.append(f"\nВсего {sum(s['wall_seconds'] for s in sess) / 3600:.2f} ч сессии 2× T4 (бюджет 4).\n")
    else:
        md.append("Нет данных.\n")

    # problems
    md.append("## 5. Проблемы и ограничения\n")
    probs = []
    for f in sorted((S4 / "errors").glob("*.txt")) if (S4 / "errors").exists() else []:
        probs.append(f"Ошибка `{f.stem}`: `{f.read_text(encoding='utf-8').strip().splitlines()[-1][:200]}`.")
    for f in sorted((S4 / "job_times").glob("*.json")) if (S4 / "job_times").exists() else []:
        nr = [j["job"] for j in read_json(f)["jobs"] if j["status"] == "not_run_budget"]
        if nr:
            probs.append(f"`{f.stem}`: остановлено по бюджету перед `{nr[0]}`.")
    probs += PROBLEMS
    md += [f"- {p}" for p in probs]
    md.append("")
    md += SUMMARY
    (ROOT / "reports" / "step4b_report.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/step4b_report.md")


OBSERVATIONS = [
    "Воспроизводимость: в закреплённом окружении SmishFuse, текстовая модель и конкатенация с seed 1 дали на "
    "чистом val те же предсказания, что в шаге 4а (574 из 574), а вероятности LogReg и LinearSVM совпали с шагом "
    "4а точно (max |Δp| = 0). Значит, версии numpy / pandas на них не влияют, и расхождение со значениями подбора "
    "шага 2 (LogReg 0.9207 локально против 0.9056 на Kaggle) — не из-за версий пакетов, а, вероятно, из-за "
    "платформы (CPU / BLAS Kaggle против Apple M); это не проверялось. Предположение шага 4а о незакреплённых "
    "numpy и pandas не подтвердилось.",
    "Чистый val: SmishFuse 0.9292 ± 0.0026 против лучшего ориентира BERT 0.9239 ± 0.0053 (+0.53 п.п., в пределах "
    "разброса BERT). Против текстовой модели +0.44 п.п. при разнице по seed +0.00 / +1.20 / +0.13: формально чуть "
    "больше разброса, но на seed 1 предсказания совпадают полностью.",
    "Обфускация: нейросети с текстом падают сильнее всего (при 20%: 0.54–0.63), модели на TF-IDF с символьными "
    "n-граммами держатся (0.82–0.83). При 20% признаки дают SmishFuse +4.66 п.п. к текстовой модели на всех "
    "трёх seed, но SmishFuse при этом на 19.7 п.п. хуже LinearSVM. У LogReg / LinearSVM качество на 10% выше, "
    "чем на чистом val: у LogReg (seed 1) меняются 7 предсказаний из 574, 4 исправляются и 2 портятся; при "
    "49 smishing и 42 spam в val это заметно сдвигает macro F1. Уровни обфускации — независимые случайные "
    "реализации, поэтому немонотонность в пределах шума одной реализации; это не исследовалось дальше.",
    "Мало данных: признаки не помогают — SmishFuse не лучше текстовой модели ни на одной доле (на 5% и 25% хуже "
    "больше разброса). Нейросети лучше LogReg на всех долях; LinearSVM на 10% выше SmishFuse и текстовой "
    "модели, но ниже конкатенации, на остальных долях ниже всех нейросетей.",
    "Перенос языка: zero-shot у всех моделей слабый (нейросети 0.54–0.56, LogReg / LinearSVM 0.36–0.41 macro F1; "
    "LinearSVM (seed 1) помечает 66 сообщений как malicious, из них верно 1 из 54 истинных). После дообучения на 177 "
    "сообщениях MOZ лучшая — текстовая модель (0.9628), SmishFuse на 1.16 п.п. ниже на всех seed.",
    "Вес гейта g во всех условиях между 0.39 и 0.61 и мало меняется от условия к условию; при этом на чистом "
    "val (seed 1) решения SmishFuse совпадают с текстовой моделью, так что g не показывает вклад признаков.",
]
PROBLEMS = [
    "Ранняя остановка нейросетей — по чистому val (в переносе — по val источника для zero-shot и по val MOZ "
    "для дообучения), параметры LogReg / LinearSVM подобраны на этом же val в шаге 2: числа на val "
    "оптимистичны для всех моделей одинаково.",
    "3 seed дают грубую оценку разброса; критерий «больше разброса» не заменяет проверку значимости.",
    "Ноутбук `smishfuse-step4b` отправлен дважды подряд (первая команда отправки прошла, но её вывод не "
    "отобразился, вторая создала версию 2). Отменить версию 1 через CLI нельзя, поэтому неизвестно, тратила "
    "ли она квоту; в разделе 4 — только сессия версии 2.",
    "Для переноса признаки MOZ стандартизованы по train mendeley и в zero-shot, и при дообучении; словари "
    "срочности / выигрыша и проверка орфографии — английские, на португальском они неинформативны.",
]
SUMMARY = [
    "## 6. Итог\n",
    "Сделано: закреплённое окружение (все версии `requirements.txt`, проверка в ноутбуке), 9 baseline на val с "
    "seed 1–3, 5 моделей × 3 условия × 3 seed; 0.54 ч GPU-сессии.\n",
    "Требует решения:\n",
    "- Признаки дают выигрыш больше разброса только при сильной обфускации (20%) и формально на чистом val; в "
    "малых данных и переносе SmishFuse не лучше или хуже текстовой модели. Продолжать ли эту архитектуру, менять "
    "ли признаки (например, на устойчивые к обфускации) или слияние.",
    "- При обфускации TF-IDF с символьными n-граммами устойчивее всех нейросетей; стоит ли добавить такую ветвь "
    "в SmishFuse.",
    "- Ориентиры шага 2 на Kaggle и на маке расходятся (LogReg 0.9056 против 0.9207 при одинаковых версиях "
    "пакетов): какие числа считать основными в статье для классических моделей (сейчас test шага 2 посчитан "
    "локально, кроме LinearSVM).",
    "- Квота версии 1 ноутбука (раздел 5) — проверить на странице квоты Kaggle.",
    "",
]


if __name__ == "__main__":
    main()
