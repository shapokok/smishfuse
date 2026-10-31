"""Builds reports/step6_kz_cases.md from results/step6/ (Kaggle notebook smishfuse-step6) and
data/kz_cases/cases.csv. Counting only (no model is run here).
Run: python scripts/make_step6_report.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "step6"
MODELS = {"lite": "Lite (LinearSVM char)", "full": "Full (каскад)", "bert": "BERT"}
INPUTS = {"original": "как написано", "translated_en": "после перевода на EN"}
VARIANTS = {"original": "исходный", "heldout10": "HELD-OUT 10%", "cyr_homoglyph": "кир. гомоглифы 10%"}
LANGS = {"en": "EN", "ru": "RU", "kk": "KK"}
SHORT = {"ham": "ham", "smishing": "smish", "spam": "spam"}


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    cases = pd.read_csv(ROOT / "data" / "kz_cases" / "cases.csv", dtype=str, keep_default_na=False)
    p = pd.read_csv(R / "predictions.csv", keep_default_na=False)
    tr = pd.read_csv(R / "translations.csv", keep_default_na=False).set_index("id")
    lat = json.loads((R / "latency.json").read_text())
    cfg = json.loads((R / "run_config.json").read_text())
    p["correct"] = p["pred"] == p["label"]

    md = ["# SmishFuse, шаг 6: кейс-стади в казахстанском контексте\n",
          "Сгенерировано `scripts/make_step6_report.py` из `results/step6/` (ноутбук `smishfuse-step6`, Kaggle CPU).\n"]
    md += [f"- {s}" for s in INTRO] + [""]

    md.append("## 1. Набор\n")
    md.append(table(["Сценарий", "Мошенническое (метка)", "Легитимное (метка)"],
                    [[sc, "smishing", cases[(cases.scenario == sc) & (cases.pair == "legit")].label.iloc[0]]
                     for sc in cases.scenario.unique()]))
    md.append("\nКаждое сообщение — на EN, RU и KK (42 сообщения). Мошеннические RU — также с замороженной "
              "обфускацией HELD-OUT 10% (7). Мошеннические RU и KK — с вариантом `cyr_homoglyph` (14), он "
              "создан только для кейс-стади. Тексты — `data/kz_cases/cases.csv`, переводы NLLB — "
              "`results/step6/translations.csv`.\n")

    md.append("## 2. Сводка: доля верных меток\n")
    rows = []
    for (lang, var, inp), g in p.groupby(["language", "variant", "input"], sort=False):
        rows.append([LANGS[lang], VARIANTS[var], INPUTS[inp], g.id.nunique()] +
                    [f"{g[g.model == m].correct.mean():.2f} ({int(g[g.model == m].correct.sum())}/{(g.model == m).sum()})"
                     for m in MODELS])
    md.append(table(["Язык", "Вариант", "Вход", "Сообщений"] + list(MODELS.values()), rows))
    md.append("")

    md.append("## 3. Пропуски мошенничества и ложные срабатывания\n")
    md.append("Пропуск — мошенническое сообщение (smishing) помечено как ham. Отдельно — помечено как spam "
              "(сообщение всё же попадает в «нежелательные», но тип угрозы определён неверно). Ложное "
              "срабатывание — легитимное сообщение помечено как smishing; отдельно — легитимное ham помечено "
              "как spam.\n")
    rows = []
    for (lang, var, inp), g in p.groupby(["language", "variant", "input"], sort=False):
        cells = []
        for m in MODELS:
            gm = g[g.model == m]
            fr, lg = gm[gm.pair == "fraud"], gm[gm.pair == "legit"]
            cells.append(f"{(fr.pred == 'ham').sum()} / {(fr.pred == 'spam').sum()} из {len(fr)}; "
                         f"{(lg.pred == 'smishing').sum()} / {((lg.label == 'ham') & (lg.pred == 'spam')).sum()} из {len(lg)}"
                         if len(gm) else "—")
        rows.append([LANGS[lang], VARIANTS[var], INPUTS[inp]] + cells)
    md.append(table(["Язык", "Вариант", "Вход"] + [f"{v}: мошенн. → ham / → spam; легит. → smish / ham → spam"
                                                    for v in MODELS.values()], rows))
    md.append("")

    md.append("## 4. По каждому сообщению\n")
    md.append("Предсказание (уверенность = максимальная вероятность). Full: † — ответ дала вторая ступень "
              "(DistilBERT).\n")
    rows = []
    for r in cases.itertuples():
        g = p[p.id == r.id]
        cells = []
        for inp in INPUTS:
            for m in MODELS:
                x = g[(g.input == inp) & (g.model == m)]
                if x.empty:
                    cells.append("—")
                    continue
                x = x.iloc[0]
                mark = "" if x.correct else " ✗"
                dag = " †" if m == "full" and str(x.routed_to_distilbert) == "True" else ""
                cells.append(f"{SHORT[x.pred]} {x.confidence:.2f}{dag}{mark}")
        rows.append([r.id, r.label] + cells)
    md.append(table(["id", "Метка"] + [f"{MODELS[m]}, {INPUTS[i]}" for i in INPUTS for m in MODELS], rows))
    md.append("\n✗ — неверно.\n")

    md.append("## 5. Задержка (Kaggle CPU, 1 поток, батч 1)\n")
    names = {"translation_nllb": "Перевод NLLB-200-distilled-600M (RU / KK → EN)",
             **{f"model_{m}_on_translation": f"{MODELS[m]} на переводе" for m in MODELS},
             **{f"translation_plus_{m}": f"Перевод + {MODELS[m]}" for m in MODELS}}
    md.append(table(["Этап", "Сообщений", "p50, мс", "p95, мс", "Среднее, мс"],
                    [[names[k], v["n"], f"{v['p50_ms']:.1f}", f"{v['p95_ms']:.1f}", f"{v['mean_ms']:.1f}"]
                     for k, v in lat.items()]))
    md.append(f"\nНастройки перевода: `{cfg['nllb']}`, {cfg['src_lang']} → {cfg['tgt_lang']}, генерация по "
              f"умолчанию модели (num_beams {cfg['generation'].get('num_beams', 'по умолчанию')}, max_length "
              f"{cfg['generation'].get('max_length', 'по умолчанию')}). Задержка — на {lat['translation_nllb']['n']} "
              "сообщениях RU / KK, поэтому p95 — грубая оценка.\n")

    md.append("## 6. Качественные примеры\n")
    md += QUALITATIVE + [""]
    md.append("## 7. Ограничения\n")
    md += [f"- {s}" for s in LIMITATIONS] + [""] + SUMMARY
    (ROOT / "reports" / "step6_kz_cases.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/step6_kz_cases.md")


INTRO = [
    "**42 сообщения — это качественная демонстрация, а не статистическая оценка.** Тексты синтетические, их "
    "написали авторы (7 сценариев × мошенническое / легитимное × EN / RU / KK); русские и казахские тексты "
    "проверены авторами.",
    "Модели заморожены и не дообучались: финальные веса seed 1 из `smishfuse-final-3class`, обученные на "
    "mendeley_3class (английский). Lite — символьный LinearSVM без аугментации, Full — каскад с "
    "замороженным tau, BERT — для сравнения.",
    "Ссылки есть только в мошеннических сообщениях и только на зарезервированный домен `.example`; телефонов "
    "нет. Сценарии 6 и 7 проверяют мошенничество без ссылок: просьба прислать или продиктовать код.",
    "Набор зафиксирован коммитом 924ff27 до запуска моделей.",
]
# Written after the run, checked against results/step6/ (2026-10-31).
QUALITATIVE = [
    "**1. Перевод спасает (казахский).** `kz1_fraud_kk` («Kaspi: карта заблокирована, подтвердите данные на "
    "kaspi-verify.example»). Как написано, все три модели отвечают ham (0.83 / 0.82 / 0.89). NLLB перевёл "
    "сообщение целиком, вместе с призывом и ссылкой. После перевода все три отвечают smishing (0.96 / 0.96 / "
    "0.97). То же с `kz5_fraud_ru` и `kz5_fraud_kk` (досрочное снятие пенсионных): ham как написано, smishing "
    "у всех моделей после перевода.",
    "",
    "**2. Перевод теряет призыв к действию (русский).** `kz1_fraud_ru` переведено как «Kaspi Bank: Your card is "
    "blocked due to suspicious activity.»: вторая фраза со ссылкой и просьбой подтвердить данные пропала. "
    "Lite на грани (smishing 0.55). У каскада SVM не уверен, ответ даёт DistilBERT — ham 0.98. BERT: ham 0.71. "
    "Ссылка пропала в 13 из 15 русских мошеннических сообщений со ссылкой и в 2 из 10 казахских.",
    "",
    "**3. Мошенничество без ссылки — ошибка и после перевода.** `kz6_fraud_ru` (бонус за код из SMS) "
    "переведено как «Beeline, you're getting a 5GB bonus!»: просьба отправить код потеряна, все три модели "
    "отвечают ham (0.98–1.00). Казахская версия переведена полностью, и Lite и каскад её ловят (smishing "
    "0.56 / 0.70), BERT — уверенно (0.97). На английском оригинале каскад ошибается (spam 0.77, ответ "
    "DistilBERT), Lite отвечает smishing с уверенностью 0.55.",
    "",
    "**4. Настоящее предупреждение банка принимается за мошенничество.** `kz7_legit` (подозрительная операция "
    "отклонена, позвоните по номеру на карте, никому не сообщайте коды). После перевода RU и KK все три модели "
    "отвечают smishing (0.74–0.95); на английском оригинале — Lite и каскад тоже (0.88 / 0.95). Слова "
    "«suspicious», «call the number», «SMS codes» совпадают с мошенническими. Модели, обученные на Mendeley, "
    "реагируют на лексику, а не на отсутствие просьбы передать данные.",
    "",
    "**5. Перевод выдумывает текст.** `kz5_legit_kk` (выписка ЕНПФ) переведено как «The following is the list "
    "of the most common types of pension benefits for the person who has a disability:». Это не перевод "
    "исходного сообщения. Lite и каскад случайно отвечают верно (ham), BERT — spam.",
]
LIMITATIONS = [
    "Замороженная HELD-OUT-обфускация на кириллице слабая. Удаление гласных работает только на латинских "
    "гласных, гомоглифы заменяют только латинские буквы. В русских сообщениях она сводится в основном к "
    "удвоению букв и искажает только латинские вставки (Kaspi, eGov, домены). Поэтому для кейс-стади добавлен "
    "отдельный вариант `cyr_homoglyph` (10% кириллических букв → латинские двойники, seed фиксирован; "
    "`scripts/make_kz_cases.py`). Он не входит в основной протокол.",
    "В трёх строках heldout10 (сценарии 2, 3, 4) обфускация задела и доменную зону (например "
    "`kasspi-prize.xаmple`). Такие домены не существуют; строки оставлены как есть.",
    "Метки легитимных сообщений сценариев 2 и 6 (реклама банка и оператора) — spam, по схеме разметки Mendeley.",
    "Модели обучены только на английском mendeley_3class. RU и KK как написано для них вне распределения; "
    "перевод NLLB — один из возможных путей, его качество отдельно не оценивалось.",
    "NLLB-200-distilled-600M с настройками генерации по умолчанию (max_length 200 токенов — не ограничение) "
    "часто переводит только первое предложение русского сообщения: ссылка пропала в 13 из 15 русских "
    "мошеннических сообщений со ссылкой. Перевод по предложениям, вероятно, это исправил бы, но не "
    "проверялся: процедура перевода задана заранее и после результатов не менялась.",
    "Даже английские версии модели классифицируют плохо: верно 7 из 14 (Lite), 6 из 14 (каскад), 9 из 14 (BERT). "
    "Современные казахстанские банковские и государственные SMS отличаются от Mendeley (сообщения 2010-х, в "
    "основном британские), и это сдвиг домена, а не только языка.",
    "Задержка перевода измерена на 49 сообщениях в один поток CPU. Многопоточный режим, GPU или меньшая модель "
    "перевода не проверялись.",
]
SUMMARY = [
    "## Итог\n",
    "- **Как написано, RU и KK модели почти не понимают.** Lite и каскад пропускают 6 из 7 мошеннических "
    "сообщений (ham) на обоих языках, BERT — 4 из 7 на русском и 5 из 7 на казахском.",
    "- **Перевод помогает, но неравномерно.** На казахском после перевода ни одна модель не пропустила "
    "мошенничество как ham (0 из 7). На русском остаётся 2–3 пропуска из 7: NLLB часто отбрасывает вторую "
    "фразу с призывом и ссылкой.",
    "- **Искажённые варианты.** Как написано, все модели распознают 1 из 7. После перевода — от 0 до 5 из 7 в "
    "зависимости от модели и варианта.",
    "- **Ложные тревоги.** Настоящее предупреждение банка о подозрительной операции (сценарий 7) все модели "
    "после перевода принимают за smishing.",
    "- **Задержка.** Перевод NLLB занимает p50 4.6 с и p95 7.5 с в один поток CPU, а сами модели — миллисекунды "
    "(Lite 3.2 мс, каскад p50 2.8 мс). Путь «перевод + модель» не real-time: в нём доминирует перевод.",
    "- 42 сообщения — качественная демонстрация, а не статистическая оценка; тексты синтетические, их "
    "написали авторы.",
    "",
]


if __name__ == "__main__":
    main()
