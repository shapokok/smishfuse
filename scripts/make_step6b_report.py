"""Builds reports/step6b_kz_adapt.md and reports/figures/step6b/adaptation_curve.pdf from results/step6/,
results/step6b/ and data/kz_adapt/ (counting and plotting only; no model is run here).
Run: python scripts/make_step6b_report.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R6, R6B = ROOT / "results" / "step6", ROOT / "results" / "step6b"
FIG = ROOT / "reports" / "figures" / "step6b"
MODELS = {"lite": "Lite", "full": "Full (каскад)", "bert": "BERT"}
VARIANTS = {"original": "исходный", "heldout10": "HELD-OUT 10%", "cyr_homoglyph": "кир. гомоглифы 10%"}
LANGS = {"en": "EN", "ru": "RU", "kk": "KK"}
LINK = re.compile(r"\S+\.[^\s.]*x[^\s.]*mp[^\s.]*|\S+\.example")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def frac(s: pd.Series) -> str:
    return f"{s.mean():.2f} ({int(s.sum())}/{len(s)})" if len(s) else "—"


def ms(v: pd.Series) -> str:
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else (f"{v.mean():.3f}" if len(v) else "—")


def section_mt(md):
    md.append("## 1. Перевод по предложениям (пост-хок вариант `mt_sentence`)\n")
    md.append("NLLB-200-distilled-600M, те же настройки, что в шаге 6. Сообщение разбивается на предложения, каждое "
              "переводится отдельно; ссылки, домены, числа и коды заменяются метками и возвращаются как есть "
              "(`src/step6/mt_sentence.py`). Модели те же замороженные Lite, Full, BERT.\n")
    a = pd.read_csv(R6 / "predictions.csv", keep_default_na=False)
    a = a[a.input == "translated_en"].assign(input="mt_whole")
    b = pd.read_csv(R6B / "mt_sentence" / "predictions.csv", keep_default_na=False)
    p = pd.concat([a, b])
    p["ok"] = p.pred == p.label
    rows = []
    for (lang, var), g in p.groupby(["language", "variant"], sort=False):
        row = [LANGS[lang], VARIANTS[var], g.id.nunique()]
        for m in MODELS:
            w = g[(g.model == m) & (g.input == "mt_whole")].ok
            s = g[(g.model == m) & (g.input == "mt_sentence")].ok
            row.append(f"{int(w.sum())} → {int(s.sum())}")
        rows.append(row)
    md.append(table(["Язык", "Вариант", "Сообщений"] + [f"{v}: верно, целиком → по предложениям" for v in MODELS.values()],
                    rows))
    tw = pd.read_csv(R6 / "translations.csv", keep_default_na=False).set_index("id")
    ts = pd.read_csv(R6B / "mt_sentence" / "translations.csv", keep_default_na=False).set_index("id")
    cases = pd.read_csv(ROOT / "data" / "kz_cases" / "cases.csv", dtype=str, keep_default_na=False).set_index("id")
    with_link = [i for i in ts.index if LINK.search(cases.loc[i, "text"])]
    lost_w = sum(not LINK.search(tw.loc[i, "translation"]) for i in with_link)
    lost_s = sum(not LINK.search(ts.loc[i, "translation"]) for i in with_link)
    lat = json.loads((R6B / "mt_sentence" / "latency.json").read_text())["translation_nllb_sentencewise"]
    md.append(f"\nСообщений RU / KK со ссылкой: {len(with_link)}. Ссылка пропала в переводе целиком: {lost_w}, по "
              f"предложениям: {lost_s}. Меток, потерянных переводчиком и дописанных в конец предложения: "
              f"{int(ts.protected_lost_and_appended.sum())} (например, «до 31 марта» → «until March 1st … 31»). "
              f"Задержка перевода по предложениям: p50 {lat['p50_ms'] / 1000:.1f} с, p95 {lat['p95_ms'] / 1000:.1f} с "
              "(1 поток CPU).\n")


def section_set(md):
    md.append("## 2. Набор для адаптации\n")
    a = pd.read_csv(ROOT / "data" / "kz_adapt" / "adapt.csv", dtype=str, keep_default_na=False)
    d = pd.read_csv(ROOT / "data" / "kz_adapt" / "draft.csv", dtype=str, keep_default_na=False)
    md.append(f"{len(a)} сообщений ({len(a) // 3} идей × RU / KK / EN), `data/kz_adapt/adapt.csv`. **Набор "
              "синтетический, создан с помощью LLM и проверен автором.** Пересечение с 63 строками кейс-стади "
              "(MinHash LSH + точный Жаккар ≥ 0.5 по символьным 5-граммам): удалено 0 "
              "(`data/kz_adapt/overlap_removed.csv`).\n")
    rows = []
    for lab in ["ham", "spam", "smishing"]:
        g, gd = a[a.label == lab], d[d.label == lab]
        rows.append([lab, len(g), int((g.same_category_as_test == "true").sum()),
                     f"{100 * gd.text.str.contains('.example', regex=False).mean():.0f}%",
                     f"{100 * g.text.str.contains('.example', regex=False).mean():.0f}%"])
    md.append(table(["Класс", "Сообщений", "Из них категории тестовых кейсов", "Со ссылкой: черновик",
                     "Со ссылкой: после проверки"], rows))
    md.append("\nСсылки (.example) добавлены после проверки автором в 14 из 36 идей spam и 10 из 38 идей ham, чтобы "
              "ссылка не была признаком только smishing. Пограничные smishing-идеи 31 (сбор пожертвований) и 32 "
              "(продление полиса ОГПО по ссылке) оставлены smishing: похожие законные рассылки существуют "
              "(колонка `borderline`).\n")


def section_adapt(md):
    md.append("## 3. Адаптация Lite\n")
    md.append("LinearSVM char без аугментации, финальные гиперпараметры, без подбора, seed 1. Обучение: Mendeley "
              "train + сообщения адаптации. A — только категории, которых нет в тестовых кейсах; B — весь набор. "
              "Размеры 50, 100, 200 — по 5 стратифицированных подвыборок, затем все сообщения. «До» — замороженный "
              "финальный Lite. Оценка на 63 строках кейс-стади как написано (без перевода).\n")
    runs = pd.read_csv(R6B / "adapt" / "runs.csv")
    p = pd.read_csv(R6B / "adapt" / "predictions.csv", keep_default_na=False)
    p["ok"] = p.pred == p.label
    cases = pd.read_csv(ROOT / "data" / "kz_cases" / "cases.csv", dtype=str, keep_default_na=False)
    adapt = pd.read_csv(ROOT / "data" / "kz_adapt" / "adapt.csv", dtype=str, keep_default_na=False)
    pc = p[p.set == "cases"].merge(cases[["id", "pair", "language", "variant", "scenario"]], on="id")

    md.append("**Mendeley VAL (основная задача), macro F1**\n")
    rows = [["до адаптации", "—", f"{runs[runs.run_id == 'before'].val_macro_f1.iloc[0]:.4f}"]]
    for cond in ["A", "B"]:
        for n, g in runs[runs.condition == cond].groupby("size"):
            rows.append([f"{cond}", n, ms(g.val_macro_f1)])
    md.append(table(["Условие", "Сообщений адаптации", "Macro F1 (среднее ± std по подвыборкам)"], rows))
    md.append("")

    def per_run(sub):
        return sub.groupby("run_id").ok.mean()

    md.append("**Кривая адаптации: доля верных на 42 исходных кейсах (EN / RU / KK как написано)**\n")
    orig = pc[pc.variant == "original"]
    curve = []
    for cond in ["A", "B"]:
        for n, g in runs[runs.condition == cond].groupby("size"):
            acc = per_run(orig[orig.run_id.isin(g.run_id)])
            fr = per_run(orig[orig.run_id.isin(g.run_id) & (orig.pair == "fraud")])
            curve.append({"condition": cond, "n": n, "acc_mean": acc.mean(), "acc_std": acc.std(ddof=1) if len(acc) > 1 else 0,
                          "fraud_mean": fr.mean(), "fraud_std": fr.std(ddof=1) if len(fr) > 1 else 0})
    b0 = orig[orig.run_id == "before"]
    md.append(table(["Условие", "Сообщений адаптации", "Доля верных (42)", "Доля распознанного мошенничества (21)"],
                    [["до", 0, f"{b0.ok.mean():.2f}", f"{b0[b0.pair == 'fraud'].ok.mean():.2f}"]] +
                    [[c["condition"], c["n"], f"{c['acc_mean']:.2f} ± {c['acc_std']:.2f}",
                      f"{c['fraud_mean']:.2f} ± {c['fraud_std']:.2f}"] for c in curve]))
    md.append("\n![Кривая адаптации](figures/step6b/adaptation_curve.pdf)\n")
    plot(curve, b0)

    final = {"до": "before", "A (все)": runs[(runs.condition == "A") & (runs.subsample == 0)].run_id.iloc[0],
             "B (все)": runs[(runs.condition == "B") & (runs.subsample == 0)].run_id.iloc[0]}
    md.append("**По языкам и вариантам (кейсы как написано): доля верных; мошенн. → ham (пропуск); легит. → smishing "
              "(ложная тревога)**\n")
    rows = []
    for (lang, var), g in pc.groupby(["language", "variant"], sort=False):
        row = [LANGS[lang], VARIANTS[var]]
        for rid in final.values():
            x = g[g.run_id == rid]
            fr, lg = x[x.pair == "fraud"], x[x.pair == "legit"]
            row.append(f"{frac(x.ok)}; {(fr.pred == 'ham').sum()}/{len(fr)}; {(lg.pred == 'smishing').sum()}/{len(lg)}")
        rows.append(row)
    md.append(table(["Язык", "Вариант"] + list(final), rows))
    md.append("")

    md.append("**Сценарии 6 и 7: мошенничество без ссылок (просьба прислать или продиктовать код)**\n")
    rows = []
    for _, r in cases[(cases.pair == "fraud") & cases.scenario.str.match(r"^[67]\.")].iterrows():
        cells = []
        for rid in final.values():
            x = pc[(pc.run_id == rid) & (pc.id == r["id"])].iloc[0]
            cells.append(f"{x.pred} {x.confidence:.2f}{'' if x.ok else ' ✗'}")
        rows.append([r["id"], *cells])
    md.append(table(["id (метка smishing)"] + list(final), rows))
    md.append("")

    md.append("**Легитимные сообщения со ссылками.** В 42 тестовых кейсах ссылок в легитимных сообщениях нет, поэтому "
              "проверка — на сообщениях набора адаптации, не использованных при обучении данной модели (для «до» — "
              "весь набор, для A (все) — сообщения категорий тестовых кейсов, для подвыборок — остаток набора).\n")
    ho = p[p.set == "adapt_held_out"].merge(adapt[["id", "text"]], on="id")
    ho = ho[(ho.label != "smishing") & ho.text.str.contains(".example", regex=False)]
    rows = []
    for name, rid in [("до", "before"), ("A (все)", final["A (все)"])]:
        x = ho[ho.run_id == rid]
        rows.append([name, len(x), frac(x.ok), int((x.pred == "smishing").sum())])
    for cond in ["A", "B"]:
        for n, g in runs[(runs.condition == cond) & (runs.subsample > 0)].groupby("size"):
            x = ho[ho.run_id.isin(g.run_id)]
            acc = x.groupby("run_id").ok.mean()
            rows.append([f"{cond}, {n} (5 подвыборок)", f"{len(x) // max(1, len(g))} в среднем",
                         f"{acc.mean():.2f} ± {acc.std(ddof=1):.2f}", int((x.pred == "smishing").sum())])
    md.append(table(["Модель", "Легит. сообщений со ссылкой", "Доля верных", "Помечено smishing (всего)"], rows))
    md.append("")
    lat = json.loads((R6B / "adapt" / "latency.json").read_text())
    md.append("**Задержка (Kaggle CPU, 1 поток, батч 1, 1000 сообщений)**\n")
    md.append(table(["Модель", "p50, мс", "p95, мс", "Сообщений/с"],
                    [[k, f"{v['p50_ms']:.3f}", f"{v['p95_ms']:.3f}", f"{v['throughput_msg_per_s']:.0f}"]
                     for k, v in lat.items()]))
    md.append("")


def plot(curve, b0):
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, key, title in zip(axes, ["acc", "fraud"], ["Доля верных (42 кейса)", "Распознано мошенничество (21)"]):
        base = b0.ok.mean() if key == "acc" else b0[b0.pair == "fraud"].ok.mean()
        ax.axhline(base, color="black", lw=0.8, ls=":", label="до адаптации")
        for cond, st in [("A", {"color": "#2166ac", "marker": "s", "ls": "--"}),
                         ("B", {"color": "#b2182b", "marker": "o", "ls": "-", "lw": 2.0})]:
            c = [x for x in curve if x["condition"] == cond]
            ax.errorbar([x["n"] for x in c], [x[f"{key}_mean"] for x in c], yerr=[x[f"{key}_std"] for x in c],
                        capsize=2, ms=4, label="A: новые категории" if cond == "A" else "B: весь набор", **st)
        ax.set_title(title)
        ax.set_xlabel("Сообщений адаптации")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("Доля")
    axes[1].legend(frameon=False, loc="lower right")
    fig.savefig(FIG / "adaptation_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    md = ["# SmishFuse, шаг 6б: пост-хок дополнения к кейс-стади\n",
          "Сгенерировано `scripts/make_step6b_report.py`. **Всё в этом отчёте — пост-хок, введено после "
          "результатов шага 6 (решение 21 в README).** 42 тестовых кейса и их варианты не менялись. **Малый "
          "синтетический набор, качественная демонстрация, а не статистическая оценка.** Вычисления — Kaggle CPU.\n"]
    section_mt(md)
    section_set(md)
    section_adapt(md)
    md.append("## 4. Качественные примеры\n")
    md += QUALITATIVE + [""]
    md.append("## 5. Ограничения\n")
    md += [f"- {s}" for s in LIMITATIONS] + [""] + SUMMARY
    (ROOT / "reports" / "step6b_kz_adapt.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/step6b_kz_adapt.md")


# Written after the run, checked against results/step6b/ (2026-11-01).
QUALITATIVE = [
    "**1. Мошенничество без ссылки распознаётся после адаптации.** `kz7_fraud_kk` (банк просит продиктовать "
    "код): до адаптации ham 0.88, после B — smishing 0.97. `kz6_fraud_ru` (бонус за код из SMS): ham 0.83 → "
    "smishing 0.87. Все 12 строк сценариев 6 и 7 после адаптации верны, до неё — 4 из 12.",
    "",
    "**2. Казахский без перевода.** `kz5_fraud_kk` (досрочное снятие пенсионных): ham 0.93 → smishing 0.96. Тот "
    "же результат, что давал перевод NLLB, но за 2.2 мс вместо 4.6 с.",
    "",
    "**3. Обратная сторона: обычные уведомления становятся «мошенничеством».** В условии A модель не видела "
    "легитимных сообщений банков, eGov, почты и ЕНПФ (это категории тестовых кейсов). Такие уведомления "
    "без ссылок она помечает как smishing: «Jusan: you have received a transfer of 15 000 tenge» (0.68), "
    "«ENPF: a contribution for April has been credited…» (0.73), «Halyk: a payment of 40 000 tenge has been "
    "confirmed…» (0.98). Модель выучила контекст (казахстанские банки и госуслуги), а не признаки обмана.",
    "",
    "**4. Предупреждение банка остаётся ложной тревогой.** `kz7_legit` (подозрительная операция отклонена, "
    "позвоните по номеру на карте) и после адаптации B помечается как smishing на всех трёх языках (0.60–0.81).",
    "",
    "**5. Реклама со ссылкой → smishing.** «Jusan: депозит под высокую ставку. Откройте онлайн на "
    "jusan-deposit.example» (казахская версия) в условии A — smishing 0.83. Ссылки в части spam и ham "
    "ложные тревоги на легитимных сообщениях со ссылками не устранили.",
]
LIMITATIONS = [
    "Набор адаптации синтетический (создан с помощью LLM, проверен автором) и мал: 108 идей, каждая на трёх языках. "
    "Три языковые версии одной идеи — не независимые примеры.",
    "Тестовые кейсы написаны теми же авторами в том же стиле, что и набор адаптации; пересечение по тексту "
    "исключено (Жаккар < 0.5), но близость стиля могла завысить эффект адаптации, особенно в условии B.",
    "Подбора гиперпараметров нет; один seed модели; разброс — только по подвыборкам набора.",
]
SUMMARY = [
    "## Итог\n",
    "- **Перевод по предложениям** сохраняет ссылки и помогает на русском (Lite 8 → 11 из 14, каскад 7 → 10), "
    "на казахском не помогает (Lite 10 → 8).",
    "- **Адаптация Lite на 324 синтетических сообщениях** подняла распознавание мошенничества в 42 кейсах с 7 "
    "до 21 из 21 и долю верных с 0.36 до 0.81 (B). Уже 50 сообщений дают 0.81–0.90 распознанного мошенничества. "
    "Условие A (только новые категории) тоже работает: мошенничество 21 из 21, доля верных 0.74.",
    "- **Основная задача не испорчена:** macro F1 на Mendeley VAL 0.896–0.910 против 0.904 до адаптации. "
    "Задержка та же: p50 2.2 мс.",
    "- **Цена — ложные тревоги.** На легитимных сообщениях со ссылками из набора доля верных 0.39–0.62. Без "
    "примеров легитимных сообщений своей категории модель помечает обычные банковские и государственные "
    "уведомления как smishing. Предупреждение банка (сценарий 7) остаётся ложной тревогой.",
    "- Набор малый и синтетический, тестовые кейсы и набор написаны в одном стиле: это качественная "
    "демонстрация, а не статистическая оценка.",
    "",
]


if __name__ == "__main__":
    main()
