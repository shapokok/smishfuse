"""Builds reports/step5a_report.md from results/step5a/ (Kaggle notebook smishfuse-step5a) and
results/step5a/published_claims.yaml (numbers from the papers).

Criteria (fixed in configs/step5a.yaml before the results): every score is the cross-fit score
(metric on each of the two val halves, averaged); "beyond the seed spread" = difference of the
means over seeds larger than the larger of the two standard deviations (ddof = 1). Descriptive,
no significance test.
Run: python scripts/make_step5a_report.py
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.registry import SHORT_NAMES  # noqa: E402

RESULTS = ROOT / "results"
S5 = RESULTS / "step5a"
CFG = yaml.safe_load((ROOT / "configs" / "step5a.yaml").read_text(encoding="utf-8"))
SETTINGS = CFG["eval_settings"]
NAMES = {"cascade": "Каскад", "average": "Усреднение SVM + DistilBERT", "svm_char_aug": "LinearSVM char (аугм.)",
         "svm_char": "LinearSVM char без аугм.", "distilbert_aug": "DistilBERT (аугм.)",
         "distilbert": "DistilBERT без аугм. (4а/4в)", "bert": "BERT (шаг 3)", "svm_step2": "LinearSVM шага 2",
         "smishfuse_v1": "SmishFuse v1", "smishfuse_v2_aug": "SmishFuse v2 (аугм.)",
         "smishfuse_v2": "SmishFuse v2 без аугм.", "mahmud2024_cnn_bigru": "Mahmud et al. 2024, CNN-BiGRU",
         "shim2024_bert_base": "Shim et al. 2024, BERT-base", "shim2024_distilbert_base": "Shim et al. 2024, DistilBERT-base",
         "shim2024_roberta_base": "Shim et al. 2024, RoBERTa-base", "linear_svm": "LinearSVM (шаг 2)",
         "logreg": "LogReg (шаг 2)"}
WORKS = {"mahmud2024": "Mahmud et al. 2024 (Systems)", "shim2024": "Shim et al. 2024 (arXiv)",
         "munoz2025": "Munoz & Islam 2025 (J. Cybersecur. Priv.)", "remmide2024": "Remmide et al. 2024 (IJIT)",
         "mishra_soni_dsmishsms": "Mishra & Soni, DSmishSMS (NCA 2021/2023)",
         "mishra_soni_smishing_detector": "Mishra & Soni, Smishing Detector (SN CS 2022)",
         "smishing4c": "Martínez-Mendoza et al. 2024, Smishing-4C (SEPLN)", "sonowal2020": "Sonowal 2020 (SN CS)",
         "mishra_soni_dataset_paper": "Mishra & Soni 2022, статья о наборе данных (SoCPaR)"}
TASK_NAMES = {"mendeley_3class": "3 класса", "mendeley_binary": "ham против smishing+spam",
              "smishing_vs_spam": "smishing против spam"}
BASELINES_4C = ["distilbert_aug", "distilbert", "bert", "svm_char_aug", "svm_char", "svm_step2"]
PREVIOUS = ["smishfuse_v1", "smishfuse_v2_aug", "smishfuse_v2"]


def name(m: str, source: str = "") -> str:
    if source == "step4b":
        return f"{SHORT_NAMES.get(m, m)} (шаг 4б)"
    return NAMES.get(m, m)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def label(s: str) -> str:
    return "чистый" if s == "clean" else f"HELD-OUT {int(s[-2:])}%"


def ms(m, s, n=None) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "нет данных"
    txt = f"{m:.4f} ± {s:.4f}" if not math.isnan(s) else f"{m:.4f}"
    return txt + (f" (n={n})" if n is not None and n != len(CFG["seeds"]) else "")


def crossfit() -> pd.DataFrame:
    """Cross-fit score per (task, model, source, setting, seed): mean of the two halves."""
    df = pd.read_csv(S5 / "compare" / "crossfit.csv")
    cols = [c for c in ["accuracy", "macro_f1", "precision_pos", "recall_pos", "f1_pos"] if c in df]
    return df.groupby(["task", "model", "source", "setting", "seed"], as_index=False)[cols].mean()


def agg(cf, task, model, setting, source=None, col="macro_f1"):
    d = cf[(cf.task == task) & (cf.model == model) & (cf.setting == setting)]
    if source:
        d = d[d.source == source]
    v = d[col].dropna()
    return (v.mean(), v.std(ddof=1), len(v)) if len(v) else (float("nan"), float("nan"), 0)


def verdict(a, b) -> tuple[float, float, bool]:
    """(difference in pp, tolerance in pp, beyond the spread and positive)."""
    d, tol = 100 * (a[0] - b[0]), 100 * max(a[1], b[1])
    return d, tol, d > tol


def section_cascade(md, cf):
    md.append("## 1. Каскад\n")
    cc = CFG["cascade"]
    md.append(f"Ступень 1 — `{cc['stage1']}`, ступень 2 — `{cc['stage2']}` (вероятности на val из шага 4в, seed k "
              "с seed k). tau выбирается на чистом val одной половины и применяется к другой, затем наоборот; "
              "оценка — среднее по двум половинам (по 287 сообщений). HELD-OUT-искажения при выборе tau не "
              "используются.\n")
    rows = []
    for m in ["cascade", "average", cc["stage1"], cc["stage2"]]:
        rows.append([name(m)] + [ms(*agg(cf, "mendeley_3class", m, s)) for s in SETTINGS])
    md.append("**Macro F1 (cross-fit), среднее ± std по 3 seed**\n")
    md.append(table(["Модель"] + [label(s) for s in SETTINGS], rows))
    md.append("")
    info = [read_json(S5 / "cascade" / f"crossfit_seed{k}.json") for k in CFG["seeds"]
            if (S5 / "cascade" / f"crossfit_seed{k}.json").exists()]
    if info:
        md.append(table(["Seed", "tau (половина 0 / 1)"] + [f"Доля во 2-ю ступень, {label(s)}" for s in SETTINGS],
                        [[i["seed"], f"{i['tau_by_half']['0']:.4f} / {i['tau_by_half']['1']:.4f}"]
                         + [f"{100 * i[s]['routed_share']:.1f}%" for s in SETTINGS] for i in info]))
        md.append("")
    curves = [read_json(S5 / "cascade" / f"curve_seed{k}.json")["curve"] for k in CFG["seeds"]
              if (S5 / "cascade" / f"curve_seed{k}.json").exists()]
    if curves:
        md.append("**Кривая: доля сообщений во второй ступени → macro F1 на всём val (без выбора tau)**\n")
        rows = []
        for i, q in enumerate(cc["curve_shares"]):
            vals = {s: pd.Series([c[s][i]["macro_f1"] for c in curves]) for s in SETTINGS}
            rows.append([f"{100 * q:.0f}%"] + [ms(v.mean(), v.std(ddof=1)) for v in vals.values()])
        md.append(table(["Во 2-ю ступень"] + [label(s) for s in SETTINGS], rows))
        md.append("\nВ кривой уходят сообщения с наименьшей максимальной вероятностью SVM. 0% — только SVM, "
                  "100% — только DistilBERT.\n")
    lat = pd.read_csv(RESULTS / "latency" / "latency.csv")
    l1 = float(lat[(lat.task == "mendeley_3class") & (lat.model == cc["latency_stage1"])]["latency_ms_median_batch1"].iloc[0])
    l2 = float(lat[(lat.task == "mendeley_3class") & (lat.model == cc["latency_stage2"])]["latency_ms_median_batch1"].iloc[0])
    if info:
        md.append("**Ожидаемая средняя задержка (расчёт, не замер)**\n")
        md.append(f"Задержка SVM ({l1:.3f} мс) + доля вызовов × задержка DistilBERT ({l2:.3f} мс). Это замеры шага 3, "
                  "Kaggle CPU, 1 поток, `results/latency/latency.csv`. Символьный LinearSVM отдельно не измерялся, "
                  "вместо него взят LinearSVM шага 2 (word+char TF-IDF). Реальный замер каскада будет на "
                  "финальном шаге.\n")
        rows = []
        for s in SETTINGS:
            share = pd.Series([i[s]["routed_share"] for i in info])
            rows.append([label(s), f"{100 * share.mean():.1f}%", f"{l1 + share.mean() * l2:.2f}"])
        md.append(table(["Val", "Доля вызовов DistilBERT (среднее по seed)", "Ожидаемая задержка, мс/сообщение"],
                        rows))
        md.append(f"\nДля сравнения: только SVM {l1:.2f} мс, только DistilBERT {l2:.2f} мс, усреднение (обе модели "
                  f"всегда) {l1 + l2:.2f} мс.\n")


def section_published(md, cf):
    md.append("## 2. Опубликованные методы\n")
    claims = yaml.safe_load((S5 / "published_claims.yaml").read_text(encoding="utf-8"))
    md.append("Заявленные числа — **из статьи**, хранятся отдельно: `results/step5a/published_claims.yaml`. "
              "Наши числа — воспроизведение по описанию: обучение на нашем чистом train, оценка на нашем val "
              "(после дедупликации, разбиение шага 1), среднее ± std по 3 seed, cross-fit по половинам val.\n")
    md.append("### 2.1. Работы с результатами на Mendeley SMS Phishing\n")
    rows = []
    for key, c in claims["on_mendeley"].items():
        rows.append([WORKS[key], c.get("dataset", c.get("dataset_split_dedup", "—")),
                     c.get("split", "—"), c.get("deduplication", "не проверено"), c["reproducible"]])
    md.append(table(["Работа", "Данные и задача (из статьи)", "Разбиение", "Дедупликация", "Воспроизводимость"], rows))
    md.append("\nПолные ссылки — в `published_claims.yaml`.\n")
    md.append("### 2.2. Заявлено и воспроизведено\n")
    pairs = [("mahmud2024_cnn_bigru", "mendeley_binary", claims["on_mendeley"]["mahmud2024"]["reported_table7_dataset3"]["CNN-BiGRU"], 1.0),
             ("shim2024_bert_base", "smishing_vs_spam", claims["on_mendeley"]["shim2024"]["reported_table1_original_data"]["BERT-base"], 100.0),
             ("shim2024_distilbert_base", "smishing_vs_spam", claims["on_mendeley"]["shim2024"]["reported_table1_original_data"]["DistilBERT-base"], 100.0),
             ("shim2024_roberta_base", "smishing_vs_spam", claims["on_mendeley"]["shim2024"]["reported_table1_original_data"]["RoBERTa-base"], 100.0)]
    rows = []
    for m, task, claim, scale in pairs:
        cl = f"acc {claim['accuracy'] / scale:.4f}, F1 {claim['f1'] / scale:.4f} (P {claim['precision'] / scale:.4f}, R {claim['recall'] / scale:.4f})"
        ours = [f"acc {ms(*agg(cf, task, m, s, 'published', 'accuracy'))}; F1+ {ms(*agg(cf, task, m, s, 'published', 'f1_pos'))}; "
                f"macro F1 {ms(*agg(cf, task, m, s, 'published'))}" for s in SETTINGS]
        rows.append([name(m), TASK_NAMES[task], cl] + ours)
    md.append(table(["Метод", "Задача", "Заявлено (из статьи; их разбиение без дедупликации)"]
                    + [f"У нас, {label(s)}" for s in SETTINGS], rows))
    md.append("\nF1+ — F1 положительного класса: malicious у Mahmud, smishing у Shim. Какой класс положительный в "
              "статьях, не указано. Accuracy и F1 из статей получены на других разбиениях без дедупликации и "
              "на test, наши — на нашем val, поэтому напрямую не сравниваются (раздел 3 — о завышении).\n")
    sel = []
    for m in CFG["published"]["shim2024"]["models"]:
        for k in CFG["seeds"]:
            f = S5 / "published" / m / f"selection_seed{k}.json"
            if f.exists():
                c = read_json(f)["chosen"]
                sel.append([name(m), k, c["batch_size"], c["lr"], c["epoch"] + 1, f"{c['inner_val_macro_f1']:.4f}"])
    if sel:
        md.append("Выбор Shim et al. (сетка статьи на 20% нашего train):\n")
        md.append(table(["Модель", "Seed", "Batch", "lr", "Эпох", "Macro F1 внутр. val"], sel))
        md.append("")
    md.append("### 2.3. Работы из приоритетного списка без результатов на Mendeley\n")
    md.append(table(["Работа", "Данные (из статьи)", "Заявлено (из статьи)", "Почему не воспроизводится"],
                    [[WORKS[k], c.get("dataset", "—"), c.get("reported", "—"),
                      c.get("reproducible", f"полный текст: {c.get('full_text', '—')}")]
                     for k, c in claims["not_on_mendeley"].items()]))
    md.append("")


def section_inflation(md):
    md.append("## 3. Завышение из-за дубликатов (train + val, 5-fold CV)\n")
    md.append("«Грязный» вариант: все сырые строки Mendeley, кроме тех, чей оставленный дубликат попал в наш test; "
              "строки групп с ничейным конфликтом меток остаются с исходными метками. «Чистый» вариант: наш "
              "дедуплицированный train + val. Одинаковый протокол: StratifiedKFold(5, shuffle), seed 1–3, "
              "среднее по фолдам, затем среднее ± std по seed.\n")
    rows = []
    for m in CFG["inflation"]["models"]:
        res = {v: [read_json(f) for f in sorted((S5 / "inflation" / m).glob(f"{v}_seed*.json"))]
               for v in ("dirty", "clean")} if (S5 / "inflation" / m).exists() else {"dirty": [], "clean": []}
        if not res["dirty"] or not res["clean"]:
            rows.append([name(m), "—", "нет данных", "", "", "", "", "", ""])
            continue
        cell = []
        for metric in ("accuracy", "macro_f1"):
            dv = pd.Series([r[metric] for r in res["dirty"]])
            cv = pd.Series([r[metric] for r in res["clean"]])
            d, tol = 100 * (dv.mean() - cv.mean()), 100 * max(dv.std(ddof=1), cv.std(ddof=1))
            cell += [ms(dv.mean(), dv.std(ddof=1)), ms(cv.mean(), cv.std(ddof=1)),
                     f"{d:+.2f} ({'больше' if abs(d) > tol else 'в пределах'} разброса {tol:.2f})"]
        dup = [sum(f["n_test_with_duplicate_in_train"] for f in r["folds"]) / r["n"] for r in res["dirty"]]
        rows.append([name(m), f"{TASK_NAMES[res['dirty'][0]['task']]}; n {res['dirty'][0]['n']} / {res['clean'][0]['n']}"]
                    + cell + [f"{100 * sum(dup) / len(dup):.1f}%"])
    md.append(table(["Модель", "Задача; n грязный / чистый", "Accuracy грязный", "Accuracy чистый",
                     "Разница, п.п.", "Macro F1 грязный", "Macro F1 чистый", "Разница, п.п.",
                     "Сообщений грязного test-фолда с дубликатом в train-фолде"], rows))
    md.append("")


def section_answer(md, cf):
    md.append("## 4. Ответ: обходит ли каскад baseline и опубликованные методы\n")
    md.append("Критерий (описательный, без значимости): разница средних cross-fit macro F1 больше наибольшего из "
              "двух std по seed. (а) — против лучшего нашего baseline в каждом условии. Кандидаты: модели шагов "
              "2–3 (на чистом val — по предсказаниям шага 4б), DistilBERT и BERT шага 4в, LinearSVM char, "
              "LinearSVM шага 2. Прежние версии SmishFuse показаны отдельно. (б) — против каждого "
              "воспроизведённого метода на его задаче; каскад сводится к задаче через вероятности ответившей "
              "ступени.\n")
    rows, a_all = [], True
    for s in SETTINGS:
        c = agg(cf, "mendeley_3class", "cascade", s)
        cands = {name(m): agg(cf, "mendeley_3class", m, s, "step4c") for m in BASELINES_4C}
        if s == "clean":
            for m in sorted(set(cf[(cf.source == "step4b") & (cf.setting == "clean")].model)):
                cands[name(m, "step4b")] = agg(cf, "mendeley_3class", m, s, "step4b")
        cands = {k: v for k, v in cands.items() if v[2]}
        best = max(cands, key=lambda k: cands[k][0])
        d, tol, ok = verdict(c, cands[best])
        a_all &= ok
        prev = max(PREVIOUS, key=lambda m: agg(cf, "mendeley_3class", m, s, "step4c")[0])
        pv = agg(cf, "mendeley_3class", prev, s, "step4c")
        dp, tp, _ = verdict(c, pv)
        rows.append([label(s), ms(*c), f"{best} {ms(*cands[best])}", f"{d:+.2f}", f"{tol:.2f}", "да" if ok else "нет",
                     f"{name(prev)} {ms(*pv)}: {dp:+.2f} (разброс {tp:.2f})"])
    md.append("**(а) Против лучшего нашего baseline (3 класса)**\n")
    md.append(table(["Val", "Каскад", "Лучший baseline", "Каскад − baseline, п.п.", "Разброс, п.п.",
                     "Выигрыш больше разброса", "Лучшая прежняя SmishFuse"], rows))
    md.append(f"\n→ (а) **{'выполнено во всех условиях' if a_all else 'выполнено не во всех условиях'}**.\n")
    rows, b_all = [], True
    for m, task in [("mahmud2024_cnn_bigru", "mendeley_binary"), ("shim2024_bert_base", "smishing_vs_spam"),
                    ("shim2024_distilbert_base", "smishing_vs_spam"), ("shim2024_roberta_base", "smishing_vs_spam")]:
        for s in SETTINGS:
            c, p = agg(cf, task, "cascade", s, "step5a"), agg(cf, task, m, s, "published")
            if not p[2]:
                rows.append([name(m), TASK_NAMES[task], label(s), ms(*c), "нет данных", "", "", ""])
                b_all = False
                continue
            d, tol, ok = verdict(c, p)
            b_all &= ok
            rows.append([name(m), TASK_NAMES[task], label(s), ms(*c), ms(*p), f"{d:+.2f}", f"{tol:.2f}",
                         "да" if ok else "нет"])
    md.append("**(б) Против воспроизведённых опубликованных методов — второй взгляд: 3-классовый каскад, "
              "сведённый к задаче метода через вероятности (основное сравнение в их постановке — раздел 4а)**\n")
    md.append(table(["Метод", "Задача", "Val", "Каскад", "Метод", "Каскад − метод, п.п.", "Разброс, п.п.",
                     "Выигрыш больше разброса"], rows))
    md.append(f"\n→ (б) **{'выполнено во всех условиях' if b_all else 'выполнено не во всех условиях'}**.\n")
    md.append(f"**Ответ: {'да' if a_all and b_all else 'нет'}** — каскад "
              + ("обходит и лучший baseline, и все воспроизведённые методы больше разброса на чистом val и на "
                 "HELD-OUT." if a_all and b_all else "не обходит одновременно лучший baseline и все воспроизведённые "
                 "методы больше разброса во всех условиях (детали в таблицах).")
              + " Предварительно, без проверки значимости.\n")


SETTING_TASKS = {"mendeley_binary": ["mahmud2024_cnn_bigru"],
                 "smishing_vs_spam": ["shim2024_bert_base", "shim2024_distilbert_base", "shim2024_roberta_base"]}
SETTING_MODELS = ["svm_char_aug", "svm_char", "svm_step2", "distilbert_aug", "distilbert", "bert"]


def section_setting(md):
    """Addendum: the cascade and our baselines trained in the published settings (main view of (b))."""
    f = S5 / "setting" / "crossfit.csv"
    md.append("## 4а. В постановке опубликованных работ (дополнение к протоколу)\n")
    md.append("Добавлено в `configs/step5a.yaml` до получения результатов (`setting_runs`). Каскад и наши baseline "
              "обучены и оценены в постановке статьи на тех же данных: Mahmud et al. — ham против spam+smishing "
              "(все сообщения train / val), Shim et al. — smishing против spam (ham исключён из train и val; "
              "в val 91 сообщение, в каждой половине около 45). Cross-fit по тем же половинам val. Каскад: "
              "LinearSVM char (аугм.) → DistilBERT (аугм.), обученные в этой постановке, tau выбирается по "
              "чистому val половины.\n")
    if not f.exists():
        md.append("Нет данных.\n")
        return True
    cf = pd.read_csv(f)
    cols = [c for c in ["accuracy", "macro_f1", "precision_pos", "recall_pos", "f1_pos"] if c in cf]
    cf = cf.groupby(["task", "model", "source", "setting", "seed"], as_index=False)[cols].mean()
    ok_all = True
    for task, pubs in SETTING_TASKS.items():
        md.append(f"**{TASK_NAMES[task]}: macro F1 (cross-fit), среднее ± std по 3 seed**\n")
        models = ["cascade", "average", *SETTING_MODELS, *pubs]
        md.append(table(["Модель"] + [label(s) for s in SETTINGS],
                        [[name(m) + (" (из статьи, воспроизведено)" if m in pubs else "")]
                         + [ms(*agg(cf, task, m, s)) for s in SETTINGS] for m in models]))
        md.append("")
        rows = []
        for s in SETTINGS:
            c = agg(cf, task, "cascade", s)
            for m in pubs:
                p_ = agg(cf, task, m, s)
                if not p_[2]:
                    rows.append([label(s), name(m), "нет данных", "", "", ""])
                    ok_all = False
                    continue
                d, tol, ok = verdict(c, p_)
                ok_all &= ok
                rows.append([label(s), name(m), f"{ms(*c)} против {ms(*p_)}", f"{d:+.2f}", f"{tol:.2f}", "да" if ok else "нет"])
            best = max(SETTING_MODELS, key=lambda m: agg(cf, task, m, s)[0])
            b = agg(cf, task, best, s)
            d, tol, ok = verdict(c, b)
            rows.append([label(s), f"лучший наш baseline: {name(best)}", f"{ms(*c)} против {ms(*b)}", f"{d:+.2f}",
                         f"{tol:.2f}", "да" if ok else "нет"])
        md.append(table(["Val", "Против", "Каскад против", "Разница, п.п.", "Разброс, п.п.", "Выигрыш больше разброса"],
                        rows))
        md.append("")
    taus = S5 / "setting" / "cascade_tau.json"
    if taus.exists():
        t = read_json(taus)
        md.append(table(["Задача", "Seed", "tau (половина 0 / 1)"] + [f"Доля во 2-ю ступень, {label(s)}" for s in SETTINGS],
                        [[TASK_NAMES[e["task"]], e["seed"], f"{e['tau_by_half']['0']:.4f} / {e['tau_by_half']['1']:.4f}"]
                         + [f"{100 * e[s]['routed_share']:.1f}%" for s in SETTINGS] for e in t]))
        md.append("")
    md.append(f"→ Против всех воспроизведённых методов в их постановке: "
              f"**{'выигрыш больше разброса во всех условиях' if ok_all else 'выигрыш больше разброса не во всех условиях'}**.\n")
    return ok_all


# Every Kaggle session of step 5a: (output folder, kernel, remark)
SESSIONS = [("smishfuse-step5a", "smishfuse-step5a", "основной прогон; задачи Shim et al. упали (ошибка маски меток)"),
            ("smishfuse-step5a-setting-v1-stale-code", "smishfuse-step5a-setting",
             "версия 1: ноутбук получил прежнюю версию датасета кода, все задачи упали"),
            ("smishfuse-step5a-setting", "smishfuse-step5a-setting",
             "версия 2: задачи Shim et al. и дополнение setting_runs")]


def section_gpu(md):
    md.append("## 6. GPU-часы\n")
    rows, total = [], 0.0
    for folder, kernel, remark in SESSIONS:
        rt = RESULTS / "kaggle_runs" / folder / "results" / "runtime" / f"{kernel}.json"
        if not rt.exists():
            rows.append([kernel, "нет данных", "", remark])
            continue
        s = [x for x in read_json(rt)["sessions"] if x["kernel"] == kernel][-1]
        total += s["wall_seconds"] / 3600
        rows.append([kernel, f"{s['wall_seconds'] / 3600:.3f}", ", ".join(map(str, s["return_codes"])), remark])
    md.append(table(["Ноутбук", "Часы сессии 2× T4", "Коды возврата", "Примечание"], rows))
    md.append(f"\nВсего {total:.2f} ч (бюджет шага 5а увеличен до 3 ч).\n")


def main():
    md = ["# SmishFuse, шаг 5а: каскад, опубликованные методы, завышение из-за дубликатов\n",
          "Сгенерировано `scripts/make_step5a_report.py` из `results/step5a/`. Предварительно: 3 seed, только val "
          "(анализ завышения — train + val), **без проверки значимости**. Test не читался. Гиперпараметры "
          "каскада без подбора (`configs/step5a.yaml`).\n"]
    for f in sorted((S5 / "environment").glob("*.json")) if (S5 / "environment").exists() else []:
        md.append(f"- `{f.stem}`: расхождений с `requirements.txt`: {len(read_json(f)['mismatch'])}.")
    md.append("")
    cf = crossfit() if (S5 / "compare" / "crossfit.csv").exists() else None
    if cf is None:
        md.append("Результатов сравнения нет.\n")
    else:
        section_cascade(md, cf)
        section_published(md, cf)
    section_inflation(md)
    if cf is not None:
        section_answer(md, cf)
    section_setting(md)
    md.append("## 5. Наблюдения\n")
    md += [f"- {o}" for o in OBSERVATIONS] + [""]
    section_gpu(md)
    md.append("## 7. Проблемы и ограничения\n")
    errs = sorted((S5 / "errors").glob("*.txt")) if (S5 / "errors").exists() else []
    probs = []
    if errs:
        last = {f.read_text(encoding="utf-8").strip().splitlines()[-1][:120] for f in errs}
        probs.append(f"В `smishfuse-step5a` упали {len(errs)} единиц работы (все задачи Shim et al.: "
                     f"{'; '.join(f'`{x}`' for x in sorted(last))}). Причина — ошибка в нашем коде: метки задачи "
                     "smishing против spam не отфильтровывались вместе с текстами; в CV не было файлов выбора "
                     "параметров. Исправлено, все эти задачи посчитаны во втором ноутбуке; файлы ошибок первого "
                     "прогона оставлены в `results/step5a/errors/`.")
    md += [f"- {p}" for p in probs + PROBLEMS] + [""] + SUMMARY
    (ROOT / "reports" / "step5a_report.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/step5a_report.md")


# Checked against results/step5a/ when the report was written (2026-10-31).
OBSERVATIONS = [
    "Каскад (3 класса). При выбранном cross-fit'ом tau во вторую ступень уходит всего 1.2–1.7% сообщений. На "
    "чистом val каскад 0.9102 ± 0.0181 — ниже лучшего baseline (DistilBERT без аугм., 0.9249) в пределах "
    "разброса. На HELD-OUT 20% каскад 0.8761 — ниже LinearSVM char (аугм.) 0.8867 больше разброса. От "
    "усреднения каскад во всех условиях отличается в пределах разброса.",
    "Кривая без выбора tau даёт максимум на чистом val при 5% сообщений во второй ступени (0.9236 ± 0.0077). "
    "Cross-fit выбирает меньшую долю: tau подбирается на половине val (287 сообщений), этот выбор шумный.",
    "Расчётная задержка каскада 3.5–3.8 мс на сообщение. У DistilBERT 57.5 мс, у усреднения 60.3 мс. Это "
    "расчёт по замерам шага 3, а не замер.",
    "Mahmud et al.: наша accuracy на чистом val 0.9803 при заявленной 0.9817. F1 malicious 0.9353 при "
    "заявленном F1 0.9715; какой F1 в статье, не указано. При HELD-OUT 20% macro F1 падает до 0.8577.",
    "В постановке Mahmud каскад выше CNN-BiGRU больше разброса во всех условиях (+2.84 … +12.60 п.п.). Лучший "
    "наш baseline каскад не обходит: на чистом val −0.32 при разбросе 0.32; на HELD-OUT 10% и 20% ниже "
    "LinearSVM char без аугм. больше разброса (−0.75 и −0.34).",
    "Shim et al.: BERT-base у нас — accuracy 0.8900 и F1 smishing 0.8933 при заявленных 0.898 и 0.911. "
    "DistilBERT и RoBERTa ниже заявленного: accuracy 0.853 и 0.856 при заявленных 0.907 и 0.916. Выбор "
    "batch / lr / эпох по сетке статьи различается от seed к seed.",
    "В постановке Shim на чистом val каскад 0.8597 ниже воспроизведённого BERT-base 0.8897 больше разброса "
    "(−3.00 при 1.88). С DistilBERT и RoBERTa разница в пределах разброса. На HELD-OUT 5/10/20% каскад выше "
    "всех трёх методов больше разброса. Лучший наш baseline он не обходит ни в одном условии: на чистом val, "
    "HELD-OUT 5% и 20% он ниже больше разброса.",
    "RoBERTa-base в CV завышения в 11 из 30 фолдов вырождается в предсказание одного класса (macro F1 "
    "0.35–0.36). Выбранные параметры: batch 8, lr 5e-5, 6 эпох.",
    "Завышение. У CNN-BiGRU (ham против остальных) грязный CV выше чистого: accuracy +0.24 п.п., macro F1 "
    "+0.96 п.п. Accuracy грязного CV 0.9824 близка к заявленной в статье 0.9817.",
    "В 3 классах и в smishing против spam грязный CV ниже чистого: LinearSVM и LogReg −1.3 п.п. accuracy и "
    "−3.6 п.п. macro F1, BERT-base и DistilBERT −3.6 и −5.3 п.п. accuracy. Все 40 групп с ничейным конфликтом "
    "меток — пары, где один и тот же текст размечен как smishing и как spam (`reports/conflicts/mendeley.csv`). "
    "В грязном варианте они дают противоречивые метки; в бинарной задаче ham против остальных такого "
    "противоречия нет. Анализ смешивает два эффекта — дубликаты и противоречивые метки; отдельно они не "
    "разделялись.",
    "В постановке Shim tau сильно меняется от половины к половине (0 … 0.99), во вторую ступень уходит от 4% до "
    "58% сообщений. В половине около 45 сообщений.",
]
PROBLEMS = [
    "Обе ступени каскада взяты из шага 4в: их ранняя остановка (DistilBERT) и параметры (LinearSVM — шаг 2) "
    "выбраны по всему чистому val. Cross-fitting честен только для tau, поэтому числа на val оптимистичны для "
    "всех моделей шага 4в.",
    "Половины val маленькие: 287 сообщений, около 24 smishing и 21 spam в каждой. Одно сообщение меняет macro F1 "
    "половины заметно.",
    "Воспроизведения: то, чего нет в статьях, задано нами (`configs/step5a.yaml`, пометки «ours»). Модель Mahmud "
    "et al. написана на PyTorch по описанию Keras-модели; GRU с recurrent dropout реализован вручную.",
    "Заявленные числа получены на других разбиениях без дедупликации; положительный класс в обеих статьях не "
    "указан. Сравнение «заявлено / у нас» поэтому качественное.",
    "Munoz & Islam 2025: полный текст не прочитан (HTTP 403), видна только аннотация. Remmide et al. 2024: "
    "числа только из обзора arXiv:2604.11429. Статья Mishra & Soni о наборе данных (SoCPaR 2022) не прочитана.",
    "Первая версия `smishfuse-step5a-setting` получила прежнюю версию датасета кода: новая версия ещё "
    "обрабатывалась на Kaggle. Все задачи упали за 0.024 ч. Её вывод лежит в "
    "`results/kaggle_runs/smishfuse-step5a-setting-v1-stale-code/`. Вторая версия запущена после проверки, что "
    "Kaggle отдаёт новую версию кода (совпадение CODE_SHA256).",
    "Для CV завышения Shim et al. взят выбор seed 1: по правилу конфига одинакового выбора у нескольких seed не "
    "нашлось.",
    "Задержка символьного LinearSVM не измерялась, в расчёте вместо неё задержка LinearSVM шага 2.",
]
SUMMARY = [
    "## Итог\n",
    "**Ответ — нет.** Каскад не обходит лучший наш baseline больше разброса ни в одном условии. На 3 классах "
    "при HELD-OUT 20% он ниже символьного LinearSVM больше разброса. Опубликованные методы каскад обходит не "
    "везде. В постановке Mahmud et al. он выше CNN-BiGRU больше разброса во всех условиях. В постановке Shim et "
    "al. — на всех уровнях HELD-OUT, но не на чистом val: там он ниже воспроизведённого BERT-base больше "
    "разброса. Сильная сторона каскада — расчётная задержка 3.5–3.8 мс против 57.5 мс у DistilBERT при "
    "качестве в пределах разброса от него на чистом val. Предварительно: 3 seed, только val, без проверки "
    "значимости.",
    "",
]


if __name__ == "__main__":
    main()
