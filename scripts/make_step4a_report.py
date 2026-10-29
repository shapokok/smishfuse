"""Builds reports/step4a_report.md from results/step4a/ and results/kaggle_runs/smishfuse-step4a/.

Run after scripts/step4a_spam_errors.py (local, saved predictions) and
scripts/collect_results.py smishfuse-step4a (Kaggle prototype run).
Message texts are not written to the report (raw data is not redistributed);
they are in reports/step4a_hard_cases.csv, which is gitignored.
Run: python scripts/make_step4a_report.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.registry import SHORT_NAMES  # noqa: E402

RESULTS = ROOT / "results"
S4 = RESULTS / "step4a"
CLASSES = ["ham", "smishing", "spam"]
NAMES = {"smishfuse": "SmishFuse (гейт)", "smishfuse_concat": "SmishFuse без гейта (конкатенация)",
         "features_mlp": "Только признаки + MLP", "distilbert_text_only": "DistilBERT без признаков (та же реализация)",
         "tfidf_features_logreg": "TF-IDF + признаки + LogReg", "logreg": "LogReg (шаг 2)",
         "linear_svm": "LinearSVM (шаг 2)", "distilbert_step3": "DistilBERT (шаг 3, веса seed 1)"}
ORDER = list(NAMES)
BASELINES = ["logreg", "linear_svm", "distilbert_step3"]


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def pct(a: int, n: int) -> str:
    return f"{a}/{n} ({a / n:.0%})" if n else "0/0"


def error_section(md: list) -> None:
    shares = pd.read_csv(S4 / "spam_error_shares.csv")
    summ = read_json(S4 / "neighbour_summary.json")
    hard = pd.read_csv(S4 / "hard_cases.csv", keep_default_na=False)
    cm = pd.read_csv(S4 / "confusion_seed1.csv")
    groups = {(g["group"], g["label"]): g for g in summ["groups"]}

    md.append("## 1. Анализ ошибок класса spam (mendeley_3class, только сохранённые предсказания test)\n")
    md.append(f"Модели шагов 2 и 3 ({len(summ['models'])}), seed {summ['seed']}; доли «mean5» — среднее по 5 seed. "
              f"В test {int(shares['n_spam_test'].iloc[0])} сообщений spam. Ничего не обучалось.\n")
    rows = [[SHORT_NAMES[r.model], f"{r.spam_recall_seed1:.3f}", f"{r.spam_to_smishing_seed1:.3f}",
             f"{r.spam_to_ham_seed1:.3f}", f"{r.spam_recall_mean5:.3f}", f"{r.spam_to_smishing_mean5:.3f}",
             f"{r.spam_to_ham_mean5:.3f}", f"{r.smishing_to_spam_seed1:.3f}", f"{r.ham_to_spam_seed1:.3f}"]
            for r in shares.itertuples()]
    md.append(table(["Модель", "Recall spam", "spam→smishing", "spam→ham", "Recall spam (mean5)",
                     "spam→smishing (mean5)", "spam→ham (mean5)", "smishing→spam", "ham→spam"], rows))
    md.append("\nДоли — от числа сообщений истинного класса. Полные матрицы ошибок (seed 1): "
              "`results/step4a/confusion_seed1.csv`.\n")
    tot = cm[cm["true"] == "spam"].groupby("pred")["count"].sum()
    md.append(f"Сумма по 9 моделям (seed 1) для истинного spam: верно {int(tot.get('spam', 0))}, → smishing "
              f"{int(tot.get('smishing', 0))}, → ham {int(tot.get('ham', 0))}.\n")

    md.append(f"### Трудные сообщения (ошибаются ≥ {summ['hard_min_models_wrong']} из {len(summ['models'])} моделей)\n")
    md.append(f"Найдено {summ['n_hard']} из {summ['n_test']} сообщений test; по истинной метке: "
              + ", ".join(f"{c} {int((hard['label'] == c).sum())}" for c in CLASSES) + ". Список с текстом и "
              "меткой — `reports/step4a_hard_cases.csv` (в git не попадает: сырые данные не распространяем), "
              "без текста — `results/step4a/hard_cases.csv`.\n")
    dirs = hard.groupby(["label", "majority_wrong_pred"]).size()
    md.append(table(["Истинная метка", "Чаще всего предсказано вместо неё", "Сообщений"],
                    [[a, b, int(n)] for (a, b), n in dirs.items()]))
    md.append("")

    md.append("### Похожие сообщения в train с другой меткой\n")
    md.append(f"Жаккар ≥ {summ['jaccard']} по символьным 5-граммам нормализованного текста (как в шаге 1). "
              f"MinHash LSH ({summ['num_perm']} перестановок, порог кандидатов {summ['lsh_candidate_threshold']}) "
              f"пропустил {summ['lsh_missed_pairs']} из {summ['pairs_exact']} пар у порога, поэтому все числа ниже — "
              f"по точному перебору всего train (4017 сообщений). Контроль — сообщения test, на которых не "
              f"ошиблась ни одна модель ({summ['n_easy_all_correct']}).\n")
    rows = []
    for c in CLASSES + ["all"]:
        h, e = groups.get(("hard", c)), groups.get(("easy", c))
        rows.append([c, pct(h["with_any_neighbour"], h["n"]) if h else "—",
                     pct(h["with_other_label_neighbour"], h["n"]) if h else "—",
                     pct(e["with_any_neighbour"], e["n"]) if e else "—",
                     pct(e["with_other_label_neighbour"], e["n"]) if e else "—"])
    md.append(table(["Метка", "Трудные: есть сосед", "Трудные: сосед с другой меткой", "Контроль: есть сосед",
                     "Контроль: сосед с другой меткой"], rows))
    other = hard[hard["n_other_label"] > 0]
    if len(other):
        md.append("\nТрудные сообщения с соседом другой метки в train: " + "; ".join(
            f"`{r.id}` ({r.label}, сосед `{r.nearest_other_id}` — {r.nearest_other_label}, J = "
            f"{float(r.max_jaccard_other):.2f})" for r in other.itertuples()) + ".")
    md.append(f"\nВнутри самого test соседей с другой меткой (J ≥ {summ['jaccard']}) у трудных сообщений: "
              f"{summ['hard_with_other_label_neighbour_in_test']}. Шаг 1 уже объединил группы с J ≥ 0.9 и решил "
              "конфликты меток голосованием, поэтому противоречия с J ≥ 0.9 здесь найтись не могут; проверка "
              "покрывает диапазон 0.5–0.9.\n")

    hs, es = groups[("hard", "spam")], groups[("easy", "spam")]
    boundary = int(((hard["label"] == "spam") & (hard["majority_wrong_pred"] == "smishing")).sum()
                   + ((hard["label"] == "smishing") & (hard["majority_wrong_pred"] == "spam")).sum())
    md.append("### Вывод по гипотезе шума разметки spam/smishing\n")
    md += [f"- {x}" for x in [
        f"Ошибки сосредоточены на границе spam/smishing и идут в обе стороны: {boundary} из {summ['n_hard']} "
        "трудных сообщений — путаница spam↔smishing (большинство ошибившихся моделей выбирает другой из этих "
        "двух классов); остальные — путаница с ham.",
        f"Прямые противоречия разметки есть: у {pct(hs['with_other_label_neighbour'], hs['n'])} трудных spam "
        f"в train есть похожее сообщение с меткой smishing (у контрольных spam — "
        f"{pct(es['with_other_label_neighbour'], es['n'])}); при ручном просмотре это один и тот же шаблон "
        "рассылки с разными метками (оба трудных spam похожи на одно и то же сообщение train). Ниже порога "
        "0.5 есть и пара внутри test: `mendeley_05863` (smishing) и `mendeley_05923` (spam) — один шаблон "
        "«FREE auction» (J = 0.44), обе ошибочно классифицируют 6 из 9 моделей.",
        f"Но у большинства трудных сообщений ({summ['n_hard'] - groups[('hard', 'all')]['with_any_neighbour']} из "
        f"{summ['n_hard']}) похожих сообщений в train нет вовсе, так что шум на уровне почти-дубликатов "
        "объясняет лишь малую часть ошибок. Трудные сообщения реже контрольных имеют похожие в train вообще: "
        f"smishing {pct(groups[('hard', 'smishing')]['with_any_neighbour'], groups[('hard', 'smishing')]['n'])} "
        f"против {pct(groups[('easy', 'smishing')]['with_any_neighbour'], groups[('easy', 'smishing')]['n'])}, "
        f"spam {pct(hs['with_any_neighbour'], hs['n'])} против {pct(es['with_any_neighbour'], es['n'])}. "
        "То есть ошибки чаще приходятся на сообщения без близкого примера в train (обобщение на новые "
        "шаблоны), чем на противоречивые дубликаты. Остальное — неоднозначность самих определений spam и "
        "smishing без текстового сходства или ошибки моделей; этот анализ их не различает.",
        "Итог: гипотеза подтверждается частично — противоречивая разметка шаблонов есть и попадает в трудные "
        "случаи чаще, чем в лёгкие, но числа малые (единицы сообщений), проверки значимости нет, и основную "
        "массу ошибок spam она не объясняет.",
    ]]
    md.append("")


def prototype_section(md: list) -> None:
    md.append("## 2. Прототип SmishFuse: сравнение на val (mendeley_3class, seed 1)\n")
    cmp_path = S4 / "val_comparison.csv"
    if not cmp_path.exists():
        md.append("Результатов прототипа нет (ноутбук `smishfuse-step4a` не собран).\n")
        return
    cfg = yaml.safe_load((ROOT / "configs" / "step4a.yaml").read_text(encoding="utf-8"))
    lex_hash = hashlib.sha256((ROOT / cfg["lexicon"]).read_bytes()).hexdigest()[:16]
    feats = read_json(S4 / "features_train_stats.json")["features"]
    md.append(
        f"- Признаки ({len(feats)}): {', '.join(f'`{f}`' for f in feats)}; стандартизация по train. Словари и "
        f"сокращатели ссылок — `{cfg['lexicon']}` (sha256 {lex_hash}…), зафиксированы до обучения.\n"
        f"- Текст: `{cfg['text']['checkpoint']}`, [CLS], дообучается; lr {cfg['text']['lr']:.3g}, batch "
        f"{cfg['text']['batch_size']}, weight decay {cfg['text']['weight_decay']}, warmup {cfg['text']['warmup_ratio']}, "
        f"до {cfg['text']['max_epochs']} эпох, patience {cfg['text']['patience']}, max_len {cfg['text']['max_len']} — "
        "как у лучшего DistilBERT шага 3 на mendeley_3class.\n"
        f"- Слияние: d = {cfg['fusion']['d']}, f' = MLP (скрытый слой {cfg['fusion']['feature_mlp_hidden']}), "
        f"e' = Linear, g = sigmoid(W[f; e] + b), h = g·f' + (1 − g)·e', dropout {cfg['fusion']['dropout']}; "
        f"головы на 3 класса и бинарную, потеря CE₃ + {cfg['fusion']['binary_loss_weight']}·CE_bin; lr новых слоёв "
        f"{cfg['fusion']['head_lr']} (энкодер — {cfg['text']['lr']:.3g}). MLP только на признаках: Adam lr "
        f"{cfg['features_only']['lr']}, до {cfg['features_only']['max_epochs']} эпох, patience "
        f"{cfg['features_only']['patience']}.\n"
        "- Подбора нет. Нейросети останавливаются по macro F1 на val (как в шаге 3), а параметры LogReg и "
        "LinearSVM подобраны на этом же val в шаге 2, поэтому все числа на val оптимистичны; сравнение "
        "предварительное. Test не читался.\n"
    )
    df = pd.read_csv(cmp_path).set_index("model")
    rows = []
    for m in ORDER:
        if m not in df.index:
            rows.append([NAMES[m], "нет результата", "", "", "", "", ""])
            continue
        r = df.loc[m]
        ep = f"{int(r['best_epoch']) + 1}/{int(r['epochs_run'])}" if pd.notna(r.get("best_epoch")) else "—"
        rows.append([NAMES[m], f"{r['macro_f1']:.4f}", f"{r['recall_ham']:.4f}", f"{r['recall_smishing']:.4f}",
                     f"{r['recall_spam']:.4f}", f"{r['f1_spam']:.4f}", ep])
    md.append(table(["Модель", "Macro F1", "Recall ham", "Recall smishing", "Recall spam", "F1 spam",
                     "Лучшая эпоха / эпох"], rows))
    n_val = int(read_json(next((S4 / "val").glob("*.json")))["n_val"])
    md.append(f"\nval: {n_val} сообщений. Предсказания: `results/step4a/val_preds/<model>.csv`.\n")
    d3 = S4 / "val" / "distilbert_step3.json"
    if d3.exists():
        r = read_json(d3)
        md.append(f"Проверка загрузки весов DistilBERT шага 3: macro F1 на val {r['macro_f1']:.4f}, записано в шаге 3 "
                  f"{r['val_macro_f1_recorded_in_step3']:.4f}.\n")

    sf = S4 / "val" / "smishfuse.json"
    if sf.exists():
        r = read_json(sf)
        md.append("### Вес гейта по классам\n")
        md.append("g — вес ветви признаков (среднее по d измерениям), 1 − g — вес текстовой ветви; среднее по "
                  "сообщениям val лучшей эпохи.\n")
        md.append(table(["Класс", "g по истинному классу (± std)", "1 − g", "g по предсказанному классу"],
                        [[c, f"{r['gate_mean_by_true_class'][c]:.3f} ± {r['gate_std_by_true_class'][c]:.3f}",
                          f"{1 - r['gate_mean_by_true_class'][c]:.3f}",
                          f"{r['gate_mean_by_pred_class'][c]:.3f}" if r['gate_mean_by_pred_class'][c] is not None
                          else "—"] for c in CLASSES]))
        md.append("")

    md.append("### Совпадение SmishFuse и текстовой ветви\n")
    if {"smishfuse", "distilbert_text_only"} <= set(df.index):
        pa = pd.read_csv(S4 / "val_preds" / "smishfuse.csv")
        pb = pd.read_csv(S4 / "val_preds" / "distilbert_text_only.csv")
        same = int((pa["y_pred"] == pb["y_pred"]).sum())
        dp = (pa.filter(like="prob_") - pb.filter(like="prob_")).abs().max(axis=1)
        md.append(
            f"Предсказанные классы SmishFuse и «DistilBERT без признаков» совпадают на {same} из {len(pa)} "
            f"сообщений val, поэтому все метрики одинаковы. Это разные модели: вероятности расходятся (максимум "
            f"|Δp| = {dp.max():.3f}, у {int((dp > 0.05).sum())} сообщений больше 0.05), macro F1 на val по эпохам "
            "различается на эпохах 1–3 и совпадает на 4-й и 5-й. Значит, ветвь признаков и гейт на val не изменили ни одного решения; "
            "средний вес гейта около 0.45 (см. выше) при этом не говорит о вкладе признаков в решения.\n")

    md.append("### Есть ли запас над лучшим baseline\n")
    refs = {"logreg": "LogReg (шаг 2)", "linear_svm": "LinearSVM (шаг 2)"}
    rec = {m: read_json(ROOT / "results" / "best_params" / "mendeley_3class" / f"{m}.json")["val_macro_f1"]
           for m in refs}
    md.append(
        "LogReg и LinearSVM, заново обученные в этой сессии Kaggle с теми же параметрами и seed 1, дали на val "
        "меньше, чем записано при подборе шага 2 на тех же train / val: " + "; ".join(
            f"{refs[m]} {df.loc[m, 'macro_f1']:.4f} против {rec[m]:.4f}" for m in refs if m in df.index) +
        ". Локальная проверка (окружение шага 2, seed 1, тот же порядок train) воспроизводит 0.9207 для LogReg; "
        "входные тексты на Kaggle те же (средние всех признаков по train совпадают). Отличается окружение: в "
        "ноутбуке 4а не были закреплены numpy и pandas (numpy 2.1.3 и pandas 2.3.3 против 2.4.6 и 3.1.0 в шаге 2), "
        "другая платформа BLAS. Поэтому запас считается двумя способами. Уточнение шага 4б: в окружении с "
        "версиями ровно из `requirements.txt` вероятности LogReg и LinearSVM совпали с шагом 4а точно, т.е. "
        "дело не в версиях пакетов, а, вероятно, в платформе (не проверялось).\n")
    if "smishfuse" in df.index and all(b in df.index for b in BASELINES):
        sf = df.loc["smishfuse", "macro_f1"]
        best = max(BASELINES, key=lambda b: df.loc[b, "macro_f1"])
        ref_best = max(rec, key=rec.get)
        ref_val = max(rec[ref_best], df.loc["distilbert_step3", "macro_f1"])
        ref_name = refs[ref_best] if rec[ref_best] >= df.loc["distilbert_step3", "macro_f1"] else NAMES["distilbert_step3"]
        ds = 100 * (df.loc["smishfuse", "f1_spam"] - df.loc[best, "f1_spam"])
        md.append(table(["Сравнение", "Лучший baseline", "Macro F1 baseline", "Macro F1 SmishFuse", "Разница, п.п."], [
            ["baseline этой сессии", NAMES[best], f"{df.loc[best, 'macro_f1']:.4f}", f"{sf:.4f}",
             f"{100 * (sf - df.loc[best, 'macro_f1']):+.2f}"],
            ["записанные значения val шагов 2–3 (осторожная оценка)", f"{ref_name} (val при подборе)"
             if ref_name in refs.values() else ref_name, f"{ref_val:.4f}", f"{sf:.4f}", f"{100 * (sf - ref_val):+.2f}"],
        ]))
        tx = df.loc["distilbert_text_only", "macro_f1"] if "distilbert_text_only" in df.index else float("nan")
        seed_vals = ", ".join(f"{max(h['val_macro_f1'] for h in read_json(f)['history']):.4f}" for f in sorted(
            (ROOT / "results" / "seed_runs" / "mendeley_3class" / "distilbert").glob("seed*.json")))
        md.append(
            f"\nF1 spam: SmishFuse {df.loc['smishfuse', 'f1_spam']:.4f}, {ds:+.2f} п.п. к {NAMES[best]} этой сессии.\n\n"
            f"Ответ: на val у SmishFuse есть запас над лучшим baseline, от {100 * (sf - ref_val):+.2f} до "
            f"{100 * (sf - df.loc[best, 'macro_f1']):+.2f} п.п. macro F1. Но весь этот запас даёт текстовая ветвь: "
            f"«DistilBERT без признаков» в той же реализации даёт ровно тот же результат ({tx:.4f}), а вариант без гейта "
            f"(конкатенация) — {df.loc['smishfuse_concat', 'macro_f1']:.4f}. Отличие нашей текстовой модели от "
            "DistilBERT шага 3 (голова d = 256, lr головы 1e-3, вспомогательная бинарная потеря) не отделено от "
            f"разброса по seed: у DistilBERT шага 3 лучший macro F1 на val по seed 1–5 — {seed_vals} "
            f"(`results/seed_runs/`), т.е. результат SmishFuse ({sf:.4f}) лежит внутри разброса одной текстовой "
            "модели по seed. Это предварительная оценка: один seed, только val, "
            f"без проверки значимости; на val {n_val} сообщений, одна ошибка в редком классе меняет macro F1 на "
            "десятые доли п.п.\n")
    else:
        md.append("Нельзя посчитать: нет результата SmishFuse или одного из baseline (раздел 4).\n")


def gpu_section(md: list) -> None:
    md.append("## 3. Израсходованные GPU-часы\n")
    f = RESULTS / "kaggle_runs" / "smishfuse-step4a" / "results" / "runtime" / "smishfuse-step4a.json"
    if not f.exists():
        md.append("Нет данных.\n")
        return
    sess = read_json(f)["sessions"]
    rows = [[i + 1, f"{s['wall_seconds'] / 3600:.2f}", s["environment"].get("gpus", "").split("\n")[0],
             ", ".join(map(str, s["return_codes"]))] for i, s in enumerate(sess)]
    md.append(table(["Сессия", "Часы", "GPU", "Код возврата"], rows))
    md.append(f"\nВсего {sum(s['wall_seconds'] for s in sess) / 3600:.2f} GPU-ч (бюджет 1.5); квота списывается по "
              "времени сессии, включая установку пакетов и загрузку моделей.\n")


def problems_section(md: list) -> None:
    md.append("## 4. Проблемы и ограничения\n")
    items = []
    for f in sorted((S4 / "errors").glob("*.txt")) if (S4 / "errors").exists() else []:
        items.append(f"Ошибка в `{f.stem}`: `{f.read_text(encoding='utf-8').strip().splitlines()[-1][:200]}` "
                     f"(`results/step4a/errors/{f.name}`).")
    items += [
        "Один seed и только val: разброс по seed и переобучение на val не оценены; на test прототип не проверялся.",
        "Признак «короткий код» (отдельное число из 4–6 цифр) срабатывает и на годы, суммы без символа валюты и "
        "время без двоеточия; телефон — последовательность с ≥ 7 цифрами. Орфографические ошибки считаются "
        "английским словарём pyspellchecker, поэтому SMS-сокращения (txt, pls, msg) и не-английские слова тоже "
        "считаются ошибками; слова из одной буквы и цифры пропускаются.",
        "Новые слои (проекции, гейт, MLP, головы) учатся с lr 1e-3, энкодер — с lr лучшего DistilBERT; у "
        "DistilBERT шага 3 голова училась с тем же lr, что и энкодер, поэтому «DistilBERT без признаков» здесь "
        "не тождествен модели шага 3 (отличаются голова, lr головы и бинарная потеря).",
        "Анализ ошибок использует предсказания seed 1; доли по 5 seed приведены для устойчивости.",
        "В ноутбуке 4а закреплены версии scikit-learn, scipy, xgboost, joblib (как в шаге 2), но не numpy и "
        "pandas. LogReg и LinearSVM на val разошлись со значениями шага 2 (раздел 2); шаг 4б показал, что "
        "причина не в версиях (с закреплёнными версиями — те же числа). Сравнение внутри сессии корректно: все "
        "модели обучены в одном окружении.",
    ]
    md += [f"- {x}" for x in items]
    md.append("")


def summary_section(md: list) -> None:
    md.append("## 5. Итог\n")
    md.append("Сделано: решения по шагу 3 записаны в README (10–12), время обучения убрано из итоговых таблиц, "
              "локальные веса BERT удалены; анализ ошибок spam по сохранённым предсказаниям 9 моделей; прототип "
              "SmishFuse и 7 сравнений на val (один seed, test не тронут) на Kaggle.\n")
    md.append("Требует решения:\n")
    md += [f"- {x}" for x in [
        "Ветвь признаков и гейт на val не изменили ни одного предсказания текстовой модели. Продолжать ли "
        "SmishFuse в этой архитектуре (и что менять: признаки, место слияния, lr новых слоёв), или сначала "
        "проверить разброс по seed (например, 3–5 seed только на val), прежде чем тратить test.",
        "Как считать baseline на val для дальнейших сравнений: записанные значения подбора шага 2 или "
        "пересчёт в одном окружении с прототипом (решено в шаге 4б: пересчёт в закреплённом окружении).",
        "Гипотеза шума разметки подтверждается частично (раздел 1); оставить как ограничение в статье или "
        "разметить спорные сообщения отдельно (не делалось).",
    ]]
    md.append("")


def main():
    md = ["# SmishFuse, шаг 4а: анализ ошибок spam и прототип SmishFuse\n",
          "Сгенерировано `scripts/make_step4a_report.py`. Полный протокол (20 попыток, 5 seed) не запускался; "
          "test в прототипе не использовался.\n"]
    error_section(md)
    prototype_section(md)
    gpu_section(md)
    problems_section(md)
    summary_section(md)
    out = ROOT / "reports" / "step4a_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
