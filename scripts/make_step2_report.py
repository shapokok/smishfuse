"""Builds reports/step2_baselines_report.md from results/.

All numbers come from results/*.csv, results/best_params, results/tables and
configs/step2.yaml; the only hand-written parts are the method notes.
Run: python scripts/make_step2_report.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import make_tables  # noqa: E402

MODELS = ["logreg", "linear_svm", "random_forest", "xgboost"]
MODEL_NAMES = {"logreg": "LogReg", "linear_svm": "LinearSVM", "random_forest": "RandomForest", "xgboost": "XGBoost"}
LOW_RECALL = 0.90


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def fmt_params(d: dict) -> str:
    parts = []
    for k, v in sorted(d.items()):
        parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
    return ", ".join(parts)


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "step2.yaml").read_text(encoding="utf-8"))
    proto = cfg["protocol"]
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    metrics = metrics[metrics["model"].isin(MODELS)]  # results/metrics.csv also holds the step-3 models
    eff = pd.read_csv(RESULTS / "efficiency.csv")
    sig = pd.read_csv(RESULTS / "significance_step2.csv")
    env = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))
    best = {(t, m): json.loads((RESULTS / "best_params" / t / f"{m}.json").read_text(encoding="utf-8"))
            for t in cfg["tasks"] for m in MODELS}
    test_counts = {}
    for t in cfg["tasks"]:
        y = pd.read_csv(RESULTS / "raw" / t / MODELS[0] / f"seed{proto['eval_seeds'][0]}.csv")["y_true"]
        test_counts[t] = ", ".join(f"{c} {n}" for c, n in y.value_counts().sort_index().items())
    means = metrics.groupby(["task", "model"]).mean(numeric_only=True)
    stds = metrics.groupby(["task", "model"]).std(ddof=1, numeric_only=True)

    md = ["# SmishFuse, шаг 2: модуль оценки и классические baseline\n"]
    md.append(
        "Сгенерировано скриптом `scripts/make_step2_report.py` из `results/`. Запуск всего шага: "
        "`.venv/bin/python src/run_step2.py`. Нейросети не обучались.\n"
    )
    md.append("## Протокол\n")
    md.append(
        f"- Задачи: {', '.join(f'`{t}`' for t in cfg['tasks'])}. Данные — сплиты шага 1 "
        "(Super SMS только в варианте `super_sms_clean`).\n"
        "- Модели: TF-IDF + LogReg, LinearSVM, RandomForest, XGBoost. Признаки TF-IDF (слова 1–2-граммы, "
        "символы 3–5-граммы или их объединение) выбираются подбором вместе с гиперпараметрами модели. "
        "Пространства поиска заданы в `configs/step2.yaml`.\n"
        f"- Подбор: Optuna TPE (seed {proto['tuning_seed']}), ровно {proto['n_trials']} попыток на пару "
        "модель-задача. Каждая попытка обучается на train и оценивается по macro F1 на val. Test при подборе "
        "не читается.\n"
        f"- Финальная оценка: лучшие параметры, seed {proto['eval_seeds'][0]}..{proto['eval_seeds'][-1]}. Seed "
        "задаёт random_state модели и перемешивание порядка train. Обучение только на train, одно предсказание "
        "на test.\n"
        f"- Метрики: accuracy, macro и weighted F1, precision/recall/F1 по классам, balanced accuracy, PR-AUC "
        "(для бинарных задач по положительному классу, для трёх классов — среднее one-vs-rest), recall при "
        f"FPR = {proto['fpr_target']:.0%} (бинарные задачи, положительный класс = malicious), ECE "
        f"({proto['ece_bins']} бинов, top-label). Предсказание = argmax вероятностей.\n"
        f"- Значимость: лучшая по среднему macro F1 модель сравнивается с остальными на предсказаниях seed "
        f"{proto['significance_seed']}: парный бутстреп разницы macro F1 ({proto['bootstrap_n']} повторов, seed "
        f"{proto['bootstrap_seed']}, 95% ДИ по перцентилям) и тест Макнемара (точный биномиальный при числе "
        f"расхождений < 25, иначе χ² с поправкой на непрерывность). Поправка Холма применяется внутри каждой "
        f"задачи, отдельно для каждого теста, α = {proto['alpha']}.\n"
        "- Юнит-тесты `tests/` (сверка с sklearn и ручным расчётом) запускаются первым шагом `run_step2.py`; "
        "при падении шаг останавливается.\n"
    )

    # 1. Tables
    md.append("## 1. Итоговые таблицы (test, среднее ± стандартное отклонение по 5 seed)\n")
    md.append("Жирным — лучшее значение в столбце, подчёркнуто — второе среди классических моделей (для ECE лучше "
              "меньшее). Таблицы со всеми моделями шагов 2 и 3: `results/tables/<task>.md|.tex`; все запуски: `results/metrics.csv`; предсказания: "
              "`results/raw/<task>/<model>/seed<k>.csv`.\n")
    for t in cfg["tasks"]:
        md.append(f"### {t}\n")
        cols = make_tables.task_columns(cfg["tasks"][t])
        md.append(make_tables.markdown(make_tables.summarize(metrics, t, cols), cols))

    # 2. Best params
    md.append("## 2. Лучшие гиперпараметры (подбор на val)\n")
    rows = []
    for t in cfg["tasks"]:
        for m in MODELS:
            b = best[(t, m)]
            v = b["vectorizer"]
            rows.append([t, MODEL_NAMES[m], v["features"], v["max_features"], v["min_df"],
                         fmt_params(b["model_params"]), f"{b['val_macro_f1']:.4f}", b["trial"],
                         f"{means.loc[(t, m), 'macro_f1']:.4f}"])
    md.append(table(["Задача", "Модель", "Признаки", "max_features", "min_df", "Параметры модели",
                     "Macro F1 val", "Лучшая попытка", "Macro F1 test (среднее)"], rows))
    md.append("\nВсе попытки: `results/tuning/<task>/<model>.csv`; лучшие параметры: "
              "`results/best_params/<task>/<model>.json`.\n")

    # 3. Significance
    md.append("## 3. Значимость различий (seed 1, лучшая модель против остальных)\n")
    rows = []
    for r in sig.itertuples():
        rows.append([r.task, MODEL_NAMES[r.best_model], MODEL_NAMES[r.other_model],
                     f"{r.diff_macro_f1:+.4f} [{r.ci95_low:+.4f}; {r.ci95_high:+.4f}]",
                     f"{fmt_p(r.p_bootstrap)} / {fmt_p(r.p_bootstrap_holm)}",
                     "да" if r.significant_bootstrap_holm else "нет",
                     f"{r.mcnemar_b_best_only_correct} / {r.mcnemar_c_other_only_correct}",
                     r.mcnemar_method, f"{fmt_p(r.p_mcnemar)} / {fmt_p(r.p_mcnemar_holm)}",
                     "да" if r.significant_mcnemar_holm else "нет"])
    md.append(table(["Задача", "Лучшая", "Против", "Δ macro F1 [95% ДИ]", "p бутстреп / Холм",
                     "Значимо (бутстреп)", "Макнемар b / c", "Метод", "p Макнемар / Холм", "Значимо (Макнемар)"],
                    rows))
    md.append("\nb — только лучшая модель права, c — только другая. Полная таблица: `results/significance_step2.csv` "
              "(сравнение всех моделей шагов 2 и 3 — `results/significance.csv`, отчёт шага 3).\n")

    # 4. Efficiency (decision 10: training time is not reported, only latency and size)
    md.append("## 4. Задержка и размер модели\n")
    lat = pd.read_csv(RESULTS / "latency" / "latency.csv").set_index(["task", "model"])
    lat_env = json.loads((RESULTS / "latency" / "environment.json").read_text(encoding="utf-8"))
    md.append(
        "Единый замер шага 3: одна сессия Kaggle для всех моделей шагов 2 и 3, "
        f"{lat_env['cpu_model']}, 1 поток. Пайплайн заново обучен на train с лучшими параметрами и seed 1; "
        f"задержка — медиана по {lat_env['n_messages']} предсказаниям с батчем 1 (TF-IDF + модель, после "
        f"{lat_env['warmup']} прогревочных вызовов) на сообщениях test, по кругу, если их меньше; размер — "
        "сериализованный joblib-пайплайн в МБ (`results/latency/latency.csv`). Время обучения в итоговые таблицы "
        "не входит (решение 10 в README): модели обучались на разных машинах. Прежние локальные замеры остаются "
        "в `results/efficiency.csv`. Задержка зависит от машины и нагрузки и не воспроизводится побайтно.\n"
    )
    rows = [[t, MODEL_NAMES[m], int(eff[(eff["task"] == t) & (eff["model"] == m)]["n_train"].iloc[0]),
             f"{lat.loc[(t, m), 'latency_ms_median_batch1']:.3f}", f"{lat.loc[(t, m), 'model_size_mb']:.2f}"]
            for t in cfg["tasks"] for m in MODELS]
    md.append(table(["Задача", "Модель", "Train", "Задержка, мс/сообщение", "Размер, МБ"], rows))
    md.append("")

    # 5. Observations
    md.append("## 5. Наблюдения\n")
    obs = []
    for t, task in cfg["tasks"].items():
        mt = means.loc[t]
        b = mt["macro_f1"].idxmax()
        obs.append(f"**{t}**: лучшая модель по среднему macro F1 — {MODEL_NAMES[b]} "
                   f"({mt.loc[b, 'macro_f1']:.4f} ± {stds.loc[(t, b), 'macro_f1']:.4f}).")
        low = []
        for c in task["classes"]:
            col = f"recall_{c}"
            if mt[col].min() < LOW_RECALL:
                vals = ", ".join(f"{MODEL_NAMES[m]} {mt.loc[m, col]:.3f}" for m in MODELS)
                low.append(f"`{c}` ({vals})")
        if low:
            obs.append(f"{t}: классы, где средний по seed recall хотя бы одной модели ниже {LOW_RECALL:.2f} "
                       "(в скобках recall всех моделей): " + "; ".join(low) + ".")
        if task["positive"] is not None:
            worst_r = mt["recall_at_fpr_1pct"].idxmin()
            obs.append(f"{t}: recall при FPR = 1% от {mt['recall_at_fpr_1pct'].min():.4f} "
                       f"({MODEL_NAMES[worst_r]}) до {mt['recall_at_fpr_1pct'].max():.4f} "
                       f"({MODEL_NAMES[mt['recall_at_fpr_1pct'].idxmax()]}).")
        gaps = [f"{MODEL_NAMES[m]} {best[(t, m)]['val_macro_f1']:.4f} → {mt.loc[m, 'macro_f1']:.4f}" for m in MODELS]
        obs.append(f"{t}: macro F1 на val (лучшая попытка) → test (среднее): " + "; ".join(gaps) + ".")
        zero = [MODEL_NAMES[m] for m in MODELS if stds.loc[(t, m), "macro_f1"] == 0]
        if zero:
            obs.append(f"{t}: стандартное отклонение macro F1 по seed равно 0 у: {', '.join(zero)}.")
        worst_ece = mt["ece"].idxmax()
        obs.append(f"{t}: наибольший ECE у {MODEL_NAMES[worst_ece]} ({mt.loc[worst_ece, 'ece']:.4f}), "
                   f"наименьший у {MODEL_NAMES[mt['ece'].idxmin()]} ({mt['ece'].min():.4f}).")
        st = sig[sig["task"] == t]
        obs.append(f"{t}: после поправки Холма значимых различий с лучшей моделью: бутстреп "
                   f"{int(st['significant_bootstrap_holm'].sum())} из {len(st)}, Макнемар "
                   f"{int(st['significant_mcnemar_holm'].sum())} из {len(st)}.")
    md += [f"- {o}" for o in obs]
    md.append("")

    md.append("## 6. Проблемы и ограничения\n")
    notes = [
        "LinearSVC не выдаёт вероятностей, поэтому он обёрнут в `CalibratedClassifierCV(method='sigmoid', cv=5, "
        "ensemble=False)`: сигмоида обучается на out-of-fold решениях 5 фолдов train, итоговая модель — одна SVM на "
        "всём train. Предсказание — argmax калиброванных вероятностей; PR-AUC и ECE относятся к калиброванной "
        "модели.",
        "Все модели шага 2 обучены на Kaggle CPU с одним и тем же кодом и версиями: LinearSVM — ноутбук "
        "`smishfuse-svm-cpu`, LogReg, RandomForest и XGBoost — `smishfuse-classic-cpu` (шаг 4в, решение 14 в README). "
        "Прежние локальные результаты этих трёх моделей и сравнение с Kaggle: "
        "`results/step4c/step2_local_vs_kaggle.csv`.",
        "Задержка и размер (раздел 4) измерены в шаге 3 на моделях, заново обученных с лучшими параметрами "
        "локального подбора. Для LogReg, RandomForest и XGBoost лучшие параметры после пересчёта на Kaggle CPU "
        "изменились (кроме RandomForest на moz_binary, LogReg на mendeley_binary и XGBoost на super_sms_clean_binary), поэтому их задержка и размер "
        "относятся к прежним параметрам и не пересчитывались.",
        "LogReg (lbfgs) не содержит случайности: seed влияет только на порядок train, поэтому разброс по seed "
        "у неё нулевой или почти нулевой. Это не 5 независимых моделей.",
        "Пространство признаков помимо двух заданных вариантов (слова 1–2, символы 3–5) включает их "
        "объединение `word+char` как третий вариант (см. `configs/step2.yaml`).",
        f"{proto['n_trials']} попыток TPE — грубый поиск: первые 10 попыток TPE в Optuna по умолчанию случайные. "
        "Разница между моделями частично может объясняться бюджетом подбора.",
        "Значимость проверяется только на предсказаниях одного seed (seed 1), как требует протокол. Разброс "
        "между seed в тесты не входит.",
        "Размеры test по классам: " + "; ".join(f"{t}: {c}" for t, c in test_counts.items()) + ". На Mendeley и "
        "MOZ редкие классы насчитывают около сотни сообщений, поэтому доверительные интервалы там широкие, а "
        "разница в 1–3 ошибки заметно двигает метрики.",
        "RandomForest обучается в 8 потоков, а предсказывает в одном: параллельное суммирование вероятностей "
        "деревьев в sklearn меняет последние биты результата от запуска к запуску.",
        "Задержка измерена в одном потоке на Kaggle CPU (единый замер шага 3); при другой нагрузке или железе "
        "числа будут другими. У moz_binary в test 505 сообщений, поэтому для 1000 измерений сообщения повторяются.",
        "XGBoost на macOS требует OpenMP-рантайм (`brew install libomp`).",
    ]
    md += [f"- {n}" for n in notes]
    md.append("")

    out = ROOT / "reports" / "step2_baselines_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
