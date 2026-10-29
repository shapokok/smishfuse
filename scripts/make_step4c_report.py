"""Builds reports/step4c_report.md from results/step4c/ (Kaggle notebook smishfuse-step4c) and the
step-2 re-run on Kaggle CPU (results/step4c/step2_local_vs_kaggle.csv).

Criteria (fixed before the results were seen, descriptive, no significance test):
(a) v2 is "not worse than the best baseline on clean val" if mean(v2) >= mean(best) - max(std_v2, std_best);
(b) v2 "beats every baseline on HELD-OUT obfuscation by more than the seed spread" if, at every
    HELD-OUT level and for every baseline, mean(v2) - mean(baseline) > max(std_v2, std_baseline).
std: ddof = 1 over the seeds.
Run: python scripts/make_step4c_report.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.registry import SHORT_NAMES  # noqa: E402

RESULTS = ROOT / "results"
S4 = RESULTS / "step4c"
CFG = yaml.safe_load((ROOT / "configs" / "step4c.yaml").read_text(encoding="utf-8"))
NAMES = {"smishfuse_v2_aug": "SmishFuse v2 (аугм.)", "smishfuse_v2": "SmishFuse v2 без аугм.",
         "distilbert_aug": "DistilBERT (аугм.)", "distilbert": "DistilBERT без аугм.",
         "smishfuse_v1": "SmishFuse v1 (4а)", "bert": "BERT (шаг 3)", "svm_char_aug": "LinearSVM char (аугм.)",
         "svm_char": "LinearSVM char без аугм.", "svm_step2": "LinearSVM шага 2 (word+char)"}
V2 = "smishfuse_v2_aug"
BASELINES = [m for m in CFG["models"] if not m.startswith("smishfuse_v2")]


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def settings():
    lv = [f"{int(round(x * 100)):02d}" for x in CFG["eval_levels"]]
    return ["clean"] + [f"train{x}" for x in lv] + [f"heldout{x}" for x in lv]


def label(s: str) -> str:
    if s == "clean":
        return "чистый"
    kind, lv = ("TRAIN", s[5:]) if s.startswith("train") else ("HELD-OUT", s[7:])
    return f"{kind} {int(lv)}%"


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(S4.glob("*/*_seed*.json")):
        r = read_json(f)
        rows.append({k: r.get(k) for k in ["model", "setting", "seed", "macro_f1", "f1_spam", "gate_mean"]}
                    | {"gate_by_class": r.get("gate_mean_by_true_class")})
    return pd.DataFrame(rows)


def stat(df, model, setting, col="macro_f1"):
    v = df[(df["model"] == model) & (df["setting"] == setting)][col].dropna()
    return (v.mean(), v.std(ddof=1), len(v)) if len(v) else (float("nan"), float("nan"), 0)


def ms(m, s, n) -> str:
    if n == 0:
        return "—"
    return f"{m:.4f} ± {s:.4f}" + ("" if n == len(CFG["seeds"]) else f" (n={n})")


def step2_section(md: list) -> None:
    md.append("## 1. Шаг 2 на Kaggle CPU (LogReg, RandomForest, XGBoost)\n")
    f = S4 / "step2_local_vs_kaggle.csv"
    if not f.exists():
        md.append("Результатов нет.\n")
        return
    d = pd.read_csv(f)
    md.append("Тот же протокол шага 2 (20 попыток Optuna на val, 5 seed на test) на Kaggle CPU "
              "(`smishfuse-classic-cpu`); macro F1 на test, среднее по 5 seed. Таблицы шагов 2–3 и значимость "
              "пересчитаны (`results/tables/`, `results/significance*.csv`).\n")
    md.append(table(["Задача", "Модель", "Локально (прежнее)", "Kaggle CPU", "Разница, п.п.", "Лучшая попытка: локально → Kaggle"],
                    [[r.task, SHORT_NAMES[r.model], f"{r.local:.4f}", f"{r.kaggle:.4f}",
                      f"{100 * (r.kaggle - r.local):+.2f}", f"{r.trial_local} → {r.trial_kaggle}"]
                     for r in d.itertuples()]))
    md.append("")


def main():
    md = ["# SmishFuse, шаг 4в: SmishFuse v2 (только val)\n",
          "Сгенерировано `scripts/make_step4c_report.py`. Предварительная оценка: "
          f"{len(CFG['seeds'])} seed, только val, **без проверки значимости**. Test в шаге 4в не читался. "
          "Гиперпараметры без подбора (`configs/step4c.yaml`).\n"]
    step2_section(md)
    if not S4.exists() or not list(S4.glob("*/*_seed*.json")):
        md.append("Результатов SmishFuse v2 нет.")
        (ROOT / "reports" / "step4c_report.md").write_text("\n".join(md), encoding="utf-8")
        return
    df = load()
    st = settings()

    md.append("## 2. Окружение и обфускация\n")
    for f in sorted((S4 / "environment").glob("*.json")) if (S4 / "environment").exists() else []:
        e = read_json(f)
        md.append(f"- `{f.stem}`: расхождений с `requirements.txt`: {len(e['mismatch'])}.")
    lv = {f.stem: read_json(f) for f in sorted((S4 / "levels").glob("*.json"))} if (S4 / "levels").exists() else {}
    md.append("- Фактическая доля изменённых символов: " +
              ", ".join(f"{label(k)} → {100 * v['realised_share']:.2f}%" for k, v in lv.items()) + ".")
    md.append("- TRAIN-типы: замены 0/o/O, 1/l/I, a/A/@, e/E/3, s/S/$ и пробелы внутри слов; HELD-OUT-типы: "
              "гомоглифы кириллицы, удвоение букв, удаление гласных (в обучении не использовались).\n")

    md.append("## 3. Macro F1 на val (среднее ± std по seed)\n")
    for part, cols in [("чистый и TRAIN-типы", st[:4]), ("HELD-OUT-типы", ["clean"] + st[4:])]:
        md.append(f"**{part}**\n")
        md.append(table(["Модель"] + [label(c) for c in cols],
                        [[NAMES[m]] + [ms(*stat(df, m, c)) for c in cols] for m in CFG["models"]]))
        md.append("")

    md.append("## 4. Ответ на вопрос шага 4в\n")
    md.append("Критерии заданы до получения результатов (описательные): (а) среднее v2 на чистом val не ниже "
              "среднего лучшего baseline больше чем на наибольшее из двух std; (б) на каждом уровне HELD-OUT "
              "разница v2 с каждым baseline больше наибольшего из двух std. Baseline: все модели шага 4в, "
              "кроме вариантов v2; для (а) также ориентиры шага 4б (`results/step4b/baseline/`).\n")
    v_m, v_s, _ = stat(df, V2, "clean")
    cands = {NAMES[m]: stat(df, m, "clean")[:2] for m in BASELINES}
    ref = RESULTS / "step4b" / "baseline"
    for d in sorted(ref.glob("*")) if ref.exists() else []:
        vals = [read_json(f)["macro_f1"] for f in d.glob("clean_seed*.json")]
        if vals:
            s = pd.Series(vals)
            cands[f"{SHORT_NAMES.get(d.name, d.name)} (шаг 4б)"] = (s.mean(), s.std(ddof=1))
    best = max(cands, key=lambda k: cands[k][0])
    bm, bs = cands[best]
    a_ok = v_m >= bm - max(v_s, bs)
    md.append(f"(а) Чистый val: {NAMES[V2]} {v_m:.4f} ± {v_s:.4f}; лучший baseline — {best} {bm:.4f} ± {bs:.4f}; "
              f"разница {100 * (v_m - bm):+.2f} п.п., допуск {100 * max(v_s, bs):.2f} п.п. → "
              f"**{'выполнено' if a_ok else 'не выполнено'}**.\n")
    rows, b_ok = [], True
    for s in st[4:]:
        vm, vs, _ = stat(df, V2, s)
        worst = None
        for m in BASELINES:
            m_, s_, n_ = stat(df, m, s)
            if n_ == 0:
                continue
            gap, tol = vm - m_, max(vs, s_)
            if worst is None or gap - tol < worst[1] - worst[2]:
                worst = (m, gap, tol)
        ok = worst is not None and worst[1] > worst[2]
        b_ok &= ok
        rows.append([label(s), f"{vm:.4f}", NAMES[worst[0]] if worst else "—",
                     f"{100 * worst[1]:+.2f}" if worst else "—", f"{100 * worst[2]:.2f}" if worst else "—",
                     "да" if ok else "нет"])
    md.append("(б) HELD-OUT: для каждого уровня — baseline, ближе всего подошедший к v2 (наименьший запас "
              "сверх разброса):\n")
    md.append(table(["Уровень", "v2", "Ближайший baseline", "v2 − baseline, п.п.", "Разброс, п.п.", "Больше разброса"],
                    rows))
    md.append(f"\n→ (б) **{'выполнено' if b_ok else 'не выполнено'}**.\n")
    md.append(f"**Ответ: {'да' if a_ok and b_ok else 'нет'}** — у SmishFuse v2 "
              + ("одновременно выполнены (а) и (б)." if a_ok and b_ok else
                 "не выполнены одновременно (а) и (б).") + " Предварительно, без проверки значимости.\n")

    md.append("## 5. Вес гейта (вклад символьной ветви)\n")
    md.append("g — вес ветви признаков f' (у v2 это символьная ветвь: SVD char TF-IDF + ручные признаки; у v1 — "
              "только ручные признаки), среднее по d измерениям и по сообщениям val, затем по seed.\n")
    gm = [m for m in ["smishfuse_v2_aug", "smishfuse_v2", "smishfuse_v1"] if m in set(df["model"])]
    md.append(table(["Модель"] + [label(c) for c in st],
                    [[NAMES[m]] + [f"{stat(df, m, c, 'gate_mean')[0]:.3f}" for c in st] for m in gm]))
    md.append("")
    for m in gm:
        c0 = stat(df, m, "clean", "gate_mean")[0]
        hi = max(st[1:], key=lambda c: abs(stat(df, m, c, "gate_mean")[0] - c0))
        md.append(f"- {NAMES[m]}: g на чистом val {c0:.3f}; наибольшее изменение — {label(hi)} "
                  f"({stat(df, m, hi, 'gate_mean')[0]:.3f}, {stat(df, m, hi, 'gate_mean')[0] - c0:+.3f}).")
    md.append("")

    md.append("## 6. Наблюдения\n")
    md += [f"- {o}" for o in OBSERVATIONS] + [""]

    md.append("## 7. GPU-часы\n")
    rt = RESULTS / "kaggle_runs" / "smishfuse-step4c" / "results" / "runtime" / "smishfuse-step4c.json"
    if rt.exists():
        sess = read_json(rt)["sessions"]
        md.append(table(["Сессия", "Часы", "Коды возврата"],
                        [[i + 1, f"{s['wall_seconds'] / 3600:.2f}", ", ".join(map(str, s["return_codes"]))]
                         for i, s in enumerate(sess)]))
        md.append(f"\nВсего {sum(s['wall_seconds'] for s in sess) / 3600:.2f} ч сессии 2× T4 (бюджет 3). "
                  "Пересчёт шага 2 шёл на Kaggle CPU и GPU-квоту не тратит.\n")
    md.append("## 8. Проблемы и ограничения\n")
    probs = [f"Ошибка `{f.stem}`: `{f.read_text(encoding='utf-8').strip().splitlines()[-1][:200]}`."
             for f in sorted((S4 / "errors").glob("*.txt"))] if (S4 / "errors").exists() else []
    probs += PROBLEMS
    md += [f"- {p}" for p in probs] + [""] + SUMMARY
    (ROOT / "reports" / "step4c_report.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/step4c_report.md")


# Checked against results/step4c/ and results/step4b/ when the report was written (2026-10-30).
OBSERVATIONS = [
    "Воспроизводимость: на чистом val предсказания SmishFuse v1, DistilBERT без аугм., BERT и LinearSVM шага 2 "
    "совпали с шагом 4б для всех трёх seed (574 из 574, разница вероятностей 0). Общий цикл обучения после "
    "добавления аугментации ведёт себя как раньше.",
    "Чистый val: v2 с аугментацией (0.8957) — худшая из нейросетей шага 4в. Ухудшают оба изменения: "
    "символьная ветвь без аугментации 0.9090 против 0.9248 у текстовой модели (−1.58 п.п., разброс 0.94), "
    "аугментация у DistilBERT 0.9110 против 0.9248 (−1.38 п.п., разброс 0.80).",
    "HELD-OUT 20%: v2 с аугментацией 0.8883 — выше всех нейросетей больше разброса (DistilBERT с аугм. "
    "+4.62 п.п., BERT +21.51, v1 +24.32) и LinearSVM шага 2 (+3.51), но вровень с символьным LinearSVM "
    "(+0.59 без аугм., +0.13 с аугм., разброс 1.10). На HELD-OUT 5% среднее v2 ниже всех baseline, на 10% ниже "
    "символьных LinearSVM и LinearSVM шага 2 больше разброса.",
    "Качество v2 с аугментацией почти не зависит от уровня искажений (0.8957 на чистом, 0.8883 на HELD-OUT 20%), "
    "но начинается ниже. Символьный LinearSVM без аугментации так же устойчив (0.9014 → 0.8823) и аугментация "
    "его не меняет больше разброса.",
    "Аугментация TRAIN-типами переносится на HELD-OUT-типы: на HELD-OUT 20% DistilBERT 0.6217 → 0.8421, "
    "v2 0.7313 → 0.8883.",
    "LinearSVM шага 2 на уровне 10% выше, чем на чистом val (0.9239 TRAIN, 0.9216 HELD-OUT против 0.9094), "
    "как и в шаге 4б. В val 49 smishing и 42 spam, несколько сообщений меняют macro F1 на доли пункта.",
    "Самый слабый класс у v2 — spam: F1 spam на чистом val 0.816 (v2 с аугм.), 0.875 у v1, 0.874 у BERT.",
    "Гейт. С аугментацией v2 даёт символьной ветви больший вес (0.611 против 0.466 без аугментации). "
    "При искажениях вес растёт монотонно в каждом seed, на 20% на +0.016…+0.024 для обоих типов. Это меньше "
    "разницы между seed (0.557–0.676 на чистом val). У v2 без аугментации вес меняется не больше чем "
    "на 0.012 и в разные стороны по seed, у v1 немного снижается во всех seed (до −0.017). Переключения на символьную ветвь нет, есть лишь небольшой сдвиг у v2 с "
    "аугментацией. Нельзя сказать, что этот сдвиг и даёт устойчивость.",
]

PROBLEMS = [
    "Ранняя остановка нейросетей — по чистому val, параметры LinearSVM и BERT подобраны на этом же val: числа "
    "на val оптимистичны.",
    "Нейросети обучались не больше 5 эпох (параметры шагов 3 и 4а, без подбора). В 7 из 18 запусков лучшая "
    "эпоха — последняя, в том числе у v1 во всех seed и у v2 с аугментацией в seed 2. Возможно, "
    "с аугментацией 5 эпох мало; это не проверялось.",
    "«DistilBERT без аугм.» здесь — текстовый вариант модели 4а (те же настройки, что у v2). Это не DistilBERT "
    "шага 3 (0.9146 на val в шаге 4б).",
    "Пересчёт шага 2 на Kaggle CPU изменил лучшие параметры в 9 из 12 пар задача–модель. Сравнение — в "
    "разделе 1 и `results/step4c/step2_local_vs_kaggle.csv`. Лучшая классическая модель на mendeley_3class "
    "теперь LinearSVM (0.9259), а не LogReg. LogReg 0.9294 → 0.9181, ниже BERT (0.9195). После поправки "
    "Холма разница лучшей модели с RandomForest и XGBoost на mendeley_3class больше не значима по бутстрепу "
    "(`significance_step2.csv`). Лучшие модели остальных задач не изменились.",
    "Платформа влияет на обучение и при тех же параметрах. XGBoost на super_sms_clean_binary: та же попытка 9 "
    "и те же параметры, но val 0.9900 локально и 0.9891 на Kaggle. LogReg на mendeley_binary: тот же macro "
    "F1, а PR-AUC и ECE отличаются в 4-м знаке.",
    "Шаги 4а и 4б брали параметры LogReg из прежнего (локального) `results/best_params`. Скрипты "
    "`src/fusion/run.py` и `src/fusion/stress.py` читают этот файл, поэтому их повторный запуск возьмёт новые "
    "параметры. `scripts/make_step4a_report.py` тоже берёт оттуда val LogReg и при перегенерации покажет "
    "0.9267 вместо 0.9207. Отчёты 4а и 4б не перегенерировались.",
    "Задержка и размер LogReg, RandomForest и XGBoost (единый замер шага 3) измерены с прежними параметрами и не "
    "пересчитывались.",
    "Аугментация нейросетей новая на каждой эпохе, LinearSVM получает одну аугментированную копию train на seed "
    "(инкрементально его не обучить) — «та же аугментация» в смысле того же распределения искажений.",
    "TF-IDF и SVD символьной ветви обучены на чистом train; при аугментации признаки пересчитываются по "
    "искажённому тексту теми же обученными преобразованиями.",
]
SUMMARY = [
    "## Итог\n",
    "**Ответ на вопрос шага 4в — нет.** SmishFuse v2 (аугментация) на чистом val на 3.36 п.п. ниже лучшего "
    "baseline (SmishFuse v1). На HELD-OUT 20% v2 больше разброса выше всех нейросетей, но не выше символьного "
    "LinearSVM, а на 5% и 10% ниже него. Гейт на символьную ветвь не переключается: вес растёт на 0.02 при "
    "разнице между seed 0.12. Предварительно: 3 seed, только val, без проверки значимости.",
    "",
]


if __name__ == "__main__":
    main()
