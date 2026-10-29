"""Builds reports/step3_neural_report.md from results/ (all numbers come from files).

Run after scripts/collect_results.py, make_tables.py and significance.py.
Missing inputs (e.g. latency not measured yet) are reported as missing.
Run: python scripts/make_step3_report.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.registry import SHORT_NAMES, STEP2_MODELS, STEP3_MODELS, ordered  # noqa: E402

RESULTS = ROOT / "results"
LOW_RECALL = 0.90


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def fmt_params(d: dict) -> str:
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(d.items()))


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def gpu_hours() -> list[list]:
    rows = []
    for f in sorted((RESULTS / "kaggle_runs").glob("*/results/runtime/*.json")):
        kernel = f.stem
        if f.parent.parent.parent.name != kernel:
            continue  # runtime files restored from another kernel's state
        if kernel.startswith("smishfuse-step4"):
            continue  # later steps report their own sessions
        for i, s in enumerate(read_json(f)["sessions"]):
            gpus = s["environment"].get("gpus", "")
            rows.append([kernel, i + 1, f"{s['wall_seconds'] / 3600:.2f}",
                         "GPU" if gpus and "unavailable" not in gpus and gpus.strip() else "CPU",
                         ", ".join(str(r) for r in s["return_codes"])])
    return rows


def short_source(src: str) -> str:
    """'/kaggle/input/.../<kernel>/results/models/...' -> 'weights from <kernel> (seed 1)'."""
    m = re.search(r"/([^/]+)/results/models/", str(src))
    if m:
        return f"веса seed 1 из `{m.group(1)}`"
    return {"re-fitted, seed 1": "заново обучен в этой сессии (seed 1, лучшие параметры)",
            "re-trained in this session (seed 1, best params)":
                "переобучен на GPU в этой сессии (seed 1, лучшие параметры)"}.get(src, str(src))


def local_ratio() -> str:
    """Range of (unified Kaggle latency) / (old local step-2 latency) over the locally measured models."""
    lat_path = RESULTS / "latency" / "latency.csv"
    if not lat_path.exists():
        return "? раз"
    eff = pd.read_csv(RESULTS / "efficiency.csv")
    loc = eff[eff["machine"] == "local_mac_m"].groupby(["task", "model"])["latency_ms_median_batch1"].median()
    lat = pd.read_csv(lat_path).set_index(["task", "model"])["latency_ms_median_batch1"]
    r = lat.loc[loc.index] / loc
    return f"{r.min():.1f}–{r.max():.1f} раза"


def main():
    step2 = yaml.safe_load((ROOT / "configs" / "step2.yaml").read_text(encoding="utf-8"))
    step3 = yaml.safe_load((ROOT / "configs" / "step3.yaml").read_text(encoding="utf-8"))
    proto = step3["protocol"]
    tasks = step2["tasks"]
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    means = metrics.groupby(["task", "model"]).mean(numeric_only=True)
    stds = metrics.groupby(["task", "model"]).std(ddof=1, numeric_only=True)
    counts = metrics.groupby(["task", "model"]).size()

    md = ["# SmishFuse, шаг 3: нейросетевые baseline\n"]
    md.append("Сгенерировано скриптом `scripts/make_step3_report.py` из `results/`. Все нейросетевые вычисления "
              "выполнены на Kaggle (2× Tesla T4 на сессию), локально — только код, отправка и сбор результатов. "
              "Метрики всех моделей пересчитаны одним кодом (`src/eval/metrics.py`) из сырых предсказаний "
              "`results/raw/`.\n")
    md.append("## Протокол\n")
    md.append(
        f"- Модели: TextCNN и BiLSTM (словные эмбеддинги с нуля), DistilBERT и BERT "
        f"(`distilbert-base-uncased`, `bert-base-uncased`; для moz_binary — `distilbert-base-multilingual-cased`, "
        f"`bert-base-multilingual-cased`), Qwen2.5-7B-Instruct в 4-bit, 8 примеров из train без дообучения.\n"
        f"- Подбор: Optuna TPE (seed {proto['tuning_seed']}), {proto['n_trials']} попыток на пару модель-задача, "
        "критерий macro F1 на val, early stopping по macro F1 на val. Пространства поиска: `configs/step3.yaml`.\n"
        f"- Финальная оценка: seed {proto['eval_seeds'][0]}..{proto['eval_seeds'][-1]} с лучшими параметрами, "
        "test только для финальной оценки. fp16 (autocast + GradScaler), "
        "`torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, "
        "фиксированные seed; побайтная воспроизводимость не требуется.\n"
        "- LLM: промпт фиксирован (`configs/llm_prompt.yaml`), подбора нет; 5 seed = 5 разных наборов из 8 "
        "примеров, сбалансированных по классам. Ответ вне допустимых слов — ошибка (`invalid`). Вероятности для "
        "PR-AUC и ECE — softmax по логитам первого токена допустимых слов.\n"
        "- Устойчивость к обрывам: каждая попытка и каждый seed сохраняются сразу; воркер не начинает новую "
        "работу после 8.5 ч сессии; повторный запуск продолжает с датасета `smishfuse-state`.\n")

    md.append("## 1. Итоговые таблицы (test, среднее ± std по seed; все модели шагов 2 и 3)\n")
    md.append("Жирным — лучшее, подчёркнуто — второе (для ECE лучше меньшее). LaTeX: `results/tables/<task>.tex`.\n")
    for t in tasks:
        md.append(f"### {t}\n")
        md.append((RESULTS / "tables" / f"{t}.md").read_text(encoding="utf-8"))
    incomplete = [f"{t}/{m}: {counts[(t, m)]} seed" for (t, m) in counts.index
                  if counts[(t, m)] != len(proto["eval_seeds"])]
    if incomplete:
        md.append("Неполные пары (не 5 seed): " + "; ".join(incomplete) + ".\n")
    missing = [f"{t}/{m}" for t in tasks for m in STEP3_MODELS if (t, m) not in counts.index]
    if missing:
        md.append("Нет результатов: " + ", ".join(missing) + ".\n")

    md.append("## 2. Лучшие гиперпараметры нейросетей (подбор на val)\n")
    rows = []
    for t in tasks:
        for m in STEP3_MODELS:
            p = RESULTS / "best_params" / t / f"{m}.json"
            if m == "qwen_fewshot" or not p.exists():
                continue
            b = read_json(p)
            rows.append([t, SHORT_NAMES[m], b.get("checkpoint") or "—", fmt_params(b["model_params"]),
                         f"{b['val_macro_f1']:.4f}", b["trial"], b["n_trials"],
                         f"{means.loc[(t, m), 'macro_f1']:.4f}" if (t, m) in means.index else "—"])
    md.append(table(["Задача", "Модель", "Checkpoint", "Параметры", "Macro F1 val", "Лучшая попытка", "Попыток",
                     "Macro F1 test"], rows))
    md.append("\nВсе попытки: `results/tuning/<task>/<model>/trial_*.json`. Параметры шага 2 — в отчёте шага 2.\n")

    md.append("## 3. Qwen2.5-7B 8-shot: невалидные ответы и KV-кэш\n")
    rows = []
    for t in tasks:
        sub = metrics[(metrics.task == t) & (metrics.model == "qwen_fewshot")]
        if len(sub):
            rows.append([t, len(sub), f"{sub['invalid_rate'].mean():.4%}", f"{sub['invalid_rate'].max():.4%}",
                         f"{sub['macro_f1'].mean():.4f} ± {sub['macro_f1'].std(ddof=1):.4f}"])
    md.append(table(["Задача", "Seed", "Доля невалидных (среднее)", "Максимум по seed", "Macro F1"], rows)
              if rows else "Результатов LLM нет.")
    md.append("\nПроверка эквивалентности префиксного KV-кэша (system + 8 примеров считаются один раз на seed): "
              "200 случайных сообщений val (seed 0), примеры seed 1, ответы без кэша (батч 8) и с кэшем (батч 32).\n")
    rows = []
    for f in sorted((RESULTS / "llm_cache_check").glob("*.json")):
        c = read_json(f)
        rows.append([c["task"], c["n_messages"], c["identical_raw_answers"], "да" if c["all_identical"] else "НЕТ",
                     f"{c['max_abs_prob_diff']:.2e}", f"{c['seconds_no_cache']:.0f} → {c['seconds_cache']:.0f}"])
    md.append(table(["Задача", "Сообщений", "Совпавших ответов", "100%", "max |Δp|", "Время, с"], rows)
              if rows else "Проверка не найдена.")
    used = step3["models"]["qwen_fewshot"]["fixed"]["prefix_cache"]
    md.append(f"\nВ финальной оценке кэш {'используется' if used else 'не используется'} "
              "(`prefix_cache` в `configs/step3.yaml`). Δp — расхождение вероятностей первого токена из-за fp16; "
              "на ответы оно не повлияло.\n")

    md.append("## 4. Значимость (seed 1, лучшая модель задачи против всех остальных, Холм внутри задачи)\n")
    sig = pd.read_csv(RESULTS / "significance.csv")
    rows = []
    for r in sig.itertuples():
        rows.append([r.task, SHORT_NAMES[r.best_model], SHORT_NAMES[r.other_model],
                     f"{r.diff_macro_f1:+.4f} [{r.ci95_low:+.4f}; {r.ci95_high:+.4f}]",
                     f"{fmt_p(r.p_bootstrap)} / {fmt_p(r.p_bootstrap_holm)}",
                     "да" if r.significant_bootstrap_holm else "нет",
                     f"{r.mcnemar_b_best_only_correct} / {r.mcnemar_c_other_only_correct}", r.mcnemar_method,
                     f"{fmt_p(r.p_mcnemar)} / {fmt_p(r.p_mcnemar_holm)}",
                     "да" if r.significant_mcnemar_holm else "нет"])
    md.append(table(["Задача", "Лучшая", "Против", "Δ macro F1 [95% ДИ]", "p бутстреп / Холм", "Значимо",
                     "Макнемар b / c", "Метод", "p Макнемар / Холм", "Значимо"], rows))
    md.append("")

    md.append("## 5. Время Kaggle по ноутбукам\n")
    md.append("Длительность сессии по `results/kaggle_runs/<kernel>/results/runtime/`. По наблюдению после "
              "smoke-теста квота GPU списывается по времени сессии (2× T4 = одна сессия).\n")
    gh = gpu_hours()
    total_gpu = sum(float(r[2]) for r in gh if r[3] == "GPU")
    gh.append(["**всего GPU**", "", f"**{total_gpu:.2f}**", "", ""])
    md.append(table(["Ноутбук", "Сессия", "Часы", "Тип", "Коды возврата воркеров"], gh))
    md.append("")

    md.append("## 6. Задержка и размер моделей (единый замер)\n")
    lat_path = RESULTS / "latency" / "latency.csv"
    if lat_path.exists():
        lat = pd.read_csv(lat_path)
        env = read_json(RESULTS / "latency" / "environment.json")
        md.append(f"Одна сессия Kaggle. CPU-модели: {env['cpu_model']}, 1 поток, батч 1, медиана по "
                  f"{env['n_messages']} сообщениям test (по кругу, если их меньше), {env['warmup']} прогревочных "
                  "вызовов, время включает предобработку. Qwen — на GPU, помечено отдельно.\n")
        rows = []
        for r in lat.itertuples():
            rows.append([r.task, SHORT_NAMES.get(r.model, r.model), r.device,
                         f"{r.latency_ms_median_batch1:.3f}" if pd.notna(getattr(r, "latency_ms_median_batch1", None)) else "—",
                         f"{r.model_size_mb:.1f}" if pd.notna(getattr(r, "model_size_mb", None)) else "—",
                         short_source(r.source)])
        md.append(table(["Задача", "Модель", "Устройство", "Задержка, мс/сообщение", "Размер, МБ", "Источник"], rows))
    else:
        md.append("Ещё не измерено (ноутбук задержки запускается после всех GPU-ноутбуков).")
    md.append("")

    md.append("## 7. Наблюдения\n")
    obs = []
    for t, task in tasks.items():
        mt = means.loc[t]
        b = mt["macro_f1"].idxmax()
        nb = [m for m in ordered(mt.index) if m in STEP3_MODELS]
        cb = [m for m in ordered(mt.index) if m in STEP2_MODELS]
        line = f"**{t}**: лучшая модель — {SHORT_NAMES[b]} ({mt.loc[b, 'macro_f1']:.4f})"
        if nb and cb:
            bn = max(nb, key=lambda m: mt.loc[m, "macro_f1"])
            bc = max(cb, key=lambda m: mt.loc[m, "macro_f1"])
            line += (f"; лучшая нейросеть {SHORT_NAMES[bn]} {mt.loc[bn, 'macro_f1']:.4f}, лучший классический "
                     f"{SHORT_NAMES[bc]} {mt.loc[bc, 'macro_f1']:.4f}")
        obs.append(line + ".")
        for c in task["classes"]:
            col = f"recall_{c}"
            low = [m for m in ordered(mt.index) if mt.loc[m, col] < LOW_RECALL]
            if low:
                obs.append(f"{t}: recall класса `{c}` ниже {LOW_RECALL:.2f} у " +
                           ", ".join(f"{SHORT_NAMES[m]} {mt.loc[m, col]:.3f}" for m in low) + ".")
    md += [f"- {o}" for o in obs]
    md.append("")

    md.append("## 8. Проблемы\n")
    probs = []
    for f in sorted((RESULTS / "errors").glob("*.txt")) if (RESULTS / "errors").exists() else []:
        t, m = f.stem.split("__")
        n = int(counts.get((t, m), 0))
        state = (f"исправлено: после правки все {n} seed получены в продолжении" if n == len(proto["eval_seeds"])
                 else f"НЕ исправлено: {n} seed")
        probs.append(f"Ошибка в `{f.stem}` (первая сессия): `{f.read_text(encoding='utf-8').strip().splitlines()[-1][:160]}`"
                     f" — {state} (`results/errors/{f.name}`).")
    deferred = []
    for f in sorted((RESULTS / "job_times").glob("*.json")) if (RESULTS / "job_times").exists() else []:
        s = read_json(f)
        n_def = sum(x["status"] == "deferred" for x in s["sessions"] if x["mode"] == "full")
        if n_def:
            deferred.append(f"{f.stem} ({n_def})")
    if deferred:
        probs.append("Работы, остановленные лимитом сессии и продолженные в следующей: " + ", ".join(deferred) + ".")
    probs += [
        "Параллельная работа двух GPU в одной сессии замедляет друг друга: 4 ядра CPU общие (токенизация, "
        "коллация, загрузка моделей). В smoke-тесте попытка BERT шла 151–181 с при работающем на втором GPU LLM "
        "против 61 с без него.",
        "Ошибки нехватки памяти Qwen (mendeley_3class, moz_binary) вызывал расчёт префиксного KV-кэша: прямой "
        "проход считал логиты словаря для всех позиций префикса во всём батче (4–6 ГБ). Исправлено "
        "`logits_to_keep=1` (кэш ключей и значений от этого не меняется); проверка эквивалентности для этих "
        "двух задач повторена на исправленном коде (раздел 3).",
        "Ноутбук `smishfuse-full-llm-bert` завершился со статусом ERROR (из-за этих ошибок), поэтому Kaggle не "
        "принимает его как источник для замера задержки: веса BERT для mendeley_3class, mendeley_binary и "
        "moz_binary переобучены в сессии замера (seed 1, лучшие параметры), см. раздел 6. На задержку и размер "
        "это не влияет (та же архитектура и чекпойнт), но эти веса не побайтно те, что дали метрики seed 1.",
        "Первая версия ноутбука `smishfuse-latency` (с этим ноутбуком в источниках) была заменена второй; отменить "
        "запуск через CLI нельзя, поэтому неизвестно, успела ли первая версия потратить квоту. Её вывод перезаписан, "
        "в таблице раздела 5 только сессия второй версии.",
        f"Задержка классических моделей на Kaggle (Xeon 2.0 ГГц) в {local_ratio()} выше прежних локальных замеров "
        "(Apple M, LogReg / RandomForest / XGBoost); "
        "в отчётах шагов 2 и 3 теперь используется только единый замер, локальные числа остаются в "
        "`results/efficiency.csv`. Время обучения моделей шага 2 измерено на разных машинах и в итоговые таблицы "
        "не входит (решение 10 в README).",
        "Детерминированный режим включён с `warn_only=True`: операции без детерминированной реализации на GPU "
        "выполняются с предупреждением, поэтому повторный запуск может дать слегка другие метрики.",
        "Вероятности Qwen — по первому токену допустимых слов; они не учитывают невалидные ответы и не равны "
        "вероятности сгенерированного ответа.",
        "Версии библиотек на Kaggle (Python 3.13, torch 2.11, transformers 5.16, scikit-learn 1.6.1) отличаются от "
        "локальных; метрики пересчитаны локально (scikit-learn 1.9.1).",
    ]
    md += [f"- {p}" for p in probs]
    md.append("")

    md.append("## 9. Итог\n")
    n_pairs = len(counts)
    full = sum(int(v) == len(proto["eval_seeds"]) for v in counts.values)
    md.append(
        f"Сделано: {len(STEP3_MODELS)} нейросетевых моделей × {len(tasks)} задачи, подбор {proto['n_trials']} попыток "
        f"на val, {len(proto['eval_seeds'])} seed на test — полных пар модель-задача {full} из {n_pairs} (вместе с "
        f"шагом 2); общие таблицы и значимость по всем моделям; единый замер задержки и размера; проверка "
        f"KV-кэша; все вычисления шага 3 — на Kaggle, GPU {total_gpu:.2f} ч.\n"
    )
    md.append("Требует решения:\n")
    md += [f"- {d}" for d in [
        "Квота первой версии `smishfuse-latency` (раздел 8) — проверить на странице квоты Kaggle.",
    ]]
    md.append("\nРешено после шага 3 (README, решения 9–11):\n")
    md += [f"- {d}" for d in [
        "Qwen2.5-7B 8-shot остаётся с фиксированным промптом без подбора, по протоколу.",
        "Веса BERT для замера задержки на mendeley_3class, mendeley_binary и moz_binary переобучены в сессии "
        "замера (seed 1, те же параметры); локальные копии этих весов удалены.",
        "Время обучения убрано из итоговых таблиц; остаются задержка и размер.",
        "Низкий recall класса `spam` на mendeley_3class разобран в шаге 4а (`reports/step4a_report.md`).",
        "Веса моделей (seed 1) хранятся только в выводе ноутбуков Kaggle, в git их нет.",
    ]]
    md.append("")
    out = ROOT / "reports" / "step3_neural_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
