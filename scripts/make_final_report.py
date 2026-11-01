"""Step 5b: every table and figure of the final evaluation, one command.

Reads results/final/ (metrics.csv, significance.csv, latency.csv, tau.json, cascade_curve.csv,
pr_curves.csv, confusion_cascade_seed1.csv, cascade_end_to_end.json) and results/step5a/published_claims.yaml;
writes reports/step5b_final_report.md and reports/figures/final/*.pdf (vector, fonts >= 8 pt, markers and
line styles for black and white; the cascade has one colour and the thickest line everywhere).
Needs matplotlib (requirements-plots.txt, local only).
Run: python scripts/make_final_report.py
"""
import json
import math
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "results" / "final"
FIG = ROOT / "reports" / "figures" / "final"
FCFG = yaml.safe_load((ROOT / "configs" / "final.yaml").read_text(encoding="utf-8"))
NAMES = {"cascade": "Каскад (итоговый)", "svm_char_aug": "LinearSVM char (аугм.) = только SVM",
         "svm_char": "LinearSVM char без аугм.", "distilbert_aug": "DistilBERT (аугм.) = только DistilBERT",
         "distilbert": "DistilBERT без аугм. (4а)", "cascade_noaug": "Каскад без аугментации",
         "average": "Усреднение SVM + DistilBERT", "smishfuse_v1": "SmishFuse v1", "smishfuse_v2": "SmishFuse v2 (аугм.)",
         "smishfuse_v2_noaug": "SmishFuse v2 без аугм.", "bert": "BERT", "logreg": "TF-IDF + LogReg",
         "linear_svm": "TF-IDF + LinearSVM (шаг 2)", "svm_step2": "TF-IDF + LinearSVM (шаг 2)",
         "random_forest": "TF-IDF + RandomForest", "xgboost": "TF-IDF + XGBoost", "textcnn": "TextCNN",
         "bilstm": "BiLSTM", "distilbert_step3": "DistilBERT (шаг 3)", "qwen_fewshot": "Qwen2.5-7B 8-shot",
         "mahmud2024_cnn_bigru": "Mahmud et al. 2024, CNN-BiGRU", "shim2024_bert_base": "Shim et al. 2024, BERT-base",
         "shim2024_distilbert_base": "Shim et al. 2024, DistilBERT-base",
         "shim2024_roberta_base": "Shim et al. 2024, RoBERTa-base"}
SETS = {"clean": "чистый", "train05": "TRAIN 5%", "train10": "TRAIN 10%", "train20": "TRAIN 20%",
        "heldout05": "HELD-OUT 5%", "heldout10": "HELD-OUT 10%", "heldout20": "HELD-OUT 20%",
        "zeroshot": "zero-shot", "finetune10": "дообучение 10%"}
COND_NAMES = {"mendeley_3class": "mendeley_3class", "setting_mahmud": "Mahmud: ham против spam+smishing",
              "setting_shim": "Shim: smishing против spam", "transfer": "Перенос mendeley_binary → moz_binary",
              "super_sms": "super_sms_clean_binary", **{f"low_data_frac{p}": f"Мало данных {int(p)}%"
                                                         for p in ("01", "05", "10", "25")}}
CASCADE_COLOR = "#b2182b"


def name(m):
    return NAMES.get(m, m)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ms(v: pd.Series, digits=4) -> str:
    v = v.dropna()
    if not len(v):
        return "нет данных"
    s = f"{v.mean():.{digits}f}"
    return s + (f" ± {v.std(ddof=1):.{digits}f}" if len(v) > 1 else "") + ("" if len(v) in (1, 5) else f" (n={len(v)})")


def cell(m, cond, model, s, col="macro_f1", digits=4):
    return ms(m[(m.condition == cond) & (m.model == model) & (m.test_set == s)][col], digits)


def mean(m, cond, model, s, col="macro_f1"):
    v = m[(m.condition == cond) & (m.model == model) & (m.test_set == s)][col].dropna()
    return v.mean() if len(v) else float("nan")


def models_of(m, cond):
    present = set(m[m.condition == cond].model)
    order = list(NAMES)
    return [x for x in order if x in present] + sorted(present - set(order))


def bold_best(rows, m, cond, models, sets):
    """Mark the best mean per column with ** (first column = model name)."""
    for j, s in enumerate(sets, start=1):
        vals = {i: mean(m, cond, mm, s) for i, mm in enumerate(models)}
        vals = {i: v for i, v in vals.items() if not math.isnan(v)}
        if vals:
            i = max(vals, key=vals.get)
            rows[i][j] = f"**{rows[i][j]}**"
    return rows


def grid(md, m, cond, sets, title, extra_rows=None):
    models = models_of(m, cond)
    rows = [[name(x)] + [cell(m, cond, x, s) for s in sets] for x in models]
    rows = bold_best(rows, m, cond, models, sets)
    md.append(f"**{title}** — macro F1, среднее ± std по 5 seed; лучшее в столбце выделено\n")
    md.append(table(["Модель"] + [SETS.get(s, s) for s in sets], rows + (extra_rows or [])))
    md.append("")


# ---------------------------------------------------------------- figures

def figures(m, lat, curve, pr, cm, e2e):
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    FIG.mkdir(parents=True, exist_ok=True)
    others = ["svm_char_aug", "svm_char", "distilbert_aug", "distilbert", "bert", "smishfuse_v1", "smishfuse_v2",
              "linear_svm", "logreg", "textcnn", "bilstm", "distilbert_step3", "random_forest", "xgboost"]
    # 16 distinct colour / marker pairs (no repetition for up to 16 models in one figure)
    greys = ["#2166ac", "#4393c3", "#1b7837", "#5aae61", "#762a83", "#e08214", "#8c510a", "#404040", "#878787",
             "#35978f", "#80cdc1", "#c51b7d", "#de77ae", "#bababa", "#01665e", "#fdb863"]
    markers = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "*", "p", "d", "8", "$\\oplus$", "$\\diamond$"]
    styles = ["--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (4, 1, 1, 1, 1, 1))] * 2

    def style(model, i):
        if model == "cascade":
            return {"color": CASCADE_COLOR, "lw": 3.0, "marker": "o", "ms": 5, "ls": "-", "zorder": 10}
        return {"color": greys[i % len(greys)], "lw": 1.1, "marker": markers[i % len(markers)], "ms": 3.5,
                "ls": styles[i % len(styles)], "zorder": 2}

    # 1. obfuscation robustness
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, kind in zip(axes, ["train", "heldout"]):
        xs, sets = [0, 5, 10, 20], ["clean", f"{kind}05", f"{kind}10", f"{kind}20"]
        for i, mod in enumerate(["cascade"] + [o for o in others if o in set(m[m.condition == "mendeley_3class"].model)]):
            ys = [mean(m, "mendeley_3class", mod, s) for s in sets]
            if all(math.isnan(y) for y in ys):
                continue
            ax.plot(xs, ys, label=name(mod), **style(mod, i))
        ax.set_title("TRAIN-типы" if kind == "train" else "HELD-OUT-типы (не в обучении)")
        ax.set_xlabel("Доля искажённых символов, %")
        ax.set_xticks(xs)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("Macro F1 (test, среднее по 5 seed)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(FIG / "obfuscation_robustness.pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. learning curve
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    fr = [1, 5, 10, 25, 100]
    conds = [f"low_data_frac{p:02d}" for p in fr[:-1]]
    for i, mod in enumerate(["cascade"] + [o for o in others if o in set(m[m.condition == "low_data_frac01"].model)]):
        ys = [mean(m, c, mod, "clean") for c in conds] + [mean(m, "mendeley_3class", mod, "clean")]
        ax.plot(fr, ys, label=name(mod), **style(mod, i))
    ax.set_xscale("log")
    ax.set_xticks(fr)
    ax.set_xticklabels([f"{x}%" for x in fr])
    ax.set_xlabel("Доля train (log)")
    ax.set_ylabel("Macro F1 (test)")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(FIG / "learning_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. quality vs latency (Pareto)
    if lat is not None:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        pts = []
        for i, r in enumerate(lat.itertuples()):
            if not str(r.device).startswith("cpu"):
                continue
            q = mean(m, "mendeley_3class", r.model, "clean")
            if math.isnan(q):
                continue
            pts.append((r.p50_ms, q, r.model))
            st = style(r.model, i)
            # names go to the legend (labels next to the points overlapped in the 40-90 ms cluster)
            ax.scatter(r.p50_ms, q, s=70 if r.model == "cascade" else 28, color=st["color"], marker=st["marker"],
                       zorder=st["zorder"], edgecolor="black", linewidth=0.5, label=name(r.model))
        front, best = [], -1
        for x, y, _ in sorted(pts):
            if y > best:
                front.append((x, y))
                best = y
        if front:
            ax.plot(*zip(*front), color="black", lw=0.8, ls="--", label="Парето-фронт")
        ax.set_xscale("log")
        ax.set_xlabel("Задержка p50, мс на сообщение (CPU, 1 поток, батч 1; log)")
        ax.set_ylabel("Macro F1 (чистый test)")
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.savefig(FIG / "quality_vs_latency.pdf", bbox_inches="tight")
        plt.close(fig)

    # 4. PR curves (malicious = smishing + spam vs ham, seed 1)
    if pr is not None:
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        for i, (mod, g) in enumerate(pr.groupby("model", sort=False)):
            st = style(mod, i)
            ax.plot(g["recall"], g["precision"], color=st["color"], lw=st["lw"], ls=st["ls"], label=name(mod),
                    zorder=st["zorder"])
        ax.set_xlabel("Recall (вредоносные)")
        ax.set_ylabel("Precision")
        ax.set_xlim(0.80, 1.0)   # the curves differ only in this corner
        ax.set_ylim(0.80, 1.005)
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(frameon=False, loc="lower left")
        fig.savefig(FIG / "pr_curves.pdf", bbox_inches="tight")
        plt.close(fig)

    # 5. cascade: share to stage 2 vs macro F1
    if curve is not None:
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        for i, s in enumerate(["clean", "heldout05", "heldout10", "heldout20"]):
            g = curve[curve.test_set == s].groupby("share")["macro_f1"].mean()
            st = {"color": CASCADE_COLOR, "lw": 3.0} if s == "clean" else {"color": greys[i], "lw": 1.2}
            ax.plot(100 * g.index, g.values, marker=markers[i], ms=3.5, ls=["-", "--", ":", "-."][i], label=SETS[s], **st)
        if e2e:
            ax.axvline(100 * e2e["routed_share"], color="black", lw=0.8, ls=(0, (1, 1)),
                       label=f"фактическая доля на test ({100 * e2e['routed_share']:.1f}%)")
        ax.set_xscale("symlog", linthresh=1)
        ticks = [0, 1, 2, 5, 10, 20, 50, 100]
        ax.set_xlim(0, 100)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Доля сообщений, отправленных в DistilBERT, % (шкала symlog)")
        ax.set_ylabel("Macro F1 (test, среднее по 5 seed)")
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(frameon=False)
        fig.savefig(FIG / "cascade_share_vs_f1.pdf", bbox_inches="tight")
        plt.close(fig)

    # 6. confusion matrix of the cascade
    if cm is not None:
        fig, ax = plt.subplots(figsize=(3.0, 2.6))
        ax.imshow(cm.values, cmap="Greys")
        for (i, j), v in pd.DataFrame(cm.values).stack().items():
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > cm.values.max() / 2 else "black")
        ax.set_xticks(range(len(cm.columns)))
        ax.set_xticklabels(cm.columns)
        ax.set_yticks(range(len(cm.index)))
        ax.set_yticklabels(cm.index)
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Истинный класс")
        ax.set_title("Каскад, чистый test, seed 1")
        fig.savefig(FIG / "confusion_cascade.pdf", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------- report

def main():
    m = pd.read_csv(F / "metrics.csv")
    sig = pd.read_csv(F / "significance.csv") if (F / "significance.csv").exists() else None
    lat = pd.read_csv(F / "latency.csv") if (F / "latency.csv").exists() else None
    curve = pd.read_csv(F / "cascade_curve.csv") if (F / "cascade_curve.csv").exists() else None
    pr = pd.read_csv(F / "pr_curves.csv") if (F / "pr_curves.csv").exists() else None
    cm = pd.read_csv(F / "confusion_cascade_seed1.csv", index_col=0) if (F / "confusion_cascade_seed1.csv").exists() else None
    e2e = json.loads((F / "cascade_end_to_end.json").read_text()) if (F / "cascade_end_to_end.json").exists() else None
    tau = json.loads((F / "tau.json").read_text()) if (F / "tau.json").exists() else {}
    claims = yaml.safe_load((ROOT / "results" / "step5a" / "published_claims.yaml").read_text(encoding="utf-8"))
    figures(m, lat, curve, pr, cm, e2e)

    md = ["# SmishFuse, шаг 5б: финальная оценка на test\n",
          "Сгенерировано `scripts/make_final_report.py` из `results/final/`. Единственное обращение к test; модели, "
          f"условия, метрики и сравнения заморожены до него в `configs/final.yaml` (коммиты f209949, 087615d). "
          f"5 seed (1–5); среднее ± std по seed (ddof = 1). tau каскада = {FCFG['method']['tau']} подбирался на val "
          "по seed 1–3 и фиксировался для всех запусков.\n"]
    for s in INTRO:
        md.append(f"- {s}")
    md.append("")

    md.append("## 1. Основное сравнение (mendeley_3class, чистый test)\n")
    models = models_of(m, "mendeley_3class")
    cols = [("macro_f1", "Macro F1"), ("accuracy", "Accuracy"), ("f1_smishing", "F1 smishing"), ("f1_spam", "F1 spam"),
            ("pr_auc", "PR-AUC"), ("ece", "ECE")]
    rows = [[name(x)] + [cell(m, "mendeley_3class", x, "clean", c) for c, _ in cols] for x in models]
    md.append(table(["Модель"] + [t for _, t in cols], rows))
    md.append("\nМодели шагов 2–3 на чистом test — их существующие предсказания (не переобучались); остальные — "
              "финальный прогон.\n")

    md.append("## 2. Постановки опубликованных работ\n")
    for cond, key in [("setting_mahmud", "mahmud2024"), ("setting_shim", "shim2024")]:
        sets = FCFG["conditions"][cond]["test_sets"]
        grid(md, m, cond, sets, COND_NAMES[cond])
        c = claims["on_mendeley"][key]
        rep = c.get("reported_table7_dataset3") or c.get("reported_table1_original_data")
        md.append("Из статьи (их разбиение без дедупликации, положительный класс не указан): " + "; ".join(
            f"{k}: acc {v['accuracy']}, F1 {v['f1']}" for k, v in rep.items()) + ".\n")

    md.append("## 3. Обфускация (mendeley_3class)\n")
    grid(md, m, "mendeley_3class", FCFG["conditions"]["mendeley_3class"]["test_sets"], "Чистый, TRAIN- и HELD-OUT-типы")
    md.append("![Устойчивость к обфускации](figures/final/obfuscation_robustness.pdf)\n")

    md.append("## 4. Мало данных (mendeley_3class, чистый test)\n")
    lows = [f"low_data_frac{p}" for p in ("01", "05", "10", "25")]
    lm = [x for x in NAMES if x in set(m[m.condition == lows[0]].model)]
    rows = [[name(x)] + [cell(m, c, x, "clean") for c in lows] + [cell(m, "mendeley_3class", x, "clean")] for x in lm]
    md.append(table(["Модель", "1%", "5%", "10%", "25%", "100%"], rows))
    md.append("\n![Кривая обучения](figures/final/learning_curve.pdf)\n")

    md.append("## 5. Перенос mendeley_binary → moz_binary\n")
    grid(md, m, "transfer", FCFG["conditions"]["transfer"]["test_sets"], COND_NAMES["transfer"])

    md.append("## 6. super_sms_clean_binary (чистый test)\n")
    grid(md, m, "super_sms", ["clean"], COND_NAMES["super_sms"])

    md.append("## 7. Абляции каскада (mendeley_3class)\n")
    sets = FCFG["conditions"]["mendeley_3class"]["test_sets"]
    ab = ["cascade", "svm_char_aug", "distilbert_aug", "cascade_noaug", "average"]
    rows = [[name(x)] + [cell(m, "mendeley_3class", x, s) for s in sets] for x in ab]
    md.append(table(["Вариант"] + [SETS[s] for s in sets], bold_best(rows, m, "mendeley_3class", ab, sets)))
    md.append("\nДоля сообщений во второй ступени (среднее по seed): " + ", ".join(
        f"{SETS[s]} {100 * mean(m, 'mendeley_3class', 'cascade', s, 'routed_share'):.1f}%" for s in sets) + ".\n")
    md.append("![Доля вызовов DistilBERT против F1](figures/final/cascade_share_vs_f1.pdf)\n")
    if tau:
        md.append("tau: mendeley_3class и мало данных — замороженные значения; 2-классовые задачи — выбраны на полном "
                  "чистом val задачи (`results/final/tau.json`): " + "; ".join(
                      f"{c} {k}: {v['tau']:.4f}" for c, d in tau.items() if isinstance(d, dict)
                      for k, v in d.items() if isinstance(v, dict) and "tau" in v) + ".\n")

    md.append("**tau и фактическая доля сообщений во второй ступени (DistilBERT) по условиям** — среднее по 5 seed\n")
    rows = []
    for cond in ["mendeley_3class", "low_data_frac01", "low_data_frac05", "low_data_frac10", "low_data_frac25",
                 "setting_mahmud", "setting_shim", "transfer", "super_sms"]:
        for s_ in sorted(set(m[(m.condition == cond) & (m.model == "cascade")].test_set),
                         key=lambda x: list(SETS).index(x) if x in SETS else 99):
            if cond in ("mendeley_3class",) or cond.startswith("low_data"):
                t = f"{FCFG['method']['tau']:.4f} (заморожен)"
            else:
                t = f"{tau.get(cond, {}).get(f'cascade:{s_}', {}).get('tau', float('nan')):.4f} (val задачи)"
            rows.append([COND_NAMES[cond], SETS.get(s_, s_), t,
                         f"{100 * mean(m, cond, 'cascade', s_, 'routed_share'):.1f}%"])
    md.append(table(["Условие", "Test", "tau", "Во 2-й ступени"], rows))
    md.append("")

    md.append("## 8. Стоимость ошибок (потери на сообщение, пропуск вредоносного : ложная тревога)\n")
    rows = []
    for cond, s in [("mendeley_3class", "clean"), ("mendeley_3class", "heldout20"), ("setting_mahmud", "clean"),
                    ("setting_mahmud", "heldout20"), ("transfer", "zeroshot"), ("transfer", "finetune10"),
                    ("super_sms", "clean")]:
        g = m[(m.condition == cond) & (m.test_set == s)]
        if g.empty:
            continue
        base = g[~g.model.isin(["cascade", "cascade_noaug", "average", "smishfuse_v1", "smishfuse_v2",
                                "smishfuse_v2_noaug", "mahmud2024_cnn_bigru"])]
        for r in FCFG["metrics"]["cost_ratios"]:
            col = f"cost_{r}to1"
            best = base.groupby("model")[col].mean().idxmin()
            rows.append([COND_NAMES[cond], SETS[s], f"{r}:1", cell(m, cond, "cascade", s, col),
                         f"{name(best)} {cell(m, cond, best, s, col)}"])
    md.append(table(["Условие", "Test", "Цена", "Каскад", "Лучший baseline по стоимости"], rows))
    md.append("\nПотери = (r × пропущенные вредоносные + ложные тревоги) / n; предсказание — argmax, без порога, "
              "учитывающего стоимость. Для smishing против spam не определено.\n")

    md.append("## 9. Значимость (seed 1; бутстреп 10 000 и Макнемар, поправка Холма внутри условия)\n")
    if sig is not None and len(sig):
        rows = [[COND_NAMES.get(r.condition, r.condition), SETS.get(r.test_set, r.test_set),
                 "лучший baseline" if r.comparison == "best_baseline" else "опубликованный", name(r.other),
                 f"{r.mean_macro_f1_cascade:.4f} / {r.mean_macro_f1_other:.4f}",
                 f"{100 * r.diff_macro_f1_seed1:+.2f} [{100 * r.ci95_low:+.2f}; {100 * r.ci95_high:+.2f}]",
                 f"{r.p_bootstrap_holm:.4f}{' *' if r.significant_bootstrap_holm else ''}",
                 f"{r.mcnemar_b_cascade_only_correct}/{r.mcnemar_c_other_only_correct}",
                 f"{r.p_mcnemar_holm:.4f}{' *' if r.significant_mcnemar_holm else ''}"] for r in sig.itertuples()]
        md.append(table(["Условие", "Test", "Сравнение", "С кем", "Macro F1 каскад / другой (5 seed)",
                         "Разница seed 1, п.п. [95% ДИ]", "p бутстреп (Холм)", "b/c", "p Макнемар (Холм)"], rows))
        md.append("\n* — значимо при α = 0.05 после поправки Холма. b — только каскад прав, c — только другая модель.\n")

    md.append("## 9б. Пост-хок анализ после финальной оценки (решение 20 в README)\n")
    md.append("**Не входил в замороженный план (`configs/final.yaml`).** Добавлен после финальной оценки по решению "
              "руководителя. Без обучения, на Kaggle CPU (`smishfuse-final-posthoc`, `src/final/posthoc.py`), из "
              "сохранённых test-предсказаний. Сравнения: символьный LinearSVM без аугментации и с ней против "
              "воспроизведённых опубликованных методов во всех их условиях. Seed 1, бутстреп 10 000 и Макнемар, "
              "поправка Холма внутри условия × test-набора отдельно для каждого теста.\n")
    ph = F / "posthoc_significance.csv"
    if ph.exists():
        d = pd.read_csv(ph)
        rows = [[COND_NAMES.get(r.condition, r.condition), SETS.get(r.test_set, r.test_set), name(r.ours), name(r.other),
                 f"{cell(m, r.condition, r.ours, r.test_set)} / {cell(m, r.condition, r.other, r.test_set)}",
                 f"{100 * r.diff_macro_f1_seed1:+.2f} [{100 * r.ci95_low:+.2f}; {100 * r.ci95_high:+.2f}]",
                 f"{r.p_bootstrap_holm:.4f}{' *' if r.significant_bootstrap_holm else ''}",
                 f"{r.mcnemar_b_ours_only_correct}/{r.mcnemar_c_other_only_correct}",
                 f"{r.p_mcnemar_holm:.4f}{' *' if r.significant_mcnemar_holm else ''}"] for r in d.itertuples()]
        md.append(table(["Условие", "Test", "Наш", "Опубликованный", "Macro F1 (5 seed)", "Разница seed 1, п.п. [95% ДИ]",
                         "p бутстреп (Холм)", "b/c", "p Макнемар (Холм)"], rows))
        md.append("\n* — значимо при α = 0.05 после поправки Холма. b — только наша модель права, c — только "
                  "опубликованная.\n")
        md += POSTHOC + [""]
    else:
        md.append("Нет данных.\n")

    md.append("## 10. Скорость (Kaggle CPU, 1 поток, батч 1)\n")
    if lat is not None:
        rows = [[name(r.model), r.device, f"{r.p50_ms:.3f}" if not pd.isna(r.p50_ms) else "—",
                 f"{r.p95_ms:.3f}" if not pd.isna(r.get("p95_ms", float("nan"))) else "—",
                 f"{r.throughput_msg_per_s:.1f}" if not pd.isna(r.get("throughput_msg_per_s", float("nan"))) else "—",
                 f"{r.size_mb:.1f}", cell(m, r.condition, r.model, "clean"), r.source] for _, r in lat.iterrows()]
        md.append(table(["Модель", "Устройство", "p50, мс", "p95, мс", "Сообщений/с", "Размер, МБ",
                         "Macro F1 чистый test", "Источник"], rows))
        if e2e:
            md.append(f"\nКаскад сквозной: {e2e['n_messages']} сообщений test подряд, во вторую ступень ушло "
                      f"{e2e['routed']} ({100 * e2e['routed_share']:.1f}%); p50 {e2e['p50_ms']:.3f} мс, p95 "
                      f"{e2e['p95_ms']:.3f} мс, среднее {e2e['mean_ms']:.3f} мс, {e2e['throughput_msg_per_s']:.1f} "
                      "сообщений/с.\n")
        md.append("![Качество против задержки](figures/final/quality_vs_latency.pdf)\n")

    md.append("## 11. Графики\n")
    md += [f"- `reports/figures/final/{f}`" for f in ["obfuscation_robustness.pdf", "learning_curve.pdf",
                                                      "quality_vs_latency.pdf", "pr_curves.pdf",
                                                      "cascade_share_vs_f1.pdf", "confusion_cascade.pdf"]]
    md.append("")
    md.append("## 12. Где каскад выигрывает и где нет\n")
    md += [f"- {o}" for o in OBSERVATIONS] + [""]
    md.append("## 13. GPU-часы\n")
    md += GPU + [""]
    md.append("## 14. Ограничения\n")
    md += [f"- {p}" for p in LIMITATIONS] + [""] + SUMMARY
    (ROOT / "reports" / "step5b_final_report.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/step5b_final_report.md and", len(list(FIG.glob("*.pdf"))), "figures")


# Written after the results, checked against results/final/ (2026-10-31).
INTRO = [
    "Каскад: LinearSVM char (аугм.) → DistilBERT (аугм.). DistilBERT вызывается, если максимальная калиброванная "
    "вероятность SVM < tau. В двухклассовых постановках tau выбран на полном чистом val задачи до подсчёта test.",
    "Значимость: только seed 1, парный бутстреп (10 000) и Макнемар, поправка Холма внутри условия × test-набора. "
    "Лучший baseline условия — наибольшее среднее macro F1 на test среди baseline (выбран по test, поэтому "
    "сравнение консервативно для каскада).",
    "Обфускация test — та же функция, что для val (по id сообщения и уровню). HELD-OUT-типы в обучении не "
    "использовались.",
]
OBSERVATIONS = [
    "**Не выигрывает у лучшего baseline.** Лучший baseline на mendeley_3class — символьный LinearSVM без "
    "аугментации: на чистом test 0.9374 против 0.9316 у каскада. Разница на seed 1 −1.85 п.п. незначима "
    "(бутстреп p = 0.088, Макнемар p = 0.18 после Холма). Ни в одном test-наборе mendeley_3class каскад не "
    "лучше лучшего baseline значимо. На TRAIN 5% он хуже по бутстрепу (−2.69 п.п., p = 0.033), по Макнемару "
    "разница незначима.",
    "**Каскад ≈ своя первая ступень.** На mendeley_3class во DistilBERT уходит 2.0–2.8% сообщений. Качество "
    "каскада практически равно LinearSVM char (аугм.): 0.9316 против 0.9310 на чистом test, 0.9326 против "
    "0.9336 на HELD-OUT 20%.",
    "**Аугментация в каскаде на test не помогла.** Абляция «каскад без аугментации» (tau 0.5772, заморожен) "
    "выше итогового каскада в 6 из 7 test-наборов: 0.9328 на чистом, 0.9398 на HELD-OUT 20%.",
    "**Устойчивость к обфускации — да, но за счёт символьного SVM.** На HELD-OUT 20% у каскада 0.9326. У "
    "нейросетей без аугментации там 0.62–0.70 (BERT 0.6998, DistilBERT 0.6462, SmishFuse v1 0.6197), у "
    "LinearSVM шага 2 — 0.9131. Символьный LinearSVM без аугментации держится так же (0.9322).",
    "**Опубликованные методы в их постановках — выигрыш на искажениях.** Против CNN-BiGRU (Mahmud) каскад "
    "значимо лучше на HELD-OUT 5/10/20% по обоим тестам (+4.47 … +13.76 п.п. на seed 1). На чистом test "
    "разница +1.73 п.п. незначима после Холма. Против Shim BERT-base, DistilBERT-base и RoBERTa-base каскад "
    "значимо лучше на HELD-OUT 10% и 20% по обоим тестам. На чистом test он лучше всех трёх по бутстрепу, по "
    "Макнемару — только RoBERTa. На HELD-OUT 5% против DistilBERT-base разница незначима.",
    "**В двухклассовых постановках каскад дорогой.** tau, выбранный на val задачи, высокий: 0.9990 у Mahmud, "
    "0.98–0.9985 в переносе. Во DistilBERT уходит 45–61% сообщений у Mahmud и 68–77% в переносе. Здесь каскад "
    "по качеству близок к DistilBERT (аугм.): у Mahmud на seed 1 их предсказания совпадают. У Shim (tau 0.7721) "
    "уходит 13–15%, у super_sms (tau 0.9691) — 3.6%.",
    "**Мало данных — проигрыш.** Каскад хуже символьного LinearSVM на всех долях по среднему. На 5% и 10% "
    "train разница на seed 1 значима по бутстрепу (−4.37 и −9.38 п.п.). На 1% на seed 1 каскад, наоборот, "
    "лучше по Макнемару (62 против 18 сообщений), но по среднему за 5 seed ниже: 0.6957 против 0.7073. При "
    "малых данных DistilBERT слабый, а во вторую ступень уходит до 18% сообщений.",
    "**Перенос на MOZ.** При дообучении на 10% каскад 0.9705 значимо лучше DistilBERT (аугм.) 0.9645 по "
    "бутстрепу (Макнемар незначим). Лучшее среднее здесь у усреднения (0.9741). В zero-shot каскад совпадает с "
    "DistilBERT (0.6115 против 0.6133; на seed 1 предсказания одинаковы).",
    "**super_sms_clean_binary.** Каскад 0.9934 против 0.9936 у BERT шага 3, разница незначима.",
    "**Стоимость ошибок.** На HELD-OUT 20% mendeley_3class каскад даёт наименьшие потери среди всех моделей "
    "при цене 1:1, 5:1 и 10:1 (например, 5:1: 0.0260 против 0.0303 у лучшего baseline). На чистом test "
    "потери на уровне DistilBERT. В постановке Mahmud на HELD-OUT 20% каскад хуже символьного SVM "
    "(5:1: 0.0301 против 0.0202).",
    "**Скорость — главный выигрыш.** Сквозной замер на всём test (1148 сообщений подряд, 1 поток CPU): p50 "
    "1.60 мс, p95 2.47 мс, среднее 3.37 мс, 297 сообщений/с, во DistilBERT ушло 2.4%. Для сравнения: "
    "DistilBERT p50 41.8 мс (21 сообщение/с), BERT 84.7 мс, символьный LinearSVM 1.46 мс (669 сообщений/с).",
]
GPU = ["| Ноутбук | Часы сессии 2× T4 |", "|---|---|",
       "| smishfuse-final-dry (пробный, val вместо test) | 0.053 |", "| smishfuse-final-3class | 0.682 |",
       "| smishfuse-final-settings | 0.958 |", "| smishfuse-final-supersms | 0.566 |", "| **Всего** | **2.26 из 3** |",
       "",
       "CPU-ноутбуки квоту GPU не тратят: smishfuse-final-dry-cpu (2 версии), smishfuse-final-cpu (2 версии), "
       "smishfuse-final-pack, smishfuse-final-assemble. Источники: `results/kaggle_runs/*/results/runtime/`."]
# Checked against results/final/posthoc_significance.csv (2026-10-31).
POSTHOC = [
    "- **Mahmud (ham против spam+smishing).** Оба символьных LinearSVM значимо лучше CNN-BiGRU на HELD-OUT "
    "5/10/20% по обоим тестам (+4.3 … +14.2 п.п. на seed 1). На чистом test разница +1.6 и +1.4 п.п. "
    "незначима, как и у каскада.",
    "- **Shim (smishing против spam), чистый test.** Значима только разница с RoBERTa-base (оба SVM, оба теста). "
    "С BERT-base и DistilBERT-base разница незначима; каскад там был лучше по бутстрепу.",
    "- **Shim, HELD-OUT 5%.** LinearSVM без аугментации значимо лучше только RoBERTa-base (оба теста), с "
    "аугментацией — RoBERTa только по бутстрепу. Разница с BERT-base и DistilBERT-base незначима.",
    "- **Shim, HELD-OUT 10% и 20%.** Оба SVM значимо лучше всех трёх методов по обоим тестам (+8.9 … +52.4 п.п.).",
    "- Вывод: выигрыш над опубликованными методами на сильных искажениях даёт уже символьный LinearSVM. Каскад "
    "добавляет значимость на чистом test против BERT-base и DistilBERT-base (Shim) по бутстрепу и на "
    "HELD-OUT 5% против BERT-base.",
]
LIMITATIONS = [
    "tau подбирался на val по seed 1–3 и фиксировался для всех запусков, в том числе для seed 4–5.",
    "Высокий tau в двухклассовых постановках оставлен как есть: так решила замороженная процедура (тот же "
    "критерий на полном val задачи, `configs/final.yaml`). В постановке Mahmud tau = 0.9990, в переносе "
    "0.9840 (zero-shot) и 0.9985 (дообучение). Поэтому во DistilBERT уходит 45–61% и 68–77% сообщений: "
    "каскад там близок к DistilBERT по качеству и намного дороже, чем на mendeley_3class (2–3% вызовов). "
    "Скорость измерялась только для mendeley_3class.",
    "Обе ступени каскада остановлены по чистому val, параметры LinearSVM подобраны на val в шаге 2. Так же "
    "устроены все модели, поэтому сравнения между ними честные, но абсолютные числа на val были оптимистичны.",
    "Значимость считается по одному seed, а различия между seed бывают сопоставимы с разницей моделей "
    "(например, мало данных 1%).",
    "В test mendeley_3class всего 99 smishing и 84 spam: одно-два сообщения заметно меняют macro F1.",
    "Модели шагов 2–3 на чистом test взяты из существующих предсказаний (5 seed), для искажённых наборов они "
    "переобучены с теми же параметрами и seed.",
    "Анализ завышения из-за дубликатов (шаг 5а) не разделяет два эффекта (решение 18 в README). Все 40 групп "
    "с ничейным конфликтом меток — пары smishing / spam с одним текстом: в 3 классах и в задаче smishing "
    "против spam они занижают оценку на «грязных» данных, а в бинарной задаче дубликаты её завышают.",
    "Стоимость ошибок посчитана для решения по argmax, без порога, учитывающего цену ошибок.",
    "Qwen2.5-7B на CPU не измерялся: 4-битная модель требует GPU. В таблице скорости — его GPU-замер шага 3.",
    "Shim et al., seed 4–5: выбор batch / lr / эпох сделан той же сеткой в этом прогоне. Seed 1–3 взяты из "
    "шага 5а.",
    "Ход работы: первая версия smishfuse-final-cpu получила старое состояние (0 восстановленных файлов, сборка "
    "упала), перезапущена. Полный вывод CPU-ноутбука скачать не удалось, поэтому он упакован в zip отдельным "
    "CPU-ноутбуком. В первой сборке не учитывались готовые test-предсказания шагов 2–3: путь в "
    "`src/final/assemble.py` был неверным. После исправления сборка повторена на Kaggle CPU, обучение и "
    "предсказания при этом не менялись. Замер скорости — одна сессия smishfuse-final-cpu v2.",
]
SUMMARY = [
    "## Итог\n",
    "Каскад значимо лучше лучшего нашего baseline только в двух условиях и только по одному из двух тестов: "
    "перенос на MOZ с дообучением на 10% (по бутстрепу; Макнемар незначим) и 1% train (по Макнемару на seed 1; "
    "по среднему за 5 seed каскад ниже: 0.6957 против 0.7073). В остальных условиях значимого выигрыша нет; "
    "качество каскада практически совпадает с первой ступенью — символьным LinearSVM. Он значимо лучше воспроизведённых опубликованных методов "
    "(CNN-BiGRU Mahmud et al.; BERT, DistilBERT, RoBERTa Shim et al.) на HELD-OUT-обфускации в их "
    "постановках. Он устойчив к искажениям: 0.93 macro F1 при 20% искажённых символов против 0.62–0.70 у "
    "нейросетей. Скорость — real-time: p50 1.6 мс на CPU при 2.4% вызовов DistilBERT. При малых данных каскад "
    "хуже символьного SVM, а в двухклассовых постановках с высоким tau вызывает DistilBERT в 45–77% случаев.",
    "",
]


if __name__ == "__main__":
    main()
