"""Figures of the paper with English labels, from the same files as the report figures (no new computation).

Sources: results/final/ (metrics.csv, latency.csv, cascade_curve.csv, cascade_end_to_end.json, pr_curves.csv,
confusion_cascade_seed1.csv) and results/step6b/adapt/ + data/kz_cases/cases.csv (adaptation curve).
MDPI layout: text width 13.8 cm (FULL figures extend into the left margin, 18 cm), lettering 8 pt, saved at
exact size as PDF (pdfLaTeX). Lite and Full have fixed colours and the thickest lines in every figure.
Needs matplotlib (requirements-plots.txt, local only).
Run: python paper_mdpi/tools/make_figures.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from names import SETS_PLAIN, name  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "results" / "final"
OUT = ROOT / "paper_mdpi" / "figures"
CM = 1 / 2.54
TEXT, FULL = 13.8 * CM, 18 * CM
CASCADE_COLOR = "#b2182b"
LITE_COLOR = "#08306b"
COLORS = ["#4393c3", "#92c5de", "#1b7837", "#5aae61", "#762a83", "#e08214", "#8c510a", "#404040", "#878787",
          "#35978f", "#80cdc1", "#c51b7d", "#de77ae", "#bababa", "#01665e", "#fdb863"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "*", "p", "d", "8", "$\\oplus$", "$\\diamond$"]
STYLES = ["--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (4, 1, 1, 1, 1, 1))] * 3
OTHERS = ["svm_char_aug", "distilbert_aug", "distilbert", "bert", "smishfuse_v1", "smishfuse_v2",
          "linear_svm", "logreg", "textcnn", "bilstm", "distilbert_step3", "random_forest", "xgboost"]


ORDER = OTHERS + ["distilbert_step3", "smishfuse_v2_noaug", "qwen_fewshot", "cascade_noaug", "average"]


def style(model: str, i: int | None = None) -> dict:
    """One fixed style per model in every figure (i is ignored for known models)."""
    if model in ORDER:
        i = ORDER.index(model)
    if model == "cascade":
        return {"color": CASCADE_COLOR, "lw": 2.4, "marker": "o", "ms": 4.5, "ls": "-", "zorder": 10}
    if model == "svm_char":
        return {"color": LITE_COLOR, "lw": 2.4, "marker": "s", "ms": 4.5, "ls": "-", "zorder": 9}
    return {"color": COLORS[i % len(COLORS)], "lw": 1.0, "marker": MARKERS[i % len(MARKERS)], "ms": 3.2,
            "ls": STYLES[i % len(STYLES)], "zorder": 2}


def mean(m, cond, model, s, col="macro_f1"):
    v = m[(m.condition == cond) & (m.model == model) & (m.test_set == s)][col].dropna()
    return v.mean() if len(v) else float("nan")


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")


PUB = {"setting_shim": ["shim2024_bert_base", "shim2024_distilbert_base", "shim2024_roberta_base"],
       "setting_mahmud": ["mahmud2024_cnn_bigru"]}
PUB_STYLE = {"shim2024_bert_base": ("#e08214", "^", "--"), "shim2024_distilbert_base": ("#762a83", "v", ":"),
             "shim2024_roberta_base": ("#1b7837", "D", "-."), "mahmud2024_cnn_bigru": ("#8c510a", "P", "--")}


def main() -> None:
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "DejaVu Sans", "legend.handlelength": 2.6})
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(F / "metrics.csv")
    present3 = set(m[m.condition == "mendeley_3class"].model)

    # obfuscation robustness (two columns)
    # published detectors in their own settings under held-out obfuscation (main text)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 7.0 * CM), layout="constrained")
    xs, sets = [0, 5, 10, 20], ["clean", "heldout05", "heldout10", "heldout20"]
    for ax, (cond, title) in zip(axes, [("setting_shim", "Shim et al. setting"),
                                        ("setting_mahmud", "Mahmud et al. setting")]):
        for mod in ["svm_char", "cascade"] + PUB[cond]:
            ys = [mean(m, cond, mod, s) for s in sets]
            sd = [m[(m.condition == cond) & (m.model == mod) & (m.test_set == s)].macro_f1.std(ddof=1) for s in sets]
            if mod in PUB_STYLE:
                c, mk, ls = PUB_STYLE[mod]
                st = {"color": c, "marker": mk, "ls": ls, "lw": 1.2, "ms": 3.6, "zorder": 3}
            else:
                st = {k: v for k, v in style(mod).items()}
            ax.errorbar(xs, ys, yerr=sd, capsize=2, label=name(mod), **st)
        ax.set_title(title)
        ax.set_xlabel("Held-out obfuscation level, %")
        ax.set_xticks(xs)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("Macro F1 (test, mean $\\pm$ sd of 5 seeds)")
    hs, ls_ = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in ls_:
                hs.append(h)
                ls_.append(lab)
    fig.legend(hs, ls_, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "published_robustness")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 7.5 * CM), sharey=True, layout="constrained")
    shown = ["svm_char", "cascade", "svm_char_aug", "distilbert_aug", "distilbert", "bert", "linear_svm"]
    for ax, kind in zip(axes, ["train", "heldout"]):
        xs, sets = [0, 5, 10, 20], ["clean", f"{kind}05", f"{kind}10", f"{kind}20"]
        for i, mod in enumerate([x for x in shown if x in present3]):
            ys = [mean(m, "mendeley_3class", mod, s) for s in sets]
            ax.plot(xs, ys, label=name(mod), **style(mod, i))
        ax.set_title("TR family (used in training)" if kind == "train" else "HO family (held out)")
        ax.set_xlabel("Obfuscation level, %")
        ax.set_xticks(xs)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("Macro F1 (test, mean of 5 seeds)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=2, frameon=False)
    save(fig, "obfuscation_robustness")
    plt.close(fig)

    # learning curve (one column, legend below)
    fig, ax = plt.subplots(figsize=(TEXT, 9.0 * CM), layout="constrained")
    fr = [1, 5, 10, 25, 100]
    conds = [f"low_data_frac{p:02d}" for p in fr[:-1]]
    present_ld = set(m[m.condition == "low_data_frac01"].model)
    for i, mod in enumerate(["svm_char", "cascade"] + [o for o in OTHERS if o in present_ld]):
        ys = [mean(m, c, mod, "clean") for c in conds] + [mean(m, "mendeley_3class", mod, "clean")]
        ax.plot(fr, ys, label=name(mod), **style(mod, i))
    ax.set_xscale("log")
    ax.set_xticks(fr)
    ax.set_xticklabels([f"{x}%" for x in fr])
    ax.minorticks_off()
    ax.set_xlabel("Share of the training data (log scale)")
    ax.set_ylabel("Macro F1 (clean test)")
    ax.grid(alpha=0.3, lw=0.5)
    fig.legend(loc="outside lower center", ncol=2, frameon=False)
    save(fig, "learning_curve")
    plt.close(fig)

    # quality versus latency, Pareto front (two columns)
    lat = pd.read_csv(F / "latency.csv")
    fig, ax = plt.subplots(figsize=(TEXT, 9.0 * CM), layout="constrained")
    pts = []
    for i, r in enumerate(lat.itertuples()):
        if not str(r.device).startswith("cpu"):
            continue
        q = mean(m, "mendeley_3class", r.model, "clean")
        if math.isnan(q):
            continue
        pts.append((r.p50_ms, q))
        st = style(r.model, i)
        ax.scatter(r.p50_ms, q, s=60 if r.model in ("cascade", "svm_char") else 24, color=st["color"], marker=st["marker"],
                   zorder=st["zorder"], edgecolor="black", linewidth=0.5, label=name(r.model))
    front, best = [], -1.0
    for x, y in sorted(pts):
        if y > best:
            front.append((x, y))
            best = y
    ax.plot(*zip(*front), color="black", lw=0.8, ls="--", label="Pareto front")
    ax.set_xscale("log")
    ax.set_xlabel("Median latency p50, ms per message (log scale)")
    ax.set_ylabel("Macro F1 (clean test)")
    ax.grid(alpha=0.3, lw=0.5)
    fig.legend(loc="outside lower center", ncol=3, frameon=False)
    save(fig, "quality_vs_latency")
    plt.close(fig)

    # share routed to stage 2 versus macro F1 (one column)
    curve = pd.read_csv(F / "cascade_curve.csv")
    e2e = json.loads((F / "cascade_end_to_end.json").read_text())
    fig, ax = plt.subplots(figsize=(9.0 * CM, 7.0 * CM), layout="constrained")
    for i, s in enumerate(["clean", "heldout05", "heldout10", "heldout20"]):
        g = curve[curve.test_set == s].groupby("share")["macro_f1"].mean()
        st = {"color": CASCADE_COLOR, "lw": 2.4} if s == "clean" else {"color": COLORS[i], "lw": 1.0}
        ax.plot(100 * g.index, g.values, marker=MARKERS[i], ms=3.2, ls=["-", "--", ":", "-."][i],
                label=SETS_PLAIN[s], **st)
    ax.axvline(100 * e2e["routed_share"], color="black", lw=0.8, ls=(0, (1, 1)), label="at $\\tau^*$")
    ax.set_xscale("symlog", linthresh=1)
    ticks = [0, 1, 2, 5, 10, 20, 50, 100]
    ax.set_xlim(0, 100)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Messages sent to DistilBERT, % (symlog scale)")
    ax.set_ylabel("Macro F1 (test, mean of 5 seeds)")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False, loc="lower left", handlelength=1.8, borderaxespad=0.3)
    save(fig, "cascade_share_vs_f1")
    plt.close(fig)

    # precision-recall curves, malicious versus ham, seed 1 (appendix, one column)
    pr = pd.read_csv(F / "pr_curves.csv")
    fig, ax = plt.subplots(figsize=(9.0 * CM, 7.0 * CM), layout="constrained")
    for i, (mod, g) in enumerate(pr.groupby("model", sort=False)):
        st = style(mod, i)
        ax.plot(g["recall"], g["precision"], color=st["color"], lw=st["lw"], ls=st["ls"], label=name(mod),
                zorder=st["zorder"])
    ax.set_xlabel("Recall (malicious)")
    ax.set_ylabel("Precision (malicious)")
    ax.set_xlim(0.80, 1.0)
    ax.set_ylim(0.80, 1.005)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "pr_curves")
    plt.close(fig)

    # confusion matrix of the cascade, seed 1 (appendix, one column)
    cm = pd.read_csv(F / "confusion_cascade_seed1.csv").set_index("y_true")
    fig, ax = plt.subplots(figsize=(8.0 * CM, 6.0 * CM), layout="constrained")
    ax.imshow(cm.values, cmap="Greys")
    for (i, j), v in pd.DataFrame(cm.values).stack().items():
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.values.max() / 2 else "black")
    ax.set_xticks(range(len(cm.columns)), cm.columns)
    ax.set_yticks(range(len(cm.index)), cm.index)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    save(fig, "confusion_cascade")
    plt.close(fig)

    # Kazakhstan case study: adaptation curve (two columns)
    curve6, base = adaptation_curve()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 6.5 * CM), sharey=True, layout="constrained")
    for ax, key, title in zip(axes, ["acc", "fraud"], ["Correct labels, all original cases",
                                                      "Fraud messages flagged as smishing"]):
        ax.axhline(base[key], color="black", lw=0.8, ls=":", label="before adaptation")
        for cond, st in [("A", {"color": "#2166ac", "marker": "s", "ls": "--", "lw": 1.0}),
                         ("B", {"color": CASCADE_COLOR, "marker": "o", "ls": "-", "lw": 2.0})]:
            c = [x for x in curve6 if x["condition"] == cond]
            ax.errorbar([x["n"] for x in c], [x[f"{key}_mean"] for x in c], yerr=[x[f"{key}_std"] for x in c],
                        capsize=2, ms=4, label="A: categories absent from the cases" if cond == "A"
                        else "B: whole adaptation set", **st)
        ax.set_title(title)
        ax.set_xlabel("Adaptation messages")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("Share")
    axes[1].legend(frameon=False, loc="lower right")
    save(fig, "adaptation_curve")
    plt.close(fig)
    explanation_figure(plt)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))


def explanation_figure(plt, top: int = 10) -> None:
    """Three test messages (results/step8, masked): contributions of n-grams to s_smishing - s_spam (eq. contrib).
    Bars: the TOP labels by absolute value; the rest of the n-grams and the intercept difference as two more bars,
    so that the bars add up to the margin s_smishing - s_spam of the message."""
    import textwrap

    s8 = ROOT / "results" / "step8"
    ex = json.loads((s8 / "examples.json").read_text())
    masked = json.loads((s8 / "examples_masked.json").read_text())
    contrib = pd.read_csv(s8 / "example_contributions.csv", keep_default_na=False)
    model = json.loads((s8 / "model.json").read_text())
    b = dict(zip(model["classes"], model["intercept"]))
    order = [("smishing_correct", "(a)"), ("spam_correct", "(b)"), ("smishing_as_spam", "(c)")]
    fig, axes = plt.subplots(3, 1, figsize=(TEXT, 17.0 * CM), layout="constrained")
    for ax, (key, tag) in zip(axes, order):
        e, g = ex[key], contrib[contrib.example == key].copy()
        g = g.reindex(g.diff_smishing_minus_spam.abs().sort_values(ascending=False).index)
        head, rest = g.head(top), g.iloc[top:]
        labels = [x if x.startswith("[") else "'" + x.replace(" ", "␣") + "'" for x in head.label]
        vals = list(head.diff_smishing_minus_spam)
        labels += [f"other n-grams ({int(rest.n_ngrams.sum())})", "intercept"]
        vals += [rest.diff_smishing_minus_spam.sum(), b["smishing"] - b["spam"]]
        margin = e["score"]["smishing"] - e["score"]["spam"]
        if abs(sum(vals) - margin) > 1e-6:
            raise ValueError(f"{key}: contributions do not add up to the margin")
        y = list(range(len(vals)))[::-1]
        ax.barh(y, vals, color=[CASCADE_COLOR if v > 0 else LITE_COLOR for v in vals], height=0.7)
        ax.set_yticks(y, labels, fontsize=7)
        ax.axvline(0, color="black", lw=0.6)
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        text = "\n".join(textwrap.wrap(masked[key]["text"], 68))
        ax.set_title(f"{tag} true: {e['y_true']}, predicted: {e['y_pred']} "
                     f"(p = {e['prob'][e['y_pred']]:.2f}); margin {margin:+.2f}\n{text}",
                     fontsize=7, loc="left")
    axes[-1].set_xlabel("Contribution to $s_{smishing} - s_{spam}$ (positive: towards smishing)")
    save(fig, "explanation_examples")
    plt.close(fig)


def adaptation_curve():
    """Same computation as scripts/make_step6b_report.py (section 3): original cases, as written."""
    r6b = ROOT / "results" / "step6b" / "adapt"
    runs = pd.read_csv(r6b / "runs.csv")
    p = pd.read_csv(r6b / "predictions.csv", keep_default_na=False)
    p["ok"] = p.pred == p.label
    cases = pd.read_csv(ROOT / "data" / "kz_cases" / "cases.csv", dtype=str, keep_default_na=False)
    pc = p[p.set == "cases"].merge(cases[["id", "pair", "variant"]], on="id")
    orig = pc[pc.variant == "original"]
    curve = []
    for cond in ["A", "B"]:
        for n, g in runs[runs.condition == cond].groupby("size"):
            acc = orig[orig.run_id.isin(g.run_id)].groupby("run_id").ok.mean()
            fr = orig[orig.run_id.isin(g.run_id) & (orig.pair == "fraud")].groupby("run_id").ok.mean()
            curve.append({"condition": cond, "n": n, "acc_mean": acc.mean(), "acc_std": acc.std(ddof=1) if len(acc) > 1 else 0,
                          "fraud_mean": fr.mean(), "fraud_std": fr.std(ddof=1) if len(fr) > 1 else 0})
    b0 = orig[orig.run_id == "before"]
    return curve, {"acc": b0.ok.mean(), "fraud": b0[b0.pair == "fraud"].ok.mean()}


if __name__ == "__main__":
    main()
