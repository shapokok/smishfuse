"""Step 5b: scoring of the final test predictions (configs/final.yaml), on the Kaggle CPU notebook.

Input: results/final/preds/ (restored from the three GPU notebooks) and results/raw/ (clean test
predictions of the step-2/3 baselines, 5 seeds). Steps, in this order:
1. cascades and ablations from the saved stage probabilities: mendeley_3class and low data with the
   frozen tau; the 2-class tasks with tau chosen on their full clean val (written to tau.json before
   any test prediction is scored);
2. metrics (src/eval/metrics.py) and the cost per message for every test prediction -> metrics.csv;
3. significance (seed 1): cascade vs the best of our baselines and vs every re-implemented published
   method, paired bootstrap + McNemar, Holm within each condition x test set -> significance.csv;
4. data of the figures: cascade share-vs-F1 curve, PR curves, confusion matrix of the cascade.
Run: python src/final/assemble.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_curve  # noqa: E402

from eval.metrics import compute_metrics  # noqa: E402
from eval.stats import holm, mcnemar, paired_bootstrap_macro_f1  # noqa: E402
from neural.common import RESULTS, load_yaml, write_json  # noqa: E402

DRY = os.environ.get("SMISHFUSE_FINAL_DRY") == "1"  # smoke test of src/final/run.py (val stands in for test)
OUT = RESULTS / ("final_dry" if DRY else "final")
PREDS = OUT / "preds"
FCFG = load_yaml("configs/final.yaml")
SEEDS = FCFG["seeds"]
CLASSES = {"mendeley_3class": ["ham", "smishing", "spam"], "mendeley_binary": ["benign", "malicious"],
           "smishing_vs_spam": ["spam", "smishing"], "moz_binary": ["benign", "malicious"],
           "super_sms_clean_binary": ["benign", "malicious"]}
POSITIVE = {"mendeley_binary": "malicious", "smishing_vs_spam": "smishing", "moz_binary": "malicious",
            "super_sms_clean_binary": "malicious"}
LOW = [f"low_data_frac{int(round(f * 100)):02d}" for f in FCFG["conditions"]["low_data"]["fractions"]]
# condition -> (task, test sets, models trained in this run)
CONDS = {"mendeley_3class": ("mendeley_3class", FCFG["conditions"]["mendeley_3class"]["test_sets"]),
         **{c: ("mendeley_3class", ["clean"]) for c in LOW},
         "setting_mahmud": ("mendeley_binary", FCFG["conditions"]["setting_mahmud"]["test_sets"]),
         "setting_shim": ("smishing_vs_spam", FCFG["conditions"]["setting_shim"]["test_sets"]),
         "transfer": ("moz_binary", FCFG["conditions"]["transfer"]["test_sets"]),
         "super_sms": ("super_sms_clean_binary", FCFG["conditions"]["super_sms_clean_binary"]["test_sets"])}
STEP23 = {"logreg": "logreg", "linear_svm": "linear_svm", "random_forest": "random_forest", "xgboost": "xgboost",
          "textcnn": "textcnn", "bilstm": "bilstm", "distilbert_step3": "distilbert", "bert": "bert",
          "qwen_fewshot": "qwen_fewshot"}  # name here -> results/raw/<task>/<name>
NOT_BASELINE = {"cascade", "cascade_noaug", "average", "smishfuse_v1", "smishfuse_v2", "smishfuse_v2_noaug",
                "mahmud2024_cnn_bigru", "shim2024_bert_base", "shim2024_distilbert_base", "shim2024_roberta_base"}
PUBLISHED = {"setting_mahmud": ["mahmud2024_cnn_bigru"],
             "setting_shim": ["shim2024_bert_base", "shim2024_distilbert_base", "shim2024_roberta_base"]}


def log(msg: str) -> None:
    print(msg, flush=True)


def read(cond: str, model: str, name: str, seed: int) -> pd.DataFrame | None:
    f = PREDS / cond / model / f"{name}_seed{seed}.csv"
    return pd.read_csv(f, dtype={"id": str}) if f.exists() else None


def probs_of(df: pd.DataFrame, classes) -> np.ndarray:
    return df[[f"prob_{c}" for c in classes]].to_numpy()


def write_preds(cond, model, name, seed, ids, y, probs, classes, extra=None) -> None:
    df = pd.DataFrame({"id": ids, "y_true": [classes[i] for i in y], "y_pred": [classes[i] for i in probs.argmax(1)]})
    for i, c in enumerate(classes):
        df[f"prob_{c}"] = probs[:, i]
    for k, v in (extra or {}).items():
        df[k] = v
    path = PREDS / cond / model / f"{name}_seed{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


# ---------------------------------------------------------------- 1. cascades

def cascade_probs(p1, p2, tau):
    routed = p1.max(1) < tau
    return np.where(routed[:, None], p2, p1), routed


def choose_tau(runs) -> float:
    """Criterion of step 5a on the full clean val, one value for all seeds (max mean macro F1, ties ->
    the smallest tau)."""
    cands = np.unique(np.concatenate([[0.0, 1.0 + 1e-9]] + [p1.max(1) for _, p1, _ in runs]))
    best = None
    for tau in cands:
        m = np.mean([f1_score(y, cascade_probs(p1, p2, tau)[0].argmax(1), labels=list(range(p1.shape[1])),
                              average="macro", zero_division=0) for y, p1, p2 in runs])
        if best is None or m > best[1]:
            best = (float(tau), float(m))
    return best[0]


def build_cascades() -> dict:
    taus = {"mendeley_3class": {"cascade": FCFG["method"]["tau"],
                                "cascade_noaug": FCFG["ablations"]["cascade_noaug"]["tau"], "source": "frozen"}}
    pairs = {"cascade": ("svm_char_aug", "distilbert_aug"), "cascade_noaug": ("svm_char", "distilbert")}
    for cond, (task, sets) in CONDS.items():
        cls = CLASSES[task]
        val_names = {"transfer": {"zeroshot": "val_zeroshot", "finetune10": "val_finetune10"}}.get(
            cond, {s: "val" for s in sets})
        for name, (s1, s2) in pairs.items():
            if name == "cascade_noaug" and cond != "mendeley_3class":
                continue  # the no-augmentation ablation is defined on mendeley_3class only
            for s in sets:
                if cond == "mendeley_3class" or cond in LOW:
                    tau = taus["mendeley_3class"][name]
                else:
                    runs = []
                    for k in SEEDS:
                        a, b = read(cond, s1, val_names[s], k), read(cond, s2, val_names[s], k)
                        if a is None or b is None:
                            continue
                        runs.append((a["y_true"].map(cls.index).to_numpy(), probs_of(a, cls), probs_of(b, cls)))
                    if not runs:
                        continue
                    tau = choose_tau(runs)
                    taus.setdefault(cond, {})[f"{name}:{s}"] = {"tau": tau, "chosen_on": val_names[s],
                                                                 "n_seeds": len(runs)}
                for k in SEEDS:
                    a, b = read(cond, s1, s, k), read(cond, s2, s, k)
                    if a is None or b is None:
                        continue
                    b = b.set_index("id").loc[a["id"]].reset_index()
                    pr, routed = cascade_probs(probs_of(a, cls), probs_of(b, cls), tau)
                    y = a["y_true"].map(cls.index).to_numpy()
                    write_preds(cond, name, s, k, a["id"], y, pr, cls, {"routed": routed.astype(int)})
                    if name == "cascade":
                        avg = (probs_of(a, cls) + probs_of(b, cls)) / 2
                        write_preds(cond, "average", s, k, a["id"], y, avg, cls)
    write_json(taus, OUT / "tau.json")
    return taus


# ---------------------------------------------------------------- 2. metrics

def cost(y, pred, task, r) -> float | None:
    """(r * missed malicious + false alarms) / n on the ham-vs-malicious view."""
    if task == "smishing_vs_spam":
        return None
    if task == "mendeley_3class":
        mal_true, mal_pred = y != 0, pred != 0
    else:
        mal_true, mal_pred = y == 1, pred == 1
    return float((r * np.sum(mal_true & ~mal_pred) + np.sum(~mal_true & mal_pred)) / len(y))


def score(df: pd.DataFrame, task: str) -> dict:
    cls = CLASSES[task]
    y = df["y_true"].map(cls.index).to_numpy()
    probs = probs_of(df, cls)
    pred = df["y_pred"].map({c: i for i, c in enumerate(cls)}).fillna(-1).astype(int).to_numpy()  # Qwen invalid -> -1
    m = compute_metrics(y, probs, cls, POSITIVE.get(task), y_pred=pred)
    for r in FCFG["metrics"]["cost_ratios"]:
        m[f"cost_{r}to1"] = cost(y, pred, task, r)
    if "routed" in df:
        m["routed_share"] = float(df["routed"].mean())
    return m


def all_metrics() -> pd.DataFrame:
    rows = []
    for cond, (task, sets) in CONDS.items():
        if not (PREDS / cond).exists():
            continue
        for mdir in sorted(p for p in (PREDS / cond).iterdir() if p.is_dir()):
            for s in sets:
                for k in SEEDS:
                    if cond in ("mendeley_3class", "super_sms") and s == "clean" and mdir.name in STEP23 and not DRY:
                        continue  # clean test of a step-2/3 baseline: its existing predictions (below)
                    df = read(cond, mdir.name, s, k)
                    if df is not None:
                        rows.append({"condition": cond, "task": task, "test_set": s, "model": mdir.name, "seed": k,
                                     "source": "final", **score(df, task)})
        if cond in ("mendeley_3class", "super_sms") and not DRY:  # the dry run has no test predictions
            for name, raw in STEP23.items():
                for k in SEEDS:
                    f = ROOT / "results" / "raw" / task / raw / f"seed{k}.csv"
                    if f.exists():
                        rows.append({"condition": cond, "task": task, "test_set": "clean", "model": name, "seed": k,
                                     "source": "steps 2-3 (results/raw)", **score(pd.read_csv(f, dtype={"id": str}), task)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics.csv", index=False, lineterminator="\n")
    log(f"[assemble] metrics: {len(df)} rows")
    return df


# ---------------------------------------------------------------- 3. significance

def seed1(cond, task, s, model) -> pd.DataFrame | None:
    if cond in ("mendeley_3class", "super_sms") and s == "clean" and model in STEP23 and not DRY:
        f = ROOT / "results" / "raw" / task / STEP23[model] / "seed1.csv"
        return pd.read_csv(f, dtype={"id": str}) if f.exists() else None
    return read(cond, model, s, 1)


def significance(metrics: pd.DataFrame) -> pd.DataFrame:
    sc = FCFG["significance"]
    rows = []
    for (cond, s), g in metrics.groupby(["condition", "test_set"]):
        task = CONDS[cond][0]
        cls = CLASSES[task]
        means = g.groupby("model")["macro_f1"].mean()
        if "cascade" not in means:
            continue
        base = means[[m for m in means.index if m not in NOT_BASELINE]]
        others = ([("best_baseline", base.idxmax())] if len(base) else []) + \
            [("published", m) for m in PUBLISHED.get(cond, []) if m in means]
        a = seed1(cond, task, s, "cascade")
        block = []
        for kind, m in others:
            b = seed1(cond, task, s, m)
            if a is None or b is None:
                continue
            b = b.set_index("id").loc[a["id"]].reset_index()
            y = a["y_true"].map(cls.index).to_numpy()
            pa = a["y_pred"].map(cls.index).to_numpy()
            pb = b["y_pred"].map({c: i for i, c in enumerate(cls)}).fillna(-1).astype(int).to_numpy()
            bs = paired_bootstrap_macro_f1(y, pa, pb, len(cls), sc["bootstrap"]["n"], sc["bootstrap"]["seed"])
            mc = mcnemar(y, pa, pb)
            block.append({"condition": cond, "test_set": s, "comparison": kind, "cascade": "cascade", "other": m,
                          "seed": sc["seed"], "n": len(y), "mean_macro_f1_cascade": float(means["cascade"]),
                          "mean_macro_f1_other": float(means[m]), "diff_macro_f1_seed1": bs["diff"],
                          "ci95_low": bs["ci_low"], "ci95_high": bs["ci_high"], "p_bootstrap": bs["p_value"],
                          "mcnemar_b_cascade_only_correct": mc["b"], "mcnemar_c_other_only_correct": mc["c"],
                          "mcnemar_method": mc["method"], "p_mcnemar": mc["p_value"]})
        for test in ("bootstrap", "mcnemar"):
            adj, rej = holm([r[f"p_{test}"] for r in block], sc["alpha"])
            for r, p, j in zip(block, adj, rej):
                r[f"p_{test}_holm"], r[f"significant_{test}_holm"] = p, bool(j)
        rows += block
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "significance.csv", index=False, lineterminator="\n")
    log(f"[assemble] significance: {len(df)} comparisons")
    return df


# ---------------------------------------------------------------- 4. figure data

def figure_data() -> None:
    cls = CLASSES["mendeley_3class"]
    rows = []
    for s in CONDS["mendeley_3class"][1]:
        for k in SEEDS:
            a, b = read("mendeley_3class", "svm_char_aug", s, k), read("mendeley_3class", "distilbert_aug", s, k)
            if a is None or b is None:
                continue
            b = b.set_index("id").loc[a["id"]].reset_index()
            p1, p2 = probs_of(a, cls), probs_of(b, cls)
            y = a["y_true"].map(cls.index).to_numpy()
            order = np.argsort(p1.max(1), kind="stable")
            for q in [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]:
                n = int(np.ceil(q * len(y)))
                pred = p1.argmax(1).copy()
                pred[order[:n]] = p2.argmax(1)[order[:n]]
                rows.append({"test_set": s, "seed": k, "share": q, "macro_f1":
                             float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))})
    pd.DataFrame(rows).to_csv(OUT / "cascade_curve.csv", index=False, lineterminator="\n")
    pr = []
    for m in ["cascade", "svm_char_aug", "distilbert_aug", "smishfuse_v1", "smishfuse_v2", "bert", "linear_svm"]:
        df = read("mendeley_3class", m, "clean", 1)
        if df is None:
            continue
        y = (df["y_true"] != "ham").astype(int).to_numpy()
        score_ = 1 - df["prob_ham"].to_numpy()
        prec, rec, _ = precision_recall_curve(y, score_)
        pr += [{"model": m, "precision": float(p), "recall": float(r)} for p, r in zip(prec, rec)]
    pd.DataFrame(pr).to_csv(OUT / "pr_curves.csv", index=False, lineterminator="\n")
    df = read("mendeley_3class", "cascade", "clean", 1)
    if df is not None:
        cm = pd.crosstab(df["y_true"], df["y_pred"]).reindex(index=cls, columns=cls, fill_value=0)
        cm.to_csv(OUT / "confusion_cascade_seed1.csv", lineterminator="\n")


def main():
    taus = build_cascades()
    log(f"[assemble] tau: {json.dumps(taus)[:500]}")
    m = all_metrics()
    significance(m)
    figure_data()


if __name__ == "__main__":
    main()
