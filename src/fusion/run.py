"""Step 4a: SmishFuse prototype vs ablations and baselines on mendeley_3class, seed 1, VAL ONLY.

The test split is never loaded here. Every model is trained on train; neural models
use early stopping on val macro F1 (as in step 3), so all val numbers are optimistic
in the same way and are a preliminary comparison only.

Compared (all on the same train / val split, seed 1):
* smishfuse, smishfuse_concat, features_mlp, distilbert_text_only  (src/fusion/model.py, GPU)
* tfidf_features_logreg: TF-IDF + standardized features, LogReg (best step-2 LogReg settings, CPU)
* logreg, linear_svm: step-2 pipelines re-fitted with best parameters and seed 1 (CPU)
* distilbert_step3: seed-1 weights of the step-3 DistilBERT (kernel source), predicted on val
Outputs: results/step4a/val/<name>.json, results/step4a/val_preds/<name>.csv,
results/step4a/val_comparison.csv, results/step4a/features_train_stats.json
Run: python src/fusion/run.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before CUDA initialises

import glob  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "baselines"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402

from fusion.features import FEATURES, Standardizer, extract_all  # noqa: E402
from fusion.model import VARIANTS  # noqa: E402
from neural.common import (RESULTS, load_configs, load_split, load_yaml, read_json, time_left,  # noqa: E402
                           write_json)

OUT = RESULTS / "step4a"


def log(msg: str) -> None:
    print(msg, flush=True)


def scores(y, pred, classes) -> dict:
    labels = list(range(len(classes)))
    rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    out = {"macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
           "accuracy": float(accuracy_score(y, pred))}
    out.update({f"recall_{c}": float(rec[i]) for i, c in enumerate(classes)})
    out.update({f"f1_{c}": float(f1[i]) for i, c in enumerate(classes)})
    return out


def save(name: str, ids, y, probs, classes, info: dict, extra: dict | None = None) -> None:
    pred = probs.argmax(axis=1)
    df = pd.DataFrame({"id": ids, "y_true": [classes[i] for i in y], "y_pred": [classes[i] for i in pred]})
    for i, c in enumerate(classes):
        df[f"prob_{c}"] = probs[:, i]
    for k, v in (extra or {}).items():
        df[k] = v
    (OUT / "val_preds").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "val_preds" / f"{name}.csv", index=False, lineterminator="\n")
    res = {"model": name, "split": "val", "seed": 1, "n_val": len(y), **scores(y, pred, classes), **info}
    write_json(res, OUT / "val" / f"{name}.json")
    log(f"[step4a] {name}: val macro F1 {res['macro_f1']:.4f}, F1 spam {res['f1_spam']:.4f}")


def done(name: str) -> bool:
    return (OUT / "val" / f"{name}.json").exists()


# ---------------------------------------------------------------- CPU comparisons

def classic_models(cfg2, data, F, classes, seed):
    import classic
    from scipy.sparse import csr_matrix, hstack

    _, tr_texts, ytr = data["train"]
    va_ids, va_texts, yva = data["val"]
    tr_texts, va_texts = np.asarray(tr_texts, dtype=object), np.asarray(va_texts, dtype=object)
    perm = np.random.default_rng(seed).permutation(len(ytr))  # same train order as the step-2 final runs
    for m in ["logreg", "linear_svm"]:
        if done(m):
            continue
        best = read_json(ROOT / "results" / "best_params" / "mendeley_3class" / f"{m}.json")
        t0 = time.perf_counter()
        model, bw = classic.build_model(m, best["model_params"], cfg2["models"][m]["fixed"], seed, 1)
        clf = classic.TextClassifier(classic.build_vectorizer(best["vectorizer"], cfg2["vectorizer_fixed"]), model, bw)
        clf.fit(tr_texts[perm], ytr[perm])
        save(m, va_ids, yva, clf.predict_proba(va_texts), classes,
             {"source": "step-2 pipeline re-fitted (best params, seed 1)", "seconds": time.perf_counter() - t0})
    name = "tfidf_features_logreg"
    if not done(name):
        best = read_json(ROOT / "results" / "best_params" / "mendeley_3class" / "logreg.json")
        t0 = time.perf_counter()
        vec = classic.build_vectorizer(best["vectorizer"], cfg2["vectorizer_fixed"])
        Xtr = hstack([vec.fit_transform(tr_texts[perm]), csr_matrix(F["train"][perm])]).tocsr()
        Xva = hstack([vec.transform(va_texts), csr_matrix(F["val"])]).tocsr()
        model, bw = classic.build_model("logreg", best["model_params"], cfg2["models"]["logreg"]["fixed"], seed, 1)
        classic.fit_model(model, bw, Xtr, ytr[perm])
        save(name, va_ids, yva, np.asarray(model.predict_proba(Xva)), classes,
             {"source": "TF-IDF (best step-2 LogReg vectorizer) + standardized features, LogReg (best C)",
              "seconds": time.perf_counter() - t0})


def distilbert_step3(data, classes, device):
    from transformers import AutoModelForSequenceClassification

    from neural.trainers import HFClassifier, TransformerSetup, predict_proba

    name = "distilbert_step3"
    if done(name):
        return
    hits = sorted(glob.glob("/kaggle/input/**/models/mendeley_3class/distilbert/seed1", recursive=True))
    if not hits:
        raise FileNotFoundError("seed-1 weights of the step-3 DistilBERT (mendeley_3class) not attached")
    d = hits[0]
    step3 = load_yaml("configs/step3.yaml")
    setup = TransformerSetup(d, {}, step3["models"]["distilbert"]["fixed"], len(classes))
    net = HFClassifier(AutoModelForSequenceClassification.from_pretrained(d)).to(device)
    va_ids, va_texts, yva = data["val"]
    probs = predict_proba(net, setup, setup.encode(va_texts), 128, device, True)
    # .../results/models/mendeley_3class/distilbert/seed1 -> .../results/seed_runs/... of the same kernel output
    ref = read_json(Path(d).parents[3] / "seed_runs" / "mendeley_3class" / "distilbert" / "seed1.json")
    # check of the reload: the step-3 run recorded val macro F1 per epoch, the best epoch's weights were kept
    save(name, va_ids, yva, probs, classes,
         {"source": d, "val_macro_f1_recorded_in_step3": max(h["val_macro_f1"] for h in ref["history"])})
    del net
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- fusion variants

def train_variant(name, cfg, data, F, classes, device):
    from fusion.train import fit

    mode = VARIANTS[name]
    yva = data["val"][2]
    res = fit(mode, cfg, (data["train"][1], data["train"][2], F["train"]), (data["val"][1], yva, F["val"]),
              len(classes), cfg["seed"], device, cfg["text"]["checkpoint"], classes.index("ham"))
    best = {"epoch": res["best_epoch"], "probs": res["val_probs"], "gate": res["val_gate"]}
    history, bs = res["history"], res["batch_size"]
    model = res["model"]
    info = {"variant": name, "mode": mode, "best_epoch": best["epoch"], "epochs_run": len(history),
            "history": history, "seconds": res["seconds"], "batch_size": bs, "device": str(device)}
    extra = {}
    if mode == "gated":
        g = best["gate"]
        info["gate_mean_by_true_class"] = {c: float(g[yva == i].mean()) for i, c in enumerate(classes)}
        info["gate_std_by_true_class"] = {c: float(g[yva == i].std()) for i, c in enumerate(classes)}
        pred = best["probs"].argmax(axis=1)
        info["gate_mean_by_pred_class"] = {c: float(g[pred == i].mean()) if (pred == i).any() else None
                                           for i, c in enumerate(classes)}
        info["gate_note"] = "g = weight of the feature branch f' (mean over the d dimensions), 1 - g = text branch"
        extra["gate_mean"] = g
    save(name, data["val"][0], yva, best["probs"], classes, info, extra)
    del model, res
    torch.cuda.empty_cache()


def main():
    import yaml

    cfg = load_yaml("configs/step4a.yaml")
    step2, _ = load_configs()
    cfg2 = yaml.safe_load((ROOT / "configs" / "step2.yaml").read_text(encoding="utf-8"))
    task = step2["tasks"][cfg["task"]]
    classes = task["classes"]
    data = {s: load_split(task, s) for s in ["train", "val"]}  # test is never loaded in step 4a
    raw = {s: extract_all(data[s][1]) for s in data}
    std = Standardizer().fit(raw["train"])
    F = {s: std.transform(raw[s]) for s in raw}
    ytr = data["train"][2]
    write_json({"features": FEATURES, "train_mean": dict(zip(FEATURES, std.mean_)),
                "train_std": dict(zip(FEATURES, std.std_)),
                "train_mean_by_class": {c: dict(zip(FEATURES, raw["train"][ytr == i].mean(axis=0)))
                                        for i, c in enumerate(classes)}},
               OUT / "features_train_stats.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[step4a] device {device}, train {len(ytr)}, val {len(data['val'][2])}, features {len(FEATURES)}")
    jobs = [("classic", lambda: classic_models(cfg2, data, F, classes, cfg["seed"])),
            ("distilbert_step3", lambda: distilbert_step3(data, classes, device))]
    jobs += [(v, (lambda v=v: None if done(v) else train_variant(v, cfg, data, F, classes, device)))
             for v in cfg["variants"]]
    failed = 0
    for name, job in jobs:
        if time_left() <= 0:  # session budget (SMISHFUSE_DEADLINE), set by kaggle_entry from budget_hours
            log(f"[step4a] budget reached, {name} not run")
            failed += 1
            continue
        try:
            job()
        except Exception:  # noqa: BLE001 - record and continue with the next comparison
            failed += 1
            (OUT / "errors").mkdir(parents=True, exist_ok=True)
            (OUT / "errors" / f"{name}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"[step4a] {name} FAILED:\n{traceback.format_exc()}")
    rows = [read_json(p) for p in sorted((OUT / "val").glob("*.json"))]
    cols = ["model", "macro_f1", "accuracy"] + [f"recall_{c}" for c in classes] + [f"f1_{c}" for c in classes] + \
           ["best_epoch", "epochs_run", "seconds"]
    pd.DataFrame(rows).reindex(columns=cols).to_csv(OUT / "val_comparison.csv", index=False, lineterminator="\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
