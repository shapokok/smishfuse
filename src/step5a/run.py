"""Step 5a runner (configs/step5a.yaml). VAL ONLY; the inflation CV uses train + val. No test text is read.

Jobs (--queue, JSON list, run in order; finished (job, seed) units are skipped on restart):
  cascade                 stage-1 / stage-2 probabilities of step 4c -> cross-fit tau, curve, averaging
  published:<method>      re-implemented published method, seeds 1-3, clean + HELD-OUT val
  inflation:<model>       5-fold CV on dirty (no dedup) and clean train + val, seeds 1-3
  compare                 cross-fit (two val halves) scores of every compared model -> compare/crossfit.csv
  setting:<task>:<model>  addendum: a step-4c baseline / cascade stage trained on mendeley_binary or
                          smishing_vs_spam, seeds 1-3, clean + HELD-OUT val of the setting
  setting_compare         addendum: cascade of the setting (cross-fit tau) and cross-fit scores -> setting/
Results: results/step5a/<job>/... The installed versions are checked against requirements.txt first.

Run: python src/step5a/run.py --queue '["cascade", "published:mahmud2024_cnn_bigru"]' --worker w0
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before CUDA initialises

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from eval.obfuscation import HELDOUT_TYPES, obfuscate_all  # noqa: E402
from fusion.envcheck import check_versions  # noqa: E402
from neural.common import DATA, RESULTS, load_configs, load_split, load_yaml, read_json, time_left, write_json  # noqa: E402
from step5a import methods  # noqa: E402

OUT = RESULTS / "step5a"
CFG = load_yaml("configs/step5a.yaml")
STEP2, _ = load_configs()
CLASSES3 = STEP2["tasks"]["mendeley_3class"]["classes"]  # ham, smishing, spam
# Binary views of mendeley_3class; class index 1 is the positive class of the published binary task.
TASKS = {"mendeley_3class": {"classes": CLASSES3, "positive": None},
         "mendeley_binary": {"classes": ["benign", "malicious"], "positive": "malicious"},
         "smishing_vs_spam": {"classes": ["spam", "smishing"], "positive": "smishing"}}
STEP4C = ROOT / "results" / "step4c"  # step-4c predictions shipped with the code dataset
STEP4B = ROOT / "results" / "step4b" / "baseline"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- data

def to_task(task: str, y3: np.ndarray):
    """Rows kept and the labels of the kept rows, from 3-class labels (ham 0, smishing 1, spam 2)."""
    if task == "mendeley_3class":
        return np.ones(len(y3), dtype=bool), y3
    if task == "mendeley_binary":
        return np.ones(len(y3), dtype=bool), (y3 != 0).astype(np.int64)
    keep = y3 != 0
    return keep, (y3[keep] == 1).astype(np.int64)  # smishing 1, spam 0


def split3(name: str):
    ids, texts, y = load_split(STEP2["tasks"]["mendeley_3class"], name)
    return np.asarray(ids, dtype=object), np.asarray(texts, dtype=object), np.asarray(y)


_eval = {}


def eval_sets():
    """setting -> val texts (3-class order): clean and the HELD-OUT obfuscation of step 4c."""
    if not _eval:
        ids, texts, _ = split3("val")
        for s in CFG["eval_settings"]:
            if s == "clean":
                _eval[s] = list(texts)
                continue
            out, stats = obfuscate_all(list(texts), list(ids), int(s[-2:]) / 100, HELDOUT_TYPES)
            ref = read_json(STEP4C / "levels" / f"{s}.json")
            if stats["sha256"] != ref["sha256"]:
                raise SystemExit(f"{s}: obfuscation differs from step 4c")
            _eval[s] = out
    return _eval


def halves(y3: np.ndarray) -> np.ndarray:
    """Half (0 / 1) of every val message, stratified by the 3-class label."""
    h = np.zeros(len(y3), dtype=np.int64)
    skf = StratifiedKFold(2, shuffle=True, random_state=CFG["cascade"]["halves_seed"])
    for k, (_, idx) in enumerate(skf.split(np.zeros(len(y3)), y3)):
        h[idx] = k
    return h


def metrics(y, pred, n_classes: int) -> dict:
    """Accuracy, macro F1; for a binary task also precision / recall / F1 of class 1 (the positive one)."""
    out = {"accuracy": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0))}
    if n_classes == 2:
        out.update({"precision_pos": float(precision_score(y, pred, zero_division=0)),
                    "recall_pos": float(recall_score(y, pred, zero_division=0)),
                    "f1_pos": float(f1_score(y, pred, zero_division=0))})
    return out


def save_preds(path: Path, ids, y, probs, classes, info: dict) -> None:
    pred = probs.argmax(axis=1)
    df = pd.DataFrame({"id": ids, "y_true": [classes[i] for i in y], "y_pred": [classes[i] for i in pred]})
    for i, c in enumerate(classes):
        df[f"prob_{c}"] = probs[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    write_json({**info, "n": len(y), **metrics(y, pred, len(classes))}, path.with_suffix(".json"))


# ---------------------------------------------------------------- cascade

def load_4c(model: str, setting: str, seed: int, ids) -> np.ndarray:
    df = pd.read_csv(STEP4C / model / f"{setting}_seed{seed}.csv", dtype={"id": str}).set_index("id").loc[list(ids)]
    return df[[f"prob_{c}" for c in CLASSES3]].to_numpy()


def choose_tau(conf: np.ndarray, p1: np.ndarray, p2: np.ndarray, y: np.ndarray) -> float:
    """Max macro F1 on these messages; ties -> the smallest tau (fewest stage-2 calls)."""
    best = None
    for tau in np.concatenate([[0.0], np.unique(conf), [1.0 + 1e-9]]):
        pred = np.where(conf >= tau, p1.argmax(1), p2.argmax(1))
        f1 = f1_score(y, pred, labels=list(range(p1.shape[1])), average="macro", zero_division=0)
        if best is None or f1 > best[1]:
            best = (float(tau), f1)
    return best[0]


def job_cascade(seed: int) -> None:
    cc = CFG["cascade"]
    ids, _, y = split3("val")
    h = halves(y)
    p1 = {s: load_4c(cc["stage1"], s, seed, ids) for s in CFG["eval_settings"]}
    p2 = {s: load_4c(cc["stage2"], s, seed, ids) for s in CFG["eval_settings"]}
    taus = {}
    for k in (0, 1):  # tau for half 1 - k is chosen on the clean val of half k
        sel = h == k
        taus[1 - k] = choose_tau(p1["clean"][sel].max(1), p1["clean"][sel], p2["clean"][sel], y[sel])
    tau_of = np.where(h == 0, taus[0], taus[1])
    info, curve = {"seed": seed, "tau_by_half": {str(k): v for k, v in taus.items()}}, {}
    for s in CFG["eval_settings"]:
        conf = p1[s].max(1)
        to2 = conf < tau_of
        probs = np.where(to2[:, None], p2[s], p1[s])
        avg = (p1[s] + p2[s]) / 2
        df = pd.DataFrame({"id": ids, "half": h, "y_true": [CLASSES3[i] for i in y], "stage1_conf": conf,
                           "routed": to2.astype(int)})
        for name, pr in [("cascade", probs), ("average", avg)]:
            df[f"{name}_pred"] = [CLASSES3[i] for i in pr.argmax(1)]
            for i, c in enumerate(CLASSES3):
                df[f"{name}_prob_{c}"] = pr[:, i]
        path = OUT / "cascade" / "predictions" / f"{s}_seed{seed}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, lineterminator="\n")
        info[s] = {"routed_share": float(to2.mean()),
                   "routed_share_by_half": {str(k): float(to2[h == k].mean()) for k in (0, 1)}}
        # descriptive curve on the full val: the given share of least confident messages -> stage 2
        order = np.argsort(conf, kind="stable")
        pts = []
        for q in cc["curve_shares"]:
            m = int(np.ceil(q * len(y)))
            pred = p1[s].argmax(1).copy()
            pred[order[:m]] = p2[s].argmax(1)[order[:m]]
            pts.append({"share": q, "n_routed": m,
                        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))})
        curve[s] = pts
    write_json(info, OUT / "cascade" / f"crossfit_seed{seed}.json")
    write_json({"seed": seed, "curve": curve}, OUT / "cascade" / f"curve_seed{seed}.json")
    log(f"[cascade] seed {seed}: tau {taus}, routed {[round(info[s]['routed_share'], 3) for s in CFG['eval_settings']]}")


# ---------------------------------------------------------------- published methods

def task_data(task: str, name: str):
    ids, texts, y3 = split3(name)
    keep, y = to_task(task, y3)
    return ids[keep], texts[keep], y, keep


def job_published(method: str, seed: int, device) -> None:
    pc = CFG["published"]
    family = "mahmud2024_cnn_bigru" if method == "mahmud2024_cnn_bigru" else "shim2024"
    task = pc[family]["task"]
    _, tr_texts, ytr, _ = task_data(task, "train")
    va_ids, _, yva, keep = task_data(task, "val")
    info = {"method": method, "task": task, "seed": seed}
    t0 = time.perf_counter()
    if family == "mahmud2024_cnn_bigru":
        model = methods.fit_mahmud(pc[family], tr_texts, ytr, seed, device)
    else:
        sc, ckpt = pc[family], pc[family]["models"][method]
        sel = methods.select_shim(sc, ckpt, tr_texts, ytr, seed, device)
        write_json(sel, OUT / "published" / method / f"selection_seed{seed}.json")
        model = methods.fit_shim(sc, ckpt, tr_texts, ytr, seed, device, sel["chosen"])
        info.update({"chosen": sel["chosen"], "checkpoint": ckpt})
    info["seconds_train"] = time.perf_counter() - t0
    for s, texts in eval_sets().items():
        probs = model.predict_proba(list(np.asarray(texts, dtype=object)[keep]))
        save_preds(OUT / "published" / method / f"{s}_seed{seed}", va_ids, yva, probs, TASKS[task]["classes"],
                   {**info, "setting": s})
    if family == "shim2024":
        model.release()
    log(f"[published] {method} seed {seed} done")


# ---------------------------------------------------------------- duplicate inflation

def inflation_data():
    """(dirty, clean) frames with id, text, y3, group; train + val only."""
    task = STEP2["tasks"]["mendeley_3class"]
    unified = pd.read_csv(DATA / task["dataset"] / "unified.csv", dtype=str, keep_default_na=False)
    test_ids = set(pd.read_csv(DATA / task["dataset"] / "test.csv", dtype=str, keep_default_na=False)["id"])
    dmap = pd.read_csv(ROOT / "results" / "step5a" / "inputs" / "mendeley_dedup_map.csv", dtype=str,
                       keep_default_na=False)
    clean_ids = set(split3("train")[0]) | set(split3("val")[0])
    removed = dmap.set_index("id")
    group = dict(zip(dmap["id"], dmap["group_id"]))
    group.update({k: g for k, g in zip(dmap["kept_id"], dmap["group_id"]) if k})
    keep = []
    for i in unified["id"]:
        if i in clean_ids:
            keep.append(True)
        elif i in removed.index:
            kept = removed.loc[i, "kept_id"]
            keep.append(kept == "" or kept not in test_ids)  # tie groups have no kept member
        else:
            keep.append(False)  # a kept member in test
    dirty = unified[keep].copy()
    assert not (set(dirty["id"]) & test_ids)
    dirty["y3"] = dirty["label"].map({c: i for i, c in enumerate(CLASSES3)}).astype(np.int64)
    dirty["group"] = [group.get(i, i) for i in dirty["id"]]
    clean = dirty[dirty["id"].isin(clean_ids)].copy()
    # clean labels are those of our splits (majority vote); the dirty rows keep their raw labels
    lab = {**dict(zip(*split3("train")[::2])), **dict(zip(*split3("val")[::2]))}
    clean["y3"] = clean["id"].map(lab).astype(np.int64)
    return dirty.reset_index(drop=True), clean.reset_index(drop=True)


def shim_choice(method: str) -> dict:
    sels = [read_json(OUT / "published" / method / f"selection_seed{k}.json")["chosen"] for k in CFG["seeds"]]
    keys = [(c["batch_size"], c["lr"], c["epoch"]) for c in sels]
    (top, n), = collections.Counter(keys).most_common(1)
    bs, lr, ep = top if n > 1 else keys[0]  # no repeated choice -> the seed-1 choice
    return {"batch_size": bs, "lr": lr, "epoch": ep}


def job_inflation(model: str, seed: int, device) -> None:
    pc = CFG["published"]
    if model in ("linear_svm", "logreg"):
        task = "mendeley_3class"
    elif model == "mahmud2024_cnn_bigru":
        task = pc["mahmud2024_cnn_bigru"]["task"]
    else:
        task = pc["shim2024"]["task"]
    choice = shim_choice(model) if model.startswith("shim2024") else None
    dirty, clean = inflation_data()
    for variant, df in [("dirty", dirty), ("clean", clean)]:
        path = OUT / "inflation" / model / f"{variant}_seed{seed}.json"
        if path.exists():
            continue
        keep, y = to_task(task, df["y3"].to_numpy())
        texts, groups = df["text"].to_numpy(dtype=object)[keep], df["group"].to_numpy(dtype=object)[keep]
        folds = []
        skf = StratifiedKFold(CFG["inflation"]["folds"], shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            if model in ("linear_svm", "logreg"):
                clf = methods.fit_classic(model, texts[tr], y[tr], seed)
            elif model == "mahmud2024_cnn_bigru":
                clf = methods.fit_mahmud(pc[model], texts[tr], y[tr], seed, device)
            else:
                clf = methods.fit_shim(pc["shim2024"], pc["shim2024"]["models"][model], texts[tr], y[tr], seed,
                                       device, choice)
            pred = clf.predict_proba(list(texts[te])).argmax(1)
            if hasattr(clf, "release"):
                clf.release()
            tr_groups = set(groups[tr])
            folds.append({"fold": k, "n_test": len(te),
                          "n_test_with_duplicate_in_train": int(sum(g in tr_groups for g in groups[te])),
                          **metrics(y[te], pred, len(TASKS[task]["classes"]))})
        res = {"model": model, "task": task, "variant": variant, "seed": seed, "n": int(len(y)),
               "class_counts": {str(c): int((y == c).sum()) for c in sorted(set(y))}, "choice": choice,
               "folds": folds}
        res.update({m: float(np.mean([f[m] for f in folds])) for m in ("accuracy", "macro_f1")})
        write_json(res, path)
        log(f"[inflation] {model} {variant} seed {seed}: acc {res['accuracy']:.4f} macro F1 {res['macro_f1']:.4f}")


# ---------------------------------------------------------------- cross-fit comparison

def job_compare() -> None:
    """Metric on each val half and their mean, for every compared model, task and setting."""
    ids, _, y3 = split3("val")
    h = halves(y3)
    rows = []

    def add(task, model, source, setting, seed, y, pred, hh):
        for k in (0, 1):
            sel = hh == k
            r = metrics(y[sel], pred[sel], len(TASKS[task]["classes"]))
            rows.append({"task": task, "model": model, "source": source, "setting": setting, "seed": seed,
                         "half": k, "n": int(sel.sum()), **r})

    def views(probs3):
        """Predictions of the three tasks from 3-class probabilities (ham, smishing, spam)."""
        bin_pred = (probs3[:, 1] + probs3[:, 2] > probs3[:, 0]).astype(np.int64)
        keep = y3 != 0
        sv = (probs3[keep, 1] > probs3[keep, 2]).astype(np.int64)
        return {"mendeley_3class": (y3, probs3.argmax(1), h), "mendeley_binary": (to_task("mendeley_binary", y3)[1],
                bin_pred, h), "smishing_vs_spam": (to_task("smishing_vs_spam", y3)[1], sv, h[keep])}

    for seed in CFG["seeds"]:
        for s in CFG["eval_settings"]:
            c = pd.read_csv(OUT / "cascade" / "predictions" / f"{s}_seed{seed}.csv", dtype={"id": str})
            for name in ("cascade", "average"):
                for task, (y, pred, hh) in views(c[[f"{name}_prob_{k}" for k in CLASSES3]].to_numpy()).items():
                    add(task, name, "step5a", s, seed, y, pred, hh)
            for m in sorted(p.name for p in STEP4C.iterdir() if (p / f"{s}_seed{seed}.csv").exists()):
                for task, (y, pred, hh) in views(load_4c(m, s, seed, ids)).items():
                    add(task, m, "step4c", s, seed, y, pred, hh)
            for method in [*CFG["published"]["shim2024"]["models"], "mahmud2024_cnn_bigru"]:
                f = OUT / "published" / method / f"{s}_seed{seed}.csv"
                if not f.exists():
                    continue
                d = pd.read_csv(f, dtype={"id": str})
                task = read_json(f.with_suffix(".json"))["task"]
                cls = TASKS[task]["classes"]
                hh = pd.Series(h, index=ids).loc[d["id"]].to_numpy()
                add(task, method, "published", s, seed, d["y_true"].map(cls.index).to_numpy(),
                    d["y_pred"].map(cls.index).to_numpy(), hh)
        for d in sorted(STEP4B.iterdir()) if STEP4B.exists() else []:  # 4b baselines: clean val only
            f = d / f"clean_seed{seed}.csv"
            if f.exists():
                b = pd.read_csv(f, dtype={"id": str}).set_index("id").loc[list(ids)]
                pred = b["y_pred"].map({c: i for i, c in enumerate(CLASSES3)}).fillna(-1).astype(int).to_numpy()
                add("mendeley_3class", d.name, "step4b", "clean", seed, y3, pred, h)
    df = pd.DataFrame(rows)
    path = OUT / "compare" / "crossfit.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    log(f"[compare] {len(df)} rows -> {path}")


# ---------------------------------------------------------------- addendum: our models in the published settings

def setting_model(model: str, task: str, seed: int, device):
    """Train one step-4c baseline / cascade stage on a 2-class setting -> {eval setting: probs}."""
    from eval.obfuscation import TRAIN_TYPES, augment
    from fusion import v2

    tr_ids, tr, ytr, _ = task_data(task, "train")
    va_ids, va, yva, keep = task_data(task, "val")
    sets = {s: list(np.asarray(t, dtype=object)[keep]) for s, t in eval_sets().items()}
    aug = v2.CFG["augmentation"]
    if model.startswith("svm"):
        import classic

        best = read_json(ROOT / "results" / "best_params" / "mendeley_3class" / "linear_svm.json")
        m, bw = classic.build_model("linear_svm", best["model_params"], methods.CFG2["models"]["linear_svm"]["fixed"],
                                    seed, 1)
        vec = v2.char_vectorizer() if model.startswith("svm_char") else \
            classic.build_vectorizer(best["vectorizer"], methods.CFG2["vectorizer_fixed"])
        clf = classic.TextClassifier(vec, m, bw)
        texts = list(tr)
        if model.endswith("_aug"):  # one augmented copy of train per seed, as in step 4c
            texts = augment(texts, list(tr_ids), seed, 0, aug["p"], aug["max_level"], TRAIN_TYPES)
        texts = np.asarray(texts, dtype=object)
        perm = np.random.default_rng(seed).permutation(len(ytr))
        clf.fit(texts[perm], ytr[perm])
        return {s: clf.predict_proba(t) for s, t in sets.items()}, {}
    if model.startswith("distilbert"):
        from fusion.train import fit, predict_texts

        zeros = lambda texts: np.zeros((len(texts), 1))  # noqa: E731 - the text-only model ignores features

        def augmented_epoch(epoch):
            t = augment(list(tr), list(tr_ids), seed, epoch, aug["p"], aug["max_level"], TRAIN_TYPES)
            return np.asarray(t, dtype=object), zeros(t)

        res = fit("text", v2.MCFG, (tr, ytr, zeros(tr)), (va, yva, zeros(va)), 2, seed, device,
                  v2.MCFG["text"]["checkpoint"], None, train_epoch=augmented_epoch if model.endswith("_aug") else None)
        out = {s: (res["val_probs"] if s == "clean" else
                   predict_texts(res["model"], res["tok"], t, zeros(t), device, v2.MCFG)[0]) for s, t in sets.items()}
        info = {"best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"]}
        del res
        torch.cuda.empty_cache()
        return out, info
    from neural.run import fit_once
    from neural.trainers import predict_proba

    step3 = load_configs()[1]
    mcfg = step3["models"]["bert"]
    best = read_json(ROOT / "results" / "best_params" / "mendeley_3class" / "bert.json")
    data = {"train": (list(tr_ids), list(tr), ytr), "val": (list(va_ids), list(va), yva)}
    res = fit_once("bert", mcfg, best["model_params"], "mendeley_3class", data, 2, seed, step3["protocol"], device)
    out = {s: predict_proba(res["model"], res["setup"], res["setup"].encode(t), mcfg["fixed"]["eval_batch_size"],
                            device, step3["protocol"]["fp16"]) for s, t in sets.items()}
    info = {"best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"]}
    del res
    torch.cuda.empty_cache()
    return out, info


def job_setting(task: str, model: str, seed: int, device) -> None:
    va_ids, _, yva, _ = task_data(task, "val")
    probs, info = setting_model(model, task, seed, device)
    for s, p in probs.items():
        save_preds(OUT / "setting" / task / model / f"{s}_seed{seed}", va_ids, yva, p, TASKS[task]["classes"],
                   {"task": task, "model": model, "seed": seed, "setting": s, **info})
    log(f"[setting] {task}/{model} seed {seed} done")


def job_setting_compare() -> None:
    """Cascade trained in each setting (cross-fit tau) and cross-fit scores of every model of the setting."""
    ids, _, y3 = split3("val")
    h_all = pd.Series(halves(y3), index=ids)
    cc, rows, taus = CFG["cascade"], [], []
    for task in CFG["setting_runs"]["tasks"]:
        va_ids, _, y, _ = task_data(task, "val")
        h, n = h_all.loc[list(va_ids)].to_numpy(), len(TASKS[task]["classes"])

        def probs(model, s, seed):
            d = pd.read_csv(OUT / "setting" / task / model / f"{s}_seed{seed}.csv", dtype={"id": str})
            return d[[f"prob_{c}" for c in TASKS[task]["classes"]]].to_numpy()

        def add(model, source, s, seed, pred):
            for k in (0, 1):
                sel = h == k
                rows.append({"task": task, "model": model, "source": source, "setting": s, "seed": seed, "half": k,
                             "n": int(sel.sum()), **metrics(y[sel], pred[sel], n)})

        for seed in CFG["seeds"]:
            p1 = {s: probs(cc["stage1"], s, seed) for s in CFG["eval_settings"]}
            p2 = {s: probs(cc["stage2"], s, seed) for s in CFG["eval_settings"]}
            tau = {}
            for k in (0, 1):
                sel = h == k
                tau[1 - k] = choose_tau(p1["clean"][sel].max(1), p1["clean"][sel], p2["clean"][sel], y[sel])
            tau_of = np.where(h == 0, tau[0], tau[1])
            entry = {"task": task, "seed": seed, "tau_by_half": {str(k): v for k, v in tau.items()}}
            for s in CFG["eval_settings"]:
                to2 = p1[s].max(1) < tau_of
                add("cascade", "setting", s, seed, np.where(to2, p2[s].argmax(1), p1[s].argmax(1)))
                add("average", "setting", s, seed, ((p1[s] + p2[s]) / 2).argmax(1))
                entry[s] = {"routed_share": float(to2.mean())}
                for m in CFG["setting_runs"]["models"]:
                    add(m, "setting", s, seed, probs(m, s, seed).argmax(1))
                for m in [*CFG["published"]["shim2024"]["models"], "mahmud2024_cnn_bigru"]:
                    f = OUT / "published" / m / f"{s}_seed{seed}.csv"
                    if f.exists() and read_json(f.with_suffix(".json"))["task"] == task:
                        d = pd.read_csv(f, dtype={"id": str}).set_index("id").loc[list(va_ids)]
                        add(m, "published", s, seed, d["y_pred"].map(TASKS[task]["classes"].index).to_numpy())
            taus.append(entry)
    path = OUT / "setting" / "crossfit.csv"
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
    write_json(taus, OUT / "setting" / "cascade_tau.json")
    log(f"[setting_compare] {len(rows)} rows -> {path}")


# ---------------------------------------------------------------- main

def done(job: str, seed: int | None) -> bool:
    kind, _, arg = job.partition(":")
    if kind == "cascade":
        return (OUT / "cascade" / f"curve_seed{seed}.json").exists()
    if kind == "published":
        return all((OUT / "published" / arg / f"{s}_seed{seed}.json").exists() for s in CFG["eval_settings"])
    if kind == "inflation":
        return all((OUT / "inflation" / arg / f"{v}_seed{seed}.json").exists() for v in ("dirty", "clean"))
    if kind == "setting":
        task, _, model = arg.partition(":")
        return all((OUT / "setting" / task / model / f"{s}_seed{seed}.json").exists() for s in CFG["eval_settings"])
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--worker", required=True)
    args = ap.parse_args()
    check_versions(args.worker, OUT / "environment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    single = ("compare", "setting_compare")  # jobs run once, not per seed
    units = [(j, s) for j in json.loads(args.queue) for s in ([None] if j in single else CFG["seeds"])]
    log(f"[step5a] worker {args.worker}, device {device}, {len(units)} units")
    times, failed = [], 0
    for job, seed in units:
        if done(job, seed):
            continue
        if time_left() <= 0 and job not in single:  # SMISHFUSE_DEADLINE; the comparisons take seconds of CPU
            log(f"[step5a] budget reached, stopping before {job} seed {seed}")
            times.append({"job": f"{job}__seed{seed}", "status": "not_run_budget"})
            failed += 1
            break
        t0 = time.perf_counter()
        kind, _, arg = job.partition(":")
        try:
            if kind == "cascade":
                job_cascade(seed)
            elif kind == "published":
                job_published(arg, seed, device)
            elif kind == "inflation":
                job_inflation(arg, seed, device)
            elif kind == "compare":
                job_compare()
            elif kind == "setting":
                task, _, model = arg.partition(":")
                job_setting(task, model, seed, device)
            elif kind == "setting_compare":
                job_setting_compare()
            else:
                raise ValueError(f"unknown job {job}")
            times.append({"job": f"{job}__seed{seed}", "status": "done", "seconds": time.perf_counter() - t0})
        except Exception:  # noqa: BLE001 - record and continue
            failed += 1
            tag = f"{job.replace(':', '_')}__seed{seed}"
            (OUT / "errors").mkdir(parents=True, exist_ok=True)
            (OUT / "errors" / f"{tag}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"[step5a] {job} seed {seed} FAILED:\n{traceback.format_exc()}")
            times.append({"job": f"{job}__seed{seed}", "status": "error", "seconds": time.perf_counter() - t0})
        write_json({"worker": args.worker, "jobs": times}, OUT / "job_times" / f"{args.worker}.json")
    write_json({"worker": args.worker, "jobs": times}, OUT / "job_times" / f"{args.worker}.json")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
