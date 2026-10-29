"""Step 4b: SmishFuse under stress conditions, VAL ONLY (no test split is read).

Conditions (configs/step4b.yaml), seeds 1-3, step-4a hyperparameters:
* baseline     all step-2 / step-3 baselines with best parameters, clean mendeley_3class val;
* obfuscation  train on clean train, evaluate on clean and obfuscated val (src/eval/obfuscation.py);
* low_data     train on nested stratified train subsets, evaluate on the full val;
* transfer     train on mendeley_binary, evaluate on moz_binary val zero-shot and after
               fine-tuning on a stratified 10% of moz_binary train.
Every (condition, model, setting, seed) is written as soon as it exists to
results/step4b/<condition>/<model>/<setting>_seed<k>.json (+ .csv predictions);
finished ones are skipped on restart. Before any work the installed versions are
compared with the pins of requirements.txt; on a mismatch the worker stops.

Run: python src/fusion/stress.py --conditions '["baseline", "transfer"]' --worker w0
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before CUDA initialises

import argparse  # noqa: E402
import importlib.metadata as md  # noqa: E402
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
import yaml  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402

from eval.obfuscation import obfuscate_all  # noqa: E402
from fusion.features import Standardizer, extract_all  # noqa: E402
from fusion.model import VARIANTS  # noqa: E402
from neural.common import RESULTS, load_configs, load_split, load_yaml, read_json, time_left, write_json  # noqa: E402

OUT = RESULTS / "step4b"
CFG = load_yaml("configs/step4b.yaml")
MCFG = load_yaml(CFG["model_config"])
STEP2, STEP3 = load_configs()
CFG2 = yaml.safe_load((ROOT / "configs" / "step2.yaml").read_text(encoding="utf-8"))
CLASSIC = {"logreg", "linear_svm", "random_forest", "xgboost"}
FUSION = set(VARIANTS)
N_JOBS = 2  # two workers share the 4 CPU cores of the Kaggle GPU machine


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- environment

def requirement_pins() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, ver = line.split("==")
            pins[name.strip()] = ver.strip()
    return pins


def check_versions(worker: str) -> None:
    rows, bad = {}, []
    for name, ver in requirement_pins().items():
        if name == "pytest":
            continue
        try:
            inst = md.version(name)
        except md.PackageNotFoundError:
            inst = None
        rows[name] = {"pinned": ver, "installed": inst}
        if inst is None or inst.split("+")[0] != ver:  # torch 2.11.0+cu128 matches the pin 2.11.0
            bad.append(name)
    write_json({"packages": rows, "mismatch": bad}, OUT / "environment" / f"{worker}.json")
    if bad:
        raise SystemExit(f"installed versions differ from requirements.txt: {bad}")


# ---------------------------------------------------------------- data

_split_cache: dict = {}
_feat_cache: dict = {}


def split(task_name: str, name: str):
    key = (task_name, name)
    if key not in _split_cache:
        ids, texts, y = load_split(STEP2["tasks"][task_name], name)
        _split_cache[key] = (np.asarray(ids, dtype=object), np.asarray(texts, dtype=object), y)
    return _split_cache[key]


def raw_features(key, texts) -> np.ndarray:
    if key not in _feat_cache:
        _feat_cache[key] = extract_all(texts)
    return _feat_cache[key]


def stratified_subset(y: np.ndarray, fraction: float, seed: int, min_per_class: int) -> np.ndarray:
    """Nested in `fraction`: per class a fixed permutation, the first n_c indices."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        perm = members[rng.permutation(len(members))]
        n = min(len(members), max(min_per_class, int(round(fraction * len(members)))))
        idx.append(perm[:n])
    return np.sort(np.concatenate(idx))


def obfuscated_val(level: float):
    ids, texts, y = split(CFG["task"], "val")
    key = ("obf", level)
    if key not in _split_cache:
        out, stats = obfuscate_all(list(texts), list(ids), level)
        _split_cache[key] = (np.asarray(out, dtype=object), stats)
        write_json(stats, OUT / "obfuscation" / "levels" / f"level_{int(round(level * 100)):02d}.json")
    return _split_cache[key]


# ---------------------------------------------------------------- results

def scores(y, pred, classes) -> dict:
    labels = list(range(len(classes)))
    rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    out = {"macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
           "accuracy": float(accuracy_score(y, pred))}
    out.update({f"recall_{c}": float(rec[i]) for i, c in enumerate(classes)})
    out.update({f"f1_{c}": float(f1[i]) for i, c in enumerate(classes)})
    return out


def result_path(cond, model, setting, seed) -> Path:
    return OUT / cond / model / f"{setting}_seed{seed}.json"


def save(cond, model, setting, seed, ids, y, probs, classes, info=None, pred=None, gate=None) -> None:
    pred = probs.argmax(axis=1) if pred is None else pred
    df = pd.DataFrame({"id": ids, "y_true": [classes[i] for i in y],
                       "y_pred": [classes[i] if i >= 0 else "invalid" for i in pred]})
    for i, c in enumerate(classes):
        df[f"prob_{c}"] = probs[:, i]
    res = {"condition": cond, "model": model, "setting": setting, "seed": seed, "split": "val", "n": len(y),
           **scores(y, pred, classes), **(info or {})}
    if gate is not None and not np.isnan(gate).all():
        df["gate_mean"] = gate
        res["gate_mean_by_true_class"] = {c: float(gate[y == i].mean()) for i, c in enumerate(classes)}
    path = result_path(cond, model, setting, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    write_json(res, path)
    log(f"[step4b] {cond}/{model}/{setting} seed {seed}: val macro F1 {res['macro_f1']:.4f}")


# ---------------------------------------------------------------- model families

def classic_fit(model_name: str, params_task: str, texts, y, seed: int):
    import classic

    best = read_json(ROOT / "results" / "best_params" / params_task / f"{model_name}.json")
    model, bw = classic.build_model(model_name, best["model_params"], CFG2["models"][model_name]["fixed"], seed, N_JOBS)
    clf = classic.TextClassifier(classic.build_vectorizer(best["vectorizer"], CFG2["vectorizer_fixed"]), model, bw)
    perm = np.random.default_rng(seed).permutation(len(y))  # train order of the step-2 final runs
    return clf.fit(texts[perm], y[perm])


def fusion_fit(name, train, val, n_classes, seed, device, checkpoint, ham_index, model=None, tok=None):
    from fusion.train import fit

    return fit(VARIANTS[name], MCFG, train, val, n_classes, seed, device, checkpoint, ham_index, model, tok)


def step3_neural(model_name: str, seed: int, device):
    from neural.run import fit_once
    from neural.trainers import predict_proba

    task_name = CFG["task"]
    mcfg = STEP3["models"][model_name]
    best = read_json(ROOT / "results" / "best_params" / task_name / f"{model_name}.json")
    data = {"train": split(task_name, "train"), "val": split(task_name, "val")}
    data = {k: (list(v[0]), list(v[1]), v[2]) for k, v in data.items()}
    n_classes = len(STEP2["tasks"][task_name]["classes"])
    res = fit_once(model_name, mcfg, best["model_params"], task_name, data, n_classes, seed, STEP3["protocol"], device)
    probs = predict_proba(res["model"], res["setup"], res["setup"].encode(data["val"][1]),
                          mcfg["fixed"]["eval_batch_size"], device, STEP3["protocol"]["fp16"])
    info = {"best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
            "best_val_macro_f1_during_training": res["best_val_macro_f1"]}
    del res
    torch.cuda.empty_cache()
    return probs, info


def qwen(seed: int):
    from neural.llm import FewShotClassifier, normalize_answer

    task_name = CFG["task"]
    task = STEP2["tasks"][task_name]
    mcfg = STEP3["models"]["qwen_fewshot"]
    tr = split(task_name, "train")
    clf = FewShotClassifier(task_name, task, mcfg, (list(tr[0]), list(tr[1]), tr[2]), seed)
    _, texts, y = split(task_name, "val")
    use_cache = bool(mcfg["fixed"]["prefix_cache"])
    bs = mcfg["fixed"]["cache_batch_size"] if use_cache else mcfg["fixed"]["batch_size"]
    order = np.argsort([len(t) for t in texts], kind="stable")
    probs = np.zeros((len(y), len(task["classes"])))
    pred = np.full(len(y), -1)
    for s in range(0, len(order), bs):
        idx = order[s : s + bs]
        raw, p, _ = clf.answer([texts[i] for i in idx], use_cache)
        for j, i in enumerate(idx):
            probs[i] = p[j]
            c = clf.answers.get(normalize_answer(raw[j]))
            pred[i] = task["classes"].index(c) if c is not None else -1
    info = {"invalid_rate": float((pred < 0).mean()), "shot_ids": clf.shot_ids, "prefix_cache": use_cache}
    del clf
    torch.cuda.empty_cache()
    return probs, pred, info


# ---------------------------------------------------------------- conditions

def job_baseline(model: str, seed: int, device) -> None:
    task_name = CFG["task"]
    classes = STEP2["tasks"][task_name]["classes"]
    ids, va, yva = split(task_name, "val")
    t0 = time.perf_counter()
    pred = None
    if model in CLASSIC:
        _, tr, ytr = split(task_name, "train")
        probs, info = classic_fit(model, task_name, tr, ytr, seed).predict_proba(va), {}
    elif model == "qwen_fewshot":
        probs, pred, info = qwen(seed)
    else:
        probs, info = step3_neural(model, seed, device)
    save("baseline", model, "clean", seed, ids, yva, probs, classes,
         {**info, "seconds": time.perf_counter() - t0}, pred=pred)


def job_obfuscation(model: str, seed: int, device) -> None:
    task_name = CFG["task"]
    classes = STEP2["tasks"][task_name]["classes"]
    ids, va, yva = split(task_name, "val")
    _, tr, ytr = split(task_name, "train")
    levels = CFG["obfuscation"]["levels"]
    sets = [("clean", va)] + [(f"obf{int(round(lv * 100)):02d}", obfuscated_val(lv)[0]) for lv in levels]
    t0 = time.perf_counter()
    if model in CLASSIC:
        clf = classic_fit(model, task_name, tr, ytr, seed)
        for name, texts in sets:
            save("obfuscation", model, name, seed, ids, yva, clf.predict_proba(texts), classes,
                 {"seconds_train": time.perf_counter() - t0})
        return
    from fusion.train import predict_texts

    std = Standardizer().fit(raw_features(("train", task_name), tr))
    F = {name: std.transform(raw_features(("val", task_name, name), texts)) for name, texts in sets}
    res = fusion_fit(model, (tr, ytr, std.transform(raw_features(("train", task_name), tr))), (va, yva, F["clean"]),
                     len(classes), seed, device, MCFG["text"]["checkpoint"], classes.index("ham"))
    info = {"best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"], "seconds_train": res["seconds"]}
    for name, texts in sets:
        probs, gate = (res["val_probs"], res["val_gate"]) if name == "clean" else \
            predict_texts(res["model"], res["tok"], texts, F[name], device, MCFG)
        save("obfuscation", model, name, seed, ids, yva, probs, classes, info, gate=gate)
    del res
    torch.cuda.empty_cache()


def job_low_data(model: str, seed: int, fraction: float, device) -> None:
    task_name = CFG["task"]
    classes = STEP2["tasks"][task_name]["classes"]
    ids, va, yva = split(task_name, "val")
    _, tr, ytr = split(task_name, "train")
    ld = CFG["low_data"]
    sub = stratified_subset(ytr, fraction, ld["subset_seed"], ld["min_per_class"])
    setting = f"frac{int(round(fraction * 100)):02d}"
    write_json({"fraction": fraction, "n": int(len(sub)),
                "per_class": {c: int((ytr[sub] == i).sum()) for i, c in enumerate(classes)}},
               OUT / "low_data" / "subsets" / f"{setting}.json")
    info = {"n_train": int(len(sub))}
    t0 = time.perf_counter()
    if model in CLASSIC:
        probs = classic_fit(model, task_name, tr[sub], ytr[sub], seed).predict_proba(va)
        save("low_data", model, setting, seed, ids, yva, probs, classes, {**info, "seconds": time.perf_counter() - t0})
        return
    Ftr = raw_features(("train", task_name), tr)[sub]
    std = Standardizer().fit(Ftr)
    res = fusion_fit(model, (tr[sub], ytr[sub], std.transform(Ftr)),
                     (va, yva, std.transform(raw_features(("val", task_name, "clean"), va))),
                     len(classes), seed, device, MCFG["text"]["checkpoint"], classes.index("ham"))
    save("low_data", model, setting, seed, ids, yva, res["val_probs"], classes,
         {**info, "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"], "seconds": res["seconds"]},
         gate=res["val_gate"])
    del res
    torch.cuda.empty_cache()


def job_transfer(model: str, seed: int, device) -> None:
    tc = CFG["transfer"]
    src, tgt = tc["source"], tc["target"]
    classes = STEP2["tasks"][tgt]["classes"]
    if STEP2["tasks"][src]["classes"] != classes:
        raise RuntimeError("source and target tasks must share the class list")
    _, s_tr, ys_tr = split(src, "train")
    _, s_va, ys_va = split(src, "val")
    t_ids, t_va, yt_va = split(tgt, "val")
    _, t_tr, yt_tr = split(tgt, "train")
    sub = stratified_subset(yt_tr, tc["finetune_fraction"], tc["subset_seed"], 1)
    setting_ft = f"finetune{int(round(tc['finetune_fraction'] * 100)):02d}"
    write_json({"n": int(len(sub)), "per_class": {c: int((yt_tr[sub] == i).sum()) for i, c in enumerate(classes)}},
               OUT / "transfer" / "subsets" / f"{setting_ft}.json")
    t0 = time.perf_counter()
    if model in CLASSIC:
        probs = classic_fit(model, src, s_tr, ys_tr, seed).predict_proba(t_va)
        save("transfer", model, "zeroshot", seed, t_ids, yt_va, probs, classes, {"seconds": time.perf_counter() - t0})
        t0 = time.perf_counter()
        both_x = np.concatenate([s_tr, t_tr[sub]])
        both_y = np.concatenate([ys_tr, yt_tr[sub]])
        probs = classic_fit(model, src, both_x, both_y, seed).predict_proba(t_va)
        save("transfer", model, setting_ft, seed, t_ids, yt_va, probs, classes,
             {"seconds": time.perf_counter() - t0, "n_train": int(len(both_y)),
              "note": "re-fitted on source train + target subset (no incremental fine-tuning)"})
        return
    std = Standardizer().fit(raw_features(("train", src), s_tr))
    F = lambda key, texts: std.transform(raw_features(key, texts))  # noqa: E731
    res = fusion_fit(model, (s_tr, ys_tr, F(("train", src), s_tr)), (s_va, ys_va, F(("val", src, "clean"), s_va)),
                     len(classes), seed, device, tc["checkpoint"], None)
    from fusion.train import predict_texts

    Ft_va = F(("val", tgt, "clean"), t_va)
    probs, gate = predict_texts(res["model"], res["tok"], t_va, Ft_va, device, MCFG)
    save("transfer", model, "zeroshot", seed, t_ids, yt_va, probs, classes,
         {"source_best_epoch": res["best_epoch"], "source_val_macro_f1": res["best_val_macro_f1"],
          "seconds": res["seconds"], "checkpoint": tc["checkpoint"]}, gate=gate)
    ft = fusion_fit(model, (t_tr[sub], yt_tr[sub], F(("train", tgt), t_tr)[sub]), (t_va, yt_va, Ft_va),
                    len(classes), seed, device, tc["checkpoint"], None, model=res["model"], tok=res["tok"])
    save("transfer", model, setting_ft, seed, t_ids, yt_va, ft["val_probs"], classes,
         {"best_epoch": ft["best_epoch"], "epochs_run": ft["epochs_run"], "seconds": ft["seconds"],
          "n_finetune": int(len(sub)), "checkpoint": tc["checkpoint"],
          "note": "same model fine-tuned on the target subset; features standardized with source train"},
         gate=ft["val_gate"])
    del res, ft
    torch.cuda.empty_cache()


def expand(condition: str) -> list[dict]:
    seeds = CFG["seeds"]
    if condition == "baseline":
        return [{"cond": "baseline", "model": m, "seed": s, "settings": ["clean"]}
                for m in CFG["baselines"] for s in seeds]
    if condition == "obfuscation":
        st = ["clean"] + [f"obf{int(round(lv * 100)):02d}" for lv in CFG["obfuscation"]["levels"]]
        return [{"cond": "obfuscation", "model": m, "seed": s, "settings": st} for m in CFG["compared"] for s in seeds]
    if condition == "low_data":
        return [{"cond": "low_data", "model": m, "seed": s, "fraction": f,
                 "settings": [f"frac{int(round(f * 100)):02d}"]}
                for f in CFG["low_data"]["fractions"] for m in CFG["compared"] for s in seeds]
    if condition == "transfer":
        st = ["zeroshot", f"finetune{int(round(CFG['transfer']['finetune_fraction'] * 100)):02d}"]
        return [{"cond": "transfer", "model": m, "seed": s, "settings": st} for m in CFG["compared"] for s in seeds]
    raise ValueError(condition)


def run_job(job: dict, device) -> None:
    if job["cond"] == "baseline":
        job_baseline(job["model"], job["seed"], device)
    elif job["cond"] == "obfuscation":
        job_obfuscation(job["model"], job["seed"], device)
    elif job["cond"] == "low_data":
        job_low_data(job["model"], job["seed"], job["fraction"], device)
    else:
        job_transfer(job["model"], job["seed"], device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--worker", required=True)
    args = ap.parse_args()
    check_versions(args.worker)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = [j for c in json.loads(args.conditions) for j in expand(c)]
    log(f"[step4b] worker {args.worker}, device {device}, {len(jobs)} jobs")
    times, failed = [], 0
    for job in jobs:
        name = f"{job['cond']}__{job['model']}__{'_'.join(job['settings'])}__seed{job['seed']}"
        if all(result_path(job["cond"], job["model"], s, job["seed"]).exists() for s in job["settings"]):
            continue
        if time_left() <= 0:  # SMISHFUSE_DEADLINE from budget_hours
            log(f"[step4b] budget reached, stopping before {name}")
            times.append({"job": name, "status": "not_run_budget"})
            failed += 1
            break
        t0 = time.perf_counter()
        try:
            run_job(job, device)
            times.append({"job": name, "status": "done", "seconds": time.perf_counter() - t0})
        except Exception:  # noqa: BLE001 - record and continue
            failed += 1
            (OUT / "errors").mkdir(parents=True, exist_ok=True)
            (OUT / "errors" / f"{name}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"[step4b] {name} FAILED:\n{traceback.format_exc()}")
            times.append({"job": name, "status": "error", "seconds": time.perf_counter() - t0})
        write_json({"worker": args.worker, "jobs": times}, OUT / "job_times" / f"{args.worker}.json")
    write_json({"worker": args.worker, "jobs": times}, OUT / "job_times" / f"{args.worker}.json")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
