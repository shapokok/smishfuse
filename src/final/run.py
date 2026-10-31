"""Step 5b: final training and prediction on TEST (configs/final.yaml), the only test evaluation.

Every job trains one model for one seed with its frozen settings and saves the probabilities on the
clean val (for tau of the 2-class cascades) and on every test set of its condition:
results/final/preds/<condition>/<model>/<set>_seed<k>.csv (id, y_true, y_pred, prob_*). Nothing is
scored here: metrics, cascades, cost and significance are computed by src/final/assemble.py.
Seed-1 models of the mendeley_3class condition and of the published methods are saved under
results/final/models/ (not downloaded; read by the latency notebook as a kernel source).

Jobs (--queue, JSON list; "<condition>:<model>" for seeds 1-5, or "<condition>:<model>@1,2,3" for a subset;
finished units are skipped):
  mendeley_3class:<m>   clean + TRAIN / HELD-OUT obfuscated test
  low_data:<m>          the 4 nested train subsets of step 4b, clean test
  setting_mahmud:<m>    ham vs spam + smishing
  setting_shim:<m>      smishing vs spam, ham removed everywhere
  transfer:<m>          mendeley_binary -> moz_binary, zero-shot and fine-tuning on 10%
  super_sms:<m>         super_sms_clean_binary, clean test
SMISHFUSE_FINAL_DRY=1: smoke test without test data (val stands in for test, seed 1, 1 epoch, small
train), output under results/final_dry/.
Run: python src/final/run.py --queue '["mendeley_3class:svm_char_aug"]' --worker w0
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before CUDA initialises

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "baselines"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from eval.obfuscation import HELDOUT_TYPES, TRAIN_TYPES, augment, obfuscate_all  # noqa: E402
from fusion.envcheck import check_versions  # noqa: E402
from neural.common import RESULTS, load_configs, load_split, load_yaml, read_json, time_left, write_json  # noqa: E402

DRY = os.environ.get("SMISHFUSE_FINAL_DRY") == "1"
OUT = RESULTS / ("final_dry" if DRY else "final")
FCFG = load_yaml("configs/final.yaml")
STEP2, STEP3 = load_configs()
CFG2 = load_yaml("configs/step2.yaml")
CFG4B = load_yaml("configs/step4b.yaml")
CFG4C = load_yaml("configs/step4c.yaml")
CFG5A = load_yaml("configs/step5a.yaml")
MCFG = load_yaml(CFG4C["model_config"])  # step-4a text / fusion model
if DRY:
    MCFG = copy.deepcopy(MCFG)
    MCFG["text"]["max_epochs"] = 1
SEEDS = [1] if DRY else FCFG["seeds"]
N_JOBS = 2  # two workers share the 4 CPU cores of the Kaggle GPU machine
TASK_OF = {"mendeley_3class": "mendeley_3class", "low_data": "mendeley_3class", "setting_mahmud": "mendeley_binary",
           "setting_shim": "smishing_vs_spam", "super_sms": "super_sms_clean_binary"}
CLASSES = {"mendeley_3class": ["ham", "smishing", "spam"], "mendeley_binary": ["benign", "malicious"],
           "smishing_vs_spam": ["spam", "smishing"], "moz_binary": ["benign", "malicious"],
           "super_sms_clean_binary": ["benign", "malicious"]}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- data

_cache: dict = {}


def data(task: str, split: str):
    """ids, texts (object arrays), y for a task; smishing_vs_spam = mendeley_3class without ham
    (smishing 1, spam 0). In the dry run "test" is the val split and train is cut to 600 messages."""
    key = (task, split)
    if key not in _cache:
        real = "val" if (DRY and split == "test") else split
        base = "mendeley_3class" if task == "smishing_vs_spam" else task
        ids, texts, y = load_split(STEP2["tasks"][base], real)
        ids, texts, y = np.asarray(ids, dtype=object), np.asarray(texts, dtype=object), np.asarray(y)
        if task == "smishing_vs_spam":
            keep = y != 0
            ids, texts, y = ids[keep], texts[keep], (y[keep] == 1).astype(np.int64)
        _cache[key] = (ids, texts, y)
    return _cache[key]


def dry_cut(ids, texts, y, n: int = 600):
    """Dry run only: a fixed random part of train."""
    if not DRY or len(y) <= n:
        return ids, texts, y
    idx = np.sort(np.random.default_rng(0).choice(len(y), n, replace=False))
    return ids[idx], texts[idx], y[idx]


def test_sets(task: str, names: list[str]) -> dict:
    """name -> test texts; obfuscation seeded by message id and level (as for val in step 4c)."""
    ids, texts, _ = data(task, "test")
    out = {}
    for name in names:
        if name == "clean":
            out[name] = list(texts)
            continue
        types = TRAIN_TYPES if name.startswith("train") else HELDOUT_TYPES
        obf, stats = obfuscate_all(list(texts), list(ids), int(name[-2:]) / 100, types)
        write_json(stats, OUT / "levels" / f"{task}_{name}.json")
        out[name] = obf
    return out


def subset_mask(y: np.ndarray, fraction: float, min_per_class: int, ref: str) -> np.ndarray:
    """The nested subset of step 4b (same function, same seed), checked against its saved counts."""
    from fusion.stress import stratified_subset

    idx = stratified_subset(y, fraction, CFG4B["low_data"]["subset_seed"], min_per_class)
    saved = read_json(ROOT / "results" / "step4b" / ref)
    if len(idx) != saved["n"]:
        raise SystemExit(f"subset {ref}: {len(idx)} messages, step 4b had {saved['n']}")
    mask = np.zeros(len(y), dtype=bool)
    mask[idx] = True
    return mask


# ---------------------------------------------------------------- model families

def best_params(task: str, model: str) -> dict:
    return read_json(ROOT / "results" / "best_params" / task / f"{model}.json")


class Classic:
    """Step-2 pipelines and the char LinearSVM of step 4c (one augmented copy of train for *_aug)."""

    def __init__(self, model: str, params_task: str):
        self.model, self.params_task = model, params_task

    def fit(self, ids, texts, y, seed: int):
        import classic
        from fusion.v2 import char_vectorizer

        name = "linear_svm" if self.model in ("svm_char_aug", "svm_char", "svm_step2") else self.model
        # svm_char*: mendeley_3class C / class weight (as in step 4c); svm_step2 = linear_svm of the task
        ptask = "mendeley_3class" if self.model.startswith("svm_char") else self.params_task
        best = best_params(ptask, name)
        m, bw = classic.build_model(name, best["model_params"], CFG2["models"][name]["fixed"], seed, N_JOBS)
        vec = char_vectorizer() if self.model.startswith("svm_char") else \
            classic.build_vectorizer(best["vectorizer"], CFG2["vectorizer_fixed"])
        self.clf = classic.TextClassifier(vec, m, bw)
        texts = list(texts)
        if self.model.endswith("_aug"):
            aug = CFG4C["augmentation"]
            texts = augment(texts, list(ids), seed, 0, aug["p"], aug["max_level"], TRAIN_TYPES)
        texts = np.asarray(texts, dtype=object)
        perm = np.random.default_rng(seed).permutation(len(y))  # train order of the step-2 final runs
        self.clf.fit(texts[perm], np.asarray(y)[perm])
        return self

    def proba(self, texts):
        return self.clf.predict_proba(np.asarray(list(texts), dtype=object))

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path / "model.joblib")


class Fusion:
    """Step-4a/4c models: text-only DistilBERT (with / without augmentation) and SmishFuse v1 / v2."""

    SPEC = {"distilbert_aug": ("text", "none", True), "distilbert": ("text", "none", False),
            "smishfuse_v1": ("gated", "hand", False), "smishfuse_v2": ("gated", "char+hand", True),
            "smishfuse_v2_noaug": ("gated", "char+hand", False)}

    def __init__(self, model: str, checkpoint: str | None = None):
        self.model = model
        self.mode, self.kind, self.aug = self.SPEC[model]
        self.checkpoint = checkpoint or MCFG["text"]["checkpoint"]

    def _features(self, seed, texts):
        from fusion.v2 import CharHandFeatures

        if self.kind == "none":
            self.feats = lambda t: np.zeros((len(t), 1))  # the text-only model ignores features
        else:
            self.feats = CharHandFeatures(self.kind, seed).fit(list(texts))

    def fit(self, ids, texts, y, seed, val, n_classes, ham_index, device, model=None, tok=None, feats_from=None):
        from fusion.train import fit

        if feats_from is None:
            self._features(seed, texts)
        else:
            self.feats = feats_from
        aug = CFG4C["augmentation"]

        def augmented_epoch(epoch):
            t = augment(list(texts), list(ids), seed, epoch, aug["p"], aug["max_level"], TRAIN_TYPES)
            return np.asarray(t, dtype=object), self.feats(t)

        _, va, yva = val
        self.res = fit(self.mode, MCFG, (np.asarray(texts, dtype=object), np.asarray(y), self.feats(list(texts))),
                       (va, yva, self.feats(list(va))), n_classes, seed, device, self.checkpoint, ham_index,
                       model=model, tok=tok, train_epoch=augmented_epoch if self.aug else None)
        self.device = device
        return self

    def proba(self, texts):
        from fusion.train import predict_texts

        texts = list(texts)
        return predict_texts(self.res["model"], self.res["tok"], texts, self.feats(texts), self.device, MCFG)[0]

    def info(self):
        return {"best_epoch": self.res["best_epoch"], "epochs_run": self.res["epochs_run"]}

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        m = self.res["model"]
        torch.save({"state_dict": {k: v.cpu() for k, v in m.state_dict().items()}, "mode": self.mode,
                    "n_features": int(self.feats(["x"]).shape[1]), "n_classes": int(m.head3.out_features),
                    "checkpoint": self.checkpoint, "model": self.model}, path / "model.pt")
        if self.kind != "none":
            joblib.dump(self.feats, path / "features.joblib")

    def release(self):
        del self.res
        torch.cuda.empty_cache()


class Step3:
    """Step-3 neural baselines (textcnn, bilstm, distilbert_step3, bert) with their best parameters."""

    def __init__(self, model: str, params_task: str):
        self.name = "distilbert" if model == "distilbert_step3" else model
        self.params_task = params_task

    def fit(self, ids, texts, y, seed, val, n_classes, device):
        from neural.run import fit_once

        self.mcfg = STEP3["models"][self.name]
        if DRY:
            self.mcfg = copy.deepcopy(self.mcfg)
            self.mcfg["fixed"]["max_epochs"] = 1
        d = {"train": (list(ids), list(texts), np.asarray(y)), "val": (list(val[0]), list(val[1]), val[2])}
        self.res = fit_once(self.name, self.mcfg, best_params(self.params_task, self.name)["model_params"],
                            "mendeley_3class", d, n_classes, seed, STEP3["protocol"], device)
        self.device = device
        return self

    def proba(self, texts):
        from neural.trainers import predict_proba

        s = self.res["setup"]
        return predict_proba(self.res["model"], s, s.encode(list(texts)), self.mcfg["fixed"]["eval_batch_size"],
                             self.device, STEP3["protocol"]["fp16"])

    def info(self):
        return {"best_epoch": self.res["best_epoch"], "epochs_run": self.res["epochs_run"]}

    def save(self, path: Path):
        self.res["setup"].save(self.res["model"], path)

    def release(self):
        del self.res
        torch.cuda.empty_cache()


class Published:
    def __init__(self, model: str):
        self.model = model

    def fit(self, ids, texts, y, seed, device):
        from step5a import methods

        pc = CFG5A["published"]
        if self.model == "mahmud2024_cnn_bigru":
            mc = copy.deepcopy(pc[self.model])
            if DRY:
                mc["epochs"] = 1
            self.m = methods.fit_mahmud(mc, texts, y, seed, device)
            self.choice = None
        else:
            sc = copy.deepcopy(pc["shim2024"])
            if DRY:
                sc.update({"batch_sizes": [16], "lrs": [5e-5], "max_epochs": 1})
            ckpt = sc["models"][self.model]
            frozen = FCFG["models"][self.model]["choice"]
            if seed in frozen and not DRY:
                bs, lr, n_ep = frozen[seed]
                self.choice = {"batch_size": bs, "lr": lr, "epoch": n_ep - 1, "source": "step 5a (frozen)"}
            else:
                sel = methods.select_shim(sc, ckpt, texts, y, seed, device)
                write_json(sel, OUT / "selection" / self.model / f"seed{seed}.json")
                self.choice = {**sel["chosen"], "source": "grid on 20% of train (this run)"}
            self.m = methods.fit_shim(sc, ckpt, texts, y, seed, device, self.choice)
        return self

    def proba(self, texts):
        return self.m.predict_proba(list(texts))

    def info(self):
        return {"choice": self.choice}

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        if self.model == "mahmud2024_cnn_bigru":
            torch.save({"state_dict": {k: v.cpu() for k, v in self.m.model.state_dict().items()},
                        "vocab": self.m.vocab, "config": self.m.mc}, path / "model.pt")
        else:
            self.m.model.save_pretrained(path)
            self.m.tok.save_pretrained(path)

    def release(self):
        if hasattr(self.m, "release"):
            self.m.release()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- saving

def save_preds(cond: str, model: str, name: str, seed: int, ids, y, probs, classes, info=None) -> None:
    pred = probs.argmax(axis=1)
    df = pd.DataFrame({"id": ids, "y_true": [classes[i] for i in y], "y_pred": [classes[i] for i in pred]})
    for i, c in enumerate(classes):
        df[f"prob_{c}"] = probs[:, i]
    path = OUT / "preds" / cond / model / f"{name}_seed{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    if info is not None:
        write_json({"condition": cond, "model": model, "seed": seed, **info},
                   OUT / "preds" / cond / model / f"info_seed{seed}.json")


def build(model: str, params_task: str, checkpoint: str | None = None):
    """params_task: whose step-2 best parameters the classical models use."""
    if model in ("svm_char_aug", "svm_char", "svm_step2", "linear_svm", "logreg", "random_forest", "xgboost"):
        return Classic(model, params_task)
    if model in Fusion.SPEC:
        return Fusion(model, checkpoint)
    if model in ("textcnn", "bilstm", "distilbert_step3", "bert"):
        return Step3(model, "mendeley_3class")
    return Published(model)


def train(m, task, tr, val, seed, device):
    ids, texts, y = tr
    n, classes = len(CLASSES[task]), CLASSES[task]
    if isinstance(m, Classic):
        return m.fit(ids, texts, y, seed)
    if isinstance(m, Fusion):
        return m.fit(ids, texts, y, seed, val, n, classes.index("ham") if "ham" in classes else None, device)
    if isinstance(m, Step3):
        return m.fit(ids, texts, y, seed, val, n, device)
    return m.fit(ids, texts, y, seed, device)


# ---------------------------------------------------------------- jobs

def job_standard(cond: str, model: str, seed: int, device, fraction=None, tag=None) -> None:
    """Train on train (or its low-data subset), predict the clean val and every test set of the condition.
    Classical models use the best step-2 parameters of mendeley_3class (settings of the papers too, as in
    the step-5a addendum) or of super_sms_clean_binary."""
    task = TASK_OF[cond]
    c = FCFG["conditions"][{"super_sms": "super_sms_clean_binary"}.get(cond, cond)]
    names = c["test_sets"]
    ids, texts, y = data(task, "train")
    if fraction is not None:
        keep = subset_mask(y, fraction, CFG4B["low_data"]["min_per_class"],
                           f"low_data/subsets/frac{int(round(fraction * 100)):02d}.json")
        ids, texts, y = ids[keep], texts[keep], y[keep]
    else:
        ids, texts, y = dry_cut(ids, texts, y)
    val = data(task, "val")
    t0 = time.perf_counter()
    params_task = "super_sms_clean_binary" if task == "super_sms_clean_binary" else "mendeley_3class"
    m = train(build(model, params_task), task, (ids, texts, y), val, seed, device)
    info = {"seconds_train": time.perf_counter() - t0, "n_train": int(len(y)),
            **(m.info() if hasattr(m, "info") else {})}
    cname = tag or cond
    save_preds(cname, model, "val", seed, val[0], val[2], m.proba(val[1]), CLASSES[task], info)
    tids, _, ty = data(task, "test")
    for name, t in test_sets(task, names).items():
        save_preds(cname, model, name, seed, tids, ty, m.proba(t), CLASSES[task])
    if seed == 1 and cond in ("mendeley_3class", "setting_mahmud", "setting_shim"):
        m.save(OUT / "models" / cond / model / "seed1")
    if hasattr(m, "release"):
        m.release()


def job_low_data(model: str, seed: int, device) -> None:
    for f in FCFG["conditions"]["low_data"]["fractions"]:
        tag = f"low_data_frac{int(round(f * 100)):02d}"
        if (OUT / "preds" / tag / model / f"clean_seed{seed}.csv").exists():
            continue
        job_standard("low_data", model, seed, device, f, tag)


def job_transfer(model: str, seed: int, device) -> None:
    tc = CFG4B["transfer"]
    src, tgt = tc["source"], tc["target"]
    s_tr, s_va = dry_cut(*data(src, "train")), data(src, "val")
    t_tr, t_va, t_te = data(tgt, "train"), data(tgt, "val"), data(tgt, "test")
    sub = subset_mask(t_tr[2], tc["finetune_fraction"], 1, "transfer/subsets/finetune10.json")
    classes = CLASSES[tgt]
    t0 = time.perf_counter()
    if model in ("svm_char_aug", "svm_char", "linear_svm", "logreg"):
        zs = Classic(model, src).fit(*s_tr, seed)
        both = tuple(np.concatenate([a, b[sub]]) for a, b in zip(s_tr, t_tr))
        ft = Classic(model, src).fit(*both, seed)
        save_preds("transfer", model, "val_zeroshot", seed, s_va[0], s_va[2], zs.proba(s_va[1]), classes,
                   {"seconds": time.perf_counter() - t0, "n_finetune_subset": int(sub.sum())})
        save_preds("transfer", model, "val_finetune10", seed, t_va[0], t_va[2], ft.proba(t_va[1]), classes)
        save_preds("transfer", model, "zeroshot", seed, t_te[0], t_te[2], zs.proba(t_te[1]), classes)
        save_preds("transfer", model, "finetune10", seed, t_te[0], t_te[2], ft.proba(t_te[1]), classes)
        return
    # neural (multilingual text branch): train on the source, then continue on the target subset
    m = Fusion(model, tc["checkpoint"]).fit(*s_tr, seed, s_va, 2, None, device)
    info = {"source_best_epoch": m.res["best_epoch"], "checkpoint": tc["checkpoint"]}
    save_preds("transfer", model, "val_zeroshot", seed, s_va[0], s_va[2], m.proba(s_va[1]), classes, info)
    save_preds("transfer", model, "zeroshot", seed, t_te[0], t_te[2], m.proba(t_te[1]), classes)
    ft = Fusion(model, tc["checkpoint"]).fit(t_tr[0][sub], t_tr[1][sub], t_tr[2][sub], seed, t_va, 2, None, device,
                                            model=m.res["model"], tok=m.res["tok"], feats_from=m.feats)
    save_preds("transfer", model, "val_finetune10", seed, t_va[0], t_va[2], ft.proba(t_va[1]), classes)
    save_preds("transfer", model, "finetune10", seed, t_te[0], t_te[2], ft.proba(t_te[1]), classes)
    del m, ft
    torch.cuda.empty_cache()


def done(cond: str, model: str, seed: int) -> bool:
    if cond == "low_data":
        return all((OUT / "preds" / f"low_data_frac{int(round(f * 100)):02d}" / model / f"clean_seed{seed}.csv").exists()
                   for f in FCFG["conditions"]["low_data"]["fractions"])
    if cond == "transfer":
        return (OUT / "preds" / "transfer" / model / f"finetune10_seed{seed}.csv").exists()
    c = FCFG["conditions"][{"super_sms": "super_sms_clean_binary"}.get(cond, cond)]
    return all((OUT / "preds" / cond / model / f"{s}_seed{seed}.csv").exists() for s in c["test_sets"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--worker", required=True)
    args = ap.parse_args()
    check_versions(args.worker, OUT / "environment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    units = []
    for j in json.loads(args.queue):  # "<condition>:<model>" or "<condition>:<model>@1,2" (a subset of seeds)
        j, _, only = j.partition("@")
        units += [(j, s) for s in SEEDS if not only or s in {int(x) for x in only.split(",")}]
    log(f"[final] worker {args.worker}, device {device}, dry {DRY}, {len(units)} units")
    times, failed = [], 0
    for job, seed in units:
        cond, _, model = job.partition(":")
        if done(cond, model, seed):
            continue
        if time_left() <= 0:  # SMISHFUSE_DEADLINE from budget_hours
            log(f"[final] budget reached, stopping before {job} seed {seed}")
            times.append({"job": f"{job}__seed{seed}", "status": "not_run_budget"})
            failed += 1
            break
        t0 = time.perf_counter()
        try:
            if cond == "low_data":
                job_low_data(model, seed, device)
            elif cond == "transfer":
                job_transfer(model, seed, device)
            else:
                job_standard(cond, model, seed, device)
            times.append({"job": f"{job}__seed{seed}", "status": "done", "seconds": time.perf_counter() - t0})
        except Exception:  # noqa: BLE001 - record and continue
            failed += 1
            tag = f"{job.replace(':', '_')}__seed{seed}"
            (OUT / "errors").mkdir(parents=True, exist_ok=True)
            (OUT / "errors" / f"{tag}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"[final] {job} seed {seed} FAILED:\n{traceback.format_exc()}")
            times.append({"job": f"{job}__seed{seed}", "status": "error", "seconds": time.perf_counter() - t0})
        write_json({"worker": args.worker, "jobs": times}, OUT / "job_times" / f"{args.worker}.json")
    write_json({"worker": args.worker, "jobs": times}, OUT / "job_times" / f"{args.worker}.json")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
