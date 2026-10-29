"""Step 4c: SmishFuse v2 (character branch + augmentation) vs baselines, VAL ONLY.

Models and settings: configs/step4c.yaml. Every model is trained on mendeley_3class train
(clean, or with TRAIN-type augmentation) and evaluated on clean val, TRAIN-type and
HELD-OUT-type obfuscated val. Results: results/step4c/<model>/<setting>_seed<k>.json
(+ .csv predictions); finished (model, seed) pairs are skipped on restart. The installed
versions are checked against requirements.txt first.

Run: python src/fusion/v2.py --queue '["smishfuse_v2_aug", "svm_char"]' --worker w0
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before CUDA initialises

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "baselines"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from eval.obfuscation import HELDOUT_TYPES, TRAIN_TYPES, augment, obfuscate_all  # noqa: E402
from fusion.envcheck import check_versions  # noqa: E402
from fusion.features import Standardizer, extract_all  # noqa: E402
from fusion.stress import save, split  # noqa: E402
from neural.common import RESULTS, load_configs, load_yaml, read_json, time_left, write_json  # noqa: E402

OUT = RESULTS / "step4c"
CFG = load_yaml("configs/step4c.yaml")
MCFG = load_yaml(CFG["model_config"])
STEP2, STEP3 = load_configs()
CFG2 = yaml.safe_load((ROOT / "configs" / "step2.yaml").read_text(encoding="utf-8"))
TASK = CFG["task"]
CLASSES = STEP2["tasks"][TASK]["classes"]


def log(msg: str) -> None:
    print(msg, flush=True)


def settings() -> list[str]:
    lv = [f"{int(round(x * 100)):02d}" for x in CFG["eval_levels"]]
    return ["clean"] + [f"train{x}" for x in lv] + [f"heldout{x}" for x in lv]


_eval_sets: dict = {}


def eval_sets() -> dict:
    """setting -> list of val texts (fixed obfuscation per level and type group)."""
    if not _eval_sets:
        ids, texts, _ = split(TASK, "val")
        _eval_sets["clean"] = list(texts)
        for group, types in [("train", TRAIN_TYPES), ("heldout", HELDOUT_TYPES)]:
            for lv in CFG["eval_levels"]:
                name = f"{group}{int(round(lv * 100)):02d}"
                out, stats = obfuscate_all(list(texts), list(ids), lv, types)
                _eval_sets[name] = out
                write_json(stats, OUT / "levels" / f"{name}.json")
    return _eval_sets


def char_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer

    cb = CFG["char_branch"]
    return TfidfVectorizer(analyzer="char", ngram_range=tuple(cb["ngram_range"]), lowercase=cb["lowercase"],
                           sublinear_tf=cb["sublinear_tf"], min_df=cb["min_df"], dtype=np.float32)


class CharHandFeatures:
    """[TruncatedSVD(char TF-IDF); hand-crafted features], standardized; all fitted on clean train."""

    def __init__(self, kind: str, seed: int):
        self.kind, self.seed = kind, seed

    def raw(self, texts) -> np.ndarray:
        hand = extract_all(texts)
        if self.kind == "hand":
            return hand
        return np.hstack([self.svd.transform(self.vec.transform(texts)), hand])

    def fit(self, texts) -> "CharHandFeatures":
        if self.kind == "char+hand":
            from sklearn.decomposition import TruncatedSVD

            self.vec = char_vectorizer()
            self.svd = TruncatedSVD(CFG["char_branch"]["svd_components"], random_state=self.seed)
            self.svd.fit(self.vec.fit_transform(texts))
        self.std = Standardizer().fit(self.raw(texts))
        return self

    def __call__(self, texts) -> np.ndarray:
        return self.std.transform(self.raw(texts))


def gate_info(gate: np.ndarray) -> dict:
    if gate is None or np.isnan(gate).all():
        return {}
    return {"gate_mean": float(gate.mean()), "gate_std": float(gate.std())}


def job_fusion(name: str, spec: dict, seed: int, device) -> None:
    from fusion.train import fit, predict_texts

    tr_ids, tr, ytr = split(TASK, "train")
    va_ids, va, yva = split(TASK, "val")
    if spec["features"] == "none":
        feats = lambda texts: np.zeros((len(texts), 1))  # noqa: E731 - the text-only model ignores features
        n_features = 1
    else:
        feats = CharHandFeatures(spec["features"], seed).fit(list(tr))
        n_features = feats(list(tr[:1])).shape[1]
    aug = CFG["augmentation"]

    def augmented_epoch(epoch):
        texts = augment(list(tr), list(tr_ids), seed, epoch, aug["p"], aug["max_level"], TRAIN_TYPES)
        return np.asarray(texts, dtype=object), feats(texts)

    train_epoch = augmented_epoch if spec["augment"] else None
    res = fit(spec["mode"], MCFG, (tr, ytr, feats(list(tr))), (va, yva, feats(list(va))), len(CLASSES), seed, device,
              MCFG["text"]["checkpoint"], CLASSES.index("ham"), train_epoch=train_epoch)
    info = {"best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"], "seconds_train": res["seconds"],
            "n_features": n_features, "augment": spec["augment"]}
    for setting, texts in eval_sets().items():
        if setting == "clean":
            probs, gate = res["val_probs"], res["val_gate"]
        else:
            probs, gate = predict_texts(res["model"], res["tok"], texts, feats(texts), device, MCFG)
        save("step4c", name, setting, seed, va_ids, yva, probs, CLASSES, {**info, **gate_info(gate)}, gate=gate,
             root=RESULTS)
    del res
    torch.cuda.empty_cache()


def job_bert(name: str, seed: int, device) -> None:
    from neural.run import fit_once
    from neural.trainers import predict_proba

    mcfg = STEP3["models"]["bert"]
    best = read_json(ROOT / "results" / "best_params" / TASK / "bert.json")
    data = {k: (list(v[0]), list(v[1]), v[2]) for k, v in {"train": split(TASK, "train"), "val": split(TASK, "val")}.items()}
    res = fit_once("bert", mcfg, best["model_params"], TASK, data, len(CLASSES), seed, STEP3["protocol"], device)
    va_ids, _, yva = split(TASK, "val")
    for setting, texts in eval_sets().items():
        probs = predict_proba(res["model"], res["setup"], res["setup"].encode(texts), mcfg["fixed"]["eval_batch_size"],
                              device, STEP3["protocol"]["fp16"])
        save("step4c", name, setting, seed, va_ids, yva, probs, CLASSES,
             {"best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"]}, root=RESULTS)
    del res
    torch.cuda.empty_cache()


def job_svm(name: str, spec: dict, seed: int) -> None:
    import classic

    tr_ids, tr, ytr = split(TASK, "train")
    va_ids, _, yva = split(TASK, "val")
    best = read_json(ROOT / "results" / "best_params" / TASK / "linear_svm.json")
    model, bw = classic.build_model("linear_svm", best["model_params"], CFG2["models"]["linear_svm"]["fixed"], seed, 1)
    vec = char_vectorizer() if spec["kind"] == "svm_char" else \
        classic.build_vectorizer(best["vectorizer"], CFG2["vectorizer_fixed"])
    clf = classic.TextClassifier(vec, model, bw)
    texts = list(tr)
    if spec["augment"]:
        aug = CFG["augmentation"]
        texts = augment(texts, list(tr_ids), seed, 0, aug["p"], aug["max_level"], TRAIN_TYPES)
    texts = np.asarray(texts, dtype=object)
    perm = np.random.default_rng(seed).permutation(len(ytr))  # train order of the step-2 final runs
    t0 = time.perf_counter()
    clf.fit(texts[perm], ytr[perm])
    for setting, ev in eval_sets().items():
        save("step4c", name, setting, seed, va_ids, yva, clf.predict_proba(ev), CLASSES,
             {"seconds_train": time.perf_counter() - t0, "augment": spec["augment"]}, root=RESULTS)


def done(name: str, seed: int) -> bool:
    return all((OUT / name / f"{s}_seed{seed}.json").exists() for s in settings())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--worker", required=True)
    args = ap.parse_args()
    check_versions(args.worker, OUT / "environment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = [(m, s) for m in json.loads(args.queue) for s in CFG["seeds"]]
    log(f"[step4c] worker {args.worker}, device {device}, {len(jobs)} jobs")
    times, failed = [], 0
    for name, seed in jobs:
        if done(name, seed):
            continue
        if time_left() <= 0:  # SMISHFUSE_DEADLINE from budget_hours
            log(f"[step4c] budget reached, stopping before {name} seed {seed}")
            times.append({"job": f"{name}__seed{seed}", "status": "not_run_budget"})
            failed += 1
            break
        spec = CFG["models"][name]
        t0 = time.perf_counter()
        try:
            if spec["kind"] == "fusion":
                job_fusion(name, spec, seed, device)
            elif spec["kind"] == "step3":
                job_bert(name, seed, device)
            else:
                job_svm(name, spec, seed)
            times.append({"job": f"{name}__seed{seed}", "status": "done", "seconds": time.perf_counter() - t0})
        except Exception:  # noqa: BLE001 - record and continue
            failed += 1
            (OUT / "errors").mkdir(parents=True, exist_ok=True)
            (OUT / "errors" / f"{name}__seed{seed}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"[step4c] {name} seed {seed} FAILED:\n{traceback.format_exc()}")
            times.append({"job": f"{name}__seed{seed}", "status": "error", "seconds": time.perf_counter() - t0})
        write_json({"worker": args.worker, "jobs": times}, OUT / "job_times" / f"{args.worker}.json")
    write_json({"worker": args.worker, "jobs": times}, OUT / "job_times" / f"{args.worker}.json")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
