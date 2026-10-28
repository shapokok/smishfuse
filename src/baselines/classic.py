"""Step 2.2: classical TF-IDF baselines (LogReg, LinearSVM, RandomForest, XGBoost).

Protocol (configs/step2.yaml):
  * tuning: Optuna TPE (seeded), exactly n_trials per (task, model); every trial
    fits on train and is scored by macro F1 on val. Test is not read during tuning.
  * final: for each seed in eval_seeds, fit on train with the best parameters
    (train order shuffled by the seed, model random_state = seed), then predict
    test once.

Outputs:
  results/tuning/<task>/<model>.csv        every trial: parameters and val macro F1
  results/best_params/<task>/<model>.json  best parameters
  results/raw/<task>/<model>/seed<k>.csv   id, y_true, y_pred, prob_<class>... on test
  results/metrics.csv                      one row per task-model-seed (deterministic)
  results/efficiency.csv                   train time, batch-1 latency, model size
                                           (wall-clock based, so not bit-reproducible)
Run: python src/baselines/classic.py                 (all tasks and models)
     python src/baselines/classic.py moz_binary      (one task)
"""
from __future__ import annotations

import io
import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval.metrics import compute_metrics  # noqa: E402

CONFIG = ROOT / "configs" / "step2.yaml"
# Overridable so the same script runs on Kaggle (read-only inputs, /kaggle/working outputs).
DATA = Path(os.environ.get("SMISHFUSE_DATA", ROOT / "data" / "processed"))
RESULTS = Path(os.environ.get("SMISHFUSE_RESULTS", ROOT / "results"))
MODELS = ["logreg", "linear_svm", "random_forest", "xgboost"]
METRIC_ORDER = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "pr_auc", "recall_at_fpr_1pct", "ece"]


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_split(task: dict, split: str):
    df = pd.read_csv(DATA / task["dataset"] / f"{split}.csv", dtype=str, keep_default_na=False)
    names = df[task["label_column"]].map(task["label_map"])
    if names.isna().any():
        raise ValueError(f"unmapped labels in {task['dataset']}/{split}: {sorted(set(df[task['label_column']][names.isna()]))}")
    y = names.map({c: i for i, c in enumerate(task["classes"])}).to_numpy(dtype=np.int64)
    return df["id"].to_numpy(), df["text"].to_numpy(dtype=object), y


# ---------------------------------------------------------------- model building

def _tfidf(kind: str, vp: dict, fixed: dict) -> TfidfVectorizer:
    common = dict(lowercase=fixed["lowercase"], sublinear_tf=fixed["sublinear_tf"],
                  max_features=vp["max_features"], min_df=vp["min_df"], dtype=np.float32)
    if kind == "word":
        return TfidfVectorizer(analyzer="word", ngram_range=tuple(fixed["word_ngram_range"]), **common)
    return TfidfVectorizer(analyzer=fixed["char_analyzer"], ngram_range=tuple(fixed["char_ngram_range"]), **common)


def build_vectorizer(vp: dict, fixed: dict):
    if vp["features"] == "word+char":
        return FeatureUnion([("word", _tfidf("word", vp, fixed)), ("char", _tfidf("char", vp, fixed))])
    return _tfidf(vp["features"], vp, fixed)


def build_model(name: str, mp: dict, fixed: dict, seed: int, n_jobs: int):
    """Returns (estimator, use_balanced_sample_weight)."""
    if name == "logreg":
        return LogisticRegression(C=mp["C"], class_weight=mp["class_weight"], solver=fixed["solver"],
                                  max_iter=fixed["max_iter"], random_state=seed), False
    if name == "linear_svm":
        svm = LinearSVC(C=mp["C"], class_weight=mp["class_weight"], max_iter=fixed["max_iter"], random_state=seed)
        return CalibratedClassifierCV(svm, method=fixed["calibration_method"], cv=fixed["calibration_cv"],
                                      ensemble=fixed["calibration_ensemble"]), False
    if name == "random_forest":
        return RandomForestClassifier(**mp, n_jobs=n_jobs, random_state=seed), False
    if name == "xgboost":
        params = {k: v for k, v in mp.items() if k != "class_weight"}
        return XGBClassifier(**params, **fixed, n_jobs=n_jobs, random_state=seed), mp["class_weight"] == "balanced"
    raise ValueError(name)


def fit_model(model, balanced_weights: bool, X, y):
    if balanced_weights:
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    if isinstance(model, RandomForestClassifier):
        # Parallel predict_proba sums tree outputs in thread-arrival order, which changes
        # the last bits of the probabilities between runs; predict single-threaded.
        model.set_params(n_jobs=1)
    return model


def _strip_vectorizer(vec) -> None:
    """Drop stop_words_ (the pruned vocabulary, kept only for introspection) before measuring size."""
    parts = [t for _, t in vec.transformer_list] if isinstance(vec, FeatureUnion) else [vec]
    for t in parts:
        if hasattr(t, "stop_words_"):
            t.stop_words_ = None


class TextClassifier:
    def __init__(self, vectorizer, model, balanced_weights: bool):
        self.vectorizer = vectorizer
        self.model = model
        self.balanced_weights = balanced_weights

    def fit(self, texts, y):
        X = self.vectorizer.fit_transform(texts)
        _strip_vectorizer(self.vectorizer)
        fit_model(self.model, self.balanced_weights, X, y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self.vectorizer.transform(texts)), dtype=np.float64)

    def set_single_thread(self) -> None:
        if hasattr(self.model, "n_jobs"):
            self.model.set_params(n_jobs=1)


# ---------------------------------------------------------------- tuning

def suggest(trial: optuna.Trial, space: dict, prefix: str) -> dict:
    out = {}
    for name, spec in space.items():
        key = prefix + name
        if spec["type"] == "float":
            out[name] = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
        elif spec["type"] == "int":
            out[name] = trial.suggest_int(key, spec["low"], spec["high"], step=spec.get("step", 1))
        elif spec["type"] == "categorical":
            out[name] = trial.suggest_categorical(key, spec["choices"])
        else:
            raise ValueError(spec)
    return out


class FeatureCache:
    """TF-IDF matrices for train/val per vectorizer config (fitted on train only)."""

    def __init__(self, cfg: dict, train_texts, val_texts):
        self.fixed = cfg["vectorizer_fixed"]
        self.train_texts, self.val_texts = train_texts, val_texts
        self.cache = {}

    def get(self, vp: dict):
        key = (vp["features"], vp["max_features"], vp["min_df"])
        if key not in self.cache:
            vec = build_vectorizer(vp, self.fixed)
            self.cache[key] = (vec.fit_transform(self.train_texts), vec.transform(self.val_texts))
        return self.cache[key]


def tune(task_name: str, model_name: str, cfg: dict, data: dict, features: FeatureCache) -> dict:
    proto = cfg["protocol"]
    mcfg = cfg["models"][model_name]
    k = len(cfg["tasks"][task_name]["classes"])
    ytr, yva = data["train"][2], data["val"][2]

    def objective(trial):
        vp = suggest(trial, cfg["vectorizer_space"], "vec_")
        mp = suggest(trial, mcfg["space"], "")
        Xtr, Xva = features.get(vp)
        model, bw = build_model(model_name, mp, mcfg["fixed"], proto["tuning_seed"], proto["n_jobs"])
        fit_model(model, bw, Xtr, ytr)
        pred = np.asarray(model.predict_proba(Xva)).argmax(axis=1)
        return f1_score(yva, pred, labels=list(range(k)), average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=proto["tuning_seed"]))
    study.optimize(objective, n_trials=proto["n_trials"], n_jobs=1)
    if len(study.trials) != proto["n_trials"]:
        raise RuntimeError(f"expected {proto['n_trials']} trials, got {len(study.trials)}")

    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    out_dir = RESULTS / "tuning" / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out_dir / f"{model_name}.csv", index=False, lineterminator="\n")

    best = study.best_trial
    best_params = {
        "task": task_name,
        "model": model_name,
        "trial": best.number,
        "val_macro_f1": best.value,
        "vectorizer": {k_[4:]: v for k_, v in best.params.items() if k_.startswith("vec_")},
        "model_params": {k_: v for k_, v in best.params.items() if not k_.startswith("vec_")},
    }
    path = RESULTS / "best_params" / task_name / f"{model_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(best_params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return best_params


# ---------------------------------------------------------------- final runs

def measure_latency_ms(clf: TextClassifier, texts, n: int, warmup: int) -> float:
    sample = [texts[i % len(texts)] for i in range(n)]
    clf.set_single_thread()
    for t in sample[:warmup]:
        clf.predict_proba([t])
    times = []
    for t in sample:
        start = time.perf_counter()
        clf.predict_proba([t])
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000)


def model_size_mb(clf: TextClassifier) -> float:
    buf = io.BytesIO()
    joblib.dump(clf, buf)
    return buf.tell() / 1e6


def final_runs(task_name: str, model_name: str, cfg: dict, data: dict, best: dict):
    proto = cfg["protocol"]
    task = cfg["tasks"][task_name]
    classes = task["classes"]
    _, train_texts, ytr = data["train"]
    test_ids, test_texts, yte = data["test"]
    metric_rows, eff_rows = [], []
    for seed in proto["eval_seeds"]:
        perm = np.random.default_rng(seed).permutation(len(ytr))
        model, bw = build_model(model_name, best["model_params"], cfg["models"][model_name]["fixed"], seed, proto["n_jobs"])
        clf = TextClassifier(build_vectorizer(best["vectorizer"], cfg["vectorizer_fixed"]), model, bw)
        start = time.perf_counter()
        clf.fit(train_texts[perm], ytr[perm])
        train_time = time.perf_counter() - start

        probs = clf.predict_proba(test_texts)
        pred = probs.argmax(axis=1)
        raw = pd.DataFrame({"id": test_ids, "y_true": [classes[i] for i in yte], "y_pred": [classes[i] for i in pred]})
        for i, c in enumerate(classes):
            raw[f"prob_{c}"] = probs[:, i]
        raw_path = RESULTS / "raw" / task_name / model_name / f"seed{seed}.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_path, index=False, lineterminator="\n")

        m = compute_metrics(yte, probs, classes, positive=task["positive"], ece_bins=proto["ece_bins"],
                            max_fpr=proto["fpr_target"])
        metric_rows.append({"task": task_name, "model": model_name, "seed": seed, **m})
        eff_rows.append({
            "task": task_name, "model": model_name, "seed": seed,
            "train_time_s": train_time,
            "latency_ms_median_batch1": measure_latency_ms(clf, test_texts, proto["latency_messages"],
                                                           proto["latency_warmup"]),
            "model_size_mb": model_size_mb(clf),
            "n_train": len(ytr),
        })
        print(f"  [{task_name}/{model_name}] seed {seed}: macro F1 {m['macro_f1']:.4f}, "
              f"train {train_time:.1f}s", flush=True)
    return metric_rows, eff_rows


# ---------------------------------------------------------------- driver

def run_task(task_name: str, cfg: dict, models: list[str]) -> None:
    task = cfg["tasks"][task_name]
    data = {s: load_split(task, s) for s in ["train", "val", "test"]}
    print(f"[{task_name}] train {len(data['train'][2])}, val {len(data['val'][2])}, test {len(data['test'][2])}",
          flush=True)
    # Tuning uses only train and val.
    features = FeatureCache(cfg, data["train"][1], data["val"][1])
    metric_rows, eff_rows = [], []
    for model_name in models:
        t0 = time.perf_counter()
        best = tune(task_name, model_name, cfg, data, features)
        print(f"[{task_name}/{model_name}] tuned in {time.perf_counter() - t0:.0f}s: val macro F1 "
              f"{best['val_macro_f1']:.4f} {best['vectorizer']} {best['model_params']}", flush=True)
        m, e = final_runs(task_name, model_name, cfg, data, best)
        metric_rows += m
        eff_rows += e
    parts = RESULTS / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(parts / f"metrics_{task_name}.csv", index=False, lineterminator="\n")
    pd.DataFrame(eff_rows).to_csv(parts / f"efficiency_{task_name}.csv", index=False, lineterminator="\n")


def combine(cfg: dict) -> None:
    """Merge per-task parts into results/metrics.csv and results/efficiency.csv."""
    order = {t: i for i, t in enumerate(cfg["tasks"])}
    for kind in ["metrics", "efficiency"]:
        frames = [pd.read_csv(p) for p in sorted((RESULTS / "parts").glob(f"{kind}_*.csv"))]
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["task", "model", "seed"], key=lambda s: s.map(order) if s.name == "task" else
                            (s.map({m: i for i, m in enumerate(MODELS)}) if s.name == "model" else s))
        if kind == "metrics":
            first = ["task", "model", "seed"] + [c for c in METRIC_ORDER if c in df.columns]
            df = df[first + [c for c in df.columns if c not in first]]
        df.to_csv(RESULTS / f"{kind}.csv", index=False, lineterminator="\n")


def write_environment() -> None:
    import scipy
    import sklearn
    import xgboost

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "scipy": scipy.__version__, "xgboost": xgboost.__version__, "optuna": optuna.__version__},
    }
    (RESULTS / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str]) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_config()
    tasks = [a for a in argv if not a.startswith("--")] or list(cfg["tasks"])
    models = MODELS
    for a in argv:
        if a.startswith("--models="):
            models = a.split("=", 1)[1].split(",")
    RESULTS.mkdir(exist_ok=True)
    write_environment()
    for task_name in tasks:
        run_task(task_name, cfg, models)
    combine(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
