"""Step 6b, POST-HOC (README decision 21): adaptation of Lite to the Kazakhstan context, Kaggle CPU.

Lite = char LinearSVM without augmentation with its final settings (no tuning), seed 1, trained on
Mendeley train + messages of the adaptation set (data/kz_adapt/adapt.csv, fixed by commit 8644b17).
Conditions: A = only categories absent from the test cases (same_category_as_test = false);
B = the whole set. Curve: 50, 100, 200 adaptation messages (5 stratified subsamples each, seeds 1-5)
and all of them. "Before" = the frozen final svm_char (seed-1 weights of smishfuse-final-3class).
Every model is evaluated on the 63 case-study rows (as written), on the adaptation messages not used
for its training, and on the Mendeley VAL split (macro F1; the test split is not read here except for
the timing texts, as in step 5b). Timing: p50 / p95 on 1000 mendeley_3class test messages, 1 thread.
Output: results/step6b/adapt/runs.csv, predictions.csv, latency.json
Run (Kaggle): python src/step6/adapt.py --queue '["run"]' --worker cpu
"""
from __future__ import annotations

import os

for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[var] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "baselines"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from fusion.envcheck import check_versions  # noqa: E402
from neural.common import RESULTS, write_json  # noqa: E402

OUT = RESULTS / "step6b" / "adapt"
CLASSES = ["ham", "smishing", "spam"]
SIZES, SUBSAMPLES = [50, 100, 200], [1, 2, 3, 4, 5]


def log(msg: str) -> None:
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--worker", required=True)
    args = ap.parse_args()
    check_versions(args.worker, OUT / "environment")
    from final import latency, run

    tr_ids, tr_texts, tr_y = run.data("mendeley_3class", "train")
    _, va_texts, va_y = run.data("mendeley_3class", "val")
    _, te_texts, _ = run.data("mendeley_3class", "test")  # timing texts only
    cases = pd.read_csv(ROOT / "data" / "kz_cases" / "cases.csv", dtype=str, keep_default_na=False)
    adapt = pd.read_csv(ROOT / "data" / "kz_adapt" / "adapt.csv", dtype=str, keep_default_na=False)
    adapt["y"] = adapt["label"].map(CLASSES.index)

    runs, preds = [], []

    def evaluate(run_id, proba, held_out: pd.DataFrame):
        val_pred = proba(va_texts).argmax(1)
        f1 = float(f1_score(va_y, val_pred, labels=[0, 1, 2], average="macro", zero_division=0))
        for name, df in (("cases", cases), ("adapt_held_out", held_out)):
            if df.empty:
                continue
            p = proba(np.asarray(df["text"].tolist(), dtype=object))
            for (_, r), pr in zip(df.iterrows(), p):
                preds.append({"run_id": run_id, "set": name, "id": r["id"], "label": r["label"],
                              "pred": CLASSES[int(pr.argmax())], "confidence": float(pr.max()),
                              **{f"prob_{c}": float(pr[i]) for i, c in enumerate(CLASSES)}})
        return f1

    # before: the frozen final model
    path = latency.find("mendeley_3class", "svm_char")
    if path is None:
        raise SystemExit("final svm_char weights not found (kernel source smishfuse-final-3class)")
    import joblib

    before = joblib.load(path / "model.joblib")
    f1 = evaluate("before", lambda t: before.predict_proba(np.asarray(list(t), dtype=object)), adapt)
    runs.append({"run_id": "before", "condition": "before", "size": 0, "subsample": 0, "n_adapt": 0, "val_macro_f1": f1})
    log(f"[adapt] before: val macro F1 {f1:.4f}")

    for cond, pool in (("A", adapt[adapt.same_category_as_test == "false"]), ("B", adapt)):
        plans = [(n, k) for n in SIZES if n < len(pool) for k in SUBSAMPLES] + [(len(pool), 0)]
        for n, k in plans:
            sub = pool if n == len(pool) else pool.loc[
                train_test_split(pool.index, train_size=n, stratify=pool["y"], random_state=k)[0]]
            held = adapt.drop(index=sub.index)
            m = run.Classic("svm_char", "mendeley_3class").fit(
                np.concatenate([tr_ids, sub["id"].to_numpy(dtype=object)]),
                np.concatenate([tr_texts, sub["text"].to_numpy(dtype=object)]),
                np.concatenate([tr_y, sub["y"].to_numpy()]), 1)
            run_id = f"{cond}_n{n}_s{k}"
            f1 = evaluate(run_id, m.proba, held)
            runs.append({"run_id": run_id, "condition": cond, "size": n, "subsample": k, "n_adapt": len(sub),
                         "n_held_out": len(held), "val_macro_f1": f1})
            log(f"[adapt] {run_id}: val macro F1 {f1:.4f}")
            if cond == "B" and k == 0:
                adapted = m
    pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False, lineterminator="\n")
    pd.DataFrame(preds).to_csv(OUT / "predictions.csv", index=False, lineterminator="\n")

    texts = list(te_texts)
    lat = {}
    for name, fn in (("before", lambda t: before.predict_proba(np.asarray([t], dtype=object))),
                     ("adapted_B_all", lambda t: adapted.proba([t]))):
        lat[name] = latency.summary(latency.timings(fn, texts))
    write_json(lat, OUT / "latency.json")
    log(f"[adapt] latency {lat}")


if __name__ == "__main__":
    main()
