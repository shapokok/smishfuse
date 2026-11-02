"""Step 8 (MDPI version), POST-HOC: reading the weights of Lite, Kaggle CPU. No training, no tuning.

Lite = the frozen final svm_char (char TF-IDF 2-5-grams, one-vs-rest LinearSVC, sigmoid calibration with
ensemble=False), seed-1 weights of smishfuse-final-3class (kernel source), as in src/step6/adapt.py.
The class score is s_k(x) = sum_g phi_g(x) w_kg + b_k and the calibrated probability is monotone in s_k when
the Platt slope a_k < 0 (checked), so the coefficients w_kg rank the n-grams by their push towards class k.

Outputs (results/step8/):
  model.json         classes, vectorizer settings, vocabulary size, intercepts, Platt (a_k, b_k), and the largest
                     difference between predict_proba and the probabilities recomputed from these parameters
  top_ngrams.csv     class, rank, ngram, weight: the TOP n-grams with the largest coefficient per class
  test_predictions.csv  id, y_true, y_pred, prob_*: seed-1 predictions on the clean test set (compared locally
                     with results/final/preds/mendeley_3class/svm_char/clean_seed1.csv)
  examples.json      three test messages chosen by a fixed rule (below): id, labels, probabilities, scores
  example_ngrams.csv example, id, ngram, tfidf, weight_<class>, contribution_<class> for every n-gram present
Selection rule (fixed before the run): among the clean test messages of each group (smishing predicted
smishing; spam predicted spam; smishing predicted spam), sort by confidence and then by id and take the
element at position (n - 1) // 2, i.e. the message with the median confidence.
Message texts are not written; the n-grams are fragments of at most 5 characters.
Run (Kaggle): python src/step8/explain.py --queue '["run"]' --worker cpu
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

from fusion.envcheck import check_versions  # noqa: E402
from neural.common import RESULTS, write_json  # noqa: E402

OUT = RESULTS / "step8"
CLASSES = ["ham", "smishing", "spam"]
TOP = 15
GROUPS = {"smishing_correct": ("smishing", "smishing"), "spam_correct": ("spam", "spam"),
          "smishing_as_spam": ("smishing", "spam")}


def log(msg: str) -> None:
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--worker", required=True)
    args = ap.parse_args()
    check_versions(args.worker, OUT / "environment")
    import joblib
    from scipy.special import expit

    from final import latency, run

    path = latency.find("mendeley_3class", "svm_char")
    if path is None:
        raise SystemExit("final svm_char weights not found (kernel source smishfuse-final-3class)")
    clf = joblib.load(path / "model.joblib")
    vec, cal = clf.vectorizer, clf.model
    if len(cal.calibrated_classifiers_) != 1:
        raise SystemExit("expected one calibrated classifier (ensemble=False)")
    cc = cal.calibrated_classifiers_[0]
    svm = cc.estimator
    if list(cal.classes_) != [0, 1, 2]:
        raise SystemExit(f"unexpected classes {cal.classes_}")
    W, b = svm.coef_, svm.intercept_
    a_k = np.array([c.a_ for c in cc.calibrators])
    b_k = np.array([c.b_ for c in cc.calibrators])
    if not (a_k < 0).all():
        raise SystemExit(f"a Platt slope is not negative: {a_k}")
    vocab = np.empty(len(vec.vocabulary_), dtype=object)
    for g, j in vec.vocabulary_.items():
        vocab[j] = g

    ids, texts, y = run.data("mendeley_3class", "test")
    X = vec.transform(texts)
    S = X @ W.T + b
    q = expit(-(S * a_k + b_k))
    P_re = q / q.sum(axis=1, keepdims=True)
    P = clf.predict_proba(texts)
    diff = float(np.abs(P - P_re).max())
    log(f"[explain] recomputed probabilities: max abs difference {diff:.2e}")
    if diff > 1e-6:
        raise SystemExit("probabilities recomputed from the weights differ from predict_proba")

    write_json({"classes": CLASSES, "analyzer": vec.analyzer, "ngram_range": list(vec.ngram_range),
                "lowercase": vec.lowercase, "sublinear_tf": vec.sublinear_tf, "n_features": len(vocab),
                "intercept": b.tolist(), "platt_a": a_k.tolist(), "platt_b": b_k.tolist(),
                "max_abs_diff_recomputed_proba": diff, "weights": "seed 1, smishfuse-final-3class"},
               OUT / "model.json")

    rows = []
    for k, c in enumerate(CLASSES):
        order = np.argsort(-W[k], kind="stable")[:TOP]
        rows += [{"class": c, "rank": r + 1, "ngram": vocab[j], "weight": float(W[k, j])} for r, j in enumerate(order)]
    pd.DataFrame(rows).to_csv(OUT / "top_ngrams.csv", index=False)

    pred = P.argmax(1)
    tp = pd.DataFrame({"id": ids, "y_true": [CLASSES[i] for i in y], "y_pred": [CLASSES[i] for i in pred],
                       **{f"prob_{c}": P[:, k] for k, c in enumerate(CLASSES)}})
    tp.to_csv(OUT / "test_predictions.csv", index=False)

    tp["conf"] = P.max(1)
    examples, grams = {}, []
    for name, (yt, yp) in GROUPS.items():
        g = tp[(tp.y_true == yt) & (tp.y_pred == yp)].sort_values(["conf", "id"]).reset_index(drop=True)
        r = g.iloc[(len(g) - 1) // 2]
        i = int(np.flatnonzero(ids == r.id)[0])
        examples[name] = {"id": r.id, "y_true": yt, "y_pred": yp, "n_in_group": len(g),
                          "prob": {c: float(P[i, k]) for k, c in enumerate(CLASSES)},
                          "score": {c: float(S[i, k]) for k, c in enumerate(CLASSES)}}
        row = X[i].tocoo()
        for j, v in zip(row.col, row.data):
            grams.append({"example": name, "id": r.id, "ngram": vocab[j], "tfidf": float(v),
                          **{f"weight_{c}": float(W[k, j]) for k, c in enumerate(CLASSES)},
                          **{f"contribution_{c}": float(v * W[k, j]) for k, c in enumerate(CLASSES)}})
        log(f"[explain] {name}: {r.id}, group of {len(g)}, confidence {r.conf:.3f}")
    write_json(examples, OUT / "examples.json")
    pd.DataFrame(grams).to_csv(OUT / "example_ngrams.csv", index=False)
    log("[explain] done")


if __name__ == "__main__":
    main()
