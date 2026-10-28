"""Classification metrics for SmishFuse.

Labels are integer class indices 0..K-1; `probs` is an (n, K) array of class
probabilities. Every metric uses the full label set 0..K-1, so a class absent
from a sample still counts (with zero_division=0), as in sklearn with labels=...
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)


def expected_calibration_error(y_true, probs, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width bins over [0, 1].

    Bin b covers (b/n_bins, (b+1)/n_bins]; confidence 0 goes to the first bin.
    ECE = sum_b |acc_b - conf_b| * n_b / n, where conf = max probability and
    acc = whether argmax equals the true label.
    """
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += abs(correct[mask].mean() - conf[mask].mean()) * mask.sum() / len(y)
    return float(ece)


def recall_at_fpr(y_true_binary, scores, max_fpr: float = 0.01) -> float:
    """Highest recall (TPR) reachable with a threshold whose FPR is <= max_fpr."""
    fpr, tpr, _ = roc_curve(np.asarray(y_true_binary), np.asarray(scores), drop_intermediate=False)
    return float(tpr[fpr <= max_fpr + 1e-12].max())


def compute_metrics(
    y_true,
    probs,
    class_names: list[str],
    positive: str | None = None,
    y_pred=None,
    ece_bins: int = 15,
    max_fpr: float = 0.01,
) -> dict:
    """All step-2 metrics as a flat dict.

    positive: name of the positive (malicious) class for binary tasks, else None.
    y_pred defaults to argmax of probs.
    """
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)
    k = len(class_names)
    labels = list(range(k))
    pred = p.argmax(axis=1) if y_pred is None else np.asarray(y_pred)

    out = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, labels=labels, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
    }
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    for i, name in enumerate(class_names):
        out[f"precision_{name}"] = prec[i]
        out[f"recall_{name}"] = rec[i]
        out[f"f1_{name}"] = f1[i]

    if positive is not None:
        pos = class_names.index(positive)
        y_bin = (y == pos).astype(int)
        out["pr_auc"] = average_precision_score(y_bin, p[:, pos])
        out["recall_at_fpr_1pct"] = recall_at_fpr(y_bin, p[:, pos], max_fpr)
    else:
        aps = []
        for i, name in enumerate(class_names):
            ap = average_precision_score((y == i).astype(int), p[:, i])
            out[f"pr_auc_{name}"] = ap
            aps.append(ap)
        out["pr_auc"] = float(np.mean(aps))  # macro one-vs-rest
    out["ece"] = expected_calibration_error(y, p, ece_bins)
    return {k_: float(v) for k_, v in out.items()}
