"""Significance tests: paired bootstrap for macro F1, McNemar, Holm correction."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import f1_score


def _f1_indicators(y_true, y_pred, k: int) -> np.ndarray:
    """(n, 3k) 0/1 matrix: per-sample tp, fp, fn indicators for every class."""
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    cols = []
    for c in range(k):
        cols += [(y == c) & (p == c), (y != c) & (p == c), (y == c) & (p != c)]
    return np.stack(cols, axis=1).astype(np.float64)


def weighted_macro_f1(indicators: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """Macro F1 for each row of `weights` (sample multiplicities), zero_division = 0."""
    counts = np.atleast_2d(weights) @ indicators
    tp, fp, fn = counts[:, 0::3], counts[:, 1::3], counts[:, 2::3]
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(axis=1)


def paired_bootstrap_macro_f1(
    y_true, pred_a, pred_b, k: int, n_boot: int = 10000, seed: int = 42, chunk: int = 500
) -> dict:
    """Paired bootstrap of macro F1(a) - macro F1(b).

    Each replicate resamples the n test messages with replacement (drawn as
    multinomial multiplicities, which is the same distribution) and evaluates
    both models on the same resample. CI = 2.5 / 97.5 percentiles. Two-sided
    p-value = min(1, 2 * min((#{d <= 0} + 1) / (B + 1), (#{d >= 0} + 1) / (B + 1))).
    """
    y = np.asarray(y_true)
    n = len(y)
    labels = list(range(k))
    observed = f1_score(y, pred_a, labels=labels, average="macro", zero_division=0) - f1_score(
        y, pred_b, labels=labels, average="macro", zero_division=0
    )
    ind_a = _f1_indicators(y, pred_a, k)
    ind_b = _f1_indicators(y, pred_b, k)
    rng = np.random.default_rng(seed)
    probs = np.full(n, 1.0 / n)
    diffs = []
    for start in range(0, n_boot, chunk):
        w = rng.multinomial(n, probs, size=min(chunk, n_boot - start)).astype(np.float64)
        diffs.append(weighted_macro_f1(ind_a, w, k) - weighted_macro_f1(ind_b, w, k))
    d = np.concatenate(diffs)
    p_le = (np.sum(d <= 0) + 1) / (n_boot + 1)
    p_ge = (np.sum(d >= 0) + 1) / (n_boot + 1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {
        "diff": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, 2 * min(p_le, p_ge))),
        "n_boot": n_boot,
    }


def mcnemar(y_true, pred_a, pred_b, exact_threshold: int = 25) -> dict:
    """McNemar test on paired correctness.

    b = a correct & b wrong, c = a wrong & b correct. If b + c < exact_threshold,
    the exact two-sided binomial test (p = 0.5) is used; otherwise the chi-square
    statistic with continuity correction, (|b - c| - 1)^2 / (b + c), df = 1.
    """
    y = np.asarray(y_true)
    ca = np.asarray(pred_a) == y
    cb = np.asarray(pred_b) == y
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "no_discordant_pairs"}
    if n < exact_threshold:
        return {"b": b, "c": c, "statistic": float(min(b, c)),
                "p_value": float(binomtest(b, n, 0.5, alternative="two-sided").pvalue), "method": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(chi2.sf(stat, 1)), "method": "chi2_cc"}


def holm(p_values, alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm step-down adjusted p-values (monotone, capped at 1) and rejections."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return [], []
    order = np.argsort(p, kind="stable")
    adj_sorted = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[order]))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj.tolist(), (adj <= alpha).tolist()
