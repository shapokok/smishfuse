import numpy as np
import pytest
from scipy.stats import chi2
from sklearn.metrics import f1_score

from eval.stats import _f1_indicators, holm, mcnemar, paired_bootstrap_macro_f1, weighted_macro_f1


def _pairs(b: int, c: int, both_right: int = 5, both_wrong: int = 3):
    """y, pred_a, pred_b with exactly b (a right, b wrong) and c (a wrong, b right) pairs."""
    y = np.zeros(b + c + both_right + both_wrong, dtype=int)
    a = np.r_[np.zeros(b), np.ones(c), np.zeros(both_right), np.ones(both_wrong)].astype(int)
    bb = np.r_[np.ones(b), np.zeros(c), np.zeros(both_right), np.ones(both_wrong)].astype(int)
    return y, a, bb


def test_mcnemar_exact():
    y, a, b = _pairs(1, 6)
    r = mcnemar(y, a, b)
    assert (r["b"], r["c"], r["method"]) == (1, 6, "exact")
    # two-sided exact binomial: 2 * P(X <= 1 | n=7, p=0.5) = 2 * (1 + 7) / 128
    assert r["p_value"] == pytest.approx(2 * 8 / 128)


def test_mcnemar_chi2():
    y, a, b = _pairs(30, 10)
    r = mcnemar(y, a, b)
    assert r["method"] == "chi2_cc"
    assert r["statistic"] == pytest.approx((20 - 1) ** 2 / 40)
    assert r["p_value"] == pytest.approx(chi2.sf(19 ** 2 / 40, 1))


def test_mcnemar_no_discordant_pairs():
    y, a, b = _pairs(0, 0)
    assert mcnemar(y, a, b)["p_value"] == 1.0


def test_mcnemar_symmetric():
    y, a, b = _pairs(3, 9)
    assert mcnemar(y, a, b)["p_value"] == pytest.approx(mcnemar(y, b, a)["p_value"])


def test_holm_known_values():
    adj, rej = holm([0.01, 0.04, 0.03], alpha=0.05)
    # sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> monotone 0.06
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert rej == [True, False, False]


def test_holm_cap_and_edge_cases():
    adj, rej = holm([0.5, 0.9])
    assert adj == pytest.approx([1.0, 1.0])
    assert rej == [False, False]
    assert holm([]) == ([], [])
    assert holm([0.02])[0] == pytest.approx([0.02])


def test_weighted_macro_f1_matches_sklearn_sample_weight():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    p = np.where(rng.random(200) < 0.7, y, rng.integers(0, 3, 200))
    w = rng.multinomial(200, np.full(200, 1 / 200), size=3).astype(float)
    got = weighted_macro_f1(_f1_indicators(y, p, 3), w, 3)
    for i in range(3):
        expected = f1_score(y, p, labels=[0, 1, 2], average="macro", sample_weight=w[i], zero_division=0)
        assert got[i] == pytest.approx(expected)


def test_bootstrap_identical_models():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.where(rng.random(300) < 0.8, y, 1 - y)
    r = paired_bootstrap_macro_f1(y, p, p, k=2, n_boot=2000, seed=7)
    assert r["diff"] == 0.0
    assert (r["ci_low"], r["ci_high"]) == (0.0, 0.0)
    assert r["p_value"] == 1.0


def test_bootstrap_clear_difference_and_determinism():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    good = np.where(rng.random(400) < 0.95, y, 1 - y)
    bad = np.where(rng.random(400) < 0.60, y, 1 - y)
    r1 = paired_bootstrap_macro_f1(y, good, bad, k=2, n_boot=2000, seed=7)
    r2 = paired_bootstrap_macro_f1(y, good, bad, k=2, n_boot=2000, seed=7)
    assert r1 == r2
    expected = f1_score(y, good, average="macro") - f1_score(y, bad, average="macro")
    assert r1["diff"] == pytest.approx(expected)
    assert 0 < r1["ci_low"] < r1["diff"] < r1["ci_high"]
    # every replicate favours `good`: p = 2 * 1 / (B + 1)
    assert r1["p_value"] == pytest.approx(2 / 2001)
