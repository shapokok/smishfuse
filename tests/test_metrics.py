import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from eval.metrics import compute_metrics, expected_calibration_error, recall_at_fpr

CLASSES3 = ["ham", "smishing", "spam"]


@pytest.fixture
def three_class():
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 0, 1])
    probs = np.array([
        [0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.3, 0.3, 0.4],
        [0.1, 0.2, 0.7], [0.2, 0.2, 0.6], [0.5, 0.1, 0.4], [0.9, 0.05, 0.05], [0.2, 0.6, 0.2],
    ])
    return y, probs


def test_three_class_matches_sklearn(three_class):
    y, probs = three_class
    pred = probs.argmax(1)
    m = compute_metrics(y, probs, CLASSES3)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert m["weighted_f1"] == pytest.approx(f1_score(y, pred, average="weighted"))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    for i, c in enumerate(CLASSES3):
        assert m[f"precision_{c}"] == pytest.approx(precision_score(y, pred, labels=[i], average="macro"))
        assert m[f"recall_{c}"] == pytest.approx(recall_score(y, pred, labels=[i], average="macro"))
        assert m[f"pr_auc_{c}"] == pytest.approx(average_precision_score(y == i, probs[:, i]))
    assert m["pr_auc"] == pytest.approx(np.mean([m[f"pr_auc_{c}"] for c in CLASSES3]))
    assert "recall_at_fpr_1pct" not in m


def test_three_class_hand_values(three_class):
    # pred = [0,0,1,1,2,2,2,0,0,1]; correct at 0,1,3,5,6,8,9 -> 7/10
    y, probs = three_class
    m = compute_metrics(y, probs, CLASSES3)
    assert m["accuracy"] == pytest.approx(0.7)
    # ham: tp=3 (idx 0,1,8), fp=1 (idx 7), fn=1 (idx 2) -> P=R=F1=0.75
    assert m["precision_ham"] == pytest.approx(0.75)
    assert m["recall_ham"] == pytest.approx(0.75)
    # spam: tp=2 (5,6), fp=1 (4), fn=1 (7) -> 2/3
    assert m["f1_spam"] == pytest.approx(2 / 3)


def test_binary_metrics():
    y = np.array([0, 0, 1, 1, 1, 0])
    p1 = np.array([0.1, 0.6, 0.8, 0.4, 0.9, 0.2])
    probs = np.stack([1 - p1, p1], axis=1)
    m = compute_metrics(y, probs, ["benign", "malicious"], positive="malicious")
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p1))
    assert m["macro_f1"] == pytest.approx(f1_score(y, (p1 >= 0.5).astype(int), average="macro"))
    assert m["recall_malicious"] == pytest.approx(2 / 3)
    assert m["precision_malicious"] == pytest.approx(2 / 3)
    # FPR <= 1% means no negative may be above the threshold: highest negative is 0.6,
    # positives above it are 0.8 and 0.9 -> recall 2/3.
    assert m["recall_at_fpr_1pct"] == pytest.approx(2 / 3)


def test_perfect_predictions():
    y = np.array([0, 1, 1, 0])
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    m = compute_metrics(y, probs, ["benign", "malicious"], positive="malicious")
    for key in ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "pr_auc", "recall_at_fpr_1pct"]:
        assert m[key] == pytest.approx(1.0)
    assert m["ece"] == pytest.approx(0.0)


def test_ece_one_sample_per_bin():
    # Confidences 0.9 (correct), 0.8 (wrong), 0.7 (correct), 0.6 (wrong) fall in 4 different bins.
    y = np.array([0, 1, 1, 1])
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    expected = (abs(1 - 0.9) + abs(0 - 0.8) + abs(1 - 0.7) + abs(0 - 0.6)) / 4
    assert expected_calibration_error(y, probs, 15) == pytest.approx(expected)


def test_ece_single_bin():
    # All confidences 0.95, accuracy 3/4 -> |0.75 - 0.95| = 0.2
    y = np.array([0, 0, 0, 1])
    probs = np.tile([0.95, 0.05], (4, 1))
    assert expected_calibration_error(y, probs, 15) == pytest.approx(0.2)


def test_ece_bin_edge_goes_to_lower_bin():
    # 0.6 = 9/15 is the right edge of bin 8 (0.5333, 0.6]; 0.62 is in bin 9.
    y = np.array([0, 1])
    probs = np.array([[0.6, 0.4], [0.62, 0.38]])
    # Separate bins: |1 - 0.6| / 2 + |0 - 0.62| / 2
    assert expected_calibration_error(y, probs, 15) == pytest.approx((0.4 + 0.62) / 2)


def test_recall_at_fpr_hand_example():
    # 100 negatives with scores 0.00..0.99, 10 positives: 5 at 2.0, 5 at 0.5.
    neg = np.arange(100) / 100
    pos = np.array([2.0] * 5 + [0.5] * 5)
    y = np.r_[np.zeros(100), np.ones(10)]
    s = np.r_[neg, pos]
    # One negative (0.99) above the threshold is allowed at FPR 1%; positives at 0.5 need 50 negatives.
    assert recall_at_fpr(y, s, 0.01) == pytest.approx(0.5)
    assert recall_at_fpr(y, s, 0.5) == pytest.approx(1.0)
