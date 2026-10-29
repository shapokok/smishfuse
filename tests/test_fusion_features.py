"""Hand-crafted SMS features (src/fusion/features.py) on hand-written strings."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fusion.features import FEATURES, Standardizer, extract  # noqa: E402


def feats(text: str) -> dict:
    return dict(zip(FEATURES, extract(text)))


def test_urls_phones_codes():
    f = feats("URGENT! You WON a £1000 prize. Claim at bit.ly/abc12 or call 08712345678. Reply STOP to 85023")
    assert f["n_urls"] == 1 and f["has_short_url"] == 1.0
    assert f["has_phone"] == 1.0 and f["has_short_code"] == 1.0
    assert f["n_currency"] == 1 and f["n_exclaim"] == 1
    assert f["n_urgency"] == 1  # urgent
    assert f["n_prize"] == 3    # won, prize, claim


def test_email_is_not_url():
    f = feats("Verify at http://secure-bank.com/login or email help@bank.co.uk")
    assert f["n_urls"] == 1 and f["has_email"] == 1.0 and f["has_short_url"] == 0.0


def test_plain_message():
    f = feats("Hey r u comin tonite? see u at 7")
    assert f["n_chars"] == 32 and f["n_words"] == 9
    assert f["has_phone"] == 0.0 and f["has_short_code"] == 0.0 and f["n_urls"] == 0
    assert f["n_misspelled"] == 2  # comin, tonite (one-letter tokens are skipped)


def test_empty_and_standardizer():
    assert all(v == 0 for v in extract(""))
    X = np.array([[1.0, 5.0], [3.0, 5.0]])
    Z = Standardizer().fit(X).transform(X)
    assert np.allclose(Z[:, 0], [-1, 1]) and np.allclose(Z[:, 1], 0)
