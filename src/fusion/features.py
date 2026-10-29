"""Hand-crafted SMS features for the SmishFuse feature branch (step 4a).

Every feature is computed from the raw message text alone; word lists and URL
shortener domains come from configs/step4a_lexicon.yaml (fixed before training).
Standardization (mean / std of train) is done by `Standardizer`.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.(?:com|net|org|co|uk|in|info|biz|"
                    r"ly|gl|gd|me|io|us|cc|tk|xyz|ru|de|nl|fr|it|es|ac|mobi|tv)\b(?:/\S*)?", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# phone: a run of digits with optional + / spaces / dashes / dots / brackets holding at least 7 digits
PHONE_RE = re.compile(r"\+?\(?\d[\d\s\-.()]{5,}\d")
SHORT_CODE_RE = re.compile(r"(?<![\d.,/:£$€])\b\d{4,6}\b(?![\d.,/:])")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

FEATURES = ["n_chars", "n_words", "n_urls", "has_short_url", "has_phone", "has_short_code", "has_email",
            "upper_ratio", "digit_ratio", "n_currency", "n_exclaim", "n_misspelled", "n_urgency", "n_prize"]


@lru_cache(maxsize=1)
def lexicon(path: str = "configs/step4a_lexicon.yaml") -> dict:
    lex = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))

    def pattern(words):
        alts = sorted((re.escape(w.lower()).replace(r"\ ", r"\s+") for w in words), key=len, reverse=True)
        return re.compile(r"(?<![\w'])(?:" + "|".join(alts) + r")(?![\w'])", re.IGNORECASE)

    return {"urgency": pattern(lex["urgency"]), "prize": pattern(lex["prize"]),
            "short": [d.lower() for d in lex["short_url_domains"]], "currency": set(lex["currency_symbols"])}


@lru_cache(maxsize=1)
def spellchecker():
    from spellchecker import SpellChecker

    return SpellChecker(language="en")


def n_phones(text: str) -> int:
    return sum(len(re.sub(r"\D", "", m.group())) >= 7 for m in PHONE_RE.finditer(text))


def extract(text: str) -> list[float]:
    lex = lexicon()
    no_email = EMAIL_RE.sub(" ", text)  # e-mail domains are not URLs
    urls = URL_RE.findall(no_email)
    no_links = URL_RE.sub(" ", no_email)
    letters = [c for c in text if c.isalpha()]
    # short codes: 4-6 digit numbers outside phone numbers, URLs and e-mails
    no_phone = PHONE_RE.sub(lambda m: " " if len(re.sub(r"\D", "", m.group())) >= 7 else m.group(), no_links)
    words = [w.lower() for w in WORD_RE.findall(no_links) if len(w) >= 2]
    misspelled = spellchecker().unknown(words) if words else set()
    return [
        len(text),
        len(text.split()),
        len(urls),
        float(any(any(d in u.lower() for d in lex["short"]) for u in urls)),
        float(n_phones(no_links) > 0),
        float(bool(SHORT_CODE_RE.search(no_phone))),
        float(bool(EMAIL_RE.search(text))),
        sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        sum(c.isdigit() for c in text) / len(text) if text else 0.0,
        sum(c in lex["currency"] for c in text),
        text.count("!"),
        sum(w in misspelled for w in words),
        len(lex["urgency"].findall(text)),
        len(lex["prize"].findall(text)),
    ]


def extract_all(texts) -> np.ndarray:
    return np.asarray([extract(t) for t in texts], dtype=np.float64)


class Standardizer:
    """(x - mean) / std with mean and std of train; constant columns get std 1."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_
