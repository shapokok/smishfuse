"""Fixed, seeded character-level obfuscation of SMS texts (step 4b stress test).

Three operations, applied to a share `level` of the non-whitespace characters of a message:
* substitution inside the groups {0, o, O}, {1, l, I}, {a, A, @}, {e, E, 3}, {s, S, $}
  (a character is replaced by a different member of its group; letters map to the
  look-alike digit / symbol and back);
* Cyrillic homoglyph: a e o p c A E O P C -> а е о р с А Е О Р С;
* space inserted inside a word (before a character whose left neighbour is a letter or digit).
Positions are drawn without replacement among the characters where at least one operation
applies; the operation is drawn uniformly among those applicable at that position.
The random stream depends only on (SEED, level, message id), so every model sees the same
obfuscated text and the result does not depend on the order of the messages.
"""
from __future__ import annotations

import hashlib

import numpy as np

SEED = 2026
GROUPS = [["0", "o", "O"], ["1", "l", "I"], ["a", "A", "@"], ["e", "E", "3"], ["s", "S", "$"]]
SUBST = {c: [d for d in g if d != c] for g in GROUPS for c in g}
HOMOGLYPH = dict(zip("aeopcAEOPC", "аеорсАЕОРС"))


def _rng(message_id: str, level: float) -> np.random.Generator:
    h = int(hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng([SEED, int(round(level * 10000)), h])


def _ops(text: str, i: int) -> list[str]:
    ops = []
    if text[i] in SUBST:
        ops.append("subst")
    if text[i] in HOMOGLYPH:
        ops.append("homoglyph")
    if i > 0 and text[i - 1].isalnum() and text[i].isalnum():
        ops.append("space")
    return ops


def obfuscate(text: str, message_id: str, level: float) -> tuple[str, int]:
    """Obfuscated text and the number of operations applied."""
    if level <= 0 or not text:
        return text, 0
    rng = _rng(message_id, level)
    eligible = [i for i in range(len(text)) if _ops(text, i)]
    n_chars = sum(not c.isspace() for c in text)
    n_ops = min(int(round(level * n_chars)), len(eligible))
    chosen = set(rng.choice(eligible, size=n_ops, replace=False).tolist()) if n_ops else set()
    out = []
    for i, c in enumerate(text):
        if i not in chosen:
            out.append(c)
            continue
        ops = _ops(text, i)
        op = ops[int(rng.integers(len(ops)))]
        if op == "subst":
            alts = SUBST[c]
            out.append(alts[int(rng.integers(len(alts)))])
        elif op == "homoglyph":
            out.append(HOMOGLYPH[c])
        else:
            out.append(" " + c)
    return "".join(out), n_ops


def obfuscate_all(texts, ids, level: float) -> tuple[list[str], dict]:
    """Obfuscated texts and statistics: realised share of changed characters, operations per type."""
    out, n_ops, n_chars = [], 0, 0
    for t, i in zip(texts, ids):
        o, k = obfuscate(t, i, level)
        out.append(o)
        n_ops += k
        n_chars += sum(not c.isspace() for c in t)
    return out, {"level": level, "n_messages": len(out), "n_ops": n_ops, "n_chars": n_chars,
                 "realised_share": n_ops / n_chars if n_chars else 0.0,
                 "sha256": hashlib.sha256("\x1f".join(out).encode("utf-8")).hexdigest()}
