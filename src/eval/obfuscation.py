"""Fixed, seeded character-level obfuscation of SMS texts (steps 4b and 4c).

Operations, applied to a share `level` of the non-whitespace characters of a message:
* subst        replacement inside the groups {0, o, O}, {1, l, I}, {a, A, @}, {e, E, 3}, {s, S, $}
               (a character becomes a different member of its group);
* homoglyph    Cyrillic look-alike: a e o p c A E O P C -> а е о р с А Е О Р С;
* space        a space inserted inside a word (before a character whose left neighbour is
               a letter or digit);
* double       a letter written twice;
* delete_vowel a vowel (a e i o u, either case) removed.
Type groups: TRAIN_TYPES (subst, space) may be used for training augmentation; HELDOUT_TYPES
(homoglyph, double, delete_vowel) only for evaluation; STEP4B_TYPES is the step-4b mix.
Positions are drawn without replacement among the characters where at least one allowed
operation applies; the operation is drawn uniformly among those applicable at that position.
The random stream depends only on (SEED, level, message id[, salt]), so every model sees the
same obfuscated text and the result does not depend on the order of the messages.
"""
from __future__ import annotations

import hashlib

import numpy as np

SEED = 2026
GROUPS = [["0", "o", "O"], ["1", "l", "I"], ["a", "A", "@"], ["e", "E", "3"], ["s", "S", "$"]]
SUBST = {c: [d for d in g if d != c] for g in GROUPS for c in g}
HOMOGLYPH = dict(zip("aeopcAEOPC", "аеорсАЕОРС"))
VOWELS = set("aeiouAEIOU")
TRAIN_TYPES = ("subst", "space")
HELDOUT_TYPES = ("homoglyph", "double", "delete_vowel")
STEP4B_TYPES = ("subst", "homoglyph", "space")


def _hash(message_id: str) -> int:
    return int(hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:16], 16)


def _rng(message_id: str, level: float, salt: tuple = ()) -> np.random.Generator:
    return np.random.default_rng([SEED, int(round(level * 10000)), _hash(message_id), *salt])


def _ops(text: str, i: int, types) -> list[str]:
    # the order of this list is part of the random stream (step-4b results depend on it)
    ops = []
    if "subst" in types and text[i] in SUBST:
        ops.append("subst")
    if "homoglyph" in types and text[i] in HOMOGLYPH:
        ops.append("homoglyph")
    if "space" in types and i > 0 and text[i - 1].isalnum() and text[i].isalnum():
        ops.append("space")
    if "double" in types and text[i].isalpha():
        ops.append("double")
    if "delete_vowel" in types and text[i] in VOWELS:
        ops.append("delete_vowel")
    return ops


def obfuscate(text: str, message_id: str, level: float, types=STEP4B_TYPES, salt: tuple = ()) -> tuple[str, int]:
    """Obfuscated text and the number of operations applied."""
    if level <= 0 or not text:
        return text, 0
    rng = _rng(message_id, level, salt)
    eligible = [i for i in range(len(text)) if _ops(text, i, types)]
    n_chars = sum(not c.isspace() for c in text)
    n_ops = min(int(round(level * n_chars)), len(eligible))
    chosen = set(rng.choice(eligible, size=n_ops, replace=False).tolist()) if n_ops else set()
    out = []
    for i, c in enumerate(text):
        if i not in chosen:
            out.append(c)
            continue
        ops = _ops(text, i, types)
        op = ops[int(rng.integers(len(ops)))]
        if op == "subst":
            alts = SUBST[c]
            out.append(alts[int(rng.integers(len(alts)))])
        elif op == "homoglyph":
            out.append(HOMOGLYPH[c])
        elif op == "space":
            out.append(" " + c)
        elif op == "double":
            out.append(c + c)
        # delete_vowel: nothing is written
    return "".join(out), n_ops


def obfuscate_all(texts, ids, level: float, types=STEP4B_TYPES) -> tuple[list[str], dict]:
    """Obfuscated texts and statistics: realised share of changed characters, operations per type."""
    out, n_ops, n_chars = [], 0, 0
    for t, i in zip(texts, ids):
        o, k = obfuscate(t, i, level, types)
        out.append(o)
        n_ops += k
        n_chars += sum(not c.isspace() for c in t)
    return out, {"level": level, "types": list(types), "n_messages": len(out), "n_ops": n_ops, "n_chars": n_chars,
                 "realised_share": n_ops / n_chars if n_chars else 0.0,
                 "sha256": hashlib.sha256("\x1f".join(out).encode("utf-8")).hexdigest()}


def augment(texts, ids, seed: int, epoch: int, p: float = 0.5, max_level: float = 0.2,
            types=TRAIN_TYPES) -> list[str]:
    """Training augmentation: each message is obfuscated with probability p at a level drawn
    uniformly from [0, max_level]; the draw depends on (message id, seed, epoch)."""
    out = []
    for t, i in zip(texts, ids):
        r = np.random.default_rng([SEED, 1, _hash(i), seed, epoch])
        if r.random() < p:
            out.append(obfuscate(t, i, float(r.uniform(0.0, max_level)), types, salt=(seed, epoch))[0])
        else:
            out.append(t)
    return out
