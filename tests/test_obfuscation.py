"""Seeded obfuscation (src/eval/obfuscation.py) on hand-written strings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.obfuscation import HOMOGLYPH, SUBST, obfuscate, obfuscate_all  # noqa: E402

TEXT = "Your account is suspended, verify at secure-bank.com or call 08001234567 now"


def test_deterministic_and_order_free():
    a, _ = obfuscate(TEXT, "id_1", 0.2)
    b, _ = obfuscate(TEXT, "id_1", 0.2)
    assert a == b
    t1, _ = obfuscate_all([TEXT, "hello there"], ["id_1", "id_2"], 0.2)
    t2, _ = obfuscate_all(["hello there", TEXT], ["id_2", "id_1"], 0.2)
    assert t1 == t2[::-1]


def test_level_and_operations():
    n_chars = sum(not c.isspace() for c in TEXT)
    for level in [0.05, 0.1, 0.2]:
        out, k = obfuscate(TEXT, "id_1", level)
        assert k == round(level * n_chars)
        # every change is an allowed substitution, homoglyph or inserted space
        assert out.replace(" ", "") != TEXT.replace(" ", "") or k == 0
        assert len(out) - len(TEXT) <= k
    assert obfuscate(TEXT, "id_1", 0.0) == (TEXT, 0)


def test_only_allowed_characters_appear():
    out, _ = obfuscate(TEXT, "id_3", 0.2)
    allowed = set(TEXT) | {" "} | set(HOMOGLYPH.values()) | {d for v in SUBST.values() for d in v}
    assert set(out) <= allowed


def test_step4b_obfuscation_unchanged():
    """The step-4b mix reproduces the recorded hashes of the obfuscated val (needs data/processed)."""
    import json

    import pandas as pd
    import pytest

    root = Path(__file__).resolve().parents[1]
    val = root / "data" / "processed" / "mendeley" / "val.csv"
    levels = sorted((root / "results" / "step4b" / "obfuscation" / "levels").glob("*.json"))
    if not val.exists() or not levels:
        pytest.skip("data/processed or step-4b results not available")
    df = pd.read_csv(val, dtype=str, keep_default_na=False)
    for f in levels:
        rec = json.loads(f.read_text())
        _, stats = obfuscate_all(df["text"].tolist(), df["id"].tolist(), rec["level"])
        assert stats["sha256"] == rec["sha256"]


def test_type_groups():
    from eval.obfuscation import HELDOUT_TYPES, TRAIN_TYPES, VOWELS, augment

    out, _ = obfuscate(TEXT, "id_4", 0.2, TRAIN_TYPES)
    assert not set(out) & set(HOMOGLYPH.values())          # no Cyrillic in TRAIN types
    held, k = obfuscate(TEXT, "id_4", 0.2, HELDOUT_TYPES)
    assert k > 0 and "$" not in held and "@" not in held   # no TRAIN substitutions
    assert held.count(" ") == TEXT.count(" ")              # no inserted spaces
    vowels_removed = sum(c in VOWELS for c in TEXT) - sum(c in VOWELS for c in held)
    assert vowels_removed >= 0
    a = augment([TEXT] * 3, ["x", "y", "z"], seed=1, epoch=0)
    assert a == augment([TEXT] * 3, ["x", "y", "z"], seed=1, epoch=0)
    assert a != augment([TEXT] * 3, ["x", "y", "z"], seed=1, epoch=1) or a == [TEXT] * 3
