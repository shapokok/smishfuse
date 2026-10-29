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
