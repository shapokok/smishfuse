"""Step 6: the Kazakhstan case-study set (data/kz_cases/cases.csv) from the checked draft.

data/kz_cases/draft.csv holds the 42 author-written messages (7 scenarios x fraud / legit x EN / RU / KK,
checked by the authors) and the RU fraud messages with the frozen HELD-OUT 10% obfuscation
(src/eval/obfuscation.py, variant heldout10). This script adds variant cyr_homoglyph, FOR THE CASE
STUDY ONLY (not part of src/eval/obfuscation.py): in the RU and KK fraud messages 10% of the Cyrillic
letters are replaced by their Latin look-alikes (а→a, е→e, о→o, р→p, с→c, у→y, х→x, к→k and the capitals),
positions drawn without replacement among the letters that have a look-alike; the number of
replacements is round(0.10 x number of Cyrillic letters), capped by the number of such letters.
Random stream: numpy default_rng([SEED, 6, sha256(id)]), SEED of src/eval/obfuscation.py.
Run: python scripts/make_kz_cases.py
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.obfuscation import SEED  # noqa: E402

LOOKALIKE = dict(zip("аеорсухкАЕОРСУХК", "aeopcyxkAEOPCYXK"))
LEVEL = 0.10


def cyr_homoglyph(text: str, message_id: str) -> tuple[str, int]:
    rng = np.random.default_rng([SEED, 6, int(hashlib.sha256(message_id.encode()).hexdigest()[:16], 16)])
    n_cyr = sum("Ѐ" <= c <= "ӿ" for c in text)
    eligible = [i for i, c in enumerate(text) if c in LOOKALIKE]
    n = min(int(round(LEVEL * n_cyr)), len(eligible))
    chosen = set(rng.choice(eligible, size=n, replace=False).tolist()) if n else set()
    return "".join(LOOKALIKE[c] if i in chosen else c for i, c in enumerate(text)), n


def main():
    with open(ROOT / "data" / "kz_cases" / "draft.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = list(rows)
    for r in rows:
        if r["variant"] == "original" and r["pair"] == "fraud" and r["language"] in ("ru", "kk"):
            text, n = cyr_homoglyph(r["text"], r["id"])
            out.append({**r, "id": f"{r['id']}_cyr_homoglyph", "variant": "cyr_homoglyph", "text": text})
            print(f"{r['id']}: {n} letters replaced")
    path = ROOT / "data" / "kz_cases" / "cases.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path}: {len(out)} rows")


if __name__ == "__main__":
    main()
