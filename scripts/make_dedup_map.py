"""Step 5a: text-free map of the Mendeley deduplication (input of the duplicate-inflation CV on Kaggle).

Reads reports/removed/mendeley_dedup.csv (step 1, gitignored because it holds message texts) and
writes results/step5a/inputs/mendeley_dedup_map.csv with id, reason, group_id, kept_id only.
Run: python scripts/make_dedup_map.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step5a" / "inputs" / "mendeley_dedup_map.csv"


def main():
    removed = pd.read_csv(ROOT / "reports" / "removed" / "mendeley_dedup.csv", dtype=str, keep_default_na=False)
    m = removed[["id", "reason", "group_id", "kept_id"]].sort_values("id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT, index=False, lineterminator="\n")
    print(f"wrote {OUT}: {len(m)} rows, {m['reason'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
