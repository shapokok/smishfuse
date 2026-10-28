"""Mendeley SMS Phishing Dataset (DOI 10.17632/f45bkkt8pr.1) -> unified schema.

Run: python src/data/load_mendeley.py
"""
import io
import zipfile

import pandas as pd

from common import fetch, finalize_load

NAME = "mendeley"
# From the dataset description on Mendeley Data.
EXPECTED = {"total": 5971, "labels": {"ham": 4844, "spam": 489, "smishing": 638}}
LABEL_MAP = {"ham": "ham", "spam": "spam", "smishing": "smishing"}


def main():
    path = fetch(NAME)
    with zipfile.ZipFile(path) as zf:
        data = zf.read("Dataset_5971.csv")
    raw = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8")

    df = pd.DataFrame(
        {
            "id": [f"{NAME}_{i:05d}" for i in range(len(raw))],
            "text": raw["TEXT"],
            "label_raw": raw["LABEL"],
        }
    )
    df["label"] = df["label_raw"].str.strip().str.lower().map(LABEL_MAP)

    bad_label = df["label"].isna()
    empty = df["text"].str.strip() == ""
    invalid = pd.concat(
        [
            df[bad_label].assign(reason="unknown_label", label=lambda d: d["label_raw"]),
            df[~bad_label & empty].assign(reason="empty_text"),
        ]
    )
    valid = df[~bad_label & ~empty]

    finalize_load(
        NAME,
        valid[["id", "text", "label"]],
        invalid[["id", "text", "label", "reason"]],
        EXPECTED,
        extra_stats={
            "raw_label_values": raw["LABEL"].value_counts().sort_index().to_dict(),
            "rows_with_U+FFFD": int(raw["TEXT"].str.contains("�", regex=False).sum()),
            "rows_with_leading_or_trailing_whitespace": int((raw["TEXT"] != raw["TEXT"].str.strip()).sum()),
            "rows_starting_with_tab": int(raw["TEXT"].str.startswith("\t").sum()),
        },
    )


if __name__ == "__main__":
    main()
