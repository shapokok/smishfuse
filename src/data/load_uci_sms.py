"""UCI SMS Spam Collection v.1 -> unified schema.

Used only to measure overlap with Mendeley (step 5); it is not split.
Run: python src/data/load_uci_sms.py
"""
import zipfile

import pandas as pd

from common import fetch, finalize_load

NAME = "uci_sms"
# From the readme shipped in the archive.
EXPECTED = {"total": 5574, "labels": {"ham": 4827, "spam": 747}}


def main():
    path = fetch(NAME)
    with zipfile.ZipFile(path) as zf:
        content = zf.read("SMSSpamCollection").decode("utf-8")
    rows = [line.split("\t", 1) for line in content.split("\n") if line]
    malformed = sum(len(r) != 2 for r in rows)
    if malformed:
        raise RuntimeError(f"{malformed} lines without a tab separator")

    df = pd.DataFrame(
        {
            "id": [f"{NAME}_{i:05d}" for i in range(len(rows))],
            "text": [r[1] for r in rows],
            "label": [r[0] for r in rows],
        }
    )
    bad_label = ~df["label"].isin(["ham", "spam"])
    empty = df["text"].str.strip() == ""
    invalid = pd.concat(
        [df[bad_label].assign(reason="unknown_label"), df[~bad_label & empty].assign(reason="empty_text")]
    )
    valid = df[~bad_label & ~empty]

    finalize_load(NAME, valid, invalid[["id", "text", "label", "reason"]], EXPECTED)


if __name__ == "__main__":
    main()
