"""MOZ-Smishing (Ali et al., AfricaNLP 2025; HF MOZNLP/MOZ-Smishing) -> unified schema.

Run: python src/data/load_moz_smishing.py
"""
import pandas as pd

from common import fetch, finalize_load

NAME = "moz_smishing"
# From the Hugging Face dataset card (552 smishing, 2,009 legitimate, 2,561 total).
EXPECTED = {"total": 2561, "labels": {"ham": 2009, "smishing": 552}}
LABEL_MAP = {"legitimate": "ham", "smishing": "smishing"}


def main():
    path = fetch(NAME)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")

    df = pd.DataFrame(
        {
            "id": [f"{NAME}_{i:05d}" for i in range(len(raw))],
            "text": raw["text"],
            "label_raw": raw["label"],
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
            "raw_label_values": raw["label"].value_counts().sort_index().to_dict(),
            "raw_source_values": raw["source"].value_counts().sort_index().to_dict(),
            "raw_ids_unique": bool(raw["id"].is_unique),
        },
    )


if __name__ == "__main__":
    main()
