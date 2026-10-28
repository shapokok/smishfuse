"""Super SMS Dataset (Salman et al., IEEE Access 2024; github.com/smspamresearch/spstudy) -> unified schema.

Run: python src/data/load_super_sms.py
"""
import io

import pandas as pd

from common import decode_utf8_with_cp1252_fallback, fetch, finalize_load

NAME = "super_sms"
# Total: paper body text ("67,018 messages"). Class counts: SpaLLM-Guard
# (Salman et al., arXiv:2501.04985, same authors): 40,837 benign / 26,181 spam.
EXPECTED = {"total": 67018, "labels": {"ham": 40837, "spam": 26181}}
LABEL_MAP = {"0": "ham", "1": "spam"}


def main():
    path = fetch(NAME)
    text, fallback_bytes = decode_utf8_with_cp1252_fallback(path.read_bytes())
    raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)

    df = pd.DataFrame(
        {
            "id": [f"{NAME}_{i:05d}" for i in range(len(raw))],
            "text": raw["SMSes"],
            "label_raw": raw["Labels"],
        }
    )
    df["label"] = df["label_raw"].str.strip().map(LABEL_MAP)

    bad_label = df["label"].isna()
    empty = df["text"].str.strip() == ""
    invalid = pd.concat(
        [
            df[bad_label & ~empty].assign(reason="missing_label", label=lambda d: d["label_raw"]),
            df[empty].assign(reason="empty_text", label=lambda d: d["label_raw"]),
        ]
    ).sort_values("id")
    valid = df[~bad_label & ~empty]

    finalize_load(
        NAME,
        valid[["id", "text", "label"]],
        invalid[["id", "text", "label", "reason"]],
        EXPECTED,
        extra_stats={
            "raw_label_values": raw["Labels"].value_counts().sort_index().to_dict(),
            "non_utf8_bytes_decoded_as_cp1252": fallback_bytes,
            "rows_with_C1_control_chars": int(raw["SMSes"].str.contains("[\x80-\x9f]", regex=True).sum()),
            # Messages that look already preprocessed: only lowercase latin letters, digits and spaces.
            "rows_lowercase_alnum_only": int(raw["SMSes"].str.fullmatch(r"[a-z0-9 ]+").sum()),
        },
    )


if __name__ == "__main__":
    main()
