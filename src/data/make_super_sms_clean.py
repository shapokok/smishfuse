"""Super SMS without any message that matches Mendeley or UCI SMS Spam Collection.

Match = same normalized text OR Jaccard(char 5-grams) >= 0.9, the same method
as dedup.py / overlap.py. The references are Mendeley and UCI *before*
deduplication (unified.csv), so messages removed there still count. LSH
candidates are retrieved in both directions (Super -> reference and
reference -> Super) and every pair is verified with exact Jaccard.

Input  : data/processed/super_sms/clean.csv (Super SMS after within-dataset dedup)
Outputs: data/processed/super_sms_clean/clean.csv
         reports/removed/super_sms_clean_cross.csv   every removed row with its best match
         reports/stats/super_sms_clean.json
The original data/processed/super_sms/ is left untouched.
Run    : python src/data/make_super_sms_clean.py
"""
import pandas as pd

from common import PROCESSED, REMOVED, STATS, read_csv, write_csv, write_json
from neardup import cross_pairs
from overlap import Prepared

NAME = "super_sms_clean"
SOURCE = "super_sms"
REFERENCES = ["mendeley", "uci_sms"]


def best_matches(sup: Prepared, ref: Prepared) -> dict:
    """Super unique-text index -> (reference row id, Jaccard) of its best match."""
    best = {}

    def offer(i, j, jac):
        if i not in best or (jac, -j) > (best[i][1], -best[i][0]):
            best[i] = (j, jac)

    for i, j, jac in cross_pairs(sup.index, ref.index):
        offer(i, j, jac)
    for j, i, jac in cross_pairs(ref.index, sup.index):
        offer(i, j, jac)
    first_id = ref.df.groupby("u")["id"].first()
    return {i: (first_id.loc[j], jac) for i, (j, jac) in best.items()}


def main():
    src = read_csv(PROCESSED / SOURCE / "clean.csv")
    columns = list(src.columns)
    sup = Prepared(src)

    matches = {}
    for ref in REFERENCES:
        print(f"[{NAME}] matching {SOURCE} against {ref} (before dedup)")
        matches[ref] = best_matches(sup, Prepared(read_csv(PROCESSED / ref / "unified.csv")))

    removed = []
    for r in sup.df.itertuples():
        hit = [ref for ref in REFERENCES if r.u in matches[ref]]
        if not hit:
            continue
        row = {"dataset": SOURCE, "id": r.id, "text": r.text, "label": r.label,
               "reason": "match_" + "_and_".join(hit)}
        for ref in REFERENCES:
            ref_id, jac = matches[ref].get(r.u, ("", ""))
            row[f"{ref}_id"] = ref_id
            row[f"{ref}_jaccard"] = round(jac, 4) if jac != "" else ""
        removed.append(row)

    removed_df = pd.DataFrame(
        removed,
        columns=["dataset", "id", "text", "label", "reason"]
        + [f"{ref}_{c}" for ref in REFERENCES for c in ("id", "jaccard")],
    )
    clean = sup.df[~sup.df["id"].isin(set(removed_df["id"]))][columns]

    write_csv(clean, PROCESSED / NAME / "clean.csv")
    write_csv(removed_df, REMOVED / f"{NAME}_cross.csv")
    stats = {
        "dataset": NAME,
        "source": f"data/processed/{SOURCE}/clean.csv",
        "references": {ref: f"data/processed/{ref}/unified.csv" for ref in REFERENCES},
        "rows_before": len(sup.df),
        "rows_after": len(clean),
        "removed": removed_df["reason"].value_counts().to_dict(),
        "removed_with_best_jaccard_1": int(
            pd.concat([removed_df[f"{ref}_jaccard"] == 1.0 for ref in REFERENCES], axis=1).any(axis=1).sum()
        ),
        "removed_by_label": removed_df["label"].value_counts().to_dict(),
        "labels_before": sup.df["label"].value_counts().to_dict(),
        "labels_after": clean["label"].value_counts().to_dict(),
        "mean_len_chars_after": round(float(clean["text"].str.len().mean()), 1),
    }
    write_json(stats, STATS / f"{NAME}.json")
    print(f"[{NAME}] {len(sup.df)} -> {len(clean)} rows; removed {stats['removed']}")


if __name__ == "__main__":
    main()
