"""Steps 7-8: stratified 70/10/20 split (seed 42) of the deduplicated data,
then a leakage check for near-duplicates across train/val/test.

The leakage check re-runs near-duplicate search with a looser LSH candidate
threshold (0.5) than dedup.py (0.7), so it is not just repeating the same search;
candidates are still verified with exact Jaccard >= 0.9.

Stratification is always by the original `label` (3 classes for Mendeley);
extra columns such as Mendeley's label_binary are carried over unchanged.
UCI is used only for overlap and is not split.
Outputs: data/processed/<name>/{train,val,test}.csv, reports/stats/split_<name>.json
Run    : python src/data/split.py
"""
import sys

from sklearn.model_selection import train_test_split

from common import PROCESSED, SEED, STATS, normalize, read_csv, write_csv, write_json
from neardup import Index, self_pairs

DATASETS = ["mendeley", "moz_smishing", "super_sms", "super_sms_clean"]
FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}
CHECK_LSH_THRESHOLD = 0.5


def split(name: str) -> dict:
    df = read_csv(PROCESSED / name / "clean.csv").sort_values("id", ignore_index=True)
    columns = list(df.columns)
    rest, test = train_test_split(df, test_size=FRACTIONS["test"], stratify=df["label"], random_state=SEED)
    val_share = FRACTIONS["val"] / (FRACTIONS["train"] + FRACTIONS["val"])
    train, val = train_test_split(rest, test_size=val_share, stratify=rest["label"], random_state=SEED)
    parts = {"train": train, "val": val, "test": test}

    # Leakage check over all rows of the clean dataset.
    split_of = {}
    for part, d in parts.items():
        for i in d["id"]:
            split_of[i] = part
    df["norm"] = df["text"].map(normalize)
    uniq = sorted(df["norm"].unique())
    uidx = {t: k for k, t in enumerate(uniq)}
    rows_by_u = df.groupby(df["norm"].map(uidx))["id"].apply(list).to_dict()
    pairs = self_pairs(Index(uniq, lsh_threshold=CHECK_LSH_THRESHOLD))
    exact_left = sum(len(v) > 1 for v in rows_by_u.values())
    cross, within = [], 0
    for i, j, jac in pairs:
        for a in rows_by_u[i]:
            for b in rows_by_u[j]:
                if split_of[a] != split_of[b]:
                    cross.append({"id_a": a, "split_a": split_of[a], "id_b": b, "split_b": split_of[b], "jaccard": jac})
                else:
                    within += 1

    for part, d in parts.items():
        write_csv(d.sort_values("id")[columns], PROCESSED / name / f"{part}.csv")

    stats = {
        "dataset": name,
        "seed": SEED,
        "fractions": FRACTIONS,
        "columns": columns,
        "sizes": {p: len(d) for p, d in parts.items()},
        "labels": {p: d["label"].value_counts().sort_index().to_dict() for p, d in parts.items()},
        **({"label_binary": {p: d["label_binary"].value_counts().sort_index().to_dict() for p, d in parts.items()}}
           if "label_binary" in columns else {}),
        "leakage_check": {
            "lsh_candidate_threshold": CHECK_LSH_THRESHOLD,
            "jaccard_threshold": 0.9,
            "exact_duplicate_texts_remaining": exact_left,
            "near_pairs_found_in_clean_data": len(pairs),
            "near_pairs_across_splits": len(cross),
            "near_pairs_within_split": within,
            "cross_split_pairs": cross[:50],
        },
    }
    write_json(stats, STATS / f"split_{name}.json")
    print(f"[{name}] sizes {stats['sizes']}; leakage: {len(cross)} cross-split near pairs, "
          f"{exact_left} exact duplicate texts, {len(pairs)} near pairs in clean data")
    return stats


def main():
    failed = [name for name in DATASETS if split(name)["leakage_check"]["near_pairs_across_splits"]]
    if failed:
        sys.exit(f"near-duplicates across splits in: {failed} (see reports/stats/split_<name>.json)")


if __name__ == "__main__":
    main()
