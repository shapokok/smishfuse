"""Steps 4 and 6: within-dataset exact / near-duplicates and label conflicts, then removal.

Duplicate groups are connected components of the graph "same normalized text
OR Jaccard(char 5-grams) >= 0.9". Per group:
  * all members share one label -> keep the member with the smallest id, drop the rest;
  * labels differ, one label has a strict majority -> keep the smallest-id member
    carrying the majority label; drop the other majority members as duplicates
    and the minority members as label_conflict_minority;
  * labels differ, tie for the top count -> drop the whole group (label_conflict_tie).

Inputs : data/processed/<name>/unified.csv
Outputs: data/processed/<name>/clean.csv
         reports/removed/<name>_dedup.csv   every removed row with the reason
         reports/conflicts/<name>.csv       every row of every conflicting group
         reports/stats/dedup_<name>.json
Run    : python src/data/dedup.py            (all datasets)
         python src/data/dedup.py mendeley   (one dataset)
"""
import sys
from collections import Counter

import pandas as pd

from common import CONFLICTS, PROCESSED, REMOVED, STATS, add_extra_columns, normalize, read_csv, write_csv, write_json
from neardup import Index, UnionFind, jaccard, self_pairs

DATASETS = ["mendeley", "moz_smishing", "super_sms", "uci_sms"]


def dedup(name: str) -> None:
    df = read_csv(PROCESSED / name / "unified.csv").sort_values("id", ignore_index=True)
    df["norm"] = df["text"].map(normalize)

    uniq = sorted(df["norm"].unique())
    uidx = {t: i for i, t in enumerate(uniq)}
    df["u"] = df["norm"].map(uidx)

    print(f"[{name}] {len(df)} rows, {len(uniq)} unique normalized texts; searching near-duplicates ...")
    index = Index(uniq)
    pairs = self_pairs(index)

    uf = UnionFind(len(uniq))
    for i, j, _ in pairs:
        uf.union(i, j)
    df["root"] = df["u"].map(uf.find)

    removed, conflicts = [], []
    group_sizes = []
    n_exact_groups = n_near_groups = 0
    n_conflict_groups = Counter()
    resolution = Counter()
    saved_by_label = Counter()
    for gid, (_, g) in enumerate(df.groupby("root", sort=True)):
        if len(g) == 1:
            continue
        group_sizes.append(len(g))
        group_id = f"{name}_g{gid:05d}"
        is_near = g["u"].nunique() > 1
        n_near_groups += is_near
        n_exact_groups += not is_near

        keep_pool = g
        if g["label"].nunique() > 1:
            # Exact conflict: some identical normalized text carries more than one label.
            exact_conflict = (g.groupby("u")["label"].nunique() > 1).any()
            ctype = "exact" if exact_conflict else "near"
            n_conflict_groups[ctype] += 1
            counts = g["label"].value_counts()
            tie = counts.iloc[0] == counts.iloc[1]
            group_labels = "|".join(f"{lab}:{n}" for lab, n in sorted(counts.items()))
            outcome = "tie_removed" if tie else f"majority_{counts.index[0]}"
            resolution["tie" if tie else "majority"] += 1
            for r in g.itertuples():
                conflicts.append(
                    {"group_id": group_id, "conflict_type": ctype, "group_labels": group_labels,
                     "resolution": outcome, "id": r.id, "label": r.label, "text": r.text}
                )
            if tie:
                for r in g.itertuples():
                    removed.append(
                        {"dataset": name, "id": r.id, "text": r.text, "label": r.label,
                         "reason": "label_conflict_tie", "group_id": group_id, "kept_id": "",
                         "jaccard_to_kept": ""}
                    )
                continue
            keep_pool = g[g["label"] == counts.index[0]]
            saved_by_label[counts.index[0]] += 1

        keeper = keep_pool.iloc[0]  # smallest id (df is sorted by id)
        for r in g.itertuples():
            if r.id == keeper["id"]:
                continue
            exact = r.u == keeper["u"]
            if r.label != keeper["label"]:
                reason = "label_conflict_minority"
            else:
                reason = "exact_duplicate" if exact else "near_duplicate"
            removed.append(
                {"dataset": name, "id": r.id, "text": r.text, "label": r.label, "reason": reason,
                 "group_id": group_id, "kept_id": keeper["id"],
                 "jaccard_to_kept": 1.0 if exact else round(jaccard(index.sets[r.u], index.sets[keeper["u"]]), 4)}
            )

    removed_df = pd.DataFrame(
        removed, columns=["dataset", "id", "text", "label", "reason", "group_id", "kept_id", "jaccard_to_kept"]
    )
    conflicts_df = pd.DataFrame(
        conflicts, columns=["group_id", "conflict_type", "group_labels", "resolution", "id", "label", "text"]
    )
    clean = add_extra_columns(name, df[~df["id"].isin(set(removed_df["id"]))])

    write_csv(clean, PROCESSED / name / "clean.csv")
    write_csv(removed_df, REMOVED / f"{name}_dedup.csv")
    write_csv(conflicts_df, CONFLICTS / f"{name}.csv")

    reasons = removed_df["reason"].value_counts().to_dict()
    stats = {
        "dataset": name,
        "rows_before": len(df),
        "rows_after": len(clean),
        # Under the previous rule every conflicting group was dropped entirely, so
        # each group resolved by majority now keeps exactly one more row.
        "rows_after_drop_whole_group_rule": len(clean) - resolution["majority"],
        "rows_saved_vs_drop_whole_group_rule": resolution["majority"],
        "rows_saved_by_label": dict(saved_by_label),
        "unique_normalized_texts": len(uniq),
        "verified_near_pairs_between_unique_texts": len(pairs),
        "duplicate_groups": len(group_sizes),
        "exact_only_groups": int(n_exact_groups),
        "groups_with_near_duplicates": int(n_near_groups),
        "largest_group": max(group_sizes, default=0),
        "removed": {k: int(v) for k, v in reasons.items()},
        "conflict_groups": dict(n_conflict_groups),
        "conflict_resolution": dict(resolution),
        "conflict_rows_by_label": conflicts_df["label"].value_counts().to_dict(),
        "labels_before": df["label"].value_counts().to_dict(),
        "labels_after": clean["label"].value_counts().to_dict(),
        "removed_by_label_and_reason": removed_df.groupby(["reason", "label"]).size().rename("n").reset_index()
        .to_dict("records"),
        "languages_after": clean["language"].value_counts().to_dict(),
        "mean_len_chars_after": round(float(clean["text"].str.len().mean()), 1),
    }
    write_json(stats, STATS / f"dedup_{name}.json")
    print(f"[{name}] removed {reasons}; conflict groups {dict(n_conflict_groups)} resolved {dict(resolution)}; "
          f"{len(df)} -> {len(clean)} rows; largest group {stats['largest_group']}")


def main():
    names = sys.argv[1:] or DATASETS
    for name in names:
        dedup(name)


if __name__ == "__main__":
    main()
