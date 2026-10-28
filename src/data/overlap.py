"""Step 5: overlap between datasets, same method as dedup.py
(normalized exact match OR Jaccard(char 5-grams) >= 0.9).

Matrix cell [A][B] = number of messages of A that have at least one match in B
(asymmetric: duplicates inside B make the two directions differ).
Computed on the deduplicated data (clean.csv). Mendeley vs UCI is additionally
computed on the data before deduplication (unified.csv), for comparison with
the figure reported in the literature.

Outputs: reports/overlap_matrix.csv, reports/overlap_matrix_exact.csv,
         reports/overlap_mendeley_uci_pairs.csv, reports/stats/overlap.json
Run    : python src/data/overlap.py
"""
import pandas as pd

from common import PROCESSED, REPORTS, STATS, normalize, read_csv, write_csv, write_json
from neardup import Index, cross_pairs

DATASETS = ["mendeley", "moz_smishing", "super_sms", "super_sms_clean", "uci_sms"]


class Prepared:
    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("id", ignore_index=True)
        self.df["norm"] = self.df["text"].map(normalize)
        self.uniq = sorted(self.df["norm"].unique())
        self.uidx = {t: i for i, t in enumerate(self.uniq)}
        self.df["u"] = self.df["norm"].map(self.uidx)
        self.index = Index(self.uniq)


def matched(a: Prepared, b: Prepared):
    """Unique-text indices of a with a match in b: (exact set, exact-or-near set, verified pairs)."""
    exact = {a.uidx[t] for t in a.uniq if t in b.uidx}
    pairs = cross_pairs(a.index, b.index)
    return exact, exact | {i for i, _, _ in pairs}, pairs


def count_rows(p: Prepared, uset: set) -> int:
    return int(p.df["u"].isin(uset).sum())


def mendeley_uci_detail(m: Prepared, u: Prepared, tag: str) -> dict:
    exact, anym, pairs = matched(m, u)
    exact_rev, any_rev, _ = matched(u, m)
    # Best UCI match per Mendeley unique text: highest J, then smallest UCI unique index.
    best = {}
    for i, j, jac in pairs:
        if i not in best or (jac, -j) > (best[i][1], -best[i][0]):
            best[i] = (j, jac)
    first_uci = u.df.groupby("u").first()
    rows = []
    for r in m.df.itertuples():
        if r.u in best:
            j, jac = best[r.u]
            rows.append({"mendeley_id": r.id, "mendeley_label": r.label, "uci_id": first_uci.loc[j, "id"],
                         "uci_label": first_uci.loc[j, "label"], "jaccard": round(jac, 4),
                         "exact_normalized": r.u in exact, "mendeley_text": r.text,
                         "uci_text": first_uci.loc[j, "text"]})
    pairs_df = pd.DataFrame(rows)
    crosstab = (pairs_df.groupby(["mendeley_label", "uci_label"]).size().rename("n").reset_index().to_dict("records")
                if len(pairs_df) else [])
    return {
        "data": tag,
        "mendeley_rows": len(m.df),
        "uci_rows": len(u.df),
        "mendeley_rows_in_uci_exact": count_rows(m, exact),
        "mendeley_rows_in_uci_exact_or_near": count_rows(m, anym),
        "uci_rows_in_mendeley_exact": count_rows(u, exact_rev),
        "uci_rows_in_mendeley_exact_or_near": count_rows(u, any_rev),
        "label_crosstab_mendeley_vs_best_uci_match": crosstab,
    }, pairs_df


def main():
    prepared = {}
    for name in DATASETS:
        print(f"[overlap] indexing {name} (clean)")
        prepared[name] = Prepared(read_csv(PROCESSED / name / "clean.csv"))

    any_m = pd.DataFrame(0, index=DATASETS, columns=DATASETS, dtype=int)
    exact_m = any_m.copy()
    for a in DATASETS:
        any_m.loc[a, a] = exact_m.loc[a, a] = len(prepared[a].df)
        for b in DATASETS:
            if a == b:
                continue
            print(f"[overlap] {a} -> {b}")
            exact, anym, _ = matched(prepared[a], prepared[b])
            exact_m.loc[a, b] = count_rows(prepared[a], exact)
            any_m.loc[a, b] = count_rows(prepared[a], anym)

    write_csv(any_m.rename_axis("dataset").reset_index(), REPORTS / "overlap_matrix.csv")
    write_csv(exact_m.rename_axis("dataset").reset_index(), REPORTS / "overlap_matrix_exact.csv")

    print("[overlap] mendeley vs uci_sms (clean and before dedup)")
    clean_stats, clean_pairs = mendeley_uci_detail(prepared["mendeley"], prepared["uci_sms"], "clean")
    raw_stats, _ = mendeley_uci_detail(
        Prepared(read_csv(PROCESSED / "mendeley" / "unified.csv")),
        Prepared(read_csv(PROCESSED / "uci_sms" / "unified.csv")),
        "before_dedup",
    )
    write_csv(clean_pairs, REPORTS / "overlap_mendeley_uci_pairs.csv")

    write_json(
        {"matrix_exact_or_near": any_m.to_dict(orient="index"), "matrix_exact": exact_m.to_dict(orient="index"),
         "mendeley_uci": [raw_stats, clean_stats]},
        STATS / "overlap.json",
    )
    print(any_m)
    print({k: v for k, v in raw_stats.items() if k != "label_crosstab_mendeley_vs_best_uci_match"})
    print({k: v for k, v in clean_stats.items() if k != "label_crosstab_mendeley_vs_best_uci_match"})


if __name__ == "__main__":
    main()
