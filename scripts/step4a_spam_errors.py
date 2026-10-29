"""Step 4a.2: error analysis of the `spam` class on mendeley_3class (saved predictions only).

* Confusion matrix of every model (seed 1) and the shares spam->smishing, spam->ham
  (seed 1 and mean over the 5 seeds).
* Hard cases: test messages misclassified by >= HARD_MIN of the 9 models (seed 1).
  reports/step4a_hard_cases.csv holds their text and label (gitignored: raw data
  is not redistributed); results/step4a/hard_cases.csv is the same without text.
* Train neighbours: Jaccard >= JACCARD over character 5-grams of normalized text
  (as in step 1). MinHash LSH candidates (verified exactly) are counted as well,
  but LSH misses pairs near the threshold, so all numbers use the exact full
  search over train (4017 messages). Control group: test messages that all 9
  models classify correctly. Additionally: neighbours with another label inside test.
Outputs: results/step4a/*.csv|json, reports/step4a_hard_cases.csv
Run: python scripts/step4a_spam_errors.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "data"))
from common import NUM_PERM, normalize  # noqa: E402
from eval.registry import MODEL_ORDER, SHORT_NAMES  # noqa: E402
from neardup import jaccard, shingles  # noqa: E402

TASK = "mendeley_3class"
CLASSES = ["ham", "smishing", "spam"]
SEED = 1
HARD_MIN = 5
JACCARD = 0.5
LSH_THRESHOLD = 0.4  # candidate threshold below JACCARD, candidates are verified exactly
RESULTS = ROOT / "results"
OUT = RESULTS / "step4a"


def load_preds(model: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "raw" / TASK / model / f"seed{seed}.csv", dtype={"id": str}, keep_default_na=False)
    return df.set_index("id")[["y_true", "y_pred"]]


def confusion(df: pd.DataFrame) -> pd.DataFrame:
    cols = CLASSES + (["invalid"] if (~df["y_pred"].isin(CLASSES)).any() else [])
    cm = pd.crosstab(df["y_true"], df["y_pred"].where(df["y_pred"].isin(CLASSES), "invalid"))
    return cm.reindex(index=CLASSES, columns=cols, fill_value=0)


def neighbours(query_ids, query_text, query_label, ref: pd.DataFrame, lsh, ref_sets, skip_self=False) -> pd.DataFrame:
    """Neighbours with Jaccard >= JACCARD in `ref`: exact full search (used for all numbers) and MinHash LSH
    candidates verified exactly (recall check of LSH)."""
    lab, rid = ref["label"].to_numpy(), ref["id"].to_numpy()
    rows = []
    for tid in query_ids:
        q = shingles(normalize(query_text[tid]))
        m = MinHash(num_perm=NUM_PERM)
        m.update_batch([s.encode("utf-8") for s in sorted(q)])
        n_lsh = sum(jaccard(q, ref_sets[j]) >= JACCARD for j in lsh.query(m) if not (skip_self and rid[j] == tid))
        hits = [(j, jaccard(q, s)) for j, s in enumerate(ref_sets) if not (skip_self and rid[j] == tid)]
        hits = [(j, jac) for j, jac in hits if jac >= JACCARD]
        same = [jac for j, jac in hits if lab[j] == query_label[tid]]
        other = [(j, jac) for j, jac in hits if lab[j] != query_label[tid]]
        best_other = max(other, key=lambda x: x[1]) if other else None
        rows.append({"id": tid, "n_neighbours": len(hits), "n_neighbours_lsh": n_lsh,
                     "n_same_label": len(same), "n_other_label": len(other),
                     "other_labels": ";".join(sorted({lab[j] for j, _ in other})),
                     "max_jaccard_same": max(same) if same else np.nan,
                     "max_jaccard_other": best_other[1] if best_other else np.nan,
                     "nearest_other_id": rid[best_other[0]] if best_other else "",
                     "nearest_other_label": lab[best_other[0]] if best_other else ""})
    return pd.DataFrame(rows)


def build_lsh(sets):
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    mhs = MinHash.bulk([[s.encode("utf-8") for s in sorted(ss)] for ss in sets], num_perm=NUM_PERM)
    with lsh.insertion_session() as session:
        for j, mh in enumerate(mhs):
            session.insert(j, mh)
    return lsh




def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODEL_ORDER if (RESULTS / "raw" / TASK / m / f"seed{SEED}.csv").exists()]
    preds = {m: load_preds(m, SEED) for m in models}
    ids = preds[models[0]].index
    y = preds[models[0]]["y_true"]
    for m in models:
        if not (preds[m].index.equals(ids) and preds[m]["y_true"].equals(y)):
            raise RuntimeError(f"test rows of {m} are not aligned")

    # confusion matrices and spam error shares
    cm_rows, share_rows = [], []
    for m in models:
        cm = confusion(preds[m])
        for t in cm.index:
            for p in cm.columns:
                cm_rows.append({"model": m, "true": t, "pred": p, "count": int(cm.loc[t, p])})
        per_seed = []
        for s in [1, 2, 3, 4, 5]:
            d = load_preds(m, s)
            sp = d[d["y_true"] == "spam"]["y_pred"]
            per_seed.append({"spam_to_smishing": (sp == "smishing").mean(), "spam_to_ham": (sp == "ham").mean(),
                             "spam_recall": (sp == "spam").mean()})
        ps = pd.DataFrame(per_seed)
        n_spam = int(cm.loc["spam"].sum())
        share_rows.append({"model": m, "n_spam_test": n_spam,
                           "spam_to_smishing_seed1": cm.loc["spam", "smishing"] / n_spam,
                           "spam_to_ham_seed1": cm.loc["spam", "ham"] / n_spam,
                           "spam_recall_seed1": cm.loc["spam", "spam"] / n_spam,
                           "spam_to_smishing_mean5": ps["spam_to_smishing"].mean(),
                           "spam_to_ham_mean5": ps["spam_to_ham"].mean(),
                           "spam_recall_mean5": ps["spam_recall"].mean(),
                           "smishing_to_spam_seed1": cm.loc["smishing", "spam"] / int(cm.loc["smishing"].sum()),
                           "ham_to_spam_seed1": cm.loc["ham", "spam"] / int(cm.loc["ham"].sum())})
    pd.DataFrame(cm_rows).to_csv(OUT / "confusion_seed1.csv", index=False, lineterminator="\n")
    pd.DataFrame(share_rows).to_csv(OUT / "spam_error_shares.csv", index=False, lineterminator="\n")

    # hard cases
    wrong = pd.DataFrame({m: preds[m]["y_pred"] != y for m in models})
    n_wrong = wrong.sum(axis=1)
    hard = n_wrong[n_wrong >= HARD_MIN].index
    easy = n_wrong[n_wrong == 0].index
    test = pd.read_csv(ROOT / "data" / "processed" / "mendeley" / "test.csv", dtype=str, keep_default_na=False)
    test_text = dict(zip(test["id"], test["text"]))
    train = pd.read_csv(ROOT / "data" / "processed" / "mendeley" / "train.csv", dtype=str, keep_default_na=False)

    hard_df = pd.DataFrame({"id": hard, "label": y[hard].to_numpy(), "n_models_wrong": n_wrong[hard].to_numpy()})
    preds_of = {m: preds[m]["y_pred"] for m in models}
    hard_df["majority_wrong_pred"] = [
        pd.Series([preds_of[m][i] for m in models if preds_of[m][i] != y[i]]).value_counts().idxmax() for i in hard]
    for m in models:
        hard_df[f"pred_{m}"] = [preds_of[m][i] for i in hard]

    train_sets = [shingles(normalize(t)) for t in train["text"]]
    lsh = build_lsh(train_sets)
    label_of = dict(zip(y.index, y))
    nb_hard = neighbours(hard, test_text, label_of, train, lsh, train_sets)
    nb_easy = neighbours(easy, test_text, label_of, train, lsh, train_sets)
    # additional: near neighbours with another label inside test itself
    test_ref = test.rename(columns={"label": "label"})
    test_sets = [shingles(normalize(t)) for t in test["text"]]
    nb_test = neighbours(hard, test_text, label_of, test_ref, build_lsh(test_sets), test_sets, skip_self=True)
    nb_test = nb_test[["id", "n_other_label", "other_labels", "max_jaccard_other", "nearest_other_id"]].add_prefix(
        "test_").rename(columns={"test_id": "id"})
    hard_df = hard_df.merge(nb_hard, on="id").merge(nb_test, on="id")
    hard_df = hard_df.sort_values(["label", "n_models_wrong", "id"], ascending=[True, False, True])
    hard_df.to_csv(OUT / "hard_cases.csv", index=False, lineterminator="\n")
    with_text = hard_df.copy()
    with_text.insert(1, "text", [test_text[i] for i in with_text["id"]])
    train_text = dict(zip(train["id"], train["text"]))
    with_text["nearest_other_train_text"] = [train_text.get(i, "") for i in with_text["nearest_other_id"]]
    with_text["test_nearest_other_text"] = [test_text.get(i, "") for i in with_text["test_nearest_other_id"]]
    with_text.to_csv(ROOT / "reports" / "step4a_hard_cases.csv", index=False, lineterminator="\n")

    # neighbour statistics: hard vs control, per true class
    nb_easy["label"] = y[nb_easy["id"]].to_numpy()
    summary = {"seed": SEED, "models": models, "hard_min_models_wrong": HARD_MIN, "jaccard": JACCARD,
               "lsh_candidate_threshold": LSH_THRESHOLD, "num_perm": NUM_PERM,
               "pairs_exact": int(pd.concat([nb_hard, nb_easy])["n_neighbours"].sum()),
               "lsh_missed_pairs": int((pd.concat([nb_hard, nb_easy])["n_neighbours"]
                                        - pd.concat([nb_hard, nb_easy])["n_neighbours_lsh"]).sum()),
               "hard_with_other_label_neighbour_in_test": int((hard_df["test_n_other_label"] > 0).sum()),
               "n_test": len(y), "n_hard": len(hard), "n_easy_all_correct": len(easy), "groups": []}
    for name, df in [("hard", hard_df), ("easy", nb_easy)]:
        for c in CLASSES + ["all"]:
            d = df if c == "all" else df[df["label"] == c]
            if not len(d):
                continue
            g = {"group": name, "label": c, "n": len(d),
                 "with_any_neighbour": int((d["n_neighbours"] > 0).sum()),
                 "with_other_label_neighbour": int((d["n_other_label"] > 0).sum())}
            for o in CLASSES:
                g[f"with_{o}_neighbour_other"] = int(d["other_labels"].str.contains(o).sum()) if o != c else None
            summary["groups"].append(g)
    (OUT / "neighbour_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[step4a] hard cases {len(hard)} (of {len(y)}), by label: "
          + ", ".join(f"{k} {v}" for k, v in hard_df["label"].value_counts().sort_index().items()))
    print(json.dumps(summary["groups"], indent=1))
    print(pd.DataFrame(share_rows).set_index("model").round(3).rename(index=SHORT_NAMES).to_string())


if __name__ == "__main__":
    main()
