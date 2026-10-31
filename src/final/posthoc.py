"""Step 5b, POST-HOC analysis after the final evaluation (README decision 20): no training, CPU only,
from the saved test predictions of results/final/preds.

Significance of the char LinearSVM (without and with augmentation) against the re-implemented published
methods in every condition of their settings (setting_mahmud: CNN-BiGRU; setting_shim: BERT, DistilBERT,
RoBERTa; clean and HELD-OUT 5/10/20%): seed 1, paired bootstrap of the macro F1 difference (10 000,
seed 42) and McNemar, Holm within each condition x test set over its comparisons, separately per test.
Output: results/final/posthoc_significance.csv
"""
from __future__ import annotations

from final import assemble
from eval.stats import holm, mcnemar, paired_bootstrap_macro_f1

OURS = ["svm_char", "svm_char_aug"]


def main() -> None:
    import pandas as pd

    sc = assemble.FCFG["significance"]
    rows = []
    for cond, published in assemble.PUBLISHED.items():
        task, sets = assemble.CONDS[cond]
        cls = assemble.CLASSES[task]
        for s in sets:
            block = []
            for ours in OURS:
                a = assemble.read(cond, ours, s, sc["seed"])
                for other in published:
                    b = assemble.read(cond, other, s, sc["seed"])
                    if a is None or b is None:
                        continue
                    b = b.set_index("id").loc[a["id"]].reset_index()
                    y = a["y_true"].map(cls.index).to_numpy()
                    pa, pb = a["y_pred"].map(cls.index).to_numpy(), b["y_pred"].map(cls.index).to_numpy()
                    bs = paired_bootstrap_macro_f1(y, pa, pb, len(cls), sc["bootstrap"]["n"], sc["bootstrap"]["seed"])
                    mc = mcnemar(y, pa, pb)
                    block.append({"condition": cond, "test_set": s, "ours": ours, "other": other, "seed": sc["seed"],
                                  "n": len(y), "diff_macro_f1_seed1": bs["diff"], "ci95_low": bs["ci_low"],
                                  "ci95_high": bs["ci_high"], "p_bootstrap": bs["p_value"],
                                  "mcnemar_b_ours_only_correct": mc["b"], "mcnemar_c_other_only_correct": mc["c"],
                                  "mcnemar_method": mc["method"], "p_mcnemar": mc["p_value"]})
            for test in ("bootstrap", "mcnemar"):
                adj, rej = holm([r[f"p_{test}"] for r in block], sc["alpha"])
                for r, p, j in zip(block, adj, rej):
                    r[f"p_{test}_holm"], r[f"significant_{test}_holm"] = p, bool(j)
            rows += block
    df = pd.DataFrame(rows)
    df.to_csv(assemble.OUT / "posthoc_significance.csv", index=False, lineterminator="\n")
    print(f"[posthoc] {len(df)} comparisons -> {assemble.OUT / 'posthoc_significance.csv'}", flush=True)


if __name__ == "__main__":
    main()
