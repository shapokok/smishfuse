"""Step 2.3: best model vs every other model, per task.

Best model = highest mean test macro F1 over the evaluation seeds. Comparisons
use the test predictions of protocol.significance_seed (seed 1):
paired bootstrap of the macro F1 difference (best - other) and McNemar.
Holm correction is applied within each task, separately for each test.
Output: results/significance.csv
Run: python scripts/significance.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.registry import ordered  # noqa: E402
from eval.stats import holm, mcnemar, paired_bootstrap_macro_f1  # noqa: E402

RESULTS = ROOT / "results"


def load_preds(task_name: str, model: str, seed: int, classes: list[str]):
    df = pd.read_csv(RESULTS / "raw" / task_name / model / f"seed{seed}.csv", dtype={"id": str})
    idx = {c: i for i, c in enumerate(classes)}
    # LLM answers outside the label set are stored as "invalid" -> -1 (always wrong).
    return df["id"].to_numpy(), df["y_true"].map(idx).to_numpy(), df["y_pred"].map(lambda v: idx.get(v, -1)).to_numpy()


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "step2.yaml").read_text(encoding="utf-8"))
    proto = cfg["protocol"]
    seed = proto["significance_seed"]
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    rows = []
    for task_name, task in cfg["tasks"].items():
        classes = task["classes"]
        k = len(classes)
        means = metrics[metrics["task"] == task_name].groupby("model")["macro_f1"].mean()
        models = ordered(means.index)
        best = max(models, key=lambda m: (means[m], -models.index(m)))
        ids_b, y, pred_best = load_preds(task_name, best, seed, classes)
        task_rows = []
        for other in models:
            if other == best:
                continue
            ids_o, y_o, pred_other = load_preds(task_name, other, seed, classes)
            if not (np.array_equal(ids_b, ids_o) and np.array_equal(y, y_o)):
                raise RuntimeError(f"{task_name}: test rows of {best} and {other} are not aligned")
            bs = paired_bootstrap_macro_f1(y, pred_best, pred_other, k, n_boot=proto["bootstrap_n"],
                                           seed=proto["bootstrap_seed"])
            mc = mcnemar(y, pred_best, pred_other)
            task_rows.append({
                "task": task_name, "best_model": best, "other_model": other, "seed": seed, "n_test": len(y),
                "macro_f1_best": f1_score(y, pred_best, labels=list(range(k)), average="macro", zero_division=0),
                "macro_f1_other": f1_score(y, pred_other, labels=list(range(k)), average="macro", zero_division=0),
                "diff_macro_f1": bs["diff"], "ci95_low": bs["ci_low"], "ci95_high": bs["ci_high"],
                "p_bootstrap": bs["p_value"], "n_boot": bs["n_boot"],
                "mcnemar_b_best_only_correct": mc["b"], "mcnemar_c_other_only_correct": mc["c"],
                "mcnemar_method": mc["method"], "p_mcnemar": mc["p_value"],
            })
        for test in ["bootstrap", "mcnemar"]:
            adj, rej = holm([r[f"p_{test}"] for r in task_rows], proto["alpha"])
            for r, a, rj in zip(task_rows, adj, rej):
                r[f"p_{test}_holm"] = a
                r[f"significant_{test}_holm"] = rj
        rows += task_rows
        print(f"[significance] {task_name}: best {best}")
    cols = ["task", "best_model", "other_model", "seed", "n_test", "macro_f1_best", "macro_f1_other",
            "diff_macro_f1", "ci95_low", "ci95_high", "p_bootstrap", "p_bootstrap_holm", "significant_bootstrap_holm",
            "n_boot", "mcnemar_b_best_only_correct", "mcnemar_c_other_only_correct", "mcnemar_method", "p_mcnemar",
            "p_mcnemar_holm", "significant_mcnemar_holm"]
    pd.DataFrame(rows)[cols].to_csv(RESULTS / "significance.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
