"""Step 4c: step-2 LogReg / RandomForest / XGBoost, local run vs Kaggle CPU rerun.

The local results are read from git commit LOCAL_COMMIT (the last commit before the Kaggle CPU
rerun replaced them); the Kaggle ones from the working tree. Output:
results/step4c/step2_local_vs_kaggle.csv with the mean / std of test macro F1 over the 5 seeds,
the best tuning trial and its val macro F1 on both platforms, and whether the chosen
hyperparameters are the same (numbers compared with a relative tolerance of 1e-9).

Run: python scripts/compare_step2_platform.py
"""
from __future__ import annotations

import io
import json
import math
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCAL_COMMIT = "c03ec36"
MODELS = ["logreg", "random_forest", "xgboost"]
OUT = ROOT / "results" / "step4c" / "step2_local_vs_kaggle.csv"


def git_show(path: str) -> str:
    return subprocess.run(["git", "show", f"{LOCAL_COMMIT}:{path}"], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout


def same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-9)
    return a == b


def main():
    old = pd.read_csv(io.StringIO(git_show("results/metrics.csv")))
    new = pd.read_csv(ROOT / "results" / "metrics.csv")
    rows = []
    for task in sorted(new.task.unique()):
        for m in MODELS:
            o = old[(old.task == task) & (old.model == m)].macro_f1
            n = new[(new.task == task) & (new.model == m)].macro_f1
            bp = f"results/best_params/{task}/{m}.json"
            bo, bn = json.loads(git_show(bp)), json.loads((ROOT / bp).read_text(encoding="utf-8"))
            rows.append({"task": task, "model": m, "local": o.mean(), "local_std": o.std(), "kaggle": n.mean(),
                         "kaggle_std": n.std(), "delta": n.mean() - o.mean(), "n_seeds": len(n),
                         "trial_local": bo["trial"], "trial_kaggle": bn["trial"],
                         "val_local": bo["val_macro_f1"], "val_kaggle": bn["val_macro_f1"],
                         "same_params": same({k: bo[k] for k in ("model_params", "vectorizer")},
                                             {k: bn[k] for k in ("model_params", "vectorizer")})})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
