"""Merges Kaggle kernel outputs into results/ and rebuilds results/metrics.csv.

Only for full-run kernels: smoke-test outputs (reduced data) must never be merged.

  python scripts/collect_results.py KERNEL [KERNEL ...]   # merge results/kaggle_runs/<KERNEL>/results
  python scripts/collect_results.py                       # only rebuild metrics.csv

Merging copies raw predictions, tuning records, best parameters and per-run
JSON files into results/ (files of the same path are replaced: a re-run of a
model on Kaggle supersedes the previous one). metrics.csv is then recomputed
from ALL results/raw/<task>/<model>/seed<k>.csv with src/eval/metrics.py, so
every model's metrics come from the same code and environment.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.metrics import compute_metrics  # noqa: E402
from eval.registry import MODEL_ORDER  # noqa: E402

RESULTS = ROOT / "results"
MERGE_DIRS = ["raw", "tuning", "best_params", "seed_runs", "job_times", "runtime", "errors", "logs", "llm_cache_check"]
METRIC_ORDER = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "pr_auc", "recall_at_fpr_1pct", "ece",
                "invalid_rate"]


def merge(kernel: str) -> None:
    if "smoke" in kernel:
        raise SystemExit(f"{kernel}: smoke-test outputs use reduced data and are never merged into results/")
    src = RESULTS / "kaggle_runs" / kernel / "results"
    if not src.exists():
        raise SystemExit(f"{src} not found; run scripts/kaggle_sync.py output {kernel} first")
    n = 0
    for d in MERGE_DIRS:
        for f in (src / d).rglob("*") if (src / d).exists() else []:
            if f.is_file() and not f.name.endswith((".tmp", ".partial.csv")):
                dst = RESULTS / f.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
    # step-2 classic.py run on Kaggle: its efficiency rows (Kaggle CPU) are kept separately.
    for f in (src / "parts").glob("efficiency_*.csv") if (src / "parts").exists() else []:
        dst = RESULTS / "kaggle_efficiency" / kernel / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
    print(f"[collect] {kernel}: {n} files merged")


def rebuild_metrics() -> pd.DataFrame:
    tasks = yaml.safe_load((ROOT / "configs" / "step2.yaml").read_text(encoding="utf-8"))["tasks"]
    proto = yaml.safe_load((ROOT / "configs" / "step2.yaml").read_text(encoding="utf-8"))["protocol"]
    rows = []
    for t, task in tasks.items():
        classes = task["classes"]
        idx = {c: i for i, c in enumerate(classes)}
        for m in MODEL_ORDER:
            for f in sorted((RESULTS / "raw" / t / m).glob("seed*.csv"), key=lambda p: int(p.stem[4:])):
                df = pd.read_csv(f, dtype={"id": str})
                y = df["y_true"].map(idx).to_numpy()
                pred = df["y_pred"].map(lambda v: idx.get(v, -1)).to_numpy()
                probs = df[[f"prob_{c}" for c in classes]].to_numpy(dtype=np.float64)
                met = compute_metrics(y, probs, classes, positive=task["positive"], y_pred=pred,
                                      ece_bins=proto["ece_bins"], max_fpr=proto["fpr_target"])
                met["invalid_rate"] = float(np.mean(pred == -1))
                rows.append({"task": t, "model": m, "seed": int(f.stem[4:]), **met})
    df = pd.DataFrame(rows)
    first = ["task", "model", "seed"] + [c for c in METRIC_ORDER if c in df.columns]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(RESULTS / "metrics.csv", index=False, lineterminator="\n")
    print(f"[collect] metrics.csv: {len(df)} rows, models {sorted(df['model'].unique())}")
    return df


def main(kernels):
    for k in kernels:
        merge(k)
    rebuild_metrics()


if __name__ == "__main__":
    main(sys.argv[1:])
