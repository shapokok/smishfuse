"""Step 5b, freezing: tau of the cascade on the FULL clean val (mendeley_3class), before the test is read.

Same criterion as step 5a (configs/step5a.yaml): macro F1 on the clean val, ties -> the smallest tau
(fewest stage-2 calls). One tau for all seeds: the candidate with the highest mean macro F1 over the
seeds whose val probabilities exist (step 4c, seeds 1-3). Candidates: 0, every distinct stage-1 max
probability of any seed, 1 + 1e-9. Reads only results/step4c/<model>/clean_seed<k>.csv (val).
Run: python scripts/freeze_tau.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["ham", "smishing", "spam"]
PAIRS = {"cascade": ("svm_char_aug", "distilbert_aug"), "cascade_noaug": ("svm_char", "distilbert")}


def load(model: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "step4c" / model / f"clean_seed{seed}.csv", dtype={"id": str}).set_index("id")


def main():
    for name, (s1, s2) in PAIRS.items():
        seeds = [k for k in (1, 2, 3, 4, 5) if (ROOT / "results" / "step4c" / s1 / f"clean_seed{k}.csv").exists()]
        runs = []
        for k in seeds:
            a, b = load(s1, k), load(s2, k).loc[load(s1, k).index]
            y = a["y_true"].map(CLASSES.index).to_numpy()
            p1, p2 = a[[f"prob_{c}" for c in CLASSES]].to_numpy(), b[[f"prob_{c}" for c in CLASSES]].to_numpy()
            runs.append((y, p1, p2))
        cands = np.unique(np.concatenate([[0.0, 1.0 + 1e-9]] + [r[1].max(1) for r in runs]))
        best = None
        for tau in cands:
            f1s = [f1_score(y, np.where(p1.max(1) >= tau, p1.argmax(1), p2.argmax(1)), labels=[0, 1, 2],
                            average="macro", zero_division=0) for y, p1, p2 in runs]
            m = float(np.mean(f1s))
            if best is None or m > best[1]:
                best = (float(tau), m, f1s, [float((p1.max(1) < tau).mean()) for _, p1, _ in runs])
        tau, m, f1s, shares = best
        print(f"{name} ({s1} -> {s2}), seeds {seeds}: tau = {tau!r}; val macro F1 per seed "
              f"{[round(f, 4) for f in f1s]}, mean {m:.4f}; share to stage 2 {[round(s, 4) for s in shares]}")


if __name__ == "__main__":
    main()
