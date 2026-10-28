"""GPU-time estimate for the full step-3 run from the smoke test (results/kaggle_runs/smishfuse-smoke).

Method (stated in the output as well):
  * trainable models: per-sample training and validation time per epoch is taken
    from the super_sms_clean_binary smoke trial (the largest smoke sample, 2143
    train messages; the smaller tasks are dominated by fixed overhead); the fixed
    per-fit overhead (model init / download from cache) = trial time minus epochs and
    encoding. One fit = overhead + epochs * (n_train * t_train + n_val * t_val).
    Epoch counts: min = patience + 1, typical = epochs of the super smoke trial,
    max = max_epochs. 25 fits per (task, model): 20 trials + 5 seeds.
  * LLM: seconds per test message measured per task in the smoke run, times the
    full test size, times 5 seeds, plus one model load per worker.
Run: python scripts/estimate_gpu.py  (writes reports/step3_gpu_estimate.md)
"""
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "results" / "kaggle_runs" / "smishfuse-smoke" / "results"
TASKS = ["mendeley_3class", "mendeley_binary", "moz_binary", "super_sms_clean_binary"]
TRAINABLE = ["textcnn", "bilstm", "distilbert", "bert"]
REF = "super_sms_clean_binary"


def sizes(task: dict) -> dict:
    d = ROOT / "data" / "processed" / task["dataset"]
    return {s: len(pd.read_csv(d / f"{s}.csv", dtype=str, keep_default_na=False)) for s in ["train", "val", "test"]}


def main():
    step2 = yaml.safe_load((ROOT / "configs" / "step2.yaml").read_text())
    step3 = yaml.safe_load((ROOT / "configs" / "step3.yaml").read_text())
    proto = step3["protocol"]
    n_fits = proto["n_trials"] + len(proto["eval_seeds"])
    rows = []
    for m in TRAINABLE:
        fixed = step3["models"][m]["fixed"]
        t = json.loads((SMOKE / "tuning" / REF / m / "trial_000.json").read_text())
        h = t["history"]
        tr = sum(x["train_seconds"] for x in h) / len(h) / t["n_train"]
        va = sum(x["val_seconds"] for x in h) / len(h) / t["n_val"]
        overhead = max(0.0, t["seconds"] - sum(x["train_seconds"] + x["val_seconds"] for x in h) - t["encode_seconds"])
        epochs = {"min": fixed["patience"] + 1, "typical": min(fixed["max_epochs"], t["epochs_run"]),
                  "max": fixed["max_epochs"]}
        for task_name in TASKS:
            n = sizes(step2["tasks"][task_name])
            per_epoch = n["train"] * tr + n["val"] * va
            row = {"task": task_name, "model": m, "n_train": n["train"], "batch_size": t["params"]["batch_size"],
                   "epoch_s": per_epoch, "overhead_s": overhead}
            for k, e in epochs.items():
                row[f"hours_{k}"] = n_fits * (overhead + e * per_epoch) / 3600
            rows.append(row)
    for task_name in TASKS:
        s = json.loads((SMOKE / "seed_runs" / task_name / "qwen_fewshot" / "seed1.json").read_text())
        n = sizes(step2["tasks"][task_name])
        per_msg = s["generation_seconds"] / s["n_test"]
        h = len(proto["eval_seeds"]) * n["test"] * per_msg / 3600
        rows.append({"task": task_name, "model": "qwen_fewshot", "n_test": n["test"], "s_per_message": per_msg,
                     "hours_min": h, "hours_typical": h, "hours_max": h})
    df = pd.DataFrame(rows)
    load_s = json.loads((SMOKE / "seed_runs" / "mendeley_3class" / "qwen_fewshot" / "seed1.json").read_text())[
        "model_load_seconds"]
    tot = df.groupby("model")[["hours_min", "hours_typical", "hours_max"]].sum()

    md = ["# SmishFuse, шаг 3: оценка GPU-времени по smoke-тесту\n", __doc__.split("Run:")[0].strip(), ""]
    md.append("| Задача | Модель | Train | batch (trial 0) | Эпоха, с | Накладные на fit, с | Часы: min | типично | max |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in df[df.model != "qwen_fewshot"].itertuples():
        md.append(f"| {r.task} | {r.model} | {r.n_train} | {r.batch_size} | {r.epoch_s:.1f} | {r.overhead_s:.1f} | "
                  f"{r.hours_min:.2f} | {r.hours_typical:.2f} | {r.hours_max:.2f} |")
    md.append("\n| Задача | LLM: test | с/сообщение (smoke) | Часы на 5 seed |\n|---|---|---|---|")
    for r in df[df.model == "qwen_fewshot"].itertuples():
        md.append(f"| {r.task} | {r.n_test} | {r.s_per_message:.3f} | {r.hours_typical:.2f} |")
    md.append(f"\nЗагрузка Qwen (скачивание + 4-bit): {load_s:.0f} с на процесс.\n")
    md.append("| Модель | Часы GPU: min | типично | max |\n|---|---|---|---|")
    for m, r in tot.iterrows():
        md.append(f"| {m} | {r.hours_min:.1f} | {r.hours_typical:.1f} | {r.hours_max:.1f} |")
    s = tot.sum()
    md.append(f"| **всего (один GPU)** | **{s.hours_min:.1f}** | **{s.hours_typical:.1f}** | **{s.hours_max:.1f}** |")
    out = ROOT / "reports" / "step3_gpu_estimate.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
