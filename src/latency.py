"""Unified latency and size measurement for all step-2 and step-3 models (one Kaggle session).

* CPU models (TF-IDF baselines, TextCNN, BiLSTM, DistilBERT, BERT): one thread,
  batch 1, median over `latency_messages` test messages (cycled if fewer),
  after `latency_warmup` warm-up calls; time covers preprocessing + model.
  Step-2 pipelines are re-fitted here on train with their best parameters and
  seed 1 (same procedure as step 2); step-3 models are loaded from the seed-1
  weights saved by the GPU kernels (attached as kernel sources).
* Qwen2.5-7B 8-shot: on the GPU (T4), batch 1, same messages, the same
  generation mode as in evaluation; size = model.get_memory_footprint().
Size: serialized model in MB (joblib for step 2, saved files for step 3).
Output: results/latency/latency.csv and results/latency/environment.json
"""
from __future__ import annotations

import os

for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[var] = "1"

import glob  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "baselines"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from neural.common import RESULTS, load_configs, load_split  # noqa: E402

STEP2 = ["logreg", "linear_svm", "random_forest", "xgboost"]
TEXTNN = ["textcnn", "bilstm"]
TRANSFORMERS = ["distilbert", "bert"]
N_MSG, WARMUP = 1000, 20


def median_ms(fn, texts) -> float:
    sample = [texts[i % len(texts)] for i in range(N_MSG)]
    for t in sample[:WARMUP]:
        fn(t)
    times = []
    for t in sample:
        start = time.perf_counter()
        fn(t)
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000)


def dir_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def find_model_dir(task: str, model: str) -> Path | None:
    hits = sorted(glob.glob(f"/kaggle/input/**/models/{task}/{model}/seed1", recursive=True))
    return Path(hits[0]) if hits else None


def step2_rows(task_name, task, cfg2, test_texts):
    import classic

    data = {s: classic.load_split(task, s) for s in ["train"]}
    rows = []
    for m in STEP2:
        best = json.loads((ROOT / "results" / "best_params" / task_name / f"{m}.json").read_text())
        model, bw = classic.build_model(m, best["model_params"], cfg2["models"][m]["fixed"], 1, 1)
        clf = classic.TextClassifier(classic.build_vectorizer(best["vectorizer"], cfg2["vectorizer_fixed"]), model, bw)
        perm = np.random.default_rng(1).permutation(len(data["train"][2]))
        clf.fit(data["train"][1][perm], data["train"][2][perm])
        clf.set_single_thread()
        buf = io.BytesIO()
        joblib.dump(clf, buf)
        rows.append({"task": task_name, "model": m, "device": "cpu_1_thread",
                     "latency_ms_median_batch1": median_ms(lambda t: clf.predict_proba([t]), test_texts),
                     "model_size_mb": buf.tell() / 1e6, "source": "re-fitted, seed 1"})
    return rows


def retrain_seed1(task_name: str, model_name: str) -> Path:
    import torch

    from neural.run import fit_once

    step2, step3 = load_configs()
    task = step2["tasks"][task_name]
    best = json.loads((ROOT / "results" / "best_params" / task_name / f"{model_name}.json").read_text())
    data = {s: load_split(task, s) for s in ["train", "val"]}
    torch.set_num_threads(os.cpu_count())
    res = fit_once(model_name, step3["models"][model_name], best["model_params"], task_name, data,
                   len(task["classes"]), 1, step3["protocol"], torch.device("cuda"))
    out = RESULTS / "latency" / "retrained" / task_name / model_name / "seed1"
    res["setup"].save(res["model"].cpu(), out)
    del res
    torch.cuda.empty_cache()
    return out


def neural_rows(task_name, test_texts):
    import torch

    from neural.trainers import BiLSTM, HFClassifier, TextCNN, encode_words, word_collate

    torch.set_num_threads(1)
    rows = []
    for m in TEXTNN + TRANSFORMERS:
        d = find_model_dir(task_name, m)
        source = str(d) if d else None
        if d is None:
            # Weights not available as a kernel source (e.g. that kernel's run ended with ERROR status):
            # re-train seed 1 with the best parameters on the GPU of this session, then measure on CPU.
            d = retrain_seed1(task_name, m)
            source = "re-trained in this session (seed 1, best params)"
            torch.set_num_threads(1)
        if m in TEXTNN:
            ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
            p = ck["params"]
            net = (TextCNN(len(ck["vocab"]), p["embed_dim"], p["num_filters"], p["kernel_sizes"], p["dropout"],
                           ck["n_classes"]) if m == "textcnn" else
                   BiLSTM(len(ck["vocab"]), p["embed_dim"], p["hidden_size"], p["num_layers"], p["dropout"],
                          ck["n_classes"]))
            net.load_state_dict(ck["state_dict"])
            net.eval()
            ks = [int(k) for k in str(p.get("kernel_sizes", "1")).split(",")]
            collate = word_collate(max(ks))

            def fn(t, net=net, ck=ck, p=p, collate=collate):
                inputs, _ = collate([(encode_words([t], ck["vocab"], p["max_len"])[0], 0)])
                with torch.no_grad():
                    torch.softmax(net(**inputs), dim=1)
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(d)
            net = HFClassifier(AutoModelForSequenceClassification.from_pretrained(d)).eval()

            def fn(t, net=net, tok=tok):
                enc = tok([t], truncation=True, max_length=128, return_tensors="pt")
                with torch.no_grad():
                    torch.softmax(net(enc["input_ids"], enc["attention_mask"]), dim=1)
        rows.append({"task": task_name, "model": m, "device": "cpu_1_thread",
                     "latency_ms_median_batch1": median_ms(fn, test_texts), "model_size_mb": dir_mb(d),
                     "source": source})
    return rows


def llm_rows(task_name, task, cfg3, test_texts):
    import torch

    from neural.llm import FewShotClassifier

    mcfg = cfg3["models"]["qwen_fewshot"]
    clf = FewShotClassifier(task_name, task, mcfg, load_split(task, "train"), 1)
    use_cache = bool(mcfg["fixed"]["prefix_cache"])

    def fn(t):
        clf.answer([t], use_cache)
        torch.cuda.synchronize()

    return [{"task": task_name, "model": "qwen_fewshot", "device": f"gpu:{torch.cuda.get_device_name(0)}",
             "latency_ms_median_batch1": median_ms(fn, test_texts),
             "model_size_mb": clf.model.get_memory_footprint() / 1e6,
             "source": f"{mcfg['checkpoint']} 4-bit, prefix_cache={use_cache}"}]


def main():
    import yaml

    cfg2 = yaml.safe_load((ROOT / "configs" / "step2.yaml").read_text())
    step2, cfg3 = load_configs()
    out = RESULTS / "latency"
    out.mkdir(parents=True, exist_ok=True)
    env = {"cpu_model": subprocess.run("lscpu | grep 'Model name' | sed 's/.*: *//'", shell=True,
                                       capture_output=True, text=True).stdout.strip(),
           "cpu_count": os.cpu_count(), "threads": 1, "n_messages": N_MSG, "warmup": WARMUP}
    rows = []
    for task_name, task in step2["tasks"].items():
        test_texts = load_split(task, "test")[1]
        for part in (lambda: step2_rows(task_name, task, cfg2, test_texts), lambda: neural_rows(task_name, test_texts),
                     lambda: llm_rows(task_name, task, cfg3, test_texts)):
            try:
                rows += part()
            except Exception as e:  # noqa: BLE001 - record and continue
                rows.append({"task": task_name, "model": "?", "source": f"error: {type(e).__name__}: {e}"})
            pd.DataFrame(rows).to_csv(out / "latency.csv", index=False, lineterminator="\n")
            print(rows[-1], flush=True)
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")


if __name__ == "__main__":
    main()
