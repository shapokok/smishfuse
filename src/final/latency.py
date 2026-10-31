"""Step 5b: speed of every final model on one Kaggle CPU session (configs/final.yaml, latency).

One thread, batch 1: per-message latency p50 / p95 over 1000 test messages of mendeley_3class (cycled,
after 20 warm-up calls), throughput = messages / total seconds, size = saved model files in MB. The
seed-1 models come from the final GPU notebooks (kernel sources, results/final/models/); a model
whose weights are missing is re-trained with seed 1 on this CPU and marked so. The cascade is also
measured end to end on the real stream: every mendeley_3class test message in file order, stage 2
only when the stage-1 max probability < tau; the realised share of stage-2 calls is reported.
Qwen2.5-7B (4-bit) cannot run on CPU: its step-3 GPU measurement is copied, marked as GPU.
Output: results/final/latency.csv, results/final/cascade_end_to_end.json
"""
from __future__ import annotations

import os

for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[var] = "1"

import glob  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "baselines"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from neural.common import RESULTS, load_yaml, write_json  # noqa: E402

DRY = os.environ.get("SMISHFUSE_FINAL_DRY") == "1"  # smoke test: few messages, no re-training
OUT = RESULTS / ("final_dry" if DRY else "final")
FCFG = load_yaml("configs/final.yaml")
N_MSG, WARMUP = FCFG["latency"]["per_model"]["messages"], FCFG["latency"]["per_model"]["warmup"]
if DRY:
    N_MSG, WARMUP = 50, 5
CPU = torch.device("cpu")
MAIN = ["svm_char_aug", "svm_char", "distilbert_aug", "distilbert", "smishfuse_v1", "smishfuse_v2",
        "smishfuse_v2_noaug", "bert", "textcnn", "bilstm", "distilbert_step3", "linear_svm", "logreg",
        "random_forest", "xgboost"]
PUBLISHED = {"mahmud2024_cnn_bigru": "setting_mahmud", "shim2024_bert_base": "setting_shim",
             "shim2024_distilbert_base": "setting_shim", "shim2024_roberta_base": "setting_shim"}


def log(msg: str) -> None:
    print(msg, flush=True)


def find(cond: str, model: str) -> Path | None:
    hits = sorted(glob.glob(f"/kaggle/input/**/final/models/{cond}/{model}/seed1", recursive=True))
    local = OUT / "models" / cond / model / "seed1"
    return Path(hits[0]) if hits else (local if local.exists() else None)


def dir_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def timings(fn, texts) -> list[float]:
    sample = [texts[i % len(texts)] for i in range(N_MSG)]
    for t in sample[:WARMUP]:
        fn(t)
    out = []
    for t in sample:
        start = time.perf_counter()
        fn(t)
        out.append(time.perf_counter() - start)
    return out


def summary(times) -> dict:
    a = np.asarray(times)
    return {"p50_ms": float(np.median(a) * 1000), "p95_ms": float(np.percentile(a, 95) * 1000),
            "mean_ms": float(a.mean() * 1000), "throughput_msg_per_s": float(len(a) / a.sum())}


# ---------------------------------------------------------------- loading the saved seed-1 models

def load(cond: str, model: str, path: Path):
    """-> function text -> probabilities (batch of one)."""
    from final import run

    if (path / "model.joblib").exists():
        clf = joblib.load(path / "model.joblib")
        clf.set_single_thread()
        return lambda t: clf.predict_proba(np.asarray([t], dtype=object))
    if model in run.Fusion.SPEC:
        from transformers import AutoTokenizer

        from fusion.model import SmishFuse
        from fusion.train import predict_texts

        ck = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        fc = run.MCFG["fusion"]
        net = SmishFuse(ck["mode"], ck["n_features"], fc["d"], fc["feature_mlp_hidden"], fc["dropout"], ck["n_classes"],
                        ck["checkpoint"])
        net.load_state_dict(ck["state_dict"])
        net.eval()
        tok = AutoTokenizer.from_pretrained(ck["checkpoint"])
        feats = joblib.load(path / "features.joblib") if (path / "features.joblib").exists() else \
            (lambda texts: np.zeros((len(texts), 1)))
        return lambda t: predict_texts(net, tok, [t], feats([t]), CPU, run.MCFG)[0]
    if model == "mahmud2024_cnn_bigru":
        from step5a import methods

        ck = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        mm = methods.MahmudModel(ck["config"], CPU)
        mm.vocab = ck["vocab"]
        emb = np.zeros((len(mm.vocab) + 1, ck["config"]["word2vec"]["vector_size"]), dtype=np.float32)
        mm.model = methods.CNNBiGRU(emb, ck["config"])
        mm.model.load_state_dict(ck["state_dict"])
        return lambda t: mm.predict_proba([t])
    if model.startswith("shim2024"):
        from transformers import AutoModelForSequenceClassification

        from step5a import methods

        tr = methods.Transformer(str(path), run.CFG5A["published"]["shim2024"]["max_len"], CPU)
        tr.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        return lambda t: tr.predict_proba([t])
    # step-3 neural baselines (setup.save format of step 3)
    from neural.trainers import BiLSTM, HFClassifier, TextCNN, encode_words, word_collate

    if (path / "model.pt").exists():
        ck = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        p = ck["params"]
        net = (TextCNN(len(ck["vocab"]), p["embed_dim"], p["num_filters"], p["kernel_sizes"], p["dropout"],
                       ck["n_classes"]) if model == "textcnn" else
               BiLSTM(len(ck["vocab"]), p["embed_dim"], p["hidden_size"], p["num_layers"], p["dropout"],
                      ck["n_classes"]))
        net.load_state_dict(ck["state_dict"])
        net.eval()
        collate = word_collate(max(int(k) for k in str(p.get("kernel_sizes", "1")).split(",")))

        def fn(t):
            inputs, _ = collate([(encode_words([t], ck["vocab"], p["max_len"])[0], 0)])
            with torch.no_grad():
                return torch.softmax(net(**inputs), dim=1)
        return fn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    net = HFClassifier(AutoModelForSequenceClassification.from_pretrained(path)).eval()

    def fn(t):
        enc = tok([t], truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            return torch.softmax(net(enc["input_ids"], enc["attention_mask"]), dim=1)
    return fn


def retrain(cond: str, model: str) -> Path:
    """Seed-1 model re-trained on this CPU (weights not available as a kernel source)."""
    from final import run

    task = run.TASK_OF[cond]
    torch.set_num_threads(os.cpu_count())
    m = run.train(run.build(model, "mendeley_3class"), task, run.data(task, "train"), run.data(task, "val"), 1, CPU)
    path = OUT / "latency_retrained" / cond / model / "seed1"
    m.save(path)
    torch.set_num_threads(1)
    return path


def main():
    from final import run

    torch.set_num_threads(1)
    _, texts, _ = run.data("mendeley_3class", "test")
    texts = list(texts)
    rows, fns = [], {}
    for model, cond in [*((m, "mendeley_3class") for m in MAIN), *PUBLISHED.items()]:
        path, source = find(cond, model), "seed-1 weights of the final run"
        if path is None and DRY:
            continue
        if path is None:
            path, source = retrain(cond, model), "re-trained with seed 1 on this CPU (weights not available)"
        fn = load(cond, model, path)
        fns[model] = fn
        rows.append({"model": model, "condition": cond, "device": "cpu_1_thread", **summary(timings(fn, texts)),
                     "size_mb": dir_mb(path), "source": source})
        log(f"[latency] {model}: p50 {rows[-1]['p50_ms']:.3f} ms, p95 {rows[-1]['p95_ms']:.3f} ms")
    # cascade end to end on the real test stream
    if not {"svm_char_aug", "distilbert_aug"} <= set(fns):
        pd.DataFrame(rows).to_csv(OUT / "latency.csv", index=False, lineterminator="\n")
        return
    tau = FCFG["method"]["tau"]
    s1, s2 = fns["svm_char_aug"], fns["distilbert_aug"]
    for t in texts[:WARMUP]:
        s1(t)
        s2(t)
    times, routed = [], 0
    for t in texts:
        start = time.perf_counter()
        p = np.asarray(s1(t))
        if p.max() < tau:
            s2(t)
            routed += 1
        times.append(time.perf_counter() - start)
    e2e = {"n_messages": len(texts), "routed": routed, "routed_share": routed / len(texts), "tau": tau, **summary(times),
           "size_mb": sum(r["size_mb"] for r in rows if r["model"] in ("svm_char_aug", "distilbert_aug"))}
    write_json(e2e, OUT / "cascade_end_to_end.json")
    rows.append({"model": "cascade", "condition": "mendeley_3class", "device": "cpu_1_thread (end to end, test stream)",
                 **{k: e2e[k] for k in ("p50_ms", "p95_ms", "mean_ms", "throughput_msg_per_s", "size_mb")},
                 "source": f"svm_char_aug + distilbert_aug, {100 * e2e['routed_share']:.1f}% to stage 2"})
    q = pd.read_csv(ROOT / "results" / "latency" / "latency.csv")
    q = q[(q.task == "mendeley_3class") & (q.model == "qwen_fewshot")].iloc[0]
    rows.append({"model": "qwen_fewshot", "condition": "mendeley_3class", "device": q["device"],
                 "p50_ms": float(q["latency_ms_median_batch1"]), "size_mb": float(q["model_size_mb"]),
                 "source": "step-3 GPU measurement (median only), not CPU"})
    pd.DataFrame(rows).to_csv(OUT / "latency.csv", index=False, lineterminator="\n")
    log(f"[latency] cascade end to end: {e2e}")


if __name__ == "__main__":
    main()
