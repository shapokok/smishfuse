"""Export of the second stage of Full (DistilBERT) to ONNX for the server of smishfuse-app, Kaggle CPU.

Full = cascade svm_char_aug -> distilbert_aug (configs/final.yaml, rule of src/final/assemble.py::cascade_probs:
stage 2 when the stage-1 max calibrated probability < tau = 0.6651). Both stages are the frozen seed-1 models of the
mendeley_3class condition:
  stage 2  distilbert_aug: weights of the Kaggle kernel smishfuse-final-3class (kernel source,
           results/final/models/mendeley_3class/distilbert_aug/seed1/model.pt; not on the local disk). Exported to
           ONNX fp32 (input_ids, attention_mask -> probs = softmax of the 3-class head) and to int8 (onnxruntime
           dynamic quantization, QInt8 weights).
  stage 1  svm_char_aug: re-trained with its frozen settings (src/final/run.py Classic: one TRAIN-type augmented
           copy of train, seed 1) and checked against its frozen val predictions; written in the Lite JSON format of
           scripts/export_lite.py ("smishfuse-lite/1"), so that the server can run Full without the device.
Checks (report.json): PyTorch CPU fp32 vs the frozen GPU (fp16 autocast) predictions on the clean test; ONNX fp32 and
int8 vs PyTorch; macro F1 on the Mendeley test for PyTorch / fp32 / int8 and the decision: int8 if its macro F1 is at
most 0.5 points below fp32, otherwise fp32; the cascade with the exported stages vs the frozen cascade predictions;
single-thread latency of onnxruntime on the Kaggle CPU.
golden_distilbert.json: 200 messages with the PyTorch fp32 probabilities (reference of the server test) and the ONNX /
stage-1 / cascade outputs. No dataset text: author-written KZ case rows (data/kz_cases), adaptation messages
(data/kz_adapt), their TRAIN (TR) and HELD-OUT (HO) obfuscation and edge cases.

Kaggle run, the same plumbing as the other kernels (scripts/kaggle_sync.py, scripts/make_kernels.py, not changed):
  python scripts/export_distilbert_onnx.py check-stage1   # local, seconds: stage 1 re-trained vs frozen val
  python scripts/export_distilbert_onnx.py push    # code hash == uploaded smishfuse-code? builds the kernel
                                                   # (kaggle/build/smishfuse-export-onnx, this script embedded with its
                                                   # sha256, checked on Kaggle together with CODE_SHA256) and pushes it
  python scripts/export_distilbert_onnx.py status
  python scripts/export_distilbert_onnx.py fetch   # the one output file export.zip -> kaggle/build/..., sha256 checked
On Kaggle the notebook runs `export_distilbert_onnx.py run` and keeps only /kaggle/working/export.zip.
"""
from __future__ import annotations

import os

for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(var, "1")

import argparse  # noqa: E402
import base64  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "smishfuse-export-onnx"
CLASSES = ["ham", "smishing", "spam"]
SEED = 1
CHECK_TOL = 1e-6
MAX_DROP_PP = 0.5
N_GOLDEN = 200
EXTRA_PIP = ["onnx==1.23.2", "onnxruntime==1.31.1", "onnxscript==0.7.2"]
WEIGHTS_GLOB = "/kaggle/input/**/final/models/mendeley_3class/distilbert_aug/seed1/model.pt"
# frozen predictions used by the checks (ids and probabilities only); results/final/preds is not part of the
# smishfuse-code dataset, so `push` embeds these files in the notebook (tar.gz, sha256 checked on Kaggle)
FROZEN_FILES = [f"results/final/preds/mendeley_3class/{m}/{s}_seed1.csv"
                for m, s in (("svm_char_aug", "val"), ("svm_char_aug", "clean"), ("distilbert_aug", "clean"),
                             ("cascade", "clean"))]
EDGE = [
    ("empty", ""),
    ("one_char", "a"),
    ("whitespace_runs", "  Ваша   карта \t\t заблокирована \n\n  код 1234  "),
    ("emoji", "\U0001F381 Вы выиграли iPhone 15! \U0001F449 http://prize-kz.top/win \U0001F525"),
    ("uppercase", "URGENT! YOUR CARD IS BLOCKED. CALL +7 701 000 00 00 NOW"),
    ("mixed_lang", "Сәлем! Your Kaspi card заблокирована, растаңыз: http://kaspi-secure.top"),
    ("url_only", "http://bank-unlock24.kz/card"),
    ("long_truncated", " ".join(["Ваша карта заблокирована, перейдите по ссылке http://bank-unlock24.kz/card"] * 12)),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- stage 1 (svm_char_aug), sklearn only

def _src_path():
    for p in (ROOT / "src", ROOT / "src" / "baselines"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def fit_stage1():
    """Exactly Classic('svm_char_aug', 'mendeley_3class').fit(..., seed 1) of src/final/run.py (no torch import)."""
    _src_path()
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    import classic
    from eval.obfuscation import TRAIN_TYPES, augment
    from neural.common import load_split, load_yaml, read_json

    task = load_yaml("configs/step2.yaml")["tasks"]["mendeley_3class"]
    assert task["classes"] == CLASSES
    ids, texts, y = load_split(task, "train")
    cfg4c = load_yaml("configs/step4c.yaml")
    cb, aug = cfg4c["char_branch"], cfg4c["augmentation"]
    vec = TfidfVectorizer(analyzer="char", ngram_range=tuple(cb["ngram_range"]), lowercase=cb["lowercase"],
                          sublinear_tf=cb["sublinear_tf"], min_df=cb["min_df"], dtype=np.float32)
    best = read_json(ROOT / "results" / "best_params" / "mendeley_3class" / "linear_svm.json")
    fixed = load_yaml("configs/step2.yaml")["models"]["linear_svm"]["fixed"]
    model, bw = classic.build_model("linear_svm", best["model_params"], fixed, SEED, 1)
    clf = classic.TextClassifier(vec, model, bw)
    t = np.asarray(augment(list(texts), list(ids), SEED, 0, aug["p"], aug["max_level"], TRAIN_TYPES), dtype=object)
    perm = np.random.default_rng(SEED).permutation(len(y))  # train order of the final runs
    clf.fit(t[perm], np.asarray(y)[perm])
    return clf, best["model_params"], fixed, {"data": "mendeley_3class train, one TRAIN-type augmented copy "
                                              f"(p {aug['p']}, level U(0, {aug['max_level']}), seed 1, epoch 0)",
                                              "n": int(len(y))}


def frozen(model: str, split: str, ids):
    import pandas as pd

    df = pd.read_csv(ROOT / "results" / "final" / "preds" / "mendeley_3class" / model / f"{split}_seed1.csv",
                     dtype={"id": str}).set_index("id").loc[list(ids)]
    return df


def check_stage1(clf) -> float:
    _src_path()
    import numpy as np

    from neural.common import load_split, load_yaml

    task = load_yaml("configs/step2.yaml")["tasks"]["mendeley_3class"]
    va_ids, va_texts, _ = load_split(task, "val")
    want = frozen("svm_char_aug", "val", va_ids)[[f"prob_{c}" for c in CLASSES]].to_numpy(dtype=np.float64)
    got = clf.predict_proba(va_texts)
    diff = float(np.abs(got - want).max())
    same = float((got.argmax(1) == want.argmax(1)).mean())
    log(f"[stage1] svm_char_aug vs frozen val seed 1: max |p - p_frozen| = {diff:.2e}, same class {same:.4f}")
    if diff > CHECK_TOL:
        raise SystemExit(f"stage 1 differs from the frozen predictions ({diff:.2e} > {CHECK_TOL})")
    return diff


def f32(x: float) -> float:
    import numpy as np

    return float(np.format_float_positional(np.float32(x), unique=True, trim="-"))


def lite_json(clf, params: dict, fixed: dict, train_info: dict, checks: dict, tau: float) -> dict:
    """Same format as scripts/export_lite.py (smishfuse-lite/1)."""
    import sklearn

    vec, cal = clf.vectorizer, clf.model
    vocab = [None] * len(vec.vocabulary_)
    for g, j in vec.vocabulary_.items():
        vocab[j] = g
    ccs = [{"coef": cc.estimator.coef_.tolist(), "intercept": cc.estimator.intercept_.tolist(),
            "platt_a": [float(c.a_) for c in cc.calibrators], "platt_b": [float(c.b_) for c in cc.calibrators]}
           for cc in cal.calibrated_classifiers_]
    return {
        "format": "smishfuse-lite/1", "name": "full_stage1",
        "description": "Stage 1 of Full: frozen final svm_char_aug, seed 1 (Mendeley 3-class train, one augmented copy)",
        "classes": CLASSES,
        "vectorizer": {"analyzer": vec.analyzer, "ngram_range": list(vec.ngram_range), "lowercase": vec.lowercase,
                       "sublinear_tf": vec.sublinear_tf, "norm": vec.norm, "use_idf": vec.use_idf,
                       "smooth_idf": vec.smooth_idf, "min_df": vec.min_df, "max_df": vec.max_df,
                       "max_features": vec.max_features, "strip_accents": vec.strip_accents, "dtype": "float32",
                       "whitespace_regex": r"\s\s+", "n_features": len(vocab)},
        "vocabulary": vocab,
        "idf": [f32(v) for v in vec.idf_],
        "classifier": {"type": "LinearSVC one-vs-rest", "C": params["C"], "class_weight": params["class_weight"],
                       "max_iter": fixed["max_iter"], "random_state": SEED},
        "calibration": {"type": "CalibratedClassifierCV", "method": fixed["calibration_method"],
                        "cv": fixed["calibration_cv"], "ensemble": fixed["calibration_ensemble"],
                        "n_calibrated_classifiers": len(ccs),
                        "rule": "p_k = expit(-(a_k * s_k + b_k)); p /= sum(p) (uniform if 0); mean over classifiers"},
        "calibrated_classifiers": ccs,
        "tau": tau,
        "provenance": {"sklearn": sklearn.__version__, "train": train_info, "checks": checks},
    }


def golden_messages() -> list[dict]:
    _src_path()
    import pandas as pd

    from eval.obfuscation import HELDOUT_TYPES, TRAIN_TYPES, obfuscate

    rows = []
    cases = pd.read_csv(ROOT / "data" / "kz_cases" / "cases.csv", dtype=str, keep_default_na=False)
    for _, r in cases.iterrows():
        rows.append({"id": f"case:{r['id']}", "lang": r["language"], "source": "kz_case", "variant": r["variant"],
                     "label": r["label"], "text": r["text"]})
    adapt = pd.read_csv(ROOT / "data" / "kz_adapt" / "adapt.csv", dtype=str, keep_default_na=False)
    picked = []
    for (lang, label), g in adapt.groupby(["language", "label"], sort=True):
        for _, r in g.sample(n=9, random_state=SEED).sort_values("id").iterrows():
            rows.append({"id": f"adapt:{r['id']}", "lang": lang, "source": "kz_adapt", "variant": "original",
                         "label": label, "text": r["text"]})
            if lang == "en" and label in ("smishing", "spam"):
                picked.append(rows[-1])
    obf_src = [r for r in picked if r["label"] == "smishing"][:6] + [r for r in picked if r["label"] == "spam"][:6]
    for r in obf_src:
        for variant, types, level in (("tr10", TRAIN_TYPES, 0.10), ("tr20", TRAIN_TYPES, 0.20),
                                      ("ho10", HELDOUT_TYPES, 0.10), ("ho20", HELDOUT_TYPES, 0.20)):
            text, _ = obfuscate(r["text"], r["id"], level, types)
            rows.append({**r, "id": f"{r['id']}|{variant}", "variant": variant, "text": text})
    for name, text in EDGE:
        rows.append({"id": f"edge:{name}", "lang": "mixed", "source": "edge", "variant": "edge", "label": None,
                     "text": text})
    if len(rows) != N_GOLDEN or len({r["id"] for r in rows}) != N_GOLDEN:
        raise SystemExit(f"golden set: {len(rows)} messages, expected {N_GOLDEN} unique")
    return rows


# ---------------------------------------------------------------- Kaggle: the export

def find_data() -> Path:
    hits = sorted(glob.glob("/kaggle/input/**/mendeley/train.csv", recursive=True))
    if not hits:
        tmp = Path("/tmp/smishfuse_data")
        for a in glob.glob("/kaggle/input/**/smishfuse-data/**/*.zip", recursive=True):
            zipfile.ZipFile(a).extractall(tmp / Path(a).stem)
        hits = sorted(glob.glob(str(tmp / "**" / "mendeley" / "train.csv"), recursive=True))
    if not hits:
        raise SystemExit("smishfuse-data not found under /kaggle/input")
    return Path(hits[0]).parents[1]


def run(out: Path):
    os.environ.setdefault("SMISHFUSE_DATA", str(find_data()))
    os.environ.setdefault("SMISHFUSE_RESULTS", str(out / "results"))
    _src_path()
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from sklearn.metrics import f1_score
    from transformers import AutoTokenizer

    from fusion.envcheck import check_versions
    from fusion.model import SmishFuse
    from neural.common import load_split, load_yaml

    t_start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "export"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    check_versions("export_onnx", out / "results" / "environment")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    report: dict = {"kernel": KERNEL, "versions": {"torch": torch.__version__, "onnx": onnx.__version__,
                                                   "onnxruntime": ort.__version__}}
    fcfg = load_yaml("configs/final.yaml")
    tau = float(fcfg["method"]["tau"])
    mcfg = load_yaml(load_yaml("configs/step4c.yaml")["model_config"])
    max_len = int(mcfg["text"]["max_len"])
    task = load_yaml("configs/step2.yaml")["tasks"]["mendeley_3class"]

    # ---- stage 2: the frozen DistilBERT
    hits = sorted(glob.glob(WEIGHTS_GLOB, recursive=True))
    if not hits:
        raise SystemExit("distilbert_aug seed-1 weights not found (kernel source smishfuse-final-3class)")
    ck = torch.load(hits[0], map_location="cpu", weights_only=True)
    if ck["mode"] != "text" or ck["model"] != "distilbert_aug" or ck["n_classes"] != 3:
        raise SystemExit(f"unexpected checkpoint: {ck['model']} {ck['mode']} {ck['n_classes']}")
    fc = mcfg["fusion"]
    net = SmishFuse(ck["mode"], ck["n_features"], fc["d"], fc["feature_mlp_hidden"], fc["dropout"], ck["n_classes"],
                    ck["checkpoint"])
    net.load_state_dict(ck["state_dict"], strict=True)
    net.eval()
    tok = AutoTokenizer.from_pretrained(ck["checkpoint"])
    report["model"] = {"weights": hits[0], "checkpoint": ck["checkpoint"], "max_len": max_len, "classes": CLASSES,
                       "head": "[CLS] -> text_proj (768 -> 256) -> head3 (256 -> 3) -> softmax"}
    log(f"[export] loaded {hits[0]}")

    class Probs(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            l3, _, _ = self.m(input_ids=input_ids, attention_mask=attention_mask)
            return torch.softmax(l3, dim=1)

    wrapper = Probs(net).eval()

    def encode(text: str):
        return tok(text, truncation=True, max_length=max_len, return_tensors="np")

    def torch_proba(text: str) -> np.ndarray:
        e = encode(text)
        with torch.no_grad():
            p = wrapper(torch.from_numpy(e["input_ids"]), torch.from_numpy(e["attention_mask"]))
        return p[0].double().numpy()

    # ---- ONNX fp32
    fp32 = stage / "distilbert_fp32.onnx"
    ex = tok(["Ваша карта заблокирована http://bank-unlock24.kz", "ok see you at 7"], padding=True,
             truncation=True, max_length=max_len, return_tensors="pt")
    args = (ex["input_ids"], ex["attention_mask"])
    names = {"input_names": ["input_ids", "attention_mask"], "output_names": ["probs"]}
    try:
        torch.onnx.export(wrapper, args, str(fp32), dynamo=False, opset_version=17, do_constant_folding=True,
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                        "probs": {0: "batch"}}, **names)
        report["exporter"] = "torch.onnx (TorchScript), opset 17"
    except Exception as e:  # noqa: BLE001  the TorchScript exporter may be unavailable in this torch
        log(f"[export] TorchScript exporter failed ({type(e).__name__}: {e}); using the dynamo exporter")
        from torch.export import Dim

        b, s = Dim("batch", min=1, max=256), Dim("seq", min=2, max=max_len)
        prog = torch.onnx.export(wrapper, args, dynamo=True, opset_version=18,
                                 dynamic_shapes={"input_ids": {0: b, 1: s}, "attention_mask": {0: b, 1: s}}, **names)
        prog.save(str(fp32), external_data=False)
        report["exporter"] = "torch.onnx (dynamo), opset 18"
    onnx.checker.check_model(str(fp32))
    log(f"[export] fp32 ONNX {fp32.stat().st_size / 1e6:.1f} MB ({report['exporter']})")

    # ---- ONNX int8 (dynamic quantization of the weights)
    int8 = stage / "distilbert_int8.onnx"
    pre = out / "distilbert_fp32_pre.onnx"
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        quant_pre_process(str(fp32), str(pre))
        src = pre
        report["quant_pre_process"] = True
    except Exception as e:  # noqa: BLE001
        log(f"[export] quant_pre_process skipped ({type(e).__name__}: {e})")
        src = fp32
        report["quant_pre_process"] = False
    quantize_dynamic(str(src), str(int8), weight_type=QuantType.QInt8, per_channel=False)
    log(f"[export] int8 ONNX {int8.stat().st_size / 1e6:.1f} MB")
    report["quantization"] = "onnxruntime.quantization.quantize_dynamic, weight_type QInt8, per_channel False"

    def session(path: Path, threads: int = 0):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    sess = {"fp32": session(fp32), "int8": session(int8)}

    def onnx_proba(name: str, text: str) -> np.ndarray:
        e = encode(text)
        return sess[name].run(["probs"], {"input_ids": e["input_ids"].astype(np.int64),
                                          "attention_mask": e["attention_mask"].astype(np.int64)})[0][0].astype(np.float64)

    tok_dir = stage / "tokenizer"
    tok.save_pretrained(str(tok_dir))

    # ---- stage 1 and its check
    stage1, params, fixed, train_info = fit_stage1()
    s1_diff = check_stage1(stage1)

    # ---- Mendeley test: PyTorch / fp32 / int8, cascade
    te_ids, te_texts, te_y = load_split(task, "test")
    P = {k: np.zeros((len(te_y), 3)) for k in ("torch", "fp32", "int8")}
    for i, t in enumerate(te_texts):
        P["torch"][i] = torch_proba(t)
        P["fp32"][i] = onnx_proba("fp32", t)
        P["int8"][i] = onnx_proba("int8", t)
        if i % 200 == 0:
            log(f"[export] test {i}/{len(te_y)}")
    fz2 = frozen("distilbert_aug", "clean", te_ids)
    fz2p = fz2[[f"prob_{c}" for c in CLASSES]].to_numpy(dtype=np.float64)
    f1 = {k: float(f1_score(te_y, v.argmax(1), labels=[0, 1, 2], average="macro")) for k, v in P.items()}
    f1["frozen_gpu"] = float(f1_score(te_y, fz2p.argmax(1), labels=[0, 1, 2], average="macro"))
    s1 = stage1.predict_proba(te_texts)
    fz1 = frozen("svm_char_aug", "clean", te_ids)[[f"prob_{c}" for c in CLASSES]].to_numpy(dtype=np.float64)
    fzc = frozen("cascade", "clean", te_ids)
    routed = s1.max(1) < tau
    drop_pp = 100 * (f1["fp32"] - f1["int8"])
    final = "int8" if drop_pp <= MAX_DROP_PP else "fp32"
    casc = {k: np.where(routed[:, None], P[k], s1) for k in ("torch", "fp32", "int8")}
    report["test"] = {
        "n": int(len(te_y)),
        "macro_f1": f1,
        "int8_drop_pp": drop_pp,
        "decision": {"final": final, "rule": f"int8 if macro F1(fp32) - macro F1(int8) <= {MAX_DROP_PP} points"},
        "torch_cpu_vs_frozen_gpu_fp16": {"max_abs_diff": float(np.abs(P["torch"] - fz2p).max()),
                                         "same_class": float((P["torch"].argmax(1) == fz2p.argmax(1)).mean())},
        "fp32_vs_torch": {"max_abs_diff": float(np.abs(P["fp32"] - P["torch"]).max()),
                          "same_class": float((P["fp32"].argmax(1) == P["torch"].argmax(1)).mean())},
        "int8_vs_torch": {"max_abs_diff": float(np.abs(P["int8"] - P["torch"]).max()),
                          "same_class": float((P["int8"].argmax(1) == P["torch"].argmax(1)).mean())},
        "stage1_vs_frozen": {"max_abs_diff": float(np.abs(s1 - fz1).max())},
        "cascade": {
            "routed_share": float(routed.mean()),
            "routed_same_as_frozen": float((routed == fzc["routed"].astype(bool).to_numpy()).mean()),
            **{f"{k}_macro_f1": float(f1_score(te_y, v.argmax(1), labels=[0, 1, 2], average="macro"))
               for k, v in casc.items()},
            "frozen_macro_f1": float(f1_score(te_y, fzc[[f"prob_{c}" for c in CLASSES]].to_numpy().argmax(1),
                                              labels=[0, 1, 2], average="macro")),
            **{f"{k}_same_class_as_frozen": float((v.argmax(1) == fzc["y_pred"].map(CLASSES.index).to_numpy()).mean())
               for k, v in casc.items()},
        },
    }
    log(f"[export] test macro F1 {json.dumps(f1)}; int8 drop {drop_pp:.3f} pp -> final {final}")

    # ---- latency on this CPU, one thread, batch 1
    lat = {}
    sample = [te_texts[i % len(te_texts)] for i in range(300)]
    for name, path in (("fp32", fp32), ("int8", int8)):
        s1t = session(path, threads=1)
        enc = [encode(t) for t in sample]
        feed = [{"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)}
                for e in enc]
        for f in feed[:20]:
            s1t.run(["probs"], f)
        ts = []
        for f in feed:
            t0 = time.perf_counter()
            s1t.run(["probs"], f)
            ts.append((time.perf_counter() - t0) * 1000)
        lat[name] = {"p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "n": len(ts)}
    report["latency_onnxruntime_1_thread"] = lat
    report["cpu"] = subprocess.run("lscpu | grep 'Model name' | sed 's/.*: *//'", shell=True, capture_output=True,
                                   text=True).stdout.strip()

    # ---- golden
    rows = golden_messages()
    texts = [r["text"] for r in rows]
    g1 = stage1.predict_proba(texts)
    worst = {"fp32": 0.0, "int8": 0.0}
    same = {"fp32": 0, "int8": 0}
    for r, p1 in zip(rows, g1):
        pt = torch_proba(r["text"])
        po = {k: onnx_proba(k, r["text"]) for k in ("fp32", "int8")}
        for k in po:
            worst[k] = max(worst[k], float(np.abs(po[k] - pt).max()))
            same[k] += int(po[k].argmax() == pt.argmax())
        to_stage2 = bool(p1.max() < tau)
        fin = pt if to_stage2 else p1
        r.update({"n_tokens": int(encode(r["text"])["input_ids"].shape[1]),
                  "torch": [float(v) for v in pt], "pred": CLASSES[int(pt.argmax())],
                  "onnx_fp32": [float(v) for v in po["fp32"]], "onnx_int8": [float(v) for v in po["int8"]],
                  "stage1": [float(v) for v in p1],
                  "cascade": {"stage": 2 if to_stage2 else 1, "label": CLASSES[int(fin.argmax())]}})
    report["golden"] = {"n": len(rows), "max_abs_diff_vs_torch": worst, "same_class_vs_torch": same}
    log(f"[export] golden: max |onnx - torch| {worst}, same class {same}")
    golden = {"format": "smishfuse-golden-distilbert/1", "classes": CLASSES, "tau": tau, "max_len": max_len,
              "reference": "PyTorch fp32 on CPU, batch 1 (no padding)", "tolerance": 1e-3, "final_model": final,
              "n": len(rows), "messages": rows}
    (stage / "golden_distilbert.json").write_text(json.dumps(golden, ensure_ascii=False, indent=0), encoding="utf-8")

    checks = {"val_vs_final_val_seed1": s1_diff, "test_vs_final_clean_seed1": report["test"]["stage1_vs_frozen"]}
    (stage / "full_stage1_svm_char_aug.json").write_text(
        json.dumps(lite_json(stage1, params, fixed, train_info, checks, tau), ensure_ascii=False,
                   separators=(",", ":")), encoding="utf-8")

    report["files"] = {str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                       for p in sorted(stage.rglob("*")) if p.is_file()}
    report["wall_seconds"] = time.time() - t_start
    (stage / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    zpath = out / "export.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage))
    log(f"[export] wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB) in {report['wall_seconds'] / 60:.1f} min")


# ---------------------------------------------------------------- local: push, status, fetch

def _plumbing():
    sys.path.insert(0, str(ROOT / "scripts"))
    import kaggle_sync
    import make_kernels

    return kaggle_sync, make_kernels


def build_dir() -> Path:
    return ROOT / "kaggle" / "build" / KERNEL


NB_RUN = r'''import base64, hashlib, io, json, pathlib, shutil, subprocess, sys, tarfile
EXPECTED_CODE_SHA256 = {code!r}
SCRIPT_SHA256 = {sha!r}
SCRIPT_B64 = {b64!r}
FROZEN_SHA256 = {fsha!r}
FROZEN_B64 = {fb64!r}
got = pathlib.Path(code, "CODE_SHA256").read_text().strip()
if got != EXPECTED_CODE_SHA256:
    raise SystemExit(f"smishfuse-code {{got[:12]}} is not the pushed code {{EXPECTED_CODE_SHA256[:12]}}")
src = base64.b64decode(SCRIPT_B64)
if hashlib.sha256(src).hexdigest() != SCRIPT_SHA256:
    raise SystemExit("embedded export script: sha256 mismatch")
script = pathlib.Path("/kaggle/working/code/scripts/export_distilbert_onnx.py")
script.parent.mkdir(parents=True, exist_ok=True)
script.write_bytes(src)
frozen = base64.b64decode(FROZEN_B64)
if hashlib.sha256(frozen).hexdigest() != FROZEN_SHA256:
    raise SystemExit("embedded frozen predictions: sha256 mismatch")
tarfile.open(fileobj=io.BytesIO(frozen), mode="r:gz").extractall("/kaggle/working/code", filter="data")
print("code", got[:12], "script", SCRIPT_SHA256[:12], "frozen predictions", FROZEN_SHA256[:12], flush=True)
p = subprocess.Popen([sys.executable, str(script), "run", "--out", "/kaggle/working/run"],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print("exit code", rc, flush=True)
# the output of the kernel is one file: export.zip (no code copy, no intermediate files)
work = pathlib.Path("/kaggle/working")
if rc == 0:
    shutil.move("/kaggle/working/run/export.zip", "/kaggle/working/export.zip")
for item in work.iterdir():
    if item.name != "export.zip":
        shutil.rmtree(item, ignore_errors=True) if item.is_dir() else item.unlink(missing_ok=True)
if rc:
    raise SystemExit(rc)
'''


def cmd_push():
    ks, mk = _plumbing()
    digest = ks.tree_hash([ks.ROOT / p for p in ks.CODE_PARTS])
    stamp = ks.BUILD / "smishfuse-code.sha256"
    uploaded = stamp.read_text().strip() if stamp.exists() else ""
    if digest != uploaded:
        raise SystemExit(f"local code {digest[:12]} != uploaded smishfuse-code {uploaded[:12] or '-'}: "
                         "run `python scripts/kaggle_sync.py code` first")
    src = Path(__file__).read_bytes()
    sha = hashlib.sha256(src).hexdigest()
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in FROZEN_FILES:
            tf.add(ROOT / rel, arcname=rel)
    frozen_tgz = buf.getvalue()
    fsha = hashlib.sha256(frozen_tgz).hexdigest()
    d = build_dir()
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    meta = {"id": f"{ks.OWNER}/{KERNEL}", "title": KERNEL, "code_file": f"{KERNEL}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [f"{ks.OWNER}/smishfuse-data", f"{ks.OWNER}/smishfuse-code"],
            "competition_sources": [], "kernel_sources": [f"{ks.OWNER}/smishfuse-final-3class"], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    pins = mk.requirement_pins() + EXTRA_PIP
    cells = [
        ("markdown", f"# {KERNEL}\n\nGenerated by `scripts/export_distilbert_onnx.py push` (SmishFuse). "
                     f"Code {digest[:12]}, script {sha[:12]}. Output: one file, export.zip.\n"),
        ("code", mk.SETUP.format(kernel=KERNEL, spec=json.dumps({"kind": "export_onnx"}), pip=pins)),
        ("code", NB_RUN.format(code=digest, sha=sha, b64=base64.b64encode(src).decode(), fsha=fsha,
                               fb64=base64.b64encode(frozen_tgz).decode())),
    ]
    (d / f"{KERNEL}.ipynb").write_text(json.dumps(mk.notebook(cells), indent=1) + "\n", encoding="utf-8")
    (d / "PUSHED.json").write_text(json.dumps({"code_sha256": digest, "script_sha256": sha, "frozen_sha256": fsha,
                                               "pushed_unix": time.time()}, indent=2) + "\n")
    log(f"[push] code {digest[:12]} (= uploaded smishfuse-code), script {sha[:12]}")
    log(ks.kaggle("kernels", "push", "-p", str(d), capture=True).strip())


def cmd_status():
    ks, _ = _plumbing()
    log(ks.kaggle("kernels", "status", f"{ks.OWNER}/{KERNEL}", capture=True).strip())


def cmd_fetch():
    ks, _ = _plumbing()
    dst = ROOT / "kaggle" / "build" / f"{KERNEL}-output"
    dst.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        try:
            ks.kaggle("kernels", "output", f"{ks.OWNER}/{KERNEL}", "-p", str(dst), "--file-pattern", r"^export\.zip$",
                      "-o", "-q", timeout=1800)
            break
        except (subprocess.TimeoutExpired, SystemExit) as e:
            log(f"[fetch] attempt {attempt} failed: {type(e).__name__}")
            if attempt == 5:
                raise
            time.sleep(30)
    z = dst / "export.zip"
    with zipfile.ZipFile(z) as zf:
        report = json.loads(zf.read("report.json"))
        for name, info in report["files"].items():
            if name == "report.json":
                continue
            h = hashlib.sha256(zf.read(name)).hexdigest()
            if h != info["sha256"]:
                raise SystemExit(f"{name}: sha256 mismatch")
        zf.extractall(dst / "export")
    log(f"[fetch] {z} ({z.stat().st_size / 1e6:.1f} MB), {len(report['files'])} files, sha256 checked")
    log(json.dumps({k: report[k] for k in ("test", "golden", "latency_onnxruntime_1_thread")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check-stage1", "push", "status", "fetch", "run"])
    ap.add_argument("--out", default="/kaggle/working/run")
    a = ap.parse_args()
    if a.cmd == "check-stage1":
        clf, *_ = fit_stage1()
        check_stage1(clf)
        log(f"[stage1] golden set: {len(golden_messages())} messages")
    elif a.cmd == "run":
        run(Path(a.out))
    else:
        {"push": cmd_push, "status": cmd_status, "fetch": cmd_fetch}[a.cmd]()


if __name__ == "__main__":
    main()
